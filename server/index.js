import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { WebSocket, WebSocketServer } from "ws";

const json = (response, status, value) => {
  response.writeHead(status, { "content-type": "application/json" });
  response.end(JSON.stringify(value));
};

function isTensor(value) {
  const supportedTypes = new Set(["float32", "float64", "int32", "int64", "uint8", "bool"]);
  if (!value || typeof value.name !== "string" || !value.name ||
      !supportedTypes.has(value.dtype) || !Array.isArray(value.shape) ||
      !Array.isArray(value.values)) {
    return false;
  }
  if (!value.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)) {
    return false;
  }
  const elementCount = value.shape.reduce((count, dimension) => count * dimension, 1);
  if (!Number.isSafeInteger(elementCount) || value.values.length !== elementCount) {
    return false;
  }
  if (value.dtype === "bool") {
    return value.values.every((entry) => typeof entry === "boolean" || entry === 0 || entry === 1);
  }
  if (value.dtype === "int64") {
    return value.values.every((entry) =>
      Number.isSafeInteger(entry) ||
      (typeof entry === "string" && /^-?(0|[1-9]\d*)$/.test(entry))
    );
  }
  return value.values.every((entry) =>
    typeof entry === "number" && Number.isFinite(entry) &&
    (value.dtype === "float32" || value.dtype === "float64" || Number.isInteger(entry)) &&
    (value.dtype !== "int32" || (entry >= -2147483648 && entry <= 2147483647)) &&
    (value.dtype !== "uint8" || (entry >= 0 && entry <= 255))
  );
}

export function createCoordinator({ port = Number(process.env.PORT ?? 8080) } = {}) {
  const workers = new Map();
  const server = createServer(async (request, response) => {
    if (request.method === "GET" && request.url === "/health") {
      return json(response, 200, { status: "ok" });
    }

    if (request.method === "GET" && request.url === "/workers") {
      const summary = [...workers.values()].map(({ workerId, modelId, partition, ready }) => ({
        workerId,
        modelId,
        partition,
        ready
      }));
      return json(response, 200, { workers: summary });
    }

    if (request.method !== "POST" || request.url !== "/infer") {
      return json(response, 404, { error: "not_found" });
    }

    let body = "";
    try {
      for await (const chunk of request) {
        body += chunk;
        if (body.length > 50 * 1024 * 1024) {
          return json(response, 413, { error: "request_too_large" });
        }
      }

    } catch (error) {
      return json(response, 400, { error: "invalid_json", message: error.message });
    }

    let modelId;
    let inputs;
    try {
      ({ modelId, inputs } = JSON.parse(body));
    } catch (error) {
      return json(response, 400, { error: "invalid_json", message: error.message });
    }
    if (typeof modelId !== "string" || !modelId || !Array.isArray(inputs) ||
        inputs.length === 0 || !inputs.every(isTensor) ||
        new Set(inputs.map((input) => input.name)).size !== inputs.length) {
      return json(response, 400, { error: "invalid_inference_request" });
    }

    try {
      const modelWorkers = [...workers.values()]
        .filter((worker) => worker.modelId === modelId && worker.ready)
        .sort((a, b) => a.partition - b.partition);
      if (modelWorkers.length === 0 ||
          modelWorkers.some((worker, index) => worker.partition !== index)) {
        return json(response, 503, { error: "model_shards_unavailable" });
      }

      const availableTensors = new Map(inputs.map((input) => [input.name, input]));
      let outputs = [];
      for (const worker of modelWorkers) {
        const stageInputs = worker.inputs.map((name) => {
          const tensor = availableTensors.get(name);
          if (!tensor) throw new Error(`missing model input: ${name}`);
          return tensor;
        });
        outputs = await executeOnWorker(worker, randomUUID(), stageInputs);
        for (const output of outputs) availableTensors.set(output.name, output);
      }
      return json(response, 200, { outputs });
    } catch (error) {
      if (error?.message.startsWith("missing model input:")) {
        return json(response, 400, { error: "missing_model_input", message: error.message });
      }
      const status = error?.message === "worker_busy" ? 503 : 502;
      return json(response, status, { error: "inference_failed", message: error.message });
    }
  });

  const websocketServer = new WebSocketServer({ server });
  websocketServer.on("connection", (socket) => {
    let worker;
    socket.on("message", (raw) => {
      let message;
      try {
        message = JSON.parse(raw.toString());
      } catch {
        socket.close(1003, "invalid JSON");
        return;
      }

      if (message.type === "worker:register") {
        if (worker || typeof message.workerId !== "string" ||
            typeof message.modelId !== "string" || !message.modelId ||
            !Number.isInteger(message.partition) || message.partition < 0 ||
            typeof message.file !== "string" || !message.file ||
            !Array.isArray(message.inputs) || !message.inputs.every((name) => typeof name === "string") ||
            new Set(message.inputs).size !== message.inputs.length ||
            !Array.isArray(message.outputs) || !message.outputs.every((name) => typeof name === "string") ||
            new Set(message.outputs).size !== message.outputs.length) {
          socket.close(1008, "invalid worker registration");
          return;
        }
        worker = {
          socket,
          workerId: message.workerId,
          modelId: message.modelId,
          partition: message.partition,
          inputs: message.inputs,
          outputs: message.outputs,
          ready: true,
          pending: new Map()
        };
        workers.set(socket, worker);
        socket.send(JSON.stringify({ type: "worker:registered" }));
        return;
      }

      if (!worker) {
        socket.close(1008, "register before sending messages");
        return;
      }

      if (message.type === "inference:result" && typeof message.requestId === "string") {
        const pending = worker.pending.get(message.requestId);
        if (!pending) return;
        worker.pending.delete(message.requestId);
        if (message.error) pending.reject(new Error(String(message.error)));
        else if (Array.isArray(message.outputs) && message.outputs.every(isTensor) &&
                 message.outputs.length === worker.outputs.length &&
                 message.outputs.every((output) => worker.outputs.includes(output.name))) {
          pending.resolve(message.outputs);
        } else {
          pending.reject(new Error("worker returned invalid tensors"));
        }
      }
    });

    socket.on("close", () => {
      if (worker) {
        workers.delete(socket);
        for (const pending of worker.pending.values()) {
          pending.reject(new Error("worker disconnected"));
        }
      }
    });
  });

  function executeOnWorker(worker, requestId, inputs) {
    if (worker.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("worker disconnected"));
    }
    if (worker.pending.size > 0) {
      return Promise.reject(new Error("worker_busy"));
    }
    return new Promise((resolve, reject) => {
      worker.pending.set(requestId, { resolve, reject });
      worker.socket.send(JSON.stringify({
        type: "inference:execute",
        requestId,
        inputs
      }), (error) => {
        if (!error) return;
        worker.pending.delete(requestId);
        reject(error);
      });
    });
  }

  return {
    server,
    listen() {
      return new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(port, () => {
          server.off("error", reject);
          resolve(server.address());
        });
      });
    },
    close() {
      for (const worker of workers.values()) worker.socket.close();
      websocketServer.close();
      return new Promise((resolve) => server.close(resolve));
    }
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const coordinator = createCoordinator();
  const address = await coordinator.listen();
  console.log(`GPURent coordinator listening on port ${address.port}`);
}
