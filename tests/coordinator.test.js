import assert from "node:assert/strict";
import { once } from "node:events";
import { afterEach, test } from "node:test";
import { WebSocket } from "ws";
import { createCoordinator } from "../server/index.js";

const coordinators = [];

async function startCoordinator() {
  const coordinator = createCoordinator({ port: 0 });
  coordinators.push(coordinator);
  const address = await coordinator.listen();
  return `http://127.0.0.1:${address.port}`;
}

async function registerWorker(baseUrl, modelId, partition, inputs, outputs, run) {
  const socket = new WebSocket(baseUrl.replace("http", "ws"));
  await once(socket, "open");
  socket.send(JSON.stringify({
    type: "worker:register",
    workerId: `worker-${partition}`,
    modelId,
    partition,
    file: `partition-${partition}.onnx`,
    inputs,
    outputs
  }));
  const [raw] = await once(socket, "message");
  assert.equal(JSON.parse(raw.toString()).type, "worker:registered");
  socket.on("message", async (data) => {
    const request = JSON.parse(data.toString());
    if (request.type === "inference:execute") {
      const outputs = await run(request.inputs);
      socket.send(JSON.stringify({
        type: "inference:result",
        requestId: request.requestId,
        outputs
      }));
    }
  });
  return socket;
}

afterEach(async () => {
  await Promise.all(coordinators.splice(0).map((coordinator) => coordinator.close()));
});

test("routes inference through model shards in partition order", async () => {
  const baseUrl = await startCoordinator();
  const sockets = [
    await registerWorker(baseUrl, "demo", 0, ["x"], ["middle"], (inputs) => [{
      ...inputs[0],
      name: "middle",
      values: inputs[0].values.map((value) => value + 1)
    }]),
    await registerWorker(baseUrl, "demo", 1, ["middle", "bias"], ["y"], (inputs) => [{
      ...inputs[0],
      name: "y",
      values: inputs[0].values.map((value, index) => value * 2 + inputs[1].values[index])
    }])
  ];

  const response = await fetch(`${baseUrl}/infer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      modelId: "demo",
      inputs: [
        { name: "x", dtype: "float32", shape: [2], values: [2, 3] },
        { name: "bias", dtype: "float32", shape: [2], values: [1, 1] }
      ]
    })
  });

  assert.equal(response.status, 200);
  assert.deepEqual((await response.json()).outputs[0].values, [7, 9]);
  sockets.forEach((socket) => socket.close());
});

test("does not start an inference pipeline with missing shard indices", async () => {
  const baseUrl = await startCoordinator();
  const socket = await registerWorker(baseUrl, "incomplete", 1, ["x"], ["y"], (inputs) => inputs);
  const response = await fetch(`${baseUrl}/infer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      modelId: "incomplete",
      inputs: [{ name: "x", dtype: "float32", shape: [1], values: [1] }]
    })
  });

  assert.equal(response.status, 503);
  socket.close();
});
