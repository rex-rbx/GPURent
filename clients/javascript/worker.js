import { randomUUID } from "node:crypto";
import { WebSocket } from "ws";
import * as ort from "onnxruntime-node";

function parseArgs(args) {
  const options = {};
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    if (!key.startsWith("--") || !args[index + 1]) throw new Error(`invalid argument: ${key}`);
    options[key.slice(2)] = args[index + 1];
  }
  for (const required of ["url", "model", "partition", "file"]) {
    if (!options[required]) throw new Error(`missing --${required}`);
  }
  return options;
}

const typeMap = {
  float32: Float32Array,
  float64: Float64Array,
  int32: Int32Array,
  int64: BigInt64Array,
  uint8: Uint8Array,
  bool: Uint8Array
};

function toOrtTensor(value) {
  const Constructor = typeMap[value.dtype];
  if (!Constructor) throw new Error(`unsupported tensor dtype: ${value.dtype}`);
  const data = value.dtype === "int64"
    ? value.values.map((entry) => BigInt(entry))
    : value.values;
  return new ort.Tensor(value.dtype, new Constructor(data), value.shape);
}

function fromOrtTensor(name, tensor) {
  const dtype = tensor.type === "string" ? null : tensor.type;
  if (!typeMap[dtype]) throw new Error(`unsupported output dtype: ${dtype}`);
  const values = Array.from(tensor.data, (value) =>
    typeof value === "bigint" ? value.toString() : value
  );
  return { name, dtype, shape: tensor.dims, values };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const sessionOptions = options.provider
    ? { executionProviders: options.provider.split(",") }
    : undefined;
  const session = await ort.InferenceSession.create(options.file, sessionOptions);
  const socket = new WebSocket(options.url);
  const workerId = randomUUID();

  socket.on("open", () => socket.send(JSON.stringify({
    type: "worker:register",
    workerId,
    modelId: options.model,
    partition: Number(options.partition),
    file: options.file,
    inputs: session.inputNames,
    outputs: session.outputNames
  })));

  socket.on("message", async (raw) => {
    const message = JSON.parse(raw.toString());
    if (message.type === "worker:registered") {
      console.log(`Worker ${workerId} registered for ${options.model} partition ${options.partition}`);
      return;
    }
    if (message.type !== "inference:execute") return;
    try {
      const feeds = Object.fromEntries(message.inputs.map((input) => [input.name, toOrtTensor(input)]));
      const result = await session.run(feeds);
      const outputs = Object.entries(result).map(([name, tensor]) => fromOrtTensor(name, tensor));
      socket.send(JSON.stringify({ type: "inference:result", requestId: message.requestId, outputs }));
    } catch (error) {
      socket.send(JSON.stringify({
        type: "inference:result",
        requestId: message.requestId,
        error: error.message
      }));
    }
  });

  socket.on("error", (error) => console.error(`WebSocket error: ${error.message}`));
  socket.on("close", () => process.exit(0));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
