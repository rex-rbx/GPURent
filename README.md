# GPURent

GPURent is an early distributed-inference foundation. A JavaScript WebSocket
coordinator routes tensor inputs through ordered ONNX model shards, each
executed by a worker written in JavaScript, Python, or C++.

The application integrating GPURent owns user consent and must obtain it before
submitting work. Workers do not prompt for consent. This repository does not
yet include GPU-temperature or game-detection adapters; connect those to the
worker runtime before offering unattended GPU work.

## Requirements

- Node.js 20+
- Python 3.10+ with `onnx` and `onnxruntime`
- For the JavaScript worker: `onnxruntime-node`
- For the C++ worker: ONNX Runtime C++ SDK, CMake, and a C++17 compiler

The workers use CPU execution by default. For GPU execution, install a CUDA
enabled ONNX Runtime distribution and select its CUDA provider. For Python,
install `onnxruntime-gpu` instead of `onnxruntime`, then pass
`--provider CUDAExecutionProvider --provider CPUExecutionProvider`. For
JavaScript, use a Linux x64 build of `onnxruntime-node` with CUDA support and
pass `--provider cuda,cpu`.

## Prepare a model

Export the model to ONNX, then partition its graph into sequential shards:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r tools/requirements.txt
python tools/split_onnx.py model.onnx --partitions 2 --output-dir shards
```

The splitter groups topologically ordered graph nodes and carries tensors
crossing each boundary to the next shard. This supports general ONNX graphs,
including LLM graphs exported with explicit inputs/outputs for their state
(such as KV cache). Tokenization, sampling, and repeated autoregressive calls
belong to the integrating application. Operators and dynamic shapes must be
supported by the ONNX Runtime execution provider installed on each worker.

## Run

Start the coordinator:

```sh
npm install
npm start
```

It listens on `ws://localhost:8080` and `http://localhost:8080`. Set
`PORT` to change the port.

Start one worker for every shard. Workers for the same model can run in any of
the three supported languages:

```sh
# JavaScript
npm run worker:js -- --url ws://localhost:8080 --model demo --partition 0 --file shards/partition-000.onnx

# Python
pip install -r clients/python/requirements.txt
python clients/python/worker.py --url ws://localhost:8080 --model demo --partition 1 --file shards/partition-001.onnx

# C++ (set ONNXRUNTIME_ROOT to an ONNX Runtime C++ SDK installation)
cmake -S clients/cpp -B clients/cpp/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build clients/cpp/build
clients/cpp/build/gpurent-worker --url ws://localhost:8080 --model demo --partition 0 --file shards/partition-000.onnx
```

Use each partition index exactly once for a complete model pipeline. The
coordinator reports `503` until every shard is connected and ready.
For GPU execution, install a CUDA-enabled ONNX Runtime build, configure the C++
client with `-DGPURENT_CUDA=ON`, and pass `--cuda` when starting it.

Submit an inference as JSON to `POST /infer`:

```json
{
  "modelId": "demo",
  "inputs": [
    {"name": "input", "dtype": "float32", "shape": [1, 4], "values": [0.1, 0.2, 0.3, 0.4]}
  ]
}
```

The response contains the final shard's named output tensors. Tensor values
travel as JSON arrays in this prototype, so large models should use a binary
transport before production deployment. The HTTP/WebSocket endpoints are
intended for trusted local networks; add authentication, TLS, quotas, and
binary tensor framing before exposing them publicly.
Represent `int64` tensor elements as decimal strings in JSON to preserve their
exact values across JavaScript runtimes.

## Protocol

Workers send `worker:register` with a `modelId`, non-negative `partition`,
`workerId`, model `file`, and the shard's ONNX input/output names. The
coordinator replies with `worker:registered`. For each stage it sends
`inference:execute` containing a request ID and only the named tensors required
by that shard; tensors from earlier stages and original model inputs remain
available to later shards. The worker responds with `inference:result` and the
output tensors. A failed stage fails the whole inference request.

The partitioner and protocol are an initial execution path, not a universal
LLM runtime. In particular, ONNX export details (including state inputs and
outputs) vary by model; validate a model's shard inputs and outputs with the
integrating application before deployment.
