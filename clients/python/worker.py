import argparse
import json
import uuid

import numpy as np
import onnxruntime as ort
import websocket

DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def tensor_from_json(value):
    dtype = DTYPES.get(value["dtype"])
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype: {value['dtype']}")
    return np.asarray(value["values"], dtype=dtype).reshape(value["shape"])


def tensor_to_json(name, value):
    array = np.asarray(value)
    dtype = str(array.dtype)
    if dtype == "bool_":
        dtype = "bool"
    if dtype not in DTYPES:
        raise ValueError(f"unsupported output dtype: {dtype}")
    values = array.reshape(-1).tolist()
    if dtype == "int64":
        values = [str(value) for value in values]
    return {
        "name": name,
        "dtype": dtype,
        "shape": list(array.shape),
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--partition", required=True, type=int)
    parser.add_argument("--file", required=True)
    parser.add_argument("--provider", action="append", dest="providers")
    args = parser.parse_args()

    session = ort.InferenceSession(args.file, providers=args.providers)
    worker_id = str(uuid.uuid4())

    def on_open(socket):
        socket.send(json.dumps({
            "type": "worker:register",
            "workerId": worker_id,
            "modelId": args.model,
            "partition": args.partition,
            "file": args.file,
            "inputs": [value.name for value in session.get_inputs()],
            "outputs": [value.name for value in session.get_outputs()],
        }))

    def on_message(socket, raw):
        message = json.loads(raw)
        if message["type"] == "worker:registered":
            print(f"Worker {worker_id} registered for {args.model} partition {args.partition}")
            return
        if message["type"] != "inference:execute":
            return
        try:
            feeds = {item["name"]: tensor_from_json(item) for item in message["inputs"]}
            results = session.run(None, feeds)
            outputs = [
                tensor_to_json(output.name, value)
                for output, value in zip(session.get_outputs(), results, strict=True)
            ]
            response = {"type": "inference:result", "requestId": message["requestId"], "outputs": outputs}
        except Exception as error:
            response = {
                "type": "inference:result",
                "requestId": message["requestId"],
                "error": str(error),
            }
        socket.send(json.dumps(response))

    websocket.WebSocketApp(
        args.url,
        on_open=on_open,
        on_message=on_message,
    ).run_forever()


if __name__ == "__main__":
    main()
