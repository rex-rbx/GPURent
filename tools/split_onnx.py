import argparse
import json
from pathlib import Path

import onnx


def partition_model(model_path, partitions, output_dir):
    model = onnx.load(str(model_path), load_external_data=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    if partitions < 1 or partitions > len(nodes):
        raise ValueError(f"partitions must be between 1 and {len(nodes)}")

    producer = {}
    consumers = {}
    initializers = {initializer.name for initializer in model.graph.initializer}
    for index, node in enumerate(nodes):
        for output in node.output:
            producer[output] = index
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(index)

    graph_outputs = {value.name for value in model.graph.output}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for partition in range(partitions):
        start = partition * len(nodes) // partitions
        end = (partition + 1) * len(nodes) // partitions
        inputs = []
        outputs = []
        for node in nodes[start:end]:
            for input_name in node.input:
                if not input_name:
                    continue
                source = producer.get(input_name)
                if (source is None and input_name not in initializers) or (
                    source is not None and source < start
                ):
                    if input_name not in inputs:
                        inputs.append(input_name)
            for output_name in node.output:
                if output_name in graph_outputs or any(
                    consumer < start or consumer >= end
                    for consumer in consumers.get(output_name, [])
                ):
                    if output_name not in outputs:
                        outputs.append(output_name)

        shard_path = output_dir / f"partition-{partition:03d}.onnx"
        onnx.utils.extract_model(
            str(model_path),
            str(shard_path),
            inputs,
            outputs,
            check_model=True,
            infer_shapes=True,
        )
        manifest.append({
            "partition": partition,
            "file": shard_path.name,
            "inputs": inputs,
            "outputs": outputs,
        })

    (output_dir / "manifest.json").write_text(
        json.dumps({"source": str(model_path), "partitions": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Split a topologically ordered ONNX graph into pipeline shards.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--partitions", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = partition_model(args.model, args.partitions, args.output_dir)
    print(f"Wrote {len(manifest)} ONNX partitions to {args.output_dir}")


if __name__ == "__main__":
    main()
