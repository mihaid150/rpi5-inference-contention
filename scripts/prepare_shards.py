import argparse
import hashlib
import json
from pathlib import Path


def tensor_digest(tensor):
    return hashlib.sha256(tensor.data).hexdigest()


def write(path, base, tensors):
    import gguf

    writer = gguf.GGUFWriter(str(path), "qwen2")
    for key, field in base.fields.items():
        if key.startswith(("GGUF.", "split.")) or key == "general.architecture":
            continue
        writer.add_key_value(
            key,
            field.contents(),
            field.types[0],
            field.types[1] if len(field.types) > 1 else None,
        )
    for tensor in tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    actual = gguf.GGUFReader(str(path))
    expected = {t.name: t for t in tensors}
    if set(expected) != {t.name for t in actual.tensors}:
        raise ValueError("Tensor coverage mismatch")
    for tensor in actual.tensors:
        source = expected[tensor.name]
        if (
            tuple(tensor.shape) != tuple(source.shape)
            or tensor.tensor_type != source.tensor_type
            or tensor_digest(tensor) != tensor_digest(source)
        ):
            raise ValueError("Shard tensor identity mismatch")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(assets, output):
    import gguf

    versions = json.loads((assets / "versions.json").read_text())
    models = {}
    for name, key in [("Q4_K_M", "model_sha256"), ("Q3_K_M", "q3_model_sha256")]:
        path = assets / "models/family" / (name + ".gguf")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != versions[key]:
                raise ValueError("Model family digest mismatch")
        models[name] = gguf.GGUFReader(str(path))
    base, donor = (models["Q4_K_M"], models["Q3_K_M"])
    other = {t.name: t for t in donor.tensors}
    if base.fields["qwen2.block_count"].contents() != 48 or set(other) != {
        t.name for t in base.tensors
    }:
        raise ValueError("Expected matched 48-layer Qwen family")
    for tensor in base.tensors:
        if not tensor.name.startswith("blk.") and tensor_digest(
            tensor
        ) != tensor_digest(other[tensor.name]):
            raise ValueError("Non-block tensors must remain identical")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for stage in range(4):
        owned = [
            t
            for t in base.tensors
            if t.name.startswith("blk.")
            and int(t.name.split(".")[1]) // 12 == stage
            or (t.name == "token_embd.weight" and stage == 0)
            or (t.name.startswith("output") and stage == 3)
        ]
        name = f"stage-{stage}.gguf"
        hashes[name] = write(output / name, base, owned)
        if stage == 0:
            mixed = [other[t.name] if t.name.startswith("blk.") else t for t in owned]
            hashes["alternate-stage-0.gguf"] = write(
                output / "alternate-stage-0.gguf", base, mixed
            )
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                hashes=hashes,
                hidden=int(base.fields["qwen2.embedding_length"].contents()),
                vocab=len(base.fields["tokenizer.ggml.tokens"].contents()),
                layers_per_stage=12,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.assets.resolve(), args.output.resolve())
