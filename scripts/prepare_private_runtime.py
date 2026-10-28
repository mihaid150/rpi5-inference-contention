import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(source, destination):
    source, destination = (Path(source), Path(destination))
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "build", "__pycache__"),
    )
    target = destination / "src/models/qwen2.cpp"
    original = target.read_text()
    text = original.replace(
        '#include "models.h"',
        '#include "models.h"\n#include <cstdlib>\n#include <set>\n#include <sstream>\n\n\nstatic bool poc_skip_attention(int layer) {\n    static const std::set<int> mask = [] {\n        std::set<int> result;\n        const char * value = std::getenv("DLI_ATTN_SKIP");\n        if (value && *value) {\n            std::stringstream stream(value);\n            std::string item;\n            while (std::getline(stream, item, \',\')) result.insert(std::stoi(item));\n        }\n        return result;\n    }();\n    return mask.count(layer) != 0;\n}',
        1,
    )
    start = text.index(
        "        // norm\n", text.index("for (int il = 0; il < n_layer; ++il)")
    )
    end = text.index('        cb(ffn_inp, "ffn_inp", il);', start)
    block = text[start:end].replace(
        "        ggml_tensor * ffn_inp =", "        ffn_inp ="
    )
    text = (
        text[:start]
        + '        if (std::getenv("DLI_ATTN_PROFILE")) {\n\n            inpSA = ggml_dup(ctx0, inpSA);\n            cb(inpSA, "poc_attn_in", il);\n        }\n        ggml_tensor * ffn_inp;\n        if (!poc_skip_attention(il)) {\n'
        + block
        + "        } else {\n\n            ffn_inp = inpSA;\n            if (il == n_layer - 1 && inp_out_ids) {\n                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);\n            }\n        }\n"
        + text[end:]
    )
    if text == original or text.count("if (!poc_skip_attention(il))") != 1:
        raise RuntimeError("Unsupported qwen2 graph layout")
    target.write_text(text)
    manifest = dict(
        source=str(source.resolve()),
        file="src/models/qwen2.cpp",
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(text.encode()).hexdigest(),
        change="Private research graph: fixed attention mask, FFN retained; optional calibration copies",
    )
    (destination.parent / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), indent=2))
