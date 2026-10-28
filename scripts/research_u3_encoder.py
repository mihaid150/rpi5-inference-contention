import hashlib
import json
from pathlib import Path


class MiniLMEncoder:

    def __init__(self, directory, threads=2):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(threads)
        directory = Path(directory)
        files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ]
        hashes = {}
        for name in files:
            h = hashlib.sha256()
            with (directory / name).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            hashes[name] = h.hexdigest()
        self.identity = dict(
            files=hashes,
            pooling="masked-mean-l2",
            max_tokens=256,
            dtype="float32",
            device="cpu",
            threads=threads,
            torch=torch.__version__,
            transformers=transformers.__version__,
        )
        self.encoder_id = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .float()
            .eval()
        )

    def token_count(self, text):
        return len(
            self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        )

    def __call__(self, texts):
        if not texts:
            return []
        encoded = self.tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt"
        )
        if encoded["input_ids"].shape[1] > 256:
            raise ValueError(
                "Encoder input exceeds 256 tokens; rechunk or explicitly revise query policy"
            )
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.torch.nn.functional.normalize(pooled, p=2, dim=1).tolist()
