# language_model/t5.py
import dataclasses as dc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class T5Hyperparameters:
    base: str = "google/flan-t5-small"
    max_output_length: int = 32
    num_beams: int = 1
    repetition_penalty: float = 1.0

    max_sequence_length: int = 512
    protected_input_length: int = 400

    learning_rate: float = 1e-4
    weight_decay: float = 0.0


class T5:
    def __init__(self, **kwargs):
        h = T5Hyperparameters(**kwargs)
        self.h = h

        self.tokenizer = AutoTokenizer.from_pretrained(h.base, use_fast=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            h.base,
            torch_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
        ).to(self.device)

        self.model.eval()

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        hp = dc.asdict(self.h)
        (path / "model_hyperparameters.json").write_text(
            __import__("json").dumps(hp, indent=2)
        )
        return str(path)