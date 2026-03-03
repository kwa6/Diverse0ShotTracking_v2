# language_model/llama.py
import dataclasses as dc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


def llama3format(messages: Any) -> str:
    """
    Minimal formatter to match experiment.py usage.
    Accepts either a preformatted string or a list/dict-like structure.
    """
    if isinstance(messages, str):
        return messages
    # best-effort stringify
    try:
        return "\n".join(str(m) for m in messages)
    except Exception:
        return str(messages)


@dataclass
class LlamaHyperparameters:
    # model identity
    base: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer_reponame: Optional[str] = None

    # prompt formatting
    format: Optional[Callable[[Any], str]] = None

    # generation
    max_output_length: int = 32
    num_beams: int = 1
    repetition_penalty: float = 1.0

    # seq lengths (used by tracker preprocessing logic)
    max_sequence_length: int = 512
    protected_input_length: int = 400

    # optimization knobs (kept for compatibility; only used if you train)
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # optional features
    quantize: Optional[str] = None  # e.g. "nf4" (GPU only, bitsandbytes)
    lora: Optional[int] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.0


class Llama:
    """
    Thin HF wrapper used by dextrous/tracker.py.

    Required by tracker:
      - save(path)
      - predict(data)  (implemented in tracker, calls model.generate internally)
      - perplexity(data) (implemented in tracker, calls model forward)
      - training(...) (implemented in tracker, calls model forward + optimizer)
    """
    def __init__(self, **kwargs):
        h = LlamaHyperparameters(**kwargs)
        self.h = h

        tok_name = h.tokenizer_reponame or h.base
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

        # device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NOTE: quantization via bitsandbytes is intentionally NOT done here.
        # Keep it minimal and stable. You can extend once GPU pipeline is verified.
        self.model = AutoModelForCausalLM.from_pretrained(
            h.base,
            torch_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
            device_map=("auto" if self.device.type == "cuda" else None),
        )

        if self.device.type == "cpu":
            self.model.to(self.device)

        # optional LoRA (only if peft installed and lora specified)
        if h.lora and get_peft_model is not None:
            cfg = LoraConfig(
                r=int(h.lora),
                lora_alpha=int(h.lora_alpha),
                lora_dropout=float(h.lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, cfg)

        self.model.eval()

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # also store hyperparams for reproducibility
        hp = dc.asdict(self.h)
        (path / "model_hyperparameters.json").write_text(
            __import__("json").dumps(hp, indent=2)
        )
        return str(path)