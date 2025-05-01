import time
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from litgpt.lora import GPT
from litgpt.generate.base import generate
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import PromptStyle

from jsonargparse import CLI
import lightning as L

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

@torch.inference_mode()
def main(
    checkpoint_path: Path = Path("./checkpoints/finetuned.ckpt"),
    prompt: str = "What food do llamas eat?",
    *,
    max_new_tokens: int = 256,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
) -> None:
    """Test the finetuned model by giving it a prompt."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(f"checkpoints/{model_name}")
    prompt_style = PromptStyle.from_name("alpaca")
    prompt = prompt_style.apply(prompt)
    encoded = tokenizer.encode(prompt, device=device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    print(f"Loading model: {checkpoint_path}")
    t0 = time.perf_counter()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = GPT.from_name(name=model_name)

    state_dict = torch.load(checkpoint_path, mmap=True, weights_only=False, map_location=device)["state_dict"]
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, assign=True, strict=False)
    print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.")
    
    with device:
        model.max_seq_length = max_returned_tokens
        model.set_kv_cache(batch_size=1)
    model.cos, model.sin = model.rope_cache(device=device)
    model.eval()
    
    L.seed_everything(42, verbose=False)

    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id, include_prompt=False)
    print(f"\n{tokenizer.decode(y)}")



if __name__ == "__main__":
    CLI(main)
