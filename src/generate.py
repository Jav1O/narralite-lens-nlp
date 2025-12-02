from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .prompts import Scenario, build_prompt


def load_model(model_name: str = "gpt2", device: Optional[str] = None):
    """
    Load tokenizer and model (GPT-2 small by default) and move model to device.
    """
    # 👇 Forzamos una implementación de atención que soporte output_attentions
    model_kwargs = {
        "attn_implementation": "eager",
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Asegurarnos de que hay pad_token para quitar warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


@torch.no_grad()
def generate_story(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 0,
    return_attentions: bool = True,
) -> Dict[str, Any]:
    """
    Generate a story continuation from a prompt.

    Returns a dict with:
      - prompt (str)
      - text (str)
      - input_ids (list[int])
      - attentions (list[Tensor]) if return_attentions=True
    """
    torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Generate continuation
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    full_ids = outputs.sequences  # (1, seq_len)

    # Re-run model on full sequence to get attentions (simpler and clearer)
    full_out = model(full_ids, output_attentions=return_attentions)

    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

    result: Dict[str, Any] = {
        "prompt": prompt,
        "text": text,
        "input_ids": full_ids[0].cpu().tolist(),
    }

    if return_attentions:
        # full_out.attentions is a tuple of length num_layers
        # each element: (batch, num_heads, seq, seq)
        result["attentions"] = [a[0].cpu() for a in full_out.attentions]

    return result
