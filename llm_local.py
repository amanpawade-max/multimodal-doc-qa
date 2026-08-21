"""
Local-CPU integration for the GPT-OSS 120B model (openai/gpt-oss-120b).

WARNING: Running a 120B-parameter model on CPU is extremely slow and may
fail due to memory limitations. This example shows how to attempt a CPU
load with transformers but includes explicit guidance and a helpful
fallback message recommending using a hosted API or a much smaller model.

Usage:
    from llm_local import generate
    print(generate("Hello world"))

Requirements (install separately):
    pip install transformers torch

If loading fails, consider using the Hugging Face Inference API or a
smaller model (e.g., EleutherAI/gpt-neo-2.7B) for local CPU experiments.
"""

from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai/gpt-oss-120b"


def load_model(device: str = "cpu", trust_remote_code: bool = True) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Attempt to load the tokenizer and model onto CPU.

    This function uses low_cpu_mem_usage to reduce peak memory where
    possible. It raises a RuntimeError with actionable advice if the
    model cannot be loaded.
    """
    if device != "cpu":
        raise ValueError("This helper is intended for local CPU usage (device='cpu').")

    try:
        # Tokenizer is usually small and safe to load
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=trust_remote_code)

        # Attempt to load model on CPU. For very large models this may still fail.
        # low_cpu_mem_usage=True reduces peak RAM by using disk-backed sharding during init
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        )

        # Ensure model is in eval mode and on CPU
        model.to("cpu")
        model.eval()

        return tokenizer, model

    except Exception as e:
        # Provide an actionable error that explains why this likely failed and what to try next
        msg = (
            f"Failed to load model '{MODEL_NAME}' on CPU: {e}\n\n"
            "Notes / next steps:\n"
            " - Running a 120B model on CPU is generally impractical; it will be extremely slow and may run out of RAM.\n"
            " - Recommended: use Hugging Face Inference API, or run on a machine with GPUs (A100/80GB or larger) or use a server like vLLM/TGI.\n"
            " - For local CPU experimentation, try a much smaller model (e.g., 'EleutherAI/gpt-neo-2.7B' or 'gpt2').\n"
            " - If you still want to proceed, consider converting/quantizing the model and using optimized runtimes (not covered here).\n"
        )
        raise RuntimeError(msg) from e


def generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate text using the locally loaded model.

    This function loads the model on each call to keep the example simple.
    For production use, load once and reuse the tokenizer/model objects.
    """
    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to CPU explicitly
    input_ids = inputs["input_ids"].to("cpu")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPT-OSS 120B local-CPU example (may fail or be extremely slow)")
    print("=" * 60 + "\n")

    example_prompt = "Write a short, friendly greeting and a single-sentence summary of climate change."
    try:
        out = generate(example_prompt, max_new_tokens=128)
        print("Model output:\n", out)
    except RuntimeError as err:
        print(str(err))
