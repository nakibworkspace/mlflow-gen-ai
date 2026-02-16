import time

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name):
    """Load and return (tokenizer, model) for the given model name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def generate_text(tokenizer, model, prompt, max_new_tokens,
                   temperature=0.7, top_k=50, repetition_penalty=1.2):
    """Generate text from a prompt.

    Returns (output_text, latency_sec, token_count).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]

    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency_sec = time.time() - start

    generated_ids = output_ids[0][input_length:]
    token_count = len(generated_ids)
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return output_text, latency_sec, token_count
