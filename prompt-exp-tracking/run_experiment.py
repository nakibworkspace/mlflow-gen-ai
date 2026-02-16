from config import (
    EXPERIMENT_NAME, MAX_NEW_TOKENS, MODEL_NAME, PROMPTS,
    REPETITION_PENALTY, TEMPERATURE, TOP_K,
)
from model import generate_text, load_model
from tracker import init_experiment, track_run


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer, model = load_model(MODEL_NAME)

    init_experiment(EXPERIMENT_NAME)

    print(f"Running {len(PROMPTS)} prompt experiments...\n")
    for i, prompt in enumerate(PROMPTS):
        output, latency, token_count = generate_text(
            tokenizer, model, prompt, MAX_NEW_TOKENS,
            TEMPERATURE, TOP_K, REPETITION_PENALTY,
        )
        track_run(i, prompt, output, latency, token_count)

        print(f"[Run {i}]")
        print(f"  Prompt:      {prompt}")
        print(f"  Output:      {output[:120]}...")
        print(f"  Latency:     {latency:.2f}s")
        print(f"  Tokens:      {token_count}\n")

    print(f"Done — {len(PROMPTS)} runs traced in experiment '{EXPERIMENT_NAME}'.")
    print("Run `mlflow ui` and open http://localhost:5000 to inspect traces.")


if __name__ == "__main__":
    main()
