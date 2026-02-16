import mlflow
import time
from transformers import pipeline

# ---------------------------
# CONFIG
# ---------------------------
EXPERIMENT_NAME = "GenAI-Labs"

PROMPTS = {
    "v1_simple": "Summarize this text:\n{input}",
    "v2_instruction": "You are an expert summarizer. Summarize concisely:\n{input}",
    "v3_bullet": "Summarize in bullet points:\n{input}"
}

# ---------------------------
# SET EXPERIMENT
# ---------------------------
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------
# LOAD MODEL (CPU Friendly)
# ---------------------------
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1  # force CPU
)

# ---------------------------
# LOAD DATASET
# ---------------------------
with open("dataset.txt") as f:
    text = f.read().strip()

# ---------------------------
# RUN EXPERIMENT
# ---------------------------
with mlflow.start_run(run_name="Prompt_Version_Comparison"):

    for version, template in PROMPTS.items():

        prompt = template.format(input=text)

        # ---- TRACE SPAN START ----
        with mlflow.start_span(name="llm_generation") as span:

            start = time.time()

            result = generator(prompt, max_length=80)[0]
            output = result["generated_text"]

            latency = time.time() - start

            # -----------------------
            # LOG GENAI TRACE DATA
            # -----------------------

            # inputs
            span.set_inputs({
                "prompt": prompt
            })

            # outputs
            span.set_outputs({
                "response": output
            })

            # attributes (metadata)
            span.set_attributes({
                "model": "distilgpt2",
                "prompt_version": version,
                "latency_sec": latency,
                "output_length": len(output),
                "task": "summarization",
                "framework": "transformers",
                "device": "cpu"
            })

            print(f"Finished {version}")

print("\n Experiment Completed — Check MLflow UI")
