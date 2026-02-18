import mlflow
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -------------------------
# CONFIG
# -------------------------
mlflow.set_experiment("GenAI-Labs")

PROMPT = "Summarize this text:\n{input}"

# -------------------------
# LOAD MODELS
# -------------------------
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# LOAD DATA
# -------------------------
text = open("dataset.txt").read().strip()
reference = open("reference.txt").read().strip()

prompt = PROMPT.format(input=text)

# -------------------------
# RUN EXPERIMENT
# -------------------------
with mlflow.start_run(run_name="Lab03_Evaluation"):

    # ---------- GENERATION SPAN ----------
    with mlflow.start_span("generation") as gen_span:

        start = time.time()
        output = generator(prompt, max_length=80)[0]["generated_text"]
        latency = time.time() - start

        gen_span.set_inputs({"prompt": prompt})
        gen_span.set_outputs({"response": output})

        gen_span.set_attributes({
            "model": "distilgpt2",
            "latency": latency
        })

        # ---------- EVALUATION 1: LENGTH ----------
        with mlflow.start_span("length_eval") as span:

            length_score = 1 / len(output.split())

            span.set_inputs({"output": output})
            span.set_outputs({"length_score": length_score})
            span.set_attributes({"metric": "inverse_length"})

        # ---------- EVALUATION 2: SEMANTIC ----------
        with mlflow.start_span("semantic_eval") as span:

            emb1 = embedder.encode(output, convert_to_tensor=True)
            emb2 = embedder.encode(reference, convert_to_tensor=True)

            similarity = float(util.cos_sim(emb1, emb2))

            span.set_inputs({
                "output": output,
                "reference": reference
            })

            span.set_outputs({"similarity_score": similarity})

            span.set_attributes({
                "metric": "cosine_similarity"
            })

print("\Evaluation tracing completed.")
