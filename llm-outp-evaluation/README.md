## Goal
Extend your tracing pipeline to evaluate LLM outputs automatically and log evaluation as child spans in MLflow.

## Objectives
- what evaluation means in GenAI
- how to score LLM outputs
- how to log evaluator spans
- how production LLM monitoring works

## Core Concepts
1. Generation ≠ Evaluation
```
Generation = model produces text
Evaluation = system judges text quality
```
### Production pipelines:
```
Prompt → LLM → Evaluators → Dashboard
```

## Evaluation Types

There are 3 major classes:

## Evaluation Metrics Summary

| Evaluation Type | Example Metric | Description |
| :--- | :--- | :--- |
| **Rule-based** | Length Score / Regex | Measures deterministic constraints and formatting. |
| **Embedding-based** | Semantic Similarity | Uses vector distance (e.g., Cosine) to check meaning. |
| **Model-based** | LLM-as-a-Judge | Uses a high-reasoning model to score quality/nuance. |

In this lab we implement:
Rule + Semantic evaluation

### Why Evaluations Are Logged as Spans
Because evaluation is a step of execution, not metadata.

Trace tree:
```
Generation span
 ├── Length evaluator
 └── Similarity evaluator
```

This gives debugging visibility.

### Project Structure
```
llm-outp-evaluation/
│
├── run_eval.py
├── dataset.txt
└── reference.txt
```

## Run Lab

Start UI:
```bash
mlflow ui
```

### Run script:
```bash
python3 run_eval.py
```

Open:
```bash
http://127.0.0.1:5000
```

🔎 What You Should See in UI

Trace tree:
```
Run
 └── generation
      ├── length_eval
      └── semantic_eval
```

Click each span to inspect:
1. inputs
2. outputs
3. scores