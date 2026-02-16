import mlflow


def init_experiment(experiment_name):
    """Set the active MLflow experiment."""
    mlflow.set_experiment(experiment_name)


def track_run(run_index, prompt, output, latency, token_count):
    """Log a single prompt experiment as an MLflow trace."""
    with mlflow.start_span(name=f"prompt-{run_index}", span_type="LLM") as span:
        span.set_inputs({"prompt": prompt})
        span.set_outputs({"output": output})
        span.set_attributes({
            "token_count": token_count,
            "latency_sec": latency,
        })
