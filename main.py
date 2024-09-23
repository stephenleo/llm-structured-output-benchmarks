import os
import pickle

import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from tqdm import tqdm

from frameworks import factory, metrics

app = typer.Typer()


@app.command()
def run_benchmark(config_path: str = "config.yaml"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} for local models")

    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)

    for config_key, config_values in configs.items():
        results = {}
        for config in config_values:
            results[config_key] = {}
            task = config["task"]
            n_runs = config["n_runs"]
            run_results = {
                "predictions": [],
                "percent_successful": [],
                "metrics": [],
                "latencies": [],
            }

            framework_instance = factory(
                config_key, task=task, device=device, **config["init_kwargs"]
            )
            logger.info(f"Using {type(framework_instance)}")

            if isinstance(framework_instance.source_data, pd.DataFrame):
                for row in tqdm(
                    framework_instance.source_data.itertuples(),
                    desc=f"Running {framework_instance.task}",
                    total=len(framework_instance.source_data),
                ):
                    if isinstance(row.labels, list):
                        labels = set(row.labels)
                    else:
                        labels = row.labels

                    # logger.info(f"Actual Text: {row.text}")
                    # logger.info(f"Actual Labels: {labels}")
                    predictions, percent_successful, framework_metrics, latencies = (
                        framework_instance.run(
                            inputs={"text": row.text},
                            n_runs=n_runs,
                            expected_response=labels,
                            task=task,
                        )
                    )
                    # logger.info(f"Predicted Labels: {predictions}")
                    run_results["metrics"].append(framework_metrics)
                    run_results["predictions"].append(predictions)
                    run_results["percent_successful"].append(percent_successful)
                    run_results["latencies"].append(latencies)
            else:
                predictions, percent_successful, _, latencies = (
                    framework_instance.run(
                        n_runs=n_runs,
                        task=task,
                    )
                )
                # logger.info(f"Predicted Labels: {predictions}")
                run_results["predictions"].append(predictions)
                run_results["percent_successful"].append(percent_successful)
                run_results["latencies"].append(latencies)

            results[config_key] = run_results

            # logger.info(f"Results:\n{results}")

            directory = f"results/{task}"
            os.makedirs(directory, exist_ok=True)

            with open(f"{directory}/{config_key}.pkl", "wb") as file:
                pickle.dump(results, file)
                logger.info(f"Results saved to {directory}/{config_key}.pkl")


@app.command()
def generate_results(
    results_data_path: str = "",
    task: str = "multilabel_classification",
):

    allowed_tasks = ["multilabel_classification", "ner", "ner_required_fields", "synthetic_data_generation"]
    if task not in allowed_tasks:
        raise ValueError(f"{task} is not allowed. Allowed values are {allowed_tasks}")
    
    if not results_data_path:
        results_data_path = f"./results/{task}"

    # Combine results from different frameworks
    results = {}

    for file_name in os.listdir(results_data_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(results_data_path, file_name)
            with open(file_path, "rb") as file:
                framework_results = pickle.load(file)
                results.update(framework_results)

    # Reliability
    percent_successful = {
        framework: value["percent_successful"]
        for framework, value in results.items()
    }
    logger.info(f"Reliability:\n{metrics.reliability_metric(percent_successful)}")

    # Latency
    latencies = {
        framework: value["latencies"]
        for framework, value in results.items()
    }
    logger.info(f"Latencies:\n{metrics.latency_metric(latencies, 95)}")

    # NER Micro Metrics
    if task in ("ner", "ner_required_fields"):
        micro_metrics_df = metrics.ner_micro_metrics(results)
        logger.info(f"NER Micro Metrics:\n{micro_metrics_df}")

    # Variety
    if task == "synthetic_data_generation":
        predictions = {
            framework: value["predictions"][0]
            for framework, value in results.items()
        }
        logger.info(f"Variety:\n{metrics.variety_metric(predictions)}")

if __name__ == "__main__":
    app()
