import os
import pickle
import sys

import typer
import yaml
from loguru import logger
from tqdm import tqdm

from frameworks import factory

app = typer.Typer()


@app.command()
def main(config_path: str = "config.yaml"):
    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)

    results = {}
    for config_key, config_values in configs.items():
        results[config_key] = {}

        for config in config_values:
            config_name = config["name"]
            n_runs = config["n_runs"]
            results[config_key][config_name] = {
                "predictions": [],
                "percent_successful": [],
                "accuracy": [],
            }

            framework_instance = factory(
                config_key, name=config_name, **config["init_kwargs"]
            )
            logger.info(f"Using {type(framework_instance)}")

            for row in tqdm(
                framework_instance.source_data.itertuples(),
                desc=f"Running {framework_instance.name}",
                total=len(framework_instance.source_data),
            ):
                logger.info(f"Actual Text: {row.text}")
                logger.info(f"Actual Labels: {set(row.labels)}")
                predictions, percent_successful, accuracy = framework_instance.run(
                    inputs={"text": row.text},
                    n_runs=n_runs,
                    expected_response=set(row.labels),
                )
                logger.info(f"Predicted Labels: {predictions}")
                results[config_key][config_name]["predictions"].append(predictions)
                results[config_key][config_name]["percent_successful"].append(
                    percent_successful
                )
                results[config_key][config_name]["accuracy"].append(accuracy)

    logger.info(f"Results:\n{results}")

    with open("results/results.pkl", "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    app()
