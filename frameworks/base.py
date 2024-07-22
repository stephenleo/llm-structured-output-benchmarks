import time
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from data_sources.data_models import multilabel_classification_model, ner_model


def response_parsing(response: Any) -> Any:
    if isinstance(response, list):
        response = {
            member.value if isinstance(member, Enum) else member for member in response
        }
    elif is_dataclass(response):
        response = asdict(response)
    elif isinstance(response, BaseModel):
        response = response.model_dump(exclude_none=True)
    return response


def calculate_metrics(
    y_true: dict[str, list[str]], y_pred: dict[str, list[str]]
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """Calculate the total True positives, False positives and False negatives for each entity in the NER task.

    Args:
        y_true (dict[str, list[str]]): The actual labels in the format {"entity1": ["value1", "value2"], "entity2": ["value3"]}
        y_pred (dict[str, list[str]]): The predicted labels in the format {"entity1": ["value1", "value2"], "entity2": ["value3"]}

    Returns:
        tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]: True positives, False positives and False negatives for each entity.
    """
    tp, fp, fn = {}, {}, {}
    for entity in y_true:
        tp[entity] = 0
        fp[entity] = 0
        fn[entity] = 0

        true_values = set(y_true.get(entity, []))
        pred_values = set(y_pred.get(entity, []))

        tp[entity] += len(true_values & pred_values)
        fp[entity] += len(pred_values - true_values)
        fn[entity] += len(true_values - pred_values)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def experiment(
    n_runs: int = 10,
    expected_response: Any = None,
    task: str = "multilabel_classification",
) -> Callable[..., tuple[list[Any], int, Optional[dict], list[list[float]]]]:
    """Decorator to run an LLM call function multiple times and return the responses

    Args:
        n_runs (int): Number of times to run the function
        expected_response (Any): The expected response. If provided, the decorator will calculate accurary too.
        task (str): The task being performed. Default is "multilabel_classification". Available options are "multilabel_classification" and "ner"

    Returns:
        Callable[..., Tuple[List[Any], int, Optional[dict], list[list[float]]]]: A function that returns a list of outputs from the function runs, percent of successful runs, metrics if expected_response is provided else None and list of latencies for each call.
    """

    def experiment_decorator(func):
        def wrapper(*args, **kwargs):
            allowed_tasks = ["multilabel_classification", "ner"]
            if task not in allowed_tasks:
                raise ValueError(
                    f"{task} is not allowed. Allowed values are {allowed_tasks}"
                )

            responses, latencies = [], []
            for _ in tqdm(range(n_runs), leave=False):

                try:
                    start_time = time.time()
                    response = func(*args, **kwargs)
                    end_time = time.time()

                    if expected_response:
                        response = response_parsing(response)

                        if "classes" in response:
                            response = response_parsing(response["classes"])

                    responses.append(response)
                    latencies.append(end_time - start_time)
                except:
                    pass

            num_successful = len(responses)
            percent_successful = num_successful / n_runs

            # Metrics calculation
            if task == "multilabel_classification" and expected_response:
                accurate = 0
                for response in responses:
                    if response == expected_response:
                        accurate += 1

                framework_metrics = {
                    "accuracy": accurate / num_successful if num_successful else 0
                }
            elif task == "ner":
                framework_metrics = []
                for response in responses:
                    framework_metrics.append(calculate_metrics(expected_response, response))

            return (
                responses,
                percent_successful,
                framework_metrics if expected_response else None,
                latencies,
            )

        return wrapper

    return experiment_decorator


class BaseFramework(ABC):
    task: str
    prompt: str
    llm_model: str
    llm_model_family: str
    retries: int
    source_data_pickle_path: str
    sample_rows: int
    response_model: Any
    device: str

    def __init__(self, *args, **kwargs) -> None:
        self.task = kwargs.get("task", "")
        self.prompt = kwargs.get("prompt", "")
        self.llm_model = kwargs.get("llm_model", "gpt-3.5-turbo")
        self.llm_model_family = kwargs.get("llm_model_family", "openai")
        self.retries = kwargs.get("retries", 0)
        self.device = kwargs.get("device", "cpu")
        source_data_pickle_path = kwargs.get("source_data_pickle_path", "")

        # Load the data
        self.source_data = pd.read_pickle(source_data_pickle_path)

        sample_rows = kwargs.get("sample_rows", 0)
        if sample_rows:
            self.source_data = self.source_data.sample(sample_rows)
            self.source_data = self.source_data.reset_index(drop=True)
        logger.info(f"Loaded source data from {source_data_pickle_path}")

        # Create the response model
        if "response_model" in kwargs:
            self.response_model = kwargs["response_model"]
        elif self.task == "multilabel_classification":
            # Identify the classes
            if isinstance(self.source_data.iloc[0]["labels"], list):
                self.classes = self.source_data["labels"].explode().unique()
            else:
                self.classes = self.source_data["labels"].unique()
            logger.info(
                f"Source data has {len(self.source_data)} rows and {len(self.classes)} classes"
            )

            self.response_model = multilabel_classification_model(self.classes)

        elif self.task == "ner":
            # Identify the entities
            self.entities = list(
                {key for d in self.source_data["labels"] for key in d.keys()}
            )

            self.response_model = ner_model(self.entities)

        logger.info(f"Response model is {self.response_model}")

    @abstractmethod
    def run(self, n_runs: int, expected_response: Any, *args, **kwargs): ...
