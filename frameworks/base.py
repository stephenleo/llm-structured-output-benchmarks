from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from data_sources.data_models import multilabel_classification_model


def response_parsing(response: Any) -> Any:
    if isinstance(response, list):
        response = {member.value for member in response}
    elif is_dataclass(response):
        response = asdict(response)
    elif isinstance(response, BaseModel):
        response = response.dict()
    return response


def experiment(
    n_runs: int = 10, expected_response: Any = None
) -> Callable[..., tuple[list[Any], int, Optional[float]]]:
    """Decorator to run an LLM call function multiple times and return the responses

    Args:
        n_runs (int): Number of times to run the function
        expected_response (set): The expected response set of classes. If provided, the decorator will calculate accurary too.

    Returns:
        Callable[..., Tuple[List[Any], int, Optional[float]]]: A function that returns a list of outputs from the function runs, percent of successful runs, accuracy of the identified classes if expected_response is provided else None.
    """

    def experiment_decorator(func):
        def wrapper(*args, **kwargs):
            classes = []
            accurate = 0
            for _ in tqdm(range(n_runs), leave=False):

                try:
                    response = func(*args, **kwargs)

                    if expected_response:
                        response = response_parsing(response)

                        if "classes" in response:
                            response = response_parsing(response["classes"])

                        if response == expected_response:
                            accurate += 1

                    classes.append(response)
                except:
                    pass

            num_successful = len(classes)
            percent_successful = num_successful / n_runs

            if expected_response:
                accuracy = accurate / num_successful if num_successful else 0

            return classes, percent_successful, accuracy if expected_response else None

        return wrapper

    return experiment_decorator


class BaseFramework(ABC):
    def __init__(
        self,
        name: str,
        prompt: str,
        llm_model: str,
        retries: int,
        source_data_pickle_path: str,
        sample_rows: int,
        response_model: Any,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.llm_model = llm_model
        self.retries = retries

        # Load the data
        self.source_data = pd.read_pickle(source_data_pickle_path)

        if sample_rows:
            self.source_data = self.source_data.sample(sample_rows)
            self.source_data = self.source_data.reset_index(drop=True)
        logger.info(f"Loaded source data from {source_data_pickle_path}")

        # Identify the classes
        # TODO: only for classification tasks
        if isinstance(self.source_data.iloc[0]["labels"], list):
            self.classes = self.source_data["labels"].explode().unique()
        else:
            self.classes = self.source_data["labels"].unique()
        logger.info(
            f"Source data has {len(self.source_data)} rows and {len(self.classes)} classes"
        )

        # Create the response model
        # TODO: only for classification tasks
        self.response_model = response_model or multilabel_classification_model(
            self.classes
        )
        logger.info(f"Response model is {self.response_model}")

    @abstractmethod
    def run(self, n_runs: int, expected_response: Any, *args, **kwargs): ...
