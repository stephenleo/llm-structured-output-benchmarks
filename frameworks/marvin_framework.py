from typing import Any

import marvin
from loguru import logger

from frameworks.base import BaseFramework, experiment


class MarvinFramework(BaseFramework):
    def __init__(
        self,
        name: str,
        prompt: str,
        llm_model: str,
        retries: int,
        source_data_pickle_path: str,
        sample_rows: int = 0,
        response_model: Any = None,
    ) -> None:
        super().__init__(
            name=name,
            prompt=prompt,
            llm_model=llm_model,
            retries=retries,
            source_data_pickle_path=source_data_pickle_path,
            sample_rows=sample_rows,
            response_model=response_model,
        )

        marvin.settings.openai.chat.completions.model = self.llm_model

    def run(
        self, n_runs: int, expected_response: Any, inputs: dict
    ) -> tuple[list[Any], float, float]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            response = marvin.cast(self.prompt.format(**inputs), self.response_model)
            return response

        predictions, percent_successful, accuracy = run_experiment(inputs)
        return predictions, percent_successful, accuracy