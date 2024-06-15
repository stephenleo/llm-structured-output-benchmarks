from typing import Any

import instructor
from openai import OpenAI

from frameworks.base import BaseFramework, experiment


class InstructorFramework(BaseFramework):
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
        self.instructor_client = instructor.patch(OpenAI())

    def run(
        self, n_runs: int, expected_response: Any, inputs: dict
    ) -> tuple[list[Any], float, float]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            response = self.instructor_client.chat.completions.create(
                model=self.llm_model,
                response_model=self.response_model,
                max_retries=self.retries,
                messages=[{"role": "user", "content": self.prompt.format(**inputs)}],
            )
            return {cat.value for cat in response.classes}

        predictions, percent_successful, accuracy = run_experiment(inputs)
        return predictions, percent_successful, accuracy
