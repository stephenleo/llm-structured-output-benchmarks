from typing import Any

import instructor
from openai import OpenAI

from frameworks.base import BaseFramework, experiment


class InstructorFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.instructor_client = instructor.patch(OpenAI())

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            response = self.instructor_client.chat.completions.create(
                model=self.llm_model,
                response_model=self.response_model,
                max_retries=self.retries,
                messages=[{"role": "user", "content": self.prompt.format(**inputs)}],
            )
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
