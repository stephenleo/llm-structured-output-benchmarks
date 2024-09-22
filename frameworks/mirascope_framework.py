import re
from typing import Any, Type

from mirascope.core import openai, prompt_template
from mirascope.integrations.tenacity import collect_errors
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt

from frameworks.base import BaseFramework, experiment


class MirascopeFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Identify all the input fields in the prompt and create the pydantic model
        # prompt_fields = re.findall(r"\{(.*?)\}", self.prompt)

    def mirascope_client(self, errors: list[ValidationError] | None = None, **kwargs):
        @retry(stop=stop_after_attempt(2), after=collect_errors(ValidationError))
        @openai.call(self.llm_model, response_model=self.response_model)
        @prompt_template("{previous_errors}\n\n" + self.prompt)
        def decorated_method(errors, **kwargs):
            return self._mirascope_client(errors, **kwargs)
        
        return decorated_method(errors, **kwargs)
    
    def _mirascope_client(self, errors, **kwargs):
        previous_errors = (
            f"Previous Errors: {errors}"
            if errors
            else "No previous errors."
        )
        return {"computed_fields": {"previous_errors": previous_errors}}

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            response = self.mirascope_client(**inputs)
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
