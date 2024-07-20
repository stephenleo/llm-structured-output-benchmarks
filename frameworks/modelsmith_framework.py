from typing import Any

from modelsmith import Forge, OpenAIModel

from frameworks.base import BaseFramework, experiment


class ModelsmithFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.forge = Forge(
            model=OpenAIModel(self.llm_model),
            response_model=self.response_model,
            max_retries=self.retries,
        )

    def run(
        self, n_runs: int, expected_response: Any, inputs: dict
    ) -> tuple[list[Any], float, float]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            response = self.forge.generate(user_input=self.prompt.format(**inputs))
            return response

        predictions, percent_successful, accuracy, latencies = run_experiment(inputs)
        return predictions, percent_successful, accuracy, latencies
