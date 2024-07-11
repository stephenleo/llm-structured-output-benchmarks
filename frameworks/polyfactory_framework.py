from typing import Any

from polyfactory.factories.pydantic_factory import ModelFactory

from frameworks.base import BaseFramework, experiment


class PolyfactoryFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        class ResponseFactory(ModelFactory):
            __model__ = self.response_model
            __randomize_collection_length__ = True
            # as a note, the response model allows repeated classes so this max
            # length is not quite correct, however it's better than the default
            # of 5
            __max_collection_length__ = len(self.classes)

        self.response_factory = ResponseFactory

    def run(
        self, n_runs: int, expected_response: Any, inputs: dict
    ) -> tuple[list[Any], float, float]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            return self.response_factory.build()

        predictions, percent_successful, accuracy, latencies = run_experiment(inputs)
        return predictions, percent_successful, accuracy, latencies
