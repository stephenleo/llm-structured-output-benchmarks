import json
from typing import Any

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import pipeline

from frameworks.base import BaseFramework, experiment


class LMFormatEnforcerFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parser = JsonSchemaParser(self.response_model.schema())
        max_length = kwargs.get("max_length", 4096)

        if self.llm_model_family == "transformers":
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                device_map=self.device,
                max_length=max_length,
            )
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(
                self.hf_pipeline.tokenizer, self.parser
            )
        else:
            raise ValueError(f"Model family: {self.llm_model_family} not supported")

    def run(
        self, n_runs: int, expected_response: Any, inputs: dict, task: str
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            prompt = self.prompt.format(
                json_schema=self.response_model.schema(), **inputs
            )
            response = self.hf_pipeline(
                prompt, prefix_allowed_tokens_fn=self.prefix_function
            )
            response = response[0]["generated_text"][len(prompt) :].strip()
            json_response = json.loads(response)
            return json_response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
