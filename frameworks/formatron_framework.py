import json
from typing import Any

from formatron.formatter import FormatterBuilder
from formatron.integrations.transformers import create_formatter_logits_processor_list
from formatron.schemas import json_schema
from outlines.fsm.json_schema import build_regex_from_schema
from transformers import AutoModelForCausalLM, AutoTokenizer

from frameworks.base import BaseFramework, experiment


class FormatronFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_length = kwargs.get("max_length", 4096)
        # whether the model returns a regex match (only for the fallback regex
        # alternative)
        self.load_json_from_re_match = False

        if self.llm_model_family != "transformers":
            raise ValueError(f"Model family: {self.llm_model_family} not supported")

        f = FormatterBuilder()
        if self.task != "multilabel_classification":
            model_schema = self.response_model.model_json_schema()
            # pydantic (v2.7.1) doesn't have a good way to include $schema,
            # c.f.: https://github.com/pydantic/pydantic/issues/1478
            model_schema["$id"] = model_schema["title"]
            model_schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
            schema = json_schema.create_schema(model_schema)
            response = f.json(schema, capture_name="json")
        else:
            # fall back to outlines's pydantic regex for:
            # - multilabel_classification task with enum: enum seems to be
            #   supported by formatron v0.4, but the output appears to be buggy
            #   at first glance, a mixture of [] and [set()]
            self.load_json_from_re_match = True
            schema = json.dumps(self.response_model.model_json_schema())
            whitespace_pattern = r" ?"
            regex_str = build_regex_from_schema(schema, whitespace_pattern)
            response = f.regex(regex_str, capture_name="json")
        f.append_line(f"{response}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        self.logits_processor = create_formatter_logits_processor_list(
            self.tokenizer, f
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_model)

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            prompt = self.prompt.format(
                json_schema=self.response_model.model_json_schema(), **inputs
            )
            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            self.logits_processor[0].reset()
            self.model.generate(
                **tokens,
                logits_processor=self.logits_processor,
                max_length=self.max_length,
            )
            response = self.logits_processor[0].formatters_captures[0]["json"]
            if self.load_json_from_re_match:
                response = json.loads(response.group(0))
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
