# 🧩 LLM Structured Output Benchmarks
<!--- BADGES: START --->
[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![DOI](https://zenodo.org/badge/815423644.svg)](https://zenodo.org/doi/10.5281/zenodo.12327266)
[![GitHub - License](https://img.shields.io/github/license/stephenleo/llm-structured-output-benchmarks?logo=github&style=flat&color=green)][#github-license]

![Github](https://img.shields.io/github/followers/stephenleo?style=social)
[![dev.to badge](https://img.shields.io/badge/-Marie%20Stephen%20Leo-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/marie-stephen-leo/)
[![dev.to badge](https://img.shields.io/badge/-Medium-blueviolet?style=flat&logo=medium)](https://stephen-leo.medium.com/)

[#github-license]: https://github.com/stephenleo/llm-structured-output-benchmarks/blob/main/LICENSE
<!--- BADGES: END --->

Benchmark various LLM Structured Output frameworks: Instructor, Mirascope, Langchain, LlamaIndex, Fructose, Marvin, Outlines, LMFormatEnforcer, etc on tasks like multi-label classification, named entity recognition, synthetic data generation, etc.

## 🏆 Benchmark Results [2024-08-25]

1. Multi-label classification
    | Framework                                                                                           |                 Model                | Reliability | Latency p95 (s) |
    |-----------------------------------------------------------------------------------------------------|:------------------------------------:|:-----------:|:---------------:|
    | [Fructose](https://github.com/bananaml/fructose)                                                    |        gpt-4o-mini-2024-07-18        |    1.000    |       1.138     |
    | [Modelsmith](https://github.com/christo-olivier/modelsmith)                                         |        gpt-4o-mini-2024-07-18        |    1.000    |       1.184     |
    | [OpenAI Structured Output](https://github.com/openai/openai-python)                                 |        gpt-4o-mini-2024-07-18        |    1.000    |       1.201     |
    | [Instructor](https://github.com/jxnl/instructor)                                                    |        gpt-4o-mini-2024-07-18        |    1.000    |       1.206     |
    | [Outlines](https://github.com/outlines-dev/outlines)                                                | unsloth/llama-3-8b-Instruct-bnb-4bit |    1.000    | 1.804<sup>*</sup> |
    | [LMFormatEnforcer](https://github.com/noamgat/lm-format-enforcer)                                   | unsloth/llama-3-8b-Instruct-bnb-4bit |    1.000    | 3.649<sup>*</sup> |
    | [Llamaindex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/openai_pydantic_program/) |        gpt-4o-mini-2024-07-18        |    0.996    |       0.853     |
    | [Marvin](https://github.com/PrefectHQ/marvin)                                                       |        gpt-4o-mini-2024-07-18        |    0.988    |       1.338     |
    | [Mirascope](https://github.com/mirascope/mirascope)                                                 |        gpt-4o-mini-2024-07-18        |    0.985    |       1.531     |
1. Named Entity Recognition
    | Framework                                                                                           |                 Model                | Reliability | Latency p95 (s) |  Precision  |   Recall    |  F1 Score   |
    |-----------------------------------------------------------------------------------------------------|:------------------------------------:|:-----------:|:---------------:|:-----------:|:-----------:|:-----------:|
    | [OpenAI Structured Output](https://github.com/openai/openai-python)                                 |        gpt-4o-mini-2024-07-18        |    1.000    |       3.459     |    0.834    |    0.748    |    0.789    |
    | [LMFormatEnforcer](https://github.com/noamgat/lm-format-enforcer)                                   | unsloth/llama-3-8b-Instruct-bnb-4bit |    1.000    | 6.573<sup>*</sup> |    0.701    |    0.262    |    0.382    |
    | [Instructor](https://github.com/jxnl/instructor)                                                    |        gpt-4o-mini-2024-07-18        |    0.998    |       2.438     |    0.776    |    0.768    |    0.772    |
    | [Mirascope](https://github.com/mirascope/mirascope)                                                 |        gpt-4o-mini-2024-07-18        |    0.989    |       3.879     |    0.768    |    0.738    |    0.752    |
    | [Llamaindex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/openai_pydantic_program/) |        gpt-4o-mini-2024-07-18        |    0.979    |       5.771     |    0.792    |    0.310    |    0.446    |
    | [Marvin](https://github.com/PrefectHQ/marvin)                                                       |        gpt-4o-mini-2024-07-18        |    0.979    |       3.270     |    0.822    |    0.776    |    0.798    |
1. Synthetic Data Generation
    | Framework                                                                                           |                 Model                | Reliability | Latency p95 (s) | Variety |
    |-----------------------------------------------------------------------------------------------------|:------------------------------------:|:-----------:|:---------------:|:-------:|
    | [Instructor](https://github.com/jxnl/instructor)                                                    |        gpt-4o-mini-2024-07-18        |    1.000    |       1.923     |  0.750  |
    | [Marvin](https://github.com/PrefectHQ/marvin)                                                       |        gpt-4o-mini-2024-07-18        |    1.000    |       1.496     |  0.010  |
    | [Llamaindex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/openai_pydantic_program/) |        gpt-4o-mini-2024-07-18        |    1.000    |       1.003     |  0.020  |
    | [Modelsmith](https://github.com/christo-olivier/modelsmith)                                         |        gpt-4o-mini-2024-07-18        |    0.970    |       2.324     |  0.835  |
    | [Mirascope](https://github.com/mirascope/mirascope)                                                 |        gpt-4o-mini-2024-07-18        |    0.790    |       3.383     |  0.886  |
    | [Outlines](https://github.com/outlines-dev/outlines)                                                | unsloth/llama-3-8b-Instruct-bnb-4bit |    0.690    | 2.354<sup>*</sup> |  0.942  |
    | [OpenAI Structured Output](https://github.com/openai/openai-python)                                 |        gpt-4o-mini-2024-07-18        |    0.650    |       1.431     |  0.877  |
    | [LMFormatEnforcer](https://github.com/noamgat/lm-format-enforcer)                                   | unsloth/llama-3-8b-Instruct-bnb-4bit |    0.650    | 2.561<sup>*</sup> |  0.662  |

<sup>*</sup> NVIDIA GeForce RTX 4080 Super GPU

## 🏃 Run the benchmark

1. Install the requirements using `pip install -r requirements.txt`
1. Set the OpenAI api key: `export OPENAI_API_KEY=sk-...`
1. Run the benchmark using `python -m main run-benchmark`
1. Raw results are stored in the `results` directory.
1. Generate the results using:
    - Multilabel classification: `python -m main generate-results`
    - NER: `python -m main generate-results --task ner`
    - Synthetic data generation: `python -m main generate-results --task synthetic_data_generation`
1. To get help on the command line arguments, add `--help` after the command. Eg., `python -m main run-benchmark --help`

## 🧪 Benchmark methodology

1. Multi-label classification:
    - **Task**: Given a text, predict the labels associated with it.
    - **Data**:
        - Base data: [Alexa intent detection dataset](https://huggingface.co/datasets/AmazonScience/massive)
        - Benchmarking test is run using synthetic data generated by running: `python -m data_sources.generate_dataset generate-multilabel-data`.
        - The synthetic data is generated by sampling and combining rows from the base data to achieve multiple classes per row according to some distribution for num classes per row. See `python -m data_sources.generate_dataset generate-multilabel-data --help` for more details.
    - **Prompt**: `"Classify the following text: {text}"`
    - **Evaluation Metrics**:
        1. Reliability: The percentage of times the framework returns valid labels without errors. The average of all the rows `percent_successful` values.
        1. Latency: The 95th percentile of the time taken to run the framework on the data.
    - **Experiment Details**: Run each row through the framework `n_runs` number of times and log the percent of successful runs for each row.
1. Named Entity Recognition
    - **Task**: Given a text, extract the entities present in it.
    - **Data**:
        - Base data: [Synthetic PII Finance dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual)
        - Benchmarking test is run using a sampled data generated by running: `python -m data_sources.generate_dataset generate-ner-data`.
        - The data is sampled from the base data to achieve number of entities per row according to some distribution. See `python -m data_sources.generate_dataset generate-ner-data --help` for more details.
    - **Prompt**: `Extract and resolve a list of entities from the following text: {text}`
    - **Evaluation Metrics**:
        1. Reliability: The percentage of times the framework returns valid labels without errors. The average of all the rows `percent_successful` values.
        1. Latency: The 95th percentile of the time taken to run the framework on the data.
        1. Precision: The micro average of the precision of the framework on the data.
        1. Recall: The micro average of the recall of the framework on the data.
        1. F1 Score: The micro average of the F1 score of the framework on the data.
    - **Experiment Details**: Run each row through the framework `n_runs` number of times and log the percent of successful runs for each row.
1. Synthetic Data Generation
    - **Task**: Generate synthetic data similar according to a Pydantic data model schema.
    - **Data**:
        - Two level nested User details Pydantic schema.
    - **Prompt**: `Generate a random person's information. The name must be chosen at random. Make it something you wouldn't normally choose.`
    - **Evaluation Metrics**:
        1. Reliability: The percentage of times the framework returns valid labels without errors. The average of all the rows `percent_successful` values.
        1. Latency: The 95th percentile of the time taken to run the framework on the data.
        1. Variety: The percent of names that are unique compared to all names generated.
    - **Experiment Details**: Run each row through the framework `n_runs` number of times and log the percent of successful runs.

## 📊 Adding new data

1. Create a new pandas dataframe pickle file with the following columns:
    - `text`: The text to be sent to the framework
    - `labels`: List of labels associated with the text
    - See `data/multilabel_classification.pkl` for an example.
1. Add the path to the new pickle file in the `./config.yaml` file under the `source_data_pickle_path` key for all the frameworks you want to test.
1. Run the benchmark using `python -m main run-benchmark` to test the new data on all the frameworks!
1. Generate the results using `python -m main generate-results`

## 🏗️ Adding a new framework

The easiest way to create a new framework is to reference the `./frameworks/instructor_framework.py` file. Detailed steps are as follows:

1. Create a .py file in frameworks directory with the name of the framework. Eg., `instructor_framework.py` for the instructor framework.
1. In this .py file create a class that inherits `BaseFramework` from `frameworks.base`.
1. The class should define an `init` method that initializes the base class. Here are the arguments the base class expects:
    - `task` (str): the task that the framework is being tested on. Obtained from `./config.yaml` file. Allowed values are `"multilabel_classification"` and `"ner"`
    - `prompt` (str): Prompt template used. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - `llm_model` (str): LLM model to be used. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - `llm_model_family` (str): LLM model family to be used. Current supported values as `"openai"` and `"transformers"`. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - `retries` (int): Number of retries for the framework. Default is $0$. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - `source_data_picke_path` (str): Path to the source data pickle file. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - `sample_rows` (int): Number of rows to sample from the source data. Useful for testing on a smaller subset of data. Default is $0$ which uses all rows in source_data_pickle_path for the benchmarking. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - `response_model` (Any): The response model to be used. Internally passed by the benchmarking script.
1. The class should define a `run` method that takes three arguments:
    - `task`: The task that the framework is being tested on. Obtained from the `task` in the `./config.yaml` file. Eg., `"multilabel_classification"`
    - `n_runs`: number of times to repeat each text
    - `expected_response`: Output expected from the framework. Use default value of `None`
    - `inputs`: a dictionary of `{"text": str}` where `str` is the text to be sent to the framework. Use default value of empty dictionary `{}`
1. This `run` method should create another `run_experiment` function that takes `inputs` as argument, runs that input through the framework and returns the output.
1. The `run_experiment` function should be annotated with the `@experiment` decorator from `frameworks.base` with `n_runs`, `expected_resposne` and `task` as arguments.
1. The `run` method should call the `run_experiment` function and return the four outputs `predictions`, `percent_successful`, `metrics` and `latencies`.
1. Import this new class in `frameworks/__init__.py`.
1. Add a new entry in the `./config.yaml` file with the name of the class as the key. The yaml entry can have the following fields
    - `task`: the task that the framework is being tested on. Obtained from `./config.yaml` file. Allowed values are `"multilabel_classification"` and `"ner"`
    - `n_runs`: number of times to repeat each text
    - `init_kwargs`: all the arguments that need to be passed to the `init` method of the class, including those mentioned in step 3 above.

## 🧭 Roadmap

1. Framework related tasks:
    | Framework                                                                                           | Multi-label classification | Named Entity Recognition | Synthetic Data Generation |
    |-----------------------------------------------------------------------------------------------------|:--------------------------:|:------------------------:|:-------------------------:|
    | [OpenAI Structured Output](https://github.com/openai/openai-python)                                 |          ✅ OpenAI         |         ✅ OpenAI       |          ✅ OpenAI       |
    | [Instructor](https://github.com/jxnl/instructor)                                                    |          ✅ OpenAI         |         ✅ OpenAI       |          ✅ OpenAI       |
    | [Mirascope](https://github.com/mirascope/mirascope)                                                 |          ✅ OpenAI         |         ✅ OpenAI       |          ✅ OpenAI       |
    | [Fructose](https://github.com/bananaml/fructose)                                                    |          ✅ OpenAI         |      🚧 In Progress     |       🚧 In Progress     |
    | [Marvin](https://github.com/PrefectHQ/marvin)                                                       |          ✅ OpenAI         |         ✅ OpenAI       |          ✅ OpenAI       |
    | [Llamaindex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/openai_pydantic_program/) |          ✅ OpenAI         |         ✅ OpenAI       |          ✅ OpenAI       |
    | [Modelsmith](https://github.com/christo-olivier/modelsmith)                                         |          ✅ OpenAI         |      🚧 In Progress     |          ✅ OpenAI       |
    | [Outlines](https://github.com/outlines-dev/outlines)                                                |     ✅ HF Transformers     |      🚧 In Progress     |     ✅ HF Transformers   |
    | [LM format enforcer](https://github.com/noamgat/lm-format-enforcer)                                 |     ✅ HF Transformers     |    ✅ HF Transformers   |     ✅ HF Transformers   |
    | [Jsonformer](https://github.com/1rgs/jsonformer)                                                    |     ❌ No Enum Support     |        💭 Planning      |         💭 Planning       |
    | [Strictjson](https://github.com/tanchongmin/strictjson)                                             |   ❌ Non-standard schema   |  ❌ Non-standard schema |   ❌ Non-standard schema  |
    | [Guidance](https://github.com/guidance-ai/guidance)                                                 |         💭 Planning        |        💭 Planning      |         💭 Planning       |
    | [DsPy](https://dspy-docs.vercel.app/docs/building-blocks/typed_predictors)                          |         💭 Planning        |        💭 Planning      |         💭 Planning       |
    | [Langchain](https://python.langchain.com/v0.2/docs/tutorials/extraction/)                           |         💭 Planning        |        💭 Planning      |         💭 Planning       |
1. Others
    - [x] Latency metrics
    - [ ] CICD pipeline for benchmark run automation
    - [ ] Async run

## 💡 Contribution guidelines

Contributions are welcome! Here are the steps to contribute:

1. Please open an issue with any new framework you would like to add. This will help avoid duplication of effort.
1. Once the issue is assigned to you, pls submit a PR with the new framework!

## 🎓 Citation

To cite LLM Structured Output Benchmarks in your work, please use the following bibtex reference:

```bibtex
@software{marie_stephen_leo_2024_12327267,
  author       = {Marie Stephen Leo},
  title        = {{stephenleo/llm-structured-output-benchmarks: 
                   Release for Zenodo}},
  month        = jun,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.12327267},
  url          = {https://doi.org/10.5281/zenodo.12327267}
}
```

## 🙏 Feedback

If this work helped you in any way, please consider ⭐ this repository to give me feedback so I can spend more time on this project.
