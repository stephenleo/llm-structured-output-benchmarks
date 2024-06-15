# LLM Structured Output Benchmarks

## Run the benchmark
1. Install the requirements using `pip install -r requirements.txt`
1. Set the OpenAI api key: `export OPENAI_API_KEY=sk-...`
1. Run the benchmark using `python -m main`
1. Results are stored in the `data` directory.

## Add a new framework
The easiest way to create a new framework is to reference the `./frameworks/instructor_framework.py` file. Detailed steps are as follows:

1. Create a .py file in frameworks directory with the name of the framework.
1. In this .py file create a class that inherits `BaseFramework` from `frameworks.base`.
1. The class should define an `init` method that initializes the base class with the expected arguments:
    - name (str): name of the task that the framework is being tested on. 
    - prompt (str): Prompt template used. Obtained from the `init_kwargs` in the `./config.yaml` file. 
    - llm_model (str): LLM model to be used. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - retries (int): Number of retries for the framework. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - source_data_picke_path (str): Path to the source data pickle file. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - sample_rows (int): Number of rows to sample from the source data. Useful for testing on a smaller subset of data. Set the default value to $0$ so that if it's not specified, all rows in source_data_pickle_path are used. Obtained from the `init_kwargs` in the `./config.yaml` file.
    - response_model (Any): The response model to be used. Internally passed by the benchmarking script. Set the default value to `None`.
1. The class should define a `run` method that takes three arguments:
    1. `inputs`: a dictionary of `{"text": str}` where `str` is the text to be sent to the framework
    1. `n_runs`: number of times to repeat each text
    1. `expected_response`: Output expected from the framework
1. This `run` method should create another `run_experiment` function that takes `inputs` as argument and runs that input through the framework and returns the output.
1. The `run_experiment` function should be annotated with the `@experiment` decorator from `frameworks.base` with `n_runs` and `expected_resposne` as arguments.
1. The `run` method should call the `run_experiment` function and return the three outputs `predictions`, `percent_successful` and `accuracy`.
1. Import this new class in `frameworks/__init__.py`.
1. Add a new entry in the `./config.yaml` file with the name of the class as the key. The yaml entry can have the following fields
    1. `name`: name of the task that the framework is being tested on
    1. `n_runs`: number of times to repeat each text
    1. `init_kwargs`: any additional arguments that need to be passed to the `init` method of the class.

