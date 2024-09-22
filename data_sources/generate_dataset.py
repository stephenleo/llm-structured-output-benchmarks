# python -m data_sources.generate_dataset generate-multilabel-data
import json
import random
from collections import Counter

import pandas as pd
from datasets import load_dataset
from loguru import logger
from rich.progress import track
from typer import Option, Typer

app = Typer()


def download_default_classification_dataset(
    text_column: str = "utt", label_column: str = "intent"
) -> pd.DataFrame:
    """Download the default classification dataset from Hugging Face's datasets library. Defaults to the AmazonScience/massive dataset.

    Args:
        text_column (str, optional): The column name for the text data. Defaults to "utt" as defined in the default dataset.
        label_column (str, optional):  The column name for the labels. Defaults to "intent" as defined in the default dataset.

    Returns:
        pd.DataFrame: A pandas DataFrame with the text and label columns.
    """
    logger.info(
        "Downloading source data from https://huggingface.co/datasets/AmazonScience/massive"
    )
    dataset = load_dataset("AmazonScience/massive", "en-US", split="test")
    dataset = dataset.select_columns([text_column, label_column])

    logger.info("Processing the text and label columns")
    dataset = dataset.rename_columns({text_column: "text", label_column: "class_label"})
    class_names = dataset.features["class_label"].names

    dataset = dataset.map(
        lambda row: {"label": class_names[row["class_label"]]},
        remove_columns=["class_label"],
    )

    return dataset.to_pandas()


def label_entity(row: dict) -> dict:
    """Convert rows to entities.

    Args:
        row (dict): row should have 'text' and 'ner_label' fields. The 'ner_label' column should be a list of dictionaries with 'start' position, 'end' position, and the NER 'label' keys.

    Returns:
        dict: A dictionary with the NER labels as keys and a list of entities as values.
    """
    entities = {}
    text = row["text"]
    for span in json.loads(row["ner_label"]):
        label = span["label"]
        if label == "date":
            # date entities are not useful in this dataset
            continue

        entity = text[span["start"] : span["end"]]

        if label not in entities:
            # New entity
            entities[label] = [entity]
        if (label in entities) and (entity not in entities[label]):
            # Add to existing entity if not duplicate
            entities[label].append(entity)
        else:
            # Duplicates
            continue

    return entities


def download_default_ner_dataset(
    text_column: str = "generated_text", label_column: str = "pii_spans"
) -> pd.DataFrame:
    """Download the default NER dataset from Hugging Face's datasets library. Defaults to the gretelai/synthetic_pii_finance_multilingual dataset.

    Args:
        text_column (str, optional): The column name for the text data. Defaults to "generated_text".
        label_column (str, optional): The column name for the labels. Defaults to "pii_spans".

    Returns:
        pd.DataFrame: A pandas DataFrame with the text and labels columns.
    """
    logger.info(
        "Downloading source data from https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual"
    )
    dataset = load_dataset("gretelai/synthetic_pii_finance_multilingual", split="test")
    dataset = dataset.filter(lambda example: example["language"] == "English")
    dataset = dataset.select_columns([text_column, label_column])
    dataset = dataset.rename_columns({text_column: "text", label_column: "ner_label"})

    logger.info("Processing the text and label columns")
    df = dataset.to_pandas()
    df["labels"] = df.apply(label_entity, axis=1)

    # Processing names
    df = df[
        ~df["labels"].apply(lambda x: "first_name" in x or "last_name" in x)
    ].reset_index(drop=True)
    
    df["labels"] = df["labels"].apply(
        lambda x: {("person_name" if k == "name" else k): v for k, v in x.items()}
    )

    df = df.drop(columns=["ner_label"])

    return df


@app.command()
def generate_multilabel_data(
    source_data_pickle_path: str = Option(
        None,
        help="Path to the source pandas dataframe pickle file. Must contain atleast two columns: one for text and one for labels",
    ),
    source_dataframe_text_column: str = Option(
        "text", help="The column name for the text data."
    ),
    source_dataframe_label_column: str = Option(
        "label", help="The column name for the labels."
    ),
    dest_num_rows: int = Option(
        100, help="Number of rows to keep in the final dataframe."
    ),
    dest_label_distribution: str = Option(
        default='{"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15}',
        help="JSON string of the probability of having each number of entities per row.",
    ),
) -> None:
    """Generate synthetic multilabel classification data by combining rows from a source dataset."""
    dest_label_distribution = json.loads(dest_label_distribution)
    dest_label_distribution = {int(k): v for k, v in dest_label_distribution.items()}
    if not source_data_pickle_path:
        logger.info("No source data pickle file provided, downloading default dataset")
        source_dataframe = download_default_classification_dataset()
    else:
        logger.info("Loading the source data from the provided pickle file")
        source_dataframe = pd.read_pickle(source_data_pickle_path)

    logger.info(f"Generating {dest_num_rows} synthetic rows")

    multilabel_data = {"text": [], "labels": []}
    for _ in track(range(dest_num_rows), description="Generating rows"):
        num_rows = random.choices(
            list(dest_label_distribution.keys()), list(dest_label_distribution.values())
        )[0]
        random_rows = source_dataframe.sample(num_rows)

        multilabel_data["text"].append(
            ". ".join(random_rows[source_dataframe_text_column].tolist())
        )
        multilabel_data["labels"].append(
            random_rows[source_dataframe_label_column].tolist()
        )

    multilabel_df = pd.DataFrame(multilabel_data)

    label_counter = Counter([len(label) for label in multilabel_df["labels"]])
    label_counter = pd.DataFrame.from_records(
        list(label_counter.items()), columns=["num_labels", "num_rows"]
    ).sort_values("num_labels")

    logger.info(f"Number of rows for each number of labels:\n{label_counter.head()}")

    logger.info(f"First 5 rows:\n{multilabel_df.head()}")
    multilabel_df.to_pickle("data/multilabel_classification.pkl")
    logger.info("Saved multilabel data to: data/multilabel_classification.pkl")


@app.command()
def generate_ner_data(
    source_data_pickle_path: str = Option(
        None,
        help="Path to the source pandas dataframe pickle file. Must contain atleast two columns: one for text and one for labels.",
    ),
    source_dataframe_text_column: str = Option(
        "text", help="The column name for the text data."
    ),
    source_dataframe_label_column: str = Option(
        "labels", help="The column name for the labels."
    ),
    dest_num_rows: int = Option(
        100, help="Number of rows to keep in the final dataframe."
    ),
    dest_label_distribution: str = Option(
        default='{"1": 0.3, "2": 0.25, "3": 0.20, "4": 0.15, "5": 0.10}',
        help="JSON string of the probability of having each number of entities per row.",
    ),
) -> None:
    """Generate synthetic NER data by combining rows from a source dataset."""
    dest_label_distribution = json.loads(dest_label_distribution)
    dest_label_distribution = {int(k): v for k, v in dest_label_distribution.items()}
    if not source_data_pickle_path:
        logger.info("No source data pickle file provided, downloading default dataset")
        source_dataframe = download_default_ner_dataset()
    else:
        logger.info("Loading the source data from the provided pickle file")
        source_dataframe = pd.read_pickle(source_data_pickle_path)

    logger.info(f"Generating {dest_num_rows} synthetic rows")

    source_dataframe["num_entities"] = source_dataframe[
        source_dataframe_label_column
    ].apply(lambda x: len(x.keys()))

    ner_data = []
    for value, fraction in track(dest_label_distribution.items()):
        num_rows = int(dest_num_rows * fraction)

        subset_df = source_dataframe[source_dataframe["num_entities"] == value]
        sampled_subset = subset_df.sample(n=num_rows, random_state=1)

        ner_data.append(sampled_subset)

    ner_df = pd.concat(ner_data).sample(frac=1).reset_index(drop=True)

    label_counter = Counter(ner_df["num_entities"])
    label_counter = pd.DataFrame.from_records(
        list(label_counter.items()), columns=["num_labels", "num_rows"]
    ).sort_values("num_labels")

    logger.info(f"Number of rows for each number of labels:\n{label_counter.head()}")

    ner_df = ner_df.drop(columns=["num_entities"])

    logger.info(f"First 5 rows:\n{ner_df.head()}")
    ner_df.to_pickle("data/ner.pkl")
    logger.info("Saved NER data to: data/ner.pkl")


if __name__ == "__main__":
    app()
