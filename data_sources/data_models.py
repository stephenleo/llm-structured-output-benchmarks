from enum import Enum

from pydantic import BaseModel


# Multilabel classification model
def multilabel_classification_model(multilabel_classes):
    class MultiLabelClassification(BaseModel):
        classes: list[
            Enum(
                "MultilabelClasses", {name: name for name in multilabel_classes}
            )
        ]

    return MultiLabelClassification
