from typing import Any

from frameworks.base import experiment
from frameworks.formatron_framework import FormatronFramework
from frameworks.fructose_framework import FructoseFramework
from frameworks.instructor_framework import InstructorFramework
from frameworks.llamaindex_framework import LlamaIndexFramework
from frameworks.lm_format_enforcer_framework import LMFormatEnforcerFramework
from frameworks.marvin_framework import MarvinFramework
from frameworks.mirascope_framework import MirascopeFramework
from frameworks.modelsmith_framework import ModelsmithFramework
from frameworks.outlines_framework import OutlinesFramework
from frameworks.vanilla_openai_framework import VanillaOpenAIFramework


def factory(class_name: str, *args, **kwargs) -> Any:
    """Factory function to create an instance of a framework class

    Args:
        class_name (str): name of the class to instantiate

    Raises:
        ValueError: If the class name is not found in the globals

    Returns:
        Any: An object of the requested framework class
    """
    if class_name in globals():
        return globals()[class_name](*args, **kwargs)
    else:
        raise ValueError(f"Invalid class name: {class_name}")
