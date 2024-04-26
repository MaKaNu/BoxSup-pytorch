"""Module with specific assert and error messages."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def check_init_msg() -> str:
    """Create message for not initialized pipeline values.

    Because they are typed as Optional they need to be checked for None.

    Args:
        value (Any): The Pipeline Process Value

    Returns:
        str: assert message
    """
    message = "Pipeline Process property is not initialized!"
    return message


def check_exists_msg(value: str) -> str:
    """Create message value which should be included into dict.

    Args:
        value (str): The input string for the Pipeline Process Value

    Returns:
        str: assert message
    """
    message = f"Inputattribute '{value}' for Pipeline Process is not set!"
    return message


def check_path_exists_msg(value: Path) -> str:
    """Create message with value as path which doesn't exists.

    Args:
        value (str): Path which doesn't exists.

    Returns:
        str: Combined ErrorMessage.
    """
    message = f"Path '{value}' does not exists!"
    return message


def check_shape_len_msg(expected: int | Tuple) -> str:
    """Create message value for length of shape asserts.

    Args:
        value (str): Tensor which is checked
        expected (int): expected len of tensors shape

    Returns:
        str: assert message
    """
    message = f"Tensor does not have the expected shape length: {expected}"
    return message
