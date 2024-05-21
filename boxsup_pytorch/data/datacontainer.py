"""Data Containers.

This module provides dataclass container which allows easy type checking to runtime.

The check_data method is integrated via the BaseContainer.
Every specific container is inherited by BaseContainer.
"""

from dataclasses import dataclass, fields

from torch import Tensor


@dataclass
class BaseContainer:
    """Base Container.

    provides the the check_data method for the inhertited classes.
    """

    def check_data(self):
        """Analyses the types of input arguments of the dataclass.

        Raises:
            RuntimeError: is raised if onee field does not fit the expected type.
        """
        container_fields = fields(self)
        for field in container_fields:
            if not isinstance(getattr(self, field.name), field.type):
                raise RuntimeError(f"Data for {field.name} is not correct type!")


@dataclass
class MaskDataContainer(BaseContainer):
    """Data Container for Mask Process."""

    images: Tensor
    masks: Tensor
    masks_cls: Tensor
    masks_iou: Tensor
    bbox_cls: Tensor
