"""Factory module for dataset factory.

copyright Matti Kaupenjohann, 2022
"""

from boxsup_pytorch.data.dataset import (
    BoxSupDatasetAll,
    BoxSupDatasetUpdateIOU,
    BoxSupDatasetUpdateMask,
    BoxSupDatasetUpdateNet,
)


class DatasetFactory:
    """Factory Class for Datasets."""

    def __init__(self):
        """Constuctor for DatasetFactory."""
        self._creators = {}

    def register_dataset(self, key: str, creator: type):
        """Register method for dataset classes which inherit from BoxSupBaseDataset.

        Args:
            format (str): String under this the class is registrated.
            creator (BoxSupBaseDataset): Class constructor which will be registered.
        """
        self._creators[key] = creator

    def get_dataset(self, key: str) -> type:
        """Return a specified registered dataset constructor.

        Args:
            key (str): String under the class is registered

        Raises:
            ValueError: If dataset with given string is not registered.

        Returns:
            type: dataset constructor
        """
        creator = self._creators.get(key)
        if not creator:
            raise ValueError(key)
        return creator


dataset_factory = DatasetFactory()
dataset_factory.register_dataset("ALL", BoxSupDatasetAll)
dataset_factory.register_dataset("MASK", BoxSupDatasetUpdateMask)
dataset_factory.register_dataset("NET", BoxSupDatasetUpdateNet)
dataset_factory.register_dataset("IOU", BoxSupDatasetUpdateIOU)
