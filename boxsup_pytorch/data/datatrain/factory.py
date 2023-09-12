"""Factory module for wagon factory.

copyright Matti Kaupenjohann, 2023
"""


from datatrain.controlwagon import DataControlWagon
from datatrain.datawagons import (
    DataWagon,
)


class WagonFactory:
    """Factory Class for DataWagons."""

    def __init__(self):
        """Constuctor for WagonFactory."""
        self._creators = {}

    def register_wagon(self, key: str, creator: type):
        """Register method for wagon classes.

        Those classes inherit from DataWagon or are of type DataControlWagon.

        Args:
            format (str): String under this the class is registrated.
            creator (type): Class constructor which will be registered.
        """
        if not issubclass(creator, DataControlWagon) or not issubclass(creator, DataWagon):
            raise TypeError(f"{creator} is not of correct class")
        self._creators[key] = creator

    def get_wagon(self, key: str) -> type:
        """Return a specified registered wagon constructor.

        Args:
            key (str): String under the class is registered

        Raises:
            ValueError: If dataset with given string is not registered.

        Returns:
            type: wagon constructor
        """
        creator = self._creators.get(key)
        if not creator:
            raise ValueError(key)
        return creator


dataset_factory = WagonFactory()
dataset_factory.register_wagon("CONTROL", DataControlWagon)
