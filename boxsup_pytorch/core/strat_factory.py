"""Factory module for strat factory.

copyright Matti Kaupenjohann, 2024
"""

from boxsup_pytorch.pipeline.strats import (
    GreedyStrat,
    MiniMaxStrat,
)


class StratFactory:
    """Factory Class for Strats."""

    def __init__(self):
        """Constuctor for StratFactory."""
        self._creators = {}

    def register_strat(self, key: str, creator: type):
        """Register method for strat classes which inherit from BaseStrat.

        Args:
            format (str): String under this the class is registrated.
            creator (BaseStrat): Class constructor which will be registered.
        """
        self._creators[key] = creator

    def get_strat(self, key: str) -> type:
        """Return a specified registered strat constructor.

        Args:
            key (str): String under the class is registered

        Raises:
            ValueError: If dataset with given string is not registered.

        Returns:
            type: strat constructor
        """
        creator = self._creators.get(key)
        if not creator:
            raise ValueError(key)
        return creator


strat_factory = StratFactory()
strat_factory.register_strat("GREEDY", GreedyStrat)
strat_factory.register_strat("MINMAX", MiniMaxStrat)
