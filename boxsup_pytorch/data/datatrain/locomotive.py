"""locomotive module.

provided classes:
    DataLocomotive
"""
from __future__ import annotations

from pathlib import Path

from datatrain.controlwagon import DataControlWagon
from datatrain.factory import WagonFactory


class DataLocomotive:
    def __init__(self, control_wagon: str, id: int, wagon_plan: Path) -> None:
        self.clutch: str = control_wagon
        self.id: int = id
        self.wagon_plan: Path = wagon_plan
        self.connected: DataControlWagon | None = None

    def connect(self, wagon_factory: WagonFactory) -> None:
        with open(self.wagon_plan, "r") as plan_file:
            wagon_plan_list = plan_file.read().split("\n")
        self.connected = wagon_factory.get_wagon(self.clutch)(wagon_plan_list)

    def to_next_station(self, station):
        raise NotImplementedError(station)
