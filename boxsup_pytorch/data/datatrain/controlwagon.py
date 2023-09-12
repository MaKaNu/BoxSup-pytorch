from __future__ import annotations

from typing import Dict, List

import torch

from boxsup_pytorch.data.datatrain.datawagons import DataWagon
from boxsup_pytorch.data.datatrain.factory import WagonFactory


class DataControlWagon:
    def __init__(self, nclutch: List[str]) -> None:
        self.nclutch: List[str] = nclutch
        self.connected: Dict[str, DataWagon] | None = None

    def connect(self, wagon_factory: WagonFactory) -> None:
        self.connected = {
            wagon_name: wagon_factory.get_wagon(wagon_name)() for wagon_name in self.nclutch
        }

    def load_wagon(self, wagon: str, data: torch.Tensor) -> None:
        raise NotImplementedError()

    def unload_wagon(self, wagon: str) -> torch.Tensor:
        raise NotImplementedError()

    def pre_trail_check(self) -> bool:
        raise NotImplementedError()
