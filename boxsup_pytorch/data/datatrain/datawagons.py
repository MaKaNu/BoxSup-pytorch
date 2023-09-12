from abc import ABC, abstractmethod


class DataWagon(ABC):
    @abstractmethod
    def check_load(self) -> bool:
        pass


class DataWagonBatch(DataWagon):
    pass


class DataWagonImage(DataWagon):
    pass


class DataWagonBBox(DataWagon):
    pass


class DataWagonMask(DataWagon):
    pass
