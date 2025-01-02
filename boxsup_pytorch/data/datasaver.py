from pathlib import Path
from pydantic import BaseModel
from torch import Tensor


class BoxSupDataSaver(BaseModel):
    data: Tensor
    path: Path
