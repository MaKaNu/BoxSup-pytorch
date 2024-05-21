"""Process Greedy Start Module."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.data.dataloader import BoxSupDataloader

# from boxsup_pytorch.data.dataloader import train_model # TODO need some fixes
from boxsup_pytorch.data.dataset import BoxSupDatasetUpdateNet
from boxsup_pytorch.model.network import FCN8s


@dataclass
class UpdateNetwork:
    network: nn.Module
    dataloader: BoxSupDataloader

    def update_logic(self, network: nn.Module):
        return network
