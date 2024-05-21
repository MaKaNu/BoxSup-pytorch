"""ErrorCalc Pipeline Module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch import Tensor

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.utils.check import check_exists_msg, check_init_msg
from boxsup_pytorch.utils.losses import Losses


@dataclass
class ErrorCalc:
    """The ErrorCalculation Collection Class."""

    losses: Losses = Losses()

    def network_inference(self, network: nn.Module, image: Tensor) -> Tensor:
        # Setup Model
        network.eval()

        # Begin Inference
        if not isinstance(image, Tensor):
            raise RuntimeError("Image is not Tensor!")
        image = image.to(network.device)
        # image = image[None, :]  # Add Dummy Batch Dim
        with torch.no_grad():
            outputs = network(image)
        return outputs[0].cpu()

    def _get_overlapping_loss(self) -> Callable:
        if GLOBAL_CONFIG.batchsize > 1:
            return self.losses.batch_overlapping_loss
        return self.losses.overlapping_loss

    def get_overlapping_loss(self) -> Callable:
        if GLOBAL_CONFIG.batchsize > 1:
            return self.losses.batch_overlapping_loss
        return self.losses.overlapping_loss
