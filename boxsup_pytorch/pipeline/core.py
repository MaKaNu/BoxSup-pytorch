"""Pipeline core module."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

from PIL.Image import Image
from torch import nn, Tensor

from boxsup_pytorch.data.dataloader import BoxSupDataloader


@runtime_checkable
class PipelineProcess(Protocol):
    """Process of a Pipeline which follows logic."""

    network: nn.Module
    dataloader: BoxSupDataloader

    def update_logic(self, **kwargs) -> nn.Module:
        """Run the Update to follow the logic.

        Args:
            network (nn.Module): Every process is network bound so the network is mandatory.
        """
        ...


@runtime_checkable
class PipelineDataProcess(Protocol):
    """Process of a Pipeline which follows data."""

    def update(self):
        """Update the internal data."""
        ...  # pragma: no cover

    def set_inputs(self, inputs: Dict[str, Tensor | Image]):
        """Set Inputs of Process."""
        ...  # pragma: no cover

    def get_outputs(self) -> Dict[str, Optional[Tensor]] | None:
        """Get Outputs of Process."""
        ...  # pragma: no cover


class DataPipeline:
    """Pipeline class wich is able to run the PipelineProcesses with specified input."""

    def __init__(self, input, config: List[PipelineDataProcess | DataPipeline]) -> None:
        """Initialze Class Pipeline.

        Args:
            input (_type_): _description_
            config (List[PipelineProcess], optional): List of PipelineProcess Instances.
                Defaults to GLOBAL_PIPELINE_CONFIG.
        """
        self.config = config
        self._get_process_inputs(config[0])
        self.input = input
        self.input_names = []

    def run(self):
        """Run the Pipeline itarative."""
        pass

    def _get_process_inputs(self, process: PipelineDataProcess | DataPipeline):
        inputs = [attr for attr in process.__dict__.keys() if attr.startswith("in_")]
        self.input_names = inputs
