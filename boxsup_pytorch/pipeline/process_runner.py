"""process_runner.py module.

<include_copyright>

This Module implements the ProcessRunner class.
The class provides the algorithm network to each of the PipelineProcesses.
The run method of the class iterates over the processes.
"""

from typing import List

from torch import nn

from boxsup_pytorch.pipeline.core import PipelineProcess


class ProcessRunner:
    """Runner to activate the update logic of the algorithm processes.

    The Process Runner holds the network between the iterating process runs.

    Implemented methods:

    - __init__
    - run
    """

    def __init__(self, network: nn.Module) -> None:
        """Initiate a Process Runner object."""
        self.network: nn.Module = network

    def run(self, processes: List[PipelineProcess]):
        """Run the provided list of Pipelineprocesses.

        Args:
            processes (List[PipelineProcess]): Holds the looping processes.
        """
        # Guards
        if self.network is None:
            raise RuntimeError("self.network is not initialized")
        for process in processes:
            process.network = self.network
            self.network = process.update_logic()
