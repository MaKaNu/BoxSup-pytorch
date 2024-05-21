"""Main module."""

import logging

import torch
from torch.backends import cudnn

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.data.dataloader import BoxSupDataloader
from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.pipeline.process_runner import ProcessRunner
from boxsup_pytorch.pipeline.update_masks import UpdateMasks
from boxsup_pytorch.pipeline.update_net import UpdateNetwork


def main():
    """Call the main routine."""
    logger = logging.getLogger(__name__)
    logger.debug("RUN: launch.py - main()")

    # load config data
    nclass = GLOBAL_CONFIG.num_classes

    dataloader_mask = BoxSupDataloader("MASK")
    dataloader_net = BoxSupDataloader("NET")

    # Specify the device the network is running on
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    # Prepare Network
    network = FCN8s(nclass=nclass, device=device).to(device)

    # Load Processes
    process_runner = ProcessRunner(network=network)

    processes = [
        UpdateMasks(network, dataloader_mask),
        UpdateNetwork(network, dataloader_net),
    ]

    process_runner.run(processes)


if __name__ == "__main__":
    main()  # pragma: no cover
