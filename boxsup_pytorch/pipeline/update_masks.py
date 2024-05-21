"""Process Greedy Start Module."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

from torch import nn
from boxsup_pytorch.config import GLOBAL_CONFIG

from boxsup_pytorch.core.strat_factory import strat_factory
from boxsup_pytorch.data.datacontainer import MaskDataContainer
from boxsup_pytorch.data.dataloader import BoxSupDataloader
from boxsup_pytorch.pipeline.error_calc import ErrorCalc
from boxsup_pytorch.pipeline.strats import GreedyStrat, MiniMaxStrat

train_logger = getLogger("train")


@dataclass
class UpdateMasks:
    """Update Dataset image masks.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    network: nn.Module
    dataloader: BoxSupDataloader

    def update_logic(self) -> nn.Module:
        global train_logger
        # TODO do we need Val? Actually not but we keep it and elimnate ladder.
        dataloaders = self.dataloader.get_data_loader()

        stage = "val"
        if "train" in dataloaders.keys():
            stage = "train"
            train_logger.info(f"Updating Masks for stage '{stage}'")

        # Load Strategy
        strat_loader = strat_factory.get_strat(GLOBAL_CONFIG.strat)
        strat = strat_loader(network=self.network)
        train_logger.info(f"run strategy {strat.name}")

        for images, masks, masks_cls, masks_iou, bbox_cls in dataloaders[stage]:
            data = MaskDataContainer(images, masks, masks_cls, masks_iou, bbox_cls)
            data.check_data()
            self.network = strat.update(data)

        return self.network

        if False:
            self.calc_error.set_inputs(input_values)
            self.calc_error.update()

            self.strat.set_inputs(input_values)
            self.strat.update()
            return self.strat.get_outputs()
        else:
            raise ValueError(
                f"""
            stage 'train' is not in dataset.
            following stages are used: {dataloaders.keys()}
            """
            )

    def __check_for_saved_iou(self):
        pass
