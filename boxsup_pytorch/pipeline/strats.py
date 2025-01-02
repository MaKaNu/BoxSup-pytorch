"""Process Greedy Start Module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import torch
from torch import nn
import torch.nn.functional as F

from boxsup_pytorch.data.datacontainer import MaskDataContainer
from boxsup_pytorch.model import network
from boxsup_pytorch.pipeline.error_calc import ErrorCalc
from boxsup_pytorch.utils.check import check_init_msg
from boxsup_pytorch.utils.common import squash_mask_layer
from boxsup_pytorch.utils.losses import Losses


@dataclass
class BaseStrat(ABC):
    """Base Strat class."""

    network: nn.Module
    error_calc = ErrorCalc()
    losses = Losses()

    @abstractmethod
    def update(self):
        """Abstract method of interface model."""
        ...

    def _check_input(self, data: MaskDataContainer):
        if not (
            hasattr(data, "images")
            or hasattr(data, "masks")
            or hasattr(data, "maks_cls")
            or hasattr(data, "maks_iou")
            or hasattr(data, "bbox_cls")
        ):
            raise RuntimeError(check_init_msg())

    def _calculate_loss(self, data: MaskDataContainer):
        overlapping_loss = self.error_calc.get_overlapping_loss()

        network_output = self.error_calc.network_inference(self.network, data.images)
        overlap_loss = overlapping_loss(data.masks_iou, data.masks_cls, data.bbox_cls)
        regression_loss = self.losses.regression_loss(network_output, data.masks)
        weighted_loss = self.losses.weighted_loss(overlap_loss, regression_loss)

        return weighted_loss

    def _sort_loss(self, loss, data: MaskDataContainer):
        # Create idx for masks which are only zero
        ignore_masks_idx = (data.masks.sum(3).sum(2) == 0).nonzero()

        if not loss.any():
            raise RuntimeError("Calculation failed!")

        # replace the weights which have zero masks with max flaot value
        batch_idxs = ignore_masks_idx[:, 0]
        mask_idxs = ignore_masks_idx[:, 1]
        loss[batch_idxs, :, mask_idxs] = torch.finfo(torch.float32).max

        # sort weights along candidate axis
        sorted_loss_idx = loss.argsort(dim=2)

        return sorted_loss_idx

    # TODO REMOVE SOON
    def _reduce_masks(self, stacked_masks):
        def first_nonzero(x, axis=0):
            nonz = x > 0
            return ((nonz.cumsum(axis) == 1) & nonz).max(axis)

        _, idx = first_nonzero(stacked_masks)
        mask = F.one_hot(idx, stacked_masks.shape[0]).permute(2, 0, 1) == 1
        target = torch.zeros_like(stacked_masks)
        target[mask] = stacked_masks[mask]
        return target.sum(dim=0)

    # TODO REMOVE SOON
    def _reduce_masks2(self, stacked_masks):
        if stacked_masks.shape[0] == 1:
            return stacked_masks

        old_non_mask = torch.ones_like(stacked_masks[0])
        merged_masks = []
        for idx, mask in enumerate(stacked_masks):
            non_mask = (mask == 0) * old_non_mask
            if not idx:
                merged_masks.append(mask)
                continue
            merged_masks.append(mask * old_non_mask)
            old_non_mask = non_mask

        return torch.stack(merged_masks).sum(0)


@dataclass
class GreedyStrat(BaseStrat):
    """Greedy Straetegy aims to get always the best candidate."""

    name: str = "greedy"

    @final
    def update(self, data: MaskDataContainer) -> nn.Module:
        self._check_input(data)
        weighted_loss = self._calculate_loss(data)
        sorted_loss_idx = self._sort_loss(weighted_loss, data)

        selected_masks = []
        for idx, batch in enumerate(data.bbox_cls):
            batch_masks = data.masks[idx]
            width, height = batch_masks.shape[1:]

            # Select valid bboxes and get bbox class
            bbox_idx = batch.nonzero().squeeze()
            bbox_classes = batch[bbox_idx]

            if bbox_idx.shape:
                selected_masks_idx = sorted_loss_idx[idx, bbox_idx][:, 0]
                stacked_labelmasks = batch_masks[selected_masks_idx]
                num_masks = stacked_labelmasks.shape[0]
                exp_bbox_classes = bbox_classes[:, None, None].expand(num_masks, width, height)
            else:
                selected_masks_idx = sorted_loss_idx[idx, bbox_idx][0]
                num_masks = 1
                stacked_labelmasks = batch_masks[selected_masks_idx].reshape(
                    [num_masks, width, height]
                )
                exp_bbox_classes = bbox_classes.unsqueeze(0)[:, None, None].expand(
                    num_masks, width, height
                )

            # Reduce all masks to single mask for training
            label_mask = squash_mask_layer(stacked_labelmasks, exp_bbox_classes)
            selected_masks.append(label_mask)

        # TODO implement the following
        # save_training_data for update of Network

        return network


@dataclass
class MiniMaxStrat(BaseStrat):
    name: str = "minimax"
    error_calc = ErrorCalc()

    def update(self):
        assert self.in_bboxes is not None, check_init_msg()
        assert self.in_masks is not None, check_init_msg()

        stacked_labelmasks = torch.zeros(self.in_bboxes.shape)
        for bbox_idx in range(self.in_bboxes.shape[1]):
            input = {
                "image": self.in_images,
                "bbox": self.in_bboxes[bbox_idx],
                "masks": self.in_masks,
            }
            self.error_calc.set_inputs(input)
            self.error_calc.update()
            output = self.error_calc.get_outputs()
            _, idx = torch.topk(output["loss"], 5)
            random_idx = torch.randint(idx.shape[0], (1,))
            selected_idx = idx[random_idx]
            selected_mask = self.in_masks[selected_idx[0]]
            class_of_bbox = torch.max(self.in_bboxes[bbox_idx])
            zero_mask = selected_mask != 0
            stacked_labelmasks[bbox_idx, zero_mask] = (
                selected_mask[zero_mask] / selected_mask[zero_mask] * class_of_bbox
            )
            self.out_labelmask = self._reduce_masks(stacked_labelmasks)
