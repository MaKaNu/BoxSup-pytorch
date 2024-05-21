"""Module for loss functions."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn

from boxsup_pytorch.utils.check import check_shape_len_msg


__all__ = ["MixSoftmaxCrossEntropyLoss", "Losses", "get_segmentation_loss"]


class Losses:
    """Loss class which provides all loss methods of this project."""

    config: Optional[Dict[str, Any]]

    def __init__(self, config=None) -> None:
        """Construct of Losses class.

        Args:
            config (Dict[str, Any]): config dictionary provided via toml
        """
        self.classes = 6 if config is None else config["num_classes"]

    def overlapping_loss(self, box: Tensor, candidates: Tensor) -> Tensor:
        """Calculate how well candidates matches the bounding box.

        Args:
            box (torch.Tensor): boundingbox [1 x w x h]
            candidates (torch.Tensor): candidate or array of candidates [N x w x h]

        Returns:
            torch.Tensor: the calculated loss
        """
        assert len(box.shape) == 2, check_shape_len_msg(2)
        assert len(candidates.shape) in [2, 3], check_shape_len_msg((2, 3))

        if len(candidates.shape) == 3:
            N = candidates.shape[0]
            classes = candidates.max(1).values.max(1).values
            binary_candidates = candidates.div(classes[..., None, None])
        else:
            N = 1
            classes = candidates.max()
            binary_candidates = candidates.div(classes)
        return (
            1
            / N
            * torch.sum(
                (1 - self.inter_over_union(binary_candidates, box))
                * self._compare_labels(candidates, box)
            )
        )

    def batch_overlapping_loss(
        self, ious: Tensor, cands_classes: Tensor, bboxes_classes: Tensor
    ) -> Tensor:
        """Calculate how well candidates matches the bounding box for a batch.

        Args:
            ious (torch.Tensor): ious of candidates over bboxes precalculated [N x M x K]
                N: Batchsize
                M: Max BBoxes of Dataset
                K: Specified Number of Candidates
            bboxes (torch.Tensor): boundingbox [N x M x w x h]
                N: Batchsize
                M: Max BBoxes of Dataset
            candidates (torch.Tensor): candidate or array of candidates [N x K x w x h]
                N: Batchsize
                K: Specified Number of Candidates

        Returns:
            torch.Tensor: the calculated loss [scalar]
        """
        compared_mask = self._compare_labels(cands_classes, bboxes_classes)
        masks_axis = 2

        N = ious.shape[2]

        return (1 / N) * torch.sum((1.0 - ious) * compared_mask, dim=masks_axis)

    def regression_loss(self, est_mask: Tensor, lab_mask: Tensor) -> Tensor:
        """Calculate logistic regression.

        Args:
            est_mask (torch.Tensor): Estimated mask [N x M x w x h]
                N: Batchsize
                M: num classes estimated by the network
            lab_mask (torch.Tensor): Label mask [N x P x w x h]
                N: Batchsize
                P: num selected maskes

        Returns:
            torch.Tensor: logistic regression loss
        """
        # TODO test the condition, here is something wrong
        if len(lab_mask.shape) == 4:  # More than 1 candidate
            # Prepare est_mask ffor vectorization.
            # To achieve vectorization we need to expand the est_mask after M.
            # To achieve it we unsqueeze first at the second dim
            # and Expand at this second dim by the size P of lab_mask.
            est_mask_expanded = est_mask.unsqueeze(2).expand(-1, -1, lab_mask.size(1), -1, -1)
        else:
            lab_mask = lab_mask.reshape(1, *lab_mask.shape)
            est_mask_expanded = 1  # THIS IS DUMMY AND MAKES NO SENSE
        loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
        return loss(est_mask_expanded, lab_mask.long()).mean(dim=(2, 3))  # dim=(1, 2)) one cand?

    def weighted_loss(self, o_loss: Tensor, r_loss: Tensor, weight: float = 3.0) -> Tensor:
        """Calculate the weighted loss.

        Args:
            o_loss (Tensor): single overlapping or array loss
            r_loss (Tensor): single regression or array loss
            weight (np.float64): weighting factor

        Returns:
            np.float64: weighted loss
        """
        num_cands = r_loss.shape[1]
        num_bboxs = o_loss.shape[1]
        exp_r_loss = r_loss.unsqueeze(1).expand(-1, num_bboxs, -1)
        exp_o_loss = o_loss.unsqueeze(2).expand(-1, -1, num_cands)
        return exp_o_loss + weight * exp_r_loss

    @staticmethod
    def _compare_labels(cands_classes: Tensor, bbox_classes: Tensor) -> Tensor:
        """Check if label of box is equal to labels of candidates.

        Args:
            box (torch.Tensor): BoundingBox
            candidates (torch.Tensor): Candidate or array of candidates

        Returns:
            torch.Tensor[bool]: True if labels are equal
        """
        if cands_classes.count_nonzero() == 0:
            return torch.ones((*bbox_classes.shape, cands_classes.shape[2]), dtype=torch.bool)
        num_cands = cands_classes.shape[1]
        num_bboxes = bbox_classes.shape[1]
        expanded_bbox_classes = bbox_classes.unsqueeze(2).expand(-1, -1, num_cands)
        expanded_cands_classes = cands_classes.unsqueeze(1).expand(-1, num_bboxes, -1)
        return expanded_bbox_classes == expanded_cands_classes

    @staticmethod
    def inter_over_union(pred: Tensor, target: Tensor) -> Tensor:
        """Calculate Intersection over Union.

        Args:
            pred (torch.Tensor): The prediction(s) px array Shape: [N ...]
            target (torch.Tensor): The target px array Shape: [1 ...]

        Returns:
            torch.Tensor: the divison of sum of intersection px by union px
        """
        dim = (1, 2) if len(pred.shape) == 3 else (0, 1)

        class_idx = target.max()
        expanded_target = target.expand_as(pred) / class_idx
        inter_mask = expanded_target.mul(pred)
        inter = inter_mask.sum(dim=dim)
        traget_area = expanded_target.sum(dim)
        pred_area = pred.sum(dim)
        union = traget_area + pred_area - inter
        return inter / union


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


def get_segmentation_loss(**kwargs):
    return MixSoftmaxCrossEntropyLoss(**kwargs)
