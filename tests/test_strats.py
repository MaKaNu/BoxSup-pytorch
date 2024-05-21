"""Test scripts for testing of strats pipeline."""

from pathlib import Path
from typing import final

from PIL import Image
import pytest
import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor

from boxsup_pytorch.data.datacontainer import MaskDataContainer
from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.pipeline.strats import BaseStrat
from boxsup_pytorch.utils.losses import Losses


def helper_load_cands() -> torch.Tensor:
    """Load candidates."""
    masks = []
    for i in range(10):
        path = Path(f"tests/data/cand{i+1:02d}.png")
        mask = pil_to_tensor(Image.open(path))
        masks.append(mask / mask.max())
    return torch.cat(masks).unsqueeze(0)


def helper_calc_ious() -> torch.Tensor:
    """Calculate IoUs. Expects that iou calculations is tested."""
    losses = Losses()
    ious = []
    for i in range(2):
        per_bbox_iou = []
        bpath = Path(f"tests/data/mask{i+1:02d}.png")
        bbox = pil_to_tensor(Image.open(bpath))
        bbox = bbox / bbox.max()
        for j in range(10):
            cpath = Path(f"tests/data/cand{j+1:02d}.png")
            cand = pil_to_tensor(Image.open(cpath))
            cand = cand / cand.max()
            iou = losses.inter_over_union(cand.squeeze(), bbox.squeeze())
            per_bbox_iou.append(iou)
        ious.append(torch.stack(per_bbox_iou))
    return torch.stack(ious).unsqueeze(0)


NETWORK = FCN8s(nclass=2, device="cpu").to("cpu")

data1 = MaskDataContainer(
    images=torchvision.io.read_image("tests/data/image.png").unsqueeze(0),  # BxCxWxH
    masks=helper_load_cands(),  # BxNxWxH
    masks_cls=torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 2, 2]]),  # BxN
    masks_iou=helper_calc_ious(),  # BxMxN
    bbox_cls=torch.tensor([[2, 1]]),  # BxM
)

pass


class DemoStrat(BaseStrat):
    """DEMO STRAT IMPLEMENTATION."""

    name: str = "demo"

    @final
    def update(self) -> None:
        """Only Demo for testing purpose of class methods."""
        pass


class TestBaseStrat:
    """Test batch_overlapping loss method of Losses class.

    We implement 1 test to test the calculation of overlapping loss based on same data as the test
    for compare class.

    Test1:
    This is the Test which analyzes the result of running the compare against not initialized cands.
    This is something what actual needed to be avoided and might be replaced with raising an Error.

    Test2:
    This tests the fresh scenario. We have prepared 4 Batches of Data with a Size of 2 x 3.
    So speaking of 2 possible BBoxes and 3 candidate Masks each initilaized the same class.
    The 4 Batches represent themself 4 different scenarios.
        Scenario 0:
        The Cands are the same class as the bboxes.
        Scenario 1:
    """

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(
                (data1),
            )
        ],
        ids=["Check Input Correct"],
    )
    def test_check_input(self, data):
        """Parametrized test for Class TestLossesOverlappingLoss."""
        strat = DemoStrat(NETWORK)

        strat._check_input(data)
