"""Test Scenario for utils losses."""

import pytest
import torch

from boxsup_pytorch.utils.losses import Losses

torch.manual_seed(42)

cands_classes_fresh = torch.tensor(
    [
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ]
)

cands_classes_trained = torch.tensor(
    [
        [[1.0, 2.0, 2.0], [2.0, 1.0, 1.0]],
        [[2.0, 1.0, 2.0], [2.0, 2.0, 2.0]],
        [[2.0, 1.0, 1.0], [1.0, 2.0, 2.0]],
        [[2.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
    ]
)

bbox_classes_real = torch.tensor(
    [
        [1.0, 2.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [2.0, 1.0],
    ]
)

expected_all_true = torch.tensor([[True, True, True], [True, True, True]])
expected_first_true = torch.tensor([[True, True, True], [False, False, False]])
expected_second_true = torch.tensor([[False, False, False], [True, True, True]])
expected_mixed_scn0 = torch.tensor([[True, False, False], [True, False, False]])
expected_mixed_scn1 = torch.tensor([[False, True, False], [False, False, False]])
expected_mixed_scn2 = torch.tensor([[False, True, True], [False, True, True]])
expected_mixed_scn3 = torch.tensor([[True, True, False], [True, True, False]])


class TestLossesClassCompareLabels:
    """Test compare label method of Losses class.

    We implement 3 tests to check both trees of the if statement of the method and two states of
    the process (Fresh Initialized and Middle of Process):

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
        The Cands are all labeled class 1 while only one bbox exists.
        Scenario 2:
        The Cands classes are all class 2 but bbox only one bbox is class 2.
        Scenario 3:
        Same as Scenario 2 but this time with class 1.

    Test3:
    This tests an already trained state. Also 4 Batches Scenario.
        Scenario 0:
        Only the first candidate has the class of the BBox.
        Scenario 1:
        The second candidate is correct for first BBox, Since BBox 2 does not existst, the result
        is always False.
        Scenario 2:
        The second and thirds candidate has same class as BBoxes.
        Scenario 3:
        The first and second candidate has same class as BBoxes.
    """

    @pytest.mark.parametrize(
        [
            "cands_classes",
            "bbox_classes",
            "exp_shape",
            "exp_dtype",
            "exp_scn0",
            "exp_scn1",
            "exp_scn2",
            "exp_scn3",
        ],
        (
            pytest.param(
                (torch.zeros(4, 2, 3)),
                (bbox_classes_real),
                ((4, 2, 3)),
                (torch.bool),
                expected_all_true,
                expected_all_true,
                expected_all_true,
                expected_all_true,
            ),
            pytest.param(
                (cands_classes_fresh),
                (bbox_classes_real),
                ((4, 2, 3)),
                (torch.bool),
                expected_all_true,
                expected_first_true,
                expected_second_true,
                expected_second_true,
            ),
            pytest.param(
                (cands_classes_trained),
                (bbox_classes_real),
                ((4, 2, 3)),
                (torch.bool),
                expected_mixed_scn0,
                expected_mixed_scn1,
                expected_mixed_scn2,
                expected_mixed_scn3,
            ),
        ),
        ids=["not_initialized_cands", "initialized_cands", "trained_cands"],
    )
    def test_compare_labels_zero_elements(
        self,
        cands_classes,
        bbox_classes,
        exp_shape,
        exp_dtype,
        exp_scn0,
        exp_scn1,
        exp_scn2,
        exp_scn3,
    ):
        """Test against cands all zeros.

        cands_classes
        """
        losses = Losses()

        result = losses._compare_labels(cands_classes, bbox_classes)
        assert result.shape == exp_shape
        assert result.dtype == exp_dtype
        # Scenarios (Batches)
        assert torch.all(result[0] == exp_scn0)
        assert torch.all(result[1] == exp_scn1)
        assert torch.all(result[2] == exp_scn2)
        assert torch.all(result[3] == exp_scn3)
