"""Test Scenario for utils losses."""

import torch

from boxsup_pytorch.utils.losses import Losses


class TestLossesClassCompareLabels:
    """Test different methods of Losses class."""

    def test_compare_labels_zero_elements(self):
        """Test against cands all zeros."""
        losses = Losses()
        torch.manual_seed(42)
        cands_classes = torch.zeros(4, 2, 3)
        bbox_classes = torch.randint(10, size=(4, 2))

        result = losses._compare_labels(cands_classes, bbox_classes)
        assert result.shape == (4, 2, 3)  # Check for expected shape
        assert torch.all(result == 1)  # Ensure all elements are 1

    def test_compare_labels_non_zero_elements(self):
        """Test against both random."""
        losses = Losses()
        torch.manual_seed(42)
        cands_classes = torch.randint(3, size=(4, 2, 3), dtype=torch.float32)
        bbox_classes = torch.randint(3, size=(4, 2), dtype=torch.float32)

        result = losses._compare_labels(cands_classes, bbox_classes)
        assert result.shape == (4, 2, 3)  # Check for expected shape
        assert result.dtype == torch.bool  # Ensure the result is a boolean tensor

    def test_compare_labels_realistic_data(self):
        """Test against both random."""
        losses = Losses()
        torch.manual_seed(42)
        cands_classes = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            ]
        )
        bbox_classes = torch.tensor([[1.0, 2.0], [0.0, 0.0], [2.0, 1.0], [1.0, 1.0]])

        result = losses._compare_labels(cands_classes, bbox_classes)
        assert result.shape == (4, 2, 3)  # Check for expected shape
        assert result.dtype == torch.bool  # Ensure the result is a boolean tensor
        assert torch.all(
            result
            == torch.tensor(
                [
                    [[False, False, False], [True, True, True]],
                    [[False, False, False], [False, False, False]],
                    [[False, False, False], [True, True, True]],
                    [[False, False, False], [False, False, False]],
                ]
            )
        )
