"""Test Cases for utils common function."""

from numpy import iterable
import pytest
import torch

from boxsup_pytorch.utils.common import squash_mask_layer, squash_mask_layer2

# DATA SINGLE MASK TEST
base_single = torch.zeros(1, 3, 3)

data_single_mask = base_single.clone()
data_single_mask[:, 1, 1] = 1

label_single_mask = base_single.clone()
label_single_mask[:, :, :] = 2

expected_single_mask = data_single_mask * 2

# DATA TWO MASK TEST
base_two = torch.zeros(2, 3, 3)

data_two_mask = base_two.clone()
data_two_mask[0, 1:, 1:] = 1
data_two_mask[1, 1:, :] = 1

label_two_mask = base_two.clone()
label_two_mask[0, :, :] = 2
label_two_mask[1, :, :] = 3

expected_two_mask = base_single.clone()
expected_two_mask[:, 1:, 1:] = 2
expected_two_mask[:, 1:, 0] = 3

# DATA MORE MASK TEST
base_more = torch.zeros(4, 3, 3)

data_more_mask = base_more.clone()
data_more_mask[0, 1, 1] = 1
data_more_mask[1, :, 2] = 1
data_more_mask[2, :, 1:] = 1
data_more_mask[3, :, :] = 1

label_more_mask = base_more.clone()
label_more_mask[0, :, :] = 2
label_more_mask[1, :, :] = 3
label_more_mask[2, :, :] = 4
label_more_mask[3, :, :] = 5

expected_more_mask = base_single.clone()
expected_more_mask[:, 1, 1] = 2
expected_more_mask[:, :, 2] = 3
expected_more_mask[:, (0, 2), 1] = 4
expected_more_mask[:, :, 0] = 5

# DATA BENCHMARK TEST

data_bench = torch.randint(high=1, size=(100, 224, 224))
label_bench = torch.linspace(2, 102, 100)[:, None, None].expand(-1, 224, 224)


class TestSquashMaskLayer:
    """Test the squash_mask_layer function.

    The function uses multiple masks which are stacked in a tensor.
    We test the following cases:

    - single mask:
        - only one mask is provided
        - expected the same mask as output
    - two masks:
        - first mask is overlapping second mask
        - exepected complete first mask, second only none overlapping.
    - more masks:
        - same as two but with four masks now

    """

    @pytest.mark.parametrize(
        "data, label, expected",
        [
            pytest.param(
                (data_single_mask),
                (label_single_mask),
                (expected_single_mask),
            ),
            pytest.param(
                (data_two_mask),
                (label_two_mask),
                (expected_two_mask),
            ),
            pytest.param(
                (data_more_mask),
                (label_more_mask),
                (expected_more_mask),
            ),
        ],
        ids=["single_mask", "two_masks", "more_masks"],
    )
    def test_parameterized2(self, data, label, expected):
        """Run all squash mask layer tasks parameterized."""
        result = squash_mask_layer2(data, label)
        assert torch.eq(result, expected).all()

    @pytest.mark.parametrize(
        "data, label, expected",
        [
            pytest.param(
                (data_single_mask),
                (label_single_mask),
                (expected_single_mask),
            ),
            pytest.param(
                (data_two_mask),
                (label_two_mask),
                (expected_two_mask),
            ),
            pytest.param(
                (data_more_mask),
                (label_more_mask),
                (expected_more_mask),
            ),
        ],
        ids=["single_mask", "two_masks", "more_masks"],
    )
    def test_parameterized(self, data, label, expected):
        """Run all squash mask layer tasks parameterized."""
        result = squash_mask_layer(data, label)
        assert torch.eq(result, expected).all()

    def test_wrong_data_shape(self):
        """Run test with wrong data shape."""
        data = torch.zeros(3, 3)
        label = torch.zeros(3, 3)
        with pytest.raises(RuntimeError):
            squash_mask_layer(data, label)

    @pytest.mark.parametrize(
        "function, data, label",
        [
            pytest.param(
                (squash_mask_layer),
                (data_bench),
                (label_bench),
            ),
            pytest.param(
                (squash_mask_layer2),
                (data_bench),
                (label_bench),
            ),
        ],
        ids=["benchmark1", "benchmark2"],
    )
    def test_bench_squash_version(self, benchmark, function, data, label):
        """Run both versions as benchmark and compare."""
        result = benchmark.pedantic(function, args=(data, label), iterations=10, rounds=30)
        print("<<--BENCH RESULT-->>", result.max())
        assert result.max() == 0
