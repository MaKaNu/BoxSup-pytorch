r"""Test Module for losses.

Each test is documented with a few explanations about the test.
This is a class based test scenario with Global test fixtures.

TestClasses:
    - TestCompareLabel
    - TestInterOUnion
    - TestOverlapping
"""

import numpy as np
import pytest

from boxsup_pytorch.losses import compare_labels, inter_o_union, overlapping_loss

# Fixture Setup


@pytest.fixture()
def bounding_box() -> np.array:
    """Pytest fixture of 2x2 BoundingBox."""
    return np.array([[0, 1], [0, 0]])


@pytest.fixture()
def bounding_box3() -> np.array:
    """Pytest fixture of 3x3 BoundingBox."""
    return np.array([[1, 1, 1], [1, 0, 2], [2, 2, 2]])


@pytest.fixture()
def multi_cand() -> np.array:
    """Pytest fixture of 3x3 multi class."""
    return np.array([[1, 1, 1], [2, 0, 1], [2, 2, 2]])


@pytest.fixture()
def overlap_cand() -> np.array:
    """Pytest fixture of 2x2 overlapping candidate."""
    return np.array([[0, 1], [0, 1]])


@pytest.fixture()
def not_overlap_cand() -> np.array:
    """Pytest fixture of 2x2 not overlapping candidate."""
    return np.array([[0, 0], [1, 0]])


@pytest.fixture()
def full_overlap_cand() -> np.array:
    """Pytest fixture of 2x2 full overlapping candidate."""
    return np.array([[0, 1], [0, 0]])


@pytest.fixture()
def not_overlap_cands(not_overlap_cand: np.array) -> np.array:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return np.array((not_overlap_cand, not_overlap_cand, not_overlap_cand))


@pytest.fixture()
def mixed_overlap_cands(
    not_overlap_cand: np.array, overlap_cand: np.array, full_overlap_cand: np.array
) -> np.array:
    """Pytest fixture of 3 2x2 mixed overlapping candidates."""
    return np.array((not_overlap_cand, overlap_cand, full_overlap_cand))


@pytest.fixture()
def full_overlap_cands(full_overlap_cand: np.array) -> np.array:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return np.array((full_overlap_cand, full_overlap_cand, full_overlap_cand))


@pytest.fixture()
def diff_label_cands(full_overlap_cand: np.array) -> np.array:
    """Pytest fixture of 3 2x2 not overlapping candidates."""
    return np.array((full_overlap_cand * 2, full_overlap_cand * 2, full_overlap_cand * 2))


# Compare Labels Tests

class TestCompareLabel:
    r"""Test scenario for compare labels.

    The formula:

    $\delta (l_B, l_S) =
    \begin{cases}
    1 & \text{if } l_B = l_S \\
    0 & \text{otherwise.}
    \end{cases}$

    Test using following inputs:

    Grid size is always 2x2
    """

    def test_compare_labels_1(self, bounding_box: np.array, overlap_cand: np.array):
        """Test 1: same class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1

        expected result = True
        """
        assert compare_labels(bounding_box, overlap_cand)

    def test_compare_labels_2(self, bounding_box: np.array, overlap_cand: np.array):
        """Test 2: different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 2

        expected result = False
        """
        assert not compare_labels(bounding_box, overlap_cand * 2)

    def test_compare_labels_3(self, bounding_box: np.array, overlap_cand: np.array):
        """Test 3: multi candidates with different class labels.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1
        - candidates: count 1, label 2

        expected result = Array[True, False]
        """
        result = compare_labels(bounding_box, np.array((overlap_cand, overlap_cand * 2)))
        assert (result == np.array((True, False))).all()


# Intersection over Union Tests

class TestInterOUnion:
    r"""Test scenario for intersection over union.

    The formula:

    $IoU (B, S) =
    {sum_of_intersecting_pixel\over sum_of_union_pixel}
    $

    $sum_of_intersecting_pixel =
    sum(B(l_B)\wedge S(l_S))
    $

    $sum_of_union_pixel =
    sum(B(l_B)\vee S(l_S))
    $

    Tests using following inputs:

    Grid size is 2x2 and 3x3
    """

    def test_inter_o_union_1(self, bounding_box: np.array, overlap_cand: np.array):
        """Test 1: even overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 2x2, overlapping

        expected result = 0.5
        """
        assert inter_o_union(bounding_box, overlap_cand) == 0.5

    def test_inter_o_union_2(self, bounding_box: np.array, full_overlap_cands: np.array):
        """Test 2: uneven overlapping.

        - bounding_box: count 1, label 1, grid 2x2
        - candidates: count 1, label 1, grid 3x2x2, full overlapping

        expected result = Array[1,1,1]
        """
        result = inter_o_union(bounding_box, full_overlap_cands)
        assert (result == np.array((1, 1, 1), dtype=np.float64)).all()

    def test_inter_o_union_3(self, bounding_box3: np.array, multi_cand: np.array):
        """Test 3: multi class.

        - bounding_box: count 1, label 1, grid 3x3
        - candidates: count 1, label 1, grid 3x3, full overlapping

        expected result = 3/5
        """
        assert inter_o_union(bounding_box3, multi_cand) == 3 / 5


# Overlapping Loss Tests

class TestOverlapping:
    r"""Test scenario for overlapping_loss.

    The loss formula:

    $\Epsilon_o = {1\over N} \sum (1 - IoU(B,S)\delta (l_B, l_S)$

    Tests using following inputs:

    Grid size is always 2x2
    """

    def test_overlapping_1(self, bounding_box: np.array, not_overlap_cands: np.array):
        """Test 1: no overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, no overlapping

        expected result = 1
        """
        assert overlapping_loss(bounding_box, not_overlap_cands) == 1

    def test_overlapping_2(self, bounding_box: np.array, mixed_overlap_cands: np.array):
        """Test 2: mixed overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping
        - candidates: count 1, label 1, overlapping
        - candidates: count 1, label 1, full overlapping

        expected result = 1/2
        """
        assert overlapping_loss(bounding_box, mixed_overlap_cands) == 1 / 2

    def test_overlapping_3(self, bounding_box: np.array, full_overlap_cands: np.array):
        """Test 3: full overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 1, full overlapping

        expected result = 0
        """
        assert overlapping_loss(bounding_box, full_overlap_cands) == 0

    def test_overlapping_4(self, bounding_box: np.array, diff_label_cands: np.array):
        """Test 4: diff overlapping.

        - bounding_box: count 1, label 1
        - candidates: count 3, label 2, full overlapping

        expected result = 0
        """
        assert overlapping_loss(bounding_box, diff_label_cands) == 0

    def test_overlapping_5(self, bounding_box: np.array, not_overlap_cand: np.array):
        """Test 5: no overlapping single.

        - bounding_box: count 1, label 1
        - candidates: count 1, label 1, no overlapping

        expected result = 1
        """
        assert overlapping_loss(bounding_box, not_overlap_cand) == 1
