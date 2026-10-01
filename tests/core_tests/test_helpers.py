"""Tests for biom3.core.helpers."""

import pytest

from biom3.core.helpers import coerce_limit_batches, optional_positive_int


@pytest.mark.parametrize("value, expected, expected_type", [
    # Fraction path: floats in (0, 1] stay float
    (0.05, 0.05, float),
    (0.5, 0.5, float),
    (1.0, 1.0, float),
    (1, 1.0, float),
    # Absolute-count path: values > 1 become int
    (2, 2, int),
    (200, 200, int),
    (200.0, 200, int),
    (5.7, 5, int),  # floor via int(); user shouldn't pass non-integer >1 anyway
    # None passes through
    (None, None, type(None)),
])
def test_coerce_limit_batches(value, expected, expected_type):
    """Cover both PL Trainer-acceptable shapes for --limit_*_batches."""
    result = coerce_limit_batches(value)
    assert result == expected
    assert type(result) is expected_type


@pytest.mark.parametrize("value, expected", [
    (None, None),
    ("None", None),
    ("none", None),
    ("", None),
    ("  ", None),
    (3, 3),
    ("3", 3),
    (" 12 ", 12),
    (5.0, 5),
])
def test_optional_positive_int(value, expected):
    result = optional_positive_int(value)
    assert result == expected
    assert result is None or type(result) is int


@pytest.mark.parametrize("value", [0, -1, "0", "-2", "abc", "2.5", 2.5, True, False, [3]])
def test_optional_positive_int_rejects(value):
    with pytest.raises(ValueError, match="num_replicas must be a positive integer"):
        optional_positive_int(value, name="num_replicas")
