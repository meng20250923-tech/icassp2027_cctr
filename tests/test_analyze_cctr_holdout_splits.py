import pytest

from scripts.analysis.analyze_cctr_holdout_splits import spearman_correlation


def test_spearman_correlation_handles_monotonic_and_reverse_sequences():
    assert spearman_correlation([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == pytest.approx(1.0)
    assert spearman_correlation([0.0, 1.0, 2.0], [5.0, 4.0, 3.0]) == pytest.approx(-1.0)
