import pytest

from recast.stats import summarize, wilson_interval


@pytest.mark.parametrize("k,n,lo,hi", [(18, 25, 52.4, 85.7), (0, 25, 0.0, 13.3), (2, 25, 2.2, 25.0),
                                       (43, 75, 46.1, 67.9)])
def test_wilson_interval(k, n, lo, hi):
    a, b = wilson_interval(k, n)
    assert round(100 * a, 1) == lo and round(100 * b, 1) == hi


def test_summarize():
    s = summarize([True, False, True, True])
    assert s["successes"] == 3 and s["rate"] == 75.0 and s["ci95"][0] < 75 < s["ci95"][1]
