import numpy as np

from analyze_song import analyze_salience


def test_step_drop_peak_and_negative_evidence():
    base = np.ones(80)
    dropped = base.copy()
    dropped[40:] = 0.2
    result = analyze_salience([dropped] * 8, [], 80, 1)
    peak = min(result["peaks"], key=lambda item: abs(item["t"] - 40))
    assert abs(peak["t"] - 40) <= 1
    assert "-" in peak["evidence"][0]


def test_uniform_curves_have_no_peaks():
    result = analyze_salience([np.ones(60)] * 8, [], 60, 1)
    assert result["peaks"] == []
