import numpy as np

from ears import tension_curve


def test_linear_rise_produces_positive_crescendo_arc():
    duration = 30
    frames = round(duration * 22050 / 512)
    rms = np.linspace(0.1, 1.0, frames)
    data = {"duration": duration, "stemTimeline": {track: [[0, duration]] for track in
            ("vocals", "drums", "bass", "guitar", "piano", "other")}}
    result = tension_curve(data, None, {track: rms for track in data["stemTimeline"]}, np)
    assert len(result["arcs"]) == 1
    assert result["arcs"][0]["type"] == "渐强"
    assert result["arcs"][0]["deltaPct"] > 0
