import builtins

import numpy as np
import soundfile as sf

import line_table


def _stems(tmp_path, sid, audio):
    path = tmp_path / "data" / "stems" / sid
    path.mkdir(parents=True)
    sf.write(path / "vocals.mp3", audio, line_table.SR, format="WAV")
    return path


def test_line_windows_and_synthetic_audio(tmp_path, monkeypatch):
    duration = 4
    samples = np.arange(duration * line_table.SR) / line_table.SR
    audio = 0.2 * np.sin(2 * np.pi * 220 * samples)
    stems = _stems(tmp_path, "song", audio)
    lines = [(0.0, "甲"), (1.0, "乙"), (3.0, "丙")]
    data = {"stemTimeline": {"vocals": [[0, 4]], "drums": [[0, 2]]}}
    monkeypatch.setattr(line_table, "_praat_metrics", lambda segment, available: (1.0, 2.0, 10.0))

    assert [(start, end) for start, end, _ in line_table.line_windows(lines, [[0, 4]])] == [
        (0.0, 1.0), (1.0, 3.0), (3.0, 4.0)]
    result = line_table.build(data, lines, stems, tmp_path / "lines.json")

    assert [row["t"] for row in result["rows"]] == [0.0, 1.0, 3.0]
    assert result["rows"][0]["accompStems"] == 1
    assert (tmp_path / "lines.json").exists()


def test_unvoiced_line_is_skipped(tmp_path, monkeypatch):
    audio = np.zeros(2 * line_table.SR)
    samples = np.arange(line_table.SR) / line_table.SR
    audio[line_table.SR:] = 0.2 * np.sin(2 * np.pi * 220 * samples)
    stems = _stems(tmp_path, "quiet", audio)
    monkeypatch.setattr(line_table, "_praat_metrics", lambda segment, available: (None, None, None))

    result = line_table.build({"stemTimeline": {"vocals": [[0, 2]]}},
                              [(0.0, "甲"), (1.0, "乙")], stems, tmp_path / "lines.json")

    assert [row["text"] for row in result["rows"]] == ["乙"]
    assert result["skipped"] == 1


def test_missing_parselmouth_degrades_to_null(tmp_path, monkeypatch, capsys):
    samples = np.arange(2 * line_table.SR) / line_table.SR
    stems = _stems(tmp_path, "nopraat", 0.2 * np.sin(2 * np.pi * 220 * samples))
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "parselmouth":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    result = line_table.build({"stemTimeline": {"vocals": [[0, 2]]}},
                              [(0.0, "甲")], stems, tmp_path / "lines.json")

    row = result["rows"][0]
    assert (row["jitterPct"], row["shimmerPct"], row["hnrDb"]) == (None, None, None)
    assert capsys.readouterr().err.strip() == "未装 parselmouth，声线三项跳过"


def test_texture_medians_and_extreme_lines():
    rows = [
        {"t": 1.0, "text": "甲", "centroidHz": 1000, "tiltDb": -9,
         "airRatio": 0.1, "jitterPct": 1.0, "shimmerPct": 3.0, "hnrDb": 12,
         "pitchStabSt": 0.5},
        {"t": 2.0, "text": "乙", "centroidHz": 1400, "tiltDb": -7,
         "airRatio": 0.2, "jitterPct": 2.0, "shimmerPct": 5.0, "hnrDb": 16,
         "pitchStabSt": 0.2},
        {"t": 3.0, "text": "丙", "centroidHz": None, "tiltDb": None,
         "airRatio": None, "jitterPct": None, "shimmerPct": None, "hnrDb": None,
         "pitchStabSt": None},
    ]

    result = line_table.texture(rows)

    assert result["centroidHz"] == 1200
    assert result["airRatio"] == 0.15
    assert result["roughestLine"] == {"t": 2.0, "text": "乙", "value": 2.0}
    assert result["steadiestLine"]["t"] == 2.0
    assert result["textureVersion"] == 3
