import numpy as np

import chroma_chords
import chords


def span(name, start, end, conf=1.3):
    return {"chord": name, "start": start, "end": end, "conf": conf}


def test_chroma_vector_recognizes_major_triad():
    vector = np.zeros(12)
    vector[[0, 4, 7]] = [1.0, 0.8, 0.6]
    name, confidence = chroma_chords._match_chord(vector, bass_root=0)
    assert name == "C"
    assert confidence >= chords.CONFIDENCE_THRESHOLD


def test_merge_dual_five_rules():
    notes = [span("C", 0, 2, 1.4), span("D", 2, 4), span("?", 4, 6, 0),
             span("Cmaj7", 6, 8, 1.3), span("F", 8, 10), span("?", 10, 12, 0)]
    chroma = [span("C", 0, 2, 1.2), span("?", 2, 4, 0), span("Em", 4, 6, 1.2),
              span("C", 6, 8, 1.2), span("G", 8, 10), span("?", 10, 12, 0)]
    merged, sources = chords.merge_dual(notes, chroma, return_sources=True)
    assert [item["chord"] for item in merged] == ["C", "D", "Em", "C", "?" ]
    assert sources == ["both", "notes", "chroma", "root-agree", None, None]
    assert merged[0]["confidence"] == 1.3
    assert merged[3]["confidence"] == 1.2


def test_window_grid_matches_notes_analyze():
    tracks = {"piano": [{"pitch": 60, "start": 0, "end": 5, "velocity": 1},
                         {"pitch": 64, "start": 0, "end": 5, "velocity": 1}]}
    analysis = chords.analyze(tracks, bpm=120, duration=5)
    notes_grid = chords.expand_to_grid(analysis["spans"], bpm=120, duration=5)
    chroma_grid = list(chroma_chords._grid(bpm=120, duration=5))
    assert len(notes_grid) == len(chroma_grid) == 5
    assert [(item["start"], item["end"]) for item in notes_grid] == chroma_grid


def test_chroma_only_degrades_cleanly():
    chroma = [span("Am", 0, 2, 1.25), span("F", 2, 4, 1.2)]
    merged, sources = chords.merge_dual([], chroma, return_sources=True)
    assert [item["chord"] for item in merged] == ["Am", "F"]
    assert sources == ["chroma", "chroma"]
