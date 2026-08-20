import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chords


def note(pitch, start, end, velocity=1.0):
    return {"pitch": pitch, "start": start, "end": end, "velocity": velocity}


def span(chord, start, end):
    return {"chord": chord, "start": start, "end": end, "confidence": 2.0}


class ChordAnalysisTests(unittest.TestCase):
    def test_major_template_and_bass_root_weighting(self):
        histogram = [0.0] * 12
        histogram[0], histogram[4], histogram[7] = 3, 2, 2
        self.assertEqual(chords._match_chord(histogram, 0)[0], "C")

    def test_confidence_below_threshold_is_unknown(self):
        histogram = [1.0] * 12
        self.assertEqual(chords._match_chord(histogram, None)[0], "?")

    def test_loop_coverage_counts_only_matched_positions(self):
        spans = [span(c, i, i + 1) for i, c in enumerate(("C", "G", "C", "G", "F", "C"))]
        loop = chords._loop(spans, 0, "major")
        self.assertEqual(loop["chords"], ["C", "G"])
        self.assertEqual(loop["coverage"], 0.67)

    def test_unknown_breaks_loop_and_stays_in_denominator(self):
        spans = [span(c, i, i + 1) for i, c in enumerate(("C", "G", "?", "C", "G"))]
        loop = chords._loop(spans, 0, "major")
        self.assertEqual(loop["count"], 2)
        self.assertEqual(loop["coverage"], 0.8)

    def test_rotation_equivalent_pattern_is_named(self):
        sequence = ("G", "Am", "F", "C") * 2
        loop = chords._loop([span(c, i, i + 1) for i, c in enumerate(sequence)], 0, "major")
        self.assertEqual(loop["name"], "1564")

    def test_null_tracks_and_empty_notes_degrade(self):
        result = chords.analyze({"vocals": None, "bass": [], "piano": None},
                                bpm=120, duration=4)
        self.assertEqual(result, {"key": None, "keyConfidence": 0.0,
                                  "spans": [], "loop": None})

    def test_malformed_note_can_be_dropped_before_analysis(self):
        raw = [None, "bad", {"pitch": 60, "start": 0, "end": 2},
               {"pitch": 61, "start": 0, "end": None}]
        cleaned = [item for item in raw if isinstance(item, dict) and
                   all(isinstance(item.get(field), (int, float))
                       for field in ("pitch", "start", "end"))]
        result = chords.analyze({"piano": cleaned}, bpm=120, duration=2)
        self.assertIsNotNone(result["key"])

    def test_abnormal_bpm_falls_back_to_two_second_window(self):
        tracks = {"piano": [note(60, 0, 4), note(64, 0, 4), note(67, 0, 4)]}
        for bpm in (-120, 0, math.nan, math.inf, "bad", None):
            with self.subTest(bpm=bpm):
                result = chords.analyze(tracks, bpm=bpm, duration=4)
                self.assertEqual([(s["start"], s["end"]) for s in result["spans"]],
                                 [(0.0, 4.0)])


if __name__ == "__main__":
    unittest.main()

