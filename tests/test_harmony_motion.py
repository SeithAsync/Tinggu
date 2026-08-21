import harmony_motion


def chord_data(**overrides):
    data = {
        "key": "C major", "keyConfidence": 0.9,
        "spans": [
            {"chord": "C", "start": 0, "end": 4, "conf": 0.91},
            {"chord": "G7", "start": 4, "end": 8, "conf": 0.82},
            {"chord": "C", "start": 8, "end": 12, "conf": 0.88},
        ],
        "loop": None,
    }
    data.update(overrides)
    return data


def test_functional_c_g7_c_states():
    result = harmony_motion.analyze(chord_data())
    assert result["mode"] == "functional"
    assert [item["state"] for item in result["states"]] == [
        "平稳", "张力上升", "强解决"]
    assert result["states"][-1]["conf"] == 0.82


def test_fit_gate_loop_and_missing_key():
    loop = {"chords": ["C", "G"], "count": 3, "coverage": 0.6, "name": None}
    assert harmony_motion.analyze(chord_data(keyConfidence=0.2, loop=loop))["mode"] == "loop"
    assert harmony_motion.analyze(chord_data(key=None)) is None


def test_question_break_and_merged_confidence_minimum():
    spans = [
        {"chord": "C", "start": 0, "end": 2, "conf": 0.9},
        {"chord": "C", "start": 2, "end": 4, "conf": 0.7},
        {"chord": "?", "start": 4, "end": 6, "conf": 0.1},
        {"chord": "F", "start": 6, "end": 8, "conf": 0.8},
    ]
    states = harmony_motion.analyze(chord_data(spans=spans))["states"]
    assert states[0] == {"start": 0.0, "end": 4.0, "state": "平稳", "conf": 0.7}
    assert states[1]["state"] == "?"
    assert states[2]["state"] == "平稳"
