"""把和弦段翻译为功能和声运动。纯标准库实现。"""

import re


PITCHES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
           "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
FUNCTIONS = {
    "major": {0: "T", 2: "S", 4: "T", 5: "S", 7: "D", 9: "T", 11: "D"},
    "minor": {0: "T", 2: "S", 3: "T", 5: "S", 7: "D", 8: "S", 10: "D"},
}


def _function(chord, tonic, mode):
    match = re.match(r"^([A-G](?:#)?)", chord or "")
    if not match:
        return "X", False
    offset = (PITCHES[match.group(1)] - tonic) % 12
    function = FUNCTIONS[mode].get(offset, "X")
    return function, function == "D" and "7" in chord


def _state(previous, current):
    if "?" in (previous, current):
        return "?"
    if "X" in (previous, current):
        return "调外"
    if previous == "D" and current == "T":
        return "解决"
    if previous == "D" and current == "D":
        return "延宕"
    if current == "D" and previous in ("T", "S"):
        return "张力上升"
    return "平稳"


def _merge(states):
    merged = []
    for item in states:
        if merged and merged[-1]["state"] == item["state"] and merged[-1]["end"] == item["start"]:
            merged[-1]["end"] = item["end"]
            merged[-1]["conf"] = round(min(merged[-1]["conf"], item["conf"]), 2)
        else:
            merged.append(dict(item))
    return merged


def analyze(chord_analysis):
    """按 spec 固定映射与状态规则生成 motion；证据不足时留白。"""
    if not chord_analysis or not chord_analysis.get("key"):
        return None
    try:
        tonic_name, mode = chord_analysis["key"].split()
        tonic = PITCHES[tonic_name]
        FUNCTIONS[mode]
    except (ValueError, KeyError):
        return None
    spans = chord_analysis.get("spans") or []
    readable_duration = functional_duration = 0.0
    annotated = []
    for span in spans:
        chord = span.get("chord") or "?"
        duration = max(0.0, float(span.get("end", 0)) - float(span.get("start", 0)))
        if chord == "?":
            annotated.append((span, "?", False))
            continue
        function, dominant7 = _function(chord, tonic, mode)
        readable_duration += duration
        if function in {"T", "S", "D"}:
            functional_duration += duration
        annotated.append((span, function, dominant7))
    share = functional_duration / readable_duration if readable_duration else 0.0
    loop = chord_analysis.get("loop") or {}
    base = {"motionVersion": 1, "functionalShare": round(share, 2), "states": [],
            "loopNote": None}
    if float(chord_analysis.get("keyConfidence") or 0) < 0.30 or share < 0.60:
        if float(loop.get("coverage") or 0) >= 0.50:
            resolutions = []
            previous = None
            for span, function, dominant7 in annotated:
                if previous and previous[1] == "D" and function == "T":
                    resolutions.append(round(float(span.get("start", 0)), 2))
                previous = (span, function, dominant7) if function != "?" else None
            base.update({"mode": "loop", "loopNote": {
                "chords": loop.get("chords") or [], "name": loop.get("name"),
                "count": loop.get("count"), "coverage": loop.get("coverage"),
                "resolutions": resolutions}})
        else:
            base["mode"] = "nonfunctional"
        return base

    states = []
    previous_function = None
    previous_dominant7 = False
    previous_confidence = None
    for span, function, dominant7 in annotated:
        start, end = float(span.get("start", 0)), float(span.get("end", 0))
        confidence = round(float(span.get("conf", span.get("confidence", 0))), 2)
        if function == "?":
            state = "?"
            previous_function = None
            previous_dominant7 = False
            previous_confidence = None
        elif previous_function is None:
            state = "延宕" if function == "D" and end - start > 4 else (
                "张力上升" if function == "D" else ("调外" if function == "X" else "平稳"))
            previous_function, previous_dominant7 = function, dominant7
        else:
            confidence = round(min(previous_confidence, confidence), 2)
            state = _state(previous_function, function)
            if state == "解决" and previous_dominant7:
                state = "强解决"
            elif function == "D" and end - start > 4:
                state = "延宕"
            previous_function, previous_dominant7 = function, dominant7
        if function != "?":
            previous_confidence = round(float(span.get("conf", span.get("confidence", 0))), 2)
        states.append({"start": round(start, 2), "end": round(end, 2),
                       "state": state, "conf": confidence})
    base.update({"mode": "functional", "states": _merge(states)})
    return base
