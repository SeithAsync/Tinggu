"""Basic Pitch 音符的三道泛音闸。

参数按 Ocean Listen（ennisaaaaaaaa-stack，MIT）校准值在本项目重写。
"""

PITCH_RANGES = {
    "vocals": (48, 84),
    "bass": (28, 52),
    "guitar": (38, 76),
    "piano": (33, 96),
    "other": (36, 84),
}
MIN_DURATION = 0.05
OVERLAP_RATIO = 0.30
OVERLAP_WINDOW = 2.0


def _overlap(left, right):
    return max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))


def filter_notes(track, notes):
    """返回该轨过闸后的音符和各闸删除计数。"""
    low, high = PITCH_RANGES[track]
    stats = {"input": len(notes), "range": 0, "duration": 0, "overlap": 0, "output": 0}
    ranged = []
    for note in notes:
        if not low <= note["pitch"] <= high:
            stats["range"] += 1
        elif note["duration"] < MIN_DURATION:
            stats["duration"] += 1
        else:
            ranged.append(note)

    kept = []
    for note in sorted(ranged, key=lambda item: (item["start"], item["end"], -item["velocity"])):
        competitors = []
        for index in range(len(kept) - 1, -1, -1):
            previous = kept[index]
            if note["start"] - previous["start"] > OVERLAP_WINDOW:
                break
            overlap = _overlap(previous, note)
            shorter = min(previous["duration"], note["duration"])
            if shorter and overlap > shorter * OVERLAP_RATIO:
                competitors.append(index)
        if not competitors:
            kept.append(note)
            continue
        strongest = max([note] + [kept[index] for index in competitors], key=lambda item: item["velocity"])
        stats["overlap"] += len(competitors) if strongest is note else 1
        if strongest is note:
            for index in competitors:
                del kept[index]
            kept.append(note)

    kept.sort(key=lambda item: (item["start"], item["pitch"]))
    stats["output"] = len(kept)
    return kept, stats
