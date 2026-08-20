"""从分轨音符识别调性、和弦与常见进行。纯标准库实现。"""

import math


PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
HARMONY_TRACKS = ("piano", "guitar", "other")
ALL_TRACKS = ("vocals", "bass", "guitar", "piano", "other")
CONFIDENCE_THRESHOLD = 1.12

TEMPLATES = (
    ("", ((0, 1.0), (4, 0.8), (7, 0.6))),
    ("m", ((0, 1.0), (3, 0.8), (7, 0.6))),
    ("7", ((0, 1.0), (4, 0.8), (7, 0.6), (10, 0.5))),
    ("maj7", ((0, 1.0), (4, 0.8), (7, 0.6), (11, 0.5))),
    ("m7", ((0, 1.0), (3, 0.8), (7, 0.6), (10, 0.5))),
    ("dim", ((0, 1.0), (3, 0.8), (6, 0.6))),
    ("sus4", ((0, 1.0), (5, 0.8), (7, 0.6))),
    ("aug", ((0, 1.0), (4, 0.8), (8, 0.6))),
)

PATTERNS = (
    ((3, 4, 2, 5), "王道进行(4536)"),
    ((0, 4, 5, 3), "1564"),
    ((5, 3, 0, 4), "6415"),
    ((0, 5, 3, 4), "1645(五十年代)"),
    ((3, 0, 4, 5), "4156"),
    ((1, 4, 0), "251"),
    ((0, 4, 5, 2, 3, 0, 3, 4), "卡农"),
    ((5, 4, 3, 2), "小调下行"),
)


def _overlap(note, start, end):
    return max(0.0, min(float(note["end"]), end) - max(float(note["start"]), start))


def _histogram(tracks, names, start=0.0, end=math.inf):
    histogram = [0.0] * 12
    for name in names:
        for note in tracks.get(name) or ():
            overlap = _overlap(note, start, end)
            if overlap:
                histogram[int(note["pitch"]) % 12] += overlap * float(note.get("velocity", 1.0))
    return histogram


def _correlation(values, profile):
    mean_values = sum(values) / 12
    mean_profile = sum(profile) / 12
    numerator = sum((x - mean_values) * (y - mean_profile) for x, y in zip(values, profile))
    denominator = math.sqrt(sum((x - mean_values) ** 2 for x in values) *
                            sum((y - mean_profile) ** 2 for y in profile))
    return numerator / denominator if denominator else 0.0


def _key(histogram):
    candidates = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for root in range(12):
            rotated = tuple(profile[(pitch - root) % 12] for pitch in range(12))
            candidates.append((_correlation(histogram, rotated), root, mode))
    best, second = sorted(candidates, reverse=True)[:2]
    confidence = max(0.0, min(1.0, (best[0] - second[0]) / 0.25))
    return f"{PITCH_NAMES[best[1]]} {best[2]}", round(confidence, 2), best[1], best[2]


def _bass_root(tracks, start, end):
    histogram = _histogram(tracks, ("bass",), start, end)
    return max(range(12), key=histogram.__getitem__) if any(histogram) else None


def _match_chord(histogram, bass_root):
    total = sum(histogram)
    active = sum(value > total * 0.05 for value in histogram) if total else 0
    if not total or active < 2:
        return "?", 0.0
    candidates = []
    for root in range(12):
        for suffix, template in TEMPLATES:
            chord_tones = {(root + interval) % 12 for interval, _ in template}
            score = sum(histogram[(root + interval) % 12] * weight for interval, weight in template)
            score -= sum(value * 0.3 for pitch, value in enumerate(histogram) if pitch not in chord_tones)
            score /= sum(weight for _, weight in template)
            if bass_root == root:
                score *= 1.25
            candidates.append((score, root, suffix))
    candidates.sort(reverse=True)
    best, second = candidates[:2]
    confidence = best[0] / max(second[0], total * 0.01) if best[0] > 0 else 0.0
    chord = PITCH_NAMES[best[1]] + best[2]
    if confidence < CONFIDENCE_THRESHOLD:
        chord = "?"
    return chord, round(confidence, 2)


def _merge_windows(windows):
    spans = []
    for chord, start, end, confidence in windows:
        if spans and spans[-1]["chord"] == chord:
            previous_duration = spans[-1]["end"] - spans[-1]["start"]
            duration = end - start
            spans[-1]["confidence"] = round(
                (spans[-1]["confidence"] * previous_duration + confidence * duration) /
                (previous_duration + duration), 2)
            spans[-1]["end"] = round(end, 2)
        else:
            spans.append({"chord": chord, "start": round(start, 2), "end": round(end, 2),
                          "confidence": confidence})
    return spans


def _rotations(sequence):
    return {tuple(sequence[index:] + sequence[:index]) for index in range(len(sequence))}


def _loop(spans, key_root, mode):
    runs = []
    run_indices = []
    for span_index, span in enumerate(spans):
        chord = span["chord"]
        if chord == "?":
            if runs and not runs[-1]:
                continue
            runs.append([])
            run_indices.append([])
        else:
            if not runs:
                runs.append([])
                run_indices.append([])
            runs[-1].append(chord)
            run_indices[-1].append(span_index)
    nonempty = [(run, indices) for run, indices in zip(runs, run_indices) if run]
    runs = [run for run, _ in nonempty]
    run_indices = [indices for _, indices in nonempty]
    readable_count = sum(map(len, runs))
    if readable_count < 4:
        return None
    roman_tonic = (key_root + 3) % 12 if mode == "minor" else key_root
    major_degrees = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
    best = None
    rooted_runs = []
    for run in runs:
        roots = [PITCH_NAMES.index(chord[:2] if len(chord) > 1 and chord[1] == "#" else chord[:1])
                 for chord in run]
        rooted_runs.append((run, roots, [PITCH_NAMES[root] for root in roots]))
    for source_run, source_roots, source_names in rooted_runs:
        for length in range(2, min(8, readable_count // 2, len(source_run)) + 1):
            for start in range(len(source_run) - length + 1):
                phrase = tuple(source_names[start:start + length])
                occurrences = []
                for run_index, (run, _, root_names) in enumerate(rooted_runs):
                    cursor = 0
                    while cursor <= len(run) - length:
                        if tuple(root_names[cursor:cursor + length]) in _rotations(list(phrase)):
                            occurrences.append((run_index, cursor))
                            cursor += length
                        else:
                            cursor += 1
                if len(occurrences) < 2:
                    continue
                covered_indices = {
                    run_indices[run_index][cursor + offset]
                    for run_index, cursor in occurrences
                    for offset in range(length)
                }
                total_duration = sum(max(0.0, span["end"] - span["start"]) for span in spans)
                covered_duration = sum(max(0.0, spans[index]["end"] - spans[index]["start"])
                                       for index in covered_indices)
                coverage = covered_duration / total_duration if total_duration else 0.0
                candidate = (coverage, length, len(occurrences), source_run[start:start + length],
                             source_roots[start:start + length])
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
    if best is None:
        return None
    coverage, length, count, display_phrase, phrase_roots = best
    degree_phrase = tuple(major_degrees.get((root - roman_tonic) % 12, -1) for root in phrase_roots)
    name = next((pattern_name for pattern, pattern_name in PATTERNS
                 if degree_phrase == pattern), None)
    if name is None:
        name = next((pattern_name for pattern, pattern_name in PATTERNS
                     if len(pattern) == length and degree_phrase in _rotations(list(pattern))), None)
    return {"chords": display_phrase, "name": name, "coverage": round(coverage, 2), "count": count}


def analyze(tracks, bpm=None, duration=None):
    """分析 notes payload 中的 tracks，返回可直接写入 chordAnalysis 的字段。"""
    all_histogram = _histogram(tracks, ALL_TRACKS)
    if not any(all_histogram):
        return {"key": None, "keyConfidence": 0.0, "spans": [], "loop": None}
    key, key_confidence, key_root, mode = _key(all_histogram)
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        bpm = 0.0
    window = 120 / bpm if math.isfinite(bpm) and bpm > 0 else 2.0
    if duration is None:
        duration = max((float(note["end"]) for name in ALL_TRACKS
                        for note in tracks.get(name) or ()), default=0.0)
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration < 0:
        duration = 0.0
    windows = []
    start = 0.0
    while start < duration:
        end = min(float(duration), start + window)
        histogram = _histogram(tracks, HARMONY_TRACKS, start, end)
        chord, confidence = _match_chord(histogram, _bass_root(tracks, start, end))
        windows.append((chord, start, end, confidence))
        start = end
    spans = _merge_windows(windows)
    return {"key": key, "keyConfidence": key_confidence, "spans": spans,
            "loop": _loop(spans, key_root, mode)}
