"""从无鼓无人声的分轨 chroma 直接识别和弦。"""

import math
import pathlib

import librosa
import numpy as np

import chords


SR = 22050
HOP = 512
HARMONY_TRACKS = ("piano", "guitar", "other")


def _stem_path(stems_dir, track):
    directory = pathlib.Path(stems_dir)
    for suffix in (".mp3", ".wav", ".flac", ".m4a"):
        path = directory / f"{track}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(track)


def _window_seconds(bpm):
    try:
        value = float(bpm)
    except (TypeError, ValueError):
        value = 0.0
    return 120 / value if math.isfinite(value) and value > 0 else 2.0


def _grid(bpm, duration):
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration < 0:
        duration = 0.0
    width = _window_seconds(bpm)
    start = 0.0
    while start < duration:
        end = min(duration, start + width)
        yield start, end
        start = end


def _match_chord(vector, bass_root=None):
    """用 chords.py 的模板匹配一个归一化 chroma 窗。"""
    values = np.asarray(vector, dtype=float)
    total = float(np.sum(values))
    if values.shape != (12,) or not np.all(np.isfinite(values)) or total <= 0:
        return "?", 0.0
    values = values / total
    candidates = []
    for root in range(12):
        for suffix, template in chords.TEMPLATES:
            tones = {(root + interval) % 12 for interval, _ in template}
            score = sum(values[(root + interval) % 12] * weight
                        for interval, weight in template)
            score -= sum(value * 0.3 for pitch, value in enumerate(values)
                         if pitch not in tones)
            score /= sum(weight for _, weight in template)
            if bass_root == root:
                score *= 1.25
            candidates.append((score, root, suffix))
    candidates.sort(reverse=True)
    best, second = candidates[:2]
    confidence = best[0] / max(second[0], 0.01) if best[0] > 0 else 0.0  # 向量已归一，下限用同量纲 0.01
    name = chords.PITCH_NAMES[best[1]] + best[2]
    if confidence < chords.CONFIDENCE_THRESHOLD:
        name = "?"
    return name, round(confidence, 2)


def _mean_window(chroma, start, end):
    first = max(0, int(math.floor(start * SR / HOP)))
    last = min(chroma.shape[1], max(first + 1, int(math.ceil(end * SR / HOP))))
    return np.mean(chroma[:, first:last], axis=1) if first < chroma.shape[1] else np.zeros(12)


def analyze_with_key(stems_dir, bpm, duration):
    """返回 chroma 路的调性与未归并逐窗 spans。"""
    harmonic = None
    for track in HARMONY_TRACKS:
        y, _ = librosa.load(_stem_path(stems_dir, track), sr=SR, mono=True)
        harmonic = y if harmonic is None else _sum_audio(harmonic, y)
    bass, _ = librosa.load(_stem_path(stems_dir, "bass"), sr=SR, mono=True)
    harmonic = harmonic if harmonic is not None else np.zeros(0, dtype=float)
    harmonic_chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=HOP)
    bass_chroma = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=HOP)
    histogram = np.sum(harmonic_chroma, axis=1)
    key, confidence, _, _ = chords._key(histogram.tolist()) if np.any(histogram) else (None, 0.0, 0, "major")
    spans = []
    for start, end in _grid(bpm, duration):
        vector = _mean_window(harmonic_chroma, start, end)
        bass_vector = _mean_window(bass_chroma, start, end)
        bass_root = int(np.argmax(bass_vector)) if np.any(bass_vector) else None
        name, chord_confidence = _match_chord(vector, bass_root)
        spans.append({"chord": name, "start": round(start, 2), "end": round(end, 2),
                      "conf": chord_confidence})
    return {"key": key, "keyConfidence": confidence, "spans": spans}


def _sum_audio(left, right):
    length = max(len(left), len(right))
    result = np.zeros(length, dtype=float)
    result[:len(left)] += left
    result[:len(right)] += right
    return result


def analyze(stems_dir, bpm, duration):
    """返回与 notes 路完全相同网格的 chroma 和弦窗。"""
    return analyze_with_key(stems_dir, bpm, duration)["spans"]
