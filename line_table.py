"""逐行对齐歌词、人声声学量、伴奏密度与和声运动。"""
import json
import pathlib
import sys

import librosa
import numpy as np


SR = 22050
HOP = 512
AIR_HZ = 5000
VOICE_ACTIVE_RATIO = 0.05
TRACKS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def window_metrics(segment, sr=SR):
    """返回任意音频切片的气声与呼吸噪声能量比。"""
    if not len(segment):
        return {"airRatio": 0.0, "breathRatio": 0.0}
    harmonic, percussive = librosa.effects.hpss(segment)
    total_energy = float(np.sum(segment ** 2))
    breath = float(np.sum(percussive ** 2) / total_energy) if total_energy else 0.0
    spectrum = np.abs(librosa.stft(segment))
    frequencies = librosa.fft_frequencies(sr=sr)
    spectrum_sum = float(np.sum(spectrum))
    air = float(np.sum(spectrum[frequencies > AIR_HZ]) / spectrum_sum) if spectrum_sum else 0.0
    return {"airRatio": round(air, 4), "breathRatio": round(breath, 4)}


def line_windows(lines, vocal_segments):
    """按下一行或十二秒封顶切窗；末行落到其后的最近人声段终点。"""
    windows = []
    endings = sorted(float(segment[1]) for segment in (vocal_segments or []))
    for index, (start, text) in enumerate(lines):
        start = float(start)
        if index + 1 < len(lines):
            end = min(float(lines[index + 1][0]), start + 12.0)
        else:
            end = next((candidate for candidate in endings if candidate >= start), start)
        if end - start >= 0.5:
            windows.append((start, end, text))
    return windows


def _praat_metrics(segment, available=True):
    if not available:
        return None, None, None
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(segment, sampling_frequency=SR)
    point = call(sound, "To PointProcess (periodic, cc)", 80, 800)
    jitter = call(point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([sound, point], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 80, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    values = (jitter * 100, shimmer * 100, hnr)
    return tuple(round(float(value), 1) if np.isfinite(value) else None for value in values)


def _tilt(segment):
    spectrum = np.mean(np.abs(librosa.stft(segment)), axis=1)
    frequencies = librosa.fft_frequencies(sr=SR)
    selected = (frequencies >= 200) & (frequencies <= 8000) & (spectrum > 0)
    if np.count_nonzero(selected) < 2:
        return None
    slope = np.polyfit(np.log10(frequencies[selected]),
                       20 * np.log10(spectrum[selected] + np.finfo(float).tiny), 1)[0]
    return round(float(slope), 1)


def _pitch_stability(segment):
    f0, voiced, _ = librosa.pyin(segment, fmin=80, fmax=800, sr=SR)
    values = f0[voiced & np.isfinite(f0)]
    if len(values) < 10:
        return None
    semitones = 12 * np.log2(values / 440.0)
    return round(float(np.std(semitones)), 2)


def _motion_at(data, start):
    motion = data.get("harmonyMotion") or {}
    return next((item.get("state", "") for item in motion.get("states", [])
                 if item.get("start", 0) <= start < item.get("end", 0)), "")


def _active_accompaniment(data, midpoint):
    timeline = data.get("stemTimeline") or {}
    return sum(any(start <= midpoint < end for start, end in timeline.get(track, []))
               for track in TRACKS if track != "vocals")


def build(data, lines, stems_dir, out_path):
    """构建行级表，写 sidecar，并返回包含版本号与 rows 的结构。"""
    vocals, _ = librosa.load(pathlib.Path(stems_dir) / "vocals.mp3", sr=SR, mono=True)
    frame_rms = librosa.feature.rms(y=vocals, hop_length=HOP)[0]
    threshold = np.percentile(frame_rms, 98) * VOICE_ACTIVE_RATIO if len(frame_rms) else 0.0
    active = frame_rms[frame_rms > threshold]
    reference = float(np.median(active)) if len(active) else 0.0
    windows = line_windows(lines, (data.get("stemTimeline") or {}).get("vocals", []))

    try:
        import parselmouth  # noqa: F401
        praat_available = True
    except ImportError:
        praat_available = False
        print("未装 parselmouth，声线三项跳过", file=sys.stderr)

    rows = []
    skipped = len(lines) - len(windows)
    praat_warned = False
    for start, end, lyric in windows:
        segment = vocals[round(start * SR):round(end * SR)]
        line_rms = float(np.sqrt(np.mean(segment ** 2))) if len(segment) else 0.0
        rms_db = 20 * np.log10(line_rms / reference) if line_rms > 0 and reference > 0 else -float("inf")
        if rms_db < -25:
            skipped += 1
            continue
        ratios = window_metrics(segment)
        centroid = librosa.feature.spectral_centroid(y=segment, sr=SR)
        try:
            jitter, shimmer, hnr = _praat_metrics(segment, praat_available)
        except Exception as error:
            jitter = shimmer = hnr = None
            if not praat_warned:
                print(f"声线三项计算跳过：{type(error).__name__}", file=sys.stderr)
                praat_warned = True
        rows.append({
            "t": round(start, 1), "text": lyric, "rmsDb": round(rms_db, 1),
            "airRatio": ratios["airRatio"], "breathRatio": ratios["breathRatio"],
            "centroidHz": round(float(np.mean(centroid))) if centroid.size else None,
            "tiltDb": _tilt(segment), "pitchStabSt": _pitch_stability(segment),
            "jitterPct": jitter, "shimmerPct": shimmer, "hnrDb": hnr,
            "accompStems": _active_accompaniment(data, (start + end) / 2),
            "motion": _motion_at(data, start),
        })

    result = {"lineVersion": 1, "rows": rows, "skipped": skipped}
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def texture(rows):
    """汇总行级声学量；不把量数翻译为主观标签。"""
    fields = ("centroidHz", "tiltDb", "airRatio", "jitterPct", "shimmerPct", "hnrDb", "pitchStabSt")
    result = {field: (round(float(np.median(values)), 2) if (values := [row[field] for row in rows
                             if row.get(field) is not None]) else None) for field in fields}
    rough = max((row for row in rows if row.get("jitterPct") is not None),
                key=lambda row: row["jitterPct"], default=None)
    steady = min((row for row in rows if row.get("pitchStabSt") is not None),
                 key=lambda row: row["pitchStabSt"], default=None)
    result["roughestLine"] = ({"t": rough["t"], "text": rough["text"], "value": rough["jitterPct"]}
                              if rough else None)
    result["steadiestLine"] = ({"t": steady["t"], "text": steady["text"], "value": steady["pitchStabSt"]}
                               if steady else None)

    def _anchor(field, pick_max=True):
        pool = [row for row in rows if row.get(field) is not None]
        if not pool:
            return None
        row = max(pool, key=lambda item: item[field]) if pick_max else min(pool, key=lambda item: item[field])
        return {"t": row["t"], "text": row["text"], "value": row[field]}

    result["airiestLine"] = _anchor("airRatio")
    result["brightestLine"] = _anchor("centroidHz")

    # 轻唱行/推声行对比（按行响度中位二分）：主副歌的粗代理，不宣称曲式
    voiced = [row for row in rows if row.get("rmsDb") is not None]
    if voiced:
        mid = float(np.median([row["rmsDb"] for row in voiced]))
        soft = [row for row in voiced if row["rmsDb"] <= mid]
        loud = [row for row in voiced if row["rmsDb"] > mid]

        def _median(pool, field):
            values = [row[field] for row in pool if row.get(field) is not None]
            return round(float(np.median(values)), 2) if values else None

        result["contrast"] = {
            "softLines": len(soft), "loudLines": len(loud),
            "soft": {field: _median(soft, field) for field in ("airRatio", "centroidHz", "hnrDb")},
            "loud": {field: _median(loud, field) for field in ("airRatio", "centroidHz", "hnrDb")},
        }
    result["textureVersion"] = 3
    return result
