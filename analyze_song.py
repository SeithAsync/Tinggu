#!/usr/bin/env python3
"""分析本地音频并将结构化结果与频谱图写入指定目录。

用法：python analyze_song.py <音频路径> <输出目录>
"""
import json
import math
import os
import sys
from pathlib import Path

# 调音旋钮：阈值常数集中在此
HOP = 512
SMOOTH_S = 0.25
BAND_ENTRY_RATIO = 0.15
BAND_SUSTAIN_S = 1.0
PERC_ENTRY_RATIO = 0.25
PERC_SUSTAIN_S = 0.5
VOCAL_RATIO = 0.45
VOCAL_MIN_SEG_S = 1.5
SED_CHUNK_S = 60
SED_SMOOTH_S = 0.5
SED_ON = 0.12
SED_OFF = 0.06
SED_MIN_SEG_S = 2.0
SED_MERGE_GAP_S = 3.0
BANDS = [("低频", 20, 120), ("中低", 120, 350), ("中频", 350, 2000), ("高频", 2000, 6000), ("气声", 6000, 11025)]
BAND_PLOT_LABELS = ["low", "low-mid", "mid", "high", "air"]

INSTRUMENT_GROUPS = {
    "吉他": ["Guitar", "Acoustic guitar", "Electric guitar", "Strum"],
    "贝斯": ["Bass guitar"],
    "鼓": ["Drum kit", "Drum", "Snare drum", "Bass drum", "Hi-hat", "Cymbal", "Percussion", "Tabla"],
    "钢琴": ["Piano", "Electric piano"],
    "合成器": ["Synthesizer", "Sampler"],
    "弦乐": ["Violin, fiddle", "Cello", "String section", "Orchestra"],
    "管乐": ["Trumpet", "Saxophone", "Flute", "Clarinet", "Trombone", "Harmonica", "Accordion"],
    "风琴": ["Organ", "Electronic organ", "Hammond organ"],
    "歌唱": ["Singing", "Male singing", "Female singing", "Child singing", "Choir", "Humming"],
    "说话声": ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Narration, monologue"],
    "儿童声": ["Children playing", "Child speech, kid speaking", "Children shouting", "Babbling"],
    "口哨": ["Whistling", "Whistle"],
}


def atomic_write_json(path, data):
    """写同目录临时文件再 os.replace，防写一半留半截 JSON（单进程 CLI，原子替换即够）。"""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    os.replace(temp, path)


SALIENCE_NAMES = ("总能量", "低频", "中低", "中频", "高频", "气声频段", "打击能量", "亮度")


def analyze_salience(curves, events, duration, fps):
    """从已在内存的八路曲线提取注意力候选区，不读取音频。"""
    import numpy as np
    seconds = max(0, int(math.ceil(duration)))
    if seconds == 0:
        return {"salienceVersion": 1, "peaks": []}
    grid = []
    for curve in curves:
        values = np.asarray(curve, dtype=float)
        row = []
        for second in range(seconds):
            start, end = round(second * fps), round((second + 1) * fps)
            chunk = values[max(0, start):max(start + 1, end)]
            row.append(float(np.mean(chunk)) if len(chunk) else 0.0)
        grid.append(np.asarray(row))
    contributions = np.zeros((len(grid), seconds))
    signed_short = np.zeros((len(grid), seconds))
    def window_mean(values, start, end, fallback):
        window = values[max(0, start):min(seconds, end)]
        return float(np.mean(window)) if len(window) else float(values[fallback])
    for channel, values in enumerate(grid):
        short = np.zeros(seconds)
        long = np.zeros(seconds)
        for t in range(seconds):
            short[t] = window_mean(values, t, t + 1, t) - window_mean(values, t - 1, t, t)
            long[t] = window_mean(values, t, t + 10, t) - window_mean(values, t - 10, t, t)
        signed_short[channel] = short
        for delta in (short, long):
            absolute = np.abs(delta)
            std = float(np.std(absolute))
            if std > 0:
                contributions[channel] += (absolute - float(np.mean(absolute))) / std
    scores = contributions.sum(axis=0)
    event_map = {}
    for event in events or []:
        second = min(seconds - 1, max(0, round(float(event.get("t", 0)))))
        scores[second] += 2.0
        event_map.setdefault(second, []).append(event.get("label", ""))
    selected = []
    for t in sorted(range(seconds), key=lambda item: scores[item], reverse=True):
        if scores[t] < 3.0 or any(abs(t - old) < 8 for old in selected):
            continue
        selected.append(t)
        if len(selected) == 5:
            break
    peaks = []
    for t in sorted(selected):
        evidence = list(dict.fromkeys(label for label in event_map.get(t, []) if label))
        for channel in np.argsort(contributions[:, t])[::-1]:
            if len(evidence) >= 3:
                break
            values = grid[channel]
            before = window_mean(values, t - 1, t, t)
            delta = float(signed_short[channel, t])
            denominator = abs(before) + float(np.mean(np.abs(values))) * 0.05
            percent = round(delta / denominator * 100) if denominator else 0
            evidence.append(f"{SALIENCE_NAMES[channel]} {percent:+d}%")
        peaks.append({"t": float(t), "score": round(float(scores[t]), 1), "evidence": evidence[:3]})
    return {"salienceVersion": 1, "peaks": peaks}


def smooth_curve(curve, sr, np):
    window_frames = max(1, round(SMOOTH_S * sr / HOP))
    return np.convolve(curve, np.ones(window_frames) / window_frames, mode="same")


def first_sustained_entry(curve, threshold, sustain_s, sr, np):
    sustain_frames = max(1, round(sustain_s * sr / HOP))
    run = 0
    for i, is_above in enumerate(curve > threshold):
        run = run + 1 if is_above else 0
        if run >= sustain_frames:
            return round(float((i - sustain_frames + 1) * HOP / sr), 1)
    return None


def merge_short_segments(mask, min_frames):
    mask = mask.copy()
    while True:
        boundaries = [0]
        boundaries.extend(i for i in range(1, len(mask)) if mask[i] != mask[i - 1])
        boundaries.append(len(mask))
        short_index = next((i for i in range(len(boundaries) - 1)
                            if boundaries[i + 1] - boundaries[i] < min_frames), None)
        if short_index is None or len(boundaries) == 2:
            return mask
        start, end = boundaries[short_index], boundaries[short_index + 1]
        if short_index == 0:
            neighbor_value = mask[end]
        elif short_index == len(boundaries) - 2:
            neighbor_value = mask[start - 1]
        else:
            prev_len = start - boundaries[short_index - 1]
            next_len = boundaries[short_index + 2] - end
            neighbor_value = mask[start - 1] if prev_len >= next_len else mask[end]
        mask[start:end] = neighbor_value


def detect_instruments(audio_path):
    import librosa
    import numpy as np
    from panns_inference import SoundEventDetection
    from panns_inference.config import labels

    audio, sr = librosa.load(str(audio_path), sr=32000, mono=True)
    duration = len(audio) / sr
    sed = SoundEventDetection(checkpoint_path=None, device="cpu")
    chunk_samples = SED_CHUNK_S * sr
    framewise_chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        framewise_chunks.append(sed.inference(chunk[None, :])[0])
    framewise = np.concatenate(framewise_chunks, axis=0)
    fps = len(framewise) / duration
    label_indices = {label: index for index, label in enumerate(labels)}
    missing_labels = []
    segments = {}
    for group_name, group_labels in INSTRUMENT_GROUPS.items():
        indices = []
        for label in group_labels:
            if label in label_indices:
                indices.append(label_indices[label])
            else:
                missing_labels.append(label)
        if not indices:
            continue
        curve = framewise[:, indices].max(axis=1)
        smooth_frames = max(1, round(SED_SMOOTH_S * fps))
        curve = np.convolve(curve, np.ones(smooth_frames) / smooth_frames, mode="same")
        raw_segments = []
        segment_start = None
        for index, probability in enumerate(curve):
            if segment_start is None and probability >= SED_ON:
                segment_start = index / fps
            elif segment_start is not None and probability < SED_OFF:
                segment_end = index / fps
                if segment_end - segment_start >= SED_MIN_SEG_S:
                    raw_segments.append([segment_start, segment_end])
                segment_start = None
        if segment_start is not None:
            segment_end = duration
            if segment_end - segment_start >= SED_MIN_SEG_S:
                raw_segments.append([segment_start, segment_end])
        merged_segments = []
        for start, end in raw_segments:
            if merged_segments and start - merged_segments[-1][1] < SED_MERGE_GAP_S:
                merged_segments[-1][1] = end
            else:
                merged_segments.append([start, end])
        if merged_segments:
            segments[group_name] = [[round(start, 1), round(end, 1)] for start, end in merged_segments]
    return {"segments": segments, "meta": {"fps": fps, "missingLabels": missing_labels}}


def analyze(audio_file, output_dir):
    import librosa
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "analysis.json"
    err_file = output_dir / "analyze_error.txt"
    y, sr = librosa.load(str(audio_file), sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    times_rms = librosa.times_like(rms, sr=sr, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    dominant_key = keys[int(np.argmax(np.mean(chroma, axis=1)))]

    y_harm, y_perc = librosa.effects.hpss(y)
    perc_rms = librosa.feature.rms(y=y_perc, hop_length=HOP)[0]
    perc_smooth = smooth_curve(perc_rms, sr, np)
    percussive_entry = first_sustained_entry(
        perc_smooth, np.percentile(perc_rms, 90) * PERC_ENTRY_RATIO, PERC_SUSTAIN_S, sr, np)

    stft_magnitude = np.abs(librosa.stft(y, hop_length=HOP))
    frequencies = librosa.fft_frequencies(sr=sr)
    band_results = []
    band_curves = []
    for name, low, high in BANDS:
        bins = (frequencies >= low) & (frequencies < high)
        band_smooth = smooth_curve(stft_magnitude[bins].mean(axis=0), sr, np)
        band_curves.append((name, band_smooth))
        entry = first_sustained_entry(band_smooth, band_smooth.max() * BAND_ENTRY_RATIO,
                                      BAND_SUSTAIN_S, sr, np)
        band_results.append({"name": name, "entry": entry})

    seg_count = 6
    seg_len = len(rms) // seg_count
    segments = []
    for i in range(seg_count):
        seg = rms[i * seg_len:(i + 1) * seg_len]
        segments.append({
            "start": round(float(times_rms[i * seg_len]), 1),
            "end": round(float(times_rms[min((i + 1) * seg_len - 1, len(times_rms) - 1)]), 1),
            "avgEnergy": round(float(np.mean(seg)), 4),
            "maxEnergy": round(float(np.max(seg)), 4),
        })
    chroma_seg_len = chroma.shape[1] // seg_count
    chroma_by_segment = [keys[int(np.argmax(chroma[:, i * chroma_seg_len:(i + 1) * chroma_seg_len].mean(axis=1)))]
                         for i in range(seg_count)]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    centroid_seg_len = len(centroid) // seg_count
    brightness_by_segment = [round(float(centroid[i * centroid_seg_len:(i + 1) * centroid_seg_len].mean()), 0)
                             for i in range(seg_count)]
    first_brightness = np.mean(brightness_by_segment[:2])
    last_brightness = np.mean(brightness_by_segment[-2:])
    brightness_ratio = last_brightness / first_brightness
    brightness_trend = "rising" if brightness_ratio > 1.15 else "falling" if brightness_ratio < 0.87 else "flat"

    harm_stft = np.abs(librosa.stft(y_harm, hop_length=HOP))
    vocal_bins = (frequencies >= 300) & (frequencies < 2500)
    vocal_curve = harm_stft[vocal_bins].sum(axis=0) / (harm_stft.sum(axis=0) + np.finfo(float).eps)
    vocal_mask = merge_short_segments(smooth_curve(vocal_curve, sr, np) > VOCAL_RATIO,
                                      max(1, round(VOCAL_MIN_SEG_S * sr / HOP)))
    vocal_segments = []
    vocal_start = None
    for i, is_vocal in enumerate(vocal_mask):
        if is_vocal and vocal_start is None:
            vocal_start = i
        elif not is_vocal and vocal_start is not None:
            vocal_segments.append([round(float(vocal_start * HOP / sr), 1), round(float(i * HOP / sr), 1)])
            vocal_start = None
    if vocal_start is not None:
        vocal_segments.append([round(float(vocal_start * HOP / sr), 1), round(duration, 1)])

    events = []
    for band in band_results:
        if band["entry"] is not None:
            events.append({"t": band["entry"], "label": f'{band["name"]}进场'})
    if percussive_entry is not None:
        events.append({"t": percussive_entry, "label": "打击进场"})
    for start, end in vocal_segments:
        events.extend(({"t": start, "label": "人声进"}, {"t": end, "label": "人声退"}))
    try:
        instrument_result = detect_instruments(audio_file)
        instruments = instrument_result["segments"]
        instruments_meta = instrument_result["meta"]
    except Exception as error:
        instruments = {}
        instruments_meta = {"error": str(error)}
    for name, instrument_segments in instruments.items():
        for start, end in instrument_segments:
            events.extend(({"t": start, "label": f"{name}进场"}, {"t": end, "label": f"{name}退潮"}))
    events.sort(key=lambda event: (event["t"], event["label"]))
    salience_events = [dict(event, label=event["label"].replace("退潮", "退场"))
                       for event in events if event["label"].startswith("人声") or
                       any(event["label"].startswith(name) for name in instruments)]
    salience = analyze_salience([rms] + [curve for _, curve in band_curves] + [perc_smooth, centroid],
                                salience_events, duration, sr / HOP)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    fig, axes = plt.subplots(4, 1, figsize=(14, 13))
    mel_s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(mel_s, ref=np.max), sr=sr, x_axis="time", y_axis="mel", ax=axes[0])
    axes[0].set_title("Mel Spectrogram")
    librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=axes[1])
    axes[1].set_title("Chromagram")
    axes[2].plot(times_rms, rms, color="#e74c3c", linewidth=0.8, label="total")
    axes[2].plot(librosa.times_like(perc_rms, sr=sr, hop_length=HOP), perc_rms,
                 color="#3498db", linewidth=0.8, label="percussive")
    axes[2].fill_between(times_rms, rms, alpha=0.3, color="#e74c3c")
    axes[2].set_title("Energy (RMS)")
    axes[2].legend()
    band_times = librosa.times_like(band_curves[0][1], sr=sr, hop_length=HOP)
    for plot_label, (_, band_curve) in zip(BAND_PLOT_LABELS, band_curves):
        axes[3].plot(band_times, band_curve, linewidth=0.8, label=plot_label)
    axes[3].set_title("Frequency Bands")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    plt.tight_layout()
    img_path = output_dir / f"{audio_file.stem}_analysis.png"
    plt.savefig(str(img_path), dpi=150)
    plt.close()

    result = {
        "name": audio_file.stem, "duration": round(duration, 1), "bpm": round(tempo), "key": dominant_key,
        "segments": segments, "spectrogram": str(img_path), "instruments": instruments,
        "instrumentsMeta": instruments_meta,
        "salience": salience, "salienceVersion": 1,
        "arrangement": {"percussiveEntry": percussive_entry, "bands": band_results,
                        "chromaBySegment": chroma_by_segment, "brightnessBySegment": brightness_by_segment,
                        "brightnessTrend": brightness_trend, "vocalSegments": vocal_segments, "events": events},
    }
    # 覆盖前把歌词和源路径从旧结果搬过来，别让重算把浅听之外的字段冲掉
    try:
        old = json.loads(result_file.read_text())
    except (OSError, json.JSONDecodeError):
        old = {}
    for keep in ("lyric", "sourcePath"):
        if keep in old:
            result[keep] = old[keep]
    atomic_write_json(result_file, result)
    err_file.unlink(missing_ok=True)


def main():
    if len(sys.argv) != 3:
        sys.exit("用法: analyze_song.py <音频路径> <输出目录>")
    audio_file = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path(sys.argv[2]).expanduser().resolve()
    try:
        analyze(audio_file, output_dir)
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analyze_error.txt").write_text(str(error) + "\n")
        raise


if __name__ == "__main__":
    main()
