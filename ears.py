#!/usr/bin/env python3
"""从本地音频生成浅听或深听报告。"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import lyrics

TRACKS = ("vocals", "drums", "bass", "guitar", "piano", "other")
STEM_CN = {"vocals": "人声", "drums": "鼓", "bass": "贝斯", "guitar": "吉他", "piano": "钢琴", "other": "其它"}
STEM_ACTIVE_RATIO = 0.12
STEM_SMOOTH_S = 0.3
STEM_MIN_SEG_S = 1.5
STEM_MERGE_GAP_S = 2.5
VOICE_WIN_S = 25
VOICE_ACTIVE_RATIO = 0.05
AIR_HZ = 5000
TAIL_MAX_S = 8
SR = 22050
HOP = 512


def read_cache(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def has_shallow_cache(path):
    data = read_cache(path)
    return "arrangement" in data and "instruments" in data


def mmss(seconds):
    seconds = round(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def lyric_lines(data):
    return ((data.get("lyric") or {}).get("lines")) or []


def line_at(lines, t):
    """取最后一个起点 <= t 的歌词句，没有则 None。"""
    hit = None
    for start, text in lines:
        if start <= t:
            hit = text
        else:
            break
    return hit


def event_text(event, lines):
    text = f"{mmss(event['t'])} {event['label']}"
    if lines and ("人声" in event["label"] or "歌唱" in event["label"]):
        句 = line_at(lines, event["t"])
        if 句:
            text += f" ♪「{句}」"
    return text


def print_lyric_section(data):
    lyric = data.get("lyric")
    if not lyric:
        return
    print("—— 歌词 ——")
    lines = lyric.get("lines") or []
    if lines:
        for _, text in lines:
            print(text)
        if (lyric.get("tlrc") or "").strip():
            print("（翻译已缓存）")
    else:
        print((lyric.get("lrc") or "").rstrip())


def print_shallow_report(data, cache_dir):
    print(f"=== 《{data.get('name') or ''}》===")
    print(f"时长 {round(data.get('duration', 0))}s | BPM {data.get('bpm', '?')} | 主导音 {data.get('key', '?')}")
    segments = data.get("segments") or []
    if segments:
        peak = max(range(len(segments)), key=lambda i: segments[i]["avgEnergy"])
        low = min(range(len(segments)), key=lambda i: segments[i]["avgEnergy"])
        bar = "".join("▁▂▃▄▅▆▇█"[min(7, int(segment["avgEnergy"] / (segments[peak]["avgEnergy"] or 1) * 7.99))]
                      for segment in segments)
        print(f"能量曲线(六段) {bar}  最烈 {segments[peak]['start']:.0f}-{segments[peak]['end']:.0f}s / 最静 {segments[low]['start']:.0f}-{segments[low]['end']:.0f}s")
    arrangement = data.get("arrangement") or {}
    print("—— 编曲时间轴 ——")
    lines = lyric_lines(data)
    event_texts = [event_text(event, lines) for event in arrangement.get("events", [])]
    for index in range(0, len(event_texts), 8):
        print(" · ".join(event_texts[index:index + 8]))
    vocal_segments = arrangement.get("vocalSegments") or []
    if vocal_segments:
        print("人声：" + ", ".join(f"{mmss(start)}-{mmss(end)}" for start, end in vocal_segments))
    else:
        print("人声：全曲器乐")
    print(f"调性走向：{' '.join(arrangement.get('chromaBySegment') or [])} | 亮度 {arrangement.get('brightnessTrend', '')}")
    print("—— 乐器出没 ——")
    instruments = data.get("instruments") or {}
    rows = sorted(instruments.items(), key=lambda item: item[1][0][0])
    if rows:
        print(" · ".join(f"{name} " + ", ".join(f"{mmss(start)}-{mmss(end)}" for start, end in spans)
                         for name, spans in rows))
    else:
        print("（模型没认出乐器）")
    instruments_meta = data.get("instrumentsMeta") or {}
    if "error" in instruments_meta:
        print(f"乐器针失灵：{instruments_meta['error']}")
    png = pathlib.Path(data.get("spectrogram") or cache_dir / f"{data.get('name', '')}_analysis.png")
    if png.exists():
        print(f"频谱图: {png}")
    print_lyric_section(data)


def ensure_shallow(audio_path, cache_dir, force):
    result_file = cache_dir / "analysis.json"
    if force or not has_shallow_cache(result_file):
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("分析中（librosa 要嚼一会儿）...", file=sys.stderr)
        subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("analyze_song.py")),
                        str(audio_path), str(cache_dir)], check=True, timeout=900)
        error_file = cache_dir / "analyze_error.txt"
        if error_file.exists():
            raise RuntimeError("分析失败:\n" + error_file.read_text())
    return read_cache(result_file)


def require_deep_dependencies():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError("深听需要额外安装：pip install -r requirements-deep.txt") from None


def split_stems(audio_path, destination):
    if all((destination / f"{track}.mp3").exists() for track in TRACKS):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_output = pathlib.Path(tempfile.mkdtemp(prefix="stems-", dir=destination.parent))
    try:
        subprocess.run([
            sys.executable, "-m", "demucs", "-n", "htdemucs_6s", "--mp3",
            "--mp3-bitrate", "128", "-j", "2", "-o", str(temp_output), str(audio_path),
        ], check=True, timeout=1800)
        source = temp_output / "htdemucs_6s" / audio_path.stem
        destination.mkdir(parents=True, exist_ok=True)
        for track in TRACKS:
            source_track = source / f"{track}.mp3"
            if not source_track.exists():
                raise RuntimeError(f"拆轨产物缺少 {track}.mp3")
            destination_track = destination / f"{track}.mp3"
            destination_track.unlink(missing_ok=True)
            shutil.move(str(source_track), str(destination_track))
    finally:
        shutil.rmtree(temp_output, ignore_errors=True)


def smooth_rms(y, librosa, np):
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    width = max(1, round(STEM_SMOOTH_S * SR / HOP))
    return np.convolve(rms, np.ones(width) / width, mode="same")


def active_segments(rms, np):
    if not len(rms) or not np.any(rms):
        return []
    active = rms > np.percentile(rms, 98) * STEM_ACTIVE_RATIO
    edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1) * HOP / SR
    ends = np.flatnonzero(edges == -1) * HOP / SR
    segments = [[float(start), float(end)] for start, end in zip(starts, ends)
                if end - start >= STEM_MIN_SEG_S]
    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] < STEM_MERGE_GAP_S:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [[round(start, 1), round(end, 1)] for start, end in merged]


def window_metrics(y, start, librosa, np):
    beginning = round(start * SR)
    segment = y[beginning:beginning + round(VOICE_WIN_S * SR)]
    _, percussive = librosa.effects.hpss(segment)
    total_energy = float(np.sum(segment ** 2))
    breath = float(np.sum(percussive ** 2) / total_energy) if total_energy else 0.0
    spectrum = np.abs(librosa.stft(segment))
    frequencies = librosa.fft_frequencies(sr=SR)
    spectrum_sum = float(np.sum(spectrum))
    air = float(np.sum(spectrum[frequencies > AIR_HZ]) / spectrum_sum) if spectrum_sum else 0.0
    rms = float(np.mean(librosa.feature.rms(y=segment)[0]))
    return {"start": round(start, 1), "breathNoiseRatio": round(breath, 4),
            "airRatio": round(air, 4), "rms": rms}


def tail_reverb(y, vocal_segments, librosa, np):
    if not vocal_segments:
        return None
    start = round(vocal_segments[-1][1] * SR)
    tail = y[start:start + round(TAIL_MAX_S * SR)]
    if not len(tail):
        return None
    rms = librosa.feature.rms(y=tail, hop_length=HOP)[0]
    if not len(rms) or not np.any(rms):
        return None
    peak = int(np.argmax(rms))
    below = np.flatnonzero(rms[peak:] <= rms[peak] * 0.1)
    if not len(below):
        return None
    return round(float(below[0] * HOP / SR), 2)


def voice_profile(y, rms, vocal_segments, librosa, np):
    threshold = np.percentile(rms, 98) * VOICE_ACTIVE_RATIO
    active = rms > threshold
    if np.count_nonzero(active) * HOP / SR < 10:
        return None
    active_indices = np.flatnonzero(active)
    first = active_indices[0] * HOP / SR
    last = active_indices[-1] * HOP / SR
    window_frames = round(VOICE_WIN_S * SR / HOP)
    candidates = []
    for start in np.arange(np.ceil(first), np.floor(last) + 0.001, 1.0):
        frame = round(start * SR / HOP)
        window_rms = rms[frame:frame + window_frames]
        window_active = active[frame:frame + window_frames]
        if np.count_nonzero(window_active) < window_frames / 2:
            continue
        candidates.append((float(np.mean(window_rms[window_active])), float(start)))
    if not candidates:
        return None
    soft = window_metrics(y, min(candidates)[1], librosa, np)
    burst = window_metrics(y, max(candidates)[1], librosa, np)
    loudness = burst["rms"] / soft["rms"] if soft["rms"] else None
    return {"softWindow": soft, "burstWindow": burst,
            "loudnessRatio": round(loudness, 1) if loudness is not None else None,
            "tailReverb": tail_reverb(y, vocal_segments, librosa, np)}


def run_deep(data, cache_dir, force):
    require_deep_dependencies()
    if "deepVersion" in data and not force:
        return data
    import librosa
    import numpy as np

    destination = cache_dir / "stems"
    split_stems(pathlib.Path(data["sourcePath"]), destination)
    timeline = {}
    vocals_y = None
    vocals_rms = None
    for track in TRACKS:
        y, _ = librosa.load(destination / f"{track}.mp3", sr=SR, mono=True)
        rms = smooth_rms(y, librosa, np)
        timeline[track] = active_segments(rms, np)
        if track == "vocals":
            vocals_y, vocals_rms = y, rms
    profile = voice_profile(vocals_y, vocals_rms, timeline["vocals"], librosa, np)
    data.update({"stemTimeline": timeline, "voiceProfile": profile, "deepVersion": 1})
    (cache_dir / "analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


def print_deep_report(data):
    print(f"=== 深听 ·《{data.get('name') or ''}》===")
    print("—— 乐器起止（六轨实测）——")
    timeline = data["stemTimeline"]
    rows = sorted(timeline.items(), key=lambda item: item[1][0][0] if item[1] else float("inf"))
    print(" · ".join(f"{STEM_CN[track]} " +
                     (", ".join(f"{mmss(start)}-{mmss(end)}" for start, end in segments) if segments else "无")
                     for track, segments in rows))
    lines = lyric_lines(data)
    vocal_segments = timeline.get("vocals") or []
    if lines and vocal_segments:
        print("—— 人声落词 ——")
        for start, end in vocal_segments:
            句 = line_at(lines, start)
            print(f"人声 {mmss(start)}-{mmss(end)}" + (f" ♪「{句}」" if 句 else ""))
    print("—— 嗓音质地 ——")
    profile = data["voiceProfile"]
    if profile is None:
        print("器乐曲，嗓音质地跳过")
        return
    soft = profile["softWindow"]
    burst = profile["burstWindow"]
    soft_lyric = f" ♪「{line_at(lines, soft['start'])}」" if lines and line_at(lines, soft["start"]) else ""
    burst_lyric = f" ♪「{line_at(lines, burst['start'])}」" if lines and line_at(lines, burst["start"]) else ""
    print(f"轻唱窗(起点 {mmss(soft['start'])})：气息噪声 {soft['breathNoiseRatio'] * 100:.1f}% | 空气感 {soft['airRatio'] * 100:.1f}%{soft_lyric}")
    print(f"爆发窗(起点 {mmss(burst['start'])})：气息噪声 {burst['breathNoiseRatio'] * 100:.1f}% | 空气感 {burst['airRatio'] * 100:.1f}%{burst_lyric}")
    tail = "null" if profile["tailReverb"] is None else f"{profile['tailReverb']:.2f}s"
    print(f"响度倍数 {profile['loudnessRatio']:.1f} | 尾音混响 {tail}")


def main(argv=None):
    parser = argparse.ArgumentParser(usage="%(prog)s <音频路径> [--deep] [--force] [--lyric 值]")
    parser.add_argument("audio", help="本地音频文件")
    parser.add_argument("--deep", action="store_true", help="运行六轨与嗓音质地深听")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重新计算")
    parser.add_argument("--lyric", help="配歌词：网易云 ID / 链接 / 本地 .lrc·.txt / \"歌名 歌手\"（自动判别）")
    args = parser.parse_args(argv)
    audio_path = pathlib.Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"音频文件不存在：{args.audio}")
    cache_dir = pathlib.Path.cwd() / "ears_cache" / audio_path.stem
    try:
        data = ensure_shallow(audio_path, cache_dir, args.force)
        if data.get("sourcePath") != str(audio_path):
            data["sourcePath"] = str(audio_path)
            (cache_dir / "analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        if args.lyric:
            data["lyric"] = lyrics.ensure_lyric(audio_path, cache_dir, args.lyric, args.force)
        print_shallow_report(data, cache_dir)
        if args.deep:
            print_deep_report(run_deep(data, cache_dir, args.force))
        return 0
    except subprocess.TimeoutExpired as error:
        print(f"失败：子进程超过 {error.timeout} 秒未完成", file=sys.stderr)
    except subprocess.CalledProcessError as error:
        print(f"失败：子进程退出码 {error.returncode}", file=sys.stderr)
    except Exception as error:
        print(str(error), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
