#!/usr/bin/env python3
"""从本地音频生成浅听或深听报告。"""
import argparse
import math
import json
import pathlib
import shutil
import subprocess
import sys
from collections import Counter
import tempfile

import lyrics

TRACKS = ("vocals", "drums", "bass", "guitar", "piano", "other")
NOTE_TRACKS = ("vocals", "bass", "guitar", "piano", "other")
NOTES_VERSION = 2      # v1.3：摘要改音域/常见音结构
CHORDS_VERSION = 2     # v1.3：音符路 + chroma 路双路印证
TEXTURE_VERSION = 3    # v1.3：声线质感带口袋标尺与锚点定义值
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
    """取词并剔掉混进来的制作名单（LRC 惯把名单塞头尾，会挂到 0:00 的人声事件上）。"""
    lines = ((data.get("lyric") or {}).get("lines")) or []
    vocals = (data.get("stemTimeline") or {}).get("vocals") or []
    return lyrics.strip_credits(lines, *lyrics.vocal_span(vocals))


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
        lyric_text = line_at(lines, event["t"])
        if lyric_text:
            text += f" ♪「{lyric_text}」"
    return text


def print_lyric_section(data):
    lyric = data.get("lyric")
    if not lyric:
        return
    print("—— 歌词 ——")
    lines = lyric.get("lines") or []
    if lines:
        previous = None
        for _, text in lines:
            if text != previous:
                print(text)
            previous = text
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


def memory_gate(required_mb, step_name):
    """检查 Linux 可用内存；没有 /proc/meminfo 的平台直接放行。"""
    meminfo = pathlib.Path("/proc/meminfo")
    if not meminfo.exists():
        return True
    available = None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) // 1024
            break
    if available is None:
        print("内存检查失败：/proc/meminfo 缺少 MemAvailable。", file=sys.stderr)
        return False
    if available < required_mb:
        print(f"内存不足：{step_name} 需要约 {required_mb}MB，当前可用 {available}MB。关闭一些程序后重试。",
              file=sys.stderr)
        return False
    return True


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


def tension_curve(data, motion, stem_rms_by_track, np):
    """张力轮廓：和声紧张度 0.5 + 力度斜率 0.3 + 配器密度 0.2。
    RMS 提力度弧线的思路来自 Ocean Listen（ennisaaaaaaaa-stack, MIT）；此处只输出机制量数。"""
    duration = float(data.get("duration") or 0)
    frame_rate = SR / HOP
    length = max((len(values) for values in stem_rms_by_track.values()), default=0)
    total = np.zeros(length)
    for values in stem_rms_by_track.values():
        total[:len(values)] += values
    std = float(np.std(total))
    timeline = data.get("stemTimeline") or {}

    def harmonic(t):
        if not motion or motion.get("mode") != "functional":
            return 0.4
        values = {"延宕": 0.9, "张力上升": 0.7, "调外": 0.5, "平稳": 0.3,
                  "解决": 0.1, "强解决": 0.0, "?": 0.4}
        state = next((item["state"] for item in motion.get("states", [])
                      if item["start"] <= t < item["end"]), None)
        return values.get(state, 0.4)

    curve = []
    for t in np.arange(0, duration + 0.001, 2.0):
        end = min(length, round(t * frame_rate) + 1)
        start = max(0, round((t - 10) * frame_rate))
        window = total[start:end]
        if len(window) > 1 and std:
            slope = float(np.polyfit(np.arange(len(window)) / frame_rate, window, 1)[0])
            dyn = min(1.0, max(0.0, slope / std))
        else:
            dyn = 0.0
        active = sum(any(start_at <= t < end_at for start_at, end_at in timeline.get(track, []))
                     for track in TRACKS)
        value = 0.5 * harmonic(float(t)) + 0.3 * dyn + 0.2 * active / 6
        curve.append([round(float(t), 1), round(value, 2)])

    smooth_width = max(1, round(3 * frame_rate))
    smoothed = np.convolve(total, np.ones(smooth_width) / smooth_width, mode="same") if length else total
    arc_parts = []
    threshold = std * 0.08
    for start in np.arange(0, duration, 8.0):
        end = min(duration, start + 8.0)
        a, b = min(length, round(start * frame_rate)), min(length, round(end * frame_rate))
        delta = float(smoothed[max(a, b - 1)] - smoothed[min(a, length - 1)]) if length and b > a else 0.0
        kind = "渐强" if delta > threshold else ("渐弱" if delta < -threshold else "平台")
        if arc_parts and arc_parts[-1]["type"] == kind:
            arc_parts[-1]["end"] = round(end, 1)
        else:
            arc_parts.append({"start": round(float(start), 1), "end": round(end, 1), "type": kind})
    if len(arc_parts) > 1 and arc_parts[-1]["end"] - arc_parts[-1]["start"] < 8:
        arc_parts[-2]["end"] = arc_parts[-1]["end"]
        arc_parts.pop()
    arcs = []
    for arc in arc_parts:
        a = min(length, round(arc["start"] * frame_rate))
        b = min(length, round(arc["end"] * frame_rate))
        edge = max(1, round(frame_rate))
        head = float(np.mean(smoothed[a:min(b, a + edge)])) if b > a else 0.0
        tail = float(np.mean(smoothed[max(a, b - edge):b])) if b > a else 0.0
        arc["deltaPct"] = round((tail - head) / (abs(head) + float(np.mean(total)) * 0.05) * 100) if length else 0
        arcs.append(arc)
    peak = max(curve, key=lambda item: item[1])[0] if curve else None
    release = None
    if peak is not None and motion:
        release = next((item["start"] for item in motion.get("states", [])
                        if item["start"] > peak and item["state"] in ("解决", "强解决")), None)
    return {"motionVersion": 1, "curve": curve, "arcs": arcs, "peak": peak, "release": release}

def build_note_payload(raw_tracks):
    """过三道闸并生成音符轨与摘要。"""
    import harmonic_filter
    filtered_tracks = {}
    filter_stats = {}
    summary = {}
    for track in NOTE_TRACKS:
        raw = raw_tracks.get(track) or []
        filtered, stats = harmonic_filter.filter_notes(track, raw)
        filtered_tracks[track] = filtered
        filter_stats[track] = stats
        pitches = [note["pitch"] for note in filtered]
        common = Counter(note["note_name"] for note in filtered).most_common(5)
        summary[track] = {
            "noteCount": len(filtered),
            "beforeFilter": len(raw),
            "pitchLow": min(pitches) if pitches else None,
            "pitchHigh": max(pitches) if pitches else None,
            "topNotes": [name for name, _ in common],
            "filterStats": stats,
        }
    return {"notesVersion": NOTES_VERSION, "tracks": filtered_tracks, "filterStats": filter_stats}, summary

def _pitch_name(pitch):
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def _merged_gaps(notes_by_track, duration):
    intervals = sorted((note["start"], note["end"])
                       for track_notes in notes_by_track.values() for note in track_notes)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor > 0.5:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor > 0.5:
        gaps.append((cursor, duration))
    return sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:3]

def print_melody(summary, notes_by_track, duration):
    """打印最多 25 行旋律线摘要。"""
    lines = ["—— 附 · 旋律线 ——"]
    for track in NOTE_TRACKS:
        item = summary.get(track) or {}
        count = item.get("noteCount", 0)
        before = item.get("beforeFilter", 0)
        removed = (before - count) / before * 100 if before else 0.0
        low, high = item.get("pitchLow"), item.get("pitchHigh")
        pitch_range = f"{_pitch_name(low)}–{_pitch_name(high)}" if low is not None else "无音高"
        common = " ".join(item.get("topNotes") or []) or "无"
        lines.append(f"{track}: {count} 音符（滤掉 {removed:.0f}%）  {pitch_range}  常见 {common}")

    vocals = notes_by_track.get("vocals") or []
    contour = []
    for start in range(0, max(0, int(duration)) + 1, 15):
        pitches = [note["pitch"] for note in vocals if start <= note["start"] < start + 15]
        if pitches:
            average = round(sum(pitches) / len(pitches))
            contour.append(f"{mmss(start)} {_pitch_name(average)}[{_pitch_name(min(pitches))}–{_pitch_name(max(pitches))}]")
    if contour:
        lines.append("人声轮廓(15s): " + " · ".join(contour))

    all_notes = [note for track_notes in notes_by_track.values() for note in track_notes]
    bin_count = max(1, int(math.ceil(duration / 10)))
    density = [sum(start <= note["start"] < start + 10 for note in all_notes)
               for start in range(0, bin_count * 10, 10)]
    gaps = _merged_gaps(notes_by_track, duration)
    reserved = 1 if gaps else 0
    if density and max(density) != min(density):
        glyphs = "▁▂▃▄▅▆▇█"
        peak = max(density) or 1
        room = max(0, 25 - len(lines) - reserved)
        for index, count in list(enumerate(density))[:room]:
            bar = glyphs[min(7, round(count / peak * 7))]
            lines.append(f"密度 {mmss(index * 10)} {bar} {count}")
    if gaps and len(lines) < 25:
        lines.append("最长静默: " + " · ".join(
            f"{mmss(start)}-{mmss(end)}({end - start:.1f}s)" for start, end in gaps))
    print("\n".join(lines[:25]))

def print_chords(data):
    """打印紧凑和弦段；缺失或稀薄时明确降级，不影响其他报告。"""
    analysis = data.get("chordAnalysis") or {}
    print("—— 地基 · 和弦 ——")
    key = analysis.get("key")
    if key:
        root, mode = key.split()
        print(f"调性: {root} {'大调' if mode == 'major' else '小调'} (置信 {analysis.get('keyConfidence', 0):.2f})")
    spans = analysis.get("spans") or []
    readable = [span for span in spans if span.get("chord") != "?"]
    total_time = sum(float(span["end"]) - float(span["start"]) for span in spans)
    read_time = sum(float(span["end"]) - float(span["start"]) for span in readable)
    coverage = read_time / total_time if total_time else 0.0
    if readable:  # "?" 窗藏进 json，比例必须报——藏细节可以，藏比例不行
        chord_list = " · ".join(f"{mmss(span['start'])} {span['chord']}"
                                for span in readable[:8 if coverage < 0.25 else 24])
        head = "和弦只认出零星几处: " if coverage < 0.25 else "进行: "
        print(f"{head}{chord_list}（可读覆盖 {coverage * 100:.0f}%）")
    if not readable:
        print("和弦稀薄/色彩复杂，读不出稳定进行")
        stats = analysis.get("sourceStats") or {}
        if stats:
            print("来源: " + " · ".join((
                f"双路印证 {(float(stats.get('both', 0)) + float(stats.get('root-agree', 0))) * 100:.0f}%",
                f"仅音符 {float(stats.get('notes', 0)) * 100:.0f}%",
                f"仅chroma {float(stats.get('chroma', 0)) * 100:.0f}%",
            )))
        return
    loop = analysis.get("loop")
    if loop:
        label = f" ({loop['name']})" if loop.get("name") else ""
        count = f" ×{loop['count']}" if loop.get("count") else ""
        print(f"主循环: {'–'.join(loop['chords'])}{label}{count} 覆盖 {loop.get('coverage', 0) * 100:.0f}%")
    else:
        print("和弦稀薄/色彩复杂，读不出稳定进行")
    stats = analysis.get("sourceStats") or {}
    if stats:
        print("来源: " + " · ".join((
            f"双路印证 {(float(stats.get('both', 0)) + float(stats.get('root-agree', 0))) * 100:.0f}%",
            f"仅音符 {float(stats.get('notes', 0)) * 100:.0f}%",
            f"仅chroma {float(stats.get('chroma', 0)) * 100:.0f}%",
        )))

def print_motion_and_tension(data):
    print("—— 地基 · 和声运动 ——")
    motion = data.get("harmonyMotion")
    if not motion:
        print("和声运动：证据不足，留白")
    elif motion.get("mode") == "nonfunctional":
        print("和声运动：非功能语法，留白")
    elif motion.get("mode") == "loop":
        note = motion.get("loopNote") or {}
        loop = "–".join(note.get("chords") or []) or "?"
        print(f"主循环: {loop} 覆盖 {float(note.get('coverage') or 0) * 100:.0f}%")
        resolutions = note.get("resolutions") or []
        if resolutions:
            print("循环内解决点: " + " · ".join(mmss(item) for item in resolutions))
    else:
        states = motion.get("states") or []
        if not states:
            print("和声运动：证据不足，留白")
        else:  # 全量 states 住 json；报告只印摘要，"?" 是断口不是状态，不上榜
            tally = Counter(item["state"] for item in states if item["state"] != "?")
            counted = " · ".join(f"{state} {count}" for state, count in tally.most_common())
            print(f"功能语法（贴合度 {motion.get('functionalShare', 0) * 100:.0f}%）· {counted}")
            resolutions = [item for item in states if item["state"] in ("解决", "强解决")]
            if resolutions:
                marks = [f"{mmss(item['start'])}{'(强)' if item['state'] == '强解决' else ''}"
                         for item in resolutions[:8]]
                more = f" 等共{len(resolutions)}处" if len(resolutions) > 8 else ""
                print("解决点: " + " · ".join(marks) + more)
            stretches = [item for item in states if item["state"] in ("延宕", "张力上升")]
            if stretches:
                longest = max(stretches, key=lambda item: item["end"] - item["start"])
                print(f"最长张力段 {mmss(longest['start'])}-{mmss(longest['end'])} {longest['state']}")

    print("—— 地基 · 张力轮廓 ——")
    tension = data.get("tension")
    if not tension:
        print("张力轮廓：证据不足，留白")
        return
    arcs = tension.get("arcs") or []
    # 全量弧线住 json；报告只印有戏的——平台和 |Δ|<20% 的微弧不上榜
    loud = [item for item in arcs if item["type"] != "平台" and abs(item["deltaPct"]) >= 20]
    if loud:
        skipped = len(arcs) - len(loud)
        print("力度弧线: " + " · ".join(
            f"{mmss(item['start'])}-{mmss(item['end'])} {item['type']} {item['deltaPct']:+d}%"
            for item in loud) + (f"（另 {skipped} 段平缓入 json）" if skipped else ""))
    else:
        print("力度弧线: 全曲平缓，无显著起伏")
    peak = "?" if tension.get("peak") is None else mmss(tension["peak"])
    release = "?" if tension.get("release") is None else mmss(tension["release"])
    print(f"张力顶点 {peak} · 回落 {release}")
    for salience in (data.get("salience") or {}).get("peaks", []):
        t = salience["t"]
        state = next((item["state"] for item in (motion or {}).get("states", [])
                      if item["start"] - 2 <= t <= item["end"] + 2), None)
        evidence = " · ".join(salience.get("evidence") or [])
        suffix = f" + {state}中" if state and state != "?" else ""
        print(f"注意力峰对表: {mmss(t)} ← {evidence}{suffix}")

def _metric(value, pattern, blank="?"):
    return blank if value is None else pattern.format(value)


def print_voice_lines(data, cache_dir, lyric_lines=()):
    print("—— 人声 · 按句 ——")
    payload = read_cache(cache_dir / "lines.json") if data.get("lineVersion") else {}
    rows = payload.get("rows") or []
    if not rows:
        if not lyric_lines or not ((data.get("stemTimeline") or {}).get("vocals") or []):
            print("器乐曲，行级表跳过")
        else:
            print("跨模态 · 行级表：证据不足，留白")
    else:
        shown = rows
        omitted = 0
        if len(rows) > 45:
            peaks = [item["t"] for item in (data.get("salience") or {}).get("peaks", [])]
            selected = {0, len(rows) - 1}
            selected.update(index for index, row in enumerate(rows)
                            if any(abs(row["t"] - peak) <= 4 for peak in peaks))
            shown = [row for index, row in enumerate(rows) if index in selected]
            omitted = len(rows) - len(shown)
        for row in shown:  # 报告行只留四样；亮度/哑度/噪比住 sidecar，供需要细粒度的读者自取
            air = _metric(row.get("airRatio"), "{:.0%}")
            stable = _metric(row.get("pitchStabSt"), "±{:.2f}st")
            motion = f" | {row['motion']}" if row.get("motion") else ""
            accomp = ("独", "薄", "中", "中", "厚", "厚")[min(5, row["accompStems"])]
            print(f"{mmss(row['t'])} 「{row['text']}」  声{row['rmsDb']:+.1f}dB 气{air} "
                  f"稳{stable} | {accomp}{motion}")
        if omitted:
            print(f"另 {omitted} 行入 sidecar")


ANCHOR_VALUE_FMT = {"airiestLine": ("气", "{:.0%}"), "brightestLine": ("", "{:.0f}Hz"),
                    "roughestLine": ("哑", "{:.1f}%"), "steadiestLine": ("", "±{:.2f}st")}


def print_texture(data):
    print("—— 人声 · 声线质感 ——")
    item = data.get("voiceTexture")
    if not item:
        print("声线质感：证据不足，留白")
        return
    print("标尺: 气声（耳语≈100·美声<5）· 亮度（暗<2000·亮>3000）· 噪比（光滑>15·毛<8）· 稳度（直线<1·大摆>3）")
    print("中位: " + " · ".join((
        f"亮度 {_metric(item.get('centroidHz'), '{:.0f}Hz')}",
        f"气声 {_metric(item.get('airRatio'), '{:.0%}')}",
        f"哑度 {_metric(item.get('jitterPct'), '{:.1f}%')}",
        f"噪比 {_metric(item.get('hnrDb'), '{:.1f}dB')}",
        f"稳度 {_metric(item.get('pitchStabSt'), '±{:.2f}st')}",
        f"倾斜 {_metric(item.get('tiltDb'), '{:.1f}dB')}",
    )))
    contrast = item.get("contrast")
    if contrast:
        soft, loud = contrast.get("soft") or {}, contrast.get("loud") or {}
        print(f"轻唱行({contrast.get('softLines', 0)}) vs 推声行({contrast.get('loudLines', 0)}): "
              f"气声 {_metric(soft.get('airRatio'), '{:.0%}')}→{_metric(loud.get('airRatio'), '{:.0%}')} · "
              f"亮度 {_metric(soft.get('centroidHz'), '{:.0f}')}→{_metric(loud.get('centroidHz'), '{:.0f}')}Hz · "
              f"噪比 {_metric(soft.get('hnrDb'), '{:.1f}')}→{_metric(loud.get('hnrDb'), '{:.1f}')}dB")
    anchors = []
    for label, key in (("最气", "airiestLine"), ("最亮", "brightestLine"),
                       ("最毛", "roughestLine"), ("最稳", "steadiestLine")):
        line = item.get(key)
        if line:
            prefix, fmt = ANCHOR_VALUE_FMT[key]
            value = line.get("value")
            tail = f" {prefix}{fmt.format(value)}" if value is not None else ""
            anchors.append(f"{label} {mmss(line['t'])}「{line['text']}」{tail}")
    if anchors:
        print("锚点: " + " · ".join(anchors))

def clean_note_tracks(payload):
    """保留合法音符；单轨或单条损坏时就地降级，不拖垮整份 notes。"""
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict):
        return {}
    cleaned = {}
    for track, notes_in_track in tracks.items():
        if not isinstance(notes_in_track, list):
            cleaned[track] = []
            continue
        cleaned[track] = [note for note in notes_in_track
                          if isinstance(note, dict) and
                          all(isinstance(note.get(field), (int, float))
                              for field in ("pitch", "start", "end"))]
    return cleaned

def print_journey(data):
    """听感时间轴：把各层数值按歌的时间编成一股辫子，
    让读报告的顺序等于歌走过的路。只变排列，输出仍是机制词。"""
    entries = []
    vocals = (data.get("stemTimeline") or {}).get("vocals") or []
    if vocals:
        entries.append((float(vocals[0][0]), 1, "人声进"))
    for peak in (data.get("salience") or {}).get("peaks", []):
        entries.append((float(peak["t"]), 0, " ·".join(peak.get("evidence") or []) or "注意力峰"))
    motion = data.get("harmonyMotion") or {}
    for item in (motion.get("states") or []):
        if item["state"] in ("解决", "强解决"):
            entries.append((float(item["start"]), 1, item["state"]))
        elif item["state"] == "延宕":
            entries.append((float(item["start"]), 1, "延宕起"))
    tension = data.get("tension") or {}
    arcs = sorted((a for a in tension.get("arcs") or [] if a["type"] != "平台" and abs(a["deltaPct"]) >= 40),
                  key=lambda a: -abs(a["deltaPct"]))[:4]
    for arc in arcs:
        entries.append((float(arc["start"]), 3, f"{arc['type']} {arc['deltaPct']:+d}%"))
    if tension.get("peak") is not None:
        entries.append((float(tension["peak"]), 0, "张力顶点"))
    if tension.get("release") is not None:
        entries.append((float(tension["release"]), 0, "回落"))
    texture = data.get("voiceTexture") or {}
    for label, key in (("最气一句", "airiestLine"), ("最毛一句", "roughestLine")):
        line = texture.get(key)
        if line:
            entries.append((float(line["t"]), 2, f"「{line['text']}」{label}"))
    if not entries:
        return
    entries.sort(key=lambda item: (item[0], item[1]))
    rows = []
    for t, _, text in entries:
        if rows and t - rows[-1][0] <= 3.0:
            rows[-1][1].append(text)
        else:
            rows.append([t, [text]])
    print("—— 听感时间轴 ——")
    for t, texts in rows[:15]:
        print(f"{mmss(t)}  " + " ｜ ".join(dict.fromkeys(texts)))
    if len(rows) > 15:
        print(f"（另 {len(rows) - 15} 拍略，细节在下方分节）")


def run_deep(data, cache_dir, force):
    require_deep_dependencies()
    import librosa
    import numpy as np

    destination = cache_dir / "stems"
    result_file = cache_dir / "analysis.json"
    line_file = cache_dir / "lines.json"

    if force:
        for field in ("deepVersion", "stemTimeline", "voiceProfile",
                      "notes", "noteSummary", "notesVersion", "chordAnalysis", "chordsVersion",
                      "harmonyMotion", "motionVersion", "tension", "lineVersion", "voiceTexture"):
            data.pop(field, None)
        try:
            line_file.unlink()
        except FileNotFoundError:
            pass

    needs_deep = force or "deepVersion" not in data
    needs_notes = force or data.get("notesVersion") != NOTES_VERSION
    needs_chords = force or needs_notes or data.get("chordsVersion") != CHORDS_VERSION
    needs_motion = force or needs_chords or "motionVersion" not in data
    needs_tension = force or needs_deep or needs_motion or "tension" not in data
    needs_lines = (force or needs_deep or needs_motion or "lineVersion" not in data
                   or not line_file.exists())
    needs_texture = (force or needs_lines or "voiceTexture" not in data
                     or (data.get("voiceTexture") or {}).get("textureVersion") != TEXTURE_VERSION)

    stem_rms_by_track = {}
    if needs_deep:
        if not all((destination / f"{track}.mp3").exists() for track in TRACKS):
            if not memory_gate(4000, "拆轨"):
                raise RuntimeError("深听已停止。")
            split_stems(pathlib.Path(data["sourcePath"]), destination)
        timeline = {}
        vocals_y = None
        vocals_rms = None
        for track in TRACKS:
            y, _ = librosa.load(destination / f"{track}.mp3", sr=SR, mono=True)
            rms = smooth_rms(y, librosa, np)
            stem_rms_by_track[track] = rms
            timeline[track] = active_segments(rms, np)
            if track == "vocals":
                vocals_y, vocals_rms = y, rms
        profile = voice_profile(vocals_y, vocals_rms, timeline["vocals"], librosa, np)
        data.update({"stemTimeline": timeline, "voiceProfile": profile, "deepVersion": 1})
        lyrics.atomic_write_json(result_file, data)

    if needs_notes:
        if not memory_gate(1500, "音符提取"):
            raise RuntimeError("深听已停止。")
        try:
            import notes

            payload, summary = build_note_payload(notes.extract(destination, out_path=None))
            data.update({"notes": payload["tracks"], "noteSummary": summary,
                         "notesVersion": NOTES_VERSION})
            lyrics.atomic_write_json(result_file, data)
            for track, stats in payload["filterStats"].items():
                print(f"音符滤网 {track}: 音域 {stats['range']} / 时长 {stats['duration']} / "
                      f"重叠 {stats['overlap']} / {stats['input']} → {stats['output']}", file=sys.stderr)
        except Exception as error:
            # 本轮失败就不留旧账冒充新结果，和弦层跟着一起撤
            print(f"本轮音符提取失败，不复用旧 notes：{type(error).__name__}", file=sys.stderr)
            for field in ("notes", "noteSummary", "notesVersion", "chordAnalysis", "chordsVersion"):
                data.pop(field, None)
            lyrics.atomic_write_json(result_file, data)

    chord_failed = False
    if needs_chords:
        import chords
        import chroma_chords

        notes_analysis = None
        chroma_analysis = None
        try:
            note_tracks = clean_note_tracks({"tracks": data.get("notes") or {}})
            if not note_tracks:
                raise ValueError("没有可用的音符轨")
            notes_analysis = chords.analyze(note_tracks, data.get("bpm"), data.get("duration"))
            if not notes_analysis.get("spans"):
                raise ValueError("没有可辨识的和弦窗")
        except Exception as error:
            notes_analysis = None
            print(f"音符路和弦警告：{type(error).__name__}", file=sys.stderr)
        try:
            chroma_analysis = chroma_chords.analyze_with_key(destination, data.get("bpm"),
                                                             data.get("duration"))
            if not chroma_analysis.get("spans"):
                raise ValueError("没有 chroma 和弦窗")
        except Exception as error:
            chroma_analysis = None
            print(f"chroma 路和弦警告：{type(error).__name__}", file=sys.stderr)
        try:
            if notes_analysis is None and chroma_analysis is None:
                raise ValueError("双路和弦均不可用")
            analysis = chords.assemble_dual(notes_analysis, chroma_analysis,
                                            data.get("bpm"), data.get("duration"))
            if not analysis.get("spans"):
                raise ValueError("没有可辨识的和弦窗")
            analysis["chordsVersion"] = CHORDS_VERSION
            data.update({"chordAnalysis": analysis, "chordsVersion": CHORDS_VERSION})
            lyrics.atomic_write_json(result_file, data)
        except Exception as error:
            chord_failed = True
            print(f"和弦分析警告：{type(error).__name__}（深听报告照常出）", file=sys.stderr)

    if needs_motion:
        import harmony_motion

        if chord_failed or not data.get("chordAnalysis"):
            data.pop("harmonyMotion", None)
            data.pop("motionVersion", None)
        else:
            motion = harmony_motion.analyze(data["chordAnalysis"])
            if motion is not None:
                data.update({"harmonyMotion": motion, "motionVersion": 1})

    if needs_lines:
        import line_table

        lines = lyric_lines(data)
        vocal_track = destination / "vocals.mp3"
        vocal_segments = (data.get("stemTimeline") or {}).get("vocals") or []
        if lines and vocal_track.exists() and vocal_segments:
            try:
                payload = line_table.build(data, lines, destination, line_file)
                data["lineVersion"] = payload["lineVersion"]
                data["voiceTexture"] = line_table.texture(payload["rows"])
            except Exception as error:
                data.pop("lineVersion", None)
                data.pop("voiceTexture", None)
                print(f"行级表跳过：{type(error).__name__}", file=sys.stderr)
        else:
            data.pop("lineVersion", None)
            data.pop("voiceTexture", None)
            print("没有歌词或没有人声轨，行级表跳过（用 --lyric 配词后可得）", file=sys.stderr)
    elif needs_texture and data.get("lineVersion"):
        import line_table

        data["voiceTexture"] = line_table.texture(read_cache(line_file).get("rows") or [])

    if needs_tension:
        if not stem_rms_by_track:
            for track in TRACKS:
                y, _ = librosa.load(destination / f"{track}.mp3", sr=SR, mono=True)
                stem_rms_by_track[track] = smooth_rms(y, librosa, np)
        data["tension"] = tension_curve(data, data.get("harmonyMotion"), stem_rms_by_track, np)

    lyrics.atomic_write_json(result_file, data)
    if chord_failed:  # 本轮算砸的和弦不进报告，但已落盘的旧账不动
        data = dict(data)
        data.pop("chordAnalysis", None)
    return data


def print_deep_report(data, cache_dir):
    """三层排列：故事（时间轴）→ 人声 → 地基（和弦/和声运动/张力），编曲底账沉附录。"""
    print(f"=== 深听 ·《{data.get('name') or ''}》===")
    lines = lyric_lines(data)
    print_journey(data)
    print_texture(data)
    print("—— 人声 · 轻唱与爆发 ——")
    profile = data["voiceProfile"]
    if profile is None:
        print("器乐曲，嗓音质地跳过")
    else:
        soft = profile["softWindow"]
        burst = profile["burstWindow"]
        soft_lyric = f" ♪「{line_at(lines, soft['start'])}」" if lines and line_at(lines, soft["start"]) else ""
        burst_lyric = f" ♪「{line_at(lines, burst['start'])}」" if lines and line_at(lines, burst["start"]) else ""
        print(f"轻唱窗(起点 {mmss(soft['start'])})：气息噪声 {soft['breathNoiseRatio'] * 100:.1f}% | 空气感 {soft['airRatio'] * 100:.1f}%{soft_lyric}")
        print(f"爆发窗(起点 {mmss(burst['start'])})：气息噪声 {burst['breathNoiseRatio'] * 100:.1f}% | 空气感 {burst['airRatio'] * 100:.1f}%{burst_lyric}")
        tail = "null" if profile["tailReverb"] is None else f"{profile['tailReverb']:.2f}s"
        print(f"响度倍数 {profile['loudnessRatio']:.1f} | 尾音混响 {tail}")
    print_voice_lines(data, cache_dir, lines)
    print_chords(data)
    print_motion_and_tension(data)
    print("—— 附 · 乐器起止（六轨实测）——")
    timeline = data["stemTimeline"]
    rows = sorted(timeline.items(), key=lambda item: item[1][0][0] if item[1] else float("inf"))
    print(" · ".join(f"{STEM_CN[track]} " +
                     (", ".join(f"{mmss(start)}-{mmss(end)}" for start, end in segments) if segments else "无")
                     for track, segments in rows))
    if data.get("notesVersion"):
        print_melody(data.get("noteSummary") or {},
                     clean_note_tracks({"tracks": data.get("notes") or {}}),
                     data.get("duration") or 0)


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
        cached_source = read_cache(cache_dir / "analysis.json").get("sourcePath")
        if cached_source and cached_source != str(audio_path):
            print("缓存来自别的文件，重新分析", file=sys.stderr)
            shutil.rmtree(cache_dir, ignore_errors=True)
        data = ensure_shallow(audio_path, cache_dir, args.force)
        if data.get("sourcePath") != str(audio_path):
            data["sourcePath"] = str(audio_path)
            lyrics.atomic_write_json(cache_dir / "analysis.json", data)
        lyric_failed = False
        if args.lyric:
            try:
                data["lyric"] = lyrics.ensure_lyric(audio_path, cache_dir, args.lyric, args.force)
            except lyrics.LyricError as error:
                print(f"配词失败：{error}（报告照常打印）", file=sys.stderr)
                lyric_failed = True
        print_shallow_report(data, cache_dir)
        if args.deep:
            print_deep_report(run_deep(data, cache_dir, args.force), cache_dir)
        return 1 if lyric_failed else 0
    except subprocess.TimeoutExpired as error:
        print(f"失败：子进程超过 {error.timeout} 秒未完成", file=sys.stderr)
    except subprocess.CalledProcessError as error:
        print(f"失败：子进程退出码 {error.returncode}", file=sys.stderr)
    except Exception as error:
        print(str(error), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
