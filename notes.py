"""逐轨离线提取音符。

思路源自 whale-listen（migratorywhale，MIT），在本项目以 Basic Pitch
ONNX 后端重写；鼓轨没有稳定音高，故不送入模型。
"""
import gc
import contextlib
import json
import pathlib
import sys

from basic_pitch.inference import predict
from pretty_midi import note_number_to_name

TRACKS = ("vocals", "bass", "guitar", "piano", "other")


def _normalise(event):
    start, end, pitch, velocity = event[:4]
    pitch = int(round(pitch))
    start = float(start)
    end = float(end)
    return {
        "pitch": pitch,
        "note_name": note_number_to_name(pitch),
        "start": round(start, 4),
        "end": round(end, 4),
        "duration": round(end - start, 4),
        "velocity": round(float(velocity), 4),
    }


def extract(stems_dir, out_path=None, predictor=predict):
    """串行提取五条有音高轨；单轨缺失或失败只警告并继续。"""
    stems_dir = pathlib.Path(stems_dir)
    out_path = pathlib.Path(out_path) if out_path is not None else None
    result = {}
    for track in TRACKS:
        source = stems_dir / f"{track}.mp3"
        if not source.exists():
            print(f"音符提取警告：缺轨 {source.name}，跳过", file=sys.stderr)
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                model_output, midi_data, events = predictor(str(source), minimum_note_length=1.0)
            result[track] = [_normalise(event) for event in events]
            del model_output, midi_data, events
        except Exception as error:
            print(f"音符提取警告：{track} 失败：{error}，跳过", file=sys.stderr)
        finally:
            gc.collect()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
