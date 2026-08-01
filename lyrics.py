#!/usr/bin/env python3
"""给一首本地歌配上歌词，写进 ears_cache/<文件名>/analysis.json 的 lyric 字段。

主路走网易云拉 LRC（字准、带时间戳、零算力），也支持喂本地 .lrc / .txt。
网易云只调 search / lyric 两个只读接口，匿名裸调，绝不带 Cookie。

用法：
  python lyrics.py <音频路径> --id <网易云歌ID>
  python lyrics.py <音频路径> --url <歌链接>
  python lyrics.py <音频路径> --search "歌名 歌手"
  python lyrics.py <音频路径> --file <本地.lrc或.txt>
"""
import argparse
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

SEARCH_URL = "https://music.163.com/api/search/get"
LYRIC_URL = "https://music.163.com/api/song/lyric?id={id}&lv=1&tv=-1"
HEADERS = {"Referer": "https://music.163.com", "User-Agent": "Mozilla/5.0 (Tinggu/1.1)"}
TIMEOUT_S = 20
SEARCH_LIMIT = 10
DURATION_TOLERANCE_S = 2.0
TAIL_MISMATCH_S = 30.0
TIMESTAMP_RE = re.compile(r"\[(\d{1,2}):(\d{1,2})(?:[.:](\d{1,3}))?\]")


class LyricError(Exception):
    """人话报错：拿不到词、词为空、参数不对等，都走这里。"""


def fetch_json(url, data=None):
    request = urllib.request.Request(url, data=data, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.URLError as error:
        raise LyricError(f"连不上网易云：{error.reason}") from None
    except (json.JSONDecodeError, ValueError):
        raise LyricError("网易云返回的不是合法 JSON，接口可能变了") from None


def netease_search(query):
    body = urllib.parse.urlencode({"s": query, "type": 1, "limit": SEARCH_LIMIT, "offset": 0}).encode()
    result = fetch_json(SEARCH_URL, data=body)
    songs = ((result.get("result") or {}).get("songs")) or []
    candidates = []
    for song in songs:
        artists = "/".join(artist.get("name", "") for artist in song.get("artists") or [])
        candidates.append({"id": song.get("id"), "name": song.get("name", ""),
                           "artist": artists, "duration_ms": song.get("duration", 0)})
    return candidates


def netease_lyric(song_id):
    result = fetch_json(LYRIC_URL.format(id=song_id))
    lrc = ((result.get("lrc") or {}).get("lyric")) or ""
    tlrc = ((result.get("tlyric") or {}).get("lyric")) or ""
    return lrc, tlrc


def extract_id_from_url(url):
    match = re.search(r"[?&#/]id=(\d+)", url) or re.search(r"song[/?#]+.*?id=(\d+)", url)
    if not match:
        match = re.search(r"id=(\d+)", url)
    if not match:
        raise LyricError("从链接里抠不出歌 ID，改用 --id 直接给数字吧（不跟短链）")
    return match.group(1)


def parse_lrc(text):
    """把 LRC 文本解析成 [[秒, 句], ...]，一行多个时间戳会拆成多条，无戳的元信息行丢弃。"""
    lines = []
    for raw in text.splitlines():
        stamps = list(TIMESTAMP_RE.finditer(raw))
        if not stamps:
            continue
        content = TIMESTAMP_RE.sub("", raw).strip()
        if not content:
            continue
        for stamp in stamps:
            minutes = int(stamp.group(1))
            seconds = int(stamp.group(2))
            fraction = stamp.group(3) or "0"
            millis = int(fraction.ljust(3, "0")[:3])
            lines.append([round(minutes * 60 + seconds + millis / 1000, 2), content])
    lines.sort(key=lambda item: item[0])
    return lines


def local_duration_s(audio_path, cache_dir):
    """先读缓存里的 duration，没有再让 librosa 读文件头。"""
    result_file = cache_dir / "analysis.json"
    try:
        cached = json.loads(result_file.read_text()).get("duration")
        if cached:
            return float(cached)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    import librosa
    return float(librosa.get_duration(path=str(audio_path)))


def lyric_from_netease(song_id, source_id, audio_path, cache_dir, warn_tail):
    lrc, tlrc = netease_lyric(song_id)
    lines = parse_lrc(lrc)
    if not lrc.strip():
        raise LyricError("这首歌没有歌词（纯音乐或无版权），没写缓存")
    if warn_tail and lines:
        duration = local_duration_s(audio_path, cache_dir)
        if lines[-1][0] > duration + TAIL_MISMATCH_S:
            print(f"歌词可能不匹配：末句在 {mmss(lines[-1][0])}，比本地音频（{mmss(duration)}）晚了 30s 以上",
                  file=sys.stderr)
    return {"source": source_id, "lrc": lrc, "tlrc": tlrc, "lines": lines}


def resolve_search(query, audio_path, cache_dir):
    candidates = netease_search(query)
    if not candidates:
        raise LyricError(f"网易云搜「{query}」一个都没搜到，换个关键词试试")
    duration = local_duration_s(audio_path, cache_dir)
    hits = [c for c in candidates if abs(c["duration_ms"] / 1000 - duration) <= DURATION_TOLERANCE_S]
    if len(hits) == 1:
        chosen = hits[0]
        print(f"时长护栏放行：{chosen['name']} - {chosen['artist']}（id {chosen['id']}）", file=sys.stderr)
        return lyric_from_netease(chosen["id"], f"netease:{chosen['id']}", audio_path, cache_dir, warn_tail=False)
    if len(hits) > 1:
        print(f"多个候选时长都对得上（本地 {mmss(duration)}），拿 --id 挑一个：", file=sys.stderr)
        print_candidates(hits)
        raise LyricError("多候选未决，请用 --id 指定")
    print(f"没有候选的时长对得上本地音频（{mmss(duration)}，护栏 ±{DURATION_TOLERANCE_S:.0f}s），"
          f"下面是 top5，确认后用 --id：", file=sys.stderr)
    print_candidates(candidates[:5])
    raise LyricError("零命中，时长护栏拦下，请核对后用 --id")


def print_candidates(candidates):
    for candidate in candidates:
        print(f"  id {candidate['id']} | {candidate['name']} | {candidate['artist']} | "
              f"{mmss(candidate['duration_ms'] / 1000)}", file=sys.stderr)


def lyric_from_file(file_path):
    path = pathlib.Path(file_path).expanduser()
    if not path.is_file():
        raise LyricError(f"歌词文件不存在：{file_path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".lrc":
        lines = parse_lrc(text)
        if not lines:
            raise LyricError(f"{path.name} 里没解析出带时间戳的歌词，是不是纯文本？改成 .txt 后缀")
        return {"source": f"file:{path.name}", "lrc": text, "tlrc": "", "lines": lines}
    if not text.strip():
        raise LyricError(f"{path.name} 是空的")
    return {"source": f"file:{path.name}", "lrc": text, "tlrc": "", "lines": []}


def mmss(seconds):
    seconds = round(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def classify_selector(value):
    """ears.py 用：一个值自动判别是 id / url / file / search。"""
    if "http" in value:
        return "url", value
    if value.isdigit():
        return "id", value
    if pathlib.Path(value).expanduser().is_file():
        return "file", value
    return "search", value


def obtain_lyric(audio_path, cache_dir, mode, value):
    """按 mode 拿到 lyric 字典（不写缓存）；失败抛 LyricError。"""
    if mode == "id":
        return lyric_from_netease(value, f"netease:{value}", audio_path, cache_dir, warn_tail=True)
    if mode == "url":
        song_id = extract_id_from_url(value)
        return lyric_from_netease(song_id, f"netease:{song_id}", audio_path, cache_dir, warn_tail=True)
    if mode == "search":
        return resolve_search(value, audio_path, cache_dir)
    if mode == "file":
        return lyric_from_file(value)
    raise LyricError(f"不认识的取词方式：{mode}")


def write_lyric_cache(cache_dir, audio_path, lyric):
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_file = cache_dir / "analysis.json"
    try:
        data = json.loads(result_file.read_text())
    except (OSError, json.JSONDecodeError):
        data = {"sourcePath": str(audio_path)}
    data["lyric"] = lyric
    result_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def ensure_lyric(audio_path, cache_dir, selector, force):
    """ears.py 的入口：缓存已有 lyric 且没 --force 就复用，否则按 selector 拉一份写进去。"""
    result_file = cache_dir / "analysis.json"
    try:
        existing = json.loads(result_file.read_text()).get("lyric")
    except (OSError, json.JSONDecodeError):
        existing = None
    if existing and not force:
        return existing
    mode, value = classify_selector(selector)
    lyric = obtain_lyric(audio_path, cache_dir, mode, value)
    write_lyric_cache(cache_dir, audio_path, lyric)
    return lyric


def main(argv=None):
    parser = argparse.ArgumentParser(usage="%(prog)s <音频路径> (--id ID | --url 链接 | --search 词 | --file 文件)")
    parser.add_argument("audio", help="本地音频文件（用于定位缓存目录和时长比对）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", help="网易云歌 ID")
    group.add_argument("--url", help="网易云歌链接")
    group.add_argument("--search", help="歌名 歌手")
    group.add_argument("--file", help="本地 .lrc 或 .txt")
    args = parser.parse_args(argv)
    audio_path = pathlib.Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"音频文件不存在：{args.audio}")
    cache_dir = pathlib.Path.cwd() / "ears_cache" / audio_path.stem
    mode, value = next((name, getattr(args, name)) for name in ("id", "url", "search", "file")
                       if getattr(args, name) is not None)
    try:
        lyric = obtain_lyric(audio_path, cache_dir, mode, value)
        write_lyric_cache(cache_dir, audio_path, lyric)
        translated = "，含翻译" if lyric["tlrc"].strip() else ""
        print(f"歌词已写入 {cache_dir / 'analysis.json'}（来源 {lyric['source']}，"
              f"{len(lyric['lines'])} 行带时间戳{translated}）")
        return 0
    except LyricError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
