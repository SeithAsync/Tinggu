import lyrics


LINES = [
    [0.5, "作词 : 黄雨篙"],
    [15.0, "熄灭蜡烛 夜就涌入"],
    [200.0, "要一束光只为我照亮"],
    [233.6, "吉他 Guitar by:宋星凯"],
    [236.6, "Mixing Engineer: 李游@55tec studio"],
    [237.8, "出品公司：北京有此山文化传媒有限公司"],
]


def test_tail_credits_stripped_with_vocal_span():
    kept = lyrics.strip_credits(LINES, 14.0, 213.0)
    assert [text for _, text in kept] == ["作词 : 黄雨篙", "熄灭蜡烛 夜就涌入", "要一束光只为我照亮"]
    # 头部名单归 parse_lrc 的 CREDIT_RE 管，strip_credits 不重复裁——这里只验尾巴


def test_no_vocal_span_keeps_everything():
    assert lyrics.strip_credits(LINES, None, None) == LINES


def test_real_lyric_inside_span_survives_pattern():
    lines = [[100.0, "策划好的告别也会痛"]]  # 命中模式词但落在人声区间内：不剔
    assert lyrics.strip_credits(lines, 10.0, 200.0) == lines


def test_vocal_span_helper():
    assert lyrics.vocal_span([]) == (None, None)
    assert lyrics.vocal_span([[14.2, 98.0], [115.0, 213.4]]) == (14.2, 213.4)
