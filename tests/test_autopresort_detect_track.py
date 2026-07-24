import importlib


def test_propagate_boxes_reuses_last_valid():
    dt = importlib.import_module("auto_presort.detect_track")
    raw = [(10, 10, 50, 50), None, None, (20, 20, 60, 60)]
    out = dt.propagate_boxes(raw)
    assert out[1] == (10, 10, 50, 50)
    assert out[2] == (10, 10, 50, 50)
    assert out[3] == (20, 20, 60, 60)


def test_propagate_boxes_leading_none_stays_none():
    dt = importlib.import_module("auto_presort.detect_track")
    out = dt.propagate_boxes([None, None, (1, 1, 2, 2)])
    assert out[0] is None and out[1] is None and out[2] == (1, 1, 2, 2)


def test_no_animal_spans_from_flags():
    dt = importlib.import_module("auto_presort.detect_track")
    flags = [True, False, False, False, False, True, True]  # 0.1..0.5 缺失=0.4s
    spans = dt.no_animal_spans(flags, fps=10.0, min_gap_sec=0.3)
    assert spans == [(0.1, 0.5)]


def test_no_animal_spans_ignores_short_dropouts():
    dt = importlib.import_module("auto_presort.detect_track")
    flags = [True, False, True, True]
    assert dt.no_animal_spans(flags, fps=10.0, min_gap_sec=0.3) == []


def test_no_animal_spans_trailing_missing():
    dt = importlib.import_module("auto_presort.detect_track")
    flags = [True, False, False, False]  # 末尾 0.1..0.4 缺失=0.3s
    assert dt.no_animal_spans(flags, fps=10.0, min_gap_sec=0.3) == [(0.1, 0.4)]
