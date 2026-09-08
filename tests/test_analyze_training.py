"""训练日志回滚分段与策略拟合指标解释的回归测试。"""

from __future__ import annotations

import csv

import pytest

from scripts.analyze_training import (
    load_metric_segments,
    load_metrics,
    main,
    report_flags,
    report_metrics,
    window_summary,
)


def _write_metrics(tmp_path, rows):
    path = tmp_path / "metrics.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["iter"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _iters(rows):
    return [int(row["iter"]) for row in rows]


def test_rollback_splits_physical_order_instead_of_globally_merging(tmp_path):
    path = _write_metrics(tmp_path, [
        {"iter": 100, "policy_loss": 1.0},
        {"iter": 102, "policy_loss": 1.2},
        {"iter": 101, "policy_loss": 2.1},
        {"iter": 102, "policy_loss": 2.2},
        {"iter": 103, "policy_loss": 2.3},
        {"iter": 99, "policy_loss": 3.0},
        {"iter": 100, "policy_loss": 3.1},
    ])
    segments = load_metric_segments(path)
    assert [_iters(rows) for rows in segments] == [[100, 102], [101, 102, 103], [99, 100]]
    assert _iters(load_metrics(path)) == [99, 100]
    assert load_metrics(path, segment=0)[-1]["policy_loss"] == "1.2"
    assert load_metrics(path, segment=1)[1]["policy_loss"] == "2.2"
    assert load_metrics(path, segment=2) == load_metrics(path, segment=-1)


def test_same_iteration_keeps_last_row_only_inside_its_segment(tmp_path):
    path = _write_metrics(tmp_path, [
        {"iter": 10, "policy_loss": "old"},
        {"iter": 10, "policy_loss": "replacement"},
        {"iter": 11, "policy_loss": "first-run"},
        {"iter": 10, "policy_loss": "rollback"},
        {"iter": 10, "policy_loss": "rollback-replacement"},
        {"iter": 11, "policy_loss": "second-run"},
    ])
    segments = load_metric_segments(path)
    assert [_iters(rows) for rows in segments] == [[10, 11], [10, 11]]
    assert [rows[0]["policy_loss"] for rows in segments] == ["replacement", "rollback-replacement"]
    assert segments[0][-1]["policy_loss"] == "first-run"


@pytest.mark.parametrize("contents", [None, "", "iter,policy_loss\n", "iter\ninvalid\nnan\ninf\n"])
def test_missing_empty_or_unparseable_metrics_return_no_segments(tmp_path, contents):
    path = tmp_path / "metrics.csv"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    assert load_metric_segments(path) == []
    assert load_metrics(path) == []
    with pytest.raises(IndexError, match="共 0 段"):
        load_metrics(path, segment=0)


def test_invalid_iteration_rows_do_not_hide_a_rollback(tmp_path):
    path = _write_metrics(tmp_path, [{"iter": value} for value in [20, "", "iter", "nan", "inf", "1.5", 19]])
    assert [_iters(rows) for rows in load_metric_segments(path)] == [[20], [19]]


@pytest.mark.parametrize("segment", [-3, -2, 2, 3])
def test_out_of_range_segment_rejected(tmp_path, segment):
    path = _write_metrics(tmp_path, [{"iter": 2}, {"iter": 1}])
    with pytest.raises(IndexError, match=f"segment={segment}.*共 2 段"):
        load_metrics(path, segment=segment)


def test_cli_reports_segment_count_and_defaults_to_last(tmp_path, capsys):
    _write_metrics(tmp_path, [{"iter": 100}, {"iter": 101}, {"iter": 90}, {"iter": 91}])
    assert main(["--log-dir", str(tmp_path), "--recent", "1"]) == 0
    report = capsys.readouterr().out
    assert "共 2 段，已选 segment=1" in report
    assert "iter 90 .. 91（共 2 行）" in report
    assert "iter 100" not in report
    assert main(["--log-dir", str(tmp_path), "--segment", "0", "--recent", "1"]) == 0
    report = capsys.readouterr().out
    assert "共 2 段，已选 segment=0" in report
    assert "iter 100 .. 101（共 2 行）" in report


def test_cli_invalid_segment_is_a_clear_error(tmp_path, capsys):
    _write_metrics(tmp_path, [{"iter": 1}])
    assert main(["--log-dir", str(tmp_path), "--segment", "1"]) == 2
    captured = capsys.readouterr()
    assert "segment=1 越界：共 1 段" in captured.err
    assert not captured.out


def test_cli_empty_file_is_a_clear_error(tmp_path, capsys):
    _write_metrics(tmp_path, [])
    assert main(["--log-dir", str(tmp_path)]) == 1
    assert "找不到或无法解析" in capsys.readouterr().err


def test_logged_kl_and_target_entropy_are_reported_with_actual_coverage():
    rows = [
        {"iter": "1", "policy_loss": "2.0", "entropy": "2.0"},
        {"iter": "2", "policy_loss": "2.0", "entropy": "2.0", "target_entropy": "1.6", "policy_kl": "0.4"},
        {"iter": "3", "target_entropy": "nan", "policy_kl": "inf"},
    ]
    window = window_summary(rows, 25)[0]
    assert window["target_entropy"] == pytest.approx(1.6)
    assert window["policy_kl"] == pytest.approx(0.4)
    lines = []
    report_metrics(rows, window=25, recent=3, lines=lines)
    report_flags(rows, lines)
    report = "\n".join(lines)
    assert "真实 policy_kl 均值 0.40000（最近 3 行中 1 行有效）" in report
    assert "目标熵 target_entropy 均值 1.60000（最近 3 行中 1 行有效）" in report
    assert "不能直接判断棋力" in report


def test_legacy_prediction_entropy_never_becomes_target_entropy_or_kl():
    rows = [{"iter": "1", "policy_loss": "1.0", "entropy": "1.0"}]
    window = window_summary(rows, 25)[0]
    assert window["policy_kl"] is None
    assert window["target_entropy"] is None
    lines = []
    report_flags(rows, lines)
    report = "\n".join(lines)
    assert "真实 policy_kl 未记录，KL 未知" in report
    assert "目标熵 target_entropy 未记录（未知）" in report
    assert "不能说明目标已学完" in report
    assert "不能据此认定棋力停滞" in report
