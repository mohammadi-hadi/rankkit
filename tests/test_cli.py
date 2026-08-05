import json

import pytest

from rankkit import __version__
from rankkit.__main__ import main, parse_etas, parse_metrics

RUN_A = [
    {"qid": f"q{i}", "ranking": ["a", "b", "c", "d"], "relevance": {"a": 2, "c": 1}}
    for i in range(40)
]
RUN_B = [
    {"qid": f"q{i}", "ranking": ["c", "a", "b", "d"], "relevance": {"a": 2, "c": 1}}
    for i in range(40)
]
CLICKS = [
    {"qid": f"q{i}", "ranking": ["a", "b", "c", "d"], "clicks": ["a"] if i % 2 else ["c"]}
    for i in range(40)
]


@pytest.fixture
def files(tmp_path):
    def write(name, records):
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return str(path)

    return {
        "a": write("run_a.jsonl", RUN_A),
        "b": write("run_b.jsonl", RUN_B),
        "clicks": write("clicks.jsonl", CLICKS),
        "dir": tmp_path,
    }


def test_parse_metrics_accepts_a_list_and_the_all_shorthand():
    assert parse_metrics("ndcg,mrr") == ["ndcg", "mrr"]
    assert "err" in parse_metrics("all")


def test_parse_metrics_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown metric"):
        parse_metrics("ndcg,f1")


def test_parse_etas_sorts_and_validates():
    assert parse_etas("1,0,0.5") == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="at least two values"):
        parse_etas("1")
    with pytest.raises(ValueError, match="must not contain negative"):
        parse_etas("-1,1")


def test_eval_prints_a_metric_table(files, capsys):
    assert main(["eval", files["a"]]) == 0
    out = capsys.readouterr().out
    assert "ndcg@10" in out
    assert "recall@10" in out
    assert "95% CI" in out


def test_eval_json_is_machine_readable(files, capsys):
    assert main(["eval", files["a"], "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["queries"] == 40
    assert 0.0 <= payload["metrics"]["ndcg"]["value"] <= 1.0
    assert payload["metrics"]["ndcg"]["ci_lo"] <= payload["metrics"]["ndcg"]["ci_hi"]


def test_eval_writes_per_query_scores_in_abevals_shape(files, capsys):
    out_path = files["dir"] / "per_query.jsonl"
    assert main(["eval", files["a"], "--per-query", str(out_path)]) == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) == 40
    assert set(rows[0]) >= {"id", "score"}


def test_eval_honours_a_smaller_cutoff(files, capsys):
    main(["eval", files["a"], "--metric", "recall", "--k", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["recall"]["value"] == pytest.approx(0.5)


def test_compare_reports_a_paired_difference(files, capsys):
    assert main(["compare", files["a"], files["b"]]) == 0
    out = capsys.readouterr().out
    assert "B - A:" in out
    assert "sign-flip permutation" in out
    assert "verdict:" in out


def test_compare_takes_one_metric_at_a_time(files, capsys):
    assert main(["compare", files["a"], files["b"], "--metric", "ndcg,mrr"]) == 2
    assert "single --metric" in capsys.readouterr().err


def test_compare_json_carries_the_planning_number(files, capsys):
    assert main(["compare", files["a"], files["b"], "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == 0.01
    assert payload["n"] == 40
    assert "queries_needed" in payload


def test_bias_alone_reports_the_corrected_estimate(files, capsys):
    assert main(["bias", files["clicks"]]) == 0
    out = capsys.readouterr().out
    assert "counting clicks:" in out
    assert "position-corrected:" in out
    assert "effective sample size" in out


def test_bias_against_a_candidate_sweeps_eta(files, capsys):
    assert main(["bias", files["clicks"], files["b"]]) == 0
    out = capsys.readouterr().out
    assert "candidate" in out
    assert "eta sensitivity" in out


def test_bias_writes_a_chart(files, capsys):
    svg_path = files["dir"] / "bias.svg"
    assert main(["bias", files["clicks"], "--svg", str(svg_path)]) == 0
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")


def test_bias_json_includes_the_diagnostics(files, capsys):
    assert main(["bias", files["clicks"], files["b"], "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostics"]["ess"] > 0
    assert len(payload["clicks_by_rank"]) == 10
    assert payload["sensitivity"][0]["eta"] == 0.0


def test_eta_zero_leaves_the_naive_estimate_alone(files, capsys):
    main(["bias", files["clicks"], "--eta", "0", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["logged"]["naive"] == pytest.approx(payload["logged"]["ips"])


def test_a_missing_file_is_an_error_not_a_traceback(files, capsys):
    assert main(["eval", str(files["dir"] / "nope.jsonl")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_malformed_json_is_reported_with_its_line(files, capsys):
    bad = files["dir"] / "bad.jsonl"
    bad.write_text('{"qid": "q1", "ranking": ["a"]}\noops\n', encoding="utf-8")
    assert main(["eval", str(bad)]) == 2
    assert "line 2" in capsys.readouterr().err


def test_an_unknown_metric_exits_two(files, capsys):
    assert main(["eval", files["a"], "--metric", "f1"]) == 2
    assert "unknown metric" in capsys.readouterr().err


def test_a_zero_cutoff_exits_two(files, capsys):
    assert main(["eval", files["a"], "--k", "0"]) == 2
    assert "k must be at least 1" in capsys.readouterr().err


def test_a_file_with_no_labels_at_all_is_reported(files, capsys):
    unlabelled = files["dir"] / "unlabelled.jsonl"
    unlabelled.write_text('{"qid": "q1", "ranking": ["a", "b"]}\n', encoding="utf-8")
    assert main(["eval", str(unlabelled)]) == 2
    assert "no query has a relevant document" in capsys.readouterr().err


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"rankkit {__version__}"
