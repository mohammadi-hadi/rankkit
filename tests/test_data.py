import json

import pytest

from rankkit.data import align, extract_clicks, extract_runs, read_jsonl


def write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"qid": "q1"}\n\n{"qid": "q2"}\n', encoding="utf-8")
    assert len(read_jsonl(path)) == 2


def test_read_jsonl_reports_the_bad_line(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"qid": "q1"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)


def test_read_jsonl_rejects_an_empty_file(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        read_jsonl(path)


def test_extract_runs_reads_graded_relevance():
    runs = extract_runs([{"qid": "q1", "ranking": ["d1", "d2"], "relevance": {"d1": 2}}])
    assert runs[0].qid == "q1"
    assert runs[0].ranking == ("d1", "d2")
    assert runs[0].gain("d1") == 2.0
    assert runs[0].gain("d2") == 0.0
    assert runs[0].n_relevant == 1


def test_relevance_may_be_a_flat_list_of_ids():
    runs = extract_runs([{"qid": "q1", "ranking": ["d1", "d2"], "relevance": ["d2"]}])
    assert runs[0].gain("d2") == 1.0
    assert runs[0].gain("d1") == 0.0


def test_numeric_ids_become_strings():
    runs = extract_runs([{"qid": 7, "ranking": [1, 2], "relevance": {"1": 1}}])
    assert runs[0].qid == "7"
    assert runs[0].ranking == ("1", "2")


def test_boolean_ids_are_rejected():
    with pytest.raises(ValueError, match="must be a string or a number"):
        extract_runs([{"qid": True, "ranking": ["d1"]}])


def test_duplicate_query_ids_are_rejected():
    with pytest.raises(ValueError, match="more than once"):
        extract_runs([{"qid": "q1", "ranking": ["d1"]}, {"qid": "q1", "ranking": ["d2"]}])


def test_a_document_cannot_appear_twice_in_one_ranking():
    with pytest.raises(ValueError, match="same document twice"):
        extract_runs([{"qid": "q1", "ranking": ["d1", "d1"]}])


def test_empty_rankings_are_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        extract_runs([{"qid": "q1", "ranking": []}])


def test_missing_ranking_is_reported_by_key():
    with pytest.raises(ValueError, match="missing 'ranking'"):
        extract_runs([{"qid": "q1"}])


def test_negative_grades_are_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        extract_runs([{"qid": "q1", "ranking": ["d1"], "relevance": {"d1": -1}}])


def test_non_numeric_grades_are_rejected():
    with pytest.raises(ValueError, match="must be a number"):
        extract_runs([{"qid": "q1", "ranking": ["d1"], "relevance": {"d1": "high"}}])


def test_custom_key_names():
    runs = extract_runs(
        [{"query": "q1", "docs": ["d1"], "labels": {"d1": 3}}],
        qid_key="query",
        rank_key="docs",
        rel_key="labels",
    )
    assert runs[0].gain("d1") == 3.0


def test_extract_clicks_records_shown_rank():
    queries = extract_clicks([{"qid": "q1", "ranking": ["d1", "d2", "d3"], "clicks": ["d3"]}])
    assert queries[0].clicks == ("d3",)
    assert queries[0].shown_rank("d3") == 3


def test_a_click_outside_the_shown_ranking_is_an_error():
    with pytest.raises(ValueError, match="not in the shown ranking"):
        extract_clicks([{"qid": "q1", "ranking": ["d1"], "clicks": ["d9"]}])


def test_queries_with_no_clicks_are_kept():
    queries = extract_clicks([{"qid": "q1", "ranking": ["d1"]}])
    assert queries[0].clicks == ()


def test_align_keeps_shared_queries_and_reports_the_rest():
    left = extract_runs([{"qid": q, "ranking": ["d1"]} for q in ("q1", "q2", "q3")])
    right = extract_runs([{"qid": q, "ranking": ["d1"]} for q in ("q2", "q3", "q4")])
    a, b, dropped = align(left, right)
    assert [r.qid for r in a] == ["q2", "q3"]
    assert [r.qid for r in b] == ["q2", "q3"]
    assert dropped == ["q1", "q4"]


def test_align_refuses_disjoint_files():
    left = extract_runs([{"qid": "q1", "ranking": ["d1"]}])
    right = extract_runs([{"qid": "q2", "ranking": ["d1"]}])
    with pytest.raises(ValueError, match="share no query ids"):
        align(left, right)
