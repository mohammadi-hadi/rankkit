import pytest

from rankkit.svg import bias_svg, write_svg

SHARES = [0.42, 0.21, 0.15, 0.09, 0.05]


def test_renders_a_well_formed_document():
    markup = bias_svg(SHARES)
    assert markup.startswith("<svg xmlns=")
    assert markup.rstrip().endswith("</svg>")


def test_one_bar_and_one_model_tick_per_rank():
    markup = bias_svg(SHARES)
    assert markup.count('<g class="bar">') == len(SHARES)
    assert markup.count('class="tick"') == len(SHARES)


def test_labels_every_rank_with_its_click_share():
    markup = bias_svg(SHARES)
    assert "42.0%" in markup
    assert "5.0%" in markup


def test_the_assumed_eta_is_written_on_the_chart():
    assert "eta = 1.5" in bias_svg(SHARES, eta=1.5)


def test_the_caption_says_the_curve_is_confounded():
    assert "mixes examination with relevance" in bias_svg(SHARES)


def test_ships_a_dark_mode():
    assert "prefers-color-scheme:dark" in bias_svg(SHARES)


def test_titles_are_escaped():
    assert "Search &amp; browse" in bias_svg(SHARES, title="Search & browse")


def test_a_tiny_bar_still_renders():
    markup = bias_svg([0.5, 0.0001])
    assert markup.count("<rect") >= 2


def test_empty_and_all_zero_inputs_are_rejected():
    with pytest.raises(ValueError, match="no ranks"):
        bias_svg([])
    with pytest.raises(ValueError, match="click share of zero"):
        bias_svg([0.0, 0.0])


def test_write_svg_round_trips(tmp_path):
    path = tmp_path / "bias.svg"
    write_svg(path, bias_svg(SHARES))
    assert path.read_text(encoding="utf-8").startswith("<svg")


def test_a_half_percent_tie_rounds_up_to_match_the_explorer():
    # 5 clicks in 400 impressions is exactly 1.25%, where Python's default
    # round-half-even would print 1.2 and the page's JS prints 1.3.
    markup = bias_svg([0.5, 5 / 400])
    assert "1.3%" in markup
    assert "1.2%" not in markup
