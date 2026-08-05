"""A self-contained SVG of the click-through curve against the assumed model.

The bars are what actually happened: the share of impressions at each rank that
were clicked. The tick on each bar is where pure examination would put it, i.e.
rank 1's observed share scaled by `(1/rank) ** eta`.

Reading the gap is the whole point. Bars falling away faster than the ticks mean
attention drops off harder than the assumed eta; bars sitting past their ticks
mean something other than position is holding those clicks up. Neither is a
propensity estimate — the curve confounds examination with relevance, which is
why eta has to come from a swap experiment rather than from this picture.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

W = 640
ROW = 26
BAR = 15
PAD_TOP = 78
PAD_BOTTOM = 44
GUTTER = 52
RIGHT = 62
RADIUS = 4


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bar_path(x0: float, x1: float, y: float, height: float, radius: float) -> str:
    """A bar rounded on the far end only, so it stays anchored to the baseline."""
    if x1 - x0 <= radius:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 0.6):.1f}" height="{height:.1f}"/>'
    return (
        f'<path d="M{x0:.1f},{y:.1f} H{x1 - radius:.1f} '
        f"A{radius},{radius} 0 0 1 {x1:.1f},{y + radius:.1f} "
        f"V{y + height - radius:.1f} "
        f"A{radius},{radius} 0 0 1 {x1 - radius:.1f},{y + height:.1f} "
        f'H{x0:.1f} Z"/>'
    )


def bias_svg(
    shares: Sequence[float],
    eta: float = 1.0,
    title: str = "Clicks by shown rank",
) -> str:
    """Render the click-through curve with the assumed examination model on top."""
    if not shares:
        raise ValueError("no ranks to draw")
    modelled = [shares[0] * (1.0 / rank) ** eta for rank in range(1, len(shares) + 1)]
    top = max(max(shares), max(modelled))
    if top <= 0:
        raise ValueError("every rank has a click share of zero, nothing to draw")
    plot = W - GUTTER - RIGHT
    height = PAD_TOP + ROW * len(shares) + PAD_BOTTOM

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
            f'width="{W}" height="{height}" font-family="ui-sans-serif, system-ui, sans-serif">'
        ),
        (
            "<style>"
            ".bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.dim{fill:#52514e}.mute{fill:#898781}"
            ".bar{fill:#2a78d6}.tick{stroke:#0b0b0b;stroke-width:2;opacity:.55}"
            ".grid{stroke:#e1e0d9;stroke-width:1}"
            ".num{font-variant-numeric:tabular-nums}"
            "@media (prefers-color-scheme:dark){"
            ".bg{fill:#1a1a19}.ink{fill:#ffffff}.dim{fill:#c3c2b7}"
            ".bar{fill:#3987e5}.tick{stroke:#ffffff}.grid{stroke:#2c2c2a}}"
            "</style>"
        ),
        f'<rect class="bg" width="{W}" height="{height}"/>',
        (
            f'<text class="ink" x="{GUTTER}" y="30" font-size="15" font-weight="600">'
            f"{_escape(title)}</text>"
        ),
        (
            f'<text class="mute" x="{GUTTER}" y="50" font-size="11.5">'
            f"Bars: observed click-through. Ticks: examination alone at eta = {eta:g}, "
            f"anchored to rank 1.</text>"
        ),
        (
            f'<text class="mute" x="{GUTTER}" y="66" font-size="11.5">'
            "Descriptive only - this curve mixes examination with relevance.</text>"
        ),
    ]

    for i, share in enumerate(shares):
        y = PAD_TOP + i * ROW
        parts.append(
            f'<line class="grid" x1="{GUTTER}" y1="{y + ROW - 4}" '
            f'x2="{W - RIGHT + 4}" y2="{y + ROW - 4}"/>'
        )
        parts.append(
            f'<text class="dim num" x="{GUTTER - 10}" y="{y + BAR - 2}" '
            f'font-size="11.5" text-anchor="end">{i + 1}</text>'
        )
        x1 = GUTTER + plot * share / top
        parts.append(f'<g class="bar">{_bar_path(GUTTER, x1, y, BAR, RADIUS)}</g>')
        mx = GUTTER + plot * modelled[i] / top
        parts.append(
            f'<line class="tick" x1="{mx:.1f}" y1="{y - 1.5}" x2="{mx:.1f}" y2="{y + BAR + 1.5}"/>'
        )
        parts.append(
            f'<text class="dim num" x="{W - RIGHT + 8}" y="{y + BAR - 2}" '
            f'font-size="11.5">{share * 100:.1f}%</text>'
        )

    baseline = PAD_TOP + ROW * len(shares)
    parts.append(
        f'<text class="mute" x="{GUTTER}" y="{baseline + 22}" font-size="11.5">'
        f"Rank (position shown to the user), 1 at the top</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(path: str | Path, markup: str) -> None:
    Path(path).write_text(markup, encoding="utf-8")
