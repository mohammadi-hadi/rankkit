"""Command line interface: `rankkit eval | compare | bias`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from . import bias as bias_mod
from .compare import bootstrap_ci, paired_compare
from .data import align, extract_clicks, extract_runs, read_jsonl
from .metrics import METRICS, max_grade_of, scores
from .svg import bias_svg, write_svg

DEFAULT_METRICS = "ndcg,mrr,map,recall"


def parse_metrics(text: str) -> list[str]:
    if text.strip() == "all":
        return list(METRICS)
    wanted = [m.strip() for m in text.split(",") if m.strip()]
    if not wanted:
        raise ValueError("no metrics requested")
    for metric in wanted:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}, expected one of {', '.join(METRICS)}")
    return wanted


def parse_etas(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--etas must be a comma separated list of numbers: {exc}") from exc
    if len(values) < 2:
        raise ValueError("--etas needs at least two values to sweep between")
    if any(v < 0 for v in values):
        raise ValueError("--etas must not contain negative values")
    return sorted(values)


def load_runs(path: str, args: argparse.Namespace):
    return extract_runs(
        read_jsonl(path),
        qid_key=args.qid_key,
        rank_key=args.rank_key,
        rel_key=args.rel_key,
    )


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_per_query(path: str, rows: list[dict[str, Any]]) -> None:
    """Write scores in the shape abeval reads, one query per line."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in rows)


def _dropped_note(dropped: list[str]) -> str:
    if not dropped:
        return ""
    shown = ", ".join(dropped[:3])
    more = f" and {len(dropped) - 3} more" if len(dropped) > 3 else ""
    return f"  (dropped {len(dropped)} unmatched: {shown}{more})"


def run_eval(args: argparse.Namespace) -> int:
    runs = load_runs(args.run, args)
    wanted = parse_metrics(args.metric)
    top = max_grade_of(runs)

    table: dict[str, dict[str, float]] = {}
    per_query: dict[str, dict[str, float]] = {}
    for metric in wanted:
        values = scores(runs, metric, k=args.k, gain=args.gain, max_grade=top)
        per_query[metric] = values
        if not values:
            raise ValueError("no query has a relevant document, so nothing can be scored")
        listed = list(values.values())
        lo, hi = bootstrap_ci(listed, level=args.level, reps=args.reps, seed=args.seed)
        table[metric] = {
            "value": sum(listed) / len(listed),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(listed),
        }

    scored = max(len(v) for v in per_query.values())
    skipped = len(runs) - scored
    depth = sum(len(r.ranking) for r in runs) / len(runs)
    relevant = sum(r.n_relevant for r in runs) / len(runs)

    if args.per_query:
        primary = wanted[0]
        rows = [
            {
                "id": qid,
                "score": value,
                **{m: per_query[m][qid] for m in wanted if qid in per_query[m]},
            }
            for qid, value in per_query[primary].items()
        ]
        write_per_query(args.per_query, rows)

    if args.json:
        emit(
            {
                "queries": len(runs),
                "scored": scored,
                "skipped": skipped,
                "k": args.k,
                "gain": args.gain,
                "level": args.level,
                "metrics": table,
            }
        )
        return 0

    note = f", {skipped} skipped for having no relevant document" if skipped else ""
    print(
        f"{len(runs)} queries, {depth:.1f} documents each, {relevant:.1f} relevant per query{note}"
    )
    print()
    width = max(len(m) for m in wanted) + len(f"@{args.k}")
    print(f"{'metric'.ljust(width)}   value    {int(args.level * 100)}% CI")
    for metric in wanted:
        row = table[metric]
        label = f"{metric}@{args.k}".ljust(width)
        print(f"{label}   {row['value']:.4f}   [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]")
    if args.per_query:
        print(f"\nper-query scores written to {args.per_query}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    wanted = parse_metrics(args.metric)
    if len(wanted) != 1:
        raise ValueError("compare takes a single --metric; run it again for another one")
    metric = wanted[0]

    runs_a = load_runs(args.run_a, args)
    runs_b = load_runs(args.run_b, args)
    aligned_a, aligned_b, dropped = align(runs_a, runs_b)
    top = max(max_grade_of(aligned_a), max_grade_of(aligned_b))
    a = scores(aligned_a, metric, k=args.k, gain=args.gain, max_grade=top)
    b = scores(aligned_b, metric, k=args.k, gain=args.gain, max_grade=top)

    result = paired_compare(
        a,
        b,
        metric=f"{metric}@{args.k}",
        level=args.level,
        reps=args.reps,
        boot_reps=args.boot_reps,
        seed=args.seed,
    )
    needed = result.queries_needed(args.target, level=args.level)

    if args.json:
        payload = dict(vars(result))
        payload["dropped"] = dropped + result.dropped
        payload["significant"] = result.significant
        payload["target"] = args.target
        payload["queries_needed"] = needed
        emit(payload)
        return 0

    print(f"{result.metric} on {result.n} shared queries{_dropped_note(dropped + result.dropped)}")
    print(f"A: {result.mean_a:.4f}   B: {result.mean_b:.4f}")
    print(
        f"B - A: {result.delta:+.4f}  [{result.ci_lo:+.4f}, {result.ci_hi:+.4f}]  "
        f"({int(args.level * 100)}% CI, {args.boot_reps} bootstrap resamples)"
    )
    print(f"p (sign-flip permutation): {result.p_permutation:.4f}")
    print(f"better on {result.wins} queries, worse on {result.losses}, tied on {result.ties}")
    if result.movers:
        movers = ", ".join(f"{qid} {delta:+.3f}" for qid, delta in result.movers)
        print(f"biggest movers: {movers}")
    if needed is not None:
        print(
            f"to detect {args.target:+.3f} at this per-query spread you would need {needed} queries"
        )
    print(
        "verdict: significant at the chosen level"
        if result.significant
        else "verdict: not distinguishable from noise at the chosen level"
    )
    if args.per_query:
        rows = [
            {"id": qid, "score": b[qid], "score_a": a[qid], "delta": b[qid] - a[qid]}
            for qid in b
            if qid in a
        ]
        write_per_query(args.per_query, rows)
        print(f"per-query scores written to {args.per_query}")
    return 0


def run_bias(args: argparse.Namespace) -> int:
    queries = extract_clicks(
        read_jsonl(args.log),
        qid_key=args.qid_key,
        rank_key=args.rank_key,
        click_key=args.click_key,
    )
    etas = parse_etas(args.etas)
    dropped: list[str] = []
    candidate = None

    if args.candidate:
        cand_runs = load_runs(args.candidate, args)
        queries, cand_runs, dropped = align(queries, cand_runs)
        candidate = bias_mod.policy_from({r.qid: r.ranking for r in cand_runs})

    logged = bias_mod.estimate(
        queries, bias_mod.logged_policy, eta=args.eta, weight=args.weight, k=args.k, clip=args.clip
    )
    diag = bias_mod.diagnose(queries, eta=args.eta, clip=args.clip)
    curve = bias_mod.clicks_by_rank(queries, max_rank=args.k)

    rows: list[bias_mod.SensitivityRow] = []
    flip = None
    cand = None
    if candidate is not None:
        cand = bias_mod.estimate(
            queries, candidate, eta=args.eta, weight=args.weight, k=args.k, clip=args.clip
        )
        rows = bias_mod.sensitivity(
            queries, candidate, etas=etas, weight=args.weight, k=args.k, clip=args.clip
        )

        def delta_at(eta: float) -> float:
            kw = {"weight": args.weight, "k": args.k, "clip": args.clip}
            here = bias_mod.estimate(queries, candidate, eta=eta, **kw)
            there = bias_mod.estimate(queries, bias_mod.logged_policy, eta=eta, **kw)
            return here.ips - there.ips

        flip = bias_mod.crossover(rows, delta_at)

    if args.svg:
        write_svg(args.svg, bias_svg(curve, eta=args.eta))

    if args.json:
        payload: dict[str, Any] = {
            "queries": diag.n_queries,
            "clicks": diag.n_clicks,
            "eta": args.eta,
            "weight": args.weight,
            "k": args.k,
            "clip": args.clip,
            "dropped": dropped,
            "clicks_by_rank": curve,
            "logged": {"naive": logged.naive, "ips": logged.ips, "snips": logged.snips},
            "diagnostics": {
                "ess": diag.ess,
                "ess_share": diag.ess_share,
                "max_weight_share": diag.max_weight_share,
                "clipped": diag.clipped,
                "relevant_per_query": diag.relevant_per_query,
            },
        }
        if cand is not None:
            payload["candidate"] = {"naive": cand.naive, "ips": cand.ips, "snips": cand.snips}
            payload["sensitivity"] = [
                {"eta": r.eta, "logged": r.baseline, "candidate": r.candidate, "delta": r.delta}
                for r in rows
            ]
            payload["crossover_eta"] = flip
        emit(payload)
        return 0

    print(
        f"{diag.n_queries} queries, {diag.n_clicks} clicks, "
        f"eta = {args.eta:g}, weight = {args.weight}{_dropped_note(dropped)}"
    )
    print()
    if cand is None:
        print(f"counting clicks:        {logged.naive:.4f}")
        print(f"position-corrected:     {logged.ips:.4f}")
    else:
        print(f"{'':<20}{'logged':>10}{'candidate':>12}")
        print(f"{'counting clicks':<20}{logged.naive:>10.4f}{cand.naive:>12.4f}")
        print(f"{'position-corrected':<20}{logged.ips:>10.4f}{cand.ips:>12.4f}")
        print()
        naive_delta = cand.naive - logged.naive
        ips_delta = cand.ips - logged.ips
        naive_winner = "candidate" if naive_delta > 0 else "logged ranking"
        ips_winner = "candidate" if ips_delta > 0 else "logged ranking"
        print(f"counting clicks says the {naive_winner} wins by {abs(naive_delta):.4f}")
        print(f"correcting for position says the {ips_winner} wins by {abs(ips_delta):.4f}")
    print()
    print(
        f"diagnostics: effective sample size {diag.ess:.0f} of {diag.n_clicks} clicks "
        f"({diag.ess_share:.1%}), largest single weight {diag.max_weight_share:.1%} of the total"
    )
    print(f"             {diag.relevant_per_query:.2f} relevant documents per query implied")
    if diag.clipped:
        print(
            f"             {diag.clipped} propensities hit the {args.clip:g} clip, which adds bias"
        )
    elif diag.max_weight_share > 0.05:
        print("             one click carries over 5% of the weight; consider --clip")

    if rows:
        print()
        print(f"eta sensitivity (candidate - logged, {args.weight}):")
        for row in rows:
            marker = "  <- assumed" if abs(row.eta - args.eta) < 1e-9 else ""
            print(f"  eta {row.eta:<5g} {row.delta:+.4f}{marker}")
        if flip is None:
            leader = "candidate" if rows[0].delta > 0 else "logged ranking"
            print(f"the {leader} leads across the whole sweep")
        else:
            print(f"the conclusion flips at eta = {flip:.2f}")
    if args.svg:
        print(f"\nchart written to {args.svg}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rankkit",
        description="Ranking evaluation with error bars and position-bias correction.",
    )
    parser.add_argument("--version", action="version", version=f"rankkit {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def shared(sub: argparse.ArgumentParser, *, clicks: bool = False) -> None:
        sub.add_argument("--qid-key", default="qid", help="field holding the query id")
        sub.add_argument("--rank-key", default="ranking", help="field holding the ranked ids")
        sub.add_argument("--rel-key", default="relevance", help="field holding the labels")
        if clicks:
            sub.add_argument("--click-key", default="clicks", help="field holding the clicked ids")
        sub.add_argument("--k", type=int, default=10, help="rank cutoff (default 10)")
        sub.add_argument("--json", action="store_true", help="machine-readable output")

    ev = subparsers.add_parser("eval", help="score one run, with confidence intervals")
    ev.add_argument("run", help="JSONL of rankings with relevance labels")
    ev.add_argument(
        "--metric", default=DEFAULT_METRICS, help=f"default {DEFAULT_METRICS}, or 'all'"
    )
    ev.add_argument("--gain", default="exp", choices=("exp", "linear"), help="NDCG gain function")
    ev.add_argument("--level", type=float, default=0.95, help="confidence level")
    ev.add_argument("--reps", type=int, default=2000, help="bootstrap resamples")
    ev.add_argument("--seed", type=int, default=0, help="resampling seed")
    ev.add_argument("--per-query", help="write per-query scores here, in abeval's input format")
    shared(ev)
    ev.set_defaults(handler=run_eval)

    cmp_ = subparsers.add_parser("compare", help="test whether one run beats another")
    cmp_.add_argument("run_a", help="baseline run")
    cmp_.add_argument("run_b", help="candidate run")
    cmp_.add_argument("--metric", default="ndcg", help="single metric to compare (default ndcg)")
    cmp_.add_argument("--gain", default="exp", choices=("exp", "linear"))
    cmp_.add_argument("--level", type=float, default=0.95, help="confidence level")
    cmp_.add_argument("--reps", type=int, default=10000, help="permutations")
    cmp_.add_argument("--boot-reps", type=int, default=2000, help="bootstrap resamples")
    cmp_.add_argument("--seed", type=int, default=0, help="resampling seed")
    cmp_.add_argument("--target", type=float, default=0.01, help="difference to plan for")
    cmp_.add_argument("--per-query", help="write per-query scores and deltas here")
    shared(cmp_)
    cmp_.set_defaults(handler=run_compare)

    bs = subparsers.add_parser("bias", help="correct click metrics for position bias")
    bs.add_argument("log", help="JSONL of shown rankings with clicks")
    bs.add_argument("candidate", nargs="?", help="optional JSONL of a candidate ranking")
    bs.add_argument("--eta", type=float, default=1.0, help="position-bias strength (default 1.0)")
    bs.add_argument("--weight", default="dcg", choices=("dcg", "recall", "rr"))
    bs.add_argument("--clip", type=float, default=0.0, help="floor on the propensity")
    bs.add_argument("--etas", default="0,0.5,1,1.5,2", help="etas to sweep")
    bs.add_argument("--svg", help="write the click-by-rank chart here")
    shared(bs, clicks=True)
    bs.set_defaults(handler=run_bias)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
