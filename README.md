# rankkit

Ranking evaluation with error bars, and click metrics that survive contact with
position bias.

Offline ranking numbers get reported as bare points: "NDCG@10 went from 0.826 to
0.862, ship it." Two things are usually missing. The first is an interval — on a
few hundred queries, differences smaller than that are routinely noise. The
second bites harder: if your labels are clicks rather than judgements, the metric
is measuring *what your current ranker put on top*, and it will quietly tell you
that every candidate is worse than production. Standard library only, no
dependencies.

**Position-bias explorer:** https://mohammadi.cv/rankkit/

## Install

```
pip install rankkit
```

Or from source: `git clone https://github.com/mohammadi-hadi/rankkit && cd rankkit && make install`.

## Is my new ranker actually better?

Point `rankkit compare` at two JSONL files that rank the same queries:

```
$ rankkit compare examples/run_a.jsonl examples/run_b.jsonl
ndcg@10 on 399 shared queries
A: 0.8260   B: 0.8617
B - A: +0.0357  [+0.0143, +0.0566]  (95% CI, 2000 bootstrap resamples)
p (sign-flip permutation): 0.0012
better on 217 queries, worse on 166, tied on 16
biggest movers: q032 +1.000, q024 -0.644, q047 -0.640
to detect +0.010 at this per-query spread you would need 3671 queries
verdict: significant at the chosen level
```

The comparison is **paired**: both systems are scored on the same queries, so
query difficulty cancels and far smaller differences become visible than two
independent samples would allow. Queries only one file ranks are dropped and
reported, never filled in with zeros.

That last line before the verdict is the one people find useful in planning
meetings. This run has a wide per-query spread, so a one-point NDCG change would
need thousands of queries to call — worth knowing *before* you commission the
labelling.

## Error bars for one run

```
$ rankkit eval examples/run_a.jsonl
400 queries, 12.0 documents each, 3.9 relevant per query, 1 skipped for having no relevant document

metric      value    95% CI
ndcg@10     0.8260   [0.8083, 0.8431]
mrr@10      0.8895   [0.8657, 0.9125]
map@10      0.7473   [0.7264, 0.7680]
recall@10   0.9611   [0.9510, 0.9708]
```

Intervals come from bootstrapping over queries, which is the unit that actually
varies. Queries with no relevant document are skipped rather than scored zero,
following trec_eval, and the count is printed so a silently shrinking
denominator cannot flatter a run.

### Handing the scores to abeval

`--per-query` writes one `{"id", "score"}` line per query, which is
[abeval](https://github.com/mohammadi-hadi/abeval)'s input format with no flags
needed:

```
$ rankkit eval examples/run_a.jsonl --metric ndcg --per-query a.jsonl
$ rankkit eval examples/run_b.jsonl --metric ndcg --per-query b.jsonl
$ abeval compare a.jsonl b.jsonl
A: 0.826   B: 0.8617   (n=399 paired items)
B - A: +0.03573  [+0.0143, +0.05671]  (95% CI)
p (sign-flip permutation): 0.0010   p (paired t): 0.0011
verdict: significant at the chosen level
```

Same verdict, reached by a separate implementation — the small differences are
resampling noise from different seeds and repetition counts.

## My only labels are clicks

This is where offline ranking evaluation usually goes wrong. A click means the
user *looked at* the document and liked it, so counting clicks measures
relevance multiplied by attention — and attention is decided by the ranker that
produced the log. Score a candidate on those clicks and it is rewarded only for
agreeing with production.

Here is the same candidate that just won by 3.6 points of NDCG on real labels,
judged from production's click log instead:

```
$ rankkit bias examples/clicks.jsonl examples/run_b.jsonl
400 queries, 425 clicks, eta = 1, weight = dcg

                        logged   candidate
counting clicks         0.8203      0.6962
position-corrected      1.3703      1.3895

counting clicks says the logged ranking wins by 0.1241
correcting for position says the candidate wins by 0.0191

diagnostics: effective sample size 230 of 425 clicks (54.1%), largest single weight 1.3% of the total
             2.38 relevant documents per query implied

eta sensitivity (candidate - logged, dcg):
  eta 0     -0.1241
  eta 0.5   -0.0827
  eta 1     +0.0191  <- assumed
  eta 1.5   +0.2834
  eta 2     +0.9990
the conclusion flips at eta = 0.94
```

Counting clicks rejects a candidate we know to be better. Weighting each click
by the inverse probability that its position was examined restores the right
verdict.

**And then the sweep talks you out of trusting it.** `eta` is the strength of
the position bias, and it is *assumed*, not measured — so the honest output is
not one number but the range of conclusions the assumption supports. Here the
winner changes at `eta = 0.94`, close enough to the assumed 1.0 that this
decision does not survive a small error in the assumption. That is a finding,
not a failure: run an interleaving test before shipping.

The diagnostics line is the other guard. 425 clicks that behave like 230 means
the estimate leans on a minority of deep clicks; `--clip` floors the propensity
to stop a handful of them dominating, at the cost of some bias:

```
$ rankkit bias examples/clicks.jsonl examples/run_b.jsonl --clip 0.1
...
  eta 1.5   +0.1006
  eta 2     +0.1537
             6 propensities hit the 0.1 clip, which adds bias
```

`--svg` writes the click-through curve with the assumed examination model drawn
over it ([examples/bias.svg](examples/bias.svg)).

### Where eta comes from

Not from your click log. A document at rank 1 collects clicks both because it
was seen more *and* because it is better, and no amount of curve fitting on
clicks-by-rank can separate those. rankkit therefore refuses to estimate `eta`
for you. Get it from a swap or randomisation experiment, take `1.0` as a
starting assumption, and read the sweep.

## Data format

JSONL, one query per line. Extra fields are ignored.

```json
{"qid": "q1", "ranking": ["d3", "d1", "d7"], "relevance": {"d1": 2, "d7": 1}}
{"qid": "q2", "ranking": ["d4", "d9"], "clicks": ["d9"]}
```

`ranking` is the order the documents were placed in, best first; in a click log
it is specifically the order **shown to the user**. `relevance` accepts graded
labels or a flat list of relevant ids. `--qid-key`, `--rank-key`, `--rel-key`
and `--click-key` rename the fields. The files in `examples/` come from
`examples/make_fixtures.py`, which is seeded and reproducible.

## Python API

Everything the CLI does is a plain function:

```python
from rankkit import estimate, extract_runs, paired_compare, read_jsonl, scores

runs = extract_runs(read_jsonl("run_a.jsonl"))
per_query = scores(runs, "ndcg", k=10)  # {qid: value}, undefined queries skipped

result = paired_compare(per_query_a, per_query_b, metric="ndcg@10")
result.delta, result.ci_lo, result.ci_hi, result.p_permutation
result.queries_needed(0.01)  # planning, not post-hoc power

est = estimate(click_queries, candidate_policy, eta=1.0, weight="dcg")
est.naive, est.ips, est.snips
```

## What's inside

| Question | Method |
|---|---|
| How good is this ranking? | NDCG (exponential or linear gain), MRR, MAP, recall@k, precision@k, hit rate, ERR |
| How sure am I? | Percentile bootstrap over queries |
| Is B better than A? | Sign-flip permutation test on paired per-query differences, with a bootstrap interval |
| How many queries do I need? | Normal-approximation planning from the observed paired spread |
| My labels are clicks | Inverse propensity scoring and its self-normalised form, with effective-sample-size and weight-concentration diagnostics |
| How much does that depend on eta? | Sweep across eta and report where the winner changes |

The position-bias correction follows Joachims, Swaminathan and Schnabel,
["Unbiased Learning-to-Rank with Biased Feedback"](https://arxiv.org/abs/1608.04468)
(2017). ERR follows Chapelle et al. (2009). The NDCG, average-precision and
precision conventions match trec_eval, and the ones implementations disagree
about are written down at the top of `rankkit/metrics.py`.

## Honest limitations

- **`eta` is an assumption.** Everything downstream of it inherits that. The
  sweep exists so you can see how much, and a crossover near your assumed value
  means you should be running an online test instead.
- **Inverse propensity scoring needs the candidate to rank the same documents.**
  A document your candidate never retrieved earns no credit, which is correct,
  but a candidate scoring a different pool is not being compared fairly.
- **Clipping changes what is being estimated.** It trades variance for bias; the
  count of clipped propensities is printed so the trade stays visible.
- **`snips` is not a competing estimate of the same thing.** It is a mean per
  relevant document rather than per query, related by the exact identity
  `snips * relevant_per_query == ips`.
- **The planning formula uses the normal approximation** and the paired spread
  you measured, so it is only as good as that estimate on small query sets.
- **Multiple comparisons are on you.** Compare ten rankers, ship the best
  p-value, and it is inflated.
- **This is not a learning-to-rank library.** It evaluates rankings; it does not
  train them.

## Sponsoring

rankkit is MIT-licensed and dependency-free, and it stays that way. Sponsoring
funds the roadmap below and the maintenance time to keep the statistics
trustworthy: [GitHub Sponsors](https://github.com/sponsors/mohammadi-hadi).

## Roadmap

- Team-draft interleaving analysis: credit assignment and a per-query preference
  test from an interleaved log.
- Propensity estimation from a swap experiment, so `eta` can be measured rather
  than assumed.
- Doubly-robust estimation, which uses a relevance model to cut the variance
  that inverse propensity scoring alone leaves behind.

## Related projects

- [abeval](https://github.com/mohammadi-hadi/abeval) — A/B-test statistics for
  LLM evals. The paired machinery here is the same, applied to ranking metrics;
  `rankkit eval --per-query` writes scores in abeval's input format.
- [calikit](https://github.com/mohammadi-hadi/calikit) — whether the
  probabilities behind your scores are calibrated at all.
- [judgekit](https://github.com/mohammadi-hadi/judgekit) — audit an LLM judge
  before you let it label a ranking.
- [judgepanel](https://github.com/mohammadi-hadi/judgepanel) — estimate judge
  accuracy without gold labels.
- [raterkit](https://github.com/mohammadi-hadi/raterkit) — audit the human
  annotations underneath everything above.

## Citing

Releases are archived on Zenodo. Cite the concept DOI
[10.5281/zenodo.21811914](https://doi.org/10.5281/zenodo.21811914), which always
resolves to the latest version; structured metadata is in
[CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
