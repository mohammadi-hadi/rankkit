"""Generate the example files, reproducibly.

The story the fixtures tell: `run_a` is the ranker in production, `run_b` is a
candidate that orders the same documents with less noise, and `clicks` is the
log production actually collected. Clicks are simulated from a *known* position
bias, so `rankkit bias` is correcting for exactly the bias that was injected.

That makes the demo a self-consistency check, not evidence that inverse
propensity scoring works on a real log — on real traffic nobody hands you eta,
which is what the sensitivity sweep is for.

    python examples/make_fixtures.py
"""

import json
import random
from pathlib import Path

SEED = 42
N_QUERIES = 400
N_DOCS = 12
ETA = 1.0

# Most documents in a candidate pool are not relevant.
GRADE_WEIGHTS = ((0, 0.68), (1, 0.16), (2, 0.10), (3, 0.06))

# Probability a user clicks a document *given that they looked at it*.
# Grade 0 is not quite zero: people do misclick on bad results.
ATTRACTION = {0: 0.05, 1: 0.35, 2: 0.65, 3: 0.88}

# The production ranker is noisier about true relevance than the candidate is.
# The gap is deliberately modest: a few points of NDCG is what a real ranking
# change looks like, and an implausibly large win would make every downstream
# number in the README too easy.
PRODUCTION_NOISE = 1.2
CANDIDATE_NOISE = 1.05

HERE = Path(__file__).parent


def sample_grade(rng: random.Random) -> int:
    roll = rng.random()
    cumulative = 0.0
    for grade, weight in GRADE_WEIGHTS:
        cumulative += weight
        if roll < cumulative:
            return grade
    return 0


def main() -> None:
    rng = random.Random(SEED)
    production, candidate, log = [], [], []

    for q in range(N_QUERIES):
        qid = f"q{q:03d}"
        docs = [f"d{q:03d}-{j:02d}" for j in range(N_DOCS)]
        grades = {doc: sample_grade(rng) for doc in docs}

        by_production = sorted(docs, key=lambda d: grades[d] + rng.gauss(0, PRODUCTION_NOISE))
        by_production.reverse()
        by_candidate = sorted(docs, key=lambda d: grades[d] + rng.gauss(0, CANDIDATE_NOISE))
        by_candidate.reverse()

        labels = {doc: grade for doc, grade in grades.items() if grade > 0}
        production.append({"qid": qid, "ranking": by_production, "relevance": labels})
        candidate.append({"qid": qid, "ranking": by_candidate, "relevance": labels})

        clicks = []
        for rank, doc in enumerate(by_production, start=1):
            examined = rng.random() < (1.0 / rank) ** ETA
            if examined and rng.random() < ATTRACTION[grades[doc]]:
                clicks.append(doc)
        log.append({"qid": qid, "ranking": by_production, "clicks": clicks})

    for name, records in (("run_a", production), ("run_b", candidate), ("clicks", log)):
        path = HERE / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(rec) + "\n" for rec in records)
        print(f"wrote {path.name}: {len(records)} queries")


if __name__ == "__main__":
    main()
