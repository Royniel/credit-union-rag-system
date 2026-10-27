# Credit Union Knowledge System

A question-answering system over five snapshots of a credit union's website, crawled between May and September 2026. It answers member questions with the value that is true now (or on a given date), cites its sources, lists the places where the site contradicts itself, and says "I don't know" when the site does not say.

The solution lives in [`my_submission/`](my_submission/). The design rationale is in [`my_submission/NOTES.md`](my_submission/NOTES.md).

## Results

Scored with the provided harness on the public practice set:

| Pillar | Score |
|---|---|
| Accuracy | 0.89 (0 stale-value hits, 0 errors) |
| Consistency | 1.0 (reworded and repeated questions agree) |
| Speed | 1.0 (p95 latency about 0.02 s per answer) |
| Ingest | about 5 s for all five snapshots |

On the harness self-test site, which includes a conflicts answer key, the system scores 1.0 overall, including conflict detection and resolution.

## How it works

The system is fully local: Python standard library only, no API keys, no network calls.

1. **Versioned index.** Every page is split into chunks: one per table row (with its caption, column names and "Rates are effective as of" date), one per `Label: value` fact or fee line, and short prose passages. Each distinct chunk is stored once with a bitmask of the snapshots that contained it, so one BM25 index serves both the latest view and any "as of" view.
2. **Source authority.** URLs are classified as official rate pages, core info pages, promos, news, blogs, documents or calculators. Rates are never quoted from calculators, blogs or news, which is where the stale numbers live.
3. **What is current.** The newest snapshot (or the latest one on or before `as_of`) defines the view. Among pages that disagree, the later effective date and the higher authority win. Counts that drift over time come from the most recently dated statement, and offers the page itself marks as ended are ignored.
4. **Qualifiers.** Jumbo vs standard, business vs personal, refinance vs purchase, intro vs ongoing rate, and term, balance and model-year tiers are parsed from the question and matched against each row. A row for a different tier is never offered as the answer.
5. **Honesty.** The answer is "I don't know" (`confident=False`, no numbers) when a named entity is not on the site, when no rate evidence exists for the product asked about, when the best evidence covers a different term or tier, or when too little of the question is covered.
6. **Conflicts.** Rate facts are tracked across pages and snapshots. Groups whose values disagree are reported with every competing value, the winner and the reason. A few prose claims (branch count, Forbes recognition, membership, assets, employees, routing number) are tracked the same way.

### Why rules and retrieval instead of an LLM

The scoring zeroes any answer that repeats a stale or wrong-tier value, rewards consistency and needs answers in under 2 seconds. Deterministic retrieval with explicit rules for authority, dates and tiers controls exactly which numbers appear in an answer, gives the same answer every time, and answers in milliseconds. The same rules also produce the conflict resolutions, so `answer()` and `conflicts()` never disagree.

## Running it

Requires Python 3.9 or newer. There is nothing to install.

The website snapshots are not included in this repository, because the data may not be redistributed. Copy `snapshots_zip/` from the participant kit into the repository root first, then:

```bash
python harness/run_eval.py --submission my_submission --eval eval/public_eval.jsonl
```

The first run unpacks the zips into `snapshots/` (git ignores both folders).

To run the self-test, which needs no snapshot data:

```bash
python harness/run_eval.py --submission my_submission \
  --snapshots harness/selftest/snapshots --eval harness/selftest/eval.jsonl \
  --conflicts-gold harness/selftest/conflicts_gold.json
```

To use the system directly:

```python
import sys
sys.path[:0] = ["harness", "my_submission"]
from run_eval import load_snapshots
from pathlib import Path
import solution

ks = solution.build()
for snap in load_snapshots(Path("snapshots")):
    ks.ingest(snap)
print(ks.answer("What's the APR for a 60-month auto loan on a 2025 model-year vehicle?").text)
print(ks.answer("What was the 10-month jumbo certificate special APY?", as_of="2026-05-31").text)
print(len(ks.conflicts()), "conflicts")
```

## Repository layout

```
my_submission/
  solution.py        build() and the ingest / answer / conflicts entry points
  kb/text.py         normalization, tokenization, term, amount and date parsing
  kb/chunker.py      page to chunks: table rows, fact lines, fee lines, passages
  kb/authority.py    page classification and publication dates
  kb/store.py        versioned chunk store and BM25 index
  kb/query.py        question intent, qualifiers and numeric constraints
  kb/answerer.py     reranking, evidence rules, answer composition, refusals
  kb/conflicts.py    contradiction detection and resolution
  NOTES.md           approach and rules for what counts as current
  requirements.txt   empty: standard library only
harness/, eval/, starter/, snapshots/README.md   from the participant kit
```

## Known limitations

- Lexical retrieval misses some paraphrases that share no words with the page text. An embedding or LLM reranker could be added on top of the same evidence rules.
- Eligibility details and some "how do I" procedures the site never states directly come back as partial answers.
- When the site's own newest statement differs from what a question expects (for example, yearly volunteer counts), the system reports the newest statement.
