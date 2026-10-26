# Notes

## API keys

None. The system is fully local, uses only the Python standard library and makes no network calls.

## Architecture

`solution.py` exposes `build()` and wires three components from the `kb/` package:

| Module | Role |
|---|---|
| `kb/chunker.py` | Splits each page into chunks: one per table row (with its caption, column names and the table's "Rates are effective as of" date), one per `Label: value` fact line or fee line, and short prose passages. |
| `kb/store.py` | A versioned chunk store with a BM25 index. Each distinct chunk is stored once with a bitmask of the snapshots that contained it, so a single index serves the latest view and every "as of" view. Pages missing from a later crawl keep their last known version in the view. |
| `kb/authority.py` | Classifies every URL (official rate page, core info page, promo, news, blog, document, calculator, junk) and reads publication dates from news and blog pages. |
| `kb/query.py` | Reads the question: intent (rate, hours, address, fee, count, link, date, change over time), product qualifiers (jumbo, business, refinance, IRA...), terms, balances, model years and any "as of" date in the text. |
| `kb/answerer.py` | Reranks retrieved chunks, enforces evidence rules, composes a short answer from the winning table rows or sentences, cites sources and declines when the evidence is missing. |
| `kb/conflicts.py` | Tracks every rate fact across pages and snapshots, reports the ones whose values disagree, and picks a winner with a stated reason. It also covers a few prose claims (branch count, Forbes recognition, membership, assets, employees, routing number). |

`answer()` takes about 10 ms. Ingesting all five snapshots takes about 5 seconds.

## How the system decides what is "current"

1. **Time.** With no `as_of`, the view is the newest snapshot. With an `as_of` date, the view is the latest snapshot taken on or before that date. A date before the first snapshot gets "I don't know". A month named in the question ("in May 2026") is treated as an `as_of` date.
2. **Source authority.** Official rate and product pages outrank core info pages, which outrank promos and landing pages, then news, then blogs. Calculators rank last. Rates are only ever quoted from official, core, promo or document pages. Calculators, blogs and news are never used as the source of a rate.
3. **Effective dates.** When two pages show the same product and tier with different numbers, the table with the later "Rates are effective as of" date and the higher authority wins. The losing value is reported in `conflict_note` and in `conflicts()`.
4. **Recency of statements.** Counts that drift over time, such as the number of branches, are taken from the most recently dated statement. Near-duplicate press releases resolve to the newest one, and passages that only mention old years are demoted.
5. **Expired offers.** Text that says an offer ended before the snapshot date is excluded.
6. **Qualifiers are different facts.** Jumbo vs standard, business vs personal, refinance vs purchase, intro vs ongoing rate, and term, balance or model-year tiers are matched against the question. A row for another tier is never offered as the answer.

## Honesty rules

The system answers "I don't know" (`confident=False`, with no numbers in the text) when:

- a named entity in the question is not on the site
- no rate evidence exists for the asked product
- the best evidence covers a different term or tier than the one asked for
- too little of the question is covered by the best evidence

## Running the scorer

The harness is not copied into this repository. Run it from the hackathon kit and point it at this folder:

```bash
python harness/run_eval.py --submission /path/to/credit-union-rag-system/my_submission --eval eval/public_eval.jsonl
```

If the solution is imported outside the scorer, it looks for `harness/contract.py` in the parent folders, or in the folder named by the `HACKATHON_HARNESS` environment variable.
