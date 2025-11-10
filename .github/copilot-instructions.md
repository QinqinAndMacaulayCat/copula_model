## Quick orientation

This repository holds a small exploratory copula/data pipeline. Key facts an AI coding agent should know before editing or running code:

- Primary code directories:
  - `data/` — a standalone ETL script `data.py` that downloads S&P500 tickers and prices and writes CSVs.
  - `copula_model/` — the (mostly empty) package; currently only `__init__.py` exists.
  - `tests/` — present but empty; no test harness detected.

## Big-picture architecture & data flow

- `data/data.py` is the main data ingestion script. It:
  1. Scrapes the S&P 500 tickers from Wikipedia using `requests` + `pandas.read_html`.
  2. Uses `yfinance` to bulk-download historical Adjusted/Close prices for those tickers.
  3. Produces two CSVs: price table and log returns.

- Important: `data.py` runs work at module import / top-level. Importing the module will start network downloads. Treat it like a script (run it directly) rather than importing it from other modules.

## Project-specific conventions & gotchas (do not assume defaults)

- Running location matters: `data.py` writes files using relative paths. If you run `python data/data.py` from the repository root it will create `sp500_prices.csv` and `sp500_log_returns.csv` in the current working directory (repo root). If you want outputs inside `data/`, run `cd data && python data.py`.

- Network I/O and mutability at import-time:
  - `get_sp500_tickers()` scrapes Wikipedia and returns tickers. The file then proceeds to call `yf.download(...)` at module top-level. Avoid importing this module in unit tests or other codepaths — it will trigger long-running network activity.

- External dependencies used (install these before running):
  - pandas, numpy, yfinance, requests
  - No requirements file detected; create an isolated venv before installing.

## How to run (macOS / zsh example)

1. Create and activate a venv:

   python3 -m venv .venv
   source .venv/bin/activate

2. Install packages:

   pip install pandas numpy yfinance requests

3. Generate data CSVs (write outputs to the `data/` directory):

   cd data
   python data.py

Notes: downloading the entire S&P500 price history is network- and time-intensive. Expect several minutes and many requests. `yfinance` uses multiple threads by default in the script (`threads=True`).

## Patterns and examples for edits

- If you want to refactor ingestion to be import-safe, move heavy work under a `main()` or `if __name__ == '__main__':` guard. Example:
  - create `def main():` that runs the download and CSV writes, and keep top-level definitions (constants and helper functions) importable.

- To update date range: modify `START_DATE` / `END_DATE` constants in `data/data.py`. The file currently sets END_DATE to `2025-11-03`.

## Integration points

- Network: Wikipedia (tickers scraping) and Yahoo Finance (price download). Be mindful of rate limits and User-Agent headers.

- File I/O: writes CSVs to relative paths. Tests or CI that import `data.data` will create files unless the module is refactored.

## Minimal checklist for PRs touching data ingestion

1. Ensure long-running network work is not executed on import.
2. Prefer making the downloader idempotent and directory-aware (accept an output directory argument).
3. Add small unit tests that mock network calls (e.g., patch `requests.get` and `yfinance.download`) before adding them to `tests/`.

---
If any part of this is unclear or you want more examples (unit tests, a requirements file, or a quick refactor to make ingestion import-safe), tell me which area to expand and I will update the file.
