# Lab 08 — Cluster and classify reviews

**Course:** TGS-2026064715 · **Maps to:** K5 K6 K7; A5 A6

## Goal

Compare a K-means elbow trace with a supervised classification baseline.

## Data

`hotel_reviews.csv` contains 120 fictional Singapore hotel and service reviews with dates, channel, rating, sentiment and property. Names and review text are synthetic; no customer records or external credentials are used.

## Steps

1. Open a terminal in this lab folder. Confirm Python 3.10+ with `python3 --version`.
2. Install the shared requirements from the parent `labs/requirements.txt` with `python3 -m pip install -r ../requirements.txt`.
3. Inspect the first rows and column names in `hotel_reviews.csv`. Check that `review_id` is unique and `sentiment` has three classes.
4. Run `python3 run.py` from this folder. Read the printed output and any files created in `output/`.
5. Explain how the result supports the lab goal, what a production dataset would add, and one failure or bias risk.

## Acceptance check

The script exits with code 0. Verify that the reported row count is 120. For model labs, inspect the holdout scores rather than training accuracy; for summaries, inspect the exported CSV/PNG.

## Reflection

How would different hotel review wording, label quality, or data leakage change this result?
