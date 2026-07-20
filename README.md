# Credit Card Fraud Detection — Two-Tier Cascade System

Machine learning-based credit card fraud detection using a two-tier cascade
architecture, built to explore the accuracy-latency trade-off in real-time
fraud screening — not just optimize for AUCPR in isolation.

## The Idea

Most fraud detection tutorials optimize purely for accuracy metrics like
AUCPR and ignore that, in production, transactions must be scored in
real time. This project measures both sides of that trade-off using a
**two-tier cascade**:

- **Tier 1** — a fast, lightweight model (Logistic Regression) that
  screens every transaction instantly.
- **Tier 2** — a slower, more accurate model (Random Forest) that only
  re-examines transactions Tier 1 is *uncertain* about.

By adjusting the escalation threshold, we trace out a full
latency-vs-accuracy trade-off curve (a Pareto frontier), rather than
reporting a single accuracy number in isolation.

## Dataset

Kaggle ULB Credit Card Fraud dataset — 284,807 transactions, only 492
labeled as fraud (~0.17%), with PCA-anonymized features (V1-V28) plus
Time and Amount. Stratified train/validation/test splits were used
throughout to preserve this class balance.

## Repository Structure

- `data/splits/` — stratified train/val/test splits (gzip-compressed)
- `models/` — trained Tier 1 (Logistic Regression) and Tier 2 (Random Forest)
  models, plus latency benchmarks
- `src/benchmarking/` — timing harness, benchmarking scripts, Pareto plotting
- `cascade_pipeline.py` — the FraudCascade class, routes transactions
  between Tier 1 and Tier 2 based on a threshold
- `results/` — comparison_table.csv and pareto_plot.png

## How It Works

```python
from cascade_pipeline import FraudCascade

cascade = FraudCascade(tier1_model, tier2_model,
                        low_threshold=0.05, high_threshold=0.95)
score, decision = cascade.predict(transaction)
```

Every transaction is scored by Tier 1 first. If Tier 1 confidence score
falls inside the threshold band, the transaction escalates to Tier 2 for
a second opinion. Otherwise, Tier 1 answer is final.

## Reproducing the Results

```bash
git clone https://github.com/shrishtigupta120807-create/credit-card-fraud-detection.git
cd credit-card-fraud-detection
cd src/benchmarking
python run_day4_benchmark.py
```

## Key Results

| System                | Mean Latency (sec) | AUCPR  |
|------------------------|--------------------:|-------:|
| Tier 1 alone            | 0.000185            | 0.7775 |
| Cascade (0.45-0.55)     | 0.000235            | 0.7777 |
| Cascade (0.35-0.65)     | 0.000306            | 0.7787 |
| Cascade (0.25-0.75)     | 0.000404            | 0.7779 |
| Cascade (0.15-0.85)     | 0.000626            | 0.7774 |
| Cascade (0.05-0.95)     | 0.001464            | 0.7785 |
| Tier 2 alone            | 0.008448            | 0.8537 |

**Findings:** Tier 1 alone is fastest but caps near 0.78 AUCPR. Tier 2
alone is most accurate (0.8537) but ~45x slower. The cascade lets you
dial between these two extremes - even at its widest threshold band, it
stays roughly 5.7x faster than running Tier 2 on everything, while only
modestly improving on Tier 1 accuracy alone.

## Limitations & Future Work

- Threshold bands were chosen based on observed score distribution rather
  than exhaustive search.
- Only 344 fraud cases in training data; results should be interpreted
  with appropriate caution.
- Latency measured on shared Colab CPU; absolute numbers will vary on
  other hardware, though relative trade-offs should hold.
