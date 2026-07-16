# credit-card-fraud-detection
Machine learning-based credit card fraud detection using multiple classical ML algorithms with fair model evaluation and data leakage prevention.
## Benchmarking 
- `src/benchmarking/timing_harness.py` — measures single-transaction inference latency for any fitted model (mean/median/min/max)
- `src/benchmarking/plotting.py` — generates the latency-vs-accuracy Pareto plot; currently tested with placeholder data, will be fed real threshold-sweep results in 
