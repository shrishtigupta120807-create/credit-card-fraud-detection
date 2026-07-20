import time
import numpy as np


def benchmark_latency(model, X, n_samples=None):
    """
    Measures single-row prediction latency for a model.
    """
    if n_samples is None:
        n_samples = len(X)

    latencies = []

    for i in range(n_samples):
        transaction = X.iloc[i]
        start = time.perf_counter()
        model.predict_proba(transaction.values.reshape(1, -1))
        end = time.perf_counter()
        latencies.append(end - start)

    latencies = np.array(latencies)

    return {
        "mean": latencies.mean(),
        "median": np.median(latencies),
        "min": latencies.min(),
        "max": latencies.max(),
    }
