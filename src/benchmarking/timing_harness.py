import pandas as pd
import numpy as np
pd.read_csv("data/splits/train.csv.gz")
from sklearn.dummy import DummyClassifier

# Load data
train = pd.read_csv("data/splits/train.csv.gz")

# Separate features (X) from the label (y)
X_train = train.drop(columns=["Class"])
y_train = train["Class"]

# Train a dummy model
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

# Sanity check: predict on the first row
first_row = X_train.iloc[[0]]   # double brackets keep it as a DataFrame, not a Series
prediction = dummy.predict(first_row)
print(prediction)

import time

# Time a single prediction
start = time.perf_counter()
prediction = dummy.predict(first_row)
end = time.perf_counter()

latency = end - start
print(f"Latency: {latency:.6f} seconds")


def benchmark_latency(model, X, n_samples=200):
    """
    Measures single-row prediction latency for a model.

    model: any fitted model with a .predict() method
    X: a DataFrame of feature rows to test on
    n_samples: how many rows to time

    Returns a dictionary of latency stats in seconds.
    """
    latencies = []

    for i in range(n_samples):
        row = X.iloc[[i]]
        start = time.perf_counter()
        model.predict(row)
        end = time.perf_counter()
        latencies.append(end - start)

    latencies = np.array(latencies)

    return {
        "mean": latencies.mean(),
        "median": np.median(latencies),
        "min": latencies.min(),
        "max": latencies.max(),
    }



results = benchmark_latency(dummy, X_train)
print(results)