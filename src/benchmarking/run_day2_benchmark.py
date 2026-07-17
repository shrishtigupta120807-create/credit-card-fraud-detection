import joblib
import pandas as pd
import numpy as np

from timing_harness import benchmark_latency


# Load the real Tier 1 model


tier1_model = joblib.load("models/tier1_model.joblib")

# Load validation data (use val, not train — you benchmark on data the model didn't train on)
val = pd.read_csv("data/splits/val.csv.gz")
X_val = val.drop(columns=["Class"])

# Run your harness on the real model
results = benchmark_latency(tier1_model, X_val)
print(results)


import numpy as np
a_numbers = np.load("models/tier1_val_latency.npy",allow_pickle=True)
print(a_numbers)


import pandas as pd

comparison_table = pd.DataFrame([
    {
        "system": "Tier 1 alone",
        "mean_latency_sec": results["mean"],
        "median_latency_sec": results["median"],
        "max_latency_sec": results["max"],
    }
])

print(comparison_table)