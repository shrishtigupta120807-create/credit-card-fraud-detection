import sys
import os
import time
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)

# Get the folder this script lives in, then go up two levels to the repo root
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(repo_root)

from cascade_pipeline import FraudCascade

# Load models
tier1_model = joblib.load(os.path.join(repo_root, "models", "tier1_model.joblib"))
tier2_model = joblib.load(os.path.join(repo_root, "models", "tier2_model.pkl"))
val = pd.read_csv(os.path.join(repo_root, "data", "splits", "val.csv.gz"))
X_val = val.drop(columns=["Class"])
y_val = val["Class"]


from timing_harness import benchmark_latency

# Compute Tier 1 alone — latency AND accuracy, freshly measured
tier1_scores = tier1_model.predict_proba(X_val)[:, 1]
tier1_latency = benchmark_latency(tier1_model, X_val)
tier1_aucpr = average_precision_score(y_val, tier1_scores)

# Compute Tier 2 alone — same idea
tier2_scores = tier2_model.predict_proba(X_val)[:, 1]
tier2_latency = benchmark_latency(tier2_model, X_val)
tier2_aucpr = average_precision_score(y_val, tier2_scores)

print("Tier 1 alone:", tier1_latency["mean"], tier1_aucpr)
print("Tier 2 alone:", tier2_latency["mean"], tier2_aucpr)



# --- Quick sanity check on one transaction ---
one_transaction = X_val.iloc[0]
cascade = FraudCascade(tier1_model, tier2_model, low_threshold=0.3, high_threshold=0.7)
score, decision = cascade.predict(one_transaction)
print("Sanity check — Score:", score, "Decision:", decision)

# --- Full threshold sweep ---
threshold_bands = [
    (0.45, 0.55),
    (0.35, 0.65),
    (0.25, 0.75),
    (0.15, 0.85),
    (0.05, 0.95),
]

sweep_results = []

for low, high in threshold_bands:
    cascade = FraudCascade(tier1_model, tier2_model, low_threshold=low, high_threshold=high)

    latencies = []
    scores = []
    escalated_count = 0

    for i in range(len(X_val)):
        transaction = X_val.iloc[i]
        start = time.perf_counter()
        score, decision = cascade.predict(transaction)
        end = time.perf_counter()
        latencies.append(end - start)
        scores.append(score)
        if decision == "escalated":
            escalated_count += 1

    latencies = np.array(latencies)
    aucpr = average_precision_score(y_val, scores)

    print(f"Band ({low}, {high}): mean_latency={latencies.mean():.6f}, aucpr={aucpr:.4f}, escalated={escalated_count}/{len(X_val)}")
    sweep_results.append({
        "low": low,
        "high": high,
        "mean_latency": latencies.mean(),
        "aucpr": aucpr,
        "escalated": escalated_count,
    })

    comparison_table = pd.DataFrame([
    {"system": "Tier 1 alone", "mean_latency_sec": tier1_latency["mean"], "aucpr": tier1_aucpr},
    {"system": "Tier 2 alone", "mean_latency_sec": tier2_latency["mean"], "aucpr": tier2_aucpr},
] + [
    {"system": f"Cascade ({low}-{high})", "mean_latency_sec": r["mean_latency"], "aucpr": r["aucpr"]}
    for (low, high), r in zip(threshold_bands, sweep_results)
])

print(comparison_table)

# Save it so it doesn't disappear when the script ends
comparison_table.to_csv(os.path.join(repo_root, "results", "comparison_table.csv"), index=False)





from plotting import plot_pareto

# Build the real lists from your actual sweep_results (not fake data)
pareto_latencies = [0.000196, 0.001278] + [r["mean_latency"] for r in sweep_results]
pareto_aucprs = [None, None] + [r["aucpr"] for r in sweep_results]  # Tier1/Tier2 alone don't have AUCPR from this run
pareto_labels = ["Tier 1 alone", "Tier 2 alone"] + [f"{low}-{high}" for low, high in threshold_bands]

# Only plot points that actually have both latency and AUCPR values
plot_latencies = [tier1_latency["mean"], tier2_latency["mean"]] + [r["mean_latency"] for r in sweep_results]
plot_aucprs = [tier1_aucpr, tier2_aucpr] + [r["aucpr"] for r in sweep_results]
plot_labels = ["Tier1 alone", "Tier2 alone"] + [f"{low}-{high}" for low, high in threshold_bands]



plot_pareto(plot_latencies, plot_aucprs, labels=plot_labels)



