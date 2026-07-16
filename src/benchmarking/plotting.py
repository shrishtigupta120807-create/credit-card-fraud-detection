import matplotlib.pyplot as plt

def plot_pareto(latencies, accuracies, labels=None):
    """
    latencies: list of mean latency values (x-axis)
    accuracies: list of AUCPR values (y-axis)
    labels: optional list of strings to label each point (e.g., threshold values)
    """
    fig, ax = plt.subplots()
    ax.scatter(latencies, accuracies)
    
    if labels:
        for x, y, label in zip(latencies, accuracies, labels):
            ax.annotate(label, (x, y))
    
    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_ylabel("AUCPR")
    ax.set_title("Latency vs. Accuracy Trade-off")
    plt.show()

# Test it with fake placeholder numbers for now
fake_latencies = [0.0001, 0.0003, 0.0008, 0.002]
fake_accuracies = [0.70, 0.80, 0.88, 0.92]
fake_labels = ["threshold=0.9", "threshold=0.7", "threshold=0.5", "threshold=0.3"]

plot_pareto(fake_latencies, fake_accuracies, fake_labels)