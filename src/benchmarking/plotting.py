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