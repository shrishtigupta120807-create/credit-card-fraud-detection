import matplotlib.pyplot as plt

def plot_pareto(latencies, accuracies, labels=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))  # also make the figure bigger
    ax.scatter(latencies, accuracies)
    ax.set_xscale('log')  # ADD THIS LINE
    if labels:
        for x, y, label in zip(latencies, accuracies, labels):
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9, rotation=15)
    ax.set_xlabel("Mean Latency (seconds, log scale)")
    ax.set_ylabel("AUCPR")
    ax.set_title("Latency vs. Accuracy Trade-off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()