"""Plot training curves for a generation and save to history/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent


def plot_generation(gen: int):
    log_dir = ROOT / "logs" / f"gen_{gen}"
    logs = [json.loads(f.read_text()) for f in sorted(log_dir.glob("child_*.json"))]
    if not logs:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for i, log in enumerate(logs):
        curve = log.get("training_curve", [])
        child = log["child"]
        val_acc = log["val_accuracy"]
        color = colors[i % len(colors)]
        label = f"child_{child} (val={val_acc:.3f})"

        if curve:
            epochs = [e["epoch"] for e in curve]
            accs = [e["acc"] for e in curve]
            ax.plot(epochs, accs, marker="o", markersize=3, color=color, label=label)
            # Mark val accuracy at the last epoch
            ax.axhline(val_acc, color=color, linestyle="--", alpha=0.4, linewidth=0.8)
        else:
            # No curve data — just show val_acc as a point
            ax.scatter([0], [val_acc], color=color, label=label, zorder=5)

    ax.set_title(f"Generation {gen} — training accuracy per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out = ROOT / "history" / f"gen_{gen}_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot   : history/gen_{gen}_curve.png")


def plot_all_generations():
    """Plot best and mean val_accuracy across all completed generations."""
    history_dir = ROOT / "history"
    summaries = sorted(history_dir.glob("gen_*_summary.json"))
    if not summaries:
        return

    gens, bests, means = [], [], []
    for path in summaries:
        data = json.loads(path.read_text())
        g = data["generation"]
        accs = [e["val_accuracy"] for e in data["leaderboard"]]
        gens.append(g)
        bests.append(max(accs))
        means.append(sum(accs) / len(accs) if accs else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, bests, marker="o", color="tab:blue", label="best val_acc")
    ax.plot(gens, means, marker="s", linestyle="--", color="tab:orange", label="mean val_acc")
    ax.fill_between(gens, means, bests, alpha=0.1, color="tab:blue")

    for g, b in zip(gens, bests):
        ax.annotate(f"{b:.3f}", (g, b), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)

    ax.set_title("HyperAgent CIFAR — accuracy over generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Val Accuracy")
    ax.set_xticks(gens)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = history_dir / "all_generations.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot   : history/all_generations.png")
