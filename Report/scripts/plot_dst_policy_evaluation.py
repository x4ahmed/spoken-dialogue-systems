"""Create the Exercise 4 and 5 figures from compact saved-run extracts."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPORT_DIR / "data"
FIGURE_DIR = REPORT_DIR / "figures"


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_trippy(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    full = grouped["Full features"]
    epochs = [int(row["epoch"]) for row in full]
    train_loss = [float(row["train_loss"]) for row in full]
    dev_rows = [row for row in full if row["dev_loss"]]
    dev_epochs = [int(row["epoch"]) for row in dev_rows]
    dev_loss = [float(row["dev_loss"]) for row in dev_rows]

    colors = {
        "Full features": "#0072B2",
        "No auxiliary": "#D55E00",
        "No history or auxiliary": "#009E73",
    }
    markers = {
        "Full features": "o",
        "No auxiliary": "s",
        "No history or auxiliary": "^",
    }

    fig, (loss_ax, jga_ax) = plt.subplots(
        2, 1, figsize=(3.25, 3.2), sharex=True, constrained_layout=True
    )
    loss_ax.plot(epochs, train_loss, color="#0072B2", marker="o", label="Train")
    loss_ax.plot(
        dev_epochs,
        dev_loss,
        color="#D55E00",
        marker="s",
        linestyle="--",
        label="Validation",
    )
    loss_ax.set_yscale("log")
    loss_ax.set_title("(a) Full-input loss", loc="left")
    loss_ax.set_ylabel("Loss (log scale)")
    loss_ax.grid(axis="y", color="0.85", linewidth=0.5)
    loss_ax.legend(frameon=False, ncol=2, loc="upper right")
    loss_ax.annotate(
        f"{train_loss[-1]:.2f}",
        (epochs[-1], train_loss[-1]),
        xytext=(-3, -9),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )
    loss_ax.annotate(
        f"{dev_loss[-1]:.2f}",
        (dev_epochs[-1], dev_loss[-1]),
        xytext=(-3, 6),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )

    for condition, condition_rows in grouped.items():
        points = [row for row in condition_rows if row["dev_jga"]]
        jga_ax.plot(
            [int(row["epoch"]) for row in points],
            [100 * float(row["dev_jga"]) for row in points],
            color=colors[condition],
            marker=markers[condition],
            label=condition,
        )
    jga_ax.set_title("(b) Validation joint goal accuracy", loc="left")
    jga_ax.set_xlabel("Training epoch")
    jga_ax.set_ylabel("JGA (%)")
    jga_ax.set_xticks([2, 4, 6, 8, 10])
    jga_ax.set_ylim(0, 48)
    jga_ax.grid(axis="y", color="0.85", linewidth=0.5)
    jga_ax.legend(frameon=False, loc="lower right")

    output_path = FIGURE_DIR / "trippy_learning_ablation.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def plot_ppo(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)

    styles = {
        "gamma .90 low LR": ("#0072B2", "o", "-"),
        "gamma .99 low LR": ("#D55E00", "s", "--"),
        "gamma .90 high LR": ("#009E73", "^", "-."),
        "gamma .99 high LR": ("#6F4E9C", "D", ":"),
    }
    labels = {
        "gamma .90 low LR": r"$\gamma=.90$, lr=$10^{-4}$",
        "gamma .99 low LR": r"$\gamma=.99$, lr=$10^{-4}$",
        "gamma .90 high LR": r"$\gamma=.90$, lr=$3\times10^{-4}$",
        "gamma .99 high LR": r"$\gamma=.99$, lr=$3\times10^{-4}$",
    }

    fig, (success_ax, actions_ax) = plt.subplots(
        2, 1, figsize=(3.25, 3.35), sharex=True, constrained_layout=True
    )
    for policy, policy_rows in grouped.items():
        policy_rows.sort(key=lambda row: int(row["dialogues"]))
        x_values = [int(row["dialogues"]) / 1000 for row in policy_rows]
        color, marker, linestyle = styles[policy]
        success_ax.plot(
            x_values,
            [100 * float(row["success_rate_strict"]) for row in policy_rows],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=labels[policy],
        )
        actions_ax.plot(
            x_values,
            [float(row["avg_actions"]) for row in policy_rows],
            color=color,
            marker=marker,
            linestyle=linestyle,
        )

    success_ax.set_title("(a) Strict success rate", loc="left")
    success_ax.set_ylabel("Success (%)")
    success_ax.set_ylim(45, 62)
    success_ax.grid(axis="y", color="0.85", linewidth=0.5)
    success_ax.legend(frameon=False, ncol=2, loc="lower right")
    success_ax.annotate(
        "best 59.4",
        (20, 59.4),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=6.5,
    )

    actions_ax.set_title("(b) Average system actions", loc="left")
    actions_ax.set_xlabel("Training dialogues (thousands)")
    actions_ax.set_ylabel("Acts per turn")
    actions_ax.set_xticks([0, 10, 20, 30, 40])
    actions_ax.set_ylim(3.7, 5.5)
    actions_ax.grid(axis="y", color="0.85", linewidth=0.5)
    actions_ax.annotate(
        "selected 4.65",
        (20, 4.6487),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=6.5,
    )

    output_path = FIGURE_DIR / "ppo_evaluation_curves.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    outputs = (
        plot_trippy(read_csv_rows(DATA_DIR / "trippy_learning_metrics.csv")),
        plot_ppo(read_csv_rows(DATA_DIR / "ppo_evaluation_metrics.csv")),
    )
    for output_path in outputs:
        print(output_path)


if __name__ == "__main__":
    main()
