"""Create the Exercise 3 evaluation figures from extracted run metrics."""

import csv
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
            "legend.fontsize": 6.8,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_learning_curves(rows):
    steps = [int(row["step"]) for row in rows]
    train_slot = [float(row["train_slot_loss"]) for row in rows]
    val_slot = [float(row["val_slot_loss"]) for row in rows]
    f1_series = {
        "Intent": [float(row["val_intent_f1_percent"]) for row in rows],
        "Slot": [float(row["val_slot_f1_percent"]) for row in rows],
        "Overall": [float(row["val_overall_f1_percent"]) for row in rows],
    }

    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#6F4E9C"

    fig, (loss_ax, f1_ax) = plt.subplots(
        2,
        1,
        figsize=(3.25, 3.15),
        sharex=True,
        constrained_layout=True,
    )

    loss_ax.plot(steps, train_slot, color=blue, marker="o", label="Train")
    loss_ax.plot(
        steps,
        val_slot,
        color=orange,
        marker="s",
        linestyle="--",
        label="Validation",
    )
    minimum_index = min(range(len(val_slot)), key=val_slot.__getitem__)
    loss_ax.scatter(
        [steps[minimum_index]],
        [val_slot[minimum_index]],
        color=orange,
        marker="*",
        s=28,
        zorder=3,
    )
    loss_ax.annotate(
        f"minimum {val_slot[minimum_index]:.3f}",
        (steps[minimum_index], val_slot[minimum_index]),
        xytext=(5, 15),
        textcoords="offset points",
        fontsize=6.5,
    )
    loss_ax.annotate(
        "Train",
        (steps[-1], train_slot[-1]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )
    loss_ax.annotate(
        "Validation",
        (steps[-1], val_slot[-1]),
        xytext=(-4, -10),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )
    loss_ax.set_title("(a) Slot-tagging loss", loc="left")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(axis="y", color="0.85", linewidth=0.5)

    f1_ax.plot(
        steps,
        f1_series["Intent"],
        color=orange,
        marker="^",
        linestyle="--",
        label="Intent",
    )
    f1_ax.plot(
        steps,
        f1_series["Slot"],
        color=green,
        marker="s",
        linestyle="-.",
        label="Slot",
    )
    f1_ax.plot(
        steps,
        f1_series["Overall"],
        color=purple,
        marker="o",
        label="Overall",
    )
    f1_ax.annotate(
        f"{f1_series['Overall'][-1]:.1f}",
        (steps[-1], f1_series["Overall"][-1]),
        xytext=(-3, 7),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )
    f1_ax.set_title("(b) Internal validation F1", loc="left")
    f1_ax.set_xlabel("Training update")
    f1_ax.set_ylabel("F1 (%)")
    f1_ax.set_ylim(50, 92)
    f1_ax.set_xticks([500, 1500, 2500, 3500, 4500, 5000])
    f1_ax.legend(frameon=False, ncol=3, loc="lower right")
    f1_ax.grid(axis="y", color="0.85", linewidth=0.5)

    output_path = FIGURE_DIR / "bertnlu_learning_curves.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def plot_act_type_f1(rows):
    labels = [row["act_type"] for row in rows]
    dev_f1 = [float(row["validation_f1_percent"]) for row in rows]
    test_f1 = [float(row["test_f1_percent"]) for row in rows]
    y_positions = list(range(len(labels)))

    blue = "#0072B2"
    orange = "#D55E00"

    fig, ax = plt.subplots(figsize=(3.25, 1.8), constrained_layout=True)
    dev_y = [value - 0.12 for value in y_positions]
    test_y = [value + 0.12 for value in y_positions]
    ax.scatter(dev_f1, dev_y, color=blue, marker="o", s=22, label="Validation")
    ax.scatter(test_f1, test_y, color=orange, marker="s", s=20, label="Test")

    for values, positions in ((dev_f1, dev_y), (test_f1, test_y)):
        for value, position in zip(values, positions):
            ax.annotate(
                f"{value:.1f}",
                (value, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=6.5,
            )

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(84.5, 90.7)
    ax.set_xlabel("F1 (%)")
    ax.grid(axis="x", color="0.85", linewidth=0.5)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )

    output_path = FIGURE_DIR / "bertnlu_act_type_f1.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    training_rows = read_csv_rows(DATA_DIR / "bertnlu_training_metrics.csv")
    act_type_rows = read_csv_rows(DATA_DIR / "bertnlu_act_type_f1.csv")
    configure_style()
    for output_path in (
        plot_learning_curves(training_rows),
        plot_act_type_f1(act_type_rows),
    ):
        print(output_path)


if __name__ == "__main__":
    main()
