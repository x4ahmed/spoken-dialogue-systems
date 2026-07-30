"""Generate the Simple MultiWOZ 2.1 and ConvLab3 overview figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPORT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPORT_DIR / "figures" / "convlab3_pipeline_overview.pdf"


def rounded_box(ax, x, y, width, height, *, facecolor, edgecolor="#333333",
                linewidth=1.0, radius=0.8, zorder=2):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.25,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, *, color="#333333", style="-|>", linewidth=1.0,
          connectionstyle="arc3", zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(7.15, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    # Dataset layer.
    rounded_box(ax, 20, 34, 60, 8, facecolor="#E8EEF5", edgecolor="#4C6680")
    ax.text(50, 39.5, "Simple MultiWOZ 2.1 - unified format",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(50, 36.5,
            "turns  |  dialogue acts  |  dialogue states  |  ontology/database  |  responses",
            ha="center", va="center", fontsize=7.5)
    arrow(ax, (50, 33.8), (50, 30.5), color="#4C6680", linewidth=1.1)
    ax.text(52, 32.2, "standardized examples and supervision",
            ha="left", va="center", fontsize=7, color="#4C6680")

    # ConvLab3 pipeline boundary.
    rounded_box(
        ax, 7, 4.5, 86, 25.5,
        facecolor="#FFFFFF", edgecolor="#6B6B6B", linewidth=1.1, radius=1.1,
        zorder=1,
    )
    ax.text(9, 28.2, "ConvLab3 PipelineAgent", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#333333")

    modules = [
        (10, "NLU", "BERTNLU", "#CFE8F3"),
        (31, "DST", "TripPy", "#D8EEDD"),
        (52, "Policy", "PPO", "#F8E4BA"),
        (73, "NLG", "template / model", "#EADCF0"),
    ]
    for x, title, method, color in modules:
        rounded_box(ax, x, 17, 17, 8, facecolor=color)
        ax.text(x + 8.5, 22.2, title, ha="center", va="center",
                fontsize=8.1, fontweight="bold")
        ax.text(x + 8.5, 19.3, method, ha="center", va="center", fontsize=7.5)

    # User input, module outputs, and response.
    ax.text(1.0, 22.3, "User", ha="left", va="center", fontsize=8,
            fontweight="bold")
    ax.text(1.0, 19.7, "utterance", ha="left", va="center", fontsize=7.3)
    arrow(ax, (6.0, 21), (9.8, 21))
    arrow(ax, (27.2, 21), (30.8, 21))
    arrow(ax, (48.2, 21), (51.8, 21))
    arrow(ax, (69.2, 21), (72.8, 21))
    arrow(ax, (90.2, 21), (94.0, 21))
    ax.text(99.0, 22.3, "System", ha="right", va="center", fontsize=8,
            fontweight="bold")
    ax.text(99.0, 19.7, "response", ha="right", va="center", fontsize=7.3)

    ax.text(29.0, 15.1, "dialogue acts", ha="center", va="center", fontsize=6.8)
    ax.text(50.0, 15.1, "belief state", ha="center", va="center", fontsize=6.8)
    ax.text(71.0, 15.1, "system act", ha="center", va="center", fontsize=6.8)

    # The state queries the database; results inform policy selection.
    rounded_box(ax, 41.5, 6.5, 17, 5.5, facecolor="#F0F0F0")
    ax.text(50, 9.25, "Ontology + database", ha="center", va="center",
            fontsize=7.5, fontweight="bold")
    arrow(ax, (39.5, 16.8), (45.0, 12.2), linewidth=0.9,
          connectionstyle="arc3,rad=0.08")
    arrow(ax, (55.0, 12.2), (60.5, 16.8), linewidth=0.9,
          connectionstyle="arc3,rad=0.08")
    ax.text(39.5, 10.9, "constraints", ha="right", va="center", fontsize=6.6)
    ax.text(60.5, 10.9, "DB result", ha="left", va="center", fontsize=6.6)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "Simple MultiWOZ 2.1 and ConvLab3 pipeline overview",
            "Author": "Ahmed Moustafa",
        },
    )
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
