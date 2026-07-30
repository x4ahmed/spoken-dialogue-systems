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
          linestyle="-", connectionstyle="arc3", zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=linewidth,
        linestyle=linestyle,
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
    ax.set_xlim(-7, 107)
    ax.set_ylim(0, 48)
    ax.axis("off")

    # Dataset layer.
    rounded_box(ax, 18, 37, 64, 8.5, facecolor="#E8EEF5", edgecolor="#4C6680")
    ax.text(50, 42.8, "Simple MultiWOZ 2.1 - unified dataset",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(50, 39.4,
            "utterances  |  dialogue acts  |  dialogue states  |  ontology + database",
            ha="center", va="center", fontsize=7.5)
    arrow(
        ax, (50, 36.8), (50, 33.5), color="#4C6680", linewidth=1.1,
        linestyle="--",
    )
    ax.text(52, 35.2, "training/evaluation data and shared resources",
            ha="left", va="center", fontsize=6.9, color="#4C6680")

    # ConvLab3 pipeline boundary.
    rounded_box(
        ax, 5, 4.5, 90, 28,
        facecolor="#FFFFFF", edgecolor="#6B6B6B", linewidth=1.1, radius=1.1,
        zorder=1,
    )
    ax.text(7, 30.6, "ConvLab3 PipelineAgent (Exercises 2 and 5)",
            ha="left", va="center",
            fontsize=9, fontweight="bold", color="#333333")

    modules = [
        (9, "NLU", "BERTNLU", "#CFE8F3"),
        (30, "DST", "RuleDST", "#D8EEDD"),
        (51, "Policy", "MLE / PPO", "#F8E4BA"),
        (72, "NLG", "TemplateNLG", "#EADCF0"),
    ]
    for x, title, method, color in modules:
        rounded_box(ax, x, 19, 17, 8, facecolor=color)
        ax.text(x + 8.5, 24.2, title, ha="center", va="center",
                fontsize=8.1, fontweight="bold")
        ax.text(x + 8.5, 21.3, method, ha="center", va="center", fontsize=7.5)

    # User input, module outputs, and response.
    ax.text(0, 24.3, "User", ha="center", va="center", fontsize=8,
            fontweight="bold")
    ax.text(0, 21.7, "utterance", ha="center", va="center", fontsize=7.3)
    arrow(ax, (3.8, 23), (8.8, 23))
    arrow(ax, (26.2, 23), (29.8, 23))
    arrow(ax, (47.2, 23), (50.8, 23))
    arrow(ax, (68.2, 23), (71.8, 23))
    arrow(ax, (89.2, 23), (97.0, 23))
    ax.text(102, 24.3, "System", ha="center", va="center", fontsize=8,
            fontweight="bold")
    ax.text(102, 21.7, "response", ha="center", va="center", fontsize=7.3)

    ax.text(28.0, 17.1, "user dialogue acts", ha="center", va="center", fontsize=6.6)
    ax.text(49.0, 17.1, "dialogue state", ha="center", va="center", fontsize=6.6)
    ax.text(70.0, 17.1, "system dialogue acts", ha="center", va="center", fontsize=6.6)

    # TripPy was trained and evaluated independently, not inserted into the
    # Exercise 2/5 interactive PipelineAgent.
    rounded_box(ax, 30, 7.4, 17, 5.7, facecolor="#E9F5EC",
                edgecolor="#5D8065", linewidth=0.9)
    ax.text(38.5, 10.9, "TripPy (Exercise 4)", ha="center", va="center",
            fontsize=7.1, fontweight="bold")
    ax.text(38.5, 8.8, "evaluated separately", ha="center", va="center",
            fontsize=6.4, color="#4F6C55")
    arrow(ax, (38.5, 13.3), (38.5, 18.8), color="#5D8065",
          linewidth=0.85, linestyle="--")

    # The policy vectorizer queries the database using belief-state
    # constraints; returned entities and counts inform policy selection and
    # action lexicalization.
    rounded_box(ax, 51, 7.4, 17, 5.7, facecolor="#F0F0F0")
    ax.text(59.5, 10.9, "Database", ha="center", va="center",
            fontsize=7.5, fontweight="bold")
    ax.text(59.5, 8.8, "query / match counts", ha="center", va="center",
            fontsize=6.4, color="#555555")
    arrow(ax, (59.5, 13.3), (59.5, 18.8), style="<->", linewidth=0.85)

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
