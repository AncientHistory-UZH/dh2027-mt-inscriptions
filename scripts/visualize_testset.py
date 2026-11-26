"""
Testset Visualization
=====================
Generates publication-ready visualizations of the testset evaluation results
and selects representative example translations for presentation slides.
"""

import pandas as pd
import numpy as np
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "testset", "results")
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")

# ── style ──────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini 2.5\nFlash",
    "claude-sonnet-4-0": "Claude\nSonnet 4",
    "gpt-5-nano-2025-08-07": "GPT-5\nNano",
}
MODEL_COLORS = {
    "gemini-2.5-flash": "#4285F4",
    "claude-sonnet-4-0": "#D97706",
    "gpt-5-nano-2025-08-07": "#10B981",
}
LANG_NAMES = {"DE": "German", "EN": "English", "FR": "French"}


def setup_style():
    """Configure matplotlib for presentation-quality output."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


# ======================================================================
# 1. GROUPED BAR CHART – All Metrics by Model × Language
# ======================================================================

def plot_grouped_bars(agg_df):
    """Three-panel grouped bar chart: BLEU, chrF, BERTScore."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=False)
    fig.patch.set_facecolor("#FAFBFC")

    metrics = [
        ("corpus_bleu", "BLEU Score", (0, 45)),
        ("corpus_chrf", "chrF Score", (30, 75)),
        ("bertscore_f1", "BERTScore F1", (0.65, 0.96)),
    ]
    languages = ["DE", "EN", "FR"]
    models = list(MODEL_LABELS.keys())
    x = np.arange(len(languages))
    width = 0.25

    for ax, (metric, title, ylim) in zip(axes, metrics):
        ax.set_facecolor("#FAFBFC")
        for i, model in enumerate(models):
            subset = agg_df[agg_df["model"] == model].set_index("language")
            vals = [subset.loc[lang, metric] if lang in subset.index else 0 for lang in languages]
            bars = ax.bar(
                x + i * width - width,
                vals,
                width * 0.9,
                label=MODEL_LABELS[model].replace("\n", " "),
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                alpha=0.9,
            )
            # Value labels
            for bar, v in zip(bars, vals):
                fmt = f"{v:.1f}" if metric != "bertscore_f1" else f"{v:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ylim[1] - ylim[0]) * 0.015,
                    fmt,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#333",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_NAMES[l] for l in languages], fontweight="medium")
        ax.set_title(title, pad=12)
        ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left", framealpha=0.9, edgecolor="#DDD")

    fig.suptitle(
        "Translation Quality: LLM Models vs. Human References",
        fontsize=20, fontweight="bold", y=1.02, color="#1a1a2e"
    )
    fig.tight_layout()

    path = os.path.join(VIS_DIR, "metrics_comparison.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    fig.savefig(path.replace(".png", ".pdf"), facecolor=fig.get_facecolor())
    print(f"  → {path}")
    plt.close(fig)


# ======================================================================
# 2. HEATMAP – Model × Language × Metric
# ======================================================================

def plot_heatmap(agg_df):
    """Compact heatmap showing all scores at a glance."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.patch.set_facecolor("#FAFBFC")

    metrics = [
        ("corpus_bleu", "BLEU", "YlOrRd", ".1f"),
        ("corpus_chrf", "chrF", "YlGnBu", ".1f"),
        ("bertscore_f1", "BERTScore F1", "RdPu", ".3f"),
    ]
    models = list(MODEL_LABELS.keys())
    model_labels = [MODEL_LABELS[m].replace("\n", " ") for m in models]
    languages = ["DE", "EN", "FR"]

    for ax, (metric, title, cmap, fmt) in zip(axes, metrics):
        matrix = []
        for model in models:
            row = []
            for lang in languages:
                val = agg_df[(agg_df["model"] == model) & (agg_df["language"] == lang)]
                row.append(val[metric].values[0] if len(val) else 0)
            matrix.append(row)

        matrix = np.array(matrix)
        sns.heatmap(
            matrix, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=[LANG_NAMES[l] for l in languages],
            yticklabels=model_labels if ax == axes[0] else [""] * len(models),
            ax=ax, cbar=False, linewidths=2, linecolor="white",
            annot_kws={"fontsize": 14, "fontweight": "bold"},
        )
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.tick_params(axis="both", length=0)

    fig.suptitle(
        "Score Heatmap: Model × Language",
        fontsize=20, fontweight="bold", y=1.05, color="#1a1a2e"
    )
    fig.tight_layout()

    path = os.path.join(VIS_DIR, "score_heatmap.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    fig.savefig(path.replace(".png", ".pdf"), facecolor=fig.get_facecolor())
    print(f"  → {path}")
    plt.close(fig)


# ======================================================================
# 3. BOX PLOTS – Score Distributions
# ======================================================================

def plot_distributions(per_df):
    """Box + strip plots showing per-inscription score distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#FAFBFC")

    metrics = [
        ("sentence_bleu", "Sentence BLEU"),
        ("sentence_chrf", "Sentence chrF"),
        ("bertscore_f1", "BERTScore F1"),
    ]
    models = list(MODEL_LABELS.keys())

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in per_df.columns:
            ax.set_visible(False)
            continue

        plot_data = per_df[per_df["model"].isin(models)].copy()
        plot_data["model_label"] = plot_data["model"].map(
            lambda m: MODEL_LABELS.get(m, m).replace("\n", " ")
        )

        sns.boxplot(
            data=plot_data, x="model_label", y=metric,
            palette=list(MODEL_COLORS.values()),
            ax=ax, width=0.5, fliersize=3,
            boxprops=dict(alpha=0.8),
            zorder=3,
        )
        sns.stripplot(
            data=plot_data, x="model_label", y=metric,
            palette=list(MODEL_COLORS.values()),
            ax=ax, size=2.5, alpha=0.3, jitter=0.2,
            zorder=2,
        )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#FAFBFC")

    fig.suptitle(
        "Per-Inscription Score Distributions",
        fontsize=20, fontweight="bold", y=1.02, color="#1a1a2e"
    )
    fig.tight_layout()

    path = os.path.join(VIS_DIR, "score_distributions.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    fig.savefig(path.replace(".png", ".pdf"), facecolor=fig.get_facecolor())
    print(f"  → {path}")
    plt.close(fig)


# ======================================================================
# 4. EXAMPLE TRANSLATIONS – One per language
# ======================================================================

def _wrap(text, width=72):
    return "\n".join(textwrap.wrap(text, width=width)) if isinstance(text, str) else ""


def _lighten(hex_color, factor):
    """Lighten a hex color towards white."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


MODEL_SHORT = {
    "gemini-2.5-flash": "Gemini Flash",
    "claude-sonnet-4-0": "Claude Sonnet",
    "gpt-5-nano-2025-08-07": "GPT-5 Nano",
}


def _draw_translation_card(ax, ex, show_scores=False, score_info=None):
    """Draw a single translation card in the given axes area.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw into (will be turned off; we draw manually).
    ex : dict
        Entry from full_translations.json.
    show_scores : bool
        If True, append metric values beside the model labels.
    score_info : dict or None
        {model: {"bleu": ..., "chrf": ..., "bert": ...}} if show_scores.
    """
    ax.axis("off")
    models = list(MODEL_LABELS.keys())

    lang = ex.get("language", "?")
    lang_name = LANG_NAMES.get(lang, lang)
    edcs_id = ex.get("edcs_id", "?")

    # ── Title + Latin ──
    ax.text(
        0.5, 0.97,
        f"EDCS-{edcs_id}  ·  {lang_name}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=16, fontweight="bold", color="#1a1a2e",
    )
    latin = _wrap(ex.get("latin_text", ""), 85)
    ax.text(
        0.5, 0.91, latin,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color="#555", style="italic", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8EEF5",
                  edgecolor="#B0C4DE", alpha=0.7),
    )

    # ── Translation rows ──
    rows = [
        ("Human Reference", ex.get("human_translation", ""),
         "#2D5F2D", "#E8F5E8", "#A5D6A5", None),
    ]
    for m in models:
        score_str = None
        if show_scores and score_info and m in score_info:
            s = score_info[m]
            score_str = f"BLEU {s.get('bleu', '?')}  ·  chrF {s.get('chrf', '?')}  ·  BERT {s.get('bert', '?')}"
        rows.append((
            MODEL_SHORT[m],
            ex.get(f"llm_{m}", "—"),
            MODEL_COLORS[m],
            _lighten(MODEL_COLORS[m], 0.88),
            _lighten(MODEL_COLORS[m], 0.6),
            score_str,
        ))

    n_rows = len(rows)
    row_height = 0.17
    top_start = 0.76

    for i, (label, text, color, bg, border, score_str) in enumerate(rows):
        y_top = top_start - i * (row_height + 0.02)

        box = FancyBboxPatch(
            (0.02, y_top - row_height), 0.96, row_height,
            boxstyle="round,pad=0.015",
            facecolor=bg, edgecolor=border, linewidth=1.2,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)

        # Label
        label_text = label
        ax.text(
            0.04, y_top - row_height / 2, label_text,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=10, fontweight="bold", color=color,
        )

        # Score badge (if present)
        if score_str:
            ax.text(
                0.96, y_top - 0.01, score_str,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#666", style="italic",
            )

        # Translation text
        wrapped = _wrap(str(text), 75)
        ax.text(
            0.18, y_top - row_height / 2, wrapped,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=9, color="#333",
        )


def plot_per_language_examples(full_translations, per_df):
    """Create one example translation card per language (DE, EN, FR)."""
    models = list(MODEL_LABELS.keys())

    for lang in ["DE", "EN", "FR"]:
        lang_name = LANG_NAMES[lang]

        # Filter valid entries for this language
        valid = [
            e for e in full_translations
            if e.get("language") == lang
            and all(
                e.get(f"llm_{m}") and not str(e.get(f"llm_{m}", "")).startswith("ERROR")
                for m in models
            )
            and isinstance(e.get("human_translation"), str)
            and len(e.get("human_translation", "")) > 15
        ]

        if not valid:
            print(f"  ⚠ No valid examples for {lang}")
            continue

        # Pick a medium-length example for each language
        valid.sort(key=lambda x: len(x.get("latin_text", "")))
        ex = valid[len(valid) // 2]

        # Get scores for this specific inscription
        edcs_id = ex.get("edcs_id")
        score_info = {}
        for m in models:
            row = per_df[(per_df["model"] == m) & (per_df["edcs_id"] == edcs_id) & (per_df["language"] == lang)]
            if len(row):
                score_info[m] = {
                    "bleu": f"{row['sentence_bleu'].values[0]:.1f}",
                    "chrf": f"{row['sentence_chrf'].values[0]:.1f}",
                    "bert": f"{row['bertscore_f1'].values[0]:.3f}" if "bertscore_f1" in row.columns else "—",
                }

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor("#FAFBFC")
        fig.suptitle(
            f"Example Translation — {lang_name}",
            fontsize=20, fontweight="bold", y=0.98, color="#1a1a2e",
        )
        _draw_translation_card(ax, ex, show_scores=True, score_info=score_info)

        path = os.path.join(VIS_DIR, f"example_{lang.lower()}.png")
        fig.savefig(path, facecolor=fig.get_facecolor())
        fig.savefig(path.replace(".png", ".pdf"), facecolor=fig.get_facecolor())
        print(f"  → {path}")
        plt.close(fig)


# ======================================================================
# 5. SIDE-BY-SIDE: Best vs. Worst Translation
# ======================================================================

def plot_best_worst_comparison(full_translations, per_df):
    """Side-by-side comparison of the best- and worst-scoring translations."""
    models = list(MODEL_LABELS.keys())

    # Use Claude (best model) as the reference for picking best/worst
    best_model = "claude-sonnet-4-0"
    model_scores = per_df[per_df["model"] == best_model].copy()

    if model_scores.empty:
        print("  ⚠ No scores available for best/worst comparison")
        return

    # Build a lookup from full_translations
    trans_lookup = {}
    for e in full_translations:
        key = (e.get("edcs_id"), e.get("language"))
        trans_lookup[key] = e

    # Filter to entries that have all models valid
    valid_scores = []
    for _, row in model_scores.iterrows():
        key = (row["edcs_id"], row["language"])
        e = trans_lookup.get(key)
        if e and all(
            e.get(f"llm_{m}") and not str(e.get(f"llm_{m}", "")).startswith("ERROR")
            for m in models
        ):
            valid_scores.append(row)

    if len(valid_scores) < 2:
        print("  ⚠ Not enough valid entries for best/worst comparison")
        return

    valid_scores_df = pd.DataFrame(valid_scores)

    # Pick best and worst by sentence BLEU (from Claude)
    best_row = valid_scores_df.loc[valid_scores_df["sentence_bleu"].idxmax()]
    worst_row = valid_scores_df.loc[valid_scores_df["sentence_bleu"].idxmin()]

    best_ex = trans_lookup[(best_row["edcs_id"], best_row["language"])]
    worst_ex = trans_lookup[(worst_row["edcs_id"], worst_row["language"])]

    # Gather per-model scores for both
    def _get_scores(edcs_id, lang):
        info = {}
        for m in models:
            row = per_df[(per_df["model"] == m) & (per_df["edcs_id"] == edcs_id) & (per_df["language"] == lang)]
            if len(row):
                info[m] = {
                    "bleu": f"{row['sentence_bleu'].values[0]:.1f}",
                    "chrf": f"{row['sentence_chrf'].values[0]:.1f}",
                    "bert": f"{row['bertscore_f1'].values[0]:.3f}" if "bertscore_f1" in row.columns else "—",
                }
        return info

    best_scores = _get_scores(best_row["edcs_id"], best_row["language"])
    worst_scores = _get_scores(worst_row["edcs_id"], worst_row["language"])

    # ── Draw side-by-side ──
    fig, (ax_good, ax_bad) = plt.subplots(1, 2, figsize=(24, 9.5))
    fig.patch.set_facecolor("#FAFBFC")

    fig.suptitle(
        "Best vs. Worst Translation Quality (by BLEU)",
        fontsize=22, fontweight="bold", y=1.01, color="#1a1a2e",
    )

    # Green header for best
    ax_good.set_title(
        f"✓  Best — BLEU {best_row['sentence_bleu']:.1f}",
        fontsize=17, fontweight="bold", color="#2D7D2D", pad=15,
    )
    _draw_translation_card(ax_good, best_ex, show_scores=True, score_info=best_scores)

    # Red header for worst
    ax_bad.set_title(
        f"✗  Worst — BLEU {worst_row['sentence_bleu']:.1f}",
        fontsize=17, fontweight="bold", color="#B91C1C", pad=15,
    )
    _draw_translation_card(ax_bad, worst_ex, show_scores=True, score_info=worst_scores)

    fig.tight_layout()

    path = os.path.join(VIS_DIR, "best_vs_worst.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    fig.savefig(path.replace(".png", ".pdf"), facecolor=fig.get_facecolor())
    print(f"  → {path}")
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main():
    setup_style()
    os.makedirs(VIS_DIR, exist_ok=True)

    # Load data
    agg_df = pd.read_csv(os.path.join(RESULTS_DIR, "aggregated_scores.csv"))
    per_df = pd.read_csv(os.path.join(RESULTS_DIR, "per_inscription_scores.csv"))
    with open(os.path.join(RESULTS_DIR, "full_translations.json"), "r", encoding="utf-8") as f:
        full_trans = json.load(f)

    print("Generating visualizations…")

    # 1. Grouped bar chart
    print("\n[1/5] Metrics comparison bars")
    plot_grouped_bars(agg_df)

    # 2. Heatmap
    print("[2/5] Score heatmap")
    plot_heatmap(agg_df)

    # 3. Distributions
    print("[3/5] Score distributions")
    plot_distributions(per_df)

    # 4. One example per language
    print("[4/5] Per-language example cards")
    plot_per_language_examples(full_trans, per_df)

    # 5. Best vs worst side-by-side
    print("[5/5] Best vs. worst comparison")
    plot_best_worst_comparison(full_trans, per_df)

    print(f"\n✓ All visualizations saved to {VIS_DIR}/")


if __name__ == "__main__":
    main()
