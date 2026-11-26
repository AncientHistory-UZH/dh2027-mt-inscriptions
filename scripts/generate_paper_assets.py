#!/usr/bin/env python3
r"""
Generate LaTeX tables and publication-quality figures for the DH2027 paper.

Outputs:
  paper_assets/tables/   -- LaTeX .tex files (standalone, can be \input{} in paper)
  paper_assets/figures/  -- PDF/PNG figures (modern style)
  paper_assets/data/     -- Clean CSVs for repository publication

Usage:
    python mt/generate_paper_assets.py
"""

import csv
import json
import os
import shutil
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "testset")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
AGG_CSV = os.path.join(RESULTS_DIR, "aggregated_scores.csv")
PER_CSV = os.path.join(RESULTS_DIR, "per_inscription_scores.csv")
STATS_JSON = os.path.join(RESULTS_DIR, "testset_statistics.json")
TRANS_CSV = os.path.join(DATA_DIR, "human-translations-testset.csv")
FILTERED_CSV = os.path.join(DATA_DIR, "filtered-testset.csv")

OUT_DIR = os.path.join(BASE_DIR, "mt", "paper_assets")
TAB_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
PUB_DIR = os.path.join(OUT_DIR, "data")

for d in [TAB_DIR, FIG_DIR, PUB_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Color Palette ────────────────────────────────────────────────────────────
C = {
    "de":    "#2563EB",  # strong blue
    "en":    "#DC2626",  # strong red
    "fr":    "#059669",  # teal-green
    "comm":  "#4F46E5",  # indigo
    "oss":   "#DB2777",  # pink
    "human": "#2563EB",
    "mach":  "#EA580C",  # warm orange
    "text":  "#111827",
    "muted": "#9CA3AF",
    "line":  "#D1D5DB",
}

# ── Global rcParams: NO seaborn, NO whitegrid ────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": 600,
    "axes.titlepad": 10,
    "axes.labelsize": 8.5,
    "axes.labelweight": 500,
    "axes.labelcolor": C["text"],
    "axes.edgecolor": "none",
    "axes.facecolor": "white",
    "axes.linewidth": 0,
    "axes.grid": False,           # OFF by default – we add it manually
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.labelsize": 7.5,
    "xtick.color": C["muted"],
    "xtick.major.size": 0,
    "ytick.labelsize": 8,
    "ytick.color": C["text"],
    "ytick.major.size": 0,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "figure.facecolor": "white",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "savefig.facecolor": "white",
})

# ── Model metadata ───────────────────────────────────────────────────────────
MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-sonnet-4-6": "Claude Sonnet 4",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gpt-5": "GPT-5",
    "gpt-5-nano-2025-08-07": "GPT-5 Nano",
    "deepseek-v3": "DeepSeek V3",
    "llama-3.1-70b": "Llama 3.1 70B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "qwen-2.5-72b": "Qwen 2.5 72B",
    "apertus-70b": "Apertus 70B",
    "apertus-8b": "Apertus 8B",
}

MODEL_CATEGORIES = {
    "gemini-2.5-flash": "Commercial",
    "gemini-3.5-flash": "Commercial",
    "claude-sonnet-4-6": "Commercial",
    "claude-sonnet-5": "Commercial",
    "gpt-5": "Commercial",
    "gpt-5-nano-2025-08-07": "Commercial",
    "deepseek-v3": "Open-source",
    "llama-3.1-70b": "Open-source",
    "llama-3.1-8b": "Open-source",
    "qwen-2.5-72b": "Open-source",
    "apertus-70b": "Open-source",
    "apertus-8b": "Open-source",
}

MODEL_ORDER = [
    "claude-sonnet-5", "claude-sonnet-4-6", "gemini-2.5-flash",
    "gemini-3.5-flash", "gpt-5", "gpt-5-nano-2025-08-07",
    "deepseek-v3", "llama-3.1-70b", "llama-3.1-8b",
    "qwen-2.5-72b", "apertus-70b", "apertus-8b",
]

LANG_ORDER = ["DE", "EN", "FR"]
LANG_NAMES = {"DE": "German", "EN": "English", "FR": "French"}

# ── Load data ────────────────────────────────────────────────────────────────
def load_agg():
    with open(AGG_CSV) as f:
        return list(csv.DictReader(f))

def load_stats():
    with open(STATS_JSON) as f:
        return json.load(f)

def load_translations():
    with open(TRANS_CSV) as f:
        return list(csv.DictReader(f))

def compute_line_stats():
    """Compute lines-per-inscription stats (lines separated by /)."""
    counts = []
    with open(FILTERED_CSV) as f:
        for row in csv.DictReader(f):
            counts.append(row["text"].count("/") + 1)
    return {
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "min": min(counts),
        "max": max(counts),
        "std": statistics.stdev(counts),
    }

agg_rows = load_agg()
stats = load_stats()


def get(model, lang, ref_source, metric):
    for r in agg_rows:
        if r["model"] == model and r["language"] == lang and r["ref_source"] == ref_source:
            return float(r[metric])
    return None


def get_avg(model, ref_source, metric, scale=1.0):
    """Average across languages."""
    vals = [get(model, lang, ref_source, metric) for lang in LANG_ORDER]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals) * scale) if vals else None


# ── Ranking helpers ──────────────────────────────────────────────────────────
def rank_models_by_bleu(ref="human"):
    """Return MODEL_ORDER sorted by avg BLEU descending."""
    scored = [(m, get_avg(m, ref, "corpus_bleu") or 0) for m in MODEL_ORDER]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored]


# ══════════════════════════════════════════════════════════════════════════════
#  TABLES
# ══════════════════════════════════════════════════════════════════════════════

def table_corpus():
    trans = load_translations()
    human_counts, gemini_counts = {}, {}
    for row in trans:
        lang = row["Sprache"]
        if row["ref_source"] == "human":
            human_counts[lang] = human_counts.get(lang, 0) + 1
        else:
            gemini_counts[lang] = gemini_counts.get(lang, 0) + 1

    src = stats["latin_source_overall"]
    line_stats = compute_line_stats()

    lines = []
    lines.append(r"% Auto-generated – do not edit")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Test corpus statistics. All inscriptions are drawn from the EDCS.}")
    lines.append(r"\label{tab:corpus}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Property} & \textbf{Value} \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Latin source texts}} \\")
    lines.append(f"Unique inscriptions & {src['unique_texts']:,} \\\\")
    lines.append(f"Total words & {src['total_words']:,} \\\\")
    lines.append(f"Mean words/inscription & {src['mean_words']:.1f} ($\\sigma$={src['std_words']:.1f}) \\\\")
    lines.append(f"Median words & {src['median_words']:.0f} \\\\")
    lines.append(f"Word range (min--max) & {src['min_words']}--{src['max_words']} \\\\")
    lines.append(f"Mean lines/inscription & {line_stats['mean']:.1f} ($\\sigma$={line_stats['std']:.1f}) \\\\")
    lines.append(f"Median lines & {line_stats['median']:.0f} \\\\")
    lines.append(f"Line range (min--max) & {line_stats['min']}--{line_stats['max']} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Reference translations}} \\")
    lines.append(f"Total reference pairs & {sum(human_counts.values()) + sum(gemini_counts.values())} \\\\")
    for lang in LANG_ORDER:
        h = human_counts.get(lang, 0)
        g = gemini_counts.get(lang, 0)
        lines.append(f"\\quad {LANG_NAMES[lang]} (human / machine) & {h} / {g} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Reference word counts (mean)}} \\")
    for lang in LANG_ORDER:
        ldata = stats["per_language"][lang]["reference_word_count"]
        lines.append(f"\\quad {LANG_NAMES[lang]} & {ldata['mean']:.1f} ($\\sigma$={ldata['std']:.1f}) \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_categories():
    """Table of inscription categories (EDCS gattung_list) in the test set."""
    # ── Read and explode multi-label categories ──
    with open(FILTERED_CSV) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    no_cat = sum(1 for r in rows
                 if not r.get("gattung_list", "").strip()
                 or r.get("gattung_list", "").strip() == "NULL")

    from collections import Counter
    cat_counter = Counter()
    for r in rows:
        gl = r.get("gattung_list", "").strip()
        if gl and gl != "NULL":
            for c in gl.split(", "):
                cat_counter[c.strip()] += 1

    # ── Group into functional (inscription types) vs. prosopographic ──
    FUNCTIONAL = [
        ("tituli sepulcrales",        "Funerary"),
        ("tituli sacri",              "Votive / sacred"),
        ("tituli operum",             "Building"),
        ("tituli honorarii",          "Honorific"),
        ("carmina",                   "Verse (\\textit{carmina})"),
        ("inscriptiones christianae", "Christian"),
        ("defixiones",                "Curse tablets"),
        ("tituli possessionis",       "Ownership"),
        ("termini",                   "Boundary stones"),
        ("signacula medicorum",       "Physician stamps"),
    ]
    PROSOPOGRAPHIC = [
        ("viri",               "Men mentioned"),
        ("mulieres",           "Women mentioned"),
        ("tria nomina",        "\\textit{Tria nomina}"),
        ("nomen singulare",    "Single name"),
        ("milites",            "Military"),
        ("liberti/libertae",   "Freed persons"),
        ("officium/professio", "Profession / office"),
        ("Augusti/Augustae",   "Emperors / empresses"),
        ("sacerdotes pagani",  "Pagan priests"),
        ("ordo equester",      "Equestrian order"),
        ("ordo decurionum",    "Decurional order"),
        ("ordo senatorius",    "Senatorial order"),
        ("seviri Augustales",  "\\textit{Seviri Augustales}"),
        ("servi/servae",       "Enslaved persons"),
        ("reges",              "Kings"),
        ("praenomen et nomen", "\\textit{Praenomen} and \\textit{nomen}"),
        ("litterae erasae",    "Erased letters"),
    ]

    lines = []
    lines.append(r"% Auto-generated – do not edit")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inscription categories (EDCS \textit{Gattung}) in the test set. "
                 r"Categories are multi-label; percentages relative to "
                 f"{total} inscriptions.}}")
    lines.append(r"\label{tab:categories}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{$n$} & \textbf{\%} \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{\textit{Inscription type}} \\")
    for key, label in FUNCTIONAL:
        n = cat_counter.get(key, 0)
        if n > 0:
            pct = n / total * 100
            lines.append(f"\\quad {label} & {n} & {pct:.1f} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{\textit{Prosopographic tags}} \\")
    for key, label in PROSOPOGRAPHIC:
        n = cat_counter.get(key, 0)
        if n > 0:
            pct = n / total * 100
            lines.append(f"\\quad {label} & {n} & {pct:.1f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"No category assigned & {no_cat} & {no_cat / total * 100:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_models():
    lines = []
    lines.append(r"% Auto-generated – do not edit")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LLMs evaluated in this study. Proprietary models were accessed via their respective APIs; open-source models via the HuggingFace Serverless Inference API.}")
    lines.append(r"\label{tab:models}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Provider} & \textbf{API / HF Identifier} \\")
    lines.append(r"\midrule")
    model_info = [
        ("Claude Sonnet 5",   "Anthropic",  "claude-sonnet-5",                       "anthropic2026claudesonnet5"),
        ("Claude Sonnet 4",   "Anthropic",  "claude-sonnet-4-6",                     "anthropic2025claude4"),
        ("Gemini 2.5 Flash",  "Google",     "gemini-2.5-flash",                      "geminiteam2025gemini25"),
        ("Gemini 3.5 Flash",  "Google",     "gemini-3.5-flash",                      "google2026gemini35flash"),
        ("GPT-5",             "OpenAI",     "gpt-5",                                 "openai2025gpt5"),
        ("GPT-5 Nano",        "OpenAI",     "gpt-5-nano-2025-08-07",                 "openai2025gpt5"),
        ("DeepSeek V3",       "DeepSeek",   "deepseek-ai/DeepSeek-V3-0324",          "deepseekai2024deepseekv3"),
        ("Llama 3.1 70B",     "Meta",       "meta-llama/Llama-3.1-70B-Instruct",     "meta2024llama3"),
        ("Llama 3.1 8B",      "Meta",       "meta-llama/Llama-3.1-8B-Instruct",      "meta2024llama3"),
        ("Qwen 2.5 72B",      "Alibaba",    "Qwen/Qwen2.5-72B-Instruct",            "qwen2024qwen25"),
        ("Apertus 70B",       "Swiss AI",   "swiss-ai/Apertus-70B-Instruct-2509",    "projectapertus2025apertus"),
        ("Apertus 8B",        "Swiss AI",   "swiss-ai/Apertus-8B-Instruct-2509",     "projectapertus2025apertus"),
    ]
    prev_type = None
    comm_models = model_info[:6]
    oss_models = model_info[6:]
    for name, provider, ident, cite in comm_models:
        lines.append(f"{name} \\cite{{{cite}}} & {provider} & \\texttt{{{ident}}} \\\\")
    lines.append(r"\midrule")
    for name, provider, ident, cite in oss_models:
        # Escape slashes in HF IDs for better line breaking
        ident_tex = ident.replace("/", "/\\allowbreak ")
        lines.append(f"{name} \\cite{{{cite}}} & {provider} & \\texttt{{{ident_tex}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def table_main_results():
    metrics = [
        ("corpus_bleu", "BLEU", 1),
        ("corpus_chrf", "chrF", 1),
        ("bertscore_f1", "BERTScore", 100),
    ]
    ref = "human"
    ranked = rank_models_by_bleu(ref)

    # Find best/second per metric×lang
    bests = {}
    for mk, _, scale in metrics:
        for lang in LANG_ORDER:
            vals = [(m, get(m, lang, ref, mk)) for m in ranked]
            vals = [(m, v) for m, v in vals if v is not None]
            vals.sort(key=lambda x: x[1], reverse=True)
            bests[(mk, lang)] = (vals[0][0] if vals else None, vals[1][0] if len(vals) > 1 else None)

    lines = []
    lines.append(r"% Auto-generated – do not edit")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Translation quality against \textbf{human} references, ranked by average BLEU. Best per language in \textbf{bold}, second-best \underline{underlined}. BERTScore shown as F1$\times$100.}")
    lines.append(r"\label{tab:results-human}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "rrr" * 3 + r"r}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{\textbf{BLEU} $\uparrow$} & \multicolumn{3}{c}{\textbf{chrF} $\uparrow$} & \multicolumn{3}{c}{\textbf{BERTScore} $\uparrow$} & \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    lines.append(r"\textbf{Model} & DE & EN & FR & DE & EN & FR & DE & EN & FR & $\overline{x}$ \\")
    lines.append(r"\midrule")

    prev_cat = None
    for model in ranked:
        cat = MODEL_CATEGORIES[model]
        if prev_cat == "Commercial" and cat == "Open-source":
            lines.append(r"\midrule")
        prev_cat = cat

        label = MODEL_LABELS[model]
        marker = "" if cat == "Commercial" else r"$^{\dagger}$"
        cells = []
        total, count = 0, 0
        for mk, _, scale in metrics:
            for lang in LANG_ORDER:
                v = get(model, lang, ref, mk)
                if v is None:
                    cells.append("--")
                    continue
                dv = v * scale
                best, second = bests[(mk, lang)]
                fmt = f"{dv:.1f}"
                if model == best:
                    fmt = r"\textbf{" + fmt + "}"
                elif model == second:
                    fmt = r"\underline{" + fmt + "}"
                cells.append(fmt)
                total += dv
                count += 1
        avg = total / count if count else 0
        cells.append(f"{avg:.1f}")
        lines.append(f"{label}{marker} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{11}{l}{\scriptsize $^{\dagger}$Open-source model} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def table_extended_metrics():
    ref = "human"
    ranked = rank_models_by_bleu(ref)
    metrics = [("corpus_meteor", "METEOR", True), ("corpus_ter", "TER", False)]

    bests = {}
    for mk, _, higher in metrics:
        for lang in LANG_ORDER:
            vals = [(m, get(m, lang, ref, mk)) for m in ranked]
            vals = [(m, v) for m, v in vals if v is not None]
            vals.sort(key=lambda x: x[1], reverse=higher)
            bests[(mk, lang)] = vals[0][0] if vals else None

    lines = []
    lines.append(r"% Auto-generated – do not edit")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{METEOR and TER against \textbf{human} references. Best per language in \textbf{bold}.}")
    lines.append(r"\label{tab:results-extended}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "rrr" * 2 + r"}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{\textbf{METEOR} $\uparrow$} & \multicolumn{3}{c}{\textbf{TER} $\downarrow$} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"\textbf{Model} & DE & EN & FR & DE & EN & FR \\")
    lines.append(r"\midrule")

    prev_cat = None
    for model in ranked:
        cat = MODEL_CATEGORIES[model]
        if prev_cat == "Commercial" and cat == "Open-source":
            lines.append(r"\midrule")
        prev_cat = cat

        label = MODEL_LABELS[model]
        marker = "" if cat == "Commercial" else r"$^{\dagger}$"
        cells = []
        for mk, _, _ in metrics:
            for lang in LANG_ORDER:
                v = get(model, lang, ref, mk)
                if v is None:
                    cells.append("--")
                    continue
                fmt = f"{v:.1f}"
                if model == bests[(mk, lang)]:
                    fmt = r"\textbf{" + fmt + "}"
                cells.append(fmt)
        lines.append(f"{label}{marker} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{7}{l}{\scriptsize $^{\\dagger}$Open-source model} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def table_ranking():
    ref = "human"
    metrics = ["corpus_bleu", "corpus_chrf", "bertscore_f1", "corpus_meteor", "corpus_ter"]
    metric_names = ["BLEU", "chrF", "BERTScore", "METEOR", "TER"]
    scales = [1, 1, 100, 1, 1]

    rows = []
    for model in MODEL_ORDER:
        avgs = {}
        for mk, mn, sc in zip(metrics, metric_names, scales):
            v = get_avg(model, ref, mk, sc)
            if v is not None:
                avgs[mn] = v
        rows.append((model, avgs))
    rows.sort(key=lambda x: x[1].get("BLEU", 0), reverse=True)

    lines = []
    lines.append(r"% Auto-generated – do not edit")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall ranking averaged across DE/EN/FR (human references), sorted by BLEU.}")
    lines.append(r"\label{tab:ranking}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{clrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{\#} & \textbf{Model} & \textbf{BLEU}$\uparrow$ & \textbf{chrF}$\uparrow$ & \textbf{BERT}$\uparrow$ & \textbf{MET.}$\uparrow$ & \textbf{TER}$\downarrow$ \\")
    lines.append(r"\midrule")
    for rank, (model, avgs) in enumerate(rows, 1):
        label = MODEL_LABELS[model]
        cat = MODEL_CATEGORIES[model]
        marker = "" if cat == "Commercial" else r"$^{\dagger}$"
        lines.append(
            f"{rank} & {label}{marker} & "
            f"{avgs.get('BLEU', 0):.1f} & "
            f"{avgs.get('chrF', 0):.1f} & "
            f"{avgs.get('BERTScore', 0):.1f} & "
            f"{avgs.get('METEOR', 0):.1f} & "
            f"{avgs.get('TER', 0):.1f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{7}{l}{\scriptsize $^{\dagger}$Open-source model} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)


def _hgrid(ax, alpha=0.25, lw=0.4, color=C["line"]):
    """Faint horizontal reference lines only."""
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(axis="x", color=color, linewidth=lw, alpha=alpha)
    ax.grid(axis="y", visible=False)


def _vgrid(ax, alpha=0.25, lw=0.4, color=C["line"]):
    """Faint vertical reference lines only."""
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(axis="y", color=color, linewidth=lw, alpha=alpha)
    ax.grid(axis="x", visible=False)


LANG_COLORS = [C["de"], C["en"], C["fr"]]


# ── Figure 1: All metrics – multi-panel horizontal bars ─────────────────────
def fig_all_metrics_bars():
    ref = "human"
    metrics = [
        ("corpus_bleu",   "BLEU ↑",      1),
        ("corpus_chrf",   "chrF ↑",      1),
        ("bertscore_f1",  "BERTScore ↑", 100),
        ("corpus_meteor", "METEOR ↑",    1),
        ("corpus_ter",    "TER ↓",       1),
    ]
    ranked = rank_models_by_bleu(ref)
    labels = [MODEL_LABELS[m] for m in ranked]
    n = len(ranked)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.6), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    y = np.arange(n)
    bh = 0.24  # bar height

    for ax, (mk, title, scale) in zip(axes, metrics):
        for i, lang in enumerate(LANG_ORDER):
            vals = [(get(m, lang, ref, mk) or 0) * scale for m in ranked]
            ax.barh(y + (i - 1) * bh, vals, bh,
                    color=LANG_COLORS[i], alpha=0.88,
                    label=LANG_NAMES[lang] if ax is axes[0] else None,
                    zorder=3)

        ax.set_title(title, pad=6, color=C["text"])
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=7.8, color=C["text"])
        ax.invert_yaxis()
        _hgrid(ax)

    axes[0].legend(loc="lower right", fontsize=7.5,
                   handlelength=1.2, borderpad=0.4, labelspacing=0.35)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06, wspace=0.18)
    _save(fig, "all_metrics_by_model")


# ── Figure 2: Heatmap ───────────────────────────────────────────────────────
def fig_heatmap():
    ref = "human"
    metrics = ["corpus_bleu", "corpus_chrf", "bertscore_f1",
               "corpus_meteor", "corpus_ter"]
    col_labels = ["BLEU ↑", "chrF ↑", "BERT ↑", "METEOR ↑", "TER ↓"]
    scales = [1, 1, 100, 1, 1]

    ranked = rank_models_by_bleu(ref)
    row_labels = [MODEL_LABELS[m] for m in ranked]
    nrows, ncols = len(ranked), len(metrics)

    # Raw values
    data = np.zeros((nrows, ncols))
    for i, m in enumerate(ranked):
        for j, (mk, sc) in enumerate(zip(metrics, scales)):
            v = get_avg(m, ref, mk, sc)
            data[i, j] = v if v is not None else 0

    # Column-wise normalize for color
    norm = data.copy()
    for j in range(ncols):
        lo, hi = norm[:, j].min(), norm[:, j].max()
        rng = hi - lo if hi - lo > 1e-9 else 1
        if j == 4:  # TER: invert
            norm[:, j] = 1 - (norm[:, j] - lo) / rng
        else:
            norm[:, j] = (norm[:, j] - lo) / rng

    # Build a smooth diverging green-yellow-red colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gyr", ["#DC2626", "#FBBF24", "#16A34A"], N=256)

    cell_w, cell_h = 1.05, 0.52
    fig_w = 1.8 + ncols * cell_w      # left margin for labels
    fig_h = 0.7 + nrows * cell_h       # top margin for col labels
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1,
              interpolation="nearest")

    # Cell separators – thin white gaps
    for i in range(nrows + 1):
        ax.axhline(i - 0.5, color="white", lw=2.2)
    for j in range(ncols + 1):
        ax.axvline(j - 0.5, color="white", lw=2.2)

    # Text annotations
    for i in range(nrows):
        for j in range(ncols):
            nv = norm[i, j]
            txt_col = "white" if nv < 0.3 or nv > 0.75 else C["text"]
            ax.text(j, i, f"{data[i, j]:.1f}",
                    ha="center", va="center", fontsize=8,
                    color=txt_col, fontweight=600)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, fontsize=8.5, fontweight=600, color=C["text"])
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=8, color=C["text"])
    ax.tick_params(length=0)
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.4)
    _save(fig, "metric_heatmap")


# ── Figure 3: Human vs. Machine refs – dumbbell chart ───────────────────────
def fig_human_vs_machine_refs():
    metrics = [
        ("corpus_bleu",  "BLEU",              1),
        ("corpus_chrf",  "chrF",              1),
        ("bertscore_f1", "BERTScore (×100)",  100),
    ]
    ranked = rank_models_by_bleu("human")
    labels = [MODEL_LABELS[m] for m in ranked]
    n = len(ranked)
    y = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True,
                             gridspec_kw={"wspace": 0.10})

    for ax, (mk, title, scale) in zip(axes, metrics):
        hv = np.array([(get_avg(m, "human", mk, scale) or 0) for m in ranked])
        gv = np.array([(get_avg(m, "gemini", mk, scale) or 0) for m in ranked])

        # Connecting segments
        for i in range(n):
            ax.plot([hv[i], gv[i]], [y[i], y[i]],
                    color=C["line"], lw=1.8, solid_capstyle="round", zorder=1)

        ax.scatter(hv, y, s=52, color=C["human"], zorder=4,
                   edgecolors="white", linewidths=0.6,
                   label="Human refs" if ax is axes[0] else None)
        ax.scatter(gv, y, s=52, color=C["mach"], zorder=4, marker="D",
                   edgecolors="white", linewidths=0.6,
                   label="Machine refs" if ax is axes[0] else None)

        ax.set_title(title, pad=6, color=C["text"])
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=7.8, color=C["text"])
        ax.invert_yaxis()
        _hgrid(ax)

    axes[0].legend(loc="lower right", fontsize=7.5, markerscale=0.85,
                   handletextpad=0.3, borderpad=0.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06, wspace=0.10)
    _save(fig, "human_vs_machine_refs")


# ── Figure 4: BLEU vs BERTScore scatter ─────────────────────────────────────
def fig_scatter():
    ref = "human"
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mkrs = {"DE": "o", "EN": "s", "FR": "^"}
    msz  = {"DE": 42, "EN": 42, "FR": 48}

    plotted = {}  # for annotation de-conflict
    for model in MODEL_ORDER:
        cc = C["comm"] if MODEL_CATEGORIES[model] == "Commercial" else C["oss"]
        for lang in LANG_ORDER:
            bleu = get(model, lang, ref, "corpus_bleu")
            bert = get(model, lang, ref, "bertscore_f1")
            if bleu is None or bert is None:
                continue
            ax.scatter(bleu, bert * 100, marker=mkrs[lang], color=cc,
                       s=msz[lang], alpha=0.72, edgecolors="white",
                       linewidth=0.6, zorder=3)
            plotted[(model, lang)] = (bleu, bert * 100)

    # Annotate a few interesting models (EN point)
    annots = {
        "deepseek-v3":   (8, 5),
        "claude-sonnet-5": (-75, -12),
        "qwen-2.5-72b":  (-60, 6),
    }
    for model, (dx, dy) in annots.items():
        pt = plotted.get((model, "EN"))
        if pt:
            ax.annotate(MODEL_LABELS[model], pt,
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.5, color=C["muted"],
                        arrowprops=dict(arrowstyle="-", color=C["line"],
                                        lw=0.5) if abs(dx) > 10 else None)

    legend_el = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#777", label="DE", markersize=5.5),
        Line2D([], [], marker="s", color="w", markerfacecolor="#777", label="EN", markersize=5.5),
        Line2D([], [], marker="^", color="w", markerfacecolor="#777", label="FR", markersize=5.5),
        Line2D([], [], marker="o", color="w", markerfacecolor=C["comm"], label="Commercial", markersize=5.5),
        Line2D([], [], marker="o", color="w", markerfacecolor=C["oss"],  label="Open-source", markersize=5.5),
    ]
    ax.legend(handles=legend_el, fontsize=7, loc="lower right",
              handletextpad=0.3, borderpad=0.5, labelspacing=0.3)
    ax.set_xlabel("Corpus BLEU", color=C["text"])
    ax.set_ylabel("BERTScore F1 (×100)", color=C["text"])
    _vgrid(ax)
    ax.grid(axis="x", color=C["line"], linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    _save(fig, "bleu_vs_bertscore")


# ── Figure 5: Length ratio ──────────────────────────────────────────────────
def fig_length_ratio():
    ref = "human"
    ranked = rank_models_by_bleu(ref)
    labels = [MODEL_LABELS[m] for m in ranked]
    n = len(ranked)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    y = np.arange(n)
    bh = 0.24

    for i, lang in enumerate(LANG_ORDER):
        vals = [(get(m, lang, ref, "length_ratio") or 1) for m in ranked]
        ax.barh(y + (i - 1) * bh, vals, bh,
                color=LANG_COLORS[i], alpha=0.88,
                label=LANG_NAMES[lang], zorder=3)

    ax.axvline(1.0, color=C["text"], ls="--", lw=0.7, alpha=0.45, zorder=2)
    ax.set_xlabel("Length ratio  (hypothesis / reference)", color=C["text"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8, color=C["text"])
    ax.invert_yaxis()
    ax.legend(fontsize=7.5, handlelength=1.2, borderpad=0.4,
              labelspacing=0.35, loc="lower right")
    _hgrid(ax)
    ax.set_xlim(0.65, 1.38)
    fig.tight_layout()
    _save(fig, "length_ratio")


# ══════════════════════════════════════════════════════════════════════════════
#  Publication data
# ══════════════════════════════════════════════════════════════════════════════
def prepare_pub_data():
    shutil.copy2(AGG_CSV, os.path.join(PUB_DIR, "aggregated_scores.csv"))
    shutil.copy2(PER_CSV, os.path.join(PUB_DIR, "per_inscription_scores.csv"))
    shutil.copy2(STATS_JSON, os.path.join(PUB_DIR, "testset_statistics.json"))

    # Testset metadata
    with open(FILTERED_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = ["edcs_id", "text", "datierung", "gattung_list"]
        with open(os.path.join(PUB_DIR, "testset_inscriptions.csv"), "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Reference summary (counts only)
    trans = load_translations()
    counts = {}
    for row in trans:
        key = f"{row['Sprache']}_{row['ref_source']}"
        counts[key] = counts.get(key, 0) + 1
    with open(os.path.join(PUB_DIR, "reference_translation_summary.json"), "w") as f:
        json.dump({
            "description": "Reference translation counts per language and source type",
            "note": "Full translations available upon request",
            "counts": counts,
        }, f, indent=2)

    readme = """# Vox Lapidum — MT Evaluation Data

## LLM Translation of Latin Epigraphic Inscriptions

This dataset accompanies the DH2027 paper.

### Files

| File | Description |
|------|-------------|
| `aggregated_scores.csv` | Corpus-level scores: 12 models × 3 languages × 3 reference tiers |
| `per_inscription_scores.csv` | Per-inscription sentence-level scores (21,960 rows) |
| `testset_inscriptions.csv` | 305 test inscriptions (EDCS-ID, Latin text, dating, genre) |
| `testset_statistics.json` | Corpus statistics |
| `reference_translation_summary.json` | Reference translation counts per language/type |

### Metrics

| Metric | Tool | Direction |
|--------|------|-----------|
| BLEU | sacrebleu | ↑ higher = better |
| chrF | sacrebleu | ↑ higher = better |
| BERTScore | bert_score | ↑ higher = better |
| METEOR | nltk | ↑ higher = better |
| TER | sacrebleu | ↓ lower = better |

### Reference Tiers

- **human** — expert human translations (~100 per language)
- **gemini** — machine-generated references via Google Gemini (~200 per language)
- **all** — combined

### Models

6 commercial + 6 open-source LLMs (see paper for details).

### License

Data: CC BY 4.0. Inscription texts from EDCS.
"""
    with open(os.path.join(PUB_DIR, "README.md"), "w") as f:
        f.write(readme)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Generating paper assets (modern style)...\n")

    tables = {
        "tab_corpus.tex": table_corpus(),
        "tab_models.tex": table_models(),
        "tab_results_main.tex": table_main_results(),
        "tab_results_extended.tex": table_extended_metrics(),
        "tab_ranking.tex": table_ranking(),
        "tab_categories.tex": table_categories(),
    }
    for fname, content in tables.items():
        path = os.path.join(TAB_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        print(f"  ✓ tables/{fname}")

    print()
    fig_all_metrics_bars()
    print("  ✓ figures/all_metrics_by_model.pdf")
    fig_heatmap()
    print("  ✓ figures/metric_heatmap.pdf")
    fig_human_vs_machine_refs()
    print("  ✓ figures/human_vs_machine_refs.pdf")
    fig_scatter()
    print("  ✓ figures/bleu_vs_bertscore.pdf")
    fig_length_ratio()
    print("  ✓ figures/length_ratio.pdf")

    print()
    prepare_pub_data()
    print("  ✓ data/")

    print(f"\nDone! → {OUT_DIR}/")


if __name__ == "__main__":
    main()
