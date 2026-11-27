#!/usr/bin/env python3
"""
Recompute BERTScore with a unified multilingual model (xlm-roberta-large).
==========================================================================

The original evaluate_testset.py passed ``lang=...`` to bert_score, which
selects *different* backbone models per language:

  - EN  → roberta-large
  - DE  → bert-base-multilingual-cased
  - FR  → bert-base-multilingual-cased

This makes BERTScore values incomparable across languages.  This script
re-scores all (model × language × ref_source) combinations using a single
backbone — ``xlm-roberta-large`` — so the numbers are comparable.

No API calls needed: it reads translations from the existing cache and
references from the testset CSVs.

After running, regenerate paper assets with:

    python mt/generate_paper_assets.py

Usage:
    python mt/recompute_bertscore.py          # from repo root
    python mt/recompute_bertscore.py --dry-run # preview without writing
"""

import argparse
import csv
import json
import os
import sys
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

BERTSCORE_MODEL = "xlm-roberta-large"

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "testset")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

CACHE_PATH = os.path.join(RESULTS_DIR, "_translation_cache.json")
HUMAN_TRANS = os.path.join(DATA_DIR, "human-translations-testset.csv")
FILTERED_CSV = os.path.join(DATA_DIR, "filtered-testset.csv")
AGG_CSV = os.path.join(RESULTS_DIR, "aggregated_scores.csv")
PER_CSV = os.path.join(RESULTS_DIR, "per_inscription_scores.csv")


def load_references():
    """Load human/gemini reference translations, returning a DataFrame."""
    df = pd.read_csv(HUMAN_TRANS)
    # Normalise EDCS IDs
    df["edcs_id_raw"] = df["edcs_id"].astype(str).str.replace("EDCS-", "", regex=False).astype(int)
    return df


def load_latin_texts():
    """Map edcs_id_raw → Latin text."""
    mapping = {}
    with open(FILTERED_CSV) as f:
        for row in csv.DictReader(f):
            eid = int(str(row["edcs_id"]).replace("EDCS-", ""))
            mapping[eid] = row["text"]
    return mapping


def load_translations():
    """Load the translation cache.  Returns {model: {(edcs_id_str, lang): text}}."""
    with open(CACHE_PATH) as f:
        raw = json.load(f)
    translations = {}
    for model, pairs in raw.items():
        trans = {}
        for key_str, text in pairs.items():
            if "|" in key_str:
                # Format: "17301017|DE"
                eid, lang = key_str.split("|", 1)
                trans[(eid.strip(), lang.strip())] = text
            elif isinstance(key_str, str) and key_str.startswith("("):
                # Format: "('17301017', 'DE')"
                parts = key_str.strip("()").split(",")
                eid = parts[0].strip().strip("'\"")
                lang = parts[1].strip().strip("'\"")
                trans[(eid, lang)] = text
            else:
                trans[key_str] = text
        translations[model] = trans
    return translations


def build_pairs(ref_df, translations, model, lang, ref_source):
    """Build aligned (hypothesis, reference, edcs_id) triples."""
    if ref_source == "all":
        subset = ref_df[ref_df["Sprache"] == lang]
    else:
        subset = ref_df[(ref_df["Sprache"] == lang) & (ref_df["ref_source"] == ref_source)]

    model_trans = translations.get(model, {})
    hypotheses, references, edcs_ids = [], [], []

    for _, row in subset.iterrows():
        key = (str(int(row["edcs_id_raw"])), lang)
        hyp = model_trans.get(key)
        if hyp is None or not isinstance(hyp, str) or hyp.startswith("ERROR"):
            continue
        ref = row["Übersetzung"]
        if not isinstance(ref, str) or len(ref.strip()) == 0:
            continue
        hypotheses.append(hyp)
        references.append(ref)
        edcs_ids.append(int(row["edcs_id_raw"]))

    return hypotheses, references, edcs_ids


def main():
    parser = argparse.ArgumentParser(description="Recompute BERTScore with xlm-roberta-large")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    # ── Import bert_score ──────────────────────────────────────────────
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("ERROR: bert_score not installed.  pip install bert_score")
        return 1

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Model: {BERTSCORE_MODEL}")
    print(f"Loading references from {HUMAN_TRANS}")
    ref_df = load_references()
    print(f"Loading translations from {CACHE_PATH}")
    translations = load_translations()
    models = sorted(translations.keys())
    langs = sorted(ref_df["Sprache"].unique())
    ref_sources = ["human", "gemini", "all"]

    print(f"Models:      {', '.join(models)}")
    print(f"Languages:   {', '.join(langs)}")
    print(f"Ref sources: {', '.join(ref_sources)}")
    n_combos = len(models) * len(langs) * len(ref_sources)
    print(f"Combinations to score: {n_combos}\n")

    if args.dry_run:
        print("[DRY RUN] Would recompute all BERTScore values.  Exiting.")
        return 0

    # ── Backup existing CSVs ───────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [AGG_CSV, PER_CSV]:
        bak = f"{path}.bak_{ts}"
        shutil.copy2(path, bak)
        print(f"  Backup: {bak}")

    # ── Load existing CSVs ─────────────────────────────────────────────
    agg_df = pd.read_csv(AGG_CSV)
    per_df = pd.read_csv(PER_CSV)

    # ── Recompute ──────────────────────────────────────────────────────
    done = 0
    for model in models:
        for lang in langs:
            for ref_source in ref_sources:
                hypotheses, references, edcs_ids = build_pairs(
                    ref_df, translations, model, lang, ref_source
                )
                if not hypotheses:
                    print(f"  {model} / {lang} / {ref_source}: no pairs → skip")
                    continue

                P, R, F1 = bert_score_fn(
                    hypotheses, references,
                    model_type=BERTSCORE_MODEL,
                    verbose=False,
                )

                # Update aggregated
                mask = (
                    (agg_df["model"] == model)
                    & (agg_df["language"] == lang)
                    & (agg_df["ref_source"] == ref_source)
                )
                agg_df.loc[mask, "bertscore_precision"] = round(P.mean().item(), 4)
                agg_df.loc[mask, "bertscore_recall"] = round(R.mean().item(), 4)
                agg_df.loc[mask, "bertscore_f1"] = round(F1.mean().item(), 4)

                # Update per-inscription (only human and gemini, not "all")
                if ref_source != "all":
                    for i, eid in enumerate(edcs_ids):
                        per_mask = (
                            (per_df["model"] == model)
                            & (per_df["language"] == lang)
                            & (per_df["edcs_id"] == eid)
                            & (per_df["ref_source"] == ref_source)
                        )
                        per_df.loc[per_mask, "bertscore_f1"] = round(F1[i].item(), 4)

                done += 1
                print(f"  [{done}/{n_combos}] {model} / {lang} / {ref_source}: "
                      f"F1={F1.mean().item():.4f}  ({len(hypotheses)} pairs)")

    # ── Write updated CSVs ─────────────────────────────────────────────
    agg_df.to_csv(AGG_CSV, index=False)
    per_df.to_csv(PER_CSV, index=False)
    print(f"\n✓ Updated {AGG_CSV}")
    print(f"✓ Updated {PER_CSV}")
    print(f"\nBackups saved with suffix .bak_{ts}")
    print(f"\nNext step:  python mt/generate_paper_assets.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

