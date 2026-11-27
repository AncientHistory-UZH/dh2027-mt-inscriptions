"""
Testset Evaluation Pipeline
============================
Translates all inscriptions from the human-annotated testset into target
languages using three LLM models, then computes translation quality metrics
(BLEU, chrF, BERTScore) against the human reference translations.

Workflow
--------
1. Load human translations from ``data/testset/human-translations-testset.csv``
   (columns: ``edcs_id``, ``Übersetzung``, ``Sprache``).
2. Load original Latin texts from ``data/testset/filtered-testset.csv``
   (column ``edcs_id`` carries the ``EDCS-`` prefix; strip it to match).
3. For each inscription × target language, call each LLM model to translate
   the Latin into the target language.
4. Compare machine translation against the human reference using BLEU, chrF
   and BERTScore.
5. Persist per-inscription scores + aggregate statistics as JSON / CSV.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import re
import sys
from collections import defaultdict
from datetime import datetime

# ── Load .env file if present ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"Loaded environment from {_env_path}")
except ImportError:
    pass  # python-dotenv not installed; rely on shell env

# ── translation helpers (reuse from translate.py) ──────────────────────
from translate import translate_single_lang, MODELS


def _check_api_keys():
    """Verify that all required API keys are set before starting."""
    required = {
        "GEMINI_API_KEY": "gemini",
        "ANTHROPIC_API_KEY": "claude/sonnet",
        "OPENAI_API_KEY": "gpt",
    }
    missing = []
    for env_var, provider in required.items():
        if not os.environ.get(env_var):
            missing.append(f"  • {env_var} ({provider})")
    if missing:
        print("\n⚠  Missing API keys – translations will fail:")
        for m in missing:
            print(m)
        print("\nSet them via environment variables or create a .env file in the project root:")
        print("  GEMINI_API_KEY=your-key")
        print("  ANTHROPIC_API_KEY=your-key")
        print("  OPENAI_API_KEY=your-key")
        print()
        resp = input("Continue anyway? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(1)

# ── evaluation metrics ─────────────────────────────────────────────────
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert_score not installed. BERTScore will be skipped.")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET_DIR = os.path.join(BASE_DIR, "data", "testset")
HUMAN_TRANS_CSV = os.path.join(TESTSET_DIR, "human-translations-testset.csv")
FILTERED_TESTSET = os.path.join(TESTSET_DIR, "filtered-testset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "testset", "results")

# Map language codes to BERTScore model language hints
LANG_MAP_BERT = {"DE": "de", "EN": "en", "FR": "fr", "IT": "it", "ES": "es"}
LANG_NAMES = {"DE": "German", "EN": "English", "FR": "French", "IT": "Italian", "ES": "Spanish"}


# ======================================================================
# 1.  DATA LOADING
# ======================================================================

def load_data():
    """Load and join human translations with original Latin texts.

    Returns
    -------
    human_df : pd.DataFrame
        Columns: edcs_id_num, edcs_id_full, inschrift_id, latin_text,
                 Übersetzung, Sprache
    """
    human_df = pd.read_csv(HUMAN_TRANS_CSV)
    # Remove any trailing empty rows
    human_df = human_df.dropna(subset=["edcs_id", "Übersetzung", "Sprache"])
    # edcs_id in human file is a plain number → cast to int-like string
    human_df["edcs_id"] = human_df["edcs_id"].astype(int)

    filtered_df = pd.read_csv(FILTERED_TESTSET)
    # Build mapping: EDCS-XXXXX → numeric part
    filtered_df["edcs_id_num"] = (
        filtered_df["edcs_id"]
        .str.replace("EDCS-", "", regex=False)
        .astype(int)
    )

    # Join to get latin text + inschrift_id
    merged = human_df.merge(
        filtered_df[["edcs_id_num", "edcs_id", "inschrift_id", "text"]].drop_duplicates(subset=["edcs_id_num"]),
        left_on="edcs_id",
        right_on="edcs_id_num",
        how="left",
        suffixes=("_human", "_filtered"),
    )
    merged.rename(columns={
        "edcs_id_human": "edcs_id_raw",
        "edcs_id_filtered": "edcs_id_full",
        "text": "latin_text",
    }, inplace=True)

    missing = merged["latin_text"].isna().sum()
    if missing:
        print(f"Warning: {missing} human translations could not be matched to a Latin text.")
        merged = merged.dropna(subset=["latin_text"])

    print(f"Loaded {len(merged)} human reference translation pairs "
          f"({merged['Sprache'].nunique()} languages, "
          f"{merged['edcs_id_raw'].nunique()} unique inscriptions).")
    return merged


# ======================================================================
# 2.  CORPUS STATISTICS
# ======================================================================

def word_count(text):
    """Simple whitespace word count after removing editorial marks."""
    if not isinstance(text, str):
        return 0
    # Remove brackets and editorial symbols
    clean = re.sub(r'[\[\]\(\)\{\}<>⟦⟧/|]', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return len(clean.split()) if clean else 0


def compute_statistics(merged_df):
    """Generate comprehensive corpus statistics for the testset."""
    stats = {}

    # ── Global counts ──────────────────────────────────────────────────
    stats["total_reference_pairs"] = len(merged_df)
    stats["unique_inscriptions"] = int(merged_df["edcs_id_raw"].nunique())
    stats["languages"] = sorted(merged_df["Sprache"].unique().tolist())
    stats["num_languages"] = len(stats["languages"])

    # ── Per-language breakdown ─────────────────────────────────────────
    per_lang = {}
    for lang, grp in merged_df.groupby("Sprache"):
        ref_wc = grp["Übersetzung"].apply(word_count)
        lat_wc = grp["latin_text"].apply(word_count)
        per_lang[lang] = {
            "language_name": LANG_NAMES.get(lang, lang),
            "num_inscriptions": int(len(grp)),
            "unique_edcs_ids": int(grp["edcs_id_raw"].nunique()),
            "reference_word_count": {
                "total": int(ref_wc.sum()),
                "mean": round(float(ref_wc.mean()), 1),
                "median": round(float(ref_wc.median()), 1),
                "min": int(ref_wc.min()),
                "max": int(ref_wc.max()),
                "std": round(float(ref_wc.std()), 1),
            },
            "latin_source_word_count": {
                "total": int(lat_wc.sum()),
                "mean": round(float(lat_wc.mean()), 1),
                "median": round(float(lat_wc.median()), 1),
                "min": int(lat_wc.min()),
                "max": int(lat_wc.max()),
                "std": round(float(lat_wc.std()), 1),
            },
        }
    stats["per_language"] = per_lang

    # ── Overall Latin source stats ─────────────────────────────────────
    unique_latin = merged_df.drop_duplicates(subset=["edcs_id_raw"])
    latin_wc = unique_latin["latin_text"].apply(word_count)
    stats["latin_source_overall"] = {
        "unique_texts": int(len(unique_latin)),
        "total_words": int(latin_wc.sum()),
        "mean_words": round(float(latin_wc.mean()), 1),
        "median_words": round(float(latin_wc.median()), 1),
        "min_words": int(latin_wc.min()),
        "max_words": int(latin_wc.max()),
        "std_words": round(float(latin_wc.std()), 1),
    }

    return stats


# ======================================================================
# 3.  TRANSLATION
# ======================================================================

def translate_testset(merged_df, cache_path=None):
    """Translate every (inscription, language) pair with each model.

    Returns
    -------
    translations : dict
        {model: {(edcs_id, lang): translated_text}}
    """
    # Try to load cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached translations from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        # Convert keys back to tuples
        translations = {}
        for model, pairs in cached.items():
            translations[model] = {tuple(k.split("|")): v for k, v in pairs.items()}
        # Check if cache is complete
        all_pairs = set()
        for _, row in merged_df.iterrows():
            all_pairs.add((str(int(row["edcs_id_raw"])), row["Sprache"]))
        
        is_complete = True
        for model in MODELS:
            if model not in translations:
                is_complete = False
                break
            for pair in all_pairs:
                if pair not in translations[model]:
                    is_complete = False
                    break
        if is_complete:
            print("Cache is complete – skipping translation API calls.")
            return translations
        print("Cache incomplete – continuing with missing translations.")
    else:
        translations = {m: {} for m in MODELS}

    # Group unique (edcs_id, lang, latin_text) triplets
    tasks = []
    for _, row in merged_df.iterrows():
        key = (str(int(row["edcs_id_raw"])), row["Sprache"])
        tasks.append((key, row["latin_text"]))
    # Deduplicate (same inscription might appear for different models)
    seen = set()
    unique_tasks = []
    for key, text in tasks:
        if key not in seen:
            seen.add(key)
            unique_tasks.append((key, text))

    total = len(unique_tasks) * len(MODELS)
    print(f"\nTranslating {len(unique_tasks)} inscription×language pairs "
          f"with {len(MODELS)} models ({total} API calls total)…")

    import concurrent.futures
    import threading

    lock = threading.Lock()
    count = 0

    for model in MODELS:
        # Filter tasks that still need translating for this model
        pending = []
        for (edcs_id, lang), text in unique_tasks:
            key = (edcs_id, lang)
            if key not in translations.get(model, {}):
                pending.append(((edcs_id, lang), text))
            else:
                count += 1

        if not pending:
            print(f"\n--- {model} --- (all cached)")
            continue

        # Use more workers for faster APIs
        workers = 5 if "gemini" in model else 10
        print(f"\n--- {model} --- ({len(pending)} calls, {workers} workers)")

        def _do_translate(args, _model=model):
            key, text = args
            result = translate_single_lang(text, key[1], _model)
            return key, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_do_translate, item): item for item in pending}
            for future in concurrent.futures.as_completed(futures):
                key, result = future.result()
                with lock:
                    translations.setdefault(model, {})[key] = result
                    count += 1
                    if count % 20 == 0:
                        print(f"  [{count}/{total}] translated")

        # Save intermediate cache after each model
        if cache_path:
            _save_cache(translations, cache_path)

    print(f"\nTranslation complete ({count} calls).")
    return translations


def _save_cache(translations, cache_path):
    """Persist translations to JSON cache (keys as pipe-separated strings)."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    serializable = {}
    for model, pairs in translations.items():
        serializable[model] = {f"{k[0]}|{k[1]}": v for k, v in pairs.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


# ======================================================================
# 4.  EVALUATION
# ======================================================================

def compute_metrics(merged_df, translations):
    """Compute BLEU, chrF, and BERTScore for each model × language.

    Returns
    -------
    per_item : list[dict]
        One dict per (model, edcs_id, language) with individual scores.
    aggregated : list[dict]
        One dict per (model, language) with aggregated corpus-level scores.
    """
    per_item = []
    aggregated = []

    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()

    for model in MODELS:
        model_trans = translations.get(model, {})
        for lang in sorted(merged_df["Sprache"].unique()):
            subset = merged_df[merged_df["Sprache"] == lang]
            hypotheses = []
            references = []
            edcs_ids = []
            latin_texts = []

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
                latin_texts.append(row["latin_text"])

            if not hypotheses:
                print(f"  {model} / {lang}: no valid pairs → skip")
                continue

            # ── Sentence-level BLEU & chrF ─────────────────────────────
            for i, (hyp, ref, eid, lat) in enumerate(
                zip(hypotheses, references, edcs_ids, latin_texts)
            ):
                sb = bleu_metric.sentence_score(hyp, [ref])
                sc = chrf_metric.sentence_score(hyp, [ref])
                item = {
                    "model": model,
                    "language": lang,
                    "edcs_id": eid,
                    "sentence_bleu": round(sb.score, 2),
                    "sentence_chrf": round(sc.score, 2),
                    "hypothesis_words": word_count(hyp),
                    "reference_words": word_count(ref),
                    "source_words": word_count(lat),
                }
                per_item.append(item)

            # ── Corpus-level BLEU & chrF ───────────────────────────────
            corpus_bleu = bleu_metric.corpus_score(hypotheses, [references])
            corpus_chrf = chrf_metric.corpus_score(hypotheses, [references])

            agg = {
                "model": model,
                "language": lang,
                "language_name": LANG_NAMES.get(lang, lang),
                "num_pairs": len(hypotheses),
                "corpus_bleu": round(corpus_bleu.score, 2),
                "corpus_chrf": round(corpus_chrf.score, 2),
            }

            # ── BERTScore ──────────────────────────────────────────────
            if HAS_BERTSCORE:
                bert_lang = LANG_MAP_BERT.get(lang, "en")
                P, R, F1 = bert_score_fn(
                    hypotheses, references, model_type="xlm-roberta-large", verbose=False
                )
                agg["bertscore_precision"] = round(P.mean().item(), 4)
                agg["bertscore_recall"] = round(R.mean().item(), 4)
                agg["bertscore_f1"] = round(F1.mean().item(), 4)

                # Also store per-item BERTScore
                for i, (p, r, f) in enumerate(zip(P, R, F1)):
                    per_item_idx = len(per_item) - len(hypotheses) + i
                    per_item[per_item_idx]["bertscore_f1"] = round(f.item(), 4)

            # ── Sentence-level aggregates ──────────────────────────────
            sent_bleus = [it["sentence_bleu"] for it in per_item
                          if it["model"] == model and it["language"] == lang]
            sent_chrfs = [it["sentence_chrf"] for it in per_item
                          if it["model"] == model and it["language"] == lang]
            agg["mean_sentence_bleu"] = round(np.mean(sent_bleus), 2)
            agg["mean_sentence_chrf"] = round(np.mean(sent_chrfs), 2)
            agg["median_sentence_bleu"] = round(float(np.median(sent_bleus)), 2)
            agg["median_sentence_chrf"] = round(float(np.median(sent_chrfs)), 2)

            # Word count stats for hypotheses
            hyp_wcs = [word_count(h) for h in hypotheses]
            ref_wcs = [word_count(r) for r in references]
            agg["mean_hypothesis_words"] = round(np.mean(hyp_wcs), 1)
            agg["mean_reference_words"] = round(np.mean(ref_wcs), 1)
            agg["length_ratio"] = round(np.sum(hyp_wcs) / max(np.sum(ref_wcs), 1), 3)

            aggregated.append(agg)
            print(f"  {model} / {lang}: {len(hypotheses)} pairs → "
                  f"BLEU={agg['corpus_bleu']:.1f}  chrF={agg['corpus_chrf']:.1f}"
                  + (f"  BERTScore-F1={agg.get('bertscore_f1', 'n/a')}" if HAS_BERTSCORE else ""))

    return per_item, aggregated


# ======================================================================
# 5.  OUTPUT & REPORTING
# ======================================================================

def save_results(stats, per_item, aggregated, translations, merged_df):
    """Persist all results to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Statistics ─────────────────────────────────────────────────────
    stats["generated_at"] = datetime.now().isoformat()
    with open(os.path.join(OUTPUT_DIR, "testset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Per-item scores ────────────────────────────────────────────────
    pi_df = pd.DataFrame(per_item)
    pi_df.to_csv(os.path.join(OUTPUT_DIR, "per_inscription_scores.csv"), index=False)

    # ── Aggregated scores ──────────────────────────────────────────────
    agg_df = pd.DataFrame(aggregated)
    agg_df.to_csv(os.path.join(OUTPUT_DIR, "aggregated_scores.csv"), index=False)

    # ── Save all translations ──────────────────────────────────────────
    full_results = []
    for _, row in merged_df.iterrows():
        entry = {
            "edcs_id": int(row["edcs_id_raw"]),
            "edcs_id_full": row.get("edcs_id_full", ""),
            "language": row["Sprache"],
            "latin_text": row["latin_text"],
            "human_translation": row["Übersetzung"],
        }
        for model in MODELS:
            key = (str(int(row["edcs_id_raw"])), row["Sprache"])
            entry[f"llm_{model}"] = translations.get(model, {}).get(key, None)
        full_results.append(entry)

    with open(os.path.join(OUTPUT_DIR, "full_translations.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # ── Summary table (LaTeX-ready) ────────────────────────────────────
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    if not agg_df.empty:
        display_cols = ["model", "language", "num_pairs", "corpus_bleu", "corpus_chrf",
                        "mean_sentence_bleu", "mean_sentence_chrf"]
        if "bertscore_f1" in agg_df.columns:
            display_cols.append("bertscore_f1")
        display_cols.extend(["length_ratio"])
        print(agg_df[display_cols].to_string(index=False))
    print("=" * 80)
    print(f"\nAll results saved to {OUTPUT_DIR}/")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 80)
    print("VOX LAPIDUM – Testset Evaluation Pipeline")
    print("=" * 80)

    merged_df = load_data()
    stats = compute_statistics(merged_df)

    # Print statistics summary
    print("\n── Corpus Statistics ──")
    print(f"  Unique inscriptions : {stats['unique_inscriptions']}")
    print(f"  Languages           : {', '.join(stats['languages'])}")
    for lang, info in stats["per_language"].items():
        print(f"  {lang} ({info['language_name']}): "
              f"{info['num_inscriptions']} refs, "
              f"avg {info['reference_word_count']['mean']} words/ref, "
              f"avg {info['latin_source_word_count']['mean']} words/src")
    print(f"  Latin overall       : {stats['latin_source_overall']['unique_texts']} texts, "
          f"avg {stats['latin_source_overall']['mean_words']} words")

    # Check API keys before translation
    _check_api_keys()

    # Translate
    cache_path = os.path.join(OUTPUT_DIR, "_translation_cache.json")
    translations = translate_testset(merged_df, cache_path=cache_path)

    # Evaluate
    print("\n── Computing Metrics ──")
    per_item, aggregated = compute_metrics(merged_df, translations)

    # Save
    save_results(stats, per_item, aggregated, translations, merged_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
