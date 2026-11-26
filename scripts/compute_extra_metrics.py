"""
Compute additional metrics (METEOR, TER) for existing translations.
Reads from _translation_cache.json and human-translations-testset.csv,
then merges results into aggregated_scores.csv.
"""
import os, sys, json, re
import pandas as pd
import numpy as np
from collections import defaultdict

# ── Load .env ──
try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env):
        load_dotenv(_env)
except ImportError:
    pass

from translate import MODELS

# NLTK METEOR
import nltk
from nltk.translate.meteor_score import meteor_score as _meteor_score
# Ensure wordnet is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

# sacrebleu TER
from sacrebleu.metrics import TER

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET_DIR = os.path.join(BASE_DIR, "data", "testset")
OUTPUT_DIR = os.path.join(TESTSET_DIR, "results")
HUMAN_CSV = os.path.join(TESTSET_DIR, "human-translations-testset.csv")
FILTERED_CSV = os.path.join(TESTSET_DIR, "filtered-testset.csv")
CACHE_PATH = os.path.join(OUTPUT_DIR, "_translation_cache.json")
AGG_CSV = os.path.join(OUTPUT_DIR, "aggregated_scores.csv")

LANG_NAMES = {"DE": "German", "EN": "English", "FR": "French"}

def word_count(text):
    if not isinstance(text, str): return 0
    clean = re.sub(r'[\[\]\(\)\{\}<>⟦⟧/|]', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return len(clean.split()) if clean else 0

def load_references():
    """Load and merge reference translations with Latin source texts."""
    hdf = pd.read_csv(HUMAN_CSV).dropna(subset=["edcs_id", "Übersetzung", "Sprache"])
    hdf["edcs_id"] = hdf["edcs_id"].astype(int)
    if "ref_source" not in hdf.columns:
        hdf["ref_source"] = "human"
    
    fdf = pd.read_csv(FILTERED_CSV)
    fdf["edcs_id_num"] = fdf["edcs_id"].str.replace("EDCS-", "", regex=False).astype(int)
    
    merged = hdf.merge(
        fdf[["edcs_id_num", "text"]].drop_duplicates(subset=["edcs_id_num"]),
        left_on="edcs_id", right_on="edcs_id_num", how="left"
    )
    merged.rename(columns={"text": "latin_text"}, inplace=True)
    merged = merged.dropna(subset=["latin_text"])
    return merged

def compute_meteor(hypothesis, reference, lang="en"):
    """Compute METEOR score for a single hypothesis/reference pair."""
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    try:
        return _meteor_score([ref_tokens], hyp_tokens)
    except Exception:
        return 0.0

def main():
    print("=" * 70)
    print("Computing METEOR & TER for all model translations")
    print("=" * 70)

    # Load translations cache
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    translations = {}
    for model, pairs in cache.items():
        translations[model] = {tuple(k.split("|")): v for k, v in pairs.items()}

    # Load references
    merged = load_references()
    print(f"Loaded {len(merged)} reference pairs")

    # TER metric from sacrebleu
    ter_metric = TER()

    # Define tiers
    tiers = [
        ("human", merged[merged["ref_source"] == "human"]),
        ("gemini", merged[merged["ref_source"] == "gemini"]),
        ("all", merged),
    ]

    results = []
    active_models = sorted(translations.keys())
    total_combos = len(active_models) * 3 * len(tiers)  # models × langs × tiers
    done = 0

    for tier_label, tier_df in tiers:
        if tier_df.empty:
            continue
        print(f"\n── Tier: {tier_label} ({len(tier_df)} refs) ──")

        for model in active_models:
            model_trans = translations.get(model, {})
            for lang in sorted(tier_df["Sprache"].unique()):
                subset = tier_df[tier_df["Sprache"] == lang]
                hypotheses = []
                references = []

                for _, row in subset.iterrows():
                    key = (str(int(row["edcs_id"])), lang)
                    hyp = model_trans.get(key)
                    if hyp is None or not isinstance(hyp, str) or hyp.startswith("ERROR"):
                        continue
                    ref = row["Übersetzung"]
                    if not isinstance(ref, str) or len(ref.strip()) == 0:
                        continue
                    hypotheses.append(hyp)
                    references.append(ref)

                if not hypotheses:
                    done += 1
                    continue

                # ── Corpus TER ──
                corpus_ter = ter_metric.corpus_score(hypotheses, [references])

                # ── Sentence-level METEOR ──
                meteor_lang = lang.lower() if lang.lower() in ["en", "de", "fr"] else "en"
                meteors = []
                for hyp, ref in zip(hypotheses, references):
                    m = compute_meteor(hyp, ref, meteor_lang)
                    meteors.append(m)

                corpus_meteor = np.mean(meteors) if meteors else 0.0

                # ── Sentence-level TER ──
                sent_ters = []
                for hyp, ref in zip(hypotheses, references):
                    st = ter_metric.sentence_score(hyp, [ref])
                    sent_ters.append(st.score)

                results.append({
                    "model": model,
                    "language": lang,
                    "ref_source": tier_label,
                    "corpus_meteor": round(corpus_meteor * 100, 2),  # as percentage
                    "corpus_ter": round(corpus_ter.score, 2),
                    "mean_sentence_meteor": round(np.mean(meteors) * 100, 2),
                    "median_sentence_meteor": round(float(np.median(meteors)) * 100, 2),
                    "mean_sentence_ter": round(np.mean(sent_ters), 2),
                    "median_sentence_ter": round(float(np.median(sent_ters)), 2),
                })

                done += 1
                print(f"  {model} / {lang} [{tier_label}]: {len(hypotheses)} pairs → "
                      f"METEOR={corpus_meteor*100:.1f}  TER={corpus_ter.score:.1f}")

    # ── Merge into aggregated_scores.csv ──
    extra_df = pd.DataFrame(results)
    
    # Load existing aggregated scores
    agg_df = pd.read_csv(AGG_CSV)
    
    # Drop old METEOR/TER columns if they exist
    for col in ["corpus_meteor", "corpus_ter", "mean_sentence_meteor",
                 "median_sentence_meteor", "mean_sentence_ter", "median_sentence_ter"]:
        if col in agg_df.columns:
            agg_df.drop(columns=[col], inplace=True)

    # Merge on model + language + ref_source
    merged_agg = agg_df.merge(
        extra_df, on=["model", "language", "ref_source"], how="left"
    )
    
    merged_agg.to_csv(AGG_CSV, index=False)
    print(f"\n✅ METEOR & TER merged into {AGG_CSV}")
    print(f"   Columns now: {list(merged_agg.columns)}")

if __name__ == "__main__":
    main()
