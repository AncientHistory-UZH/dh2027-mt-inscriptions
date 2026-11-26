# Vox Lapidum — MT Evaluation Experiment

LLM translation of Latin epigraphic inscriptions, evaluated against human
reference translations in German, English and French.

## What is in this folder

```
├── data/
│   └── testset/
│       ├── filtered-testset.csv              # 307 Latin inscriptions from EDCS
│       ├── human-translations-testset.csv    # 915 reference translations (DE/EN/FR, human + Gemini)
│       └── results/
│           ├── _translation_cache.json       # all LLM translations (12 models × 305 inscriptions × 3 langs)
│           ├── full_translations.json        # human refs + all LLM translations side by side
│           ├── aggregated_scores.csv         # corpus-level BLEU, chrF, BERTScore, TER, METEOR per model × lang
│           ├── per_inscription_scores.csv    # sentence-level scores (21 960 rows)
│           ├── testset_statistics.json       # corpus statistics
│           └── visualizations/               # score distributions, heatmaps, examples (PDF + PNG)
├── scripts/
│   ├── translate.py                # model list + API routing (Gemini, Anthropic, OpenAI, DeepInfra)
│   ├── evaluate_testset.py         # main pipeline: translate → evaluate → save (with incremental cache)
│   ├── compute_extra_metrics.py    # TER + METEOR on cached translations
│   ├── generate_paper_assets.py    # LaTeX tables + figures for the paper
│   ├── visualize_testset.py        # score visualizations
│   ├── analyse_alternate_readings.py  # epigraphic alternate-reading analysis
│   ├── prepare_ground_truth.py     # ref_source enrichment (human vs Gemini)
│   └── preprocess.py               # filter Greek / short inscriptions from EDCS dump
├── requirements.txt
├── .env.template
└── .gitignore
```

## How the results were produced

The main pipeline is `evaluate_testset.py`. It translates every inscription
into every target language with every model, then computes BLEU, chrF and
BERTScore. Translations are cached in `_translation_cache.json`: when the
script is re-run, only models missing from the cache trigger API calls.

The twelve models were added incrementally: `MODELS` in `translate.py` was
updated in batches (commercial APIs first, then open-source models via
DeepInfra) and the pipeline was re-run each time. The cache accumulated all
outputs across runs.

After the main evaluation, `compute_extra_metrics.py` added TER and METEOR
from the same cache.

## Testset

305 Latin inscriptions selected from the EDCS (Epigraphic Database Clauss-Slaby),
each with reference translations in up to three languages:

| Language | Human references | Gemini references | Total |
|----------|-----------------|-------------------|-------|
| DE       | ~100            | ~200              | ~305  |
| EN       | ~100            | ~200              | ~305  |
| FR       | ~100            | ~200              | ~305  |

The `ref_source` column in `human-translations-testset.csv` distinguishes the
two tiers.

## Models

12 LLMs, 6 commercial + 6 open-source:

| Model | Provider | API |
|-------|----------|-----|
| gemini-2.5-flash | Google | Gemini API |
| gemini-3.5-flash | Google | Gemini API |
| claude-sonnet-4-6 | Anthropic | Anthropic API |
| claude-sonnet-5 | Anthropic | Anthropic API |
| gpt-5 | OpenAI | OpenAI API |
| gpt-5-nano-2025-08-07 | OpenAI | OpenAI API |
| deepseek-v3 | DeepSeek | DeepInfra (OpenAI-compatible) |
| llama-3.1-8b | Meta | DeepInfra (OpenAI-compatible) |
| llama-3.1-70b | Meta | DeepInfra (OpenAI-compatible) |
| qwen-2.5-72b | Alibaba | DeepInfra (OpenAI-compatible) |
| apertus-8b | — | DeepInfra (OpenAI-compatible) |
| apertus-70b | — | DeepInfra (OpenAI-compatible) |

## Metrics

| Metric | Tool | Direction |
|--------|------|-----------|
| BLEU | sacrebleu | ↑ higher = better |
| chrF | sacrebleu | ↑ higher = better |
| BERTScore | bert_score | ↑ higher = better |
| METEOR | nltk | ↑ higher = better |
| TER | sacrebleu | ↓ lower = better |

## Reproducing the evaluation (no API keys needed)

The translation cache already contains all LLM outputs. To recompute the
metrics from cached translations:

```bash
pip install -r requirements.txt
cd scripts/
python evaluate_testset.py          # uses _translation_cache.json, skips API calls
python compute_extra_metrics.py     # adds TER + METEOR
python visualize_testset.py         # regenerates visualizations
python generate_paper_assets.py     # regenerates LaTeX tables + figures
```

## Re-running translations (needs API keys)

Copy `.env.template` to `.env` in the repository root and fill in your keys.
Delete or rename `_translation_cache.json` to force fresh translations.

```bash
cp .env.template .env
# edit .env with your keys
cd scripts/
python evaluate_testset.py
```

## Inscription texts

Inscription texts are from the EDCS (Epigraphic Database Clauss-Slaby,
https://edcs.hist.uzh.ch). `filtered-testset.csv` contains the Latin source
texts with EDCS identifiers.

## License

Code: MIT. Inscription texts from EDCS. Human translations by the project team.
