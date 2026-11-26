# Vox Lapidum — MT Evaluation Experiment

LLM translation of Latin epigraphic inscriptions, evaluated against human
reference translations in German, English and French.

## What is in this folder

```
mt-experiment/
├── data/
│   └── testset/
│       ├── filtered-testset.csv              # 305 Latin inscriptions from EDCS
│       ├── human-translations-testset.csv    # 915 human + Gemini reference translations (DE/EN/FR)
│       └── results/
│           ├── _translation_cache.json       # all LLM translations (12 models × 305 inscriptions × 3 langs)
│           ├── full_translations.json        # human refs + all LLM translations side by side
│           ├── aggregated_scores.csv         # corpus-level BLEU, chrF, BERTScore, TER, METEOR per model × lang
│           ├── per_inscription_scores.csv    # sentence-level scores (21 960 rows)
│           ├── testset_statistics.json       # corpus statistics
│           └── visualizations/               # score distributions, heatmaps, examples (PDF + PNG)
├── scripts/
│   ├── translate.py                # LLM translation (Gemini, Claude, GPT)
│   ├── evaluate_testset.py         # main pipeline: translate → evaluate → save
│   ├── compute_extra_metrics.py    # TER + METEOR on cached translations
│   ├── generate_paper_assets.py    # LaTeX tables + figures for the paper
│   ├── visualize_testset.py        # score visualizations
│   ├── analyse_alternate_readings.py  # epigraphic alternate-reading analysis
│   ├── prepare_ground_truth.py     # ref_source enrichment (human vs Gemini)
│   ├── preprocess.py               # filter Greek / short inscriptions from EDCS dump
│   ├── evaluate.py                 # earlier evaluation script (BLEURT, COMET)
│   └── visualize.py                # earlier visualization script
├── requirements.txt
├── .env.template
└── .gitignore
```

## Testset

305 Latin inscriptions selected from the EDCS (Epigraphic Database Clauss-Slaby),
each with human reference translations in up to three languages:

| Language | Human references | Gemini references | Total |
|----------|-----------------|-------------------|-------|
| DE       | ~100            | ~200              | ~305  |
| EN       | ~100            | ~200              | ~305  |
| FR       | ~100            | ~200              | ~305  |

The `ref_source` column in `human-translations-testset.csv` distinguishes the
two tiers.

## Models

12 LLMs, 6 commercial + 6 open-source:

| Model | Provider |
|-------|----------|
| gemini-2.5-flash | Google |
| gemini-3.5-flash | Google |
| claude-sonnet-4-6 | Anthropic |
| claude-sonnet-5 | Anthropic |
| gpt-5 | OpenAI |
| gpt-5-nano-2025-08-07 | OpenAI |
| deepseek-v3 | DeepSeek |
| llama-3.1-8b | Meta (via DeepInfra) |
| llama-3.1-70b | Meta (via DeepInfra) |
| qwen-2.5-72b | Alibaba (via DeepInfra) |
| apertus-8b | (via DeepInfra) |
| apertus-70b | (via DeepInfra) |

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
cp .env.template ../.env  # one level up, where scripts expect it
# edit ../.env with your keys
cd scripts/
python evaluate_testset.py
```

## Inscription texts

Inscription texts are from the EDCS (Epigraphic Database Clauss-Slaby,
https://edcs.hist.uzh.ch). `filtered-testset.csv` contains the Latin source
texts with EDCS identifiers.

## License

Code: MIT. Inscription texts from EDCS. Human translations by the project team.

