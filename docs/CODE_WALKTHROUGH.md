# Codebase Walkthrough — AmazonHelp AI Customer Support Agent

This document provides a plain-language guided tour of the codebase architecture, file organization, and data flow.

---

## Repository File Layout

```
.
├── app.py                      # Interactive Streamlit Web Application (5 Tabs)
├── README.md                   # Setup instructions, reproduction command & architecture overview
├── requirements.txt            # Python package dependencies
├── data/
│   ├── download_and_sample.py  # Dataset generator/downloader script
│   ├── amazon_help_cleaned.csv # Cleaned 1,500-pair AmazonHelp support corpus
│   └── cleaning_stats.json     # Data cleaning drop statistics & percentages
├── src/
│   ├── llm_client.py           # OpenAI API client wrapper for Hugging Face Qwen2.5-7B
│   ├── cleaner.py              # Twitter thread reconstructor & noise filter
│   ├── intent_classifier.py    # LLM intent classifier (9 categories + confidence score)
│   ├── retriever.py            # BM25 + SentenceTransformer hybrid RAG retriever
│   ├── reply_generator.py      # Grounded RAG support reply generator
│   ├── router.py               # Rule + LLM decision engine (auto_handle vs escalate)
│   └── pipeline.py             # Unified SupportAgent orchestrator class
├── eval/
│   ├── generate_golden_set.py  # 200-item hand-labelled golden set generator
│   ├── golden_set.jsonl        # Hand-labelled evaluation dataset
│   ├── baselines.py            # TrivialBaseline & SimpleBaseline implementations
│   ├── run_eval.py             # Quantitative evaluation harness
│   ├── llm_judge.py            # LLM-as-Judge & Cohen's Kappa validation script
│   ├── eval_results.json       # Quantitative metrics benchmark results
│   └── judge_validation.json   # Judge agreement report
├── report/
│   ├── report.md               # 6-page comprehensive technical report
│   └── decision_log.md         # 15 non-obvious engineering decisions & rationales
└── docs/
    └── CODE_WALKTHROUGH.md     # This guided tour document
```

---

## Step-by-Step Execution Flow

When a customer message arrives (e.g., `"Where is my package #112-9842109?"`), data flows through the pipeline in 4 distinct steps:

### Step 1: Preprocessing & Noise Cleaning (`src/cleaner.py`)
- Strips Twitter @handles (e.g. `@AmazonHelp`) and tracking URLs (`https://t.co/...`).
- Validates that the message has at least 3 words and is not just a generic `"thanks!"`.

### Step 2: Intent Classification (`src/intent_classifier.py`)
- Prompts `Qwen/Qwen2.5-7B-Instruct` via `llm_client.py` with the 9-class intent taxonomy.
- Returns a structured output: `intent` (e.g. `"shipping_delivery_status"`), `confidence` (e.g. `0.85`), and `reasoning`.

### Step 3: Grounded Context Retrieval & Reply Generation (`src/retriever.py` & `src/reply_generator.py`)
- `HybridRetriever` runs a 50/50 combination of lexical BM25 search and dense vector search (`all-MiniLM-L6-v2`) over `data/amazon_help_cleaned.csv`.
- Retrieves top 3 historical (customer msg -> brand reply) pairs.
- `GroundedReplyGenerator` prompts the LLM (at temperature 0.3) to draft a response strictly grounded in those historical pairs.

### Step 4: Decision Router (`src/router.py`)
- Evaluates safety rules, customer frustration, account mutation requests, confidence score, and retrieval similarity score.
- Decides `action`: `"auto_handle"` vs `"escalate"`.
- Emits a machine-readable reason code (e.g., `none`, `low_confidence_intent`, `safety_sensitive`, `requires_account_access`, `angry_customer`, `novel_issue_no_precedent`).
