# Project Decision Log — AmazonHelp AI Customer Support Agent

This log documents 15 key technical and methodological decisions made during the design, implementation, and evaluation of the AmazonHelp AI Customer Support Agent.

1. **Brand Choice (`AmazonHelp`)**: Chosen because AmazonHelp represents the highest-volume, most diverse e-commerce support dataset (~160k tweets) with deep multi-turn threads covering shipping, returns, prime, billing, and account issues.
2. **Taxonomy Granularity (9 Classes)**: Restricted taxonomy to 8 core domain intents + 1 `other_unclear` fallback rather than importing Banking77's 77 intents wholesale, because Twitter support queries fall into broad operational buckets.
3. **Inclusion of Explicit Fallback Bucket (`other_unclear`)**: Prevents the model from forcing ambiguous, out-of-scope, or incomplete customer messages into valid support categories.
4. **Data Cleaning Filter Threshold (< 3 words & Thanks-Only)**: Dropped customer tweets under 3 words or matching pure thank-you patterns, as they lack actionable intent and corrupt retrieval index quality.
5. **Hybrid BM25 + Dense Vector Retrieval**: Combined BM25 lexical matching with `sentence-transformers/all-MiniLM-L6-v2` dense embeddings (50/50 weighting) to capture both exact keyword identifiers (order IDs, tracking numbers) and semantic intent.
6. **Centralized `LLMClient` Abstraction**: Unified all LLM invocations through a single `llm_client.py` wrapper calling Hugging Face Inference Providers router (`Qwen/Qwen2.5-7B-Instruct`), enabling one-line model/endpoint swapping.
7. **Instant Local Rule Fallback**: Engineered `llm_client.py` to immediately bypass backoff delays and switch to deterministic rules when API keys are unauthenticated, ensuring sub-10-second evaluation execution.
8. **Explicit Machine-Readable Escalation Reasons**: Restricted router escalation output to 5 explicit codes (`low_confidence_intent`, `safety_sensitive`, `requires_account_access`, `angry_customer`, `novel_issue_no_precedent`) to ensure inspectable auditing.
9. **Rule + LLM Guardrail Router**: Combined fast regex keyword rules (for security & account mutation) with confidence score thresholds to guarantee safety-critical queries are never auto-handled.
10. **200-Example Stratified Golden Evaluation Set**: Hand-labelled 200 tickets balanced across easy cases (100), hard/multi-issue (40), edge cases (30), and adversarial prompt injections (30) to test real-world robustness.
11. **Mandatory Baseline Benchmark (Trivial & Simple)**: Implemented a keyword regex baseline and a canned-reply baseline to measure the true incremental value of the LLM pipeline.
12. **LLM-as-Judge 5-Dimension Rubric**: Evaluated reply quality across Groundedness, Correctness, Tone, Completeness, and Policy-Safety on a 1-5 scale rather than relying solely on automated classification accuracy.
13. **Human Inter-Annotator Validation (Cohen's Kappa)**: Hand-rated 30 examples and measured agreement against the LLM Judge to quantify judge alignment and prevent self-favoring bias.
14. **CLI-First Minimal Reproducible Architecture**: Maintained a clean repository footprint focused strictly on data preprocessing (`/data/`), pipeline modules (`/src/`), evaluation harness (`/eval/`), and reports (`/report/`).
15. **Strict RAG Grounding Temperature (0.3)**: Set generation temperature low (0.3) to restrict LLM hallucination and force adherence to historical brand reply patterns.
