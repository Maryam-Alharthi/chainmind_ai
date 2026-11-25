# Project summary for portfolio/CV

**ChainMind AI — Bilingual Supply Chain Intelligence Copilot**

Built a Streamlit application for auditable inventory search and order feasibility across a synthetic multi-warehouse catalog. Implemented deterministic stock aggregation and shortage calculations, bilingual alias-aware search, safe draft replenishment reports, and synthetic Arabic/English NLP datasets with group-aware splits and validation. Added NLP baselines for intent classification, rule-based BIO entity extraction, lexical retrieval, and optional local multilingual embedding/FAISS and Cross-Encoder experiments. Produced reproducible evaluation artifacts with sliced metrics, bootstrap intervals, error examples, and explicit limitations.

**Measured project corpus:** 150 synthetic SKUs, 300 labeled requests, 100 BIO examples, 100 relevance pairs, plus 100 intent and 100 retrieval challenge queries. Evaluation values are reported in `EVALUATION_SUMMARY.md`; these small template-generated datasets are not representative of operational performance.

**Technology:** Python 3.11, Streamlit, pandas, scikit-learn-style custom baseline, Hugging Face Transformers/Sentence-Transformers (optional local cache), FAISS (optional), ReportLab, pytest.
