# ChainMind AI architecture

## Runtime boundaries

- `app.py`: Streamlit pages, cached data loading, forms, charts, downloads, and user-visible explanations.
- `engine.py`: validated inventory records and deterministic SKU search, stock aggregation, order feasibility, summary, and draft report logic.
- `nlp/`: normalization, attention, intent baseline, rule-based NER, semantic retrieval, and evaluation helpers.
- `data/synthetic/`: reproducibly generated catalog and labeled task corpora. All files are synthetic.
- `scripts/`: dataset generator, data integrity audit, and evaluation report generator.
- `reports/nlp_evaluation.json`: computed report from the fixed seed corpus; regenerate with the command in README.

Inventory availability is computed per warehouse row as `max(on_hand - reserved, 0)`, then summed by SKU. Order shortage is `max(requested - available, 0)`. These calculations run in Python, independent of language models. Draft purchasing recommendations are informational only and do not change inventory or place orders.

## NLP evaluation flow

The group manifest assigns product families to train, validation, or test before creating the related requests, NER examples, and semantic pairs. The separate challenge query set uses families held out from training. The classifier fits only on train; validation and test are for measurement. Retrieval compares lexical TF-IDF with locally cached multilingual Sentence-Transformers + FAISS and optional Cross-Encoder reranking. The optional models load only from an existing local Hugging Face cache; app interactions do not download model weights.

The classifier and NER baselines are deliberately identified as custom Naive Bayes and rule-based extraction. They are not fine-tuned transformer models. Attention demonstrations distinguish deterministic/random teaching examples from outputs extracted from an available cached pretrained model.

## Data flow

```text
synthetic CSV/JSONL -> schema and leakage validation -> baseline/model evaluation -> JSON report
         |                                                           |
         +-> Streamlit inventory and NLP views <---------------------+
```

## Trust and limitations

The application is read-only. API credentials are not stored in source files; no API request is made in offline mode. Synthetic templates and a small number of held-out families limit conclusions about real language, suppliers, and inventory. Retrieval scores are candidate signals, not proof of product equivalence. Review ambiguous matches before operational use.
