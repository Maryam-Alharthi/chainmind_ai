# Applied NLP lab mapping

| Lab | Implementation | Evidence and boundary |
|---|---|---|
| 1. Text processing | `nlp_utils.py`; NLP Intelligence tab | Arabic/English normalization, PII masking, regex tokenization, token audit, fertility/unknown-token fields, tokenizer comparison when cached. Heuristic segmentation is labeled as such. |
| 2. Attention | `nlp/attention.py`; NLP Intelligence tab | Scaled dot-product, multi-head, causal and padding-mask utilities, visualization and description comparison. Random examples and pretrained cached attention are distinguished. |
| 3A. Topic classification | `nlp/intent_classifier.py`; `requests.csv`; Evaluation tab | Five intents, family-grouped splits, custom word/character n-gram Naive Bayes, confusion matrix, slices and bootstrap intervals. No transformer fine-tuning is claimed. |
| 3B. NER | `ner_bio.jsonl`; `nlp/entities.py` | BIO spans for SKU, quantity, product, supplier, warehouse; alignment checks and subword labels with continuation pieces masked as `-100`. App extraction is rule-based, not trained NER. |
| 4. Arabic pipeline | `nlp_utils.py`; NLP Intelligence tab | Arabic normalization, variation/tokenizer comparison, clitic heuristic, multilingual and Arabic tokenizer inspection when cached. No Arabic model fine-tuning is claimed. |
| 5. Semantic search | `nlp/semantic_search.py`; Semantic Search tab | TF-IDF baseline and optional multilingual sentence embeddings + FAISS; optional Cross-Encoder reranking; Recall@10/MRR@10 and challenge queries. Models must already be cached. |
| 6. Evaluation | `nlp/evaluation.py`; `scripts/run_nlp_evaluation.py`; report JSON | Test slices, seeded bootstrap intervals, errors, fixed-threshold negative-query checks, ambiguity metrics, and explicit dataset limitations. |

All included datasets are synthetic. Their sizes, split counts, generation assumptions, and actual computed measurements are documented in README and `EVALUATION_SUMMARY.md`.
