# Evaluation summary

Generated from `synthetic-seed-62042026` with seed `62042026`. These are actual computed results from the checked-in synthetic data; they are not estimates of real-world performance.

## Intent classification

| Split | n | Accuracy | Macro-F1 | 95% bootstrap CI, accuracy | 95% bootstrap CI, Macro-F1 |
|---|---:|---:|---:|---|---|
| validation | 40 | 0.950 | 0.948 | [0.875, 1.0] | [0.8586666666666666, 1.0] |
| test | 50 | 1.000 | 1.000 | [1.0, 1.0] | [1.0, 1.0] |
| challenge_test | 100 | 0.680 | 0.667 | [0.58, 0.77] | [0.5658020280661035, 0.7587706833396488] |

Challenge split contains 100 requests and is separate from the original grouped test split. The classifier is a custom Naive Bayes baseline, trained only on train data. See `nlp_evaluation.json` for confusion matrices, slices, and errors.

## NER

Rule-based held-out test: 15 examples, 75 gold entities; precision 1.000, recall 1.000, F1 1.000. The rules use a training supplier vocabulary and the product/warehouse catalog. This templated catalog-assisted score is not a learned-model result and does not establish catalog-independent generalization.

## Semantic retrieval challenge

Each method sees the same 100 held-out challenge queries and 25 candidate SKUs. There are 75 positive/ambiguous queries and 25 negative queries. Metrics below are computed from ranked candidate lists.

| Method | Top-1 accuracy (positive/ambiguous) | Recall@10 | MRR@10 | Negative false-accept rate at fixed threshold 0.2 |
|---|---:|---:|---:|---:|
| Lexical TF-IDF | 0.947 | 1.000 | 0.960 | 0.000 |
| Sentence-Transformer + FAISS | 0.813 | 0.987 | 0.873 | 0.000 |
| Cross-Encoder reranked | 0.973 | 0.987 | 0.976 | 0.000 |

The pretrained systems were measured with locally cached multilingual weights. On a fresh installation without those cached weights, only the lexical baseline is available; the app reports that fallback. The Cross-Encoder is a reranking experiment, not a purchasing decision. See the JSON report for seeded bootstrap intervals, language/query-type slices, ambiguity metrics, and retrieval errors.

## Limitations

- All evaluation corpora are synthetic and mostly template-generated; scores do not estimate production performance.
- Topic classification is a custom Multinomial Naive Bayes baseline trained on train only; validation/test were not used for fitting.
- NER rules are evaluated on 15 held-out examples from templated sentences; this is not a learned model. Product entities use the available inventory catalog as a lexicon, so the score is not a catalog-independent generalization estimate.
- All challenge queries and relevance labels are synthetic. Use their measurements for controlled comparison, not estimates of production performance.
- Pretrained transformer attention is available only when optional dependencies and a model are already cached locally.

Reproduce with `python scripts/run_nlp_evaluation.py`. The evaluation script uses the fixed corpus and seed; changing data or package/model versions can change measurements.
