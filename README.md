# ChainMind AI — Supply Chain Intelligence Copilot

ChainMind is a bilingual, read-only inventory assistant and order feasibility dashboard built for supply chain planners, warehouse teams, and procurement analysts. It demonstrates deterministic inventory arithmetic and optional LLM tool calling over synthetic sample records.

## Problem and solution

Teams need quick, auditable answers to product availability and shortage questions across warehouses. ChainMind searches SKUs and aliases, aggregates reserved-adjusted stock, calculates deficits in Python, surfaces low stock, and prepares a draft replenishment report. It never modifies inventory or submits an order.

## Features

- Inventory dashboard with data-derived SKU, available unit, low-stock, warehouse, estimated on-hand value, and replenishment metrics.
- Category and warehouse stock charts, alerts, search, CSV export, and upload validation.
- Arabic/English SKU and alias matching with normalized Arabic text and similarity-ranked candidates.
- Multi-line order feasibility, warehouse breakdown, shortages, and cost estimates when unit costs exist.
- The workspace offers local Qwen3 1.7B through Ollama, an offline rules demo, and an optional OpenAI provider.
- Inspectable tool inputs/results and PDF inventory/order draft reports.
- Automated regression coverage, including BRG-6204: requested 220, available 170, shortage 50.

## Agent architecture

`app.py` handles the Streamlit interface. `engine.py` validates inventory records and implements deterministic lookup, matching, stock, shortage, low-stock, summary, and order-line functions. The optional model chooses among read-only tools (`search_products`, `get_stock`, `check_order`, `list_low_stock`); Python computes every inventory quantity. Offline mode uses transparent rules and the same functions.

## Selected NLP techniques and system architecture

The Streamlit interface stays in `app.py`; deterministic stock calculations remain in `engine.py`. Lightweight NLP logic is organized in `nlp/`, reusable text utilities remain in `nlp_utils.py`, and reproducible corpora/tools live under `data/synthetic/` and `scripts/`.

- **Lab 1, text processing:** Unicode and Arabic letter normalization, English case/space normalization, email and Saudi mobile masking (SKU patterns are preserved), regex token audit, and a simple Arabic clitic heuristic. Input text is displayed unchanged beside processed text.
- **Lab 2, attention:** deterministic scaled dot-product and multi-head random attention with an optional causal mask is shown only as an educational random example. The app can display last-layer attention from a locally cached multilingual sentence-transformer model when optional dependencies and weights are available. Random and pretrained outputs are labeled separately.
- **Lab 3A, classification:** a custom Multinomial Naive Bayes baseline with word, bigram, and character n-gram features trains from the 210 family-grouped train requests. It is available for website inference without model downloads. Validation and test remain separate.
- **Lab 3B, NER:** the website uses an explicitly rule-based baseline (SKU/quantity patterns, supplier lexicon, warehouse vocabulary, and inventory product/alias lookup). BIO examples are checked for subword/token/span consistency. No trained NER model is claimed.
- **Lab 4, Arabic:** normalization and clitic heuristics are integrated. The UI can compare cached multilingual and Arabic pretrained tokenizers if locally available. No dialect labels or dialect fine-tuning are claimed.
- **Lab 5, retrieval:** exact SKU lookup has priority, followed by a lightweight multilingual character/word TF-IDF cosine index. The optional local path uses Sentence-Transformers with FAISS and can rerank with a cached multilingual Cross-Encoder. Similarity is a candidate-ranking signal, not proof of part equivalence.
- **Lab 6, evaluation:** reproducible Accuracy, Macro-F1, confusion matrix, language/length/category slices, observed error examples, Recall@10, MRR@10, and seeded 2,000-resample percentile bootstrap intervals are exported. Validation is used for diagnostics; the test split is not used to fit or tune the classifier.

### Measured results

The latest computed evaluation is summarized in [docs/EVALUATION_SUMMARY.md](docs/EVALUATION_SUMMARY.md); the full report with slices, error cases, and bootstrap intervals is [reports/nlp_evaluation.json](reports/nlp_evaluation.json). It includes the original grouped validation/test sets plus separate 100-query classification and retrieval challenge sets.

The retrieval challenge compares lexical TF-IDF, multilingual Sentence-Transformer + FAISS, and Cross-Encoder reranking on the same 100 queries and 25 held-out candidate SKUs. These are real computed scores, but the data is small and synthetic. In this run, top-1 accuracy was 0.947 for TF-IDF, 0.813 for embeddings, and 0.973 after Cross-Encoder reranking; all three had zero false accepts on 25 negative queries at the fixed threshold 0.2. Do not interpret these as operational performance. The pretrained measurements require model weights already available in the local cache.

The intent classifier is a custom Naive Bayes baseline, and NER is rule-based catalog-assisted extraction. Neither is described as a fine-tuned transformer. The NER score is measured on 15 templated held-out examples and is not a catalog-independent estimate. Refer to the report for exact split sizes and confidence intervals.

### Semantic search smoke queries

With the cached multilingual model and FAISS enabled, the verified hybrid search ranked BRG-6204 first for `رمان بلي للمكينة`, `bearing 6204`, and `أحتاج محمل 6204`; `machine bearing` ranked a bearing first; and `برغي ماكينة M6` ranked machine screw SCR-03 first, ahead of unrelated products. The Cross-Encoder reranking test also ranked BRG-6204 first for `bearing 6204`. Each row shows its retrieval/reranking method and component scores. Similarity remains a search aid; confirm SKU and approved specifications before substitution.

## Synthetic dataset phase

All files in `data/synthetic/` are generated from hand-authored supply-chain vocabularies and deterministic rules. They contain no personal, customer, or confidential company information. Each record is marked synthetic. The fixed default seed is `62042026`.

| File | Contents |
|---|---|
| `inventory.csv` | 150 unique SKUs across Riyadh, Jeddah, Dammam, Madinah, and Abha, with English and Arabic product descriptions. It has 151 warehouse rows because BRG-6204 is stocked at two sites. English/Arabic names, aliases and categories, on-hand/reserved/reorder values, unit cost in SAR, product-family IDs, and a group split are included. |
| `requests.csv` | 300 Arabic and English requests (150 per language) labeled evenly across `stock_inquiry`, `shortage_check`, `replenishment`, `shipping`, or `purchasing` (60 each). SKU, quantity, warehouse, language, family group, template group, split, and synthetic marker are included. |
| `ner_bio.jsonl` | 100 examples with whitespace-aligned `tokens` and `bio_tags`, plus entity spans. Every example contains SKU, quantity, product, supplier, and warehouse entities. This is rule-generated annotation, not trained NER output. |
| `semantic_pairs.csv` | 50 bilingual queries, each paired with one relevant document (`relevance=2`) and one non-relevant document (`relevance=0`), for 100 labeled pairs total. |
| `challenge_requests.csv` | 100 synthetic intent challenge requests, grouped by held-out families and kept separate from training. |
| `challenge_search.csv` | 100 synthetic relevance queries (including Arabic, mixed-language, ambiguity, and negative queries) over held-out candidate product families. |
| `split_manifest.json` | Seed, family-to-split assignments, and split method. |

Splits are assigned by product family so related SKU variants remain together. Request template groups are family-specific and all family members remain in one split. The generator checks exact duplicate request text, high-similarity request pairs across splits (SequenceMatcher threshold 0.97), BIO alignment, schema/labels, semantic pair labels, and family assignment consistency. Current allocation is 70/13/17 percent by family (21 train, 4 validation, 5 test). Resulting counts are 210/40/50 requests, 70/15/15 NER examples, and 72/14/14 semantic pairs (train/validation/test). These are dataset construction checks, not model evaluation results.

The compact `data/sample_inventory.csv` remains unchanged for regression tests. The dashboard’s default example loads `data/synthetic/inventory.csv` (150 SKUs); BRG-6204 still has 170 available across Riyadh and Jeddah, so a request for 220 has a shortage of 50.

Regenerate, validate, train the small baseline, and export evaluation from the project root:

```bash
python scripts/generate_synthetic_data.py
python scripts/validate_synthetic_data.py
python scripts/run_nlp_evaluation.py
```

Use `--seed <integer>` to create a different deterministic realization, or `--out-dir <path>` to write elsewhere. Tests compare regenerated bytes against the checked-in default-seed files.

## Technology

Python, Streamlit, pandas, NumPy, ReportLab, pytest. A small pure-Python Naive Bayes baseline and TF-IDF search need no heavyweight model stack. Local Qwen uses Python’s standard HTTP client to reach Ollama on loopback.

Optional locally cached transformer, multilingual embedding, FAISS, and Cross-Encoder experiments are listed in `requirements-nlp.txt`. They are not required for website startup. The semantic-search UI defaults to the Sentence-Transformers + FAISS hybrid when dependencies are installed; exact SKU lookup still runs first. Otherwise the UI uses TF-IDF and shows the fallback method per result. The app uses locally cached weights only and does not download models during a search.

Install the optional stack inside the activated `chainmind-ai` environment with `python -m pip install -r requirements-nlp.txt`. The tested embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (pinned cached revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`); optional reranking uses `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (revision `1427fd652930e4ba29e8149678df786c240d8825`). Cache these weights separately when network access and disk space permit. The Arabic tokenizer comparison can also use `CAMeL-Lab/bert-base-arabic-camelbert-mix`. Model files are excluded from Git.

## Isolated local environment (recommended)

Use a dedicated Conda environment with Python 3.11. These commands leave the Anaconda `base` environment unchanged:

```bash
conda create -n chainmind-ai python=3.11 -y
conda activate chainmind-ai
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_synthetic_data.py
python scripts/run_nlp_evaluation.py
```

Then launch the app from the project directory:

```bash
streamlit run app.py
```

The dependency constraints keep NumPy below 2 and install compatible `numexpr`, `bottleneck`, and PyArrow versions in this environment. The NumPy 2.4.6 warnings from packages in a pre-existing Anaconda base are avoided by using `chainmind-ai`; do not downgrade or uninstall packages from `base`. If Conda is unavailable, create an isolated Python 3.11 `venv` and install the same requirements there.

To confirm which Python and packages are active:

```bash
python --version
python -m pip show numpy pandas numexpr bottleneck
```

## Configuration

- `OPENAI_API_KEY` (optional): enables API tool calling. You can also enter it in the sidebar. Never commit credentials.
- `OPENAI_MODEL` (optional): model name available to your API account; defaults to `gpt-5.4-mini`.

Offline demo mode needs neither credential nor network access. API usage may incur provider charges.

## Sample use cases

- Ask `Check whether we can fulfill 220 units of BRG-6204.` Expected: 170 available, shortage 50.
- Search `bearing 6204` or `رمان بلي 6204` and confirm the exact SKU.
- Add order lines in Order Planner and export a draft report.
- Upload a CSV using the schema below to calculate dashboard values from that inventory.

## Inventory CSV schema

Required columns: `sku,name,aliases,category,warehouse,on_hand,reserved,reorder_point,unit_cost_sar`. Separate aliases with `|`. Quantities and costs must be nonnegative; reserved stock cannot exceed on-hand. Upload size is capped at 10 MB. Stock is aggregated by SKU across warehouses, with available = on_hand - reserved.

## Reports and Arabic

PDF reports contain summary metrics, order quantities, shortage and cost calculations, warehouse records, and current evaluation summaries. Arabic RTL shaping is not supported by the installed ReportLab default fonts, so the PDF discloses the limitation. The Reports workspace also offers a UTF-8 HTML report with RTL Arabic cells as a readable alternative; no system font files are redistributed.

## Testing

```bash
python -m pytest -q tests
```

Tests cover arithmetic, matching/Arabic aliases, stock aggregation, invalid inventory, offline agent, report generation, BRG-6204 regression, generated corpus validation, BIO alignment, byte-for-byte reproducibility, NLP preprocessing, baseline metrics, semantic ranking, attention, and Streamlit page smoke checks. Run `python scripts/validate_synthetic_data.py` for an explicit dataset audit.

## Screenshots

Run the application locally, open the dashboard and order planner, then capture screenshots with your operating system. Use synthetic/sample data and avoid exposing API keys.

## Further project documentation

See [architecture](docs/ARCHITECTURE.md), [SDAIA lab mapping](docs/SDAIA_LAB_MAPPING.md), [evaluation summary](docs/EVALUATION_SUMMARY.md), [demo script](docs/DEMO_SCRIPT.md), and [CV summary](docs/PROJECT_CV_SUMMARY.md). Screenshots can be captured locally from Overview, NLP lab, Semantic search, Evaluation, and Reports in the sidebar.

The inventory and all NLP corpora are synthetic. The dashboard loads the 150-SKU catalogue at `data/synthetic/inventory.csv` by default; the compact `sample_inventory.csv` remains available as a fixture. Future work: evaluate on consented or properly licensed natural bilingual requests, assess robust Arabic morphology, evaluate retrieval on a larger held-out query set, improve PDF Arabic shaping with a licensed font, and separately review the existing agent.

## Portfolio summary

Built a bilingual supply-chain intelligence prototype with reproducible synthetic data, deterministic inventory planning, an offline multilingual NLP baseline, rule-based entity extraction, candidate-ranked inventory retrieval, and leakage-aware evaluation reports. Reported metrics are measured only on the included synthetic splits and are not presented as production performance.

## Local Qwen on Apple Silicon

In **Workspace → Copilot mode**, select **Local AI · Qwen3 1.7B** (the default). In **Inventory copilot**, send an Arabic or English message using the composer. The conversation keeps previous replies in the current session. After a stock lookup, you can reply with a quantity such as `220 units` or `٢٢٠ وحدة`. Stock and order replies include warehouse tables. Use **New chat** to clear the thread. Qwen reads the request and selects a stock lookup, order check, or low-stock list using structured JSON. Python resolves the product and calculates the answer from the currently loaded inventory. Expand the trace to inspect the model decision and supporting records.

The local provider uses `127.0.0.1:11434`, a 2,048-token context, batch size 16, CPU inference requested, and unloads after each request. It requires a running Ollama server with `qwen3:1.7b` installed. If free disk falls below 3 GiB or macOS reports elevated memory pressure or new swap-outs, the local request stops. Select **Offline demo** to use the inventory tools while memory is constrained. The earlier standalone inference checks below predate this integration.

Verified on 2026-11-25 on an Apple M1 with 8 GiB unified memory and macOS 15.6.1. Ollama 0.40.18 was installed per user at `~/Applications/Ollama.app` from the [official signed release](https://github.com/ollama/ollama/releases/tag/v0.40.18). The archive passed its release SHA-256 check and the extracted app passed macOS code-signature verification. The API served requests on `127.0.0.1:11434` with cloud features disabled. The local model list contained only `qwen3:1.7b` (Q4_K_M, 1,359,293,444 bytes, digest prefix `8f68893c685c`). Free disk measured 9.61 GiB before installation and approximately 6.59 GiB after the tests. These are whole-volume measurements.

Real requests to `/api/chat` used a 2,048-token context, temperature zero and thinking disabled. The first Arabic response defined inventory and appended a sentence in Chinese. With an Arabic-only system instruction it returned “المخزون هو المحتوى الذي يُخزّن في مخازن الشركة أو المنشأة لاستخدامه في العمليات اليومية.” The first English response described a discrepancy with expected or recorded inventory. With order fulfillment specified it returned “Inventory shortage occurs when the available stock is insufficient to meet the demand for a product.”

In a separate local tool test the model requested `echo_sku` with the sole argument `{"sku":"BRG-6204"}`. The test client checked the name and arguments, ran a Python function that returned its input, and sent that result back in a tool message. Qwen replied “The returned SKU is: BRG-6204.” The model was subsequently unloaded from RAM and remains installed for the next request.

### Install and start

Check space with `df -h "$HOME"`. As a planning estimate for the smaller model, aim for about **8 GiB free before installation** to allow for the model, app, installer, and several GiB for macOS swap and application data. This is a working-space budget. The model shares RAM with macOS and other applications. Start with a 2,048-token context on this 8 GiB Mac.

1. Download the official [Ollama macOS installer](https://ollama.com/download/mac), open the DMG, and copy Ollama.app into `~/Applications` for a per-user installation. Create that directory if needed. The commands below use the bundled CLI directly.
2. In terminal 1 start a foreground server bound to loopback.

   ```bash
   export PATH="$HOME/Applications/Ollama.app/Contents/Resources:$PATH"
   OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 OLLAMA_CONTEXT_LENGTH=2048 ollama serve
   ```

3. In terminal 2 verify the version and listener. The listener should show `127.0.0.1:11434`. Inspect the installed tags, then fetch the selected tag only if missing.

   ```bash
   export PATH="$HOME/Applications/Ollama.app/Contents/Resources:$PATH"
   export OLLAMA_HOST=127.0.0.1:11434
   ollama --version
   lsof -nP -iTCP:11434 -sTCP:LISTEN
   curl --noproxy '*' -fsS http://127.0.0.1:11434/api/version
   ollama list
   ollama show qwen3:1.7b >/dev/null || ollama pull qwen3:1.7b
   ollama run qwen3:1.7b 'مرحبا، اشرح بإيجاز ما هو المخزون'
   ollama run qwen3:1.7b 'Explain inventory shortage in one sentence.'
   ```

### Verify a local tool call

Send this request to the same server. Inspect the JSON for a structured entry in `message.tool_calls` with function name `echo_sku` and argument `sku` equal to `BRG-6204`. The request only supplies a tool schema. It performs no inventory operation.

```bash
curl --noproxy '*' -fsS http://127.0.0.1:11434/api/chat -H 'Content-Type: application/json' -d '{
  "model":"qwen3:1.7b","stream":false,"think":false,
  "options":{"temperature":0,"num_ctx":2048},
  "messages":[{"role":"user","content":"Call echo_sku once with SKU BRG-6204."}],
  "tools":[{"type":"function","function":{"name":"echo_sku","description":"Echo a supplied SKU.","parameters":{"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"]}}}]
}'
```

Unload the model with `ollama stop qwen3:1.7b`. Stop the foreground service with Ctrl-C in terminal 1. If the port is occupied, inspect the listener before starting a second instance. For a connection refusal, check terminal 1 and the local version endpoint. If storage runs low, stop the pull and recheck free space. A writable external volume can hold the model by setting `OLLAMA_MODELS` to a directory on that volume in terminal 1 before starting the service. The internal drive still needs room for the app and swap.
