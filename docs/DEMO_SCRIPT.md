# Portfolio demo script (about 5 minutes)

1. Start `streamlit run app.py` and open **Overview** using the full synthetic catalog: 150 SKUs across five warehouses.
2. Use search with `BRG-6204`, `bearing 6204`, and `رمان بلي 6204`; explain that a candidate score helps search but ambiguous matches need a human selection.
3. In **Order planner** request 220 units of BRG-6204. Show 170 available and shortage 50, then inspect warehouse availability and download a draft report. Emphasize that this does not place a purchase order.
4. Open **NLP lab**. Show a bilingual preprocessing example, the intent labels, and the difference between rule NER and model-based extraction.
5. Open **Semantic search** and compare lexical results with pretrained embeddings if the optional model is cached. If unavailable, explain the local fallback rather than implying a model ran.
6. Open **Evaluation** and discuss the challenge set size, subgroup results, uncertainty intervals, synthetic-template limitations, and recorded error examples.
7. Download the HTML report to demonstrate readable Arabic/English with RTL table cells. Explain that the current PDF font setup does not shape Arabic.

Do not describe synthetic benchmark scores as production accuracy. Keep API credentials out of screenshots and recordings.
