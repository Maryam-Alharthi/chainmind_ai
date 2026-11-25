import os
import importlib.util
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from engine import clean_inventory, match_items, low_stock, inventory_summary, assess_order_lines
from copilot_chat import render_chat
from nlp_utils import normalize_text, normalize_english, mask_pii, text_audit, arabic_clitic_segments, compare_cached_tokenizers
from nlp.classification import load_model, LABELS
from nlp.semantic_search import search_inventory, rerank_with_local_cross_encoder
from nlp.ner_baseline import extract_entities
from nlp.attention import random_attention, pretrained_attention

from workspace_ui import (ASSETS, PAGES, brand, inventory_metrics, page_header, panel_heading,
                          replenishment_queue, stylesheet, warehouse_balances)

ROOT = Path(__file__).parent
st.set_page_config(page_title='ChainMind AI | Supply chain workspace',
                   page_icon=str(ASSETS / 'chainmind.svg'), layout='wide')
stylesheet()


def pdf_report(df, notes='', order_lines=None, evaluation_summary=None):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4)
    styles=getSampleStyleSheet(); summary=inventory_summary(df); story=[
      Paragraph('CHAINMIND AI | INVENTORY & ORDER REPORT',styles['Title']),
      Paragraph('Generated '+datetime.now().strftime('%Y-%m-%d %H:%M')+' | Synthetic demo inventory',styles['Normal']),Spacer(1,12),
      Paragraph(f"Inventory summary: {summary['total_skus']} SKUs · {summary['available_units']} available units · {summary['low_stock_skus']} replenishment items · {summary['warehouses']} warehouses · Estimated on-hand value SAR {summary['estimated_value_sar']:,.2f}",styles['Normal']),Spacer(1,12)]
    warehouse_summary=df.groupby('warehouse',as_index=False).agg(on_hand=('on_hand','sum'),reserved=('reserved','sum'),available=('available','sum'))
    story.append(Paragraph('Warehouse breakdown',styles['Heading2']))
    wh_data=[['Warehouse','On hand','Reserved','Available']]+[[str(r.warehouse),str(int(r.on_hand)),str(int(r.reserved)),str(int(r.available))] for _,r in warehouse_summary.iterrows()]
    wh_table=Table(wh_data,repeatRows=1);wh_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#172b46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.3,colors.grey)]));story.extend([wh_table,Spacer(1,12)])
    if order_lines:
        story.append(Paragraph('Requested quantities and deterministic availability',styles['Heading2']))
        data=[['SKU','Requested','Available','Shortage','Est. cost SAR']]
        for p in order_lines:
            if 'error' not in p: data.append([p['sku'],str(p['requested']),str(p['available']),str(p['shortage']),str(p['estimated_replenishment_cost_sar'] or 0)])
        t=Table(data,repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#172b46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.3,colors.grey),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.HexColor('#ecf3fa')])]));story.extend([t,Spacer(1,12)])
    if evaluation_summary:
        cls=evaluation_summary['classification']['test'];ner=evaluation_summary['ner'];sem=evaluation_summary['semantic_search']['challenge_test']
        semantic_metrics=sem['sentence_transformer_faiss'] if 'metrics' in sem['sentence_transformer_faiss'] else sem['lexical_tfidf']
        story += [Paragraph('NLP evaluation snapshot (synthetic held-out data)',styles['Heading2']),
          Paragraph(f"Intent baseline: n={cls['n']}, accuracy={cls['accuracy']:.3f}, macro-F1={cls['macro_f1']:.3f}.",styles['Normal']),
          Paragraph(f"Rule NER baseline: n={ner['test_examples']}, exact entity F1={ner['metrics']['f1']:.3f}.",styles['Normal']),
          Paragraph(f"Challenge retrieval ({semantic_metrics.get('positive_queries',0)} positive/ambiguous queries): Recall@10={semantic_metrics.get('recall@10',0):.3f}, MRR@10={semantic_metrics.get('mrr@10',0):.3f}.",styles['Normal']),
          Paragraph('Metrics use synthetic, family-held-out data and are not estimates of production performance.',styles['Italic']),Spacer(1,12)]
    story.append(Paragraph('Inventory by SKU and warehouse',styles['Heading2']))
    data=[['SKU','Product','Warehouse','Available','Reorder']]
    for _,r in df.iterrows(): data.append([str(r.sku),str(r['name'])[:25],str(r.warehouse),str(int(r.available)),str(int(r.reorder_point))])
    t=Table(data,repeatRows=1,colWidths=[67,145,100,65,60]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#172b46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.HexColor('#ecf3fa')]),('GRID',(0,0),(-1,-1),.25,colors.grey),('FONTSIZE',(0,0),(-1,-1),8)]));story.append(t)
    replenish=low_stock(df)
    story.append(Spacer(1,12));story.append(Paragraph('Draft replenishment review (human approval required)',styles['Heading2']))
    rdata=[['SKU','Product','Available','Reorder point','Draft top-up']]+[[str(r['sku']),str(r['name'])[:32],str(int(r['available'])),str(int(r['reorder_point'])),str(max(0,int(r['reorder_point'])-int(r['available'])))] for r in replenish]
    if len(rdata)==1:rdata.append(['—','No current low-stock items','','',''])
    rt=Table(rdata,repeatRows=1);rt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#172b46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.3,colors.grey)]));story.append(rt)
    story += [Spacer(1,12),Paragraph('Replenishment recommendation: review any shortage or SKU at/below reorder point; this is a draft for human approval only.',styles['Normal']),Paragraph('Arabic narrative is available in the dashboard. Arabic PDF shaping is unavailable with the installed default ReportLab fonts; no unreadable Arabic text is embedded.',styles['Italic']),Paragraph('Read-only report. No orders placed and no live inventory modified.',styles['Italic'])]
    doc.build(story);return buf.getvalue()

def html_report(df,evaluation_summary=None,order_lines=None):
    """Accessible UTF-8 HTML report uses system fonts and logical RTL text direction."""
    from html import escape as esc
    summary=inventory_summary(df);now=datetime.now().strftime('%Y-%m-%d %H:%M')
    warehouses=df.groupby('warehouse',as_index=False).agg(on_hand=('on_hand','sum'),reserved=('reserved','sum'),available=('available','sum'))
    lines=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
      '<title>ChainMind Supply Chain Report</title><style>body{font:16px/1.55 system-ui,-apple-system,"Geeza Pro","Noto Naskh Arabic",Tahoma,sans-serif;background:#f8fafc;color:#172033;margin:2rem auto;max-width:1100px;padding:0 1rem}h1,h2{color:#0f172a}table{width:100%;border-collapse:collapse;margin:1rem 0 2rem}th,td{border:1px solid #cbd5e1;padding:.55rem;text-align:left}th{background:#e2e8f0}.ar{direction:rtl;unicode-bidi:plaintext;text-align:right}.notice{padding:1rem;border-left:4px solid #0891b2;background:#ecfeff}</style>',
      '<body><header><h1>ChainMind AI — Supply Chain Report</h1>',f'<p>Generated {esc(now)} · <strong>Synthetic demo data</strong></p></header>',
      '<section><h2>Inventory summary</h2><ul>',
      f"<li>Total SKUs: {summary['total_skus']}</li><li>Available units: {summary['available_units']:,}</li><li>Low-stock SKUs: {summary['low_stock_skus']}</li><li>Warehouses: {summary['warehouses']}</li><li>Estimated on-hand value: SAR {summary['estimated_value_sar']:,.2f}</li></ul></section>",
      '<section><h2>Warehouse breakdown</h2><table><thead><tr><th>Warehouse</th><th>On hand</th><th>Reserved</th><th>Available</th></tr></thead><tbody>']
    for _,r in warehouses.iterrows():lines.append(f"<tr><td>{esc(str(r.warehouse))}</td><td>{int(r.on_hand)}</td><td>{int(r.reserved)}</td><td>{int(r.available)}</td></tr>")
    lines.append('</tbody></table></section>')
    if order_lines:
        lines.append('<section><h2>Order lines and draft replenishment</h2><table><thead><tr><th>SKU</th><th>Requested</th><th>Available</th><th>Shortage</th><th>Recommendation</th></tr></thead><tbody>')
        for row in order_lines:
            if 'error' not in row:lines.append(f"<tr><td>{esc(str(row['sku']))}</td><td>{row['requested']}</td><td>{row['available']}</td><td>{row['shortage']}</td><td>{'Human review for draft replenishment' if row['shortage'] else 'Fulfillable from available stock'}</td></tr>")
        lines.append('</tbody></table></section>')
    if evaluation_summary:
        cls=evaluation_summary['classification']['test'];ner=evaluation_summary['ner'];sem=evaluation_summary['semantic_search']['challenge_test']
        retrieval=sem['sentence_transformer_faiss'] if 'metrics' in sem['sentence_transformer_faiss'] else sem['lexical_tfidf']
        lines += ['<section><h2>NLP evaluation snapshot</h2><ul>',
          f"<li>Intent classifier: n={cls['n']}, accuracy={cls['accuracy']:.3f}, Macro-F1={cls['macro_f1']:.3f}</li>",
          f"<li>Rule-based NER: {ner['test_examples']} examples, exact entity F1={ner['metrics']['f1']:.3f}</li>",
          f"<li>Semantic challenge retrieval: {retrieval['positive_queries']} positive/ambiguous queries, Recall@10={retrieval['recall@10']:.3f}, MRR@10={retrieval['mrr@10']:.3f}</li>",
          '<li>All reported evaluation data are synthetic and do not estimate production performance.</li></ul></section>']
    lines += ['<section><h2>Inventory supporting records</h2><table><thead><tr><th>SKU</th><th>Product</th><th lang="ar" dir="rtl" class="ar">المنتج</th><th>Warehouse</th><th>On hand</th><th>Reserved</th><th>Available</th></tr></thead><tbody>']
    for _,r in df.iterrows():lines.append(f"<tr><td>{esc(str(r.sku))}</td><td>{esc(str(r['name']))}</td><td lang=\"ar\" dir=\"rtl\" class=\"ar\">{esc(str(r.get('name_ar','')))}</td><td>{esc(str(r.warehouse))}</td><td>{int(r.on_hand)}</td><td>{int(r.reserved)}</td><td>{int(r.available)}</td></tr>")
    lines += ['</tbody></table></section><p class="notice">Draft recommendations require human approval. No purchase order was placed and live inventory was not changed. Arabic text is rendered in HTML with RTL direction and local system fonts.</p></body></html>']
    return '\n'.join(lines).encode('utf-8')

@st.cache_data
def load_example(): return pd.read_csv(ROOT/'data'/'synthetic'/'inventory.csv',dtype={'sku':str})

@st.cache_resource
def load_topic_baseline():
    model_path=ROOT/'models'/'topic_classifier.json'
    if not model_path.exists():return None
    return load_model(model_path)

@st.cache_data
def load_ner_supplier_lexicon():
    path=ROOT/'data'/'synthetic'/'ner_bio.jsonl'
    if not path.exists():return []
    return list({e['text'] for line in path.read_text(encoding='utf-8').splitlines() if line
                 for e in json.loads(line)['entities'] if e['type']=='SUPPLIER'})

with st.sidebar:
    brand()
    st.html('<p class="cm-rail-label">Workspace</p>')
    page = st.radio('Workspace navigation', list(PAGES), key='workspace_nav',
                    label_visibility='collapsed', width='stretch')
    st.divider()
    st.html('<p class="cm-rail-label">Data source</p>')
    with st.expander('Load inventory CSV'):
        uploaded = st.file_uploader('Inventory file', type=['csv'], max_upload_size=10)
        st.caption('Required columns')
        st.code('sku, name, aliases, category, warehouse, on_hand, reserved, reorder_point, unit_cost_sar', language=None, wrap_lines=True)
    mode = st.selectbox('Copilot mode', ['Local AI · Qwen3 1.7B', 'Offline demo', 'OpenAI API'], key='copilot_mode')
    if mode == 'Local AI · Qwen3 1.7B':
        st.caption('Local inference through Ollama. Inventory access is read-only.')
    elif mode == 'OpenAI API':
        api_key = st.text_input('OpenAI API key', type='password', value=os.getenv('OPENAI_API_KEY', ''))
        model_name = st.text_input('API model', value=os.getenv('OPENAI_MODEL', 'gpt-5.4-mini'))
    else:
        st.caption('Deterministic rules · no AI model. Inventory access is read-only.')

try:
    if uploaded is not None and uploaded.size > 10*1024*1024: raise ValueError('Upload exceeds 10 MB limit')
    raw=pd.read_csv(uploaded,dtype={'sku':str}) if uploaded is not None else load_example()
    inventory=clean_inventory(raw)
except Exception as exc:
    st.error('Inventory load failed: '+str(exc));st.stop()

source_id = getattr(uploaded, 'file_id', None)
if source_id != st.session_state.get('inventory_source'):
    st.session_state.pop('order_results', None)
    st.session_state.pop('order_lines', None)
    st.session_state['inventory_source'] = source_id
source_name = uploaded.name if uploaded is not None else 'Synthetic inventory'
if page == 'Inventory copilot':
    render_chat(inventory, mode, source_id, source_name, pdf_report,
                api_key if mode == 'OpenAI API' else '', model_name if mode == 'OpenAI API' else '')
else:
    page_header(page, source_name)

if page == 'Overview':
    summary = inventory_summary(inventory)
    alerts = low_stock(inventory)
    inventory_metrics(summary)
    left, right = st.columns([1.15, 1])
    with left.container(border=True, key='sites'):
        warehouse_balances(inventory)
        with st.expander('View warehouse records'):
            balances = inventory.groupby('warehouse', as_index=False)[['on_hand', 'reserved', 'available']].sum()
            st.dataframe(balances, hide_index=True, width='stretch')
    with right.container(border=True, key='replenishment'):
        replenishment_queue(alerts, summary['replenishment_skus'])
        with st.expander('View all low-stock items'):
            st.dataframe(pd.DataFrame(alerts), hide_index=True, width='stretch')
    with st.expander('Available stock by category'):
        category = inventory.groupby('category')['available'].sum().sort_values()
        st.bar_chart(category, horizontal=True, color='#79E7E1')
    panel_heading('catalog', 'Inventory ledger', f'{len(inventory):,} records')
    q = st.text_input('Search SKU, product name, or alias', key='inventory_search', persist_state='session')
    filtered=inventory
    if q.strip():
        candidates=match_items(inventory,q,top_k=inventory.sku.nunique()); valid={x['sku'] for x in candidates if x['score']>=.35}
        filtered=inventory[inventory.sku.isin(valid)]
    st.dataframe(filtered,hide_index=True,width='stretch')
    st.download_button('Download inventory CSV',inventory.to_csv(index=False).encode('utf-8-sig'),file_name='inventory_export.csv',mime='text/csv')
if page == 'Order planner':
    product_options=inventory.groupby('sku').name.first().to_dict()
    with st.form('order_form'):
        panel_heading('orders', 'Order lines')
        lines=[]
        line_count = st.session_state.get('order_line_count', 1)
        for i in range(line_count):
            c1,c2=st.columns([3,1]); sku=c1.selectbox(f'Product line {i+1}', ['—']+list(product_options), format_func=lambda x: x if x=='—' else f'{x} · {product_options[x]}',key=f'order_sku_{i}', persist_state='session')
            qty=c2.number_input(f'Quantity {i+1}',min_value=0,max_value=1000000,value=0,step=1,key=f'order_qty_{i}', persist_state='session')
            if sku!='—' and qty>0: lines.append({'sku':sku,'quantity':qty})
        with st.container(horizontal=True):
            submitted = st.form_submit_button('Calculate order', type='primary')
            if line_count < 5 and st.form_submit_button('Add order line'):
                st.session_state['order_line_count'] = line_count + 1
                st.rerun()
    if submitted:
        st.session_state['order_results']=assess_order_lines(inventory,lines)
        st.session_state['order_lines']=lines
    if 'order_results' in st.session_state:
        results=st.session_state['order_results']
        if not results: st.info('Choose at least one product and a positive quantity.')
        for p in results:
            if 'error' in p: st.error(p['error']);continue
            with st.container(border=True):
                st.markdown(f"**{p['sku']} · {p['name']}** — Requested {p['requested']} · Available {p['available']} · Shortage {p['shortage']}")
                st.caption('Warehouse availability (aggregated total above; warehouses are not double-counted):')
                st.dataframe(pd.DataFrame(p['warehouses']),hide_index=True,width='stretch')
                if p['shortage'] and p['estimated_replenishment_cost_sar'] is not None: st.caption(f"Draft replenishment estimate: SAR {p['estimated_replenishment_cost_sar']:,.2f}")
        st.download_button('Download draft order report PDF',data=pdf_report(inventory,order_lines=results),file_name='chainmind_order_draft.pdf',mime='application/pdf')
if page == 'NLP lab':
    sample_query='Check whether we can fulfill 220 units of BRG-6204. Please email ops@example.com.'
    with st.container(border=True, key='nlp_input'):
        panel_heading('nlp', 'Text inspection')
        text = st.text_area('Arabic or English supply-chain text', value=sample_query, height=100,
                            key='nlp_text', persist_state='session')
    if text.strip():
        p1,p2=st.columns(2)
        p1.markdown('**Original source text**');p1.code(text,language=None)
        processed=mask_pii(text);normalized=normalize_text(processed)
        p2.markdown('**PII-masked and normalized**');p2.code(normalized,language=None)
        audit=text_audit(processed)
        st.caption(f"Regex demo tokenizer · {audit['token_count']} tokens · source text is retained and never overwritten.")
        with st.expander('Tokenization audit'):
            st.write(audit['tokens'])
            if any('\u0600'<=c<='\u06ff' for c in text):
                st.write('Arabic clitic heuristic:',arabic_clitic_segments(text))
                st.caption('The clitic segmenter is a transparent rule heuristic, not a pretrained Arabic morphological analyzer.')
                if st.button('Compare locally cached multilingual and Arabic tokenizers',key='arabic_tokenizer_compare'):
                    try:st.json(compare_cached_tokenizers(text))
                    except Exception as exc:st.info(str(exc))
        baseline=load_topic_baseline()
        if baseline:
            predicted=baseline.predict_one(text);st.metric('Predicted intent · local Naive Bayes baseline',predicted)
        else:st.warning('Topic baseline artifact is missing. Run `python scripts/run_nlp_evaluation.py` to train and save it.')
        entities=extract_entities(text,inventory,load_ner_supplier_lexicon())
        st.markdown('**Entity extraction · rule-based baseline**')
        st.dataframe(pd.DataFrame(entities,columns=['type','text','start','end','method']),hide_index=True,width='stretch')
        if entities and any(e['type']=='SKU' for e in entities):
            sku=next(e['text'] for e in entities if e['type']=='SKU')
            st.markdown('**Matched inventory candidates**')
            st.dataframe(pd.DataFrame(match_items(inventory,sku,top_k=3)),hide_index=True,width='stretch')
        st.caption('Processing stages: preserve input → mask obvious PII → normalize → predict intent → extract rule-based entities → query inventory records. No model is called as an agent.')
    st.divider()
    panel_heading('evaluation', 'Attention inspection')
    attention_text=st.text_input('Short text for attention inspection','Check bearing stock in Riyadh',key='attention_text')
    causal=st.checkbox('Apply causal mask to random demo',value=False)
    if st.button('Show deterministic random attention',key='random_attention'):
        tokens=text_audit(attention_text)['tokens'][:12];result=random_attention(tokens,heads=2,seed=6204,causal=causal)
        st.caption('Randomly initialized Q/K demonstration. These weights are not from a trained model.')
        import altair as alt
        rows=[{'head':f'Head {h+1}','query':result['tokens'][i],'key':result['tokens'][j],'weight':matrix[i][j]} for h,matrix in enumerate(result['heads']) for i in range(len(matrix)) for j in range(len(matrix))]
        heat=alt.Chart(pd.DataFrame(rows)).mark_rect().encode(x=alt.X('key:N',sort=result['tokens'],title='Key token'),y=alt.Y('query:N',sort=result['tokens'],title='Query token'),color=alt.Color('weight:Q',title='Weight')).facet(column='head:N')
        st.altair_chart(heat,width='stretch')
        st.caption('Single-sequence example has no padding tokens. Padding masks are relevant when batching unequal-length sequences.')
    if st.button('Try cached pretrained multilingual attention',key='real_attention'):
        try:
            result=pretrained_attention(attention_text)
            st.caption('Real attention from cached pretrained model: '+result['model'])
            import altair as alt
            mat=result['mean'];tokens=result['tokens'];rows=[{'query':tokens[i],'key':tokens[j],'weight':mat[i][j]} for i in range(len(mat)) for j in range(len(mat))]
            st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_rect().encode(x=alt.X('key:N',title='Key token'),y=alt.Y('query:N',title='Query token'),color=alt.Color('weight:Q',title='Attention')),width='stretch')
        except Exception as exc:st.info(str(exc))

if page == 'Semantic search':
    has_embedding = all(importlib.util.find_spec(m) is not None for m in ('sentence_transformers','faiss'))
    with st.container(border=True, key='searchbox'):
        panel_heading('search', 'Catalog lookup')
        query = st.text_input('Search in Arabic or English', 'رمان بلي 6204', key='semantic_query', persist_state='session')
        with st.expander('Retrieval settings'):
            use_transformer = st.checkbox('Multilingual Sentence-Transformers + FAISS', value=has_embedding,
                                          disabled=not has_embedding, key='dense', persist_state='session')
            rerank = st.checkbox('Cross-Encoder reranking', disabled=not has_embedding,
                                 key='rerank', persist_state='session')
            if not has_embedding:
                st.caption('Install requirements-nlp.txt in the project environment to enable embedding search.')
            st.caption('Models load from the local cache. Exact SKU matches take priority.')
        backend = 'Sentence-Transformers + FAISS' if use_transformer else 'TF-IDF fallback'
        st.caption('Selected retrieval: ' + backend + (' · Cross-Encoder requested' if rerank else ''))
        submitted_search = st.button('Search inventory', type='primary', key='semantic_search')
    if submitted_search and query.strip():
        try:
            with st.spinner('Searching inventory records…'):
                results=search_inventory(inventory,query,top_k=10,use_local_transformer=use_transformer)
                if rerank:results=rerank_with_local_cross_encoder(query,results[:10])
            table=pd.DataFrame(results)
            if not table.empty:
                cols=[x for x in ['sku','name','name_ar','category','score','method','retrieval_method','rerank_score','available','warehouses'] if x in table]
                if 'model_warning' in table:
                    warning=table['model_warning'].dropna()
                    if not warning.empty:st.warning(f"Transformer search unavailable; showing local lexical candidates. {warning.iloc[0]}")
                else:
                    ambiguous=len(results)>1 and results[0]['score']-results[1]['score']<0.06 and results[0]['score']>=0.25
                    if ambiguous:st.warning('Several products have similar scores. Review the candidates and confirm the exact SKU; similarity does not establish interchangeability.')
                if 'rerank_warning' in table:
                    warning=table['rerank_warning'].dropna()
                    if not warning.empty:st.warning(f"Cross-Encoder unavailable; showing original retrieval order. {warning.iloc[0]}")
                if results and results[0].get('method')=='cross_encoder_reranked':st.caption('Search method shown per row: Cross-Encoder reranked (retrieval method is listed alongside it).')
                elif results:st.caption(f"Search method used: {results[0].get('method','unknown')}.")
                st.dataframe(table[cols],hide_index=True,width='stretch')
                st.caption('Stock values are computed by the inventory engine and aggregate reserved-adjusted availability across warehouses.')
            else:st.info('No candidates found.')
        except Exception as exc:st.error(f'Local model unavailable: {exc}')

if page == 'Evaluation':
    report_path=ROOT/'reports'/'nlp_evaluation.json'
    if not report_path.exists():st.warning('Evaluation report missing. Run `python scripts/run_nlp_evaluation.py`.')
    else:
        report=json.loads(report_path.read_text(encoding='utf-8'))
        panel_heading('evaluation', 'Intent classification', 'Naive Bayes')
        classification = []
        for split in ('validation', 'test'):
            block = report['classification'][split]
            bounds = block['bootstrap_95_ci']
            row = {'Split': split.title(), 'Requests': block['n']}
            for metric, label in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')]:
                lo, hi = bounds[metric + '_95_ci']
                row[label] = block[metric]
                row[label + ' · 95% CI'] = f'{lo:.3f} – {hi:.3f}'
            classification.append(row)
        st.dataframe(pd.DataFrame(classification), hide_index=True, width='stretch')
        with st.expander('Test confusion matrix and dataset slices'):
            st.dataframe(pd.DataFrame(report['classification']['test']['confusion_matrix']).T, width='stretch')
            slices = report['classification']['test']['slices']
            rows = [{'Group': kind, 'Slice': label, 'Examples': m['n'], 'Accuracy': m['accuracy'],
                     'Macro-F1': m['macro_f1']} for kind, values in slices.items() for label, m in values.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        ner = report['ner']
        panel_heading('nlp', 'Named entity recognition', f"{ner['test_examples']} test examples")
        st.caption(ner['method'])
        st.dataframe(pd.DataFrame([ner['metrics']]), hide_index=True, width='stretch')
        with st.expander('Results by entity type'):
            st.dataframe(pd.DataFrame(ner['by_entity_type']).T, width='stretch')
        panel_heading('search', 'Semantic retrieval')
        semantic_eval = report['semantic_search']['challenge_test']
        scores = [{'Method': name.replace('_', ' '), 'Positive queries': row.get('positive_queries'),
                   'Recall@10': row['recall@10'], 'MRR@10': row['mrr@10']}
                  for name, row in semantic_eval.items() if isinstance(row, dict) and 'recall@10' in row]
        if scores:
            st.dataframe(pd.DataFrame(scores), hide_index=True, width='stretch')
        with st.expander('Retrieval slices, intervals and model details'):
            st.json(semantic_eval)
        st.markdown('#### Limitations and observed errors')
        for item in report['limitations']:st.markdown('- '+item)
        with st.expander('Classification and NER error examples'):
            st.json({'validation_classification':report['classification']['validation']['errors'],'test_classification':report['classification']['test']['errors'],'ner':report['ner']['errors'],'semantic':report['semantic_search']['errors']})
        st.download_button('Download reproducible evaluation JSON',data=report_path.read_bytes(),file_name='chainmind_nlp_evaluation.json',mime='application/json')

if page == 'Reports':
    evaluation_path=ROOT/'reports'/'nlp_evaluation.json'
    evaluation_data=json.loads(evaluation_path.read_text(encoding='utf-8')) if evaluation_path.exists() else None
    with st.container(border=True, key='pdf'):
        panel_heading('reports', 'Inventory and evaluation', 'PDF')
        st.caption('Warehouse ledger, replenishment summary and available evaluation results. English print layout.')
        st.download_button('Download PDF', data=pdf_report(inventory, evaluation_summary=evaluation_data),
                           file_name='chainmind_supply_chain_report.pdf', mime='application/pdf')
    with st.container(border=True, key='html'):
        panel_heading('catalog', 'Bilingual inventory report', 'HTML')
        st.caption('Arabic and English records with right-to-left text support.')
        st.download_button('Download HTML', data=html_report(inventory, evaluation_data),
                           file_name='chainmind_supply_chain_report.html', mime='text/html')
    if evaluation_path.exists():
        with st.container(border=True, key='json'):
            panel_heading('evaluation', 'Evaluation results', 'JSON')
            st.caption('Measured scores, dataset slices and recorded errors.')
            st.download_button('Download JSON', data=evaluation_path.read_bytes(),
                               file_name='chainmind_nlp_evaluation.json', mime='application/json')
