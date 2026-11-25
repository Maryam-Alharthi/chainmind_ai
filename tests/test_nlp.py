import json
from pathlib import Path

import pandas as pd

from engine import clean_inventory
from nlp_utils import normalize_text,normalize_english,mask_pii,arabic_clitic_segments,text_audit
from nlp.classification import LABELS,load_requests,train_from_rows
from nlp.evaluation import classification_metrics,bootstrap_intervals,retrieval_metrics,ner_entity_metrics
from nlp.ner_baseline import extract_entities,parse_ner_jsonl,template_entities
from nlp.semantic_search import LexicalInventoryIndex,inventory_documents,search_inventory,rerank_with_local_cross_encoder
from nlp.attention import random_attention

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'synthetic'

def inventory():
    return clean_inventory(pd.read_csv(ROOT/'data'/'sample_inventory.csv',dtype={'sku':str}))

def full_inventory():
    return clean_inventory(pd.read_csv(DATA/'inventory.csv',dtype={'sku':str}))

def test_bilingual_text_processing_pii_and_sku_preservation():
    assert normalize_text('إِمْدادٌ   الرياض')=='امداد الرياض'
    assert normalize_english('  BEARING   6204 ' )=='bearing 6204'
    text=mask_pii('Send BRG-6204 to ops@example.com or 0551234567 / +966551234567')
    assert 'BRG-6204' in text and text.count('[PHONE]')==2 and '[EMAIL]' in text
    assert text_audit('فحص BRG-6204')['token_count']>=2

def test_arabic_clitic_heuristic_is_explicit_and_returns_tokens():
    segmented=arabic_clitic_segments('والصمام في المستودع')
    assert segmented and all(isinstance(x,str) for x in segmented)
    assert 'والصمام' not in segmented

def test_intent_baseline_train_only_and_predicts_known_labels():
    rows=load_requests(DATA/'requests.csv')
    model=train_from_rows(rows)
    assert set(model.labels)==set(LABELS)
    assert sum(model.class_docs.values())==sum(r['split']=='train' for r in rows)==210
    assert model.predict_one('Check whether inventory is available for BRG-6204') in LABELS

def test_classification_metrics_and_bootstrap_are_computed():
    y=['a','b','a','b'];pred=['a','a','a','b'];labels=['a','b']
    m=classification_metrics(y,pred,labels)
    assert m['n']==4 and m['accuracy']==.75 and 0<m['macro_f1']<=1
    ci=bootstrap_intervals(y,pred,labels,n_resamples=100,seed=3)
    assert ci['resamples']==100 and len(ci['accuracy_95_ci'])==2

def test_semantic_index_exact_sku_and_arabic_alias_search():
    df=inventory();docs=inventory_documents(df);index=LexicalInventoryIndex(docs)
    assert index.search('BRG-6204',1)[0]['method']=='exact_sku'
    assert index.search('رمان بلي 6204',1)[0]['sku']=='BRG-6204'
    assert search_inventory(df,'bearing 6204',1)[0]['sku']=='BRG-6204'
    assert index.search('Check unknown SKU ZXQ-9001',10)==[]

def test_exact_sku_results_preserve_query_order_and_catalog_data(monkeypatch):
    import nlp.semantic_search as semantic
    def unexpected_load(*args):
        raise AssertionError('Exact SKU lookup should bypass model initialization')
    monkeypatch.setattr(semantic, '_cached_transformer_index', unexpected_load)
    df = inventory()
    query = 'Check brg-6204, BRG-6204 and PMP-01'
    rows = search_inventory(df, query, 2, use_local_transformer=True)
    assert [row['sku'] for row in rows] == ['BRG-6204', 'PMP-01']
    assert all(row['method'] == 'exact_sku' and row['score'] == 1.0 for row in rows)
    assert rows[0]['available'] == 170 and rows[0]['warehouses']
    assert 'model_warning' not in rows[0]
    assert search_inventory(df, query, 1) == rows[:1]
    assert search_inventory(df, 'ZXQ-9001', use_local_transformer=True) == []


def test_optional_embedding_fallback_when_cached_model_unavailable(monkeypatch):
    import nlp.semantic_search as semantic
    def fail(_docs):raise OSError('no local model snapshot')
    monkeypatch.setattr(semantic,'_cached_transformer_index',fail)
    results=semantic.search_inventory(inventory(),'bearing 6204',2,use_local_transformer=True)
    assert results and results[0]['method'] in {'tfidf_cosine','fuzzy_candidate'}
    assert 'model_warning' in results[0]

def test_optional_cross_encoder_fallback_when_model_unavailable(monkeypatch):
    import nlp.semantic_search as semantic
    def fail():raise OSError('no local cross-encoder snapshot')
    monkeypatch.setattr(semantic,'load_cross_encoder',fail)
    original=[{'sku':'BRG-6204','text':'Bearing 6204','method':'sentence_transformer_faiss_hybrid'}]
    result=semantic.rerank_with_local_cross_encoder('bearing 6204',original)
    assert result[0]['method']=='sentence_transformer_faiss_hybrid'
    assert 'rerank_warning' in result[0]

def test_real_multilingual_embeddings_and_faiss_when_model_is_cached():
    import pytest
    pytest.importorskip('sentence_transformers')
    from huggingface_hub import snapshot_download
    try:snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',local_files_only=True)
    except Exception:pytest.skip('Optional multilingual embedding model is not cached.')
    from nlp.semantic_search import LocalSentenceTransformerFAISS
    index=LocalSentenceTransformerFAISS(inventory_documents(inventory()))
    assert index.search('BRG-6204',1)[0]['method']=='exact_sku'
    result=index.search('رمان بلي 6204',3)
    assert result and result[0]['sku']=='BRG-6204'
    assert all(x['method']=='sentence_transformer_faiss_hybrid' for x in result)

def test_real_multilingual_retrieval_ranks_relevant_products_for_bilingual_queries():
    import pytest
    pytest.importorskip('sentence_transformers');pytest.importorskip('faiss')
    from huggingface_hub import snapshot_download
    try:snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42',local_files_only=True,allow_patterns=['*.json','*.txt','*.model','*.safetensors','pytorch_model.bin','1_Pooling/*'])
    except Exception:pytest.skip('Optional multilingual embedding model is not cached.')
    df=full_inventory()
    expected={
        'رمان بلي للمكينة':'BRG-6204',
        'bearing 6204':'BRG-6204',
        'أحتاج محمل 6204':'BRG-6204',
        'machine bearing':'BRG-',
        'برغي ماكينة M6':'SCR-03',
    }
    for query,sku in expected.items():
        results=search_inventory(df,query,top_k=5,use_local_transformer=True)
        assert results and results[0]['method']=='sentence_transformer_faiss_hybrid'
        assert results[0]['sku']==sku or (sku.endswith('-') and results[0]['sku'].startswith(sku)), (query,results[0]['sku'])

def test_real_cross_encoder_reranking_when_model_is_cached():
    import pytest
    pytest.importorskip('sentence_transformers')
    from huggingface_hub import snapshot_download
    try:snapshot_download('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',revision='1427fd652930e4ba29e8149678df786c240d8825',local_files_only=True,allow_patterns=['*.json','*.txt','*.model','*.safetensors','pytorch_model.bin'])
    except Exception:pytest.skip('Optional Cross-Encoder model is not cached.')
    df=full_inventory()
    results=search_inventory(df,'bearing 6204',top_k=10,use_local_transformer=True)
    reranked=rerank_with_local_cross_encoder('bearing 6204',results)
    assert reranked and all(row['method']=='cross_encoder_reranked' for row in reranked)
    assert 'rerank_score' in reranked[0] and not any('rerank_warning' in row for row in reranked)
    assert reranked[0]['sku']=='BRG-6204'

def test_real_pretrained_attention_when_model_is_cached():
    import pytest
    pytest.importorskip('transformers')
    from huggingface_hub import snapshot_download
    try:snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',local_files_only=True)
    except Exception:pytest.skip('Optional multilingual transformer model is not cached.')
    from nlp.attention import pretrained_attention
    result=pretrained_attention('Check BRG-6204 stock in Riyadh.')
    assert result['kind']=='pretrained_transformer' and result['tokens']
    assert len(result['mean'])==len(result['tokens'])

def test_ner_rule_extractor_and_heldout_bio_spans():
    df=inventory()
    found=extract_entities('Need 220 units of BRG-6204 from Gulf Parts Company in Riyadh',df,['Gulf Parts Company'])
    assert {e['type'] for e in found}>={'SKU','QUANTITY','SUPPLIER','WAREHOUSE'}
    heldout=parse_ner_jsonl(DATA/'ner_bio.jsonl','test')
    assert len(heldout)==15 and all(len(x['tokens'])==len(x['bio_tags']) for x in heldout)
    assert len(template_entities(heldout[0]['text']))==5
    assert ner_entity_metrics([('x','SKU','A')],[('x','SKU','A')])['f1']==1

def test_retrieval_metrics_and_random_attention_mask():
    m=retrieval_metrics({'q1':['a','b'],'q2':['x','z']},{'q1':'b','q2':'x'},k=2)
    assert m['recall@2']==1 and m['mrr@2']==.75
    a=random_attention(['need','stock','now'],heads=2,seed=4,causal=True)
    assert a['kind']=='random_demo' and a['mean'][0][1]==0 and abs(sum(a['mean'][1])-1)<1e-8
