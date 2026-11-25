"""Train lightweight request baseline and export reproducible held-out evaluation."""
from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nlp.classification import LABELS,load_requests,train_from_rows,save_model
from nlp.evaluation import classification_metrics,bootstrap_intervals,ner_entity_metrics,retrieval_metrics,challenge_retrieval_metrics
from nlp.ner_baseline import parse_ner_jsonl,extract_entities
from nlp.semantic_search import inventory_documents,LexicalInventoryIndex,LocalSentenceTransformerFAISS,load_cross_encoder
from engine import clean_inventory
import pandas as pd

DATA=ROOT/'data'/'synthetic';MODEL=ROOT/'models'/'topic_classifier.json';REPORT=ROOT/'reports'/'nlp_evaluation.json'

def evaluate():
    requests=load_requests(DATA/'requests.csv');model=train_from_rows(requests);save_model(model,MODEL)
    manifest=json.loads((DATA/'split_manifest.json').read_text(encoding='utf-8'))
    results={'dataset':f"synthetic-seed-{manifest['seed']}",'seed':manifest['seed'],'classification':{},'ner':{},'semantic_search':{},'limitations':[]}
    categories={}
    for r in csv.DictReader((DATA/'inventory.csv').open(encoding='utf-8')):categories[r['sku']]=r['category']
    for split in ('validation','test'):
        rows=[r for r in requests if r['split']==split];truth=[r['intent'] for r in rows];pred=model.predict([r['text'] for r in rows])
        block=classification_metrics(truth,pred,LABELS);block['bootstrap_95_ci']=bootstrap_intervals(truth,pred,LABELS,2000,6204)
        block['slices']={}
        slices={'language':lambda r:r['language'],'length':lambda r:'short' if len(r['text'].split())<=12 else 'long',
                'category':lambda r:categories.get(r['sku'],'unknown')}
        for slice_name,keyfn in slices.items():
            groups=defaultdict(list)
            for i,r in enumerate(rows):groups[keyfn(r)].append(i)
            block['slices'][slice_name]={key:classification_metrics([truth[i] for i in ix],[pred[i] for i in ix],LABELS) for key,ix in groups.items()}
        block['errors']=[{'text':r['text'],'expected':truth[i],'predicted':pred[i],'family_id':r['family_id']}
                         for i,r in enumerate(rows) if truth[i]!=pred[i]]
        results['classification'][split]=block
    challenge_requests=list(csv.DictReader((DATA/'challenge_requests.csv').open(encoding='utf-8')))
    challenge_truth=[r['intent'] for r in challenge_requests];challenge_pred=model.predict([r['text'] for r in challenge_requests])
    challenge_block=classification_metrics(challenge_truth,challenge_pred,LABELS)
    challenge_block['slices']={}
    for dimension,keyfn in [('language',lambda r:r['language']),('length',lambda r:'short' if len(r['text'].split())<=10 else 'long')]:
        groups=defaultdict(list)
        for i,row in enumerate(challenge_requests):groups[keyfn(row)].append(i)
        challenge_block['slices'][dimension]={key:classification_metrics([challenge_truth[i] for i in ix],[challenge_pred[i] for i in ix],LABELS) for key,ix in groups.items()}
    challenge_block['errors']=[{'text':r['text'],'expected':challenge_truth[i],'predicted':challenge_pred[i],'family_id':r['family_id']}
                               for i,r in enumerate(challenge_requests) if challenge_truth[i]!=challenge_pred[i]]
    challenge_block['bootstrap_95_ci']=bootstrap_intervals(challenge_truth,challenge_pred,LABELS,2000,6204)
    results['classification']['challenge_test']=challenge_block
    ner_rows=parse_ner_jsonl(DATA/'ner_bio.jsonl');ner_train=[r for r in ner_rows if r['split']=='train'];ner=[r for r in ner_rows if r['split']=='test']
    inventory_for_ner=clean_inventory(pd.read_csv(DATA/'inventory.csv',dtype={'sku':str}))
    known_suppliers=list({e['text'] for row in ner_train for e in row['entities'] if e['type']=='SUPPLIER'})
    gold=[];predicted=[];per_example=[]
    for i,ex in enumerate(ner):
        g={(i,e['type'],e['text'].casefold()) for e in ex['entities']}
        p={(i,e['type'],e['text'].casefold()) for e in extract_entities(ex['text'],inventory_for_ner,known_suppliers)}
        gold.extend(g);predicted.extend(p);per_example.append({'text':ex['text'],'gold':sorted(g),'predicted':sorted(p)})
    results['ner']={'method':'Rule-based SKU/quantity detection plus training-split supplier lexicon, known warehouse names, and current inventory product/alias lookup; not a trained NER model.',
                    'test_examples':len(ner),'metrics':ner_entity_metrics(gold,predicted),
                    'by_entity_type':{kind:ner_entity_metrics([x for x in gold if x[1]==kind],[x for x in predicted if x[1]==kind]) for kind in ('SKU','QUANTITY','PRODUCT','SUPPLIER','WAREHOUSE')},
                    'errors':[x for x in per_example if x['gold']!=x['predicted']]}
    inventory=clean_inventory(pd.read_csv(DATA/'inventory.csv',dtype={'sku':str}));docs=inventory_documents(inventory)
    semantic=list(csv.DictReader((DATA/'semantic_pairs.csv').open(encoding='utf-8')))
    test_pairs=[r for r in semantic if r['split']=='test'];grouped=defaultdict(list)
    for row in test_pairs:grouped[row['query_id']].append(row)
    rankings={};targets={};queries={}
    family_to_split={f'FAM-{k}':v for k,v in manifest['family_splits'].items()}
    product_family={r['sku']:r['family_id'] for r in csv.DictReader((DATA/'inventory.csv').open(encoding='utf-8'))}
    for qid,pairs in grouped.items():
        query=pairs[0]['query'];split=pairs[0]['split'];allowed=[d for d in docs if family_to_split.get(product_family.get(d['sku'],''))==split]
        index=LexicalInventoryIndex(allowed);ranked=index.search(query,len(allowed));rankings[qid]=[r['sku'] for r in ranked]
        targets[qid]=next(r['document_id'] for r in pairs if int(r['relevance'])>0);queries[qid]=query
    metric=retrieval_metrics(rankings,targets,10)
    # Query bootstrap for intervals, fixed 2000 resamples.
    import random
    rng=random.Random(6204);qids=list(rankings);boot_recall=[];boot_rr=[]
    for _ in range(2000):
        sample=[rng.choice(qids) for _ in qids]
        ranks=[(rankings[q].index(targets[q])+1) if targets[q] in rankings[q] else None for q in sample]
        boot_recall.append(sum(r is not None and r<=10 for r in ranks)/len(ranks));boot_rr.append(sum((1/r if r else 0) for r in ranks)/len(ranks))
    def ci(x):x.sort();return [x[int(.025*(len(x)-1))],x[int(.975*(len(x)-1))]]
    challenge_search=list(csv.DictReader((DATA/'challenge_search.csv').open(encoding='utf-8')))
    test_docs=[d for d in docs if family_to_split.get(product_family.get(d['sku'],''))=='test']
    lexical_index=LexicalInventoryIndex(test_docs)
    embedding_index=None;embedding_error=None
    try:embedding_index=LocalSentenceTransformerFAISS(test_docs)
    except Exception as exc:embedding_error=f"Unavailable locally: {type(exc).__name__}: {exc}"
    lexical_ranked={};embedding_ranked={};reranked={};cross_error=None
    challenge_queries=[r['query'] for r in challenge_search]
    for row in challenge_search:
        lexical_ranked[row['query_id']]=lexical_index.search(row['query'],len(test_docs))
        if embedding_index is not None:embedding_ranked[row['query_id']]=embedding_index.search(row['query'],len(test_docs))
    if embedding_index is not None:
        try:
            cross=load_cross_encoder()
            pairs=[]
            for row in challenge_search:
                for candidate in embedding_ranked[row['query_id']]:pairs.append((row['query'],candidate['text']))
            cross_scores=cross.predict(pairs,show_progress_bar=False);cursor=0
            for row in challenge_search:
                base=embedding_ranked[row['query_id']];chunk=cross_scores[cursor:cursor+len(base)];cursor+=len(base)
                reranked[row['query_id']]=sorted([{**item,'rerank_score':float(score)} for item,score in zip(base,chunk)],key=lambda x:x['rerank_score'],reverse=True)
        except Exception as exc:cross_error=f"Unavailable locally: {type(exc).__name__}: {exc}"
    semantic_result={'standard_pair_test':{'queries':len(rankings),'candidate_products_per_query':sorted({len(x) for x in rankings.values()}),
      'metrics':metric,'bootstrap_95_ci':{'resamples':2000,'seed':6204,'recall@10':ci(boot_recall),'mrr@10':ci(boot_rr)},
      'limitation':'Standard pairs use product names in query text; use challenge_test for paraphrase robustness.'},
      'challenge_test':{'lexical_tfidf':challenge_retrieval_metrics(challenge_search,lexical_ranked),
                        'sentence_transformer_faiss':challenge_retrieval_metrics(challenge_search,embedding_ranked) if embedding_index else {'status':embedding_error},
                        'cross_encoder_reranked':challenge_retrieval_metrics(challenge_search,reranked) if reranked else {'status':cross_error or 'Not run; no locally cached reranker.'},
                        'candidate_skus':len(test_docs)},
      'embedding_model':'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2','index':'FAISS IndexFlatIP over normalized multilingual embeddings',
      'errors':[]}
    if embedding_ranked:
        semantic_result['errors']=challenge_retrieval_metrics(challenge_search,embedding_ranked)['errors']
    results['semantic_search']=semantic_result
    results['limitations']=[
      'All evaluation corpora are synthetic and mostly template-generated; scores do not estimate production performance.',
      'Topic classification is a custom Multinomial Naive Bayes baseline trained on train only; validation/test were not used for fitting.',
      'NER rules are evaluated on 15 held-out examples from templated sentences; this is not a learned model. Product entities use the available inventory catalog as a lexicon, so the score is not a catalog-independent generalization estimate.',
      'All challenge queries and relevance labels are synthetic. Use their measurements for controlled comparison, not estimates of production performance.',
      'Pretrained transformer attention is available only when optional dependencies and a model are already cached locally.' ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(results,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
    return results

if __name__=='__main__':
    report=evaluate()
    for split,metrics in report['classification'].items():print(split,'n=',metrics['n'],'accuracy=',round(metrics['accuracy'],4),'macro_f1=',round(metrics['macro_f1'],4))
    print('ner test',report['ner']['metrics'])
    for method in ('lexical_tfidf','sentence_transformer_faiss','cross_encoder_reranked'):
        metrics=report['semantic_search']['challenge_test'][method]
        if 'status' in metrics:print('semantic',method,metrics['status'])
        else:print('semantic',method,{k:metrics.get(k) for k in ('positive_queries','negative_queries','top1_accuracy','recall@10','mrr@10','negative_false_accept_rate_at_fixed_threshold')})
