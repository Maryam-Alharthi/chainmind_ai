"""Metrics, sliced evaluation, error examples, and deterministic bootstrap intervals."""
from __future__ import annotations
from collections import Counter
import random

def classification_metrics(y_true,y_pred,labels):
    matrix={a:{b:0 for b in labels} for a in labels}
    for a,b in zip(y_true,y_pred):matrix[a][b]+=1
    f1s=[]
    for label in labels:
        tp=matrix[label][label];fp=sum(matrix[o][label] for o in labels if o!=label);fn=sum(matrix[label][o] for o in labels if o!=label)
        p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
        f1s.append(2*p*r/(p+r) if p+r else 0)
    return {'n':len(y_true),'accuracy':sum(a==b for a,b in zip(y_true,y_pred))/max(1,len(y_true)),
            'macro_f1':sum(f1s)/max(1,len(f1s)),'confusion_matrix':matrix}

def bootstrap_intervals(y_true,y_pred,labels,n_resamples=2000,seed=6204):
    rng=random.Random(seed);n=len(y_true);acc=[];f1=[]
    for _ in range(n_resamples):
        ix=[rng.randrange(n) for _ in range(n)]
        m=classification_metrics([y_true[i] for i in ix],[y_pred[i] for i in ix],labels)
        acc.append(m['accuracy']);f1.append(m['macro_f1'])
    def ci(vals):
        vals.sort();return [vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]]
    return {'method':'percentile bootstrap','resamples':n_resamples,'confidence':.95,'seed':seed,
            'accuracy_95_ci':ci(acc),'macro_f1_95_ci':ci(f1)}

def ner_entity_metrics(gold,pred):
    g=set(gold);p=set(pred);tp=len(g&p)
    precision=tp/len(p) if p else 0;recall=tp/len(g) if g else 0
    return {'gold_entities':len(g),'predicted_entities':len(p),'true_positive':tp,'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0}

def retrieval_metrics(rankings,targets,k=10):
    recalls=[];rr=[]
    for query,ranked in rankings.items():
        target=targets[query]
        rank=(ranked.index(target)+1) if target in ranked else None
        recalls.append(rank is not None and rank<=k);rr.append(1/rank if rank else 0)
    return {'queries':len(rankings),f'recall@{k}':sum(recalls)/max(1,len(recalls)),f'mrr@{k}':sum(rr)/max(1,len(rr))}

def challenge_retrieval_metrics(rows,ranked_by_query,k=10,negative_threshold=.2,seed=6204,n_resamples=2000):
    """Score candidate sets, multi-answer ambiguity queries, and explicit no-match negatives."""
    positive=[r for r in rows if not str(r['is_negative']).lower()=='true']
    negative=[r for r in rows if str(r['is_negative']).lower()=='true']
    ranks=[];top1=[];amb_expected=[];amb_pred=[];slices={}
    errors=[];neg_scores=[]
    for row in rows:
        ranked=ranked_by_query.get(row['query_id'],[]);skus=[x['sku'] for x in ranked]
        score=float(ranked[0]['score']) if ranked else 0.0
        is_neg=str(row['is_negative']).lower()=='true'
        expected=set(filter(None,row['expected_skus'].split('|')))
        ambiguous=len(expected)>1;detected=len(ranked)>1 and score>=negative_threshold and score-float(ranked[1]['score'])<.06
        if not is_neg:
            rank=next((i+1 for i,sku in enumerate(skus) if sku in expected),None)
            ranks.append(rank);top1.append(bool(skus and skus[0] in expected));amb_expected.append(ambiguous);amb_pred.append(detected)
            if rank is None or (rank!=1 and not ambiguous):errors.append({'query':row['query'],'expected_skus':sorted(expected),'top_sku':skus[0] if skus else None,'query_type':row['query_type']})
        else:
            neg_scores.append(score)
            if score>=negative_threshold:errors.append({'query':row['query'],'expected_skus':[],'top_sku':skus[0] if skus else None,'query_type':row['query_type'],'error':'false_accept_at_fixed_threshold'})
    rng=random.Random(seed);boot_recall=[];boot_mrr=[]
    for _ in range(n_resamples):
        sample=[ranks[rng.randrange(len(ranks))] for _ in ranks]
        boot_recall.append(sum(x is not None and x<=k for x in sample)/max(1,len(sample)))
        boot_mrr.append(sum(1/x if x else 0 for x in sample)/max(1,len(sample)))
    def interval(values):
        values=sorted(values);return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]]
    tp=sum(a and p for a,p in zip(amb_expected,amb_pred));ap=sum(amb_pred);ag=sum(amb_expected)
    ambiguity={'n':len(amb_expected),'true_positive':tp,'precision':tp/ap if ap else 0,'recall':tp/ag if ag else 0,
               'f1':2*tp/(ap+ag) if ap+ag else 0}
    for dimension,fn in [('language',lambda r:r['language']),('length',lambda r:'short' if len(r['query'].split())<=8 else 'long'),('query_type',lambda r:r['query_type'])]:
        buckets={}
        for row in positive:buckets.setdefault(fn(row),[]).append(row)
        slices[dimension]={}
        for key,items in buckets.items():
            rr=[];hits=[]
            for row in items:
                ranked=ranked_by_query.get(row['query_id'],[]);expected=set(filter(None,row['expected_skus'].split('|')))
                rank=next((i+1 for i,x in enumerate(ranked) if x['sku'] in expected),None);rr.append(1/rank if rank else 0);hits.append(bool(rank and rank<=k))
            slices[dimension][key]={'n':len(items),'recall_at_k':sum(hits)/len(hits),'mrr':sum(rr)/len(rr)}
    return {'positive_queries':len(positive),'negative_queries':len(negative),'top1_accuracy':sum(top1)/max(1,len(top1)),
      f'recall@{k}':sum(x is not None and x<=k for x in ranks)/max(1,len(ranks)),
      f'mrr@{k}':sum(1/x if x else 0 for x in ranks)/max(1,len(ranks)),
      'negative_false_accept_rate_at_fixed_threshold':sum(x>=negative_threshold for x in neg_scores)/max(1,len(neg_scores)),
      'negative_threshold':negative_threshold,'ambiguity_detection':ambiguity,'slices':slices,
      'bootstrap_95_ci':{'resamples':n_resamples,'seed':seed,f'recall@{k}':interval(boot_recall),f'mrr@{k}':interval(boot_mrr)},
      'errors':errors}
