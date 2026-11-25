"""Lightweight, reproducible multilingual Multinomial Naive Bayes baseline."""
from __future__ import annotations
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from nlp_utils import normalize_text

LABELS=['stock_inquiry','shortage_check','replenishment','shipping','purchasing']

def features(text):
    normalized=normalize_text(text)
    tokens=re.findall(r'[\w\u0600-\u06FF]+',normalized)
    out=Counter()
    for i,t in enumerate(tokens):
        out['w:'+t]+=1
        if i:out['b:'+tokens[i-1]+'_'+t]+=1
    # Character trigrams capture SKU fragments and Arabic orthographic variants.
    compact='^'+' '.join(tokens)+'$'
    for word in tokens:
        padded='^'+word+'$'
        for n in (3,4):
            for i in range(max(0,len(padded)-n+1)):out[f'c{n}:'+padded[i:i+n]]+=1
    return out

class MultinomialNB:
    def __init__(self,alpha=0.5):self.alpha=alpha
    def fit(self,examples):
        self.class_docs=Counter();self.feature_counts=defaultdict(Counter);self.class_totals=Counter();self.vocabulary=set()
        for text,label in examples:
            fs=features(text);self.class_docs[label]+=1;self.feature_counts[label].update(fs);self.class_totals[label]+=sum(fs.values());self.vocabulary.update(fs)
        self.labels=sorted(self.class_docs)
        if set(self.labels)!=set(LABELS):raise ValueError('Training data must contain every configured intent label')
        self.vocabulary=sorted(self.vocabulary);return self
    def predict_one(self,text):
        fs=features(text);total_docs=sum(self.class_docs.values());vocab_size=max(len(self.vocabulary),1)
        scores={}
        for label in self.labels:
            score=math.log(self.class_docs[label]/total_docs)
            denom=self.class_totals[label]+self.alpha*vocab_size
            for feat,count in fs.items():
                score+=count*math.log((self.feature_counts[label][feat]+self.alpha)/denom)
            scores[label]=score
        return max(scores,key=scores.get)
    def predict(self,texts):return [self.predict_one(x) for x in texts]
    def to_dict(self):
        return {'model_type':'Multinomial Naive Bayes; word/bigram/character n-gram features','alpha':self.alpha,
                'labels':self.labels,'class_docs':dict(self.class_docs),'class_totals':dict(self.class_totals),
                'vocabulary':self.vocabulary,'feature_counts':{k:dict(v) for k,v in self.feature_counts.items()}}
    @classmethod
    def from_dict(cls,data):
        model=cls(data['alpha']);model.labels=data['labels'];model.class_docs=Counter(data['class_docs']);model.class_totals=Counter(data['class_totals'])
        model.vocabulary=data['vocabulary'];model.feature_counts=defaultdict(Counter,{k:Counter(v) for k,v in data['feature_counts'].items()});return model

def load_requests(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def train_from_rows(rows):
    train=[(r['text'],r['intent']) for r in rows if r['split']=='train']
    return MultinomialNB().fit(train)

def save_model(model,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(model.to_dict(),ensure_ascii=False,separators=(',',':')),encoding='utf-8')

def load_model(path):return MultinomialNB.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
