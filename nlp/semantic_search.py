"""Inventory retrieval: exact SKU, lightweight TF-IDF cosine, optional local ST+FAISS."""
from __future__ import annotations
import re
from functools import lru_cache
import json
from collections import Counter
import math
import numpy as np
from engine import normalize, match_items

def inventory_documents(df):
    docs=[]
    for sku,group in df.groupby('sku',sort=False):
        row=group.iloc[0]
        docs.append({'sku':str(sku),'name':str(row['name']),'name_ar':str(row.get('name_ar','')),
          'category':str(row['category']),'aliases':str(row['aliases']),
          'description_en':str(row.get('description_en','')),'description_ar':str(row.get('description_ar','')),
          'text':' '.join((str(sku),str(row['name']),str(row.get('name_ar','')),str(row['aliases']),str(row['category']),str(row.get('category_ar','')),
                           str(row.get('description_en','')),str(row.get('description_ar','')))),
          'available':int(group['available'].sum()),
          'warehouses':', '.join(f"{r.warehouse}: {int(r.available)}" for _,r in group.iterrows())})
    return docs

def _exact_sku_results(query, documents, by_sku, top_k):
    """Return None for free text, or the ordered, deduplicated catalog hits."""
    text = str(query).strip().casefold()
    refs = [text] if text in by_sku else re.findall(r'(?<!\w)[a-z]{2,5}-[a-z0-9-]+(?!\w)', text)
    if not refs:
        return None
    return [
        {**documents[by_sku[ref]], 'score': 1.0, 'method': 'exact_sku'}
        for ref in dict.fromkeys(refs) if ref in by_sku
    ][:top_k]


def _terms(text):
    s=normalize(text); words=re.findall(r'[\w\u0600-\u06FF]+',s)
    terms=Counter('w:'+w for w in words)
    for w in words:
        padded='^'+w+'$'
        for n in (2,3,4):
            for i in range(max(0,len(padded)-n+1)):terms[f'c{n}:'+padded[i:i+n]]+=1
    return terms

class LexicalInventoryIndex:
    """Offline TF-IDF/cosine baseline. Scores are lexical cosine, not embedding similarity."""
    def __init__(self,documents):
        self.documents=documents
        counters=[_terms(d['text']) for d in documents]
        df=Counter(k for c in counters for k in c)
        n=max(1,len(documents));self.idf={k:math.log(1+n/v)+1 for k,v in df.items()}
        self.vectors=[]
        for c in counters:
            v={k:value*self.idf[k] for k,value in c.items()};norm=math.sqrt(sum(x*x for x in v.values())) or 1
            self.vectors.append({k:x/norm for k,x in v.items()})
        self.by_sku={d['sku'].casefold():i for i,d in enumerate(documents)}
    def search(self,query,top_k=10):
        exact = _exact_sku_results(query, self.documents, self.by_sku, top_k)
        if exact is not None:
            return exact
        c=_terms(query);v={k:value*self.idf.get(k,0) for k,value in c.items()};norm=math.sqrt(sum(x*x for x in v.values())) or 1
        v={k:x/norm for k,x in v.items()}
        scores=[sum(weight*self.vectors[i].get(k,0) for k,weight in v.items()) for i in range(len(self.documents))]
        method='tfidf_cosine'
        if not any(scores):
            names=match_items_from_documents(self.documents,query,top_k)
            return [{**d,'score':float(s),'method':'fuzzy_candidate'} for d,s in names]
        order=sorted(range(len(scores)),key=lambda i:(scores[i],self.documents[i]['sku']),reverse=True)[:max(1,top_k)]
        return [{**self.documents[i],'score':float(scores[i]),'method':method} for i in order]

def match_items_from_documents(documents,query,top_k):
    phrase=normalize(query);ranked=[]
    for d in documents:
        names=[d['sku'],d['name'],d['name_ar'],*d['aliases'].split('|')]
        score=max((__import__('difflib').SequenceMatcher(None,phrase,normalize(n)).ratio() for n in names if n),default=0)
        ranked.append((d,score))
    return sorted(ranked,key=lambda x:x[1],reverse=True)[:top_k]

class LocalSentenceTransformerFAISS:
    """Optional multilingual embeddings. Loads local cache only; does not download models."""
    MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    def __init__(self,documents):
        self.documents=documents
        self.model=_load_sentence_model(self.MODEL)
        vectors=self.model.encode([d['text'] for d in documents],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype('float32')
        # On macOS ARM, importing FAISS before PyTorch inference can trigger an OpenMP crash.
        import faiss
        self.index=faiss.IndexFlatIP(vectors.shape[1]);self.index.add(vectors)
        self.by_sku={d['sku'].casefold():i for i,d in enumerate(documents)}
        self.lexical=LexicalInventoryIndex(documents)
        identity_docs=[{**d,'text':' '.join((d['name'],d['name_ar'],d['aliases']))} for d in documents]
        self.identity_lexical=LexicalInventoryIndex(identity_docs)
    def search(self,query,top_k=10):
        exact = _exact_sku_results(query, self.documents, self.by_sku, top_k)
        if exact is not None:
            return exact
        q=self.model.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
        scores,indices=self.index.search(q,len(self.documents))
        dense_score={self.documents[int(i)]['sku']:float(s) for s,i in zip(scores[0],indices[0]) if i>=0}
        lexical_rows=self.lexical.search(query,len(self.documents))
        lexical_score={row['sku']:float(row['score']) for row in lexical_rows}
        identity_rows=self.identity_lexical.search(query,len(self.documents))
        identity_score={row['sku']:float(row['score']) for row in identity_rows}
        # Dense multilingual retrieval understands paraphrases. A higher-weight
        # name/alias signal anchors catalog language; broad lexical context has
        # lower weight so generic descriptions do not swamp product identity.
        # These cosine features are all normalized to [0, 1].
        ranked=[]
        for sku,index in self.by_sku.items():
            document_sku=self.documents[index]['sku']
            dense=dense_score.get(document_sku,0.0)
            lexical=lexical_score.get(document_sku,0.0)
            identity=identity_score.get(document_sku,0.0)
            fused=.30*dense+.60*identity+.10*lexical
            ranked.append((fused,index,dense_score.get(self.documents[index]['sku'],0.0),
                           identity_score.get(self.documents[index]['sku'],0.0),lexical_score.get(self.documents[index]['sku'],0.0)))
        ranked.sort(key=lambda row:(row[0],self.documents[row[1]]['sku']),reverse=True)
        return [{**self.documents[i],'score':float(fused),'embedding_score':dense,'identity_score':identity,
                 'lexical_score':lexical,
                 'method':'sentence_transformer_faiss_hybrid'}
                for fused,i,dense,identity,lexical in ranked[:max(1,top_k)]]

@lru_cache(maxsize=2)
def _load_sentence_model(model_name):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    # Pin a tested revision and fetch only the files needed by the PyTorch
    # Sentence-Transformers runtime (not ONNX/OpenVINO export variants).
    revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
    snapshot=snapshot_download(model_name,revision=revision,local_files_only=True,
        allow_patterns=['*.json','*.txt','*.model','*.safetensors','pytorch_model.bin','1_Pooling/*'])
    return SentenceTransformer(snapshot,local_files_only=True)

def rerank_with_local_cross_encoder(query,results):
    """Rerank top candidates only when a multilingual cross-encoder is cached locally."""
    if not results or all(row.get('method')=='exact_sku' for row in results):
        return results
    try:
        model=load_cross_encoder()
        scores=model.predict([(query,row['text']) for row in results])
    except Exception as exc:
        return [{**row,'rerank_warning':str(exc)} for row in results]
    reranked=[]
    for row,score in zip(results,scores):
        reranked.append({**row,'retrieval_method':row.get('method'),
                         'method':'cross_encoder_reranked','rerank_score':float(score)})
    return sorted(reranked,key=lambda r:r['rerank_score'],reverse=True)

@lru_cache(maxsize=1)
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    from huggingface_hub import snapshot_download
    model='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    revision='1427fd652930e4ba29e8149678df786c240d8825'
    snapshot=snapshot_download(model,revision=revision,local_files_only=True,
        allow_patterns=['*.json','*.txt','*.model','*.safetensors','pytorch_model.bin'])
    return CrossEncoder(snapshot,local_files_only=True)

def search_inventory(df,query,top_k=10,use_local_transformer=False):
    docs=inventory_documents(df)
    by_sku = {doc['sku'].casefold(): i for i, doc in enumerate(docs)}
    exact = _exact_sku_results(query, docs, by_sku, top_k)
    if exact is not None:
        return exact
    if use_local_transformer:
        try:index=_cached_transformer_index(json.dumps(docs,ensure_ascii=False,sort_keys=True))
        except Exception as exc:
            fallback=LexicalInventoryIndex(docs).search(query,top_k)
            return [{**row,'model_warning':f'{type(exc).__name__}: {exc}'} for row in fallback]
    else:index=LexicalInventoryIndex(docs)
    return index.search(query,top_k)

@lru_cache(maxsize=8)
def _cached_transformer_index(serialized_documents):
    return LocalSentenceTransformerFAISS(json.loads(serialized_documents))
