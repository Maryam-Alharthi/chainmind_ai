"""Educational random attention plus optional locally cached pretrained attention."""
from __future__ import annotations
import numpy as np

def random_attention(tokens,heads=2,seed=6204,causal=False):
    """Create deterministic, explicitly RANDOM scaled dot-product attention maps."""
    tokens=list(tokens)[:24]
    if not tokens:return {'tokens':[],'heads':[],'mean':[]}
    rng=np.random.default_rng(seed);n=len(tokens);dim=16
    matrices=[]
    for _ in range(heads):
        q=rng.normal(size=(n,dim));k=rng.normal(size=(n,dim));logits=q@k.T/np.sqrt(dim)
        if causal:logits=np.where(np.triu(np.ones((n,n),dtype=bool),1),-1e9,logits)
        logits-=logits.max(axis=1,keepdims=True);weights=np.exp(logits);weights/=weights.sum(axis=1,keepdims=True)
        matrices.append(weights)
    return {'tokens':tokens,'heads':[x.tolist() for x in matrices],'mean':np.mean(matrices,axis=0).tolist(),'kind':'random_demo'}

def pretrained_attention(text,model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """Load only an already cached Hugging Face model; never downloads from app interaction."""
    try:
        import torch
        from transformers import AutoModel,AutoTokenizer
        from huggingface_hub import snapshot_download
    except ImportError as exc:raise RuntimeError('Optional transformer dependencies are not installed. See requirements-nlp.txt.') from exc
    try:
        snapshot=snapshot_download(model_name,local_files_only=True)
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        model=AutoModel.from_pretrained(snapshot,output_attentions=True,local_files_only=True)
    except OSError as exc:raise RuntimeError(f'Model {model_name} is not cached locally. Download it separately before enabling real attention.') from exc
    encoded=tokenizer(text,return_tensors='pt',truncation=True,max_length=64)
    model.eval()
    with torch.no_grad():output=model(**encoded,output_attentions=True)
    weights=output.attentions[-1][0].mean(dim=0).cpu().numpy()
    tokens=tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    return {'tokens':tokens,'mean':weights.tolist(),'kind':'pretrained_transformer','model':model_name}
