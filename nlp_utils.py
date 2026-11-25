"""Small, dependency-light Lab 1 text utilities; tokenizer claims require a named tokenizer."""
import re
import unicodedata

EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?966|00966|0)?5\d{8}(?!\w)")
TOKEN = re.compile(r"[\u0600-\u06FF]+|[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\d+(?:[.,]\d+)?|[^\w\s]", re.UNICODE)

def normalize_text(text):
    """Unicode normalize and standardize common Arabic letter variants/diacritics."""
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = text.replace('ى', 'ي').replace('ة', 'ه').replace('ـ', '')
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_english(text):
    """Normalize English casing, Unicode compatibility, and whitespace."""
    text=unicodedata.normalize('NFKC',str(text)).casefold()
    return re.sub(r'\s+',' ',text).strip()

def mask_pii(text):
    """Mask emails and Saudi mobile numbers while preserving SKU-like identifiers."""
    text = EMAIL.sub('[EMAIL]', str(text))
    return PHONE.sub('[PHONE]', text)

def tokenize(text):
    """Transparent regex tokenization for audit/demo, not a pretrained tokenizer."""
    return TOKEN.findall(normalize_text(text))

def text_audit(text, tokenizer=None):
    """Return token count and optional real tokenizer metrics when one is passed."""
    tokens = tokenize(text)
    result = {'normalized_text': normalize_text(text), 'tokens': tokens, 'token_count': len(tokens),
              'tokenizer': 'regex-demo' if tokenizer is None else tokenizer.__class__.__name__}
    if tokenizer is not None:
        encoded = tokenizer(str(text), add_special_tokens=False)
        ids = encoded['input_ids']
        words = tokens
        result['subword_count'] = len(ids)
        result['fertility'] = len(ids) / max(len(words), 1)
        unk = getattr(tokenizer, 'unk_token_id', None)
        result['unknown_token_rate'] = (sum(i == unk for i in ids) / len(ids)) if ids and unk is not None else None
    return result

def arabic_clitic_segments(text):
    """Educational heuristic for common attached Arabic conjunction/preposition/article clitics.

    This is a readable rule baseline, not a linguistically complete morphological analyzer.
    """
    tokens=tokenize(text); output=[]
    prefixes=('وال','بال','كال','فال','لل','ال','و','ف','ب','ك','ل')
    for token in tokens:
        if not re.fullmatch(r'[\u0600-\u06FF]+',token):
            output.append(token);continue
        remaining=token; pieces=[]
        for prefix in prefixes:
            if remaining.startswith(prefix) and len(remaining)-len(prefix)>=3:
                pieces.append(prefix);remaining=remaining[len(prefix):]
                if prefix in ('وال','بال','كال','فال') and remaining.startswith('ال'):
                    pieces.append('ال');remaining=remaining[2:]
                break
        if remaining.startswith('ال') and len(remaining)>4:
            pieces.append('ال');remaining=remaining[2:]
        pieces.append(remaining)
        output.extend(pieces)
    return output

def compare_cached_tokenizers(text,model_names=None):
    """Compare multilingual/Arabic pretrained tokenizers from local cache only."""
    try:from transformers import AutoTokenizer
    except ImportError as exc:raise RuntimeError('Optional Transformers dependency is not installed; see requirements-nlp.txt.') from exc
    from huggingface_hub import snapshot_download
    model_names=model_names or ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2','CAMeL-Lab/bert-base-arabic-camelbert-mix']
    out=[]
    for name in model_names:
        try:
            snapshot=snapshot_download(name,local_files_only=True)
            tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
            ids=tokenizer.encode(str(text),add_special_tokens=False)
            out.append({'model':name,'tokens':tokenizer.convert_ids_to_tokens(ids),'token_count':len(ids)})
        except Exception:out.append({'model':name,'unavailable':'not cached locally'})
    return out
