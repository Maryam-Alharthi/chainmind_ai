"""Rule-based supply-chain entity extraction, separate from trained NER."""
from __future__ import annotations
import re
from engine import normalize

WAREHOUSES={
 'riyadh':['riyadh','الرياض'],'jeddah':['jeddah','جدة'],'dammam':['dammam','الدمام'],
 'madinah':['madinah','medina','المدينة المنورة'],'abha':['abha','أبها','ابها']}

def extract_entities(text,inventory=None,suppliers=None):
    source=str(text);entities=[]
    def add(kind,value,start,end):
        if value and not any(e['type']==kind and e['start']==start and e['end']==end for e in entities):
            entities.append({'type':kind,'text':value,'start':start,'end':end,'method':'rule-based'})
    for m in re.finditer(r'(?<![\w])(?:[A-Z]{2,5}-[A-Z0-9-]+)(?![\w])',source,re.I):add('SKU',m.group(),m.start(),m.end())
    sku_spans=[(m.start(),m.end()) for m in re.finditer(r'(?<![\w])(?:[A-Z]{2,5}-[A-Z0-9-]+)(?![\w])',source,re.I)]
    masked=list(source)
    for a,b in sku_spans:masked[a:b]=' '* (b-a)
    quantity_pattern=r'(?<!\w)(\d+(?:[.,]\d+)?)(?=\s*(?:units?|pieces?|pcs?|items?|قطعة|قطع|وحدة|وحدات|حبة|حبات))'
    for m in re.finditer(quantity_pattern,''.join(masked),re.I):add('QUANTITY',m.group(1),m.start(1),m.end(1))
    if inventory is not None:
        candidates=[]
        for _,row in inventory.drop_duplicates('sku').iterrows():
            for name in [str(row.get('name','')),str(row.get('name_ar','')),*str(row.get('aliases','')).split('|')]:
                name=name.strip()
                if len(name)>=3:candidates.append(name)
        for name in sorted(set(candidates),key=len,reverse=True):
            patt=re.compile(re.escape(name),re.I)
            for m in patt.finditer(source):add('PRODUCT',m.group(),m.start(),m.end())
    if suppliers:
        for name in sorted(set(suppliers),key=len,reverse=True):
            for m in re.finditer(re.escape(name),source,re.I):add('SUPPLIER',m.group(),m.start(),m.end())
    normalized=normalize(source)
    for canonical,variants in WAREHOUSES.items():
        for variant in variants:
            # normalized offset search is adequate for canonical Latin/Arabic strings with common Arabic normalization.
            pos=normalized.find(normalize(variant))
            if pos>=0:
                m=re.search(re.escape(variant),source,re.I)
                if m:add('WAREHOUSE',m.group(),m.start(),m.end())
                break
    return sorted(entities,key=lambda e:(e['start'],e['end']))

def parse_ner_jsonl(path,split=None):
    import json
    rows=[json.loads(line) for line in open(path,encoding='utf-8') if line.strip()]
    return [r for r in rows if split is None or r['split']==split]

def template_entities(text):
    """General grammatical-span rules for the released synthetic annotation format."""
    out=[];sku=re.search(r'\b[A-Z]{2,5}-[A-Z0-9-]+\b',text,re.I)
    if sku:out.append(('SKU',sku.group()))
    qty=re.search(r'(?<!\w)(\d+)(?=\s*(?:units?|وحدة))',text,re.I)
    if qty:out.append(('QUANTITY',qty.group(1)))
    if 'SKU' in text and sku:
        m=re.search(r'units? of (.+?) SKU',text,re.I)
        if m:out.append(('PRODUCT',m.group(1).strip()))
        m=re.search(r'from (.+?) at ',text,re.I)
        if m:out.append(('SUPPLIER',m.group(1).strip()))
        m=re.search(r'at (.+?)\.?$',text,re.I)
        if m:out.append(('WAREHOUSE',m.group(1).strip()))
    elif sku:
        m=re.search(r'وحدة من (.+?) الصنف',text)
        if m:out.append(('PRODUCT',m.group(1).strip()))
        after=text[sku.end():]
        m=re.search(r'من (.+?) في ',after)
        if m:out.append(('SUPPLIER',m.group(1).strip()))
        m=re.search(r'في (.+?)\.?$',after)
        if m:out.append(('WAREHOUSE',m.group(1).strip()))
    return out
