"""Validate synthetic dataset schemas, arithmetic, labels, duplicates, BIO spans, and splits."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=ROOT/'data'/'synthetic'
SPLITS={'train','validation','test'}
INTENTS={'stock_inquiry','shortage_check','replenishment','shipping','purchasing'}
ENTITY_TYPES={'SKU','QUANTITY','PRODUCT','SUPPLIER','WAREHOUSE'}
INTENT_CUES={
 'stock_inquiry':re.compile(r'stock|inventory|available|how many|quantity|\bكم\b|الكمية|رصيد|مخزون|المتاح|جاهزة',re.I),
 'shortage_check':re.compile(r'cover|shortage|deficit|compare|need|fulfill|عجز|نقص|يغطي|طلب|تلبية|نحتاج',re.I),
 'replenishment':re.compile(r'reorder|replenish|restock|top-up|إعادة|تزويد|تعبئة|ترميم|منخفض',re.I),
 'shipping':re.compile(r'ship|dispatch|shipment|شحن|صرف',re.I),
 'purchasing':re.compile(r'purchas|procurement|requisition|recommendation|شراء|توريد|مشتري',re.I),
}

def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def normalize(text):return re.sub(r'[^\w\u0600-\u06FF]+',' ',text.lower()).strip()

def near_duplicate_pairs(rows,threshold=.97):
    """Return high-similarity text pairs from different splits for human review."""
    found=[]
    for i,left in enumerate(rows):
        a=normalize(left['text'])
        for right in rows[i+1:]:
            if left['split']==right['split']:continue
            b=normalize(right['text'])
            if a and b and SequenceMatcher(None,a,b).ratio()>=threshold:
                found.append((left['request_id'],right['request_id']))
    return found

def validate(data_dir=DEFAULT_DATA):
    inv=read_csv(data_dir/'inventory.csv');requests=read_csv(data_dir/'requests.csv')
    sem=read_csv(data_dir/'semantic_pairs.csv');challenge_requests=read_csv(data_dir/'challenge_requests.csv');challenge_search=read_csv(data_dir/'challenge_search.csv')
    with (data_dir/'ner_bio.jsonl').open(encoding='utf-8') as f:ner=[json.loads(line) for line in f if line.strip()]
    manifest=json.loads((data_dir/'split_manifest.json').read_text(encoding='utf-8'))
    assert manifest['synthetic'] is True and manifest['seed']>0
    assert len({r['sku'] for r in inv})==150, 'Expected exactly 150 inventory SKUs'
    assert {r['warehouse'] for r in inv}=={'Riyadh','Jeddah','Dammam','Madinah','Abha'}
    assert all(r['synthetic']=='True' and r['name_ar'] and r['category_ar'] and r['description_en'] and r['description_ar'] and '|' in r['aliases'] for r in inv)
    assert len({(r['sku'],r['warehouse']) for r in inv})==len(inv)
    for r in inv:
        on,res,rop,cost=map(float,(r['on_hand'],r['reserved'],r['reorder_point'],r['unit_cost_sar']))
        assert min(on,res,rop,cost)>=0 and res<=on
    brg=[r for r in inv if r['sku']=='BRG-6204']
    available=sum(int(r['on_hand'])-int(r['reserved']) for r in brg)
    assert len(brg)==2 and available==170
    # The preserved fixture continues to prove requested 220 -> shortage 50.
    assert max(0,220-available)==50
    assert len(requests)==300 and all(r['synthetic']=='True' for r in requests)
    assert {r['intent'] for r in requests}==INTENTS
    assert {r['language'] for r in requests}=={'en','ar'}
    assert all(INTENT_CUES[r['intent']].search(r['request_core']) for r in requests), 'Request core text does not support its intent label'
    assert len({normalize(r['text']) for r in requests})==len(requests), 'Duplicate request text found'
    assert not near_duplicate_pairs(requests), 'Near duplicate requests cross split boundaries'
    template_splits=defaultdict(set)
    for r in requests:template_splits[r['template_group']].add(r['split'])
    assert all(len(splits)==1 for splits in template_splits.values()), 'Request template group leaked across splits'
    assert len(ner)==100 and all(r['synthetic'] is True for r in ner)
    for ex in ner:
        assert len(ex['tokens'])==len(ex['bio_tags']), ex['text']
        assert ' '.join(ex['tokens'])==ex['text']
        seen=set()
        for i,tag in enumerate(ex['bio_tags']):
            if tag=='O':continue
            prefix,kind=tag.split('-',1);assert kind in ENTITY_TYPES
            if prefix=='B':seen.add(kind)
            else:assert prefix=='I' and i>0 and ex['bio_tags'][i-1] in {f'B-{kind}',f'I-{kind}'}, ex['text']
        assert seen==ENTITY_TYPES
        for entity in ex['entities']:
            assert ex['bio_tags'][entity['start']]==f"B-{entity['type']}"
            assert ' '.join(ex['tokens'][entity['start']:entity['end']])==entity['text']
            if entity['type']=='SKU':assert re.fullmatch(r'[A-Z]{2,5}-[A-Z0-9-]+',entity['text'])
            if entity['type']=='QUANTITY':assert entity['text'].isdigit()
            if entity['type']=='PRODUCT':
                assert (re.search(r'units of '+re.escape(entity['text'])+r' SKU',ex['text'],re.I) or
                        re.search(r'وحدة من '+re.escape(entity['text'])+r' الصنف',ex['text']))
            if entity['type']=='SUPPLIER':
                assert (re.search(r'from '+re.escape(entity['text'])+r' at ',ex['text'],re.I) or
                        re.search(r'من '+re.escape(entity['text'])+r' في ',ex['text']))
            if entity['type']=='WAREHOUSE':assert ex['text'].rstrip().endswith(entity['text']+' .')
    assert len(sem)==100 and all(r['synthetic']=='True' for r in sem)
    assert {int(r['relevance']) for r in sem}=={0,2}
    by_query=defaultdict(list)
    for r in sem:by_query[r['query_id']].append(r)
    assert len(by_query)==50 and all(sorted(int(x['relevance']) for x in pair)==[0,2] for pair in by_query.values())
    # Across all resources, product families and paired documents stay inside one split.
    family_splits=defaultdict(set)
    for row in inv+requests+ner:
        family_splits[row['family_id']].add(row['split'])
    for pair in sem:
        family_splits[pair['query_family_id']].add(pair['split'])
        family_splits[pair['document_family_id']].add(pair['split'])
    assert all(len(splits)==1 for splits in family_splits.values()), 'Related family records leaked across splits'
    family_to_split={f"FAM-{family}":split for family,split in manifest['family_splits'].items()}
    assert all(family_to_split[r['family_id']]==r['split'] for r in inv+requests+ner)
    assert all(family_to_split[r['query_family_id']]==r['split'] and family_to_split[r['document_family_id']]==r['split'] for r in sem)
    assert all({r['split'] for r in collection}==SPLITS for collection in (requests,ner,sem))
    assert all(r['split'] in SPLITS for r in requests+ner+sem)
    assert len(challenge_requests)==100 and len(challenge_search)==100
    assert all(r['synthetic']=='True' and r['split']=='challenge_test' for r in challenge_requests+challenge_search)
    assert {r['intent'] for r in challenge_requests}==INTENTS
    assert {r['query_type'] for r in challenge_search}>={'en_paraphrase','ar_paraphrase','mixed_typo','ambiguous_family','negative_missing_sku'}
    test_families={f'FAM-{name}' for name,split in manifest['family_splits'].items() if split=='test'}
    assert all(r['family_id'] in test_families for r in challenge_requests)
    positives=[r for r in challenge_search if r['is_negative']=='False']
    negatives=[r for r in challenge_search if r['is_negative']=='True']
    assert len(positives)==75 and len(negatives)==25
    assert all(set(r['expected_skus'].split('|')).issubset({x['sku'] for x in inv}) for r in positives)
    assert all(r['family_id'] in test_families and set(r['expected_skus'].split('|')).issubset({x['sku'] for x in inv if x['family_id']==r['family_id']}) for r in positives)
    assert all(not any(sku in r['query'] for sku in (x['sku'] for x in inv)) for r in positives)
    assert len({normalize(r['query']) for r in challenge_search})==len(challenge_search)
    assert set(manifest['family_splits'].values())==SPLITS
    return {'inventory_skus':len({r['sku'] for r in inv}),'inventory_rows':len(inv),'requests':len(requests),'ner_examples':len(ner),'semantic_pairs':len(sem),
            'challenge_requests':len(challenge_requests),'challenge_search':len(challenge_search)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA);args=p.parse_args()
    print(json.dumps(validate(args.data_dir),indent=2))

if __name__=='__main__':main()
