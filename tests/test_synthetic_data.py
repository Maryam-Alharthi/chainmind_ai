import csv
import hashlib
import json
from pathlib import Path

from scripts.generate_synthetic_data import generate, SEED
from scripts.validate_synthetic_data import validate

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'synthetic'

def test_generated_corpora_validate_and_preserve_brg_regression():
    counts=validate(DATA)
    assert counts=={'inventory_skus':150,'inventory_rows':151,'requests':300,'ner_examples':100,'semantic_pairs':100,'challenge_requests':100,'challenge_search':100}
    rows=list(csv.DictReader((DATA/'inventory.csv').open(encoding='utf-8')))
    brg=[r for r in rows if r['sku']=='BRG-6204']
    available=sum(int(r['on_hand'])-int(r['reserved']) for r in brg)
    assert available==170 and max(0,220-available)==50

def test_generated_corpora_are_reproducible(tmp_path):
    out=tmp_path/'synthetic'
    generate(out,SEED)
    for name in ('inventory.csv','requests.csv','ner_bio.jsonl','semantic_pairs.csv','challenge_requests.csv','challenge_search.csv','split_manifest.json'):
        current=(DATA/name).read_bytes(); regenerated=(out/name).read_bytes()
        assert hashlib.sha256(current).digest()==hashlib.sha256(regenerated).digest()

def test_ner_bio_alignment_covers_required_entity_types():
    examples=[json.loads(line) for line in (DATA/'ner_bio.jsonl').read_text(encoding='utf-8').splitlines()]
    types={tag.split('-',1)[1] for ex in examples for tag in ex['bio_tags'] if tag!='O'}
    assert types=={'SKU','QUANTITY','PRODUCT','SUPPLIER','WAREHOUSE'}
    for ex in examples:
        sku=[e['text'] for e in ex['entities'] if e['type']=='SKU']
        assert len(sku)==1 and sku[0] in ex['text']
