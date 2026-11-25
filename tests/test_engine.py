from pathlib import Path
import pandas as pd
from engine import clean_inventory,match_items,inventory_lookup,assess_order,low_stock,demo_agent,assess_order_lines,inventory_summary,normalize,parse_inventory_query

ROOT=Path(__file__).resolve().parents[1]
df=clean_inventory(pd.read_csv(ROOT/'data/sample_inventory.csv',dtype={'sku':str}))

def test_totals():
    b=inventory_lookup(df,'BRG-6204')
    assert b['on_hand']==195 and b['reserved']==25 and b['available']==170

def test_shortage():
    p=assess_order(df,'BRG-6204',220)
    assert p['shortage']==50 and not p['fulfillable']

def test_matching():
    assert match_items(df,'رمان بلي 6204')[0]['sku']=='BRG-6204'

def test_low_stock():
    assert any(x['sku']=='PMP-01' for x in low_stock(df))

def test_demo():
    answer,trace=demo_agent(df,'BRG-6204 220')
    assert '50' in answer and any(x['tool']=='check_order' for x in trace)

def test_arabic_identifier_is_not_inferred_as_requested_quantity():
    parsed=parse_inventory_query(df,'أبغى رمان بلي 6204')
    assert {k:parsed[k] for k in ('resolved_sku','requested_quantity','available_quantity','shortage','quantity_status')} == {
        'resolved_sku':'BRG-6204','requested_quantity':None,'available_quantity':170,'shortage':None,'quantity_status':'missing'}
    answer,trace=demo_agent(df,'أبغى رمان بلي 6204')
    assert 'المتاح: 170' in answer and 'Quantity not specified' in answer and 'كم وحدة تحتاج؟' in answer
    assert not any(row['tool']=='check_order' for row in trace)

def test_arabic_and_english_explicit_quantity_patterns():
    cases=[
        ('220 قطعة من رمان بلي 6204',220),
        ('أحتاج 220 وحدة من BRG-6204',220),
        ('220 units of bearing 6204',220),
        ('Check 220 units of BRG-6204',220),
    ]
    for query,quantity in cases:
        parsed=parse_inventory_query(df,query)
        assert parsed['resolved_sku']=='BRG-6204'
        assert parsed['requested_quantity']==quantity
        assert parsed['available_quantity']==170 and parsed['shortage']==50
        assert parsed['quantity_status']=='specified'

def test_stock_only_queries_do_not_calculate_shortage():
    for query in ('كم المتوفر من رمان بلي 6204','Check stock for BRG-6204'):
        parsed=parse_inventory_query(df,query)
        assert parsed['resolved_sku']=='BRG-6204'
        assert parsed['requested_quantity'] is None and parsed['shortage'] is None
        assert parsed['available_quantity']==170 and parsed['quantity_status']=='missing'

def test_brg_regression():
    p=assess_order(df,'BRG-6204',220)
    assert (p['requested'],p['available'],p['shortage'])==(220,170,50)
    parsed=parse_inventory_query(df,'Check 220 units of BRG-6204')
    assert (parsed['requested_quantity'],parsed['available_quantity'],parsed['shortage'])==(220,170,50)

def test_english_matching_and_alias():
    assert match_items(df,'bearing 6204')[0]['sku']=='BRG-6204'
    assert match_items(df,'رَمَّان بِلِي 6204')[0]['sku']=='BRG-6204'

def test_missing_sku_and_bad_quantity():
    assert inventory_lookup(df,'MISSING')['error']=='SKU not found'
    assert 'error' in assess_order(df,'BRG-6204',1.5)

def test_warehouse_stock_aggregated_once():
    item=inventory_lookup(df,'BRG-6204')
    assert item['available']==sum(x['available'] for x in item['warehouses'])==170

def test_multiple_order_lines_and_cost():
    results=assess_order_lines(df,[{'sku':'BRG-6204','quantity':220}])
    assert results[0]['shortage']==50
    assert results[0]['estimated_replenishment_cost_sar']==2250

def test_inventory_summary_uses_loaded_data():
    s=inventory_summary(df)
    assert s['total_skus']==8 and s['warehouses']==3 and s['estimated_value_sar']>0

def test_invalid_csv_and_missing_columns():
    import pytest
    with pytest.raises(ValueError,match='Missing required columns'):
        clean_inventory(pd.DataFrame({'sku':['A']}))
    bad=df.drop(columns=['available']).copy();bad.loc[0,'reserved']=999999
    with pytest.raises(ValueError,match='Reserved stock'):
        clean_inventory(bad)

def test_ambiguous_candidate_list_is_ranked_without_silent_selection():
    candidates=match_items(df,'bearing')
    assert len(candidates)>1 and candidates[0]['score']>=candidates[1]['score']

def test_pdf_report_generation():
    from app import pdf_report
    import json
    evaluation=json.loads((ROOT/'reports/nlp_evaluation.json').read_text(encoding='utf-8'))
    pdf=pdf_report(df,order_lines=[assess_order(df,'BRG-6204',220)],evaluation_summary=evaluation)
    assert pdf.startswith(b'%PDF') and len(pdf)>1000

def test_accessible_html_report_has_rtl_arabic_and_disclosure():
    from app import html_report
    html=html_report(df).decode('utf-8')
    assert '<meta charset="utf-8">' in html and 'dir="rtl"' in html and 'المنتج' in html
    assert 'Synthetic demo data' in html and 'human approval' in html

def test_lab1_nlp_utilities():
    from nlp_utils import normalize_text,mask_pii,text_audit
    assert normalize_text('إِمدادٌ')=='امداد'
    assert '[EMAIL]' in mask_pii('contact ops@example.com')
    assert text_audit('bearing 6204')['token_count']==2
