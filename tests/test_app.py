from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'

def test_streamlit_nlp_and_semantic_pages_smoke():
    at=AppTest.from_file(APP).run()
    assert len(at.exception)==0
    nav = at.radio(key='workspace_nav')
    assert len(nav.options) == 7 and nav.value == 'Overview'
    assert any('150' in node.proto.body and 'Catalog SKUs' in node.proto.body for node in at.get('html'))
    nav.set_value('Semantic search').run()
    at.text_input(key='semantic_query').set_value('BRG-6204')
    at.button(key='semantic_search').click()
    at.run()
    assert len(at.exception)==0
    assert not at.error
    results, = [widget.value for widget in at.dataframe]
    assert results[['sku', 'available', 'method', 'score']].to_dict('records') == [
        {'sku': 'BRG-6204', 'available': 170, 'method': 'exact_sku', 'score': 1.0}
    ]

def test_random_attention_research_visualization_smoke():
    at=AppTest.from_file(APP).run()
    at.radio(key='workspace_nav').set_value('NLP lab').run()
    at.button(key='random_attention').click()
    at.run()
    assert len(at.exception)==0
    assert any('Randomly initialized Q/K' in str(x.value) for x in at.caption)


def test_planner_and_reports_through_workspace_navigation():
    app = AppTest.from_file(APP).run()
    nav = app.radio(key='workspace_nav')
    nav.set_value('Order planner').run()
    app.selectbox(key='order_sku_0').set_value('BRG-6204')
    app.number_input(key='order_qty_0').set_value(220)
    next(b for b in app.button if b.label == 'Add order line').click().run()
    assert len(app.number_input) == 2
    next(b for b in app.button if b.label == 'Calculate order').click().run()
    assert app.session_state['order_results'][0]['shortage'] == 50
    for page in ('Evaluation', 'Reports', 'Order planner'):
        app.radio(key='workspace_nav').set_value(page).run()
        assert not app.exception and not app.error
        if page == 'Reports':
            assert len(app.get('download_button')) == 3
    assert app.number_input(key='order_qty_0').value == 220
