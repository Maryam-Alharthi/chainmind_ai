import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import local_copilot as local
from engine import clean_inventory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stub_ollama(monkeypatch):
    """Real loopback transport with a controlled response, no model load."""
    requests = []
    state = {"plan": {"action": "order", "product": "BRG-6204"}, "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/api/chat"
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(state["status"])
            self.end_headers()
            response = {"model": local.MODEL, "done": True,
                        "message": {"content": json.dumps(state["plan"])}}
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        monkeypatch.setattr(local, "OLLAMA_PORT", server.server_port)
        monkeypatch.setattr(local, "_check_resources", lambda baseline=None: 0)
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.03})
        worker.start()
        try:
            yield state, requests
        finally:
            server.shutdown()
            worker.join()


def test_local_workspace_order_and_error(stub_ollama):
    state, calls = stub_ollama
    app = AppTest.from_file(ROOT / "app.py", default_timeout=15).run()
    app.radio(key="workspace_nav").set_value("Inventory copilot").run()
    assert app.selectbox(key="copilot_mode").value.startswith("Local AI")
    app.chat_input[0].set_value("Check 220 units of BRG-6204").run()
    assert not app.exception
    assert "Available: 170. Shortage: 50" in app.session_state["chat_turns"][-1]["reply"]
    assert app.session_state["chat_turns"][-1]["trace"][-1]["tool"] == "check_order"
    assert len(calls) == 1 and calls[0]["model"] == "qwen3:1.7b"
    assert calls[0]["options"]["num_batch"] == 16
    assert calls[0]["options"]["num_ctx"] == 2048 and calls[0]["keep_alive"] == 0
    state["status"] = 404
    app.chat_input[0].set_value("Check 220 units of BRG-6204").run()
    assert any("ollama list" in err.value for err in app.error)
    assert "error" in app.session_state["chat_turns"][-1]
    assert len(app.chat_message) == 2
    app.selectbox(key="copilot_mode").select("Offline demo").run()
    app.chat_input[0].set_value("Check 220 units of BRG-6204").run()
    assert len(calls) == 2 and "50" in app.session_state["chat_turns"][-1]["reply"]


@pytest.mark.parametrize(("prompt", "action", "phrase", "check"), [
    ("أبغى رمان بلي 6204", "order", "رمان بلي 6204", "كم وحدة تحتاج"),
    ("أحتاج 220 وحدة من BRG-6204", "order", "BRG-6204", "العجز: 50"),
    ("Show low stock", "low_stock", "", "PMP-01"),
    ("Check stock BRG-9999", "stock", "BRG-6204", "confirm one SKU"),
    ("Buy items", "unsupported", "", "Ask about product"),
])
def test_local_inventory_actions(stub_ollama, prompt, action, phrase, check):
    state, _ = stub_ollama
    state["plan"] = {"action": action, "product": phrase}
    df = clean_inventory(pd.read_csv(ROOT / "data/sample_inventory.csv"))
    answer, _ = local.local_agent(df, prompt)
    assert check in answer


@pytest.mark.parametrize("prompt", [
    "How many units of bearing 6204 are available?",
    "كم المتوفر من رمان بلي 6204؟",
])
def test_workspace_scopes_misclassified_stock_question_to_named_product(stub_ollama, prompt):
    state, _ = stub_ollama
    state["plan"] = {"action": "low_stock", "product": ""}
    app = AppTest.from_file(ROOT / "app.py", default_timeout=15).run()
    app.radio(key="workspace_nav").set_value("Inventory copilot").run()
    app.chat_input[0].set_value(prompt).run()
    assert not app.exception and not app.error
    trace = app.session_state["chat_turns"][-1]["trace"]
    assert trace[0]["result"] == state["plan"]
    assert [step["tool"] for step in trace[1:]] == ["resolve_stock_intent", "get_stock"]
    assert trace[-1]["arguments"] == {"sku": "BRG-6204"}
    assert trace[-1]["result"]["available"] == 170
    assert "170" in app.session_state["chat_turns"][-1]["reply"]
    assert "shortage" not in trace[-1]["result"]


def test_local_uses_loaded_inventory(stub_ollama):
    df = clean_inventory(pd.DataFrame([dict(sku="BRG-6204", name="Bearing", aliases="bearing",
        category="Parts", warehouse="A", on_hand=5, reserved=2, reorder_point=1, unit_cost_sar=2)]))
    answer, _ = local.local_agent(df, "Check 220 units of BRG-6204")
    assert "Available: 3. Shortage: 217" in answer


def test_pressure_stops_before_network(monkeypatch):
    def stop(baseline=None):
        raise local.LocalModelError("memory pressure")
    monkeypatch.setattr(local, "_check_resources", stop)
    with pytest.raises(local.LocalModelError, match="memory pressure"):
        local._request_plan("Check stock")


@pytest.mark.parametrize(("prompt", "action", "answer_text", "tool"), [
    ("Check 220 units of BRG-6204", "low_stock", "Shortage: 50", "check_order"),
    ("أحتاج ٢٢٠ قطعة من brg-6204", "stock", "العجز: 50", "check_order"),
    ("Check stock for BRG-6204", "stock", "Available: 170", "get_stock"),
    ("List items below 220 units", "low_stock", "reorder point", "list_low_stock"),
    ("Check 220 units of BRG-9999", "low_stock", "confirm one SKU", "ollama.chat"),
    ("Check 220 units of BRG-6204 and PMP-01", "low_stock", "confirm one SKU", "ollama.chat"),
    ("Purchase 220 units of BRG-6204", "unsupported", "Ask about product", "ollama.chat"),
])
def test_route_with_exact_order_evidence(stub_ollama, prompt, action, answer_text, tool):
    state, _ = stub_ollama
    state["plan"] = {"action": action, "product": "BRG-6204"}
    inventory = clean_inventory(pd.read_csv(ROOT / "data/sample_inventory.csv"))
    answer, trace = local.local_agent(inventory, prompt)
    assert answer_text in answer
    assert trace[0]["result"]["action"] == action
    assert trace[-1]["tool"] == tool


def test_chat_grounded_followups_and_new_conversation(stub_ollama):
    state, calls = stub_ollama
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=15).run()
    app.radio(key='workspace_nav').set_value('Inventory copilot').run()
    state['plan'] = {'action': 'stock', 'product': 'BRG-6204'}
    app.chat_input[0].set_value('كم المتوفر من BRG-6204؟').run()
    assert app.dataframe[0].value.iloc[0]['المتاح'] == 170
    app.chat_input[0].set_value('٢٢٠ وحدة').run()
    assert '220 وحدة من BRG-6204' in calls[-1]['messages'][-1]['content']
    assert app.dataframe[2].value.to_dict('records') == [{'المطلوب': 220, 'المتاح': 170, 'العجز': 50}]
    assert app.dataframe[3].value['المتاح'].sum() == 170
    app.chat_input[0].set_value('اعرضها في جدول').run()
    assert not app.exception and len(calls) == 2
    assert app.dataframe[4].value.equals(app.dataframe[2].value)
    app.radio(key='workspace_nav').set_value('Overview').run()
    app.radio(key='workspace_nav').set_value('Inventory copilot').run()
    assert len(app.chat_message) == 3 and len(calls) == 2
    app.button(key='new_chat').click().run()
    assert not app.chat_message and not app.session_state['chat_turns']


def test_chat_preserves_provider_markdown_and_sends_history(monkeypatch):
    import copilot_chat
    output = '### Stock comparison\n\n| SKU | Status |\n| --- | --- |\n| BRG-6204 | Available |\n\n- Review warehouse balances.'
    histories = []
    def fake_agent(df, prompt, key, model, *, history):
        histories.append(history)
        return output, []
    monkeypatch.setattr(copilot_chat, 'ai_agent', fake_agent)
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=15).run()
    app.radio(key='workspace_nav').set_value('Inventory copilot').run()
    app.selectbox(key='copilot_mode').set_value('OpenAI API').run()
    next(w for w in app.text_input if w.label == 'OpenAI API key').set_value('test-placeholder').run()
    app.chat_input[0].set_value('Compare stock').run()
    assert any(w.value == output for w in app.markdown)
    app.chat_input[0].set_value('Explain the comparison').run()
    assert histories[-1] == [{'role': 'user', 'content': 'Compare stock'}, {'role': 'assistant', 'content': output}]
    app.selectbox(key='copilot_mode').set_value('Offline demo').run()
    assert not app.chat_message
    app.chat_input[0].set_value('Show low stock').run()
    assert len(app.dataframe[0].value) == 21
