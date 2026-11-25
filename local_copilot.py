"""Local Qwen selects a read-only inventory operation from a constrained schema."""
import http.client
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

from engine import assess_order, inventory_lookup, low_stock, normalize, parse_inventory_query

OLLAMA_PORT = 11434
MODEL = "qwen3:1.7b"
ACTIONS = ("stock", "order", "low_stock", "unsupported")
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "product": {"type": "string"},
    },
    "required": ["action", "product"],
    "additionalProperties": False,
}
INSTRUCTION = (
    "Route this English or Arabic inventory request. Choose stock for a product lookup, "
    "order for an order quantity, low_stock for a list of items needing replenishment, "
    "or unsupported for other tasks including placing a purchase. Copy the product name "
    "or SKU verbatim from the request, without the requested quantity. Use an empty "
    "product for low_stock or unsupported. Return JSON matching the schema."
)


class LocalModelError(RuntimeError):
    pass


def _check_resources(baseline=None):
    if shutil.disk_usage(Path.home()).free < 3 * 2**30:
        raise LocalModelError("Local analysis stopped: free disk space fell below 3 GiB.")
    if sys.platform != "darwin":
        return None
    try:
        pressure = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True, timeout=2
        ).strip()
        stats = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=2)
        count = int(re.search(r"^Swapouts:\s*(\d+)", stats, re.M)[1])
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise LocalModelError("Could not check macOS memory status. Try Offline demo.") from exc
    if pressure != "1":
        raise LocalModelError("Local analysis stopped: macOS reports elevated memory pressure.")
    if baseline is not None and count > baseline:
        raise LocalModelError("Local analysis stopped: macOS reported new swap-outs.")
    return count


def _request_plan(prompt):
    baseline = _check_resources()
    body = {
        "model": MODEL, "stream": False, "think": False, "keep_alive": 0,
        "format": PLAN_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 2048, "num_batch": 16, "num_gpu": 0, "num_predict": 96},
        "messages": [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": prompt}],
    }
    connection = http.client.HTTPConnection("127.0.0.1", OLLAMA_PORT, timeout=90)
    done = threading.Event()
    interruptions = []

    def monitor(sock):
        while not done.wait(0.15):
            try:
                _check_resources(baseline)
            except LocalModelError as exc:
                interruptions.append(exc)
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    watcher = None
    try:
        connection.connect()
        watcher = threading.Thread(target=monitor, args=(connection.sock,), daemon=True)
        watcher.start()
        connection.request("POST", "/api/chat", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status == 404:
            raise LocalModelError(f"Ollama could not find {MODEL}. Check the installed models with ollama list.")
        if response.status != 200:
            raise LocalModelError(f"Ollama returned HTTP {response.status}. Check the local server.")
        reply = json.loads(response.read())
        if reply.get("done_reason") == "length" or reply.get("model") != MODEL or not reply.get("done"):
            raise LocalModelError("Incomplete local model response. Try a shorter request.")
        plan = json.loads(reply["message"]["content"])
        if not isinstance(plan, dict) or plan.get("action") not in ACTIONS or not isinstance(plan.get("product"), str):
            raise ValueError("Invalid inventory plan")
        _check_resources(baseline)
        return plan
    except (OSError, http.client.HTTPException) as exc:
        raise LocalModelError("Cannot complete the request to Ollama on 127.0.0.1:11434. Check that ollama serve is running.") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise LocalModelError("Qwen returned an unreadable inventory request. Try rephrasing.") from exc
    finally:
        done.set()
        if watcher:
            watcher.join()
        connection.close()
        if interruptions:
            raise interruptions[0]


def _named_products(inventory, prompt):
    """Identify catalog names and aliases written as whole phrases in the request."""
    text = normalize(prompt)
    skus = set()
    for row in inventory.drop_duplicates("sku").itertuples(index=False):
        names = [row.name, getattr(row, "name_ar", ""), *row.aliases.split("|")]
        for name in names:
            if not isinstance(name, str) or not name.strip():
                continue
            if re.search(r"(?<!\w)" + re.escape(normalize(name)) + r"(?!\w)", text):
                skus.add(row.sku)
                break
    return skus


def local_agent(inventory, prompt):
    """Use Qwen for intent and catalog records for every inventory figure."""
    if not prompt.strip() or len(prompt) > 2000:
        raise LocalModelError("Enter an inventory question of 1–2,000 characters.")
    plan = _request_plan(prompt)
    trace = [{"tool": "ollama.chat", "model": MODEL, "result": plan}]
    arabic = bool(re.search(r"[\u0621-\u064a]", prompt))
    action = plan["action"]
    parsed = parse_inventory_query(inventory, prompt)
    # Resolve literal identifiers from the user's text independently of the model.
    catalog = {str(code).casefold(): str(code) for code in inventory.sku.unique()}
    tokens = re.findall(r"\w+(?:[-./]\w+)*", prompt)
    mentions = {token.casefold() for token in tokens if token.casefold() in catalog
                or re.fullmatch(r"[a-z]{2,6}-[a-z0-9-]*\d[a-z0-9-]*", token, re.I)}
    explicit_sku = catalog.get(next(iter(mentions))) if len(mentions) == 1 else None
    qty = parsed["requested_quantity"]
    shorthand = explicit_sku and re.fullmatch(re.escape(explicit_sku) + r"\s+\d+", prompt.strip(), re.I)
    if mentions and qty is not None and (parsed["quantity_has_unit"] or shorthand):
        if explicit_sku is None:
            return ("يرجى تحديد رمز صنف واحد من جدول المخزون." if arabic else
                    "Please confirm one SKU from the loaded inventory."), trace
        # Keep unsupported requests (e.g. a purchase instruction) out of execution.
        if action in ("stock", "low_stock"):
            trace.append({"tool": "resolve_order_intent", "result": {
                "reason": "Exact catalog SKU and requested quantity", "sku": explicit_sku, "quantity": qty}})
            action = "order"

    if action == "unsupported":
        return ("اسأل عن توفر صنف أو عجز طلب أو انخفاض المخزون." if arabic else
                "Ask about product availability, an order shortage, or low-stock items."), trace
    # A named product scopes an availability question to that catalog record.
    # Keep explicit low-stock/replenishment requests on the list route.
    named_stock_sku = None
    if action == "low_stock" and not parsed["quantity_has_unit"]:
        list_cue = re.search(
            r"low[- ]stock|reorder|replenish|below|حد (?:اعاده )?الطلب|اعاده طلب|تجديد المخزون|منخفض|نقص",
            normalize(prompt),
        )
        if not list_cue:
            products = {explicit_sku} if mentions else _named_products(inventory, prompt)
            if products:
                if None in products or len(products) != 1:
                    return ("يرجى تحديد رمز صنف واحد من جدول المخزون." if arabic else
                            "Please confirm one SKU from the loaded inventory."), trace
                named_stock_sku = next(iter(products))
                action = "stock"
                trace.append({"tool": "resolve_stock_intent", "result": {
                    "reason": "Single catalog product in the request", "sku": named_stock_sku}})
    if action == "low_stock":
        rows = low_stock(inventory)
        trace.append({"tool": "list_low_stock", "arguments": {}, "result": rows})
        heading = "الأصناف عند حد إعادة الطلب أو تحته" if arabic else "Items at or below their reorder point"
        lines = [f"{row['sku']} · {row['name']} — {int(row['available'])}" for row in rows]
        return "\n".join([f"{heading} ({len(rows)})", *lines]), trace

    phrase = plan["product"].strip()
    sku = parsed["resolved_sku"]
    if named_stock_sku:
        sku = named_stock_sku
    elif phrase and normalize(phrase) in normalize(prompt):
        sku = parse_inventory_query(inventory, phrase)["resolved_sku"]
    if mentions:
        sku = explicit_sku
    stock = inventory_lookup(inventory, sku) if sku else {}
    if "available" not in stock:
        return ("يرجى تحديد رمز صنف واحد من جدول المخزون." if arabic else
                "Please confirm one SKU from the loaded inventory."), trace
    sku = stock["sku"]
    if action == "order" and qty is not None:
        order = assess_order(inventory, sku, qty)
        trace.append({"tool": "check_order", "arguments": {"sku": sku, "quantity": qty}, "result": order})
        if "error" in order:
            return order["error"], trace
        if arabic:
            return f"{sku} — المطلوب: {qty}، المتاح: {order['available']}، العجز: {order['shortage']}.", trace
        return f"{sku} — Requested: {qty}. Available: {order['available']}. Shortage: {order['shortage']}.", trace
    trace.append({"tool": "get_stock", "arguments": {"sku": sku}, "result": stock})
    if arabic:
        return f"{sku} — المتاح: {stock['available']}. كم وحدة تحتاج؟", trace
    return f"{sku} — Available: {stock['available']}. How many units do you need?", trace
