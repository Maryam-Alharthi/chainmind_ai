"""Presentation components for the ChainMind operations workspace."""
from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets" / "ui"
# Each glyph uses the same 24-unit grid, stroke and optical margin.
PATHS = {
    "overview": "M3 3h7v8H3z M14 3h7v5h-7z M3 15h7v6H3z M14 12h7v9h-7z",
    "copilot": "M4 5h16v12h-8l-5 4v-4H4z M8 9l2 2-2 2 M13 13h3",
    "orders": "M5 4h14v17l-3-2-4 2-4-2-3 2z M8 8h8 M8 12h6",
    "search": "M16 16l5 5 M18 10a8 8 0 1 1-16 0 8 8 0 1 1 16 0 M7 10h6",
    "nlp": "M7 4H4v16h3 M17 4h3v16h-3 M8 9h8 M8 13h5",
    "evaluation": "M3 3v17h18 M7 16v-4 M12 16V8 M17 16V5",
    "reports": "M5 3h9l5 5v13H5z M14 3v6h5 M8 13h8 M8 17h6",
    "catalog": "M3 7l9-5 9 5-9 5z M3 7v10l9 5 9-5V7 M12 12v10",
    "stock": "M3 5h18v15H3z M3 10h18 M9 14h6",
    "alert": "M3 19l9-16 9 16z M12 9v4 M12 16v.3",
    "value": "M3 6h18v13H3z M3 10h18 M14 15h4",
    "warehouse": "M3 20V8l9-5 9 5v12z M7 20V10h10v10 M7 14h10 M7 17h10",
    "database": "M3 6c0-4 18-4 18 0s-18 4-18 0v12c0 4 18 4 18 0V6 M3 12c0 4 18 4 18 0",
    "lock": "M5 10h14v11H5z M8 10V7a4 4 0 0 1 8 0v3 M12 14v3",
}
PAGES = {
    "Overview": ("overview", "Inventory overview", "Availability and replenishment across your warehouse network."),
    "Inventory copilot": ("copilot", "Inventory copilot", "Ask about stock, order quantities and shortages in English or Arabic."),
    "Order planner": ("orders", "Order planner", "Check demand against available units and prepare a replenishment draft."),
    "Semantic search": ("search", "Semantic search", "Find catalog records by SKU, product name or description in either language."),
    "NLP lab": ("nlp", "NLP lab", "Inspect how a supply-chain request is normalized, classified and tagged."),
    "Evaluation": ("evaluation", "Evaluation", "Inspect the measured results on held-out synthetic requests."),
    "Reports": ("reports", "Reports", "Export the loaded inventory and the latest available evaluation."),
}


def glyph(name):
    return (f'<svg aria-hidden="true" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'fill="none" stroke="#79e7e1" stroke-width="1.65" stroke-linejoin="round" '
            f'stroke-linecap="round"><path d="{PATHS[name]}"/></svg>')


def svg_image(markup):
    # Streamlit sanitizes inline SVG elements. Embed the authored artwork as an image.
    data = b64encode(markup.encode()).decode()
    return f'<img alt="" src="data:image/svg+xml;base64,{data}">'


def stylesheet():
    rules = []
    for index, (icon, _, _) in enumerate(PAGES.values()):
        data = b64encode(glyph(icon).encode()).decode()
        rules.append(f'.st-key-workspace_nav [role="radiogroup"] label:has(input[value="{index}"]) '
                     f'{{ --nav-icon: url("data:image/svg+xml;base64,{data}"); }}')
    st.html('<style>' + (ASSETS / 'workspace.css').read_text() + '\n'.join(rules) + '</style>')


def brand():
    mark = svg_image((ASSETS / "chainmind.svg").read_text())
    st.html(f'<div class="cm-brand">{mark}<div><strong>ChainMind <span>AI</span></strong>'
            '<small>Supply chain intelligence</small></div></div>')


def page_header(page, source):
    _, title, description = PAGES[page]
    drawing = svg_image((ASSETS / "depot.svg").read_text()) if page == "Overview" else ""
    st.html(f'<div class="cm-context"><span>Workspace &nbsp;/&nbsp; {escape(page)}</span>'
            f'<span class="cm-source">{svg_image(glyph("database"))}<bdi>{escape(source)}</bdi></span></div>'
            f'<header class="cm-header{" cm-cover" if drawing else ""}"><div>'
            f'<h1>{title}</h1><p>{description}</p></div>'
            f'<div aria-hidden="true" class="cm-depot">{drawing}</div></header>')


def inventory_metrics(summary):
    cells = []
    for icon, label, number, unit in (
        ("catalog", "Catalog SKUs", summary['total_skus'], ""),
        ("stock", "Available inventory", summary['available_units'], "units"),
        ("alert", "Low-stock items", summary['low_stock_skus'], ""),
        ("value", "On-hand value", round(summary['estimated_value_sar']), "SAR"),
    ):
        cells.append(f'<div class="cm-kpi cm-{icon}"><dt>{svg_image(glyph(icon))}{label}</dt>'
                     f'<dd>{number:,} <small>{unit}</small></dd></div>')
    st.html('<dl class="cm-kpis">' + ''.join(cells) + '</dl>')


def panel_heading(icon, title, detail=""):
    st.html(f'<div class="cm-panel-title">{svg_image(glyph(icon))}<h2>{escape(title)}</h2>'
            f'<span>{escape(detail)}</span></div>')


def warehouse_balances(inventory):
    totals = inventory.groupby('warehouse')['available'].sum().sort_values(ascending=False)
    panel_heading("warehouse", "Warehouse balances", f"{len(totals)} sites")
    maximum = max(1, totals.max())
    rows = []
    for site, value in totals.items():
        rows.append(f'<div class="cm-balance"><dt dir="auto">{escape(str(site))}</dt>'
                    f'<dd>{int(value):,}</dd><div aria-hidden="true"><i style="width:{100*value/maximum:.3f}%">'
                    '</i></div></div>')
    st.html('<p class="cm-note">Available units by warehouse</p><dl class="cm-balances">' + ''.join(rows) + '</dl>')


def replenishment_queue(alerts, count):
    panel_heading("alert", "Replenishment queue", f"{count} SKUs")
    if not alerts:
        st.caption("All products are above their reorder points.")
        return
    ranked = sorted(alerts, key=lambda row: row['reorder_point'] - row['available'], reverse=True)
    rows = []
    for item in ranked[:4]:
        short = int(item['reorder_point'] - item['available'])
        rows.append(f'<li><div><code>{escape(str(item["sku"]))}</code>'
                    f'<span dir="auto">{escape(str(item["name"]))}</span></div>'
                    f'<strong>{short:,}</strong></li>')
    st.html('<p class="cm-note">Largest gaps to reorder point · units</p><ol class="cm-queue">' +
            ''.join(rows) + '</ol>')
