"""Session chat and presentation of read-only inventory tool results."""
from functools import partial
from html import escape
import re

import pandas as pd
import streamlit as st

from engine import ai_agent, demo_agent, normalize
from local_copilot import local_agent
from workspace_ui import ASSETS, svg_image


LABELS = {
    'sku': ('SKU', 'رمز الصنف'), 'name': ('Product', 'الصنف'),
    'available': ('Available', 'المتاح'), 'reserved': ('Reserved', 'المحجوز'),
    'on_hand': ('On hand', 'الرصيد الفعلي'), 'requested': ('Requested', 'المطلوب'),
    'shortage': ('Shortage', 'العجز'), 'warehouse': ('Warehouse', 'المستودع'),
    'reorder_point': ('Reorder point', 'حد إعادة الطلب'), 'score': ('Match score', 'درجة المطابقة'),
}


def evidence(trace):
    """Prefer order records to redundant stock lookups for the same SKU."""
    records = {}
    low_stock = None
    candidates = []
    for step in trace:
        tool, result = step['tool'], step['result']
        if tool in ('get_stock', 'check_order') and isinstance(result, dict) and 'available' in result:
            sku = result['sku']
            if tool == 'check_order' or sku not in records:
                records[sku] = result
        elif tool == 'list_low_stock' and isinstance(result, list):
            low_stock = result
        elif tool == 'search_products' and isinstance(result, list):
            candidates = result
    return list(records.values()), low_stock, candidates


def contextual_prompt(prompt, turns, inventory):
    """Resolve short follow-ups only after a single, verified product answer."""
    if not turns or turns[-1].get('error'):
        return prompt
    records, low_stock, _ = evidence(turns[-1]['trace'])
    if len(records) != 1 or low_stock is not None:
        return prompt
    sku = records[0]['sku']
    text = normalize(prompt).strip(' ؟?!.,')
    ar = bool(re.search('[ء-ي]', text)) or (text.isdigit() and turns[-1]['ar'])
    if re.fullmatch(r'(?:show (?:me )?|which |what )?(?:the )?warehouses?(?: breakdown)?|'
                    r'warehouse availability|(?:في اي |اي |حسب )?المستودعات|توزيع المستودعات', text):
        return f'كم المتوفر من {sku} في المستودعات؟' if ar else f'Check stock for {sku} by warehouse'
    quantity = re.fullmatch(
        r'(?:(?:i need|what about|how about|احتاج|ابغي|اريد)\s+)?(\d{1,7})'
        r'(?:\s*(?:units?|pieces?|pcs|وحده|وحدات|قطعه|قطع))?', text)
    if quantity:
        number = int(quantity[1])
        # A bare numeric model name/alias remains a product query.
        if text.isdigit():
            tokens = inventory[['sku', 'name', 'aliases']].astype(str).agg(' '.join, axis=1)
            if tokens.str.contains(rf'(?<!\d){number}(?!\d)', regex=True).any():
                return prompt
        return f'أحتاج {number} وحدة من {sku}' if ar else f'Check {number} units of {sku}'
    return prompt


def table_followup(prompt):
    text = normalize(prompt).strip(' ؟?!.،,')
    return bool(re.fullmatch(
        r'(?:as a table|(?:show|put|format|display) (?:it|that|this|the results?) (?:in|as) a table|'
        r'(?:show|make) (?:a |the )?table|(?:اعرض(?:ها|ه)? |رتب(?:ها|ه)? )?(?:في |علي شكل |ب)?جدول)', text))


def table(rows, columns, ar):
    data = pd.DataFrame(rows)[columns].rename(columns={c: LABELS[c][int(ar)] for c in columns})
    st.dataframe(data, hide_index=True, width='stretch')


def render_evidence(turn, mode):
    ar = turn['ar']
    def tr(en, arabic):
        return arabic if ar else en

    records, low_stock, candidates = evidence(turn['trace'])
    # Free-form model output uses Streamlit's Markdown parser for tables, lists,
    # headings and code. For the compact local/rules replies use verified records.
    if (mode == 'OpenAI API' and not turn.get('table_only')) or (not records and low_stock is None and not turn.get('table_only')):
        st.markdown(turn['reply'], unsafe_allow_html=False)
    for item in records:
        st.text(f"{item['sku']} · {item['name']}")
        if turn.get('table_only'):
            columns = ['requested', 'available', 'shortage'] if 'requested' in item else ['on_hand', 'reserved', 'available']
            table([item], columns, ar)
            if item.get('warehouses'):
                table(item['warehouses'], ['warehouse', 'available'], ar)
            continue
        if 'requested' in item:
            gap = item['shortage']
            if gap:
                st.markdown(tr(
                    f"**The order is short by {gap:,} units.** You can allocate {item['available']:,} of the {item['requested']:,} requested.",
                    f"**ينقص الطلب {gap:,} وحدة.** يمكن تغطية {item['available']:,} وحدة من أصل {item['requested']:,} وحدة مطلوبة."))
            else:
                st.markdown(tr('**The full order can be fulfilled from available stock.**',
                               '**يمكن تغطية الطلب بالكامل من المخزون المتاح.**'))
            table([item], ['requested', 'available', 'shortage'], ar)
        else:
            st.markdown(tr(f"**{item['available']:,} units are available** after reservations.",
                           f"**المتاح بعد خصم المحجوز هو {item['available']:,} وحدة.**"))
            table([item], ['on_hand', 'reserved', 'available'], ar)
        if item.get('warehouses'):
            st.markdown(tr('**Availability by warehouse**', '**المتاح حسب المستودع**'))
            table(item['warehouses'], ['warehouse', 'available'], ar)
        if item.get('shortage'):
            cost = item.get('estimated_replenishment_cost_sar')
            note = tr(f"Review a draft replenishment for {gap:,} units.",
                      f"يُقترح مراجعة طلب تعويض بمقدار {gap:,} وحدة.")
            if cost is not None:
                note += tr(f" At the recorded unit cost, the estimate is SAR {cost:,.2f}.",
                           f" التكلفة التقديرية وفق سعر الوحدة المسجل هي {cost:,.2f} ريال سعودي.")
            st.markdown(note)
        elif 'requested' not in item and mode != 'OpenAI API':
            st.markdown(tr('How many units do you need for the order?', 'كم وحدة تحتاج للطلب؟'))
    if low_stock is not None:
        if low_stock:
            st.markdown(tr(f'**{len(low_stock)} products are at or below their reorder points.**',
                           f'**بلغ عدد الأصناف عند حد إعادة الطلب أو دونه {len(low_stock)}.**'))
            table(low_stock, ['sku', 'name', 'available', 'reorder_point'], ar)
        else:
            st.markdown(tr('All products are above their reorder points.', 'جميع الأصناف أعلى من حد إعادة الطلب.'))
    elif candidates and not records:
        table(candidates, ['sku', 'name', 'score'], ar)


def show_user(prompt):
    st.html(f'<div class="cm-user-row"><div class="cm-user-bubble" dir="auto" aria-label="User">'
            f'{escape(prompt)}</div></div>')


def show_answer(turn, index, mode, inventory, pdf_report):
    with st.chat_message('assistant', avatar=str(ASSETS / 'chainmind.svg')):
        st.caption(mode if mode != 'Offline demo' else 'Offline demo · rules')
        if turn.get('error'):
            st.error(turn['error'])
            return
        with st.container(key=f"chat_{'ar' if turn['ar'] else 'en'}_{index}"):
            render_evidence(turn, mode)
        label = 'سجل الأدوات والتقرير' if turn['ar'] else 'Tool trace and report'
        if turn.get('table_only'):
            label = 'سجلات الرد السابق' if turn['ar'] else 'Evidence from the previous reply'
        with st.expander(label):
            st.json(turn['trace'], expanded=False)
            records, _, _ = evidence(turn['trace'])
            orders = [r for r in records if 'requested' in r]
            st.download_button('تنزيل تقرير المخزون (PDF بالإنجليزية)' if turn['ar'] else 'Download inventory PDF (English)',
                               data=partial(pdf_report, inventory, order_lines=orders),
                               file_name='chainmind_inventory_report.pdf', mime='application/pdf',
                               key=f'chat_report_{index}', on_click='ignore')


def render_chat(inventory, mode, source_id, source_name, pdf_report, api_key='', model_name=''):
    context = (source_id, mode, model_name)
    if st.session_state.get('chat_context') != context:
        st.session_state['chat_turns'] = []
        st.session_state['chat_context'] = context
    turns = st.session_state['chat_turns']
    head, actions = st.columns([3, 1], vertical_alignment='center')
    with head:
        st.html(f'<header class="cm-chat-head"><h1>Inventory copilot</h1><p><bdi>{escape(source_name)}</bdi>'
                f' · {escape(mode)}</p></header>')
    if actions.button('New chat', key='new_chat', disabled=not turns):
        turns.clear()
        st.rerun()

    prompt = st.chat_input('Message the inventory copilot…', key='copilot_input', max_chars=2000, submit_mode='disable')
    if not turns and not prompt:
        mark = svg_image((ASSETS / 'chainmind.svg').read_text())
        st.html(f'<div class="cm-chat-start">{mark}<h2>What would you like to check?</h2>'
                '<p>Ask about availability, compare an order, or find items to replenish.</p></div>')
        sku = str(inventory.iloc[0]['sku'])
        with st.container(horizontal=True, horizontal_alignment='center', key='chat_suggestions'):
            for i, example in enumerate([f'Check stock for {sku}', f'أحتاج 220 وحدة من {sku}', 'Show low stock']):
                st.button(example, key=f'chat_example_{i}', on_click=st.session_state.update,
                          args=({'copilot_input': example},))
    with st.container(key='chat_thread'):
        for index, turn in enumerate(turns):
            show_user(turn['prompt'])
            show_answer(turn, index, mode, inventory, pdf_report)
        if prompt and prompt.strip():
            show_user(prompt)
            expanded = contextual_prompt(prompt, turns, inventory)
            turn = {'prompt': prompt, 'ar': bool(re.search('[ء-ي]', expanded)), 'trace': []}
            if table_followup(prompt):
                prior = turns[-1]['trace'] if turns and not turns[-1].get('error') else []
                records, low_stock, candidates = evidence(prior)
                if records or low_stock is not None or candidates:
                    turn.update(trace=prior, table_only=True, reply=turns[-1]['reply'])
                else:
                    turn['reply'] = ('ما الصنف أو قائمة المخزون التي تريد عرضها في جدول؟' if turn['ar']
                                     else 'Which product or inventory list would you like to see in a table?')
                turns.append(turn)
                st.rerun()
            with st.spinner('Checking inventory records…'):
                try:
                    if mode == 'Offline demo':
                        reply, trace = demo_agent(inventory, expanded)
                    elif mode == 'OpenAI API':
                        if not api_key.strip():
                            raise ValueError('Enter an API key in the sidebar.')
                        history = [message for previous in turns if not previous.get('error')
                                   for message in ({'role':'user','content':previous['prompt']},
                                                   {'role':'assistant','content':previous['reply']})]
                        reply, trace = ai_agent(inventory, expanded, api_key.strip(), model_name, history=history)
                    else:
                        reply, trace = local_agent(inventory, expanded)
                    turn.update(reply=reply, trace=trace)
                except Exception as exc:
                    message = str(exc) if mode != 'OpenAI API' or not api_key.strip() else 'Check the API key, model and connection.'
                    turn['error'] = f'Copilot unavailable. {message}'
            turns.append(turn)
            st.rerun()
