"""Generate reproducible, explicitly synthetic ChainMind supply-chain datasets."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

SEED = 62042026
WAREHOUSES = ["Riyadh", "Jeddah", "Dammam", "Madinah", "Abha"]
WAREHOUSES_AR = {"Riyadh": "الرياض", "Jeddah": "جدة", "Dammam": "الدمام", "Madinah": "المدينة المنورة", "Abha": "أبها"}
CATEGORY_AR = {"Mechanical":"ميكانيكا","Electrical":"كهرباء","HVAC":"تكييف وتهوية","Plumbing":"سباكة","Electronics":"إلكترونيات","Safety":"سلامة"}
SUPPLIERS = [
    ("Gulf Parts Company", "شركة قطع الخليج"), ("Al Waha Industrial Supply", "شركة الواحة للتوريد الصناعي"),
    ("Desert Line Components", "مؤسسة خط الصحراء للمكونات"), ("Red Sea Equipment Trading", "تجارة معدات البحر الأحمر"),
    ("Najd Technical Materials", "مواد نجد الفنية"), ("Eastern Gate Industrial", "بوابة الشرق الصناعية"),
    ("Oasis Mechanical Supply", "توريد الواحة الميكانيكي"), ("Summit Electrical Trading", "قمة للتجارة الكهربائية"),
    ("Palm Route HVAC Supplies", "مستلزمات مسار النخيل للتكييف"), ("Horizon Safety Products", "منتجات الأفق للسلامة"),
]
# Five closely related variants per family deliberately remain in the same split.
FAMILIES = [
    ("BRG", "Mechanical", "Bearing", "محمل", ["6204", "6205", "6206", "6207", "6208"], 45),
    ("MTR", "Electrical", "Electric motor", "محرك كهربائي", ["0.75 kW", "1.5 kW", "2 kW", "3 kW", "5 kW"], 1750),
    ("FLT", "HVAC", "HEPA air filter", "فلتر هواء هيبا", ["H10", "H11", "H12", "H13", "H14"], 170),
    ("VLV", "Plumbing", "Brass ball valve", "صمام كروي نحاسي", ["15 mm", "20 mm", "25 mm", "32 mm", "40 mm"], 90),
    ("SNS", "Electronics", "Temperature sensor", "حساس حرارة", ["T10", "T20", "T30", "T40", "T50"], 320),
    ("CBL", "Electrical", "Industrial cable", "كيبل صناعي", ["2 m", "5 m", "10 m", "15 m", "20 m"], 55),
    ("PMP", "Mechanical", "Centrifugal water pump", "مضخة مياه طرد مركزي", ["0.5 HP", "1 HP", "1.5 HP", "2 HP", "3 HP"], 1100),
    ("GSK", "Mechanical", "Rubber gasket", "حشية مطاطية", ["12 mm", "18 mm", "24 mm", "30 mm", "36 mm"], 12),
    ("BPD", "Mechanical", "Brake pad set", "طقم فحمات فرامل", ["A1", "A2", "B1", "B2", "C1"], 140),
    ("BLT", "Mechanical", "Hex bolt", "مسمار سداسي", ["M6", "M8", "M10", "M12", "M16"], 2.5),
    ("NUT", "Mechanical", "Lock nut", "صامولة قفل", ["M6", "M8", "M10", "M12", "M16"], 1.4),
    ("SCR", "Mechanical", "Machine screw", "برغي ماكينة", ["M4", "M5", "M6", "M8", "M10"], 1.1),
    ("FAN", "HVAC", "Axial ventilation fan", "مروحة تهوية محورية", ["150 mm", "200 mm", "250 mm", "300 mm", "350 mm"], 260),
    ("BEL", "Mechanical", "Drive belt", "سير نقل حركة", ["A32", "A38", "A44", "B50", "B62"], 38),
    ("PUL", "Mechanical", "V-belt pulley", "بكرة سير", ["60 mm", "80 mm", "100 mm", "120 mm", "150 mm"], 75),
    ("CBR", "Electrical", "Circuit breaker", "قاطع كهربائي", ["10 A", "16 A", "20 A", "32 A", "40 A"], 42),
    ("REL", "Electrical", "Control relay", "مرحّل تحكم", ["12 V", "24 V", "110 V", "220 V", "380 V"], 28),
    ("FUS", "Electrical", "Cartridge fuse", "مصهر خرطوشي", ["2 A", "5 A", "10 A", "16 A", "25 A"], 8),
    ("LED", "Electronics", "LED indicator lamp", "مصباح بيان ليد", ["red 12 V", "green 12 V", "red 24 V", "green 24 V", "amber 24 V"], 16),
    ("CON", "Electrical", "Power connector", "موصل طاقة", ["2-pin", "3-pin", "4-pin", "5-pin", "6-pin"], 12),
    ("GLV", "Safety", "Protective work gloves", "قفازات عمل واقية", ["size S", "size M", "size L", "size XL", "size XXL"], 24),
    ("MSK", "Safety", "Dust respirator mask", "قناع تنفس للغبار", ["P1", "P2", "P3", "N95", "FFP2"], 18),
    ("GGL", "Safety", "Protective safety glasses", "نظارات سلامة واقية", ["clear", "tinted", "anti-fog", "over-spec", "sealed"], 32),
    ("HLM", "Safety", "Industrial hard hat", "خوذة صناعية", ["white", "yellow", "blue", "red", "vented"], 58),
    ("TAP", "Plumbing", "Thread seal tape", "شريط إحكام القلاوظ", ["12 mm", "19 mm", "25 mm", "12 mm heavy", "19 mm heavy"], 4),
    ("HOS", "Plumbing", "Reinforced water hose", "خرطوم مياه مقوى", ["10 m", "15 m", "20 m", "25 m", "30 m"], 85),
    ("PPE", "Safety", "High-visibility vest", "سترة عالية الوضوح", ["size S", "size M", "size L", "size XL", "size XXL"], 36),
    ("OIL", "Mechanical", "Hydraulic oil filter", "فلتر زيت هيدروليكي", ["HF10", "HF20", "HF30", "HF40", "HF50"], 95),
    ("BAT", "Electronics", "Rechargeable battery pack", "حزمة بطارية قابلة للشحن", ["12 V 2 Ah", "12 V 5 Ah", "18 V 2 Ah", "18 V 5 Ah", "24 V 5 Ah"], 210),
    ("WIP", "HVAC", "Industrial air wiper", "ماسحة هواء صناعية", ["300 mm", "400 mm", "500 mm", "600 mm", "700 mm"], 110),
]
INTENTS = ["stock_inquiry", "shortage_check", "replenishment", "shipping", "purchasing"]
EN_CONTEXT = [
 "This review supports the quarterly maintenance plan for {product}.", "Please reconcile this with the service team's count of {product}.",
 "The night shift flagged the {product} bin for a routine check.", "Use this result when preparing the next regional stock summary for {product}.",
 "The workshop requested a traceable balance for {product} before its next service window.", "Include the {product} line in the depot's weekly review.",
 "A technician asked for an evidence-based check of {product}.", "This item is part of the preventive maintenance kit review for {product}.",
 "The branch is checking its planned usage of {product} for the coming cycle.", "Please add the finding to the stock-control notes for {product}.",
 "The operations team is reconciling the {product} shelf after a cycle count.", "Use the result to support a maintenance schedule review for {product}.",
 "The inventory clerk needs a clear record for the {product} item card.", "A workshop workpack includes {product} and needs a stock check.",
 "This request follows a routine bin audit for {product}.", "The regional planner is reviewing expected consumption of {product}.",
 "Please retain the calculation with the maintenance record for {product}.", "The warehouse team is preparing a branch transfer review for {product}.",
 "A scheduled inspection raised a question about the {product} balance.", "The service desk needs stock evidence before assigning {product}.",
 "This check will inform the next equipment-maintenance review for {product}.", "The branch controller is reconciling a shelf label for {product}.",
 "Please use the latest cycle-count note when reviewing {product}.", "The work-order planner is checking the {product} requirement.",
 "This stock question came from the weekly workshop coordination meeting about {product}.", "The depot is preparing a reserve check for the {product} range.",
 "A maintenance supervisor requested a documented balance for {product}.", "The materials team is validating the next service-kit list containing {product}.",
 "The location lead asked for a careful review of the {product} bin.", "This item was included in a routine replenishment review for {product}."]
AR_CONTEXT = [
 "يدعم هذا الفحص خطة الصيانة الدورية لـ{product}.", "يرجى مطابقة النتيجة مع جرد فريق الخدمة لـ{product}.",
 "أشار فريق المناوبة الليلية إلى ضرورة مراجعة خانة {product}.", "استخدم النتيجة في ملخص المخزون الإقليمي القادم للصنف {product}.",
 "طلبت الورشة رصيداً موثقاً لـ{product} قبل موعد الخدمة التالي.", "أضف الصنف {product} إلى مراجعة المستودع الأسبوعية.",
 "طلب فني فحصاً موثقاً للصنف {product}.", "يدخل هذا الصنف ضمن مراجعة عدة الصيانة الوقائية لـ{product}.",
 "يراجع الفرع الاستخدام المخطط للصنف {product} في الدورة القادمة.", "يرجى إضافة النتيجة إلى ملاحظات ضبط المخزون للصنف {product}.",
 "يطابق فريق العمليات رف {product} بعد الجرد الدوري.", "استخدم النتيجة لدعم مراجعة جدول الصيانة الخاص بـ{product}.",
 "يحتاج أمين المخزن إلى سجل واضح لبطاقة الصنف {product}.", "تتضمن حزمة عمل الورشة الصنف {product} وتحتاج إلى فحص مخزون.",
 "جاء هذا الطلب بعد تدقيق اعتيادي لخانة {product}.", "يراجع المخطط الإقليمي الاستهلاك المتوقع للصنف {product}.",
 "يرجى حفظ الحساب مع سجل الصيانة للصنف {product}.", "يجهز فريق المستودع مراجعة تحويل للصنف {product} بين الفروع.",
 "أثار الفحص المجدول سؤالاً حول رصيد {product}.", "يحتاج مكتب الخدمة إلى دليل مخزون قبل تخصيص {product}.",
 "يدعم هذا الفحص المراجعة القادمة لصيانة المعدات المتعلقة بـ{product}.", "يطابق مسؤول الفرع ملصق الرف الخاص بـ{product}.",
 "يرجى الرجوع إلى أحدث ملاحظة جرد عند مراجعة {product}.", "يتحقق مخطط أوامر العمل من احتياج {product}.",
 "ورد سؤال المخزون خلال اجتماع تنسيق الورشة الأسبوعي حول {product}.", "يجهز المستودع مراجعة الكمية المحجوزة لمجموعة {product}.",
 "طلب مشرف الصيانة توثيق رصيد {product}.", "يتحقق فريق المواد من قائمة عدة الخدمة القادمة التي تضم {product}.",
 "طلب مسؤول الموقع مراجعة دقيقة لخانة {product}.", "أدرج الصنف {product} في مراجعة إعادة التزويد الدورية."]
EN_REQUESTS = {
 "stock_inquiry": ["Check the available stock for {sku} at {warehouse}.", "How many units of {sku} can we allocate from {warehouse}?", "Please confirm current inventory for {sku} near {warehouse}.", "Look up {sku} stock across our network; nearest site is {warehouse}.", "What quantity of {sku} is on hand for dispatch from {warehouse}?"] ,
 "shortage_check": ["Can stock of {sku} cover an order for {qty} units at {warehouse}?", "Check whether {qty} units of {sku} leave a shortage at {warehouse}.", "Compare demand for {qty} pieces of {sku} with stock in {warehouse}.", "We need {qty} units of {sku}; identify any stock deficit for {warehouse}.", "Is an order of {qty} {sku} units fulfillable using stock near {warehouse}?"] ,
 "replenishment": ["Review the reorder level for {sku} at {warehouse}.", "Should {sku} be replenished at the {warehouse} site?", "Prepare a draft restock quantity for {sku} near {warehouse}.", "Flag {sku} for replenishment if its available balance is low at {warehouse}.", "Check the reorder point and suggest a draft top-up for {sku} in {warehouse}."] ,
 "shipping": ["Check dispatch availability for {sku} from {warehouse}.", "Can {warehouse} ship {qty} units of {sku}?", "Find stock of {sku} suitable for shipment from {warehouse}.", "Review the warehouse allocation for shipping {sku} via {warehouse}.", "Confirm whether {qty} pieces of {sku} can ship from {warehouse}."] ,
 "purchasing": ["Estimate a draft purchase quantity for {sku} after checking {warehouse} stock.", "Review a purchase request for {qty} units of {sku}; use {warehouse} inventory evidence.", "Should procurement review {sku} based on the balance at {warehouse}?", "Prepare a read-only purchasing recommendation for {sku} near {warehouse}.", "Check whether a draft requisition is needed for {qty} {sku} units at {warehouse}."]}
AR_REQUESTS = {
 "stock_inquiry": ["تحقق من الكمية المتاحة للصنف {sku} في مستودع {warehouse}.", "كم وحدة من {sku} يمكن تخصيصها من {warehouse}؟", "أحتاج رصيد المخزون الحالي للصنف {sku} بالقرب من {warehouse}.", "ابحث عن مخزون {sku} في الشبكة، وأقرب موقع هو {warehouse}.", "ما كمية {sku} الجاهزة للصرف من {warehouse}؟"],
 "shortage_check": ["هل يغطي مخزون {sku} طلباً مقداره {qty} وحدة في {warehouse}؟", "احسب العجز المحتمل لطلب {qty} قطعة من {sku} في {warehouse}.", "قارن طلب {qty} وحدة من {sku} بالمخزون المتاح في {warehouse}.", "نحتاج {qty} وحدة من {sku}؛ تحقق من النقص في {warehouse}.", "هل يمكن تلبية طلب {qty} من {sku} باستخدام مخزون {warehouse}؟"],
 "replenishment": ["راجع حد إعادة الطلب للصنف {sku} في {warehouse}.", "هل يحتاج {sku} إلى إعادة تزويد في مستودع {warehouse}؟", "جهز كمية مقترحة كمسودة لتعبئة مخزون {sku} قرب {warehouse}.", "نبهني إذا كان رصيد {sku} منخفضاً في {warehouse}.", "تحقق من حد الطلب واقترح تعبئة مبدئية للصنف {sku} في {warehouse}؟"],
 "shipping": ["تحقق من جاهزية شحن الصنف {sku} من {warehouse}.", "هل يستطيع {warehouse} شحن {qty} وحدة من {sku}؟", "ابحث عن مخزون {sku} المناسب للشحن من {warehouse}.", "راجع تخصيص الصنف {sku} للشحن عبر {warehouse}.", "أكد إمكانية شحن {qty} قطعة من {sku} من {warehouse}."] ,
 "purchasing": ["قدر كمية شراء مسودة للصنف {sku} بعد فحص مخزون {warehouse}.", "راجع طلب شراء {qty} وحدة من {sku} بالاستناد إلى مخزون {warehouse}.", "هل ينبغي للمشتريات مراجعة الصنف {sku} حسب رصيد {warehouse}؟", "جهز توصية شراء للقراءة فقط للصنف {sku} قرب {warehouse}.", "تحقق من الحاجة لمسودة طلب توريد {qty} من {sku} في {warehouse}."]}

def split_families(seed: int = SEED) -> dict[str, str]:
    groups = [f[0] for f in FAMILIES]
    random.Random(seed).shuffle(groups)
    # Fixed 70/13/17 family proportions; every split contains whole product families.
    return {family: ("train" if i < 21 else "validation" if i < 25 else "test") for i, family in enumerate(groups)}

def products(seed: int):
    rng = random.Random(seed)
    rows=[]
    for fi, (prefix, category, en, ar, variants, cost) in enumerate(FAMILIES):
        for vi, variant in enumerate(variants):
            sku = "BRG-6204" if prefix == "BRG" and vi == 0 else f"{prefix}-{vi+1:02d}"
            name=f"{en} {variant}"
            name_ar=f"{ar} {variant}"
            aliases=f"{variant} {en}|{en} {variant}|{name_ar}|قطعة {name_ar}"
            family=f"FAM-{prefix}"
            description_en=f"Replacement {category.lower()} component for maintenance use. Rating or size: {variant}. Verify dimensions and approved specifications before substitution."
            description_ar=f"مكوّن بديل لفئة {CATEGORY_AR[category]} لأعمال الصيانة. المقاس أو التصنيف: {variant}. يجب التحقق من الأبعاد والمواصفات المعتمدة قبل الاستبدال."
            rows.append({"sku":sku,"name":name,"aliases":aliases,"category":category,"warehouse":WAREHOUSES[(fi+vi)%5],
                         "on_hand":rng.randint(18,360),"reserved":0,"reorder_point":rng.randint(20,90),
                         "unit_cost_sar":round(cost*(0.8+rng.random()*0.4),2),"name_ar":name_ar,"category_ar":CATEGORY_AR[category],
                         "description_en":description_en,"description_ar":description_ar,"family_id":family})
    # Retain the original, tested multi-warehouse inventory arithmetic exactly.
    brg=next(r for r in rows if r['sku']=='BRG-6204')
    brg.update(name="Bearing 6204",name_ar="محمل 6204",aliases="ball bearing 6204|رمان بلي 6204|محمل 6204",warehouse="Riyadh",on_hand=120,reserved=20,reorder_point=60,unit_cost_sar=45)
    rows.append({**brg,"warehouse":"Jeddah","on_hand":75,"reserved":5})
    for r in rows:
        if r['sku']!='BRG-6204': r['reserved']=rng.randint(0,min(r['on_hand'],30))
    return rows

def make_requests(items, family_split, seed):
    rng=random.Random(seed+1); rows=[]
    for ix,p in enumerate(items):
        family=p['family_id']
        fi=next(i for i,f in enumerate(FAMILIES) if f[0]==family[4:])
        for k in range(2):
            intent=INTENTS[(ix%5+k)%5]
            lang="en" if (ix+k)%2==0 else "ar"
            w=p['warehouse']; w=WAREHOUSES[(WAREHOUSES.index(w)+k)%5]
            qty=rng.randint(4,180)
            templates=(EN_REQUESTS if lang=='en' else AR_REQUESTS)[intent]
            core=templates[(fi+k)%len(templates)].format(sku=p['sku'],qty=qty,warehouse=w if lang=='en' else WAREHOUSES_AR[w])
            context=(EN_CONTEXT if lang=='en' else AR_CONTEXT)[fi].format(product=p['name'] if lang=='en' else p['name_ar'])
            text=f"{core} {context}"
            rows.append({"request_id":f"REQ-{len(rows)+1:04d}","text":text,"request_core":core,"language":lang,"intent":intent,
                         "sku":p['sku'],"quantity":qty,"warehouse":w,"family_id":family,
                         "template_group":f"{family}-{intent}-{lang}","split":family_split[family[4:]],"synthetic":True})
    return rows

def ner_record(p, supplier, warehouse, quantity, lang, family_split):
    if lang=='en':
        tokens=["Please","supply",str(quantity),"units","of",*p['name'].split(),"SKU",p['sku'],"from",*supplier[0].split(),"at",warehouse,"."]
        product_start=5; sku_at=5+len(p['name'].split())+1; supplier_at=sku_at+2; wh_at=supplier_at+len(supplier[0].split())+1
        types=[(2,3,"QUANTITY"),(product_start,product_start+len(p['name'].split()),"PRODUCT"),(sku_at,sku_at+1,"SKU"),(supplier_at,supplier_at+len(supplier[0].split()),"SUPPLIER"),(wh_at,wh_at+1,"WAREHOUSE")]
        sentence=" ".join(tokens)
    else:
        prod=p['name_ar'].split(); sup=supplier[1].split(); w=WAREHOUSES_AR[warehouse]
        tokens=["يرجى","توفير",str(quantity),"وحدة","من",*prod,"الصنف",p['sku'],"من",*sup,"في",w,"."]
        product_start=5; sku_at=5+len(prod)+1; supplier_at=sku_at+2; wh_at=supplier_at+len(sup)+1
        types=[(2,3,"QUANTITY"),(product_start,product_start+len(prod),"PRODUCT"),(sku_at,sku_at+1,"SKU"),(supplier_at,supplier_at+len(sup),"SUPPLIER"),(wh_at,wh_at+1,"WAREHOUSE")]
        sentence=" ".join(tokens)
    tags=["O"]*len(tokens); entities=[]
    for start,end,kind in types:
        tags[start]=f"B-{kind}"
        for j in range(start+1,end): tags[j]=f"I-{kind}"
        entities.append({"type":kind,"start":start,"end":end,"text":" ".join(tokens[start:end])})
    return {"text":sentence,"tokens":tokens,"bio_tags":tags,"entities":entities,"sku":p['sku'],"family_id":p['family_id'],
            "split":family_split[p['family_id'][4:]],"synthetic":True}

def make_ner(items, family_split):
    selected=[]
    # 70 train + 15 validation + 15 test examples, selected by complete families.
    for split, family_count in (("train",14),("validation",3),("test",3)):
        fams=[f"FAM-{f[0]}" for f in FAMILIES if family_split[f[0]]==split][:family_count]
        for family in fams:
            selected.extend([p for p in items if p['family_id']==family][:5])
    result=[]
    for i,p in enumerate(selected):
        supplier=SUPPLIERS[i%len(SUPPLIERS)]; warehouse=WAREHOUSES[(i*3)%len(WAREHOUSES)]
        result.append(ner_record(p,supplier,warehouse,5+(i*7)%95,"en" if i%2==0 else "ar",family_split))
    return result

def make_semantic(items, family_split, seed):
    rng=random.Random(seed+2); result=[]
    candidates={s:[p for p in items if family_split[p['family_id'][4:]]==s] for s in ("train","validation","test")}
    counts={"train":36,"validation":7,"test":7}; qid=0
    for split,count in counts.items():
        pool=candidates[split]
        for n in range(count):
            p=pool[n%len(pool)]; others=[x for x in pool if x['family_id']!=p['family_id']]
            neg=others[n%len(others)]
            lang="en" if qid%2==0 else "ar"
            query=(p['name']+" availability" if lang=='en' else p['name_ar']+" التوفر")
            for doc,label,candidate in ((p,2,p),(neg,0,neg)):
                doc_text=f"{doc['sku']} | {doc['name']} | {doc['name_ar']} | {doc['aliases']} | {doc['category']} | {doc['description_en']} | {doc['description_ar']}"
                result.append({"query_id":f"Q-{qid+1:03d}","query":query,"language":lang,"document_id":candidate['sku'],
                               "document":doc_text,"relevance":label,"query_family_id":p['family_id'],"document_family_id":doc['family_id'],
                               "split":split,"synthetic":True})
            qid+=1
    rng.shuffle(result)
    return result

def make_challenge_data(items,family_split):
    test_families=[f"FAM-{f[0]}" for f in FAMILIES if family_split[f[0]]=='test']
    held=[p for p in items if p['family_id'] in test_families]
    family_specs={p['family_id']:[] for p in held}
    for p in held:family_specs[p['family_id']].append(p)
    req_templates={
      'stock_inquiry':[
       ('en','Could you tell me the unreserved balance for {sku} at {warehouse}?'),('ar','كم قطعة غير محجوزة من {sku} في {warehouse}؟'),
       ('mixed','Can I allocate {sku} من فرع {warehouse}؟'),('en','Is any usable quantity of {sku} sitting at {warehouse}?'),('ar','أحتاج رصيد الصنف {sku} الجاهز للصرف في {warehouse}.')],
      'shortage_check':[
       ('en','Demand is {qty} pieces; would supply of {sku} fall short in {warehouse}?'),('ar','الطلب {qty} قطعة، هل سيظهر عجز للصنف {sku} في {warehouse}؟'),
       ('mixed','هل الكمية تكفي لـ {qty} units من {sku}؟'),('en','Compare the {qty}-unit requirement with the free stock of {sku}.'),('ar','قارن احتياج {qty} وحدة برصيد {sku} المتاح.')],
      'replenishment':[
       ('en','Put {sku} on the restock review list for {warehouse}.'),('ar','هل ندرج {sku} ضمن قائمة تعويض النواقص في {warehouse}؟'),
       ('mixed','Review reorder for {sku} في {warehouse}.'),('en','Suggest a draft top-up if {sku} is below its buffer at {warehouse}.'),('ar','راجع مستوى الأمان واقترح تعبئة مبدئية لـ {sku}.')],
      'shipping':[
       ('en','Can the {warehouse} team dispatch {qty} of {sku} today?'),('ar','هل يمكن تجهيز شحنة من {qty} وحدة للصنف {sku} عبر {warehouse}؟'),
       ('mixed','Check delivery-ready stock for {sku} من {warehouse}.'),('en','Find a dispatch location for {qty} pieces of {sku}.'),('ar','تحقق من جاهزية صرف {sku} للشحن من {warehouse}.')],
      'purchasing':[
       ('en','Does the buyer need to raise a draft requisition for {sku}?'),('ar','هل تستدعي حالة {sku} إعداد مسودة طلب توريد؟'),
       ('mixed','Prepare a purchase review for {sku} بالاستناد إلى المخزون.'),('en','Estimate a procurement top-up of {qty} for {sku}; do not place an order.'),('ar','راجع الحاجة الشرائية للصنف {sku} دون تنفيذ شراء.')]
    }
    challenge_requests=[]
    for i in range(100):
        intent=list(req_templates)[i//20];lang,text=req_templates[intent][i%5]
        p=held[i%len(held)];family=family_specs[p['family_id']]
        warehouse=p['warehouse'];warehouse_ar=WAREHOUSES_AR[warehouse];qty=10+(i*13)%140
        sku=p['sku'] if i%4 else ''
        text=text.format(sku=sku or ('هذا الصنف' if lang!='en' else 'the requested part'),warehouse=warehouse if lang!='ar' else warehouse_ar,qty=qty)
        challenge_requests.append({'request_id':f'CHREQ-{i+1:03d}','text':text,'intent':intent,'language':lang,'sku':sku,
          'quantity':qty,'warehouse':warehouse,'family_id':p['family_id'],'split':'challenge_test','synthetic':True})
    challenge_search=[]
    for i,p in enumerate(held):
        spec=p['name'].replace(FAMILIES[next(k for k,f in enumerate(FAMILIES) if f[0]==p['family_id'][4:])][2],'').strip()
        # A matched paraphrase in each language, neither copied from product name nor alias.
        family_i=next(k for k,f in enumerate(FAMILIES) if f[0]==p['family_id'][4:]);_,category,en,ar,_,_=FAMILIES[family_i]
        english={'Mechanical':'mechanical maintenance part','Electrical':'power system spare','HVAC':'air handling element','Plumbing':'water line fitting','Electronics':'electronic service component','Safety':'protective work equipment'}[category]
        arabic={'Mechanical':'قطعة صيانة ميكانيكية','Electrical':'مستلزم لمنظومة الطاقة','HVAC':'عنصر لمسار معالجة الهواء','Plumbing':'وصلة لشبكة المياه','Electronics':'مكوّن إلكتروني للخدمة','Safety':'معدة للوقاية المهنية'}[category]
        # Preserve the rating/size term, but describe its function with vocabulary absent from the name.
        variants=[('en_paraphrase',f"Find the {english} rated {spec} for the maintenance kit."),
                  ('ar_paraphrase',f"أبحث عن {arabic} بالمقاس أو التصنيف {spec} لعدة الصيانة."),
                  ('mixed_typo',f"مطلوب {arabic} spare specification {spec} للصيانه، الرجاء البحث.")]
        for kind,q in variants:
            if kind=='mixed_typo' and i>=20:continue
            challenge_search.append({'query_id':f'CHQ-{len(challenge_search)+1:03d}','query':q,'query_type':kind,'language':'en' if kind=='en_paraphrase' else 'ar' if kind=='ar_paraphrase' else 'mixed',
              'expected_skus':p['sku'],'is_negative':False,'family_id':p['family_id'],'split':'challenge_test','synthetic':True})
    # Add five genuinely ambiguous family-level descriptions: all five variants are acceptable candidates.
    for family_id,products_in_family in list(family_specs.items())[:5]:
        p=products_in_family[0];prefix=family_id[4:]
        ambiguous_en={'GLV':'hand protection','MSK':'breathing protection','GGL':'eye protection','HLM':'head protection','PPE':'visibility equipment',
          'BRG':'rotating supports','MTR':'drive units','FLT':'air treatment elements','VLV':'flow-control parts','SNS':'measurement devices',
          'CBL':'power transmission','PMP':'fluid-moving units','GSK':'sealing parts','BPD':'braking components','BLT':'fastening hardware',
          'NUT':'threaded retainers','SCR':'threaded fasteners','FAN':'air circulation units','BEL':'drive transmission','PUL':'rotating transmission',
          'CBR':'circuit protection','REL':'switching components','FUS':'circuit protection','LED':'status indicators','CON':'electrical connections',
          'TAP':'thread sealing','HOS':'fluid transfer','OIL':'fluid filtration','BAT':'portable power','WIP':'air cleaning'}[prefix]
        ambiguous_ar={'GLV':'وقاية اليد','MSK':'حماية التنفس','GGL':'حماية العين','HLM':'حماية الرأس','PPE':'معدات الرؤية الواضحة'}
        hint_ar=ambiguous_ar.get(prefix,'مستلزمات الصيانة')
        q=f"Which {ambiguous_en} variants should I compare for maintenance?" if len(challenge_search)%2==0 else f"ما الخيارات المناسبة من فئة {hint_ar} لأعمال الموقع؟"
        challenge_search.append({'query_id':f'CHQ-{len(challenge_search)+1:03d}','query':q,'query_type':'ambiguous_family','language':'mixed','expected_skus':'|'.join(x['sku'] for x in products_in_family),
          'is_negative':False,'family_id':family_id,'split':'challenge_test','synthetic':True})
    negatives=[
      'Find a replacement for non-catalog part ZXQ-9917; no matching item should exist.',
      'أبحث عن قطعة غير موجودة برمز ZZX-8401 ولا يوجد لها بديل في القائمة.',
      'Need the imaginary component code QQQ-0000 for a discontinued unit.',
      'هل يتوفر الصنف الوهمي ABC-7777 في أي مستودع؟',
      'Spare for a fictional machine, reference NO-SUCH-42, not a real inventory part.',
    ]
    for i in range(25):
        q=negatives[i%len(negatives)]
        # Make each negative distinct while retaining a clearly absent identifier.
        code=f"ZXQ-{9000+i}"
        q=re.sub(r'(ZXQ-9917|ZZX-8401|QQQ-0000|ABC-7777|NO-SUCH-42)',code,q)
        challenge_search.append({'query_id':f'CHQ-{len(challenge_search)+1:03d}','query':q,'query_type':'negative_missing_sku','language':'en' if i%2==0 else 'ar',
          'expected_skus':'','is_negative':True,'family_id':'','split':'challenge_test','synthetic':True})
    return challenge_requests,challenge_search

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def generate(out_dir: Path, seed: int = SEED):
    rng=random.Random(seed)
    family_split=split_families(seed)
    inv=products(seed)
    for p in inv:p['split']=family_split[p['family_id'][4:]];p['synthetic']=True
    sku_items=list({p['sku']:p for p in inv}.values())
    requests=make_requests(sku_items,family_split,seed)
    ner=make_ner(sku_items,family_split)
    semantic=make_semantic(sku_items,family_split,seed)
    challenge_requests,challenge_search=make_challenge_data(sku_items,family_split)
    out_dir.mkdir(parents=True,exist_ok=True)
    write_csv(out_dir/'inventory.csv',inv)
    write_csv(out_dir/'requests.csv',requests)
    write_jsonl(out_dir/'ner_bio.jsonl',ner)
    write_csv(out_dir/'semantic_pairs.csv',semantic)
    write_csv(out_dir/'challenge_requests.csv',challenge_requests)
    write_csv(out_dir/'challenge_search.csv',challenge_search)
    with (out_dir/'split_manifest.json').open('w',encoding='utf-8') as f:
        json.dump({"synthetic":True,"seed":seed,"family_splits":family_split,"split_method":"fixed-seed family-group split 70/13/17"},f,ensure_ascii=False,indent=2)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data'/'synthetic')
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args();generate(args.out_dir,args.seed)
    print(f"Generated explicitly synthetic datasets in {args.out_dir} using seed {args.seed}")

if __name__=='__main__':main()
