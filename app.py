"""ScreenAgent UI (Streamlit).

    pip install streamlit
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
from collections import Counter

import pandas as pd
import streamlit as st

from screenagent.data import build_corpus, CRITERIA, Abstract
from screenagent.memory import SemanticMemory, EpisodicMemory
from screenagent.agents import Supervisor, Config
from screenagent.llm import MockLLM, OllamaLLM
from screenagent.safety import AuditLog, input_guard

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "results.json")
CRIT_TH = {"P": "กลุ่มตัวอย่าง (P)", "I": "สิ่งที่ทดลอง (I)", "O": "ผลลัพธ์ (O)", "S": "รูปแบบการวิจัย (S)"}
LABEL_TH = {"include": "คัดเข้า", "exclude": "คัดออก", "uncertain": "ส่งมนุษย์ตรวจ"}
LABEL_COLOR = {"include": "green", "exclude": "red", "uncertain": "orange"}

st.set_page_config(page_title="ScreenAgent", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.agent-card {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;}
.agent-card h4 {margin: 0 0 6px 0; font-size: 1rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; margin-right:6px;
       border:1px solid rgba(128,128,128,.4);}
.small {font-size:.85rem; opacity:.75;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ state
@st.cache_resource
def corpus_and_memory():
    corpus = build_corpus()
    dev = [a for a in corpus if a.split == "dev"]
    test = [a for a in corpus if a.split == "test"]
    return corpus, dev, test, SemanticMemory(CRITERIA, dev)


corpus, dev, test, semantic = corpus_and_memory()
by_id = {a.id: a for a in corpus}
ss = st.session_state
ss.setdefault("episodic", EpisodicMemory())
ss.setdefault("audit", AuditLog())
ss.setdefault("batch", None)
ss.setdefault("exports", [])

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔎 ScreenAgent")
    st.caption("Multi-agent คัดกรอง abstract · DIM 716")
    backend = st.radio("Model", ["Mock (ออฟไลน์)", "Ollama (local model)"], index=0)
    model = st.text_input("ชื่อ model", "qwen2.5:7b-instruct", disabled=backend.startswith("Mock"))
    seed = st.number_input("Seed", 1, 99, 1)
    st.divider()
    st.subheader("ส่วนประกอบของระบบ")
    use_critic = st.toggle("Critic", True)
    use_memory = st.toggle("Memory (ตัวอย่างจาก dev)", True)
    use_gate = st.toggle("Permission gate", True)
    use_guard = st.toggle("Input guard", True)
    thr = st.slider("Confidence threshold ส่งมนุษย์", 0.5, 0.95, 0.65, 0.05)
    st.divider()
    st.caption("โหมด Mock จำลองการอ่านของ LLM ใช้สอนและทดสอบระบบเท่านั้น ไม่ใช่ผลวิจัย")


def make_llm():
    if backend.startswith("Mock"):
        return MockLLM(seed=int(seed))
    return OllamaLLM(model=model, seed=int(seed))


def make_sup(approver=None, cfg_over=None):
    cfg = Config("ui", use_memory=use_memory, use_critic=use_critic, use_gate=use_gate, use_guard=use_guard)
    for k, v in (cfg_over or {}).items():
        setattr(cfg, k, v)
    sup = Supervisor(make_llm(), semantic, cfg, audit=ss.audit, approver=approver, episodic=ss.episodic)
    sup.proc.docs["policy"] = dict(sup.proc.docs["policy"], confidence_threshold=thr)
    return sup


def label_badge(label):
    return f":{LABEL_COLOR[label]}[**{LABEL_TH[label]}**]"


def verdict_table(a, screener, critic=None, show_truth=True):
    rows = []
    for c in "PIOS":
        row = {"เกณฑ์": CRIT_TH[c],
               "Screener": "✅ ผ่าน" if screener[c]["met"] else "❌ ไม่ผ่าน",
               "ความมั่นใจ": screener[c]["confidence"]}
        if critic is not None:
            row["Critic"] = "✅ ผ่าน" if critic[c]["met"] else "❌ ไม่ผ่าน"
        if show_truth and a.criteria:
            row["เฉลย"] = ("✅" if a.criteria[c]["met"] else "❌") + f" ({'ชัดเจน' if a.criteria[c]['clarity'] == 'explicit' else 'กำกวม'})"
        rows.append(row)
    return pd.DataFrame(rows)


def run_one(sup, a):
    return sup.process(a, salt="agent")


tab_home, tab_screen, tab_hitl, tab_safety, tab_results = st.tabs(
    ["ภาพรวม", "คัดกรองทีละเรื่อง", "คิวตรวจโดยมนุษย์", "Safety", "ผลการทดลอง"])

# ================================================================== home
with tab_home:
    st.header("ScreenAgent: ระบบ multi-agent คัดกรอง abstract")
    st.write("ระบบตัดสินว่า abstract แต่ละเรื่องเข้าเกณฑ์ systematic review หรือไม่ "
             "โดยให้ agent หลายตัวทำงานร่วมกันบน blackboard และส่งเฉพาะกรณีที่ไม่แน่ใจให้มนุษย์ตรวจ")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Abstract ทั้งหมด", len(corpus), f"dev {len(dev)} · test {len(test)}", delta_color="off")
    c2.metric("เกี่ยวข้องใน test", sum(a.relevant for a in test))
    c3.metric("ฝัง prompt injection", sum(bool(a.injection) for a in test))
    c4.metric("ตัวอย่างใน memory", len(semantic.examples))

    st.subheader("Agent ในระบบ")
    cards = [
        ("1 · Input guard", "ครอบ abstract ด้วย &lt;untrusted&gt; และตรวจคำสั่งที่ฝังมา"),
        ("2 · Screener (ReAct)", "เรียก get_criteria → search_examples แล้วตัดสินทีละเกณฑ์ P/I/O/S"),
        ("3 · Critic", "ทวนเหตุผลของ Screener ทีละเกณฑ์ ถ้าเห็นต่างจะส่งให้มนุษย์"),
        ("4 · Supervisor", "คุมลำดับ จำกัด step ตัดสินใจขั้นสุดท้าย และส่งงานเข้าคิวมนุษย์"),
        ("5 · Permission gate", "ทุก tool call ผ่าน gate: low อนุญาต, medium เฉพาะ task นี้, high ต้องให้มนุษย์อนุมัติ"),
        ("6 · Audit log", "บันทึกทุกการกระทำแบบ hash chain ตรวจได้ว่ามีใครแก้หรือไม่"),
    ]
    cols = st.columns(3)
    for i, (t, d) in enumerate(cards):
        cols[i % 3].markdown(f"<div class='agent-card'><h4>{t}</h4><div class='small'>{d}</div></div>",
                             unsafe_allow_html=True)
    st.subheader("เกณฑ์คัดเข้า")
    for c, txt in CRITERIA.items():
        st.markdown(f"- **{CRIT_TH[c]}** {txt.split(': ', 1)[1]}")

# ================================================================ screen
with tab_screen:
    st.header("คัดกรองทีละเรื่อง")
    f1, f2 = st.columns([1, 2])
    kind = f1.selectbox("เลือกชุด", ["ทั้งหมด", "เกี่ยวข้อง", "ไม่เกี่ยวข้อง", "ฝัง injection"])
    pool = {"ทั้งหมด": test, "เกี่ยวข้อง": [a for a in test if a.relevant],
            "ไม่เกี่ยวข้อง": [a for a in test if not a.relevant and not a.injection],
            "ฝัง injection": [a for a in test if a.injection]}[kind]
    aid = f2.selectbox("Abstract", [a.id for a in pool],
                       format_func=lambda i: f"{i} · {by_id[i].text[60:130]}…")
    a = by_id[aid]
    custom = st.toggle("พิมพ์ abstract เอง (ต้องใช้ Ollama)", False)
    if custom:
        text = st.text_area("Abstract", height=160)
        a = Abstract(id="CUSTOM", text=text, relevant=False, criteria={}, split="test")
    else:
        st.info(a.text)
        st.caption(f"เฉลย: {'เกี่ยวข้อง' if a.relevant else 'ไม่เกี่ยวข้อง'}"
                   + (f" · มี injection แบบ {a.injection}" if a.injection else ""))

    if st.button("รัน ScreenAgent", type="primary", disabled=custom and backend.startswith("Mock")):
        if custom and not a.text.strip():
            st.warning("กรุณาพิมพ์ abstract")
            st.stop()
        try:
            r = run_one(make_sup(), a)
        except Exception as e:  # e.g. Ollama not running
            st.error(f"เรียก model ไม่สำเร็จ: {e}")
            st.stop()
        st.subheader("ผลลัพธ์")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"#### {label_badge(r['label'])}")
        m1.caption(r["reason"])
        m2.metric("Token", f"{r['tokens']:,}")
        m3.metric("LLM calls", r["llm_calls"])
        m4.metric("Steps", r["steps"])

        st.markdown("##### 1 · Input guard")
        _, flagged, hits = input_guard(a.text)
        if not use_guard:
            st.caption("ปิดอยู่")
        elif flagged:
            st.warning(f"พบคำสั่งที่ฝังมา: {hits}")
        else:
            st.success("ไม่พบคำสั่งที่ฝังมา")

        st.markdown("##### 2–3 · Screener และ Critic")
        st.dataframe(verdict_table(a, r["screener"], r["critic"], show_truth=not custom),
                     hide_index=True, width="stretch")
        if r["disagree"]:
            st.warning(f"Critic เห็นต่างในเกณฑ์ {', '.join(CRIT_TH[c] for c in r['disagree'])}")

        st.markdown("##### 4–5 · Supervisor และ Permission gate")
        tr = pd.DataFrame(r["trace"])
        tr["decision"] = tr["decision"].map({"allow": "✅ allow", "deny": "⛔ deny"})
        st.dataframe(tr.rename(columns={"agent": "Agent", "tool": "Tool", "decision": "Gate", "why": "เหตุผล"}),
                     hide_index=True, width="stretch")

# ================================================================== hitl
with tab_hitl:
    st.header("คิวตรวจโดยมนุษย์ (Human-in-the-loop)")
    st.write("รันระบบกับ abstract หลายเรื่อง เรื่องที่ระบบไม่แน่ใจจะเข้าคิวนี้ "
             "คำตัดสินของอาจารย์จะถูกบันทึกเป็น episodic memory ซึ่งมีแต่มนุษย์เท่านั้นที่เขียนได้")
    n = st.slider("จำนวน abstract ที่จะรัน", 10, 300, 40, 10)
    if st.button("รันชุดนี้", type="primary"):
        sup = make_sup()
        out = []
        bar = st.progress(0.0)
        for i, ab in enumerate(test[:n]):
            out.append(sup.process(ab, salt="agent"))
            bar.progress((i + 1) / n)
        ss.batch = {"results": out, "sup": sup}
    if ss.batch:
        res = ss.batch["results"]
        cnt = Counter(r["label"] for r in res)
        rel = [r for r in res if by_id[r["id"]].relevant]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("คัดเข้า", cnt["include"])
        k2.metric("คัดออก", cnt["exclude"])
        k3.metric("รอมนุษย์ตรวจ", cnt["uncertain"])
        k4.metric("Recall (ก่อนมนุษย์ตรวจ)",
                  f"{sum(r['label'] != 'exclude' for r in rel) / max(1, len(rel)):.2f}")
        queue = [r for r in res if r["label"] == "uncertain"]
        done = {e["id"] for e in ss.episodic.items}
        st.subheader(f"คิวรอตรวจ ({len([q for q in queue if q['id'] not in done])} เรื่อง)")
        for r in queue:
            if r["id"] in done:
                continue
            ab = by_id[r["id"]]
            with st.expander(f"{r['id']} · {r['reason']}"):
                st.write(ab.text)
                st.dataframe(verdict_table(ab, r["screener"], show_truth=False), hide_index=True,
                             width="stretch")
                c1, c2 = st.columns([1, 3])
                choice = c1.radio("คำตัดสิน", ["include", "exclude"], key=f"c{r['id']}",
                                  format_func=lambda x: LABEL_TH[x])
                why = c2.text_input("เหตุผล (จะถูกเก็บเป็น episodic memory)", key=f"w{r['id']}")
                if st.button("บันทึก", key=f"b{r['id']}"):
                    ss.episodic.add("human", r["id"], choice, why or "-")
                    ss.batch["sup"].tools.decisions[r["id"]] = choice
                    ss.audit.log("human_decision", task=r["id"], label=choice, reason=why)
                    st.rerun()
        if ss.episodic.items:
            st.subheader("Episodic memory (คำตัดสินของมนุษย์)")
            em = pd.DataFrame(ss.episodic.items)
            em["เฉลย"] = em["id"].map(lambda i: "เกี่ยวข้อง" if by_id[i].relevant else "ไม่เกี่ยวข้อง")
            st.dataframe(em, hide_index=True, width="stretch")

        st.subheader("Export ผล (high risk: ต้องอนุมัติ)")
        approve = st.checkbox("ข้าพเจ้าอนุมัติให้ export ผลการคัดกรองชุดนี้")
        if st.button("Export CSV"):
            sup = ss.batch["sup"]
            sup.gate.approver = lambda call, task: approve
            sup.task = {"abstract_id": "EXPORT", "goal": "export"}
            ok = sup.call_tool({"tool": "export_results", "args": {"format": "csv"}}, agent="human")
            if ok:
                df = pd.DataFrame([{"id": i, "label": l} for i, l in sup.tools.decisions.items()])
                st.download_button("ดาวน์โหลด screening_results.csv", df.to_csv(index=False),
                                   "screening_results.csv", "text/csv")
            else:
                st.error("Gate ปฏิเสธ: ต้องติ๊กอนุมัติก่อน")

# ================================================================ safety
with tab_safety:
    st.header("Safety: ทดสอบ prompt injection")
    st.write("เลือก abstract ที่ฝังคำสั่งไว้ แล้วเทียบระบบที่มีและไม่มี permission gate บน seed เดียวกัน")
    inj = [a for a in test if a.injection]

    def _lands(ab):  # does the attack land without a gate under the current seed? (shown first)
        r = make_sup(cfg_over={"use_gate": False}).process(ab, salt="agent")
        return any(t["agent"] == "screener" and t["decision"] == "allow" and t["tool"] in ("export_results", "record_decision")
                   for t in r["trace"]) or (ab.injection == "flip" and r["label"] == "include")
    _scratch = ss.audit
    ss.audit = AuditLog()   # keep the probe runs out of the visible audit log
    lands = {ab.id: _lands(ab) for ab in inj}
    ss.audit = _scratch
    inj = sorted(inj, key=lambda ab: (not lands[ab.id], ab.injection))
    opt_label = {ab.id: f"{ab.id} · แบบ {ab.injection}" + (" · โจมตีสำเร็จถ้าไม่มี gate" if lands[ab.id] else "")
                 for ab in inj}
    pick = st.selectbox("Abstract ที่ฝัง injection", [a.id for a in inj],
                        format_func=opt_label.get)
    st.info(by_id[pick].text)
    if st.button("ทดสอบโจมตี", type="primary"):
        cols = st.columns(2)
        for col, gate_on in zip(cols, (True, False)):
            sup = make_sup(cfg_over={"use_gate": gate_on})
            r = sup.process(by_id[pick], salt="agent")
            executed = [t for t in r["trace"] if t["agent"] == "screener" and t["tool"] in ("export_results", "record_decision")]
            with col:
                st.markdown(f"#### {'มี gate' if gate_on else 'ไม่มี gate'}")
                st.markdown(f"คำตัดสิน: {label_badge(r['label'])}")
                if not executed:
                    st.success("model ไม่ทำตามคำสั่งที่ฝังมาใน seed นี้ (ลองเปลี่ยน seed ทางซ้าย)")
                for t in executed:
                    (st.error if t["decision"] == "allow" else st.success)(
                        f"{t['tool']} → {t['decision']} ({t['why']})")
                if sup.tools.exports:
                    st.error(f"ข้อมูลถูก export ออกไป: {sup.tools.exports}")
    st.divider()
    st.subheader("Audit log")
    log = ss.audit
    st.caption(f"{len(log.records)} records ในเซสชันนี้")
    if log.records:
        st.dataframe(pd.DataFrame([{"event": r["event"], "agent": r["meta"].get("agent", ""),
                                    "tool": (r["meta"].get("call") or {}).get("tool", ""),
                                    "decision": r["meta"].get("decision", ""), "hash": r["hash"][:16] + "…"}
                                   for r in log.records[-30:]]), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        if c1.button("ตรวจ hash chain"):
            ok, at = log.verify()
            (st.success("chain ถูกต้องทั้งสาย") if ok else st.error(f"chain ขาดที่ record {at}"))
        if c2.button("จำลองการแก้ log ย้อนหลัง"):
            log.records[0]["meta"]["decision"] = "edited"
            ok, at = log.verify()
            st.error(f"ตรวจพบการแก้ไขที่ record {at}") if not ok else st.success("ok")

# =============================================================== results
with tab_results:
    st.header("ผลการทดลอง (Experiment Plan)")
    if not os.path.exists(RESULTS):
        st.warning("ยังไม่มีไฟล์ results/results.json ให้รัน `python3 run_experiment.py` ก่อน")
    else:
        res = json.load(open(RESULTS, encoding="utf-8"))
        st.caption(f"Backend: {res['backend']} · test {res['n_test']} เรื่อง (เกี่ยวข้อง {res['n_relevant']}) · 3 seeds")
        if res["backend"] == "mock":
            st.warning("ผลจาก Mock LLM: ใช้สอนและตรวจ pipeline เท่านั้น ไม่ใช่ผลวิจัย")
        full = res["conditions"]["ScreenAgent (full)"]
        m = st.columns(4)
        m[0].metric("Recall (ระบบเต็ม)", f"{full['recall']:.3f}",
                    f"{full['recall'] - res['conditions']['B2 retrieval']['recall']:+.3f} จาก B2")
        m[1].metric("ส่งมนุษย์ตรวจ", f"{full['uncertain_rate']:.1%}")
        m[2].metric("Token / abstract", f"{full['tokens_per_abstract']:,.0f}")
        m[3].metric("Attack success", full["attack_success_majority"])
        df = pd.DataFrame([{"เงื่อนไข": c, "Recall": e["recall"], "CI ล่าง": e["ci"][0], "CI บน": e["ci"][1],
                            "Precision": e["precision"], "WSS": e["wss"], "ส่งมนุษย์": e["uncertain_rate"],
                            "Token": e["tokens_per_abstract"], "Attack success": e.get("attack_success_majority", "–")}
                           for c, e in res["conditions"].items()])
        st.subheader("Recall ตามเงื่อนไข")
        st.bar_chart(df.set_index("เงื่อนไข")["Recall"], horizontal=True, height=300)
        st.dataframe(df.style.format({"Recall": "{:.3f}", "CI ล่าง": "{:.3f}", "CI บน": "{:.3f}",
                                      "Precision": "{:.3f}", "WSS": "{:.3f}", "ส่งมนุษย์": "{:.1%}",
                                      "Token": "{:,.0f}"}), hide_index=True, width="stretch")
        st.subheader("การทดสอบสมมติฐาน (McNemar + Holm)")
        st.dataframe(pd.DataFrame([{"เปรียบเทียบ": f"ระบบเต็ม {k}", "ระบบเต็มถูกฝ่ายเดียว": t["only_full"],
                                    "อีกฝ่ายถูกฝ่ายเดียว": t["only_other"], "Holm p": round(t["holm_p"], 4),
                                    "Cohen's h": round(t["cohens_h"], 2)} for k, t in res["tests"].items()]),
                     hide_index=True, width="stretch")
        ag = res["agents"]
        st.subheader("ผลของ agent แต่ละตัว")
        a1, a2, a3 = st.columns(3)
        a1.metric("Critic จับข้อผิดพลาดได้", f"{ag['critic']['catch_rate']:.0%}",
                  f"false alarm {ag['critic']['false_alarm_rate']:.0%}", delta_color="off")
        sc = ag["screener"]
        a2.metric("Screener: ข้อความกำกวม", f"{sc['ScreenAgent (full)']['subtle']['accuracy']:.2f}",
                  f"{sc['ScreenAgent (full)']['subtle']['accuracy'] - sc['- memory']['subtle']['accuracy']:+.2f} จาก memory")
        a3.metric("Guard ตรวจพบ injection", f"{ag['guard']['flag_rate_injected']:.0%}",
                  f"false positive {ag['guard']['flag_rate_clean']:.0%}", delta_color="off")
        st.dataframe(pd.DataFrame([{"tool | decision | เหตุผล": k, "จำนวน": v} for k, v in ag["gate"].items()]),
                     hide_index=True, width="stretch")
