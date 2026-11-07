"""Browser bridge for the GitHub Pages version of ScreenAgent.

Runs inside Pyodide (Python compiled to WebAssembly). It mirrors app.py but
returns plain JSON so index.html can draw the UI. Only MockLLM is available in
the browser; Ollama needs a local server and the Streamlit app.
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter

from screenagent.data import build_corpus, CRITERIA
from screenagent.memory import SemanticMemory, EpisodicMemory
from screenagent.agents import Supervisor, Config
from screenagent.llm import MockLLM
from screenagent.safety import AuditLog, input_guard

CORPUS = build_corpus()
DEV = [a for a in CORPUS if a.split == "dev"]
TEST = [a for a in CORPUS if a.split == "test"]
BY_ID = {a.id: a for a in CORPUS}
SEMANTIC = SemanticMemory(CRITERIA, DEV)

S = {"episodic": EpisodicMemory(), "audit": AuditLog(), "batch": None, "lands": {}}


def _out(x):
    return json.dumps(x, ensure_ascii=False, default=str)


def _sup(o, audit=None, **over):
    o = json.loads(o) if isinstance(o, str) else o
    cfg = Config("ui", use_memory=o["memory"], use_critic=o["critic"],
                 use_gate=o["gate"], use_guard=o["guard"])
    for k, v in over.items():
        setattr(cfg, k, v)
    sup = Supervisor(MockLLM(seed=int(o["seed"])), SEMANTIC, cfg,
                     audit=audit if audit is not None else S["audit"], episodic=S["episodic"])
    sup.proc.docs["policy"] = dict(sup.proc.docs["policy"], confidence_threshold=float(o["thr"]))
    return sup


def _ab(a):
    return {"id": a.id, "text": a.text, "relevant": a.relevant,
            "injection": a.injection, "criteria": a.criteria}


def overview():
    return _out({
        "n": len(CORPUS), "dev": len(DEV), "test": len(TEST),
        "relevant": sum(a.relevant for a in TEST),
        "injected": sum(bool(a.injection) for a in TEST),
        "memory": len(SEMANTIC.examples),
        "criteria": {c: t.split(": ", 1)[-1] for c, t in CRITERIA.items()},
    })


def pool(kind):
    p = {"all": TEST, "rel": [a for a in TEST if a.relevant],
         "irr": [a for a in TEST if not a.relevant and not a.injection],
         "inj": [a for a in TEST if a.injection]}[kind]
    return _out([{"id": a.id, "snip": a.text[60:130]} for a in p])


def abstract(aid):
    return _out(_ab(BY_ID[aid]))


def screen(aid, o):
    a = BY_ID[aid]
    r = _sup(o).process(a, salt="agent")
    _, flagged, hits = input_guard(a.text)
    r["guard"] = {"flagged": flagged, "hits": hits}
    r["abstract"] = _ab(a)
    return _out(r)


def run_batch(n, o):
    sup = _sup(o)
    res = [sup.process(a, salt="agent") for a in TEST[:int(n)]]
    S["batch"] = {"results": res, "sup": sup}
    return batch_state()


def batch_state():
    b = S["batch"]
    if not b:
        return _out(None)
    res = b["results"]
    cnt = Counter(r["label"] for r in res)
    rel = [r for r in res if BY_ID[r["id"]].relevant]
    done = {e["id"] for e in S["episodic"].items}
    queue = [{"id": r["id"], "reason": r["reason"], "text": BY_ID[r["id"]].text,
              "screener": r["screener"]}
             for r in res if r["label"] == "uncertain" and r["id"] not in done]
    epi = [dict(e, truth=BY_ID[e["id"]].relevant) for e in S["episodic"].items]
    return _out({"n": len(res), "include": cnt["include"], "exclude": cnt["exclude"],
                 "uncertain": cnt["uncertain"],
                 "recall": sum(r["label"] != "exclude" for r in rel) / max(1, len(rel)),
                 "queue": queue, "episodic": epi})


def human_decide(aid, label, why):
    S["episodic"].add("human", aid, label, why or "-")
    S["batch"]["sup"].tools.decisions[aid] = label
    S["audit"].log("human_decision", task=aid, label=label, reason=why)
    return batch_state()


def export(approve):
    sup = S["batch"]["sup"]
    sup.gate.approver = lambda call, task: bool(approve)
    sup.task = {"abstract_id": "EXPORT", "goal": "export"}
    ok = sup.call_tool({"tool": "export_results", "args": {"format": "csv"}}, agent="human")
    if not ok:
        return _out({"ok": False})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "label"])
    for i, l in sup.tools.decisions.items():
        w.writerow([i, l])
    return _out({"ok": True, "csv": buf.getvalue()})


def _lands(a, o):
    r = _sup(o, audit=AuditLog(), use_gate=False).process(a, salt="agent")
    return any(t["agent"] == "screener" and t["decision"] == "allow"
               and t["tool"] in ("export_results", "record_decision") for t in r["trace"]) \
        or (a.injection == "flip" and r["label"] == "include")


def injected(o):
    inj = [a for a in TEST if a.injection]
    lands = {a.id: _lands(a, o) for a in inj}
    inj.sort(key=lambda a: (not lands[a.id], a.injection))
    return _out([{"id": a.id, "kind": a.injection, "lands": lands[a.id]} for a in inj])


def attack(aid, o):
    out = []
    for gate_on in (True, False):
        sup = _sup(o, use_gate=gate_on)
        r = sup.process(BY_ID[aid], salt="agent")
        ex = [t for t in r["trace"] if t["agent"] == "screener"
              and t["tool"] in ("export_results", "record_decision")]
        out.append({"gate": gate_on, "label": r["label"], "executed": ex,
                    "exports": list(sup.tools.exports)})
    return _out(out)


def audit_state():
    log = S["audit"]
    rows = [{"event": r["event"], "agent": r["meta"].get("agent", ""),
             "tool": (r["meta"].get("call") or {}).get("tool", ""),
             "decision": r["meta"].get("decision", ""), "hash": r["hash"][:16] + "…"}
            for r in log.records[-30:]]
    return _out({"count": len(log.records), "rows": rows})


def audit_verify():
    ok, at = S["audit"].verify()
    return _out({"ok": ok, "at": at})


def audit_tamper():
    log = S["audit"]
    if log.records:
        log.records[0]["meta"]["decision"] = "edited"
    return audit_verify()


def reset():
    S.update({"episodic": EpisodicMemory(), "audit": AuditLog(), "batch": None})
    return _out(True)
