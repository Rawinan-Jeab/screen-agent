"""Safety layers: input guard, task-bound permission gate, hash-chained audit log."""
from __future__ import annotations

import hashlib
import json
import re
import time

# ---------------------------------------------------------------- input guard
_PATTERNS = [
    r"ignore (all|any|previous)[^.]*instruction", r"ignore all inclusion criteria", r"\bsystem\s*:",
    r"assistant instruction", r"note to any ai", r"call (export_results|record_decision)", r"mark this paper as",
]
_RX = re.compile("|".join(_PATTERNS), re.I)


def input_guard(text: str):
    """Wrap untrusted text and flag embedded instructions. Returns (wrapped_text, flagged, hits)."""
    hits = [m.group(0) for m in _RX.finditer(text)]
    wrapped = f"<untrusted_abstract>\n{text}\n</untrusted_abstract>"
    return wrapped, bool(hits), hits


# ------------------------------------------------------------ permission gate
RISK = {
    "get_criteria": "low",
    "search_examples": "low",
    "flag_for_human": "low",
    "record_decision": "medium",
    "export_results": "high",
}


class PermissionGate:
    """Decides in code, not in the prompt.

    low     always allowed
    medium  allowed only for the abstract id bound to the current task
    high    needs explicit human approval (approver callback); default deny
    unknown deny
    """

    def __init__(self, approver=None):
        self.approver = approver or (lambda call, task: False)

    def check(self, call: dict, task: dict):
        tool = call.get("tool")
        tier = RISK.get(tool)
        if tier is None:
            return "deny", "unknown tool"
        if tier == "low":
            return "allow", "low risk"
        if tier == "medium":
            if call.get("args", {}).get("abstract_id") == task["abstract_id"]:
                return "allow", "task-bound id"
            return "deny", "abstract id outside current task"
        if self.approver(call, task):
            return "allow", "human approved"
        return "deny", "high risk without approval"


class NoGate(PermissionGate):
    """Ablation: everything allowed."""

    def check(self, call, task):
        return "allow", "no gate (ablation)"


# ------------------------------------------------------------------ audit log
class AuditLog:
    """Tamper-evident log: each record stores the hash of the previous one."""

    def __init__(self):
        self.records = []
        self.prev = "0" * 64

    def log(self, event: str, **meta):
        rec = {"ts": time.time(), "event": event, "meta": meta, "prev": self.prev}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
        self.records.append(rec)
        self.prev = rec["hash"]
        return rec

    def verify(self):
        prev = "0" * 64
        for i, r in enumerate(self.records):
            body = {k: v for k, v in r.items() if k != "hash"}
            if r["prev"] != prev:
                return False, i
            if hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != r["hash"]:
                return False, i
            prev = r["hash"]
        return True, None

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, default=str) + "\n")
