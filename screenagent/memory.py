"""Four memory layers (see the design doc, section 4).

Working    per-abstract blackboard shared by all agents, cleared after each abstract
Semantic   criteria + labelled DEV examples, retrieved by TF-IDF cosine (stand-in for ChromaDB)
Episodic   human corrections; only the HITL interface may write (agents are read-only)
Procedural prompts / rubric / policy, frozen and hashed before the test run
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = set("a an the of and or to in on for with was were we this that by from as at is are be".split())


def _tokens(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Blackboard(dict):
    """Working memory: one dict per abstract. Every agent reads/writes it; Supervisor clears it."""

    def post(self, author: str, key: str, value):
        self.setdefault("_log", []).append({"author": author, "key": key})
        self[key] = value


class SemanticMemory:
    """Criteria + labelled examples. Retrieval is restricted to the dev split to prevent test leakage."""

    def __init__(self, criteria: dict, examples: list):
        self.criteria = dict(criteria)
        bad = [e.id for e in examples if e.split != "dev"]
        if bad:
            raise ValueError(f"semantic memory may only hold dev examples, got {bad[:3]}")
        self.examples = list(examples)
        self._df = Counter()
        self._vecs = []
        for e in self.examples:
            tf = Counter(_tokens(e.text))
            self._df.update(tf.keys())
            self._vecs.append(tf)
        self._n = len(self.examples)
        self._vecs = [self._weight(tf) for tf in self._vecs]

    def _weight(self, tf):
        v = {t: (1 + math.log(c)) * math.log((1 + self._n) / (1 + self._df.get(t, 0)) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, text: str, k: int = 5):
        q = self._weight(Counter(_tokens(text)))
        scored = []
        for e, v in zip(self.examples, self._vecs):
            s = sum(q[t] * v.get(t, 0.0) for t in q)
            scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [{"id": e.id, "text": e.text, "label": "include" if e.relevant else "exclude", "score": round(s, 3)}
                for s, e in scored[:k]]


class EpisodicMemory:
    """Human corrections. Agents can read; only the role 'human' can write (blocks memory poisoning)."""

    def __init__(self):
        self.items = []

    def add(self, role: str, abstract_id: str, label: str, reason: str):
        if role != "human":
            raise PermissionError("only the human reviewer may write episodic memory")
        self.items.append({"id": abstract_id, "label": label, "reason": reason})

    def recall(self, k=3):
        return self.items[-k:]


class ProceduralMemory:
    """Prompts, critic rubric and policy. freeze() records a hash; any later change raises."""

    def __init__(self, **docs):
        self.docs = dict(docs)
        self._hash = None

    def digest(self):
        return hashlib.sha256(json.dumps(self.docs, sort_keys=True).encode()).hexdigest()

    def freeze(self):
        self._hash = self.digest()
        return self._hash

    def get(self, key):
        if self._hash and self.digest() != self._hash:
            raise RuntimeError("procedural memory changed after freeze")
        return self.docs[key]
