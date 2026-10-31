import json, pathlib, tempfile, unittest
import area_exam_runner as runner

class ContractTests(unittest.TestCase):
    def test_exact_model_identity_requires_observed_identity(self):
        self.assertFalse(runner.identity_matches("argo:gpt-5.6-sol", None))
        self.assertFalse(runner.identity_matches("argo:gpt-5.6-sol", "fallback/model"))
        self.assertTrue(runner.identity_matches("argo:gpt-5.6-sol", "argo:gpt-5.6-sol"))

    def test_answer_acceptance_rejects_reasoning_only_and_mismatch(self):
        self.assertEqual(runner.answer_state({"content":"", "reasoning_content":"thinking", "finish_reason":"stop", "error":None}, True), "EMPTY")
        self.assertEqual(runner.answer_state({"content":"A", "finish_reason":"stop", "error":None}, False), "IDENTITY_MISMATCH")
        self.assertEqual(runner.answer_state({"content":"A", "finish_reason":"length", "error":None}, True), "LENGTH_CAPPED")
        self.assertEqual(runner.answer_state({"content":"A", "finish_reason":"stop", "error":None}, True), "VALID")

    def test_grade_parser_rejects_extra_text_and_bad_types(self):
        self.assertIsNone(runner.parse_grade('preamble {"score":1,"match":true,"confidence":0.9,"reasoning":"ok"}'))
        self.assertIsNone(runner.parse_grade('{"score":0.5,"match":true,"confidence":0.9,"reasoning":"ok"}'))
        self.assertIsNone(runner.parse_grade('{"score":1,"match":"yes","confidence":0.9,"reasoning":"ok"}'))
        self.assertEqual(runner.parse_grade('{"score":1,"match":true,"confidence":0.9,"reasoning":"ok"}')["score"],1)

    def test_resume_counts_only_explicit_terminal_records(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"x.jsonl"
            p.write_text('\n'.join([json.dumps({"owner":"x","question_id":"q1","model_label":"m","state":"VALID"}),json.dumps({"owner":"x","question_id":"q2","model_label":"m","state":"ERROR"}),'{bad']))
            got=runner.completed_keys(p,("owner","question_id","model_label"),runner.ANSWER_TERMINAL)
            self.assertIn(("x","q1","m"),got); self.assertIn(("x","q2","m"),got)

    def test_atomic_key_uses_lane_owner(self):
        self.assertEqual(runner.atomic_key("yeil","q9","M"),("yeil","q9","M"))

    def test_timeout_floor_supports_long_compression(self):
        import os
        old=os.environ.get("ARGONIUM_REQUEST_TIMEOUT")
        try:
            os.environ["ARGONIUM_REQUEST_TIMEOUT"]="1800"
            self.assertEqual(runner.effective_timeout(45),1800)
        finally:
            if old is None: os.environ.pop("ARGONIUM_REQUEST_TIMEOUT",None)
            else: os.environ["ARGONIUM_REQUEST_TIMEOUT"]=old

if __name__ == '__main__': unittest.main()
