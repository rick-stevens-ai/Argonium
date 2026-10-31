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
        self.assertEqual(runner.answer_state({"content":"This approximation cannot preserve energy.", "finish_reason":"stop", "error":None}, True), "VALID")
        self.assertEqual(runner.answer_state({"content":"I cannot answer that request.", "finish_reason":"stop", "error":None}, True), "REFUSAL")
        self.assertEqual(runner.answer_state({"content":"I can’t answer that request.", "finish_reason":"stop", "error":None}, True), "REFUSAL")
        self.assertEqual(runner.answer_state({"content":"I am not able to answer that request.", "finish_reason":"stop", "error":None}, True), "REFUSAL")
        self.assertEqual(runner.answer_state({"content":"I will not answer that request.", "finish_reason":"stop", "error":None}, True), "REFUSAL")

    def test_grade_parser_rejects_extra_text_and_bad_types(self):
        self.assertIsNone(runner.parse_grade('preamble {"score":1,"match":true,"confidence":0.9,"reasoning":"ok"}'))
        self.assertIsNone(runner.parse_grade('{"score":0.5,"match":true,"confidence":0.9,"reasoning":"ok"}'))
        self.assertIsNone(runner.parse_grade('{"score":1,"match":"yes","confidence":0.9,"reasoning":"ok"}'))
        self.assertEqual(runner.parse_grade('{"score":1,"match":true,"confidence":0.9,"reasoning":"ok"}')["score"],1)
        self.assertEqual(runner.parse_grade('<think>private</think>\n{"score":1,"match":true,"confidence":0.9,"reasoning":"ok"}')["score"],1)
        self.assertIsNone(runner.parse_grade('<think>unterminated {"score":1,"match":true,"confidence":0.9,"reasoning":"ok"}'))

    def test_resume_counts_only_explicit_terminal_records(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"x.jsonl"
            p.write_text('\n'.join([json.dumps({"owner":"x","question_id":"q1","model_label":"m","state":"VALID"}),json.dumps({"owner":"x","question_id":"q2","model_label":"m","state":"ERROR"}),'{bad']))
            got=runner.completed_keys(p,("owner","question_id","model_label"),runner.ANSWER_TERMINAL)
            self.assertIn(("x","q1","m"),got); self.assertIn(("x","q2","m"),got)

    def test_invalid_grades_are_resumable_not_terminal(self):
        self.assertNotIn("INVALID_GRADE",runner.GRADE_TERMINAL)
        self.assertNotIn("GRADER_IDENTITY_MISMATCH",runner.GRADE_TERMINAL)

    def test_redacted_exam_contains_exact_public_keyset(self):
        row={"question_id":"q1","format":"qa","question":"What?","answer":"secret","rubric":"secret","source_path":"secret"}
        self.assertEqual(runner.redact_question(row),{"question_id":"q1","format":"qa","question":"What?"})

    def test_mc_public_view_has_choices_but_no_private_key(self):
        row={"question_id":"q1","format":"mc","question":"Choose.","options":["one","two","three","four","five"],"answer":"B. two","rubric":"secret","source_path":"secret"}
        public=runner.redact_question(row)
        self.assertEqual(set(public),{"question_id","format","question"})
        self.assertIn("A. one",public["question"]);self.assertIn("E. five",public["question"])
        self.assertNotIn("secret",json.dumps(public))

    def test_actual_candidate_call_payload_excludes_private_fields(self):
        with tempfile.TemporaryDirectory() as td:
            captured=[];old_call=runner.call
            try:
                def fake_call(route,messages,max_tokens,timeout):
                    captured.append(messages)
                    return {"raw":'{"model":"route","choices":[]}',"content":"B. two","reasoning_content":"","finish_reason":"stop","usage":{},"actual_model":"route","latency_seconds":0.1,"http_status":200}
                runner.call=fake_call
                q={"question_id":"q1","format":"mc","question":"Choose.","options":["one","two","three","four","five"],"answer":"B. two SECRET_ANSWER","rubric":"SECRET_RUBRIC","source_path":"SECRET_PATH","evidence_span":"SECRET_EVIDENCE"}
                out=pathlib.Path(td)/"answer.jsonl";runner.answer_one("x",q,"m","route",out)
                payload=json.dumps(captured)
                for secret in ("SECRET_ANSWER","SECRET_RUBRIC","SECRET_PATH","SECRET_EVIDENCE"):
                    self.assertNotIn(secret,payload)
                self.assertIn("A. one",payload);self.assertIn("E. five",payload)
            finally: runner.call=old_call

    def test_mc_leak_markers_and_key_parser(self):
        clean={"question":"Choose. A. one B. two C. three D. four E. five","answer":"C. three"}
        opts=runner.mc_options(clean);self.assertEqual(runner.mc_correct_label(clean,opts),"C");self.assertEqual(runner.mc_leakage(clean,opts),[])
        for marker in ("*two","✓ two","➡ two","➤ two","__two__","_two_","~~two~~","`two`","# two","two (correct answer)","[x] two","(selected) two","selected option two","Correct: B","key B","selected: B","checkmark B","Answer: B"):
            row={"question":"Choose.","options":["one",marker,"three","four","five"],"answer":"B. "+marker}
            self.assertTrue(runner.mc_leakage(row,runner.mc_options(row)))

    def full_exam(self):
        rows=[]
        for i in range(500):
            label="ABCDE"[i%5];opts=[f"choice-{i}-{x}" for x in "ABCDE"]
            rows.append({"question_id":f"m{i}","owner":"x","area":"a","format":"mc","question":f"MC question {i}?","options":opts,"correct_label":label,"correct_option":opts["ABCDE".index(label)],"answer":f"{label}. {opts['ABCDE'.index(label)]}","rubric":"exact option","source_id":f"osti:{i}","source_path":"/x","source_sha256":"0"*64,"evidence_span":"evidence span long enough for validation", "evidence_start":0,"evidence_end":40,"difficulty":"introductory" if i<100 else ("intermediate" if i<350 else "advanced"),"skills":["s"]})
        for i in range(500):
            rows.append({"question_id":f"q{i}","owner":"x","area":"a","format":"qa","question":f"QA question {i}?","answer":"answer","rubric":"rubric","source_id":f"osti:{500+i}","source_path":"/x","source_sha256":"0"*64,"evidence_span":"evidence span long enough for validation", "evidence_start":0,"evidence_end":40,"difficulty":"introductory" if i<100 else ("intermediate" if i<350 else "advanced"),"skills":["s"]})
        return rows

    def test_exact_equilateral_key_distribution_and_single_mutations(self):
        rows=self.full_exam();self.assertEqual(runner.validate_exam_shape(rows),[])
        rows[0]["correct_label"]="B";rows[0]["correct_option"]=rows[0]["options"][1];rows[0]["answer"]="B. "+rows[0]["options"][1]
        self.assertTrue(any("mc_key_distribution" in e for e in runner.validate_exam_shape(rows)))
        rows=self.full_exam();rows[0]["options"][0]="*"+rows[0]["options"][0]
        self.assertTrue(any("answer leakage" in e for e in runner.validate_exam_shape(rows)))

    def test_conflicting_private_mc_keys_fail_closed(self):
        rows=self.full_exam()
        rows[0]["correct_option"]=rows[0]["options"][1]
        rows[0]["answer"]="B. "+rows[0]["options"][1]
        errors=runner.validate_exam_shape(rows)
        self.assertTrue(any("MC key cannot be resolved" in e for e in errors))

    def test_exam_shape_rejects_partial_or_wrong_mix(self):
        self.assertTrue(runner.validate_exam_shape([]))
        base={"question_id":"q","owner":"x","area":"a","format":"qa","question":"What?","answer":"A","rubric":"R","source_id":"osti:1","source_path":"/x","source_sha256":"0"*64,"evidence_span":"span","evidence_start":0,"evidence_end":4,"difficulty":"introductory","skills":["s"]}
        self.assertTrue(any("row_count" in e for e in runner.validate_exam_shape([base])))

    def test_qwen_farm_has_three_distinct_exact_routes(self):
        self.assertEqual(len(runner.GRADERS),3)
        self.assertEqual(len(set(runner.GRADERS)),3)

    def test_v3_contract_has_exact_roster_and_unresolved_routes(self):
        p=pathlib.Path(__file__).parents[1]/"AREA_EXAM_11K_CONTRACT.json"
        roster=runner.load_roster(p)
        self.assertEqual(len(roster),17)
        self.assertEqual(dict(roster)["Claude Sonnet 5.5"],None)
        self.assertEqual(dict(roster)["GPT-5.6 Sol"],"argo:gpt-5.6-sol")

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
