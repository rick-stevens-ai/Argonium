import json, pathlib, tempfile, unittest
import validate_osti5000_exam as v

class OstiMembershipTests(unittest.TestCase):
    def test_accepts_exact_snapshot_member_and_selected_text_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td); src=root/'paper.txt'; src.write_text('alpha beta gamma delta')
            rec={"paper_id":"osti:1","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":v.sha256(src)}]}
            snap=root/'snapshot.jsonl';snap.write_text(json.dumps(rec)+'\n')
            q={"question_id":"q","source_id":"osti:1","source_path":str(src),"source_sha256":v.sha256(src),"evidence_span":"beta gamma","evidence_start":6,"evidence_end":16}
            self.assertEqual(v.validate_rows([q],v.load_snapshot(snap)),[])
    def test_rejects_nonmember_and_member_wrong_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);src=root/'a.txt';src.write_text('source text here');other=root/'other.txt';other.write_text('source text here')
            snap=root/'s.jsonl';snap.write_text(json.dumps({"paper_id":"osti:1","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":v.sha256(src)}]})+'\n')
            base={"question_id":"q","source_sha256":v.sha256(other),"evidence_span":"source","evidence_start":0,"evidence_end":6}
            errors=v.validate_rows([{**base,"source_id":"textures:x","source_path":str(other)},{**base,"question_id":"q2","source_id":"osti:1","source_path":str(other)}],v.load_snapshot(snap))
            self.assertTrue(any('not in OSTI-5000' in e for e in errors));self.assertTrue(any('path not registered' in e for e in errors))
if __name__=='__main__':unittest.main()
