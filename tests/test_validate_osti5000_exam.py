import json, pathlib, tempfile, unittest
import validate_osti5000_exam as v

class OstiMembershipTests(unittest.TestCase):
    def accepted_manifest(self, root, first, count=5000):
        rows=[first]
        for i in range(2,count+1):
            src=root/f'paper-{i}.txt';src.write_text(f'paper {i} source')
            rows.append({"paper_id":f"osti:{i}","status":"ACCEPT","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":v.sha256(src)}]})
        p=root/'accepted.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows));return p

    def test_accepts_exact_accepted_manifest_member_and_selected_text_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td); src=root/'paper.txt'; src.write_text('alpha beta gamma delta')
            rec={"paper_id":"osti:1","status":"ACCEPT","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":v.sha256(src)}]}
            snap=self.accepted_manifest(root,rec)
            q={"question_id":"q","source_id":"osti:1","source_path":str(src),"source_sha256":v.sha256(src),"evidence_span":"beta gamma","evidence_start":6,"evidence_end":16}
            self.assertEqual(v.validate_rows([q],v.load_accepted_manifest(snap)),[])
    def test_rejects_nonmember_and_member_wrong_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);src=root/'a.txt';src.write_text('source text here');other=root/'other.txt';other.write_text('source text here')
            snap=self.accepted_manifest(root,{"paper_id":"osti:1","status":"ACCEPT","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":v.sha256(src)}]})
            base={"question_id":"q","source_sha256":v.sha256(other),"evidence_span":"source","evidence_start":0,"evidence_end":6}
            errors=v.validate_rows([{**base,"source_id":"textures:x","source_path":str(other)},{**base,"question_id":"q2","source_id":"osti:1","source_path":str(other)}],v.load_accepted_manifest(snap))
            self.assertTrue(any('not in OSTI-5000' in e for e in errors));self.assertTrue(any('path not registered' in e for e in errors))

    def test_rejects_non_5000_duplicate_and_nonaccepted_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);src=root/'a.txt';src.write_text('source')
            base={"paper_id":"osti:1","status":"ACCEPT","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":v.sha256(src)}]}
            for rows,needle in [([base],"exactly 5000"),([base]*5000,"duplicate"),([{**base,"paper_id":f"osti:{i}","status":"REJECT" if i==9 else "ACCEPT"} for i in range(5000)],"non-ACCEPT")]:
                p=root/'bad.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
                with self.assertRaisesRegex(ValueError,needle):v.load_accepted_manifest(p)

    def test_rejects_unhashed_selected_source_and_conflicting_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);src=root/'a.txt';src.write_text('source');digest=v.sha256(src)
            cases=[
                ({"paper_id":"osti:1","status":"ACCEPT","selected_text_path":str(src),"source_files":[]},"not hash-bound"),
                ({"paper_id":"osti:1","status":"ACCEPT","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":"bad"}]},"valid sha256"),
                ({"paper_id":"osti:1","status":"ACCEPT","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":digest},{"path":str(src),"sha256":"0"*64}]},"conflicting hashes"),
            ]
            for first,needle in cases:
                p=self.accepted_manifest(root,first)
                with self.assertRaisesRegex(ValueError,needle):v.load_accepted_manifest(p)

    def test_rejects_missing_null_blank_and_nonstr_paper_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);src=root/'a.txt';src.write_text('source');digest=v.sha256(src)
            base={"status":"ACCEPT","selected_text_path":str(src),"source_files":[{"path":str(src),"sha256":digest}]}
            for bad in ({}, {"paper_id":None}, {"paper_id":""}, {"paper_id":"   "}, {"paper_id":17}):
                first={**base,**bad};p=self.accepted_manifest(root,first)
                with self.assertRaisesRegex(ValueError,"paper_id must be a non-empty string"):v.load_accepted_manifest(p)
if __name__=='__main__':unittest.main()
