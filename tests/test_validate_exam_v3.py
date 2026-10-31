import copy, unittest
import validate_exam_v3 as v


def mc(qid,label,mark=''):
    opts=[f'alpha{mark}', 'beta', 'gamma', 'delta', 'epsilon']
    return {'question_id':qid,'owner':'x','area':'a','format':'mc','question':'Pick one','options':opts,'correct_option':opts['ABCDE'.index(label)],'answer':opts['ABCDE'.index(label)],'rubric':'exact','source_id':'s','source_path':'/tmp/x','source_sha256':'0'*64,'evidence_span':'source evidence long enough','evidence_start':0,'evidence_end':27,'difficulty':'introductory','skills':['reasoning']}

class McSafetyTests(unittest.TestCase):
    def test_rejects_visible_answer_markers(self):
        bad=['*alpha','alpha*','**alpha**','✓ alpha','alpha (correct)','[correct] alpha','alpha <-','alpha ->','answer → alpha','alpha ✅','alpha ☒','alpha ★','<u>alpha</u>','__alpha__']
        for i,s in enumerate(bad):
            r=mc(str(i),'A');r['options'][0]=s;r['correct_option']=s;r['answer']=s
            self.assertTrue(v.mc_leak_errors(r),s)

    def test_rejects_question_and_rubric_answer_disclosure(self):
        r=mc('q','B');r['question']='Pick one. Correct answer: B';self.assertTrue(v.mc_leak_errors(r))
        r=mc('q','B');r['rubric']='Accept option B';self.assertTrue(v.mc_leak_errors(r))
        r=mc('q','B');r['question']='Answer → B';self.assertTrue(v.mc_leak_errors(r))
        r=mc('q','B');r['rubric']='Correct: B';self.assertTrue(v.mc_leak_errors(r))

    def test_balanced_exactly_100_per_letter_for_500_mc(self):
        rows=[mc(str(i),"ABCDE"[i%5]) for i in range(500)]
        self.assertEqual(v.correct_label_counts(rows),{'A':100,'B':100,'C':100,'D':100,'E':100})
        self.assertEqual(v.balance_errors(rows),[])

    def test_rejects_unbalanced_distribution(self):
        rows=[mc(str(i),'A') for i in range(500)]
        self.assertTrue(v.balance_errors(rows))

    def test_rejects_balanced_wrong_sizes(self):
        for n in (0,5,10,495,505):
            rows=[mc(str(i),'ABCDE'[i%5]) for i in range(n)]
            self.assertTrue(v.balance_errors(rows),n)

    def test_public_exam_removes_answer_key_fields(self):
        r=mc('q','C');r['evidence']='SECRET';r['source']='SECRET';r['nested']={'answer':'SECRET'};r['rubric_detail']='SECRET'
        public=v.public_question(r)
        for k in ('answer','correct_option','rubric','evidence_span','source_path','source_sha256'):
            self.assertNotIn(k,public)
        for k in ('evidence','source','nested','rubric_detail','source_id','evidence_start','evidence_end'):
            self.assertNotIn(k,public)
        self.assertEqual(public['options'],r['options'])

if __name__=='__main__':unittest.main()
