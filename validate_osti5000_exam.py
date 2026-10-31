#!/usr/bin/env python3
"""Fail-closed OSTI-5000 source membership and evidence validator."""
import argparse,hashlib,json
from pathlib import Path

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def load_snapshot(path):
 idx={}
 with Path(path).open() as f:
  for n,line in enumerate(f,1):
   if not line.strip():continue
   r=json.loads(line);pid=r['paper_id'];allowed={}
   if r.get('selected_text_path'): allowed[str(Path(r['selected_text_path']).expanduser().resolve())]=None
   for s in r.get('source_files') or []:
    if s.get('path'): allowed[str(Path(s['path']).expanduser().resolve())]=s.get('sha256')
   idx[pid]=allowed
 return idx

def validate_rows(rows,index):
 errors=[];cache={}
 for i,r in enumerate(rows,1):
  qid=r.get('question_id',f'row-{i}');pid=str(r.get('source_id',''))
  if pid not in index:errors.append(f'{qid}: source_id not in OSTI-5000 snapshot: {pid}');continue
  p=Path(str(r.get('source_path',''))).expanduser().resolve();ps=str(p)
  if ps not in index[pid]:errors.append(f'{qid}: source path not registered for OSTI-5000 paper {pid}: {p}');continue
  if not p.is_file():errors.append(f'{qid}: source absent {p}');continue
  if ps not in cache:
   b=p.read_bytes();cache[ps]=(b,b.decode('utf-8',errors='replace'),hashlib.sha256(b).hexdigest())
  b,text,digest=cache[ps]
  expected=r.get('source_sha256')
  registered=index[pid][ps]
  if expected!=digest:errors.append(f'{qid}: question source hash mismatch')
  if registered and registered!=digest:errors.append(f'{qid}: snapshot source hash mismatch')
  try:s,e=int(r['evidence_start']),int(r['evidence_end'])
  except Exception:errors.append(f'{qid}: invalid evidence offsets');continue
  if not 0<=s<e<=len(text):errors.append(f'{qid}: evidence offsets out of bounds');continue
  if text[s:e]!=r.get('evidence_span'):errors.append(f'{qid}: exact evidence span mismatch')
 return errors

def main():
 ap=argparse.ArgumentParser();ap.add_argument('exam',type=Path);ap.add_argument('--snapshot',type=Path,required=True);a=ap.parse_args()
 rows=[]
 for n,line in enumerate(a.exam.open(),1):
  try:rows.append(json.loads(line))
  except Exception as e:print(json.dumps({'valid':False,'errors':[f'line {n}: {e}']}));raise SystemExit(1)
 errors=validate_rows(rows,load_snapshot(a.snapshot));o={'schema':'argonium-osti5000-membership-v1','valid':not errors,'rows':len(rows),'snapshot':str(a.snapshot),'snapshot_sha256':sha256(a.snapshot),'exam':str(a.exam),'exam_sha256':sha256(a.exam),'errors':errors};print(json.dumps(o,indent=2));raise SystemExit(0 if not errors else 1)
if __name__=='__main__':main()
