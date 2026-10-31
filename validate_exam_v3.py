#!/usr/bin/env python3
"""MC safety, balance, and public/private exam separation for Argonium."""
import argparse,collections,json,re
from pathlib import Path

LABELS='ABCDE'
PUBLIC_FIELDS={'question_id','owner','area','format','question','options','difficulty','skills'}
# Indicators that visibly identify an option as privileged/correct.
MARKER_PATTERNS=[
 r'\*|__|<\/?u\b', r'✓|✔|✅|☑|☒|★', r'\bcorrect\b', r'\banswer\s*key\b',
 r'\[(?:x|correct|answer)\]', r'\((?:correct|answer)\)', r'(?:<-|->|→|←)\s*(?:correct|answer)?\s*$',
 r'^\s*(?:correct|answer)\s*(?::|[-=]|→|->)',
]

def option_label(row):
 opts=row.get('options')
 if not isinstance(opts,list) or len(opts)!=5:return None
 correct=row.get('correct_option'); answer=row.get('answer')
 try:i=opts.index(correct)
 except ValueError:return None
 if opts.count(correct)!=1 or answer!=correct:return None
 return LABELS[i]

def mc_leak_errors(row):
 if row.get('format')!='mc':return []
 errors=[];opts=row.get('options')
 if not isinstance(opts,list) or len(opts)!=5:return ['options must be exactly five strings']
 for i,opt in enumerate(opts):
  if not isinstance(opt,str) or not opt.strip():errors.append(f'option {LABELS[i]} empty/non-string');continue
  for pat in MARKER_PATTERNS:
   if re.search(pat,opt,re.I):errors.append(f'option {LABELS[i]} contains answer marker {pat}')
 for field in ('question','rubric'):
  text=str(row.get(field,''))
  if re.search(r'\b(?:(?:correct\s*(?:answer|option)?|answer)\s*(?::|is|→|->)?|accept\s+option)\s*[A-E]\b',text,re.I):
   errors.append(f'{field} discloses answer label')
 if option_label(row) is None:errors.append('correct_option/answer must equal one unique option')
 return errors

def correct_label_counts(rows):
 c=collections.Counter(option_label(r) for r in rows if r.get('format')=='mc')
 c.pop(None,None);return {x:c.get(x,0) for x in LABELS}

def balance_errors(rows):
 mc=[r for r in rows if r.get('format')=='mc'];n=len(mc);counts=correct_label_counts(mc);errors=[]
 if n!=500:errors.append(f'MC count {n}, expected exactly 500')
 target=100
 if any(counts[x]!=target for x in LABELS):errors.append(f'correct-label distribution {counts}, expected {target} each')
 return errors

def public_question(row):
 return {k:row[k] for k in PUBLIC_FIELDS if k in row}

def load(path):
 rows=[]
 with Path(path).open() as f:
  for n,line in enumerate(f,1):
   try:rows.append(json.loads(line))
   except Exception as e:raise ValueError(f'line {n}: {e}')
 return rows

def validate(rows):
 errors=[]
 for r in rows:
  errors.extend(f"{r.get('question_id')}: {e}" for e in mc_leak_errors(r))
 errors.extend(balance_errors(rows))
 return errors

def write_public(rows,path):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:
  for r in rows:f.write(json.dumps(public_question(r),ensure_ascii=False,separators=(',',':'))+'\n')
 tmp.replace(path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('exam',type=Path);ap.add_argument('--public-out',type=Path);a=ap.parse_args();rows=load(a.exam);errors=validate(rows)
 if not errors and a.public_out:write_public(rows,a.public_out)
 out={'schema':'argonium-mc-safety-v1','valid':not errors,'rows':len(rows),'mc_rows':sum(r.get('format')=='mc' for r in rows),'correct_label_counts':correct_label_counts(rows),'public_out':str(a.public_out) if a.public_out and not errors else None,'errors':errors};print(json.dumps(out,indent=2));raise SystemExit(0 if not errors else 1)
if __name__=='__main__':main()
