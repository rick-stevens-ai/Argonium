#!/usr/bin/env python3
"""Contract-safe Argonium 11K answer/grade runner.

Answers and grades are separate phases. Exact route identity is fail-closed.
Every append is fsync'd; resume only recognizes explicit terminal states.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, threading, time
import urllib.error, urllib.request
from pathlib import Path
import validate_osti5000_exam as osti5000

BASE=os.environ.get("ARGONIUM_BASE_URL","http://127.0.0.1:4000/v1/chat/completions")
KEY=os.environ.get("ARGONIUM_API_KEY","stevens")
ANSWER_TERMINAL={"VALID","REFUSAL","EMPTY","ERROR","TIMEOUT","LENGTH_CAPPED","IDENTITY_MISMATCH","UNRESOLVED_ROUTE"}
GRADE_TERMINAL={"VALID_GRADE","FAILURE_ZERO"}
GRADERS=("pecans/qwen3.8-27b","uicgpu/qwen3.8-27b","chiatta/qwen38-27b")
LOCK=threading.Lock()

def sha(b): return hashlib.sha256(b).hexdigest()
def slug(s): return re.sub(r"[^a-z0-9]+","-",s.lower()).strip("-")
def atomic_key(owner,qid,label): return owner,qid,label

LEAK_PATTERNS=(
    (re.compile(r"\*"),"asterisk"),
    (re.compile(r"[✓✔✅☑★➜➤➡→]"),"answer marker glyph"),
    (re.compile(r"__(?:[^_]|_(?!_))+__"),"markdown underline/bold"),
    (re.compile(r"(?<!\w)_(?!_)[^_\n]+_(?!\w)"),"markdown emphasis"),
    (re.compile(r"~~[^~\n]+~~"),"markdown strikethrough"),
    (re.compile(r"`[^`\n]+`"),"markdown inline code"),
    (re.compile(r"(?m)^\s*#{1,6}\s+"),"markdown heading"),
    (re.compile(r"(?i)\b(?:correct\s+(?:answer|option|choice)|answer\s*key|keyed\s+answer)\b"),"answer-key phrase"),
    (re.compile(r"(?i)[\[(](?:x|✓|correct|right|selected)[\])]"),"checked marker"),
    (re.compile(r"(?i)\b(?:selected|right)\s+(?:answer|option|choice)\b"),"selected-answer phrase"),
    (re.compile(r"(?i)\b(?:correct|key|selected|checkmark)\s*[:=-]?\s*[A-E]\b"),"textual key marker"),
    (re.compile(r"(?i)(?:^|\s)(?:answer|key)\s*[:=-]\s*[A-E]\b"),"explicit key label"),
)

def mc_options(row):
    opts=row.get("options")
    if isinstance(opts,list) and len(opts)==5 and all(isinstance(x,str) and x.strip() for x in opts):
        return dict(zip("ABCDE",[x.strip() for x in opts]))
    q=str(row.get("question", ""))
    matches=list(re.finditer(r"(?:^|\s)([A-E])[.)]\s+",q))
    if [m.group(1) for m in matches]!=list("ABCDE") or re.search(r"(?:^|\s)[F-Z][.)]\s+",q): return None
    return {m.group(1):q[m.end():(matches[i+1].start() if i+1<len(matches) else len(q))].strip() for i,m in enumerate(matches)}

def mc_correct_label(row,opts):
    labels=[]
    label=str(row.get("correct_label") or "").strip().upper()
    if label in opts:labels.append(label)
    correct=row.get("correct_option")
    if isinstance(correct,str):
        if correct.strip().upper() in opts:labels.append(correct.strip().upper())
        hits=[k for k,v in opts.items() if re.sub(r"\s+"," ",v).strip().lower()==re.sub(r"\s+"," ",correct).strip().lower()]
        if len(hits)==1:labels.append(hits[0])
    m=re.match(r"^([A-E])(?:[.)]|\s|$)",str(row.get("answer","")).strip(),re.I)
    if m:labels.append(m.group(1).upper())
    resolved=set(labels)
    return next(iter(resolved)) if len(resolved)==1 else None

def mc_leakage(row,opts):
    text="\n".join([str(row.get("question","")),*opts.values()])
    return [name for rx,name in LEAK_PATTERNS if rx.search(text)]

def redact_question(row):
    """Materialize the exact candidate-visible boundary."""
    question=str(row["question"])
    if row.get("format")=="mc" and isinstance(row.get("options"),list):
        # Candidate receives choices but never private key fields.
        question=question.rstrip()+"\n"+"\n".join(f"{k}. {v}" for k,v in zip("ABCDE",row["options"]))
    return {"question_id":row["question_id"],"format":row["format"],"question":question}

def validate_exam_shape(rows):
    required={"question_id","owner","area","format","question","answer","rubric","source_id","source_path","source_sha256","evidence_span","evidence_start","evidence_end","difficulty","skills"}
    errors=[]; ids=[]; questions=[]
    from collections import Counter
    formats=Counter(); difficulty=Counter(); correct_labels=Counter()
    for i,r in enumerate(rows,1):
        missing=required-set(r)
        if missing: errors.append(f"row {i}: missing {sorted(missing)}"); continue
        ids.append(str(r["question_id"])); questions.append(re.sub(r"\s+"," ",str(r["question"])).strip().lower())
        formats[str(r["format"])]+=1; difficulty[str(r["difficulty"])]+=1
        if not all(str(r[k]).strip() for k in ("question","answer","rubric")): errors.append(f"row {i}: empty question/answer/rubric")
        if r["format"]=="mc":
            opts=mc_options(r)
            if not opts:errors.append(f"row {i}: MC must contain exactly five A-E choices");continue
            leaks=mc_leakage(r,opts)
            if leaks:errors.append(f"row {i}: candidate-visible answer leakage: {','.join(leaks)}")
            label=mc_correct_label(r,opts)
            if not label:errors.append(f"row {i}: MC key cannot be resolved")
            else:correct_labels[label]+=1
    if len(rows)!=1000: errors.append(f"row_count={len(rows)} expected=1000")
    if len(ids)!=len(set(ids)): errors.append("duplicate question_id")
    if len(questions)!=len(set(questions)): errors.append("duplicate normalized question")
    if formats!={"mc":500,"qa":500}: errors.append(f"format_mix={dict(formats)}")
    if difficulty!={"introductory":200,"intermediate":500,"advanced":300}: errors.append(f"difficulty_mix={dict(difficulty)}")
    if correct_labels!={"A":100,"B":100,"C":100,"D":100,"E":100}: errors.append(f"mc_key_distribution={dict(correct_labels)} expected=100 each")
    return errors

def preflight_exam(lane,accepted_manifest):
    if accepted_manifest is None: raise SystemExit("--accepted-manifest is required for answers and grades")
    exam=lane/"EXAM.jsonl"
    try: rows=[json.loads(x) for x in exam.read_text().splitlines() if x.strip()]
    except Exception as e: raise SystemExit(f"invalid EXAM.jsonl: {e}")
    errors=validate_exam_shape(rows)
    try: index=osti5000.load_accepted_manifest(accepted_manifest)
    except Exception as e: raise SystemExit(f"invalid accepted OSTI-REP-5000 manifest: {e}")
    errors.extend(osti5000.validate_rows(rows,index))
    if errors: raise SystemExit("exam preflight failed: "+"; ".join(errors[:20]))
    return rows

def identity_matches(requested, observed):
    return isinstance(observed,str) and observed.strip()==requested

def answer_state(resp, identity_ok):
    if not identity_ok: return "IDENTITY_MISMATCH"
    if resp.get("error"):
        return "TIMEOUT" if "timed" in str(resp["error"]).lower() else "ERROR"
    content=str(resp.get("content") or "").strip()
    if not content: return "EMPTY"  # reasoning is evidence, never a final answer
    if resp.get("finish_reason")=="length": return "LENGTH_CAPPED"
    # Conservative first-person refusal only: scientific answers often say
    # a method "cannot" do something, which is subject matter, not refusal.
    if re.search(r"\b(?:i\s+(?:cannot|can't|can’t|am unable to|am not able to|must refuse|will not)|i'm unable to|i’m unable to)\b",content,re.I): return "REFUSAL"
    return "VALID"

def final_content(text):
    """Remove only complete provider-exposed think blocks; keep boundary strict."""
    s=(text or "").strip()
    if "<think>" in s or "</think>" in s:
        s=re.sub(r"<think>.*?</think>","",s,flags=re.S).strip()
        if "<think>" in s or "</think>" in s: return None
    return s

def parse_grade(text):
    final=final_content(text)
    if final is None:return None
    try: g=json.loads(final)
    except Exception: return None
    if not isinstance(g,dict) or set(g)!={"score","match","confidence","reasoning"}: return None
    if type(g["score"]) is not int or g["score"] not in (0,1): return None
    if type(g["match"]) is not bool: return None
    if type(g["confidence"]) not in (int,float) or isinstance(g["confidence"],bool) or not 0<=g["confidence"]<=1: return None
    if not isinstance(g["reasoning"],str): return None
    return g

def completed_keys(path,key_fields,terminal):
    out=set()
    if not path.exists(): return out
    for line in path.read_text(errors="ignore").splitlines():
        try:
            r=json.loads(line)
            if r.get("state") in terminal and all(r.get(x) is not None for x in key_fields):
                out.add(tuple(r[x] for x in key_fields))
        except Exception: pass
    return out

def append_record(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n"
    with LOCK:
        with path.open("a") as f: f.write(encoded); f.flush(); os.fsync(f.fileno())

def effective_timeout(requested):
    """Permit long reasoning/compression endpoints without source edits."""
    return max(float(requested), float(os.environ.get("ARGONIUM_REQUEST_TIMEOUT", "900")))

def call(route,messages,max_tokens,timeout):
    body={"model":route,"messages":messages,"temperature":0,"max_tokens":max_tokens}
    req=urllib.request.Request(BASE,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    start=time.time()
    try:
        with urllib.request.urlopen(req,timeout=effective_timeout(timeout)) as h: raw=h.read()
        d=json.loads(raw); ch=(d.get("choices") or [{}])[0]; msg=ch.get("message") or {}
        return {"raw":raw.decode(errors="replace"),"content":msg.get("content") or "","reasoning_content":msg.get("reasoning_content") or msg.get("reasoning") or "","finish_reason":ch.get("finish_reason"),"usage":d.get("usage"),"actual_model":d.get("model"),"latency_seconds":round(time.time()-start,3),"http_status":200}
    except urllib.error.HTTPError as e:
        raw=e.read().decode(errors="replace"); return {"raw":raw,"error":f"HTTPError:{e.code}","http_status":e.code,"latency_seconds":round(time.time()-start,3)}
    except Exception as e:
        return {"raw":"","error":f"{type(e).__name__}:{e}","http_status":None,"latency_seconds":round(time.time()-start,3)}

def load_roster(contract):
    d=json.loads(contract.read_text()); out=[]
    for x in d["models"]:
        route=x.get("requested_route",x.get("preferred_route"))
        if route=="historical_argonium_config": route=None
        out.append((x["label"],route))
    return out

def qualify(lane,contract,accepted_manifest):
    preflight_exam(lane,accepted_manifest)
    rows=[]; roster=load_roster(contract); prompt=[{"role":"user","content":"Reply with exactly ROUTE_OK."}]
    for label,route in roster+[("Qwen grader",x) for x in GRADERS]:
        if not route:
            rows.append({"label":label,"requested_route":None,"state":"UNRESOLVED_ROUTE","identity_match":False}); continue
        resp=call(route,prompt,32,45); ident=identity_matches(route,resp.get("actual_model")); st=answer_state(resp,ident)
        if st=="VALID" and resp.get("content","").strip()!="ROUTE_OK.": st="BAD_CANARY"
        rows.append({"label":label,"requested_route":route,"actual_model":resp.get("actual_model"),"identity_match":ident,"state":st,"content":resp.get("content"),"finish_reason":resp.get("finish_reason"),"usage":resp.get("usage"),"latency_seconds":resp.get("latency_seconds"),"http_status":resp.get("http_status"),"error":resp.get("error"),"raw_sha256":sha(resp.get("raw","").encode())})
    tmp=lane/"MODEL_ROSTER.json.tmp"; final=lane/"MODEL_ROSTER.json"; tmp.write_text(json.dumps({"schema":"argonium-model-roster-v2","rows":rows},indent=2)+"\n"); os.replace(tmp,final)
    print(json.dumps(rows,indent=2))

def answer_one(owner,q,label,route,path):
    public=redact_question(q)
    resp=call(route,[{"role":"system","content":"Answer the question. MC: state chosen letter and option. QA: answer concisely."},{"role":"user","content":public["question"]}],1024,240)
    ident=identity_matches(route,resp.get("actual_model")); st=answer_state(resp,ident)
    row={"owner":owner,"question_id":q["question_id"],"model_label":label,"public_input_sha256":sha(json.dumps(public,sort_keys=True,separators=(",",":")).encode()),"requested_route":route,"actual_model":resp.get("actual_model"),"identity_match":ident,"state":st,"content":resp.get("content", ""),"reasoning_content":resp.get("reasoning_content", ""),"finish_reason":resp.get("finish_reason"),"usage":resp.get("usage"),"latency_seconds":resp.get("latency_seconds"),"http_status":resp.get("http_status"),"error":resp.get("error"),"raw_response":resp.get("raw", ""),"raw_sha256":sha(resp.get("raw","").encode())}
    append_record(path,row)

def run_answers(lane,contract,workers,accepted_manifest):
    owner=lane.name; qs=preflight_exam(lane,accepted_manifest)
    public_rows=[redact_question(q) for q in qs]
    public_path=lane/"PUBLIC_EXAM.jsonl"; public_tmp=lane/"PUBLIC_EXAM.jsonl.tmp"
    public_tmp.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in public_rows)); os.replace(public_tmp,public_path)
    qualified=json.loads((lane/"MODEL_ROSTER.json").read_text())["rows"]
    qmap={x["label"]:x for x in qualified if x["label"]!="Qwen grader"}
    for label,route in load_roster(contract):
        p=lane/"answers"/(slug(label)+".jsonl"); done=completed_keys(p,("owner","question_id","model_label"),ANSWER_TERMINAL)
        if not route or qmap.get(label,{}).get("state")!="VALID":
            for q in qs:
                key=atomic_key(owner,q["question_id"],label)
                if key not in done: append_record(p,{"owner":owner,"question_id":q["question_id"],"model_label":label,"requested_route":route,"state":"UNRESOLVED_ROUTE","content":"","reasoning_content":""})
            continue
        todo=[q for q in qs if atomic_key(owner,q["question_id"],label) not in done]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex: list(ex.map(lambda q:answer_one(owner,q,label,route,p),todo))

def grade_one(owner,q,a,route,path):
    if a["state"]!="VALID":
        append_record(path,{"owner":owner,"question_id":q["question_id"],"model_label":a["model_label"],"state":"FAILURE_ZERO","score":0,"match":False,"confidence":1,"reasoning":"candidate terminal failure counts zero","grader_route":None}); return
    prompt=f'''Return exactly one JSON object with keys score (0 or 1), match (boolean), confidence (0..1), reasoning (string).\nQUESTION:\n{q["question"]}\nREFERENCE ANSWER:\n{q["answer"]}\nRUBRIC:\n{q["rubric"]}\nCANDIDATE ANSWER:\n{a.get("content","")}'''
    resp=call(route,[{"role":"system","content":"Strict blind scientific grader. JSON only."},{"role":"user","content":prompt}],512,180)
    ident=identity_matches(route,resp.get("actual_model"))
    if not ident: st="GRADER_IDENTITY_MISMATCH"; g=None
    else: g=parse_grade(resp.get("content")); st="VALID_GRADE" if g else "INVALID_GRADE"
    row={"owner":owner,"question_id":q["question_id"],"model_label":a["model_label"],"state":st,"score":g.get("score") if g else None,"match":g.get("match") if g else None,"confidence":g.get("confidence") if g else None,"reasoning":g.get("reasoning") if g else "strict JSON parse/identity failure","grader_route":route,"actual_grader_model":resp.get("actual_model"),"grader_identity_match":ident,"latency":resp.get("latency_seconds"),"raw_response":resp.get("raw"),"raw_sha256":sha(resp.get("raw","").encode())}
    append_record(path,row)

def run_grades(lane,workers,accepted_manifest):
    owner=lane.name; qs={r["question_id"]:r for r in preflight_exam(lane,accepted_manifest)}
    qrows=json.loads((lane/"MODEL_ROSTER.json").read_text())["rows"]
    healthy=[x["requested_route"] for x in qrows if x["label"]=="Qwen grader" and x["state"]=="VALID"]
    expected=set(GRADERS)
    if len(set(healthy))<2 or not set(healthy)<=expected:
        raise SystemExit("Qwen farm requires at least two distinct exact qualified grader routes")
    for ap in sorted((lane/"answers").glob("*.jsonl")):
        ans=[json.loads(x) for x in ap.read_text().splitlines()]; gp=lane/"grades"/ap.name; done=completed_keys(gp,("owner","question_id","model_label"),GRADE_TERMINAL)
        todo=[a for a in ans if atomic_key(owner,a.get("question_id"),a.get("model_label")) not in done]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            def assigned(a,i):
                alternatives=[r for r in healthy if r!=a.get("requested_route")]
                if a.get("requested_route") in expected and not alternatives:
                    raise SystemExit("Qwen candidate requires a different qualified Qwen grader backend")
                return (alternatives or healthy)[i%len(alternatives or healthy)]
            fut=[ex.submit(grade_one,owner,qs[a["question_id"]],a,assigned(a,i),gp) for i,a in enumerate(todo)]; [f.result() for f in fut]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["qualify","answers","grades"]);ap.add_argument("--lane",type=Path,required=True);ap.add_argument("--contract",type=Path,required=True);ap.add_argument("--accepted-manifest",type=Path);ap.add_argument("--workers",type=int,default=2);a=ap.parse_args()
    if a.workers<1 or a.workers>4: raise SystemExit("workers must be 1..4")
    {"qualify":lambda:qualify(a.lane,a.contract,a.accepted_manifest),"answers":lambda:run_answers(a.lane,a.contract,a.workers,a.accepted_manifest),"grades":lambda:run_grades(a.lane,a.workers,a.accepted_manifest)}[a.mode]()
if __name__=="__main__":main()
