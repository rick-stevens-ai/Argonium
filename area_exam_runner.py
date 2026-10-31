#!/usr/bin/env python3
"""Contract-safe Argonium 11K answer/grade runner.

Answers and grades are separate phases. Exact route identity is fail-closed.
Every append is fsync'd; resume only recognizes explicit terminal states.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, threading, time
import urllib.error, urllib.request
from pathlib import Path

BASE=os.environ.get("ARGONIUM_BASE_URL","http://127.0.0.1:4000/v1/chat/completions")
KEY=os.environ.get("ARGONIUM_API_KEY","stevens")
ANSWER_TERMINAL={"VALID","REFUSAL","EMPTY","ERROR","TIMEOUT","LENGTH_CAPPED","IDENTITY_MISMATCH","UNRESOLVED_ROUTE"}
GRADE_TERMINAL={"VALID_GRADE","FAILURE_ZERO","INVALID_GRADE","GRADER_IDENTITY_MISMATCH"}
GRADERS=("pecans/qwen3.8-27b","uicgpu/qwen3.8-27b")
LOCK=threading.Lock()

def sha(b): return hashlib.sha256(b).hexdigest()
def slug(s): return re.sub(r"[^a-z0-9]+","-",s.lower()).strip("-")
def atomic_key(owner,qid,label): return owner,qid,label

def identity_matches(requested, observed):
    return isinstance(observed,str) and observed.strip()==requested

def answer_state(resp, identity_ok):
    if not identity_ok: return "IDENTITY_MISMATCH"
    if resp.get("error"):
        return "TIMEOUT" if "timed" in str(resp["error"]).lower() else "ERROR"
    content=str(resp.get("content") or "").strip()
    if not content: return "EMPTY"  # reasoning is evidence, never a final answer
    if resp.get("finish_reason")=="length": return "LENGTH_CAPPED"
    if re.search(r"\b(cannot|can't|unable to|refuse)\b",content,re.I): return "REFUSAL"
    return "VALID"

def parse_grade(text):
    try: g=json.loads((text or "").strip())
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
        route=x.get("preferred_route")
        if route=="historical_argonium_config": route=None
        out.append((x["label"],route))
    return out

def qualify(lane,contract):
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
    resp=call(route,[{"role":"system","content":"Answer the question. MC: state chosen letter and option. QA: answer concisely."},{"role":"user","content":q["question"]}],1024,240)
    ident=identity_matches(route,resp.get("actual_model")); st=answer_state(resp,ident)
    row={"owner":owner,"question_id":q["question_id"],"model_label":label,"requested_route":route,"actual_model":resp.get("actual_model"),"identity_match":ident,"state":st,"content":resp.get("content", ""),"reasoning_content":resp.get("reasoning_content", ""),"finish_reason":resp.get("finish_reason"),"usage":resp.get("usage"),"latency_seconds":resp.get("latency_seconds"),"http_status":resp.get("http_status"),"error":resp.get("error"),"raw_response":resp.get("raw", ""),"raw_sha256":sha(resp.get("raw","").encode())}
    append_record(path,row)

def run_answers(lane,contract,workers):
    owner=lane.name; qs=[json.loads(x) for x in (lane/"EXAM.jsonl").read_text().splitlines()]
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
    if a["state"] not in {"VALID","LENGTH_CAPPED"}:
        append_record(path,{"owner":owner,"question_id":q["question_id"],"model_label":a["model_label"],"state":"FAILURE_ZERO","score":0,"match":False,"confidence":1,"reasoning":"candidate terminal failure counts zero","grader_route":None}); return
    prompt=f'''Return exactly one JSON object with keys score (0 or 1), match (boolean), confidence (0..1), reasoning (string).\nQUESTION:\n{q["question"]}\nREFERENCE ANSWER:\n{q["answer"]}\nRUBRIC:\n{q["rubric"]}\nCANDIDATE ANSWER:\n{a.get("content","")}'''
    resp=call(route,[{"role":"system","content":"Strict blind scientific grader. JSON only."},{"role":"user","content":prompt}],512,180)
    ident=identity_matches(route,resp.get("actual_model"))
    if not ident: st="GRADER_IDENTITY_MISMATCH"; g=None
    else: g=parse_grade(resp.get("content")); st="VALID_GRADE" if g else "INVALID_GRADE"
    row={"owner":owner,"question_id":q["question_id"],"model_label":a["model_label"],"state":st,"score":g.get("score") if g else None,"match":g.get("match") if g else None,"confidence":g.get("confidence") if g else None,"reasoning":g.get("reasoning") if g else "strict JSON parse/identity failure","grader_route":route,"actual_grader_model":resp.get("actual_model"),"grader_identity_match":ident,"latency":resp.get("latency_seconds"),"raw_response":resp.get("raw"),"raw_sha256":sha(resp.get("raw","").encode())}
    append_record(path,row)

def run_grades(lane,workers):
    owner=lane.name; qs={r["question_id"]:r for r in map(json.loads,(lane/"EXAM.jsonl").read_text().splitlines())}
    qrows=json.loads((lane/"MODEL_ROSTER.json").read_text())["rows"]
    healthy=[x["requested_route"] for x in qrows if x["label"]=="Qwen grader" and x["state"]=="VALID"]
    if not healthy: raise SystemExit("No exact healthy Qwen grader route")
    for ap in sorted((lane/"answers").glob("*.jsonl")):
        ans=[json.loads(x) for x in ap.read_text().splitlines()]; gp=lane/"grades"/ap.name; done=completed_keys(gp,("owner","question_id","model_label"),GRADE_TERMINAL)
        todo=[a for a in ans if atomic_key(owner,a.get("question_id"),a.get("model_label")) not in done]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            fut=[ex.submit(grade_one,owner,qs[a["question_id"]],a,healthy[i%len(healthy)],gp) for i,a in enumerate(todo)]; [f.result() for f in fut]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["qualify","answers","grades"]);ap.add_argument("--lane",type=Path,required=True);ap.add_argument("--contract",type=Path,required=True);ap.add_argument("--workers",type=int,default=2);a=ap.parse_args()
    if a.workers<1 or a.workers>4: raise SystemExit("workers must be 1..4")
    {"qualify":lambda:qualify(a.lane,a.contract),"answers":lambda:run_answers(a.lane,a.contract,a.workers),"grades":lambda:run_grades(a.lane,a.workers)}[a.mode]()
if __name__=="__main__":main()
