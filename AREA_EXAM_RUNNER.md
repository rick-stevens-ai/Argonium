# Argonium Area Exam runner

`area_exam_runner.py` is the contract-safe runner for the 11-member area-exam campaign.

## Guarantees

- exact requested/observed model identity; missing identity fails closed;
- reasoning-only output is not accepted as a final answer;
- strict JSON grade envelope with no prose recovery;
- Qwen-only grader qualification;
- append-only, fsync-backed atomic answer and grade rows;
- resume by `(owner, question_id, model_label)` terminal state;
- bounded `1..4` workers per lane;
- timeout floor controlled by `ARGONIUM_REQUEST_TIMEOUT`.

For endpoints performing long context compression or reasoning, raise the timeout rather than relaunching:

```bash
export ARGONIUM_REQUEST_TIMEOUT=1800  # 30 minutes
```

## Required order

```bash
python area_exam_runner.py qualify --lane <lane> --contract <contract> --accepted-manifest <accepted-5000.jsonl>
python area_exam_runner.py answers --lane <lane> --contract <contract> --accepted-manifest <accepted-5000.jsonl> --workers 2
python area_exam_runner.py grades  --lane <lane> --contract <contract> --accepted-manifest <accepted-5000.jsonl> --workers 2
```

Never start `answers` until `qualify` creates an exact-identity `MODEL_ROSTER.json`.

Before qualification or model spend, validate the exam against the immutable accepted 5,000-paper manifest:

```bash
python validate_osti5000_exam.py <lane>/EXAM.jsonl \
  --accepted-manifest /path/to/OSTI_REP5000_ACCEPTED.jsonl
```

The validator rejects a development snapshot, any count other than exactly 5,000,
duplicate paper IDs, and any row not explicitly marked `status: ACCEPT`.
