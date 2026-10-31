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
python area_exam_runner.py qualify --lane <lane> --contract <contract>
python area_exam_runner.py answers --lane <lane> --contract <contract> --workers 2
python area_exam_runner.py grades  --lane <lane> --contract <contract> --workers 2
```

Never start `answers` until `qualify` creates an exact-identity `MODEL_ROSTER.json`.
