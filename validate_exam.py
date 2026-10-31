#!/usr/bin/env python3
"""Validate Argonium exam: no answer leaks + equilateral distribution."""
import json, re, sys, pathlib, collections

def validate(path):
    data = json.loads(pathlib.Path(path).read_text())
    # Handle both list and dict with questions key
    questions = data if isinstance(data, list) else data.get('questions', data.get('data', []))
    n = len(questions)
    print(f'Validating {path}: {n} questions')

    # 1. No leaks: check question text for (*), CORRE*, correct answer patterns
    leak_patterns = [r'\(\*\)', r'CORRECT ANSWER', r'Correct answer is', r'\*.*correct', r'answer.*is.*\d+\)']
    leaks = []
    for i, q in enumerate(questions):
        qtext = q.get('question','') + ' ' + ' '.join(q.get('choices', []))
        for pat in leak_patterns:
            if re.search(pat, qtext, re.I):
                leaks.append((i, pat, qtext[:100]))
    if leaks:
        print(f'FAIL: {len(leaks)} leaks found')
        for idx, pat, snippet in leaks[:10]:
            print(f'  Q{idx} pat={pat}: {snippet}')
        return False
    else:
        print('PASS: no answer leaks in question text')

    # 2. Equilateral distribution: correct answer positions should be uniform
    # answer field is "The correct answer is N) ..." — extract N
    positions = []
    for q in questions:
        ans = q.get('answer','')
        m = re.search(r'correct answer is (\d+)', ans, re.I)
        if m:
            positions.append(int(m.group(1)))

    if not positions:
        print('WARN: no positions extracted — check answer format')
        return True

    from collections import Counter
    cnt = Counter(positions)
    num_choices = max(positions) if positions else 0
    expected = n / num_choices if num_choices else 0
    print(f'Positions: {dict(cnt)}  expected ~{expected:.1f} each (num_choices={num_choices})')
    # Allow 15% deviation
    for pos in range(1, num_choices+1):
        actual = cnt.get(pos, 0)
        if abs(actual - expected) / expected > 0.15:
            print(f'FAIL: position {pos} has {actual} vs expected {expected:.1f} (>{expected*0.15:.1f} deviation)')
            return False
    print('PASS: equilateral distribution (within 15%)')
    return True

if __name__ == '__main__':
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)
