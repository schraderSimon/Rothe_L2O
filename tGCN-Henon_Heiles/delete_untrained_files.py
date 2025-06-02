from pathlib import Path

for txt in Path('./outputs/').glob('*.txt'):
    with open(txt, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    if num_lines < 25:
        txt.unlink()
        print(f"Deleted {txt} ({num_lines} lines)")