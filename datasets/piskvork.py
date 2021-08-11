from pathlib import Path


def load_piskvork_record(path):
    actions = []
    with open(path, 'r') as file:
        for line in file.readlines()[2:-1]:
            row, col = [int(c) for c in line.strip().split(',')][:2]
            actions.append((row, col))
    return actions


def load_piskvork_records(root):
    for path in Path(root).rglob('*.rec'):
        yield load_piskvork_record(path)
