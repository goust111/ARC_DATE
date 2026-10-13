import random
from collections import Counter, deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(comp)
    return comps


def _box(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), max(rs), min(cs), max(cs)


def _mode(vals):
    return Counter(int(v) for v in vals if v != 0).most_common(1)[0][0]


def _repair(block):
    h, w = block.shape
    visible = [(r, c, int(block[r, c])) for r, c in zip(*np.where(block != 0))]
    best = (h, w)
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            classes = {}
            ok = True
            for r, c, color in visible:
                key = (r % ph, c % pw)
                if key in classes and classes[key] != color:
                    ok = False
                    break
                classes[key] = color
            if ok and all((r % ph, c % pw) in classes for r, c in zip(*np.where(block == 0))):
                best = (ph, pw)
                break
        if best != (h, w):
            break
    ph, pw = best
    classes = {}
    for r, c, color in visible:
        classes.setdefault((r % ph, c % pw), color)
    out = block.copy()
    for r, c in zip(*np.where(out == 0)):
        out[r, c] = classes.get((r % ph, c % pw), 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid != 0)
    panel_comps = [comp for comp in comps if any(grid[r, c] in (2, 7) for r, c in comp)]
    if not panel_comps:
        return grid.copy()
    target = max(panel_comps, key=lambda comp: sum(1 for r, c in comp if grid[r, c] == 2))
    tr0, tr1, tc0, tc1 = _box(target)
    target_block = grid[tr0:tr1 + 1, tc0:tc1 + 1].copy()
    left_color = _mode(target_block[:, 0])
    right_color = _mode(target_block[:, -1])

    motif_comps = [comp for comp in comps if comp not in panel_comps]
    motif = max(motif_comps, key=len) if motif_comps else []
    mr0, mr1, mc0, mc1 = _box(motif)
    block = grid[mr0:mr1 + 1, mc0:mc1 + 1].copy()
    block = _repair(block)
    if block.shape[0] != target_block.shape[0] or block.shape[1] != target_block.shape[1] - 2:
        block = block[:target_block.shape[0], :target_block.shape[1] - 2]

    motif_left = _mode(block[:, 0])
    motif_right = _mode(block[:, -1])
    if motif_left == right_color and motif_right == left_color:
        block = np.fliplr(block)
    out = target_block.copy()
    out[:, 1:-1] = block
    return out


def _panel(grid, r0, c0, h, w, left, right, fill, reds, oranges):
    grid[r0:r0 + h, c0:c0 + w] = fill
    grid[r0:r0 + h, c0] = left
    grid[r0:r0 + h, c0 + w - 1] = right
    cells = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]
    random.shuffle(cells)
    for r, c in cells[:reds]:
        grid[r, c] = 2
    for r, c in cells[reds:reds + oranges]:
        grid[r, c] = 7


def generate_input() -> np.ndarray:
    h, w = 20, 18
    grid = np.zeros((h, w), dtype=int)
    ph, pw = random.choice([(6, 7), (7, 7), (6, 6)])
    side_pairs = [(1, 3), (4, 6), (8, 9)]
    red_counts = [2, 4, 3]
    random.shuffle(red_counts)
    positions = [(1, 1), (1, 10), (10, 1)]
    for i, (r, c) in enumerate(positions):
        left, right = side_pairs[i]
        fill = random.choice([x for x in [1, 3, 4, 6, 8, 9] if x not in (left, right)])
        _panel(grid, r, c, ph, pw, left, right, fill, red_counts[i], random.randint(1, 3))
    target_i = red_counts.index(max(red_counts))
    left, right = side_pairs[target_i]
    inner_w = pw - 2
    motif = np.zeros((ph, inner_w), dtype=int)
    for r in range(ph):
        for c in range(inner_w):
            motif[r, c] = left if c % 2 == 0 else right
    if random.random() < 0.5:
        motif = np.fliplr(motif)
    for _ in range(random.randint(2, 4)):
        r = random.randrange(1, ph - 1)
        c = random.randrange(1, inner_w - 1)
        motif[r, c] = 0
    mr, mc = 13, 10
    grid[mr:mr + ph, mc:mc + inner_w] = motif
    return grid
