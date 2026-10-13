import random
from collections import deque
import numpy as np


MAP = {8: 2, 3: 1, 5: 4, 1: 3}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _process(quad):
    comps = _components((quad != 0) & (quad != 7))
    pattern = max(comps, key=len)
    control = next(comp[0] for comp in comps if comp is not pattern and len(comp) == 1)
    ctrl_color = int(quad[control])
    rows = [r for r, _ in pattern]
    cols = [c for _, c in pattern]
    crop = quad[min(rows):max(rows) + 1, min(cols):max(cols) + 1]
    fill = MAP[ctrl_color]
    out = np.where(crop == 0, fill, 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    parts = [
        _process(grid[:6, :6]),
        _process(grid[:6, 6:]),
        _process(grid[6:, :6]),
        _process(grid[6:, 6:]),
    ]
    top_h = max(parts[0].shape[0], parts[1].shape[0])
    bot_h = max(parts[2].shape[0], parts[3].shape[0])
    left_w = max(parts[0].shape[1], parts[2].shape[1])
    right_w = max(parts[1].shape[1], parts[3].shape[1])
    out = np.zeros((top_h + bot_h, left_w + right_w), dtype=int)
    placements = [(0, 0, parts[0]), (0, left_w, parts[1]), (top_h, 0, parts[2]), (top_h, left_w, parts[3])]
    for r, c, arr in placements:
        out[r:r + arr.shape[0], c:c + arr.shape[1]] = arr
    return out


def generate_input() -> np.ndarray:
    shapes = [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        [(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)],
    ]
    grid = np.zeros((12, 12), dtype=int)
    ctrl_colors = random.sample(list(MAP), 4)
    for idx, (qr, qc) in enumerate([(0, 0), (0, 6), (6, 0), (6, 6)]):
        color = random.choice([c for c in [1, 3, 5, 8] if c != ctrl_colors[idx]])
        shape = shapes[idx]
        sr, sc = random.randint(0, 2), random.randint(0, 2)
        for dr, dc in shape:
            grid[qr + sr + dr, qc + sc + dc] = color
        for _ in range(50):
            r, c = random.randint(0, 5), random.randint(0, 5)
            clear = grid[qr + r, qc + c] == 0 and all(
                not (0 <= r + dr < 6 and 0 <= c + dc < 6) or grid[qr + r + dr, qc + c + dc] == 0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if clear:
                grid[qr + r, qc + c] = ctrl_colors[idx]
                break
        else:
            return generate_input()
        for _ in range(random.randint(1, 2)):
            for _try in range(20):
                r, c = random.randint(0, 5), random.randint(0, 5)
                if grid[qr + r, qc + c] == 0:
                    grid[qr + r, qc + c] = 7
                    break
    return grid
