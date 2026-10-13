import random
import numpy as np


def _split_panels(grid):
    sep = 7
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    starts = [0] + [c + 1 for c in cols]
    ends = [c for c in cols] + [grid.shape[1]]
    return [grid[:, s:e] for s, e in zip(starts, ends) if e > s]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    panels = _split_panels(grid)[:3]
    h, w = panels[0].shape
    known = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=grid.dtype)
    for panel in panels:
        for r in range(h):
            for c in range(w):
                if panel[r, c] != 3:
                    out[r, c] = panel[r, c]
                    known[r, c] = True
    changed = True
    while changed:
        changed = False
        for r in range(h):
            for c in range(w):
                if known[r, c]:
                    continue
                equivalents = [(r, c), (h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)]
                for rr, cc in equivalents:
                    if known[rr, cc]:
                        out[r, c] = out[rr, cc]
                        known[r, c] = True
                        changed = True
                        break
    return out


def _symmetric_pattern(h, w, rng):
    colors = [0, 1, 2, 4, 6, 8, 9]
    pat = np.zeros((h, w), dtype=int)
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            color = rng.choice(colors)
            for rr, cc in ((r, c), (h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)):
                pat[rr, cc] = color
    if np.count_nonzero(pat) < h:
        pat[h // 2, w // 2] = 1
    return pat


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.choice([6, 7, 8])
    w = rng.choice([5, 6, 7])
    pattern = _symmetric_pattern(h, w, rng)
    panels = [pattern.copy() for _ in range(3)]
    anchor = (rng.randint(1, h - 2), rng.randint(1, w - 2))
    rects = [
        (max(0, anchor[0] - 1), max(0, anchor[1] - 1), min(h, anchor[0] + 1), min(w, anchor[1] + 2)),
        (max(0, anchor[0] - 1), max(0, anchor[1]), min(h, anchor[0] + 2), min(w, anchor[1] + 2)),
        (max(0, anchor[0]), max(0, anchor[1] - 1), min(h, anchor[0] + 2), min(w, anchor[1] + 1)),
    ]
    for panel, (r0, c0, r1, c1) in zip(panels, rects):
        panel[r0:r1, c0:c1] = 3
    out = np.full((h, 3 * w + 2), 7, dtype=int)
    out[:, 0:w] = panels[0]
    out[:, w + 1:2 * w + 1] = panels[1]
    out[:, 2 * w + 2:3 * w + 2] = panels[2]
    return out
