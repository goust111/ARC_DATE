import random
import numpy as np


def _find_frames(grid):
    h, w = grid.shape
    frames = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = grid[r:r + 3, c:c + 3]
            border = [int(block[i, j]) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
            center = int(block[1, 1])
            if all(x == 5 for x in border) and center not in (0, 5):
                frames.append((c, center))
    return [center for _, center in sorted(frames)]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    sep_cols = []
    for c in range(w):
        rows = np.where(grid[:, c] == 2)[0]
        if len(rows) >= 3 and np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            sep_cols.append(c)
    c1, c2 = sep_cols[:2]
    panel_h = int(max(np.where(grid == 2)[0]) + 1)
    panels = [grid[:panel_h, :c1], grid[:panel_h, c1 + 1:c2], grid[:panel_h, c2 + 1:]]
    min_w = min(p.shape[1] for p in panels)
    panels = [p[:, :min_w] for p in panels]
    color_to_panel = {}
    for p in panels:
        vals = [int(x) for x in p.reshape(-1) if x not in (0, 2)]
        if vals:
            color_to_panel[vals[0]] = p
    out = np.zeros((panel_h, min_w), dtype=grid.dtype)
    for color in _find_frames(grid):
        panel = color_to_panel[color]
        mask = panel != 0
        out[mask] = panel[mask]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    h, panel_h, pw = 15, 10, 5
    w = pw * 3 + 2
    colors = [1, 3, 4]
    for _ in range(1000):
        grid = np.zeros((h, w), dtype=int)
        sep1, sep2 = pw, pw * 2 + 1
        grid[:panel_h, sep1] = 2
        grid[:panel_h, sep2] = 2
        masks = []
        base = np.zeros((panel_h, pw), dtype=bool)
        for r in range(1, panel_h - 1):
            base[r, r % pw] = True
        for i in range(3):
            mask = base.copy()
            for _p in range(12):
                mask[rng.randrange(panel_h), rng.randrange(pw)] = True
            masks.append(mask)
        for i, color in enumerate(colors):
            left = i * (pw + 1)
            panel = np.zeros((panel_h, pw), dtype=int)
            panel[masks[i]] = color
            grid[:panel_h, left:left + pw] = panel
        order = rng.sample(colors, 3)
        for left, color in zip([1, 7, 13], order):
            grid[12:15, left:left + 3] = 5
            grid[13, left + 1] = color
        return grid
    return generate_input()
