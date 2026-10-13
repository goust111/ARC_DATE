import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    blue = np.argwhere(grid == 1)
    r0, c0 = blue.min(axis=0)
    r1, c1 = blue.max(axis=0)
    valid_rows = {r for r in range(grid.shape[0]) if np.any((grid[r, :-1] == 1) & (grid[r, 1:] == 1))}
    valid_cols = {c for c in range(grid.shape[1]) if np.any((grid[:-1, c] == 1) & (grid[1:, c] == 1))}
    gaps = set()
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            if grid[r, c] == 0 and (r in valid_rows or c in valid_cols):
                gaps.add((r, c))
    center_r = (r0 + r1) / 2
    center_c = (c0 + c1) / 2
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in zip(*np.where((grid != 0) & (grid != 1)))]
    for r, c, color in markers:
        out[r, c] = 0
        if r in valid_rows:
            dc = 1 if c < center_c else -1
            rr, cc = r, c
            while 0 <= cc + dc < grid.shape[1]:
                cc += dc
                if (rr, cc) in gaps:
                    out[rr, cc] = color
                    break
        elif c in valid_cols:
            dr = 1 if r < center_r else -1
            rr, cc = r, c
            while 0 <= rr + dr < grid.shape[0]:
                rr += dr
                if (rr, cc) in gaps:
                    out[rr, cc] = color
                    break
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 16)
    grid = np.zeros((h, w), dtype=int)
    rows = [3, 6, 9, 12]
    cols = [3, 6, 9, 12]
    rows = [r for r in rows if r < h - 1]
    cols = [c for c in cols if c < w - 1]
    for r in rows:
        grid[r, cols[0] : cols[-1] + 1] = 1
    for c in cols:
        grid[rows[0] : rows[-1] + 1, c] = 1
    candidates = []
    for r in rows:
        for c in cols[1:-1]:
            candidates.append(("h", r, c))
    for c in cols:
        for r in rows[1:-1]:
            candidates.append(("v", r, c))
    random.shuffle(candidates)
    colors = random.sample([2, 3, 4, 7, 8, 9], random.randint(3, 5))
    used_lines = set()
    for color, (kind, r, c) in zip(colors, candidates):
        key = (kind, r if kind == "h" else c)
        if key in used_lines:
            continue
        used_lines.add(key)
        grid[r, c] = 0
        if kind == "h":
            side = random.choice(["left", "right"])
            mc = cols[0] - 1 if side == "left" else cols[-1] + 1
            if 0 <= mc < w:
                grid[r, mc] = color
        else:
            side = random.choice(["top", "bottom"])
            mr = rows[0] - 1 if side == "top" else rows[-1] + 1
            if 0 <= mr < h:
                grid[mr, c] = color
    return grid
