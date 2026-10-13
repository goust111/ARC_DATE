import random
import numpy as np


GLYPHS = {
    2: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    3: [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],
}


def _intervals(lines, n):
    cuts = [-1] + sorted(lines) + [n]
    return [(cuts[i] + 1, cuts[i + 1] - 1) for i in range(len(cuts) - 1) if cuts[i] + 1 <= cuts[i + 1] - 1]


def _base(panel):
    vals, counts = np.unique(panel[panel != 5], return_counts=True)
    if len(vals) == 0:
        return 0
    return int(vals[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """Expand red/green controls to glyph masks, swap masks across each horizontal panel pair, and recolor by opposite bases."""
    out = grid.copy()
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == 5)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == 5)]
    rows = _intervals(sep_rows, h)
    cols = _intervals(sep_cols, w)
    if len(rows) < 2 or len(cols) < 2:
        return out
    rows = rows[:2]
    cols = cols[:2]
    legend = (slice(0, min(4, h)), slice(0, min(8, w)))

    panels = {}
    masks = {}
    bases = {}
    expanded = grid.copy()
    for ri, (r0, r1) in enumerate(rows):
        for ci, (c0, c1) in enumerate(cols):
            panel = grid[r0 : r1 + 1, c0 : c1 + 1]
            base = _base(panel)
            bases[(ri, ci)] = base
            mask = set()
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if r < 4 and c < 8:
                        continue
                    col = int(grid[r, c])
                    if col in GLYPHS:
                        for dr, dc in GLYPHS[col]:
                            rr, cc = r + dr, c + dc
                            if r0 <= rr <= r1 and c0 <= cc <= c1:
                                expanded[rr, cc] = col
                                mask.add((rr - r0, cc - c0))
            panels[(ri, ci)] = (r0, r1, c0, c1)
            masks[(ri, ci)] = mask

    out[:] = expanded
    out[legend] = 0
    for ri in range(2):
        left = panels[(ri, 0)]
        right = panels[(ri, 1)]
        lb, rb = bases[(ri, 0)], bases[(ri, 1)]
        for r0, r1, c0, c1, base in [(*left, lb), (*right, rb)]:
            out[r0 : r1 + 1, c0 : c1 + 1] = base
        for dr, dc in masks[(ri, 1)]:
            rr, cc = left[0] + dr, left[2] + dc
            if left[0] <= rr <= left[1] and left[2] <= cc <= left[3]:
                out[rr, cc] = rb
        for dr, dc in masks[(ri, 0)]:
            rr, cc = right[0] + dr, right[2] + dc
            if right[0] <= rr <= right[1] and right[2] <= cc <= right[3]:
                out[rr, cc] = lb
    for r in sep_rows:
        out[r, :] = 5
    for c in sep_cols:
        out[:, c] = 5
    out[legend] = 0
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((14, 20), dtype=int)
    bases = random.sample([4, 6, 8, 9], 4)
    g[:6, :9] = bases[0]
    g[:6, 10:] = bases[1]
    g[7:, :9] = bases[2]
    g[7:, 10:] = bases[3]
    g[6, :] = 5
    g[:, 9] = 5
    # small legend
    for r, c in [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]:
        g[r, c] = 2
    for r, c in [(1, 5), (0, 4), (0, 6), (2, 4), (2, 6)]:
        g[r, c] = 3
    controls = [(3, random.choice([5, 6, 7]), 2), (4, random.choice([13, 14, 15]), 3),
                (10, random.choice([3, 4, 5]), 3), (9, random.choice([15, 16, 17]), 2)]
    if random.random() < 0.65:
        controls.append((random.choice([10, 11]), random.choice([5, 6, 7]), 2))
        controls.append((random.choice([10, 11]), random.choice([16, 17, 18]), 3))
    for r, c, col in controls:
        g[r, c] = col
    return g
