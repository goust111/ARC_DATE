import random
import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the outside motif into a canvas at every border gap.

    Input structure: A large colored hollow rectangular canvas has one-pixel
    gaps along some border sides. A compact multicolor stamp sprite lies outside
    the canvas.
    Rule: Preserve the original stamp and canvas. For every gap, place a copy of
    the stamp just inside the corresponding side, aligning the stamp center
    column or row with the gap coordinate. Gaps are processed in row-major order,
    so later stamps overwrite earlier stamps where they overlap.
    Key insight: A gap encodes both the stamping side and the exact row or column
    alignment; sides without gaps produce no stamp.
    """
    out = grid.copy()
    h, w = grid.shape
    comps = _components(grid != BG)
    canvas = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(canvas)
    canvas_color = int(grid[canvas[0]])
    stamp_comps = [c for c in comps if c is not canvas and not any(r0 <= r <= r1 and c0 <= cc <= c1 for r, cc in c)]
    if not stamp_comps:
        return out
    stamp = max(stamp_comps, key=len)
    sr0, sc0, sr1, sc1 = _bbox(stamp)
    vals = grid[sr0:sr1 + 1, sc0:sc1 + 1]
    sh, sw = vals.shape
    gaps = []
    for c in range(c0, c1 + 1):
        if grid[r0, c] == BG:
            gaps.append((r0, c, "top"))
        if grid[r1, c] == BG:
            gaps.append((r1, c, "bottom"))
    for r in range(r0 + 1, r1):
        if grid[r, c0] == BG:
            gaps.append((r, c0, "left"))
        if grid[r, c1] == BG:
            gaps.append((r, c1, "right"))
    for gr, gc, side in sorted(gaps):
        if side == "top":
            tr, tc = r0 + 1, gc - sw // 2
        elif side == "bottom":
            tr, tc = r1 - sh, gc - sw // 2
        elif side == "left":
            tr, tc = gr - sh // 2, c0 + 1
        else:
            tr, tc = gr - sh // 2, c1 - sw
        for rr in range(sh):
            for cc in range(sw):
                if vals[rr, cc] != BG:
                    nr, nc = tr + rr, tc + cc
                    if r0 < nr < r1 and c0 < nc < c1:
                        out[nr, nc] = int(vals[rr, cc])
    return out


def generate_input() -> np.ndarray:
    h = w = 26
    grid = np.zeros((h, w), dtype=int)
    col = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    r0, c0, r1, c1 = 4, 5, 19, 21
    grid[r0, c0:c1 + 1] = col
    grid[r1, c0:c1 + 1] = col
    grid[r0:r1 + 1, c0] = col
    grid[r0:r1 + 1, c1] = col
    possible = [(r0, random.randint(c0 + 3, c1 - 3)), (r1, random.randint(c0 + 3, c1 - 3)),
                (random.randint(r0 + 3, r1 - 3), c0), (random.randint(r0 + 3, r1 - 3), c1)]
    for r, c in random.sample(possible, random.randint(2, 4)):
        grid[r, c] = BG
    stamp_cols = random.sample([x for x in [1, 2, 3, 4, 6, 7, 8, 9] if x != col], 3)
    stamp = np.array([[0, stamp_cols[0], 0], [stamp_cols[1], stamp_cols[1], stamp_cols[2]], [0, stamp_cols[2], 0]])
    grid[21:24, 2:5] = stamp
    return grid
