import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Move same-colored fragments into the missing side gaps of outline frames.

    Input structure: black background with several colored rectangular hollow
    outlines, each outline having one straight missing border gap and one small
    same-colored fragment elsewhere on the canvas.
    Rule: for each color, treat the largest component as the damaged outline,
    infer the missing cells on its bounding-box border, erase the other same
    colored fragment components, and paint the missing border cells with that
    color.
    Key insight: color pairs a fragment to an outline, but the gap position is
    recovered from the outline geometry rather than from the fragment's location.
    """
    out = grid.copy()
    for color in sorted(int(x) for x in np.unique(grid) if x != 0):
        comps = _components(grid == color)
        if len(comps) < 2:
            continue
        outline = max(comps, key=len)
        frag_comps = [c for c in comps if c is not outline]
        for comp in frag_comps:
            for r, c in comp:
                out[r, c] = 0
        rows = [p[0] for p in outline]
        cols = [p[1] for p in outline]
        r1, r2, c1, c2 = min(rows), max(rows), min(cols), max(cols)
        border = {(r1, c) for c in range(c1, c2 + 1)}
        border |= {(r2, c) for c in range(c1, c2 + 1)}
        border |= {(r, c1) for r in range(r1, r2 + 1)}
        border |= {(r, c2) for r in range(r1, r2 + 1)}
        for r, c in border:
            if grid[r, c] == 0:
                out[r, c] = color
    return out


def _place_rect(grid, r, c, h, w, color, side, gap_len):
    grid[r:r + h, c] = color
    grid[r:r + h, c + w - 1] = color
    grid[r, c:c + w] = color
    grid[r + h - 1, c:c + w] = color
    if side == "top":
        start = random.randint(c + 1, c + w - gap_len - 1)
        grid[r, start:start + gap_len] = 0
        return [(0, i) for i in range(gap_len)]
    if side == "bottom":
        start = random.randint(c + 1, c + w - gap_len - 1)
        grid[r + h - 1, start:start + gap_len] = 0
        return [(0, i) for i in range(gap_len)]
    if side == "left":
        start = random.randint(r + 1, r + h - gap_len - 1)
        grid[start:start + gap_len, c] = 0
        return [(i, 0) for i in range(gap_len)]
    start = random.randint(r + 1, r + h - gap_len - 1)
    grid[start:start + gap_len, c + w - 1] = 0
    return [(i, 0) for i in range(gap_len)]


def _free(grid, r, c, h, w):
    return 0 <= r and 0 <= c and r + h <= grid.shape[0] and c + w <= grid.shape[1] and np.all(grid[r:r + h, c:c + w] == 0)


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 26), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    anchors = [(2, 2), (2, 14), (13, 3), (13, 15)]
    random.shuffle(anchors)
    for color, (r, c) in zip(colors, anchors):
        rh, rw = random.randint(5, 7), random.randint(5, 8)
        side = random.choice(["top", "bottom", "left", "right"])
        gap_len = random.randint(3, min(5, rh - 2 if side in ("left", "right") else rw - 2))
        shape = _place_rect(grid, r, c, rh, rw, color, side, gap_len)
        if random.random() < 0.5:
            shape = [(-dc, dr) for dr, dc in shape]
        frs = [p[0] for p in shape]
        fcs = [p[1] for p in shape]
        fh, fw = max(frs) - min(frs) + 1, max(fcs) - min(fcs) + 1
        norm = [(rr - min(frs), cc - min(fcs)) for rr, cc in shape]
        for _ in range(100):
            rr = random.randint(1, grid.shape[0] - fh - 1)
            cc = random.randint(1, grid.shape[1] - fw - 1)
            if _free(grid, rr - 1, cc - 1, fh + 2, fw + 2):
                for a, b in norm:
                    grid[rr + a, cc + b] = color
                break
    return grid
