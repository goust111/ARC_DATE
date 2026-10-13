import random
from collections import deque
import numpy as np


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


def _rotate(points, k):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    rel = [(r - r0, c - c0) for r, c in points]
    h, w = r1 - r0 + 1, c1 - c0 + 1
    for _ in range(k % 4):
        rel = [(c, h - 1 - r) for r, c in rel]
        h, w = w, h
    return rel


def _external_zero(arr):
    h, w = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if arr[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if arr[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == 0 and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    green = np.argwhere(grid == 3)
    fr0, fc0 = green.min(axis=0)
    fr1, fc1 = green.max(axis=0)
    border = {(fr0, c) for c in range(fc0, fc1 + 1)} | {(fr1, c) for c in range(fc0, fc1 + 1)}
    border |= {(r, fc0) for r in range(fr0, fr1 + 1)} | {(r, fc1) for r in range(fr0, fr1 + 1)}
    missing = sorted(p for p in border if grid[p] != 3)
    blue = [(int(r), int(c)) for r, c in np.argwhere(grid == 1)]
    red_comps = _components(grid == 2)
    single = next((comp[0] for comp in red_comps if len(comp) == 1), None)
    lcomp = next((comp for comp in red_comps if len(comp) == 3), None)
    if single is None or lcomp is None or not missing:
        return out
    br0, br1 = min(r for r, _ in blue), max(r for r, _ in blue)
    bc0, bc1 = min(c for _, c in blue), max(c for _, c in blue)
    corners = [(br0, bc0), (br0, bc1), (br1, bc1), (br1, bc0)]
    sidx = min(range(4), key=lambda i: abs(corners[i][0] - single[0]) + abs(corners[i][1] - single[1]))
    lr0, lr1 = min(r for r, _ in lcomp), max(r for r, _ in lcomp)
    lc0, lc1 = min(c for _, c in lcomp), max(c for _, c in lcomp)
    lcorners = [(lr0, lc0), (lr0, lc1), (lr1, lc1), (lr1, lc0)]
    miss_corner = next(i for i, p in enumerate(lcorners) if p not in lcomp)
    k = (miss_corner - sidx) % 4
    rotated = _rotate(blue, k)
    target_set = set(missing)
    placed = None
    for rr, cc in target_set:
        for pr, pc in rotated:
            off = (rr - pr, cc - pc)
            cand = {(r + off[0], c + off[1]) for r, c in rotated}
            if cand == target_set:
                placed = cand
                break
        if placed:
            break
    if placed is None:
        placed = target_set
    out[out == 1] = 0
    out[out == 2] = 0
    for r, c in placed:
        out[r, c] = 3
    exterior = _external_zero(out)
    interior = (out == 0) & (~exterior)
    out[interior] = 6
    outside = _external_zero(out)
    frame_pts = list(map(tuple, np.argwhere(out == 3)))
    for r, c in frame_pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and out[nr, nc] == 0 and outside[nr, nc]:
                    out[nr, nc] = 2
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((12, 12), dtype=int)
    top, left = random.randint(2, 3), random.randint(2, 3)
    size = 7
    bottom, right = top + size - 1, left + size - 1
    grid[top, left:right + 1] = 3
    grid[bottom, left:right + 1] = 3
    grid[top:bottom + 1, left] = 3
    grid[top:bottom + 1, right] = 3
    gap_start = left + random.randint(2, 3)
    missing = [(top, c) for c in range(gap_start, gap_start + 3)]
    for p in missing:
        grid[p] = 0
    lr, lc = top - 1, gap_start + 2
    l_cells = {(lr, lc - 1), (lr + 1, lc - 1), (lr + 1, lc)}
    for r, c in l_cells:
        grid[r, c] = 2
    candidates = [(1, 10), (8, 1), (8, 10)]
    random.shuffle(candidates)
    for br, bc in candidates:
        single = (br - 1 if br > 0 else br + 3, bc - 1 if bc > 0 else bc + 1)
        blue_cells = {(br + i, bc) for i in range(3)}
        if any(not (0 <= r < 12 and 0 <= c < 12) or grid[r, c] != 0 for r, c in blue_cells):
            continue
        if any(max(abs(single[0] - r), abs(single[1] - c)) <= 1 for r, c in l_cells):
            continue
        for r, c in blue_cells:
            grid[r, c] = 1
        grid[single] = 2
        return grid
    return grid
