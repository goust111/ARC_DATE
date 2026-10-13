import random
from collections import deque
import numpy as np


def _label(mask):
    h, w = mask.shape
    lab = -np.ones((h, w), dtype=int)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or lab[r, c] >= 0:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            lab[r, c] = idx
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and lab[nx, ny] < 0:
                        lab[nx, ny] = idx
                        q.append((nx, ny))
            comps.append(comp)
    return lab, comps


def _has_hole(pixels):
    rs, cs = [p[0] for p in pixels], [p[1] for p in pixels]
    r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
    h, w = r1 - r0 + 1, c1 - c0 + 1
    solid = np.zeros((h, w), dtype=bool)
    for r, c in pixels:
        solid[r - r0, c - c0] = True
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if not solid[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not solid[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any((~solid) & (~seen)))


def _fragments(grid):
    h, w = grid.shape
    rows = [int(r) for r in np.where(grid[:, 0] == 2)[0]]
    cols = [int(c) for c in np.where(grid[0] == 2)[0]]
    barrier = np.zeros((h, w), dtype=bool)
    for r in rows:
        barrier[r, :] = True
    for c in cols:
        barrier[:, c] = True
    lab, comps = _label(~barrier)
    sizes = {i: len(c) for i, c in enumerate(comps)}
    assigned = {}
    for r, c in map(tuple, np.argwhere(grid == 1)):
        if lab[r, c] >= 0:
            assigned[(r, c)] = int(lab[r, c])
            continue
        choices = []
        for dr, dc in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and lab[nr, nc] >= 0:
                priority = 0 if (dr, dc) in ((-1, 0), (0, -1)) else 1
                choices.append((-sizes[int(lab[nr, nc])], priority, int(lab[nr, nc])))
        assigned[(r, c)] = min(choices)[2] if choices else -1
    seen = set()
    frags = []
    for p, region in assigned.items():
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (x + dx, y + dy)
                if q in assigned and q not in seen and assigned[q] == region:
                    seen.add(q)
                    stack.append(q)
        frags.append(comp)
    return frags


def _place_slide(out, pixels, upward):
    h, w = out.shape
    dr = -1 if upward else 1
    cur = 0
    while True:
        trial = cur + dr
        moved = [(r + trial, c) for r, c in pixels]
        if any(r < 0 or r >= h or out[r, c] != 0 for r, c in moved):
            break
        cur = trial
    for r, c in pixels:
        out[r + cur, c] = 1


def main(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    frags = _fragments(grid)
    closed = [f for f in frags if _has_hole(f)]
    open_ = [f for f in frags if not _has_hole(f)]
    closed.sort(key=lambda f: min(r for r, _ in f))
    open_.sort(key=lambda f: max(r for r, _ in f), reverse=True)
    for frag in closed:
        _place_slide(out, frag, True)
    for frag in open_:
        _place_slide(out, frag, False)
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(14, 17)
    g = np.zeros((h, w), dtype=int)
    rr = random.randint(6, 8)
    cc = random.randint(6, 8)
    g[0, cc] = 2
    g[rr, 0] = 2
    r0 = rr - random.randint(4, 5)
    c0 = cc - random.randint(4, 5)
    r0 = max(1, r0)
    c0 = max(1, c0)
    g[r0:r0 + 5, c0:c0 + 5] = 1
    g[r0 + 1:r0 + 4, c0 + 1:c0 + 4] = 0
    g[r0 + 2, c0 + 2] = 0
    br = min(h - 4, rr + 1)
    bc = min(w - 4, cc + 1)
    g[br:br + 3, bc:bc + 3] = 1
    for c in range(c0 + 4, bc + 1):
        g[rr, c] = 1
    for r in range(rr, br + 1):
        g[r, bc] = 1
    return g
