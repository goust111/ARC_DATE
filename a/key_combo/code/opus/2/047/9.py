import numpy as np
from collections import deque


def _connected(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _hole_count(comp: np.ndarray, obj_mask: np.ndarray):
    if len(comp) == 0:
        return 0
    r0, c0 = comp.min(axis=0)
    r1, c1 = comp.max(axis=0)
    sub = obj_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    black = (sub == 0)

    vis = np.zeros((h, w), dtype=bool)
    q = deque()
    for c in range(w):
        for r in (0, h - 1):
            if 0 <= r < h and black[r, c] and not vis[r, c]:
                q.append((r, c))
                vis[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
    holes = 0
    for rr in range(h):
        for cc in range(w):
            if black[rr, cc] and not vis[rr, cc]:
                holes += 1
                q = deque([(rr, cc)])
                vis[rr, cc] = True
                while q:
                    cr, cc2 = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc2 + dc
                        if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
    return holes


def _map_color(h):
    if h == 0:
        return 4
    if h == 1:
        return 1
    if h == 2:
        return 3
    if h == 3:
        return 2
    return 4


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sep_cols = np.where(np.all(g == 1, axis=0))[0]
    sep = int(sep_cols[0]) if len(sep_cols) else w // 2
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    keep_mask = left == 5
    panel = np.where(keep_mask, right, np.zeros_like(right))

    comp_mask = panel == 8
    out = panel.astype(np.int64).copy()
    for comp in _connected(comp_mask):
        h0 = int(comp.shape[0])
        if h0 == 0:
            continue
        holes = _hole_count(comp, panel)
        color = _map_color(holes)
        for r, c in comp:
            out[int(r), int(c)] = color
    return out


def _add_ring(panel, top, left, include_divider=False):
    # adds a 4x4 ring that has one hole
    for c in range(left, left + 4):
        panel[top, c] = 8
        panel[top + 3, c] = 8
    for r in range(top + 1, top + 3):
        panel[r, left] = 8
        panel[r, left + 3] = 8
    if include_divider:
        for r in range(top + 1, top + 3):
            panel[r, left + 1] = 8


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 9
    w = int(rng.integers(4, 7))
    if w < 4:
        w = 4
    left = np.full((h, w), 5, dtype=np.int64)

    right = np.zeros((h, w), dtype=np.int64)
    # object A: one-hole ring
    top_a = int(rng.integers(0, h - 3))
    left_a = int(rng.integers(0, max(1, w - 3)))
    _add_ring(right, top_a, left_a, include_divider=False)

    # object B: two-hole ring (ring + divider), place far enough to avoid overlap
    if top_a < 3 and top_a + 4 <= h - 4:
        top_b = top_a + 4
    elif top_a >= 3 and top_a - 4 >= 0:
        top_b = top_a - 4
    else:
        top_b = 0 if top_a >= h / 2 else h - 4
        if top_b < 0:
            top_b = 0
        if top_b > h - 4:
            top_b = h - 5
    left_b = int(rng.integers(0, max(1, w - 3)))

    _add_ring(right, top_b, left_b, include_divider=True)

    # object C: solid block (0 hole) to ensure unchanged object exists
    top_c = int(rng.integers(0, h - 1))
    left_c = int(rng.integers(0, max(1, w - 1)))
    max_rows = h - top_c
    rows = int(rng.integers(1, min(3, max_rows) + 1))
    max_cols = w - left_c
    cols = int(rng.integers(1, min(3, max_cols) + 1))
    right[top_c : top_c + rows, left_c : left_c + cols] = 8

    # masking: break part of B's divider so its topology changes
    div_cells = [(top_b + 1, left_b + 1), (top_b + 2, left_b + 1)]
    for cell in div_cells:
        if rng.random() < 0.8:
            left[cell] = 0

    grid = np.concatenate([left, np.ones((h, 1), dtype=np.int64), right], axis=1)
    return grid
