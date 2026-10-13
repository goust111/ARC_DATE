import random
from collections import deque

import numpy as np


def _flood_from_marker(g: np.ndarray):
    H, W = g.shape
    pts = [(r, c) for r in range(H) for c in range(W) if int(g[r, c]) == 6]
    if not pts:
        return np.zeros((H, W), dtype=bool)
    r0, c0 = pts[0]
    q = deque([(r0, c0)])
    visited = np.zeros((H, W), dtype=bool)
    visited[r0, c0] = True

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if visited[nr, nc] or int(g[nr, nc]) != 0:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return visited


def _components(g: np.ndarray, color: int):
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) != color:
                continue
            q = [(r, c)]
            visited[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if nx < 0 or nx >= H or ny < 0 or ny >= W:
                        continue
                    if visited[nx, ny] or int(g[nx, ny]) != color:
                        continue
                    visited[nx, ny] = True
                    q.append((nx, ny))
            comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = np.zeros((H, W), dtype=int)

    reachable = _flood_from_marker(g)

    mask_ctrl = (g != 0) & (g != 5) & (g != 6) & (g != 9)
    if not mask_ctrl.any():
        return out
    rs = np.where(mask_ctrl.any(axis=1))[0]
    cs = np.where(mask_ctrl.any(axis=0))[0]
    cr0, cr1 = rs.min(), rs.max()
    cc0, cc1 = cs.min(), cs.max()
    ctrl = g[cr0 : cr1 + 1, cc0 : cc1 + 1]

    comps = _components(g, 5)
    if not comps:
        return out

    top_rows = sorted({min(r for r, _ in comp) for comp in comps})
    left_cols = sorted({min(c for _, c in comp) for comp in comps})

    for comp in comps:
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, c0 = min(rows), min(cols)
        r1, c1 = max(rows), max(cols)

        reachable_comp = False
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if int(g[nr, nc]) == 0 and reachable[nr, nc]:
                    reachable_comp = True
                    break
            if reachable_comp:
                break

        ri = top_rows.index(r0) if r0 in top_rows else None
        ci = left_cols.index(c0) if c0 in left_cols else None
        if ri is None or ci is None:
            continue

        if ri >= ctrl.shape[0] or ci >= ctrl.shape[1]:
            continue
        new_color = int(ctrl[ri, ci])

        if reachable_comp and new_color not in (0, 5, 6, 9):
            for r, c in comp:
                out[r, c] = new_color
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = 0

    return out


def generate_input() -> np.ndarray:
    H = 20
    W = 20
    g = np.zeros((H, W), dtype=int)
    g[0, 0] = 6

    rows = random.randint(3, 4)
    cols = random.randint(3, 4)

    palette = random.sample([1, 2, 3, 4, 7, 8, 9], 5)
    base_r = random.randint(1, 2)
    base_c = random.randint(1, 3)

    # place 3x3 template lattice with separators
    for rr in range(rows):
        for cc in range(cols):
            r0 = base_r + rr * 4
            c0 = base_c + cc * 4
            for r in range(3):
                for c in range(3):
                    if r in (0, 2) or c in (0, 2):
                        g[r0 + r, c0 + c] = 5

    # optional separator wall to block some templates
    if rows > 1:
        wall_row = base_r + 4 + random.randint(0, max(0, rows - 2)) * 4 - 1
        wall_row = max(1, min(H - 2, wall_row))
        for c in range(base_c, base_c + cols * 4 - 1):
            if g[wall_row, c] == 0:
                g[wall_row, c] = 9

    # control matrix
    ctrl_r0 = base_r + rows * 4 + 2
    ctrl_c0 = base_c
    for r in range(rows):
        for c in range(cols):
            if ctrl_r0 + r < H and ctrl_c0 + c < W:
                g[ctrl_r0 + r, ctrl_c0 + c] = palette[(r * cols + c + random.randint(0, 4)) % len(palette)]

    return g
