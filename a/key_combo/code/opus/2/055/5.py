import random
from collections import deque
from typing import List, Tuple

import numpy as np


def _components_zero(g: np.ndarray):
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) != 0:
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
                    if visited[nx, ny] or int(g[nx, ny]) != 0:
                        continue
                    visited[nx, ny] = True
                    q.append((nx, ny))
            comps.append(pts)
    return comps


def _collect_containers(g: np.ndarray):
    H, W = g.shape
    comps = _components_zero(g)
    containers = []
    for comp in comps:
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        # border-only frame
        if len(comp) != 2 * (h + w) - 4:
            continue
        ok = True
        s = set(comp)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                is_border = (rr in (r0, r1)) or (cc in (c0, c1))
                if is_border != ((rr, cc) in s):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        containers.append((r0, r1, c0, c1))
    return containers


def _find_marker_and_target(g: np.ndarray, cont):
    r0, r1, c0, c1 = cont
    H, W = g.shape

    # marker candidate per side
    for c in range(c0, c1 + 1):
        if c0 - 1 >= 0 and int(g[r0 - 1, c]) == 9:
            mr, mc = r0 - 1, c
            for cc in range(mc + 1, W):
                if int(g[mr, cc]) == 9:
                    return "right", (mr, mc), (mr, cc), (0, 1)
        if c1 + 1 < W and int(g[r1 + 1, c]) == 9:
            mr, mc = r1 + 1, c
            for cc in range(mc - 1, -1, -1):
                if int(g[mr, cc]) == 9:
                    return "up", (mr, mc), (mr, cc), (-1, 0)

    for r in range(r0, r1 + 1):
        if r0 - 1 >= 0 and int(g[r, c0 - 1]) == 9:
            mr, mc = r, c0 - 1
            for cc in range(1, 1):
                pass
            for cc in range(mc + 1, W):
                if int(g[mr, cc]) == 9:
                    # marker is left side, move right
                    return "right", (mr, mc), (mr, cc), (0, 1)
        if c0 <= c1 and False:
            pass

    for r in range(r0, r1 + 1):
        if c1 + 1 < W and int(g[r, c1 + 1]) == 9:
            mr, mc = r, c1 + 1
            for cc in range(mc - 1, -1, -1):
                if int(g[mr, cc]) == 9:
                    return "left", (mr, mc), (mr, cc), (0, -1)
        if c1 + 1 < W and int(g[r, c1 + 1]) == 9:
            pass

    # robust side detection by explicit edges:
    for c in range(c0, c1 + 1):
        if c0 - 1 >= 0 and int(g[r0, c]) == 9:
            mr, mc = r0, c
            for rr in range(mr + 1, H):
                if int(g[rr, mc]) == 9:
                    return "down", (mr, mc), (rr, mc), (1, 0)
        if c1 + 1 < W and int(g[r1, c]) == 9:
            mr, mc = r1, c
            for rr in range(mr - 1, -1, -1):
                if int(g[rr, mc]) == 9:
                    return "up", (mr, mc), (rr, mc), (-1, 0)

    # fallback: check left/right sides only
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0 and int(g[r, c0 - 1]) == 9:
            mr, mc = r, c0 - 1
            for cc in range(mc + 1, W):
                if int(g[mr, cc]) == 9:
                    return "right", (mr, mc), (mr, cc), (0, 1)
    return None


def _order_gray(interior_coords: List[Tuple[int, int]], direction: str):
    if direction == "right":
        return sorted(interior_coords, key=lambda x: (x[1], x[0]))
    if direction == "left":
        return sorted(interior_coords, key=lambda x: (-x[1], x[0]))
    if direction == "down":
        return sorted(interior_coords, key=lambda x: (x[0], x[1]))
    return sorted(interior_coords, key=lambda x: (-x[0], x[1]))


def _draw_contour(g: np.ndarray, mask: np.ndarray, color: int):
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if g[r, c] != 6 or not mask[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not mask[nr, nc]:
                        g[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    g0 = np.array(grid, dtype=int)
    H, W = g0.shape
    out = np.full((H, W), 6, dtype=int)

    containers = _collect_containers(g0)
    contour_targets = []

    for r0, r1, c0, c1 in containers:
        parsed = _find_marker_and_target(g0, (r0, r1, c0, c1))
        if parsed is None:
            continue
        direction, (mr, mc), (tr, tc), (dr, dc) = parsed
        D = abs(tr - mr) + abs(tc - mc)

        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        interior_vals = { (r, c): int(g0[r, c]) for r, c in interior }
        gray_coords = [(r, c) for r, c in interior if interior_vals[(r, c)] == 5]
        G = len(gray_coords)
        S = min(D, G)

        gray_order = _order_gray(gray_coords, direction)
        charged = set(gray_order[:S])

        # translated position
        nr0, nr1 = r0 + dr * S, r1 + dr * S
        nc0, nc1 = c0 + dc * S, c1 + dc * S
        if nr0 < 0 or nr1 >= H or nc0 < 0 or nc1 >= W:
            nr0, nr1, nc0, nc1 = r0, r1, c0, c1

        # frame
        for c in range(nc0, nc1 + 1):
            out[nr0, c] = 0
            out[nr1, c] = 0
        for r in range(nr0, nr1 + 1):
            out[r, nc0] = 0
            out[r, nc1] = 0

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                v = interior_vals[(r, c)]
                if v == 7:
                    pass
                nr = r + dr * S
                nc = c + dc * S
                if (r, c) in charged:
                    v = 7
                if v not in (5, 7):
                    v = 6
                out[nr, nc] = v

        # determine full charge with original interior gray count vs steps
        fully = (S == G)
        # contour info
        mask = np.zeros((H, W), dtype=bool)
        for rr in range(nr0, nr1 + 1):
            for cc in range(nc0, nc1 + 1):
                mask[rr, cc] = True
        contour_targets.append((mask, 3 if fully else 2))

    # draw green first then red
    for mask, color in contour_targets:
        if color == 3:
            for r in range(H):
                for c in range(W):
                    if out[r, c] != 6:
                        continue
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                                out[r, c] = 3
                                break
                        else:
                            continue
                        break

    for mask, color in contour_targets:
        if color == 2:
            for r in range(H):
                for c in range(W):
                    if out[r, c] != 6:
                        continue
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                                out[r, c] = 2
                                break
                        else:
                            continue
                        break

    # erase markers and targets
    out[g0 == 9] = 6

    return out


def generate_input() -> np.ndarray:
    H = 18
    W = 18
    g = np.full((H, W), 6, dtype=int)

    num = random.randint(2, 4)
    used = np.zeros((H, W), dtype=bool)
    cols = random.sample([1, 2, 3, 4, 7], num)

    # Keep fixed orientation: some left/right/up/down
    for i in range(num):
        ih = random.randint(3, 5)
        iw = random.randint(2, 4)
        Hc = ih + 2
        Wc = iw + 2
        r0 = 2 + i * 4
        if r0 + Hc >= H - 2:
            r0 = H - Hc - 2
        c0 = 2 + i * 3
        if c0 + Wc >= W - 2:
            c0 = 2

        r1, c1 = r0 + Hc - 1, c0 + Wc - 1

        # draw border
        for c in range(c0, c1 + 1):
            g[r0, c] = 0
            g[r1, c] = 0
        for r in range(r0 + 1, r1):
            g[r, c0] = 0
            g[r, c1] = 0

        # fill interior random with 5/7
        gray_cells = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if random.random() < 0.65:
                    g[r, c] = 5
                    gray_cells.append((r, c))
                else:
                    g[r, c] = 7
        if not gray_cells:
            g[r0 + 1, c0 + 1] = 5
            gray_cells.append((r0 + 1, c0 + 1))

        G = len(gray_cells)

        # Marker side alternates
        if i % 2 == 0:
            # marker at left, target to right => move right
            mr, mc = r0 + 1, c0 - 1
            if mc < 0:
                mr, mc = r0 + 2, c1 + 1
                D = random.randint(1, 4)
                tc = mc - D
            else:
                space = max(1, W - (c1 + 1))
                D = random.randint(1, min(6, space))
                tc = c1 + D
        else:
            # marker at right, target to left => move left
            mr, mc = r0 + 1, c1 + 1
            if mc >= W:
                mc = c0 - 1
            D = random.randint(1, min(6, max(1, c0)))
            tc = c0 - D

        if i == 0 and G > 1:
            # force full charge
            D = G
        if i == 1:
            # force partial if possible
            if G > 1:
                D = max(1, G - 1)

        # marker and target
        if 0 <= mc < W:
            g[mr, mc] = 9
        if 0 <= mr < H and 0 <= tc < W:
            g[mr, tc] = 9

    return g
