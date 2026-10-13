from __future__ import annotations

import numpy as np


def _find_arrow_and_direction(g: np.ndarray) -> tuple[int, int, int, int]:
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w - 1):
            b = g[r : r + 2, c : c + 2]
            if np.sum(b == 5) != 3:
                continue
            # determine missing corner
            coords = (b == 5)
            if coords[0, 0] == 0:
                # missing top-left
                return r, c, -1, -1
            if coords[0, 1] == 0:
                # missing top-right
                return r, c + 1, -1, 1
            if coords[1, 0] == 0:
                # missing bottom-left
                return r + 1, c, 1, -1
            if coords[1, 1] == 0:
                # missing bottom-right
                return r + 1, c + 1, 1, 1
    raise ValueError("arrow not found")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    if h != w:
        n = min(h, w)
        g = g[:n, :n]
        h = n

    try:
        sr, sc, dr, dc = _find_arrow_and_direction(g)
    except Exception:
        return g.copy()

    wall = 7
    H, W = g.shape
    r, c = sr, sc
    bounces = 0
    step_limit = max(H, W) * max(H, W) * 8
    for _ in range(step_limit):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if g[nr, nc] == wall:
            # Front hit (head-on): invert both axes
            dr = -dr
            dc = -dc
            bounces += 1
            continue
        side_hit = False
        if 0 <= r + dr < H and g[r + dr, c] == wall:
            dr = -dr
            bounces += 1
            side_hit = True
        if 0 <= c + dc < W and g[r, c + dc] == wall:
            dc = -dc
            bounces += 1
            side_hit = True
        if side_hit:
            continue
        r, c = nr, nc

    # Step 4: partition by bounce count
    divisor = bounces + 1
    if divisor <= 0 or H % divisor != 0:
        return g.copy()
    blk = H // divisor
    br = sr // blk
    bc = sc // blk
    rr0, cc0 = br * blk, bc * blk
    return g[rr0 : rr0 + blk, cc0 : cc0 + blk].copy()


_RNG = np.random.default_rng()


def _simulate(g: np.ndarray, sr: int, sc: int, dr: int, dc: int) -> int:
    h, w = g.shape
    r, c = sr, sc
    bounces = 0
    wall = 7
    for _ in range(h * w * 8):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if g[nr, nc] == wall:
            # Front hit (head-on): invert both axes
            dr = -dr
            dc = -dc
            bounces += 1
            continue
        side_hit = False
        if 0 <= r + dr < h and g[r + dr, c] == wall:
            dr = -dr
            bounces += 1
            side_hit = True
        if 0 <= c + dc < w and g[r, c + dc] == wall:
            dc = -dc
            bounces += 1
            side_hit = True
        if side_hit:
            continue
        r, c = nr, nc
    return bounces


def generate_input() -> np.ndarray:
    while True:
        n = int(_RNG.choice([12, 16, 20]))
        g = np.zeros((n, n), dtype=int)
        palette = [1, 2, 3, 4, 8]
        for r in range(n):
            for c in range(n):
                g[r, c] = int(_RNG.choice(palette))
        orient = int(_RNG.integers(0, 4))

        # place arrow L-shape
        if orient == 0:
            top = int(_RNG.integers(1, n - 3))
            left = int(_RNG.integers(1, n - 3))
            # missing top-left
            arrow = [(top, left + 1), (top + 1, left), (top + 1, left + 1)]
            start = (top, left)
            dr, dc = -1, -1
        elif orient == 1:
            top = int(_RNG.integers(1, n - 3))
            left = int(_RNG.integers(1, n - 3))
            # missing top-right
            arrow = [(top, left), (top + 1, left), (top + 1, left + 1)]
            start = (top, left + 1)
            dr, dc = -1, 1
        elif orient == 2:
            top = int(_RNG.integers(1, n - 3))
            left = int(_RNG.integers(1, n - 3))
            # missing bottom-left
            arrow = [(top, left), (top, left + 1), (top + 1, left + 1)]
            start = (top + 1, left)
            dr, dc = 1, -1
        else:
            top = int(_RNG.integers(1, n - 3))
            left = int(_RNG.integers(1, n - 3))
            # missing bottom-right
            arrow = [(top, left), (top, left + 1), (top + 1, left)]
            start = (top + 1, left + 1)
            dr, dc = 1, 1

        for rr, cc in arrow:
            g[rr, cc] = 5
        g[start[0], start[1]] = 5

        # put 1~3 walls to induce reflections
        candidates = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1) if g[r, c] != 5]
        _RNG.shuffle(candidates)
        wall_count = int(_RNG.integers(1, 4))
        for idx in range(min(wall_count, len(candidates))):
            wr, wc = candidates[idx]
            g[wr, wc] = 7

        b = _simulate(g, start[0], start[1], dr, dc)
        if b >= 1 and n % (b + 1) == 0:
            return g
