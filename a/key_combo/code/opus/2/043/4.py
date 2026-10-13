import random
from collections import deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _invert_binary(tile: np.ndarray) -> np.ndarray:
    out = tile.copy()
    out[tile == BLUE] = YELLOW
    out[tile == YELLOW] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape

    # detect gray rectangle frame
    frame_mask = grid == GREY
    comps = _components(frame_mask)
    if not comps:
        return grid
    comp = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return grid

    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    tile = grid[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1].copy()
    t_inv = _invert_binary(tile)

    outside_bbox = np.zeros((h, w), dtype=bool)
    outside_bbox[: , :] = True
    outside_bbox[r0 : r1 + 1, c0 : c1 + 1] = False
    region_mask = (grid == BLACK) & outside_bbox & (grid != RED)
    comps_black = _components(region_mask)
    if not comps_black:
        return grid

    # anchor region is one containing bottom-left
    anchor_idx = 0
    bl = (h - 1, 0)
    for i, compi in enumerate(comps_black):
        if bl in compi:
            anchor_idx = i
            break
    anchor_tile = t_inv
    other_tile = tile

    out = grid.copy()
    for i, compi in enumerate(comps_black):
        sub = anchor_tile if i == anchor_idx else other_tile
    for i, compi in enumerate(comps_black):
        if not compi:
            continue
        rs = min(p[0] for p in compi)
        cs = min(p[1] for p in compi)
        rh, rw = sub.shape
        tile_use = anchor_tile if i == anchor_idx else other_tile
        rh, rw = tile_use.shape
        for r, c in compi:
            vr = (r - rs) % rh
            vc = (c - cs) % rw
            out[r, c] = tile_use[vr, vc]
    return out


def generate_input() -> np.ndarray:
    h = 15
    w = 15
    while True:
        g = np.full((h, w), BLACK, dtype=int)
        ih = random.randint(3, 4)
        iw = random.randint(3, 4)
        fw = iw + 2
        fh = ih + 2
        r0 = random.randint(1, h - fh - 2)
        c0 = random.randint(1, w - fw - 2)
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        g[r0, c0 : c1 + 1] = GREY
        g[r1, c0 : c1 + 1] = GREY
        g[r0 : r1 + 1, c0] = GREY
        g[r0 : r1 + 1, c1] = GREY

        need_blue = 2
        need_yellow = 2
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                g[rr, cc] = random.choice([BLUE, YELLOW])
                need_blue -= 1 if g[rr, cc] == BLUE else 0
                need_yellow -= 1 if g[rr, cc] == YELLOW else 0
        # ensure both colors present
        if not (need_blue <= 0 and need_yellow <= 0):
            continue

        if random.random() < 0.5:
            cwall = random.randint(1, c0 - 2) if c0 > 2 else random.randint(c1 + 3, w - 2)
            if cwall < w:
                g[:, cwall] = RED
        else:
            rwall = random.randint(1, r0 - 2) if r0 > 2 else random.randint(r1 + 3, h - 2)
            if rwall < h:
                g[rwall, :] = RED

        return g


_generate_input_single = generate_input


def generate_inputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    attempts = 0
    while len(outputs) < 20 and attempts < 200:
        attempts += 1
        g = _generate_input_single()
        arr = np.array(g, dtype=int, copy=True)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] > 30 or arr.shape[1] > 30:
            continue
        outputs.append(arr)

    while len(outputs) < 20:
        if outputs:
            outputs.append(outputs[-1].copy())
        else:
            outputs.append(np.zeros((1, 1), dtype=int))
    return outputs


    random.seed(0)
    print(main(generate_input()).shape)
