from __future__ import annotations

import numpy as np

from collections import deque

BLACK = 0
GREY = 5
MAROON = 9
ORANGE = 7
TEAL = 8


def _neighbors4(r: int, c: int, H: int, W: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < H:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < W:
        yield r, c + 1


def _components(mask: np.ndarray):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_gray_frame(grid: np.ndarray):
    H, W = grid.shape
    comps = _components(grid == GREY)
    best = None
    best_size = -1
    for comp in comps:
        if len(comp) < 12:
            continue
        r0, c0 = comp[:, 0].min(), comp[:, 1].min()
        r1, c1 = comp[:, 0].max(), comp[:, 1].max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        perimeter = 2 * (h + w) - 4
        if len(comp) != perimeter:
            continue
        ok = True
        for rr in range(r0, r1 + 1):
            if grid[rr, c0] != GREY or grid[rr, c1] != GREY:
                ok = False
                break
        if not ok:
            continue
        for cc in range(c0, c1 + 1):
            if grid[r0, cc] != GREY or grid[r1, cc] != GREY:
                ok = False
                break
        if not ok:
            continue
        if len(comp) > best_size:
            best_size = len(comp)
            best = (r0, c0, r1, c1)
    return best


def _fill_components(grid: np.ndarray):
    comps = _components(grid != GREY)
    infos = []
    for comp in comps:
        areas = len(comp)
        infos.append((areas, comp))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    H, W = out.shape

    # Step1: recover maroon by reflection partners
    maroons = np.argwhere(out == MAROON)
    for r, c in maroons:
        r = int(r)
        c = int(c)
        candidates = [(H - 1 - r, c), (r, W - 1 - c), (H - 1 - r, W - 1 - c)]
        repl = None
        for rr, cc in candidates:
            if 0 <= rr < H and 0 <= cc < W and out[rr, cc] != MAROON:
                repl = int(out[rr, cc])
                break
        if repl is not None:
            out[r, c] = repl

    # Step2: flood fill on non-gray regions (black + color pixels)
    infos = _fill_components(out)
    if not infos:
        return out
    sizes = [a for a, _ in infos]
    min_size = min(sizes)
    max_size = max(sizes)
    uniq_min = sizes.count(min_size) == 1
    uniq_max = sizes.count(max_size) == 1

    for a, comp in infos:
        if a == min_size and uniq_min:
            out[comp[:, 0], comp[:, 1]] = ORANGE
        elif a == max_size and uniq_max:
            out[comp[:, 0], comp[:, 1]] = TEAL
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    out = np.zeros((11, 11), dtype=int)

    # Gray symmetric wall skeleton
    for r in range(2, 9):
        out[r, 2] = GREY
        out[r, 8] = GREY
    for c in range(2, 9):
        out[2, c] = GREY
        out[8, c] = GREY
    for r in range(4, 7):
        out[r, 4] = GREY
        out[r, 6] = GREY

    # Symmetric blue/red interior pattern around center (5,5)
    palette = [1, 2, 3, 6]
    for _ in range(18):
        r = int(rng.integers(3, 8))
        c = int(rng.integers(3, 8))
        color = int(rng.choice(palette))
        rs = [r, 10 - r]
        cs = [c, 10 - c]
        for rr in rs:
            for cc in cs:
                out[rr, cc] = color

    # Add maroon occlusions at random unique locations
    all_cells = [(r, c) for r in range(11) for c in range(11) if out[r, c] != MAROON]
    rng.shuffle(all_cells)
    for r, c in all_cells[:6]:
        if out[r, c] == MAROON:
            continue
        out[r, c] = MAROON

    return out

