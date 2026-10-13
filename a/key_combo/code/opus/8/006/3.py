from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2


def _neighbor_dirs(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            out.append((dr, dc))
    return out


def _top_run_end(mask: np.ndarray, r: int, c: int) -> int:
    h = mask.shape[0]
    rr = r
    while rr + 1 < h and mask[rr + 1, c]:
        rr += 1
    return rr


def _bottom_run_start(mask: np.ndarray, r: int, c: int) -> int:
    rr = r
    while rr - 1 >= 0 and mask[rr - 1, c]:
        rr -= 1
    return rr


def _left_run_end(mask: np.ndarray, r: int, c: int) -> int:
    w = mask.shape[1]
    cc = c
    while cc + 1 < w and mask[r, cc + 1]:
        cc += 1
    return cc


def _right_run_start(mask: np.ndarray, r: int, c: int) -> int:
    cc = c
    while cc - 1 >= 0 and mask[r, cc - 1]:
        cc -= 1
    return cc


def _stretch_top(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    rows = np.where(mask.any(axis=1))[0]
    r = int(rows.min())
    if r == 0:
        return mask
    cols = np.where(mask[r])[0]
    support = {int(c) for c in cols if r + 1 < mask.shape[0] and mask[r + 1, c]}
    out = mask.copy()
    out[r, cols] = False
    for c in cols:
        out[0, c] = True
    for c in support:
        end = _top_run_end(mask, r, c)
        out[0 : end + 1, c] = True
    return out


def _stretch_bottom(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    rows = np.where(mask.any(axis=1))[0]
    r = int(rows.max())
    last = mask.shape[0] - 1
    if r == last:
        return mask
    cols = np.where(mask[r])[0]
    support = {int(c) for c in cols if r - 1 >= 0 and mask[r - 1, c]}
    out = mask.copy()
    out[r, cols] = False
    for c in cols:
        out[last, c] = True
    for c in support:
        start = _bottom_run_start(mask, r, c)
        out[start : last + 1, c] = True
    return out


def _stretch_left(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    cols = np.where(mask.any(axis=0))[0]
    c = int(cols.min())
    if c == 0:
        return mask
    rows = np.where(mask[:, c])[0]
    support = {int(r) for r in rows if c + 1 < mask.shape[1] and mask[r, c + 1]}
    out = mask.copy()
    out[rows, c] = False
    for r in rows:
        out[r, 0] = True
    for r in support:
        end = _left_run_end(mask, r, c)
        out[r, 0 : end + 1] = True
    return out


def _stretch_right(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    cols = np.where(mask.any(axis=0))[0]
    c = int(cols.max())
    last = mask.shape[1] - 1
    if c == last:
        return mask
    rows = np.where(mask[:, c])[0]
    support = {int(r) for r in rows if c - 1 >= 0 and mask[r, c - 1]}
    out = mask.copy()
    out[rows, c] = False
    for r in rows:
        out[r, last] = True
    for r in support:
        start = _right_run_start(mask, r, c)
        out[r, start : last + 1] = True
    return out


def _endpoint_info(mask: np.ndarray) -> list[tuple[int, int, tuple[int, int]]]:
    pts: list[tuple[int, int, tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        dirs = _neighbor_dirs(mask, int(r), int(c))
        if len(dirs) != 1:
            continue
        dr, dc = dirs[0]
        pts.append((int(r), int(c), (-dr, -dc)))
    return pts


def _fill_short_gaps(mask: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    out = mask.copy()
    changed = True
    while changed:
        changed = False
        endpoints = _endpoint_info(out)
        row_map: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        col_map: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        for r, c, facing in endpoints:
            row_map.setdefault(r, []).append((c, facing))
            col_map.setdefault(c, []).append((r, facing))

        for r, items in row_map.items():
            items.sort()
            for i in range(len(items) - 1):
                c1, f1 = items[i]
                c2, f2 = items[i + 1]
                gap = c2 - c1 - 1
                if f1 != (0, 1) or f2 != (0, -1) or not (1 <= gap <= 3):
                    continue
                if blocked[r, c1 + 1 : c2].any() or out[r, c1 + 1 : c2].any():
                    continue
                out[r, c1 + 1 : c2] = True
                changed = True

        for c, items in col_map.items():
            items.sort()
            for i in range(len(items) - 1):
                r1, f1 = items[i]
                r2, f2 = items[i + 1]
                gap = r2 - r1 - 1
                if f1 != (1, 0) or f2 != (-1, 0) or not (1 <= gap <= 3):
                    continue
                if blocked[r1 + 1 : r2, c].any() or out[r1 + 1 : r2, c].any():
                    continue
                out[r1 + 1 : r2, c] = True
                changed = True

    return out


def _draw_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _add_extra_bars(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))
    random.shuffle(inner_rows)
    random.shuffle(inner_cols)

    if inner_rows and random.random() < 0.75:
        mask[inner_rows[0], c0 : c1 + 1] = True
    if inner_cols and random.random() < 0.75:
        mask[r0 : r1 + 1, inner_cols[0]] = True
    if len(inner_rows) >= 2 and random.random() < 0.35:
        mask[inner_rows[1], c0 : c1 + 1] = True
    if len(inner_cols) >= 2 and random.random() < 0.35:
        mask[r0 : r1 + 1, inner_cols[1]] = True


def _add_gap_fill_hooks(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r1 - r0 >= 4 and c1 - c0 >= 4 and random.random() < 0.6:
        mid_c = random.randint(c0 + 1, c1 - 1)
        mask[r0 + 1 : r1, mid_c] = True
        if mid_c - c0 >= 2:
            cut_col = random.randint(c0 + 1, mid_c - 1)
            mask[r0 + 1, cut_col] = True
        if c1 - mid_c >= 2:
            cut_col = random.randint(mid_c + 1, c1 - 1)
            mask[r1 - 1, cut_col] = True


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Compress red coarse-block presence into edge stretch instructions, stretch the central blue rectilinear drawing, then repair short new gaps.

    Input structure:
    - The grid is square, divisible into a 3x3 array of equal coarse blocks, with black background.
    - Sparse red pixels may appear in any coarse block; only the four edge-middle blocks act as direction markers.
    - Blue pixels form a connected one-cell-thick rectilinear line drawing entirely inside the center coarse block.

    Rule:
    - Read each coarse block as marked iff it contains at least one red pixel.
    - Use marks at top, bottom, left, and right edge-middle blocks as instructions to stretch the outermost blue side in that direction to the corresponding grid edge.
    - When a side is moved outward, keep its coordinate span and extend the perpendicular supporting segments so the moved side stays attached to the drawing.
    - After all requested stretches, fill any horizontal or vertical gap of length at most three between two blue endpoints that face each other.
    - Preserve the original red pixels.

    Key insight:
    - The red pattern is not output itself; edge-middle red occupancy is a coarse spatial code that says which outer blue sides to push to the canvas edges.
    """
    grid = np.asarray(input_grid, dtype=int)
    h, w = grid.shape
    assert grid.ndim == 2 and h == w and h % 3 == 0

    k = h // 3
    blue = grid == BLUE
    blocked = grid == RED

    def marked(br: int, bc: int) -> bool:
        return bool(np.any(grid[br * k : (br + 1) * k, bc * k : (bc + 1) * k] == RED))

    if marked(0, 1):
        blue = _stretch_top(blue)
    if marked(2, 1):
        blue = _stretch_bottom(blue)
    if marked(1, 0):
        blue = _stretch_left(blue)
    if marked(1, 2):
        blue = _stretch_right(blue)

    blue = _fill_short_gaps(blue, blocked)

    out = grid.copy()
    out[out == BLUE] = BG
    out[blue] = BLUE
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        k = random.randint(4, 6)
        size = 3 * k
        grid = np.zeros((size, size), dtype=int)

        local = np.zeros((k, k), dtype=bool)
        r0 = random.randint(1, max(1, k - 3))
        c0 = random.randint(1, max(1, k - 3))
        r1 = random.randint(r0 + 1, k - 2)
        c1 = random.randint(c0 + 1, k - 2)

        _draw_rect(local, r0, c0, r1, c1)
        _add_extra_bars(local, r0, c0, r1, c1)
        _add_gap_fill_hooks(local, r0, c0, r1, c1)

        grid[k : 2 * k, k : 2 * k][local] = BLUE

        directions = random.sample(["top", "bottom", "left", "right"], random.randint(2, 4))
        block_map = {"top": (0, 1), "bottom": (2, 1), "left": (1, 0), "right": (1, 2)}
        used_red: set[tuple[int, int]] = set()

        def place_red(br: int, bc: int, count: int) -> None:
            choices = [
                (r, c)
                for r in range(br * k, (br + 1) * k)
                for c in range(bc * k, (bc + 1) * k)
                if (r, c) not in used_red and grid[r, c] != BLUE
            ]
            random.shuffle(choices)
            assert len(choices) >= count
            for r, c in choices[:count]:
                grid[r, c] = RED
                used_red.add((r, c))

        for name in directions:
            br, bc = block_map[name]
            place_red(br, bc, random.randint(1, 2))

        noise_blocks = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
        random.shuffle(noise_blocks)
        for br, bc in noise_blocks[: random.randint(0, 2)]:
            place_red(br, bc, 1)

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if np.count_nonzero(out == BLUE) <= np.count_nonzero(grid == BLUE):
            continue
        return grid

    raise RuntimeError("generate_input failed")
