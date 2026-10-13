from __future__ import annotations

from collections import deque
import random

import numpy as np

BG = 0
GRAY = 5
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_separator_row(g: np.ndarray) -> int:
    H, W = map(int, g.shape)
    for r in range(H):
        if bool(np.all(g[r, :] == GRAY)):
            return r
    raise ValueError("no full gray separator row found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compute intersection and symmetric-difference maps from two panel masks and cast diagonal rays from intersection pixels.

    Input structure:
    - Black background.
    - Odd-height grid with a full gray row splitting top and bottom panels.
    - Top panel contains blue foreground pixels, bottom panel contains red foreground pixels on black.
    - Foreground cells are only those two colors.

    Rule:
    - Remove the separator row and compare panels position-wise.
    - Cells foreground in both panels become green.
    - Cells foreground in exactly one panel become yellow.
    - From each green cell, cast diagonal rays in four diagonal directions until blocked.
    - Rays are blocked (without painting the blocking cell) by yellow pixels, by already-painted green pixels, or grid boundary.

    Key insight:
    - The separator is not only a split of input data but a structural prior that turns two masks into three-valued seed pixels (none/intersection/diff), which then drive directional ray growth.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H >= 3 and H % 2 == 1
    sep = _find_separator_row(g)
    assert sep == H // 2

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape

    top_fg = top != BG
    bottom_fg = bottom != BG

    out = np.zeros_like(top, dtype=int)
    both = top_fg & bottom_fg
    one = top_fg ^ bottom_fg
    out[both] = GREEN
    out[one] = YELLOW

    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    sources = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1]) if out[r, c] == GREEN]
    for r0, c0 in sources:
        for dr, dc in dirs:
            r, c = r0 + dr, c0 + dc
            while _in_bounds(out.shape, r, c):
                cell = int(out[r, c])
                if cell in {GREEN, YELLOW}:
                    break
                out[r, c] = GREEN
                r += dr
                c += dc

    return out


def _sample_non_adjacent_cells(limit: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count <= 0:
        return []

    cells = []
    pool = list(limit)
    random.shuffle(pool)
    while pool and len(cells) < count:
        r, c = pool.pop()
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells):
            continue
        cells.append((r, c))
    return cells


def _random_panel_mask(H: int, W: int, total: int) -> set[tuple[int, int]]:
    choices = [(r, c) for r in range(H) for c in range(W)]
    random.shuffle(choices)
    selected: set[tuple[int, int]] = set()

    for r, c in choices:
        if len(selected) >= total:
            break
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in selected):
            continue
        selected.add((r, c))
    return selected


def generate_input() -> np.ndarray:
    for _ in range(1600):
        panel_h = random.randint(3, 6)
        W = random.randint(11, 15)
        H = 2 * panel_h + 1
        g = np.full((H, W), BG, dtype=int)

        sep = panel_h
        g[sep, :] = GRAY

        total_top = random.randint(5, 10)
        total_bottom = random.randint(5, 10)
        inter_count = random.randint(2, min(4, total_top, total_bottom))

        all_pos = [(r, c) for r in range(panel_h) for c in range(W)]
        random.shuffle(all_pos)

        intersection = _sample_non_adjacent_cells(all_pos, inter_count)
        remaining = [p for p in all_pos if p not in set(intersection)]
        if len(remaining) < total_top + total_bottom - 2 * inter_count:
            continue

        top_only_needed = total_top - inter_count
        bottom_only_needed = total_bottom - inter_count
        if top_only_needed + bottom_only_needed < 3:
            bottom_only_needed = 3 - top_only_needed
            if bottom_only_needed < 0:
                top_only_needed = 3
                bottom_only_needed = 0

        random.shuffle(remaining)
        top_only = remaining[:top_only_needed]
        bottom_only = remaining[top_only_needed : top_only_needed + bottom_only_needed]

        top_set = set(intersection) | set(top_only)
        bottom_set = set(intersection) | set(bottom_only)
        if len(top_set) == 0 or len(bottom_set) == 0:
            continue
        if len(top_set) < 5 or len(bottom_set) < 5:
            continue

        for r, c in top_set:
            g[r, c] = BLUE
        for r, c in bottom_set:
            br, bc = panel_h + 1 + r, c
            if g[br, bc] == BLUE:
                continue
            g[br, bc] = RED

        try:
            _ = main(g)
        except Exception:
            continue
        return g

    raise RuntimeError("failed to generate valid puzzle 2 input")

