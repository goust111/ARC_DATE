from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np

BG = 0
MAIN = 1
MARKER = 2
EDGE = 5

GEN_RETRIES = 400
PATCH_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _edge_direction(grid: np.ndarray) -> tuple[int, int]:
    edges: list[tuple[int, int]] = []
    if np.all(grid[0, :] == EDGE):
        edges.append((-1, 0))
    if np.all(grid[-1, :] == EDGE):
        edges.append((1, 0))
    if np.all(grid[:, 0] == EDGE):
        edges.append((0, -1))
    if np.all(grid[:, -1] == EDGE):
        edges.append((0, 1))
    assert len(edges) == 1, f"expected exactly one full grey edge, got {edges}"
    return edges[0]


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if _in_bounds(mask.shape, rr, cc) and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(np.count_nonzero(mask))


def _classify_markers(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    repaired: list[tuple[int, int]] = []
    isolated: list[tuple[int, int]] = []
    for r0, c0 in np.argwhere(grid == MARKER):
        r = int(r0)
        c = int(c0)
        blue_neighbors = sum(
            1
            for rr, cc in _neighbors4(r, c)
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == MAIN
        )
        if blue_neighbors >= 3:
            repaired.append((r, c))
        else:
            isolated.append((r, c))
    return repaired, isolated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded red defects, then extend isolated red rays toward the edge until a repaired scar or the blue body blocks them.

    Input structure:
    - A black background grid contains one full grey edge line on exactly one border.
    - One irregular blue connected object sits away from that grey border.
    - Red pixels are split between embedded defects inside the blue object and isolated background markers.

    Rule:
    - Any red pixel with at least three blue 4-neighbors is an embedded defect and is repainted blue; its coordinate is remembered as a repaired scar.
    - The grey border gives the ray direction: top means up, bottom means down, left means left, right means right.
    - Each isolated red marker stays red and paints a straight red ray through background cells in that direction, stopping before the first repaired scar, blue object pixel, or grid boundary.

    Key insight:
    - The repair step does more than recolor defects: those repaired coordinates become hidden blocking points that determine each ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    dr, dc = _edge_direction(g)
    repaired, isolated = _classify_markers(g)

    out = g.copy()
    repaired_set = set(repaired)
    for r, c in repaired:
        out[r, c] = MAIN

    h, w = out.shape
    for r, c in isolated:
        rr = r + dr
        cc = c + dc
        while 0 <= rr < h and 0 <= cc < w:
            if (rr, cc) in repaired_set:
                break
            if int(out[rr, cc]) == MAIN:
                break
            if int(out[rr, cc]) == BG:
                out[rr, cc] = MARKER
            rr += dr
            cc += dc

    return out


def _repair_column_choices(start: int, width: int, count: int) -> list[tuple[int, ...]]:
    interior = list(range(start + 1, start + width - 1))
    return [
        combo
        for combo in itertools.combinations(interior, count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]


def _make_patch(repair_count: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(PATCH_RETRIES):
        patch_h = random.randint(5, 7)
        bottom_w = random.randint(7, 8) if repair_count == 3 else random.randint(5, 7)
        patch_w = bottom_w + random.randint(1, 3)

        widths = [0] * patch_h
        widths[-1] = bottom_w
        widths[-2] = bottom_w

        narrowed = False
        for r in range(patch_h - 3, -1, -1):
            max_drop = min(2, widths[r + 1] - 3)
            drop = random.randint(0, max_drop)
            widths[r] = widths[r + 1] - drop
            narrowed = narrowed or drop > 0
        if widths[0] > bottom_w - 1:
            continue

        lefts = [0] * patch_h
        lefts[-1] = random.randint(0, patch_w - bottom_w)
        lefts[-2] = lefts[-1]

        shifted = False
        for r in range(patch_h - 3, -1, -1):
            width = widths[r]
            below_left = lefts[r + 1]
            below_right = below_left + widths[r + 1] - 1
            lo = max(0, below_left - (width - 2))
            hi = min(patch_w - width, below_right - 1)
            options = [x for x in range(lo, hi + 1) if abs(x - below_left) <= 1]
            if not options:
                options = list(range(lo, hi + 1))
            if not options:
                break
            lefts[r] = random.choice(options)
            shifted = shifted or lefts[r] != below_left
        else:
            patch = np.zeros((patch_h, patch_w), dtype=int)
            for r, (left, width) in enumerate(zip(lefts, widths)):
                patch[r, left : left + width] = MAIN

            area = int(np.count_nonzero(patch == MAIN))
            if not (15 <= area <= 30):
                continue
            if not _connected(patch == MAIN):
                continue
            if not (narrowed or shifted):
                continue

            combos = _repair_column_choices(lefts[-1], bottom_w, repair_count)
            combos = [
                combo
                for combo in combos
                if all(patch[-2, c] == MAIN for c in combo)
            ]
            if not combos:
                continue

            repair_cols = list(random.choice(combos))
            repairs = [(patch_h - 1, c) for c in repair_cols]
            return patch, repairs
    return None


def _stamp(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _trace_stop(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    repaired: set[tuple[int, int]],
) -> tuple[str, tuple[int, int] | None]:
    dr, dc = direction
    r, c = start
    rr, cc = r + dr, c + dc
    while _in_bounds(grid.shape, rr, cc):
        if (rr, cc) in repaired:
            return "repair", (rr, cc)
        if int(grid[rr, cc]) == MAIN:
            return "main", (rr, cc)
        rr += dr
        cc += dc
    return "boundary", None


def _valid_input(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.array_equal(main(inp), out):
        return False

    repaired, isolated = _classify_markers(inp)
    if len(repaired) not in (2, 3):
        return False
    if len(isolated) not in (2, 3):
        return False

    dr, dc = _edge_direction(inp)
    repaired_set = set(repaired)
    traces = set()
    for cell in isolated:
        line_key = cell[1] if dr != 0 else cell[0]
        if line_key in traces:
            return False
        traces.add(line_key)

        stop_kind, stop_cell = _trace_stop(inp, cell, (dr, dc), repaired_set)
        if stop_kind != "repair" or stop_cell is None:
            return False

        r, c = cell
        if any(int(inp[rr, cc]) == MAIN for rr, cc in _neighbors4(r, c) if _in_bounds(inp.shape, rr, cc)):
            return False

    main_mask = out == MAIN
    if int(np.count_nonzero(main_mask)) < 15 or int(np.count_nonzero(main_mask)) > 30:
        return False
    return _connected(main_mask)


def _build_canonical() -> tuple[np.ndarray, np.ndarray] | None:
    repair_count = random.randint(2, 3)
    made = _make_patch(repair_count)
    if made is None:
        return None
    patch, local_repairs = made
    patch_h, patch_w = patch.shape

    min_h = max(13, patch_h + 5)
    min_w = max(13, patch_w + 4)
    if min_h > 18 or min_w > 18:
        return None

    grid_h = random.randint(min_h, 18)
    grid_w = random.randint(min_w, 18)

    top = random.randint(2, grid_h - patch_h - 3)
    left = random.randint(2, grid_w - patch_w - 2)

    out = np.zeros((grid_h, grid_w), dtype=int)
    out[0, :] = EDGE
    _stamp(out, patch, top, left)

    repairs = [(top + r, left + c) for r, c in local_repairs]
    inp = out.copy()
    for r, c in repairs:
        inp[r, c] = MARKER

    isolated_count = random.randint(2, repair_count)
    chosen_repairs = random.sample(repairs, isolated_count)
    for r, c in chosen_repairs:
        max_gap = grid_h - 2 - r
        if max_gap < 2:
            return None
        gap = random.randint(2, min(5, max_gap))
        inp[r + gap, c] = MARKER

    return inp, main(inp)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_canonical()
        if built is None:
            continue
        inp, out = built
        turns = random.randint(0, 3)
        inp = np.rot90(inp, k=turns)
        out = np.rot90(out, k=turns)
        if _valid_input(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
