from __future__ import annotations

import random

import numpy as np


BG = 0
SEGMENT_MIN_LEN = 2
SEGMENT_MAX_LEN = 4
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
MAX_GRID_SIZE = 30
MAX_GEN_RETRIES = 400
EMPTY_SLOT_SIZE = 4
SLOT_MARGIN = 1


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _segment_from_coords(coords: np.ndarray, color: int) -> dict:
    assert SEGMENT_MIN_LEN <= len(coords) <= SEGMENT_MAX_LEN
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        endpoints = ((r, cols[0]), (r, cols[-1]))
        orientation = "h"
    else:
        assert len(cols) == 1
        c = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1))
        endpoints = ((rows[0], c), (rows[-1], c))
        orientation = "v"
    return {
        "color": int(color),
        "coords": coords,
        "cells": {tuple(map(int, xy)) for xy in coords.tolist()},
        "orientation": orientation,
        "endpoints": endpoints,
    }


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    assert a[1] == b[1]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _build_candidate(
    grid: np.ndarray,
    start_seg: dict,
    target_seg: dict,
    start_ep: tuple[int, int],
    target_ep: tuple[int, int],
) -> dict | None:
    if start_seg["orientation"] == "h":
        if start_ep[0] == target_ep[0] or start_ep[1] == target_ep[1]:
            return None
        turn = (start_ep[0], target_ep[1])
        first_leg = abs(target_ep[1] - start_ep[1])
    else:
        if start_ep[0] == target_ep[0] or start_ep[1] == target_ep[1]:
            return None
        turn = (target_ep[0], start_ep[1])
        first_leg = abs(target_ep[0] - start_ep[0])

    path = _line_cells(start_ep, turn) + _line_cells(turn, target_ep)[1:]
    h, w = grid.shape
    for r, c in path:
        if not (0 <= r < h and 0 <= c < w):
            return None
    for r, c in path[1:-1]:
        if int(grid[r, c]) != BG:
            return None
    return {"path": tuple(path), "first_leg": int(first_leg)}


def _solve_color(grid: np.ndarray, color: int) -> tuple[tuple[int, int], ...]:
    comps = _components_of_color(grid, color)
    assert len(comps) == 2, f"color {color} must form exactly two segments"
    segs = [_segment_from_coords(comp, color) for comp in comps]

    candidates: dict[tuple[tuple[int, int], ...], dict] = {}
    for start_idx, target_idx in ((0, 1), (1, 0)):
        start_seg = segs[start_idx]
        target_seg = segs[target_idx]
        for start_ep in start_seg["endpoints"]:
            for target_ep in target_seg["endpoints"]:
                cand = _build_candidate(grid, start_seg, target_seg, start_ep, target_ep)
                if cand is not None:
                    candidates[cand["path"]] = cand

    assert candidates, f"color {color} has no legal L path"
    best_first_leg = max(cand["first_leg"] for cand in candidates.values())
    best_paths = [cand["path"] for cand in candidates.values() if cand["first_leg"] == best_first_leg]
    assert len(best_paths) == 1, f"color {color} must have a unique longest-first-leg path"
    return best_paths[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each color-matched pair of short segments with an L-shaped bridge whose first leg follows the start segment's orientation.

    Input structure:
    - The background is black.
    - Every non-black color appears as exactly two disjoint straight segments of length 2-4.
    - Each segment is axis-aligned and fully contiguous, with no extra pixels of that color elsewhere.

    Rule:
    - Pair the two segments of each color.
    - Enumerate the legal one-turn endpoint-to-endpoint bridges that extend outward from one segment, travel straight until aligned with an endpoint of the partner segment, then turn 90 degrees into that endpoint.
    - Keep the unique legal bridge whose first straight leg is longer than the competing ordered alternative; that first leg must be horizontal for a horizontal start segment and vertical for a vertical start segment.
    - Draw only the background cells along that bridge, leaving the original segments in place and never covering other colored pixels.

    Key insight:
    The segment orientation does not just describe the objects: it determines the direction of the bridge's first leg, and the chosen ordered bridge is the one that stays straight longer before turning.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = [int(c) for c in np.unique(g) if int(c) != BG]

    paths_by_color: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in colors:
        paths_by_color[color] = _solve_color(g, color)

    out = g.copy()
    painted: set[tuple[int, int]] = set()
    for color, path in paths_by_color.items():
        for r, c in path[1:-1]:
            assert int(out[r, c]) == BG, "bridges may only paint background cells"
            assert (r, c) not in painted, "bridges may not overlap each other"
            out[r, c] = int(color)
            painted.add((r, c))
    return out


def _maybe_transpose(arr: np.ndarray, do_it: bool) -> np.ndarray:
    return arr.T.copy() if do_it else arr.copy()


def _maybe_flipud(arr: np.ndarray, do_it: bool) -> np.ndarray:
    return np.flipud(arr).copy() if do_it else arr.copy()


def _maybe_fliplr(arr: np.ndarray, do_it: bool) -> np.ndarray:
    return np.fliplr(arr).copy() if do_it else arr.copy()


def _make_pair_arrays(color: int, start_horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    seg_a = random.randint(SEGMENT_MIN_LEN, SEGMENT_MAX_LEN)
    seg_b = random.randint(SEGMENT_MIN_LEN, SEGMENT_MAX_LEN)
    leg2 = random.randint(2, 4)
    leg1 = random.randint(leg2 + 1, 5)

    h = leg2 + seg_b
    w = seg_a + leg1
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    inp[0, :seg_a] = color
    inp[leg2 : leg2 + seg_b, w - 1] = color

    out[:] = inp
    out[0, seg_a - 1 : w] = color
    out[: leg2 + 1, w - 1] = color

    transpose = not start_horizontal
    flip_ud = bool(random.getrandbits(1))
    flip_lr = bool(random.getrandbits(1))

    inp = _maybe_transpose(inp, transpose)
    out = _maybe_transpose(out, transpose)
    inp = _maybe_flipud(inp, flip_ud)
    out = _maybe_flipud(out, flip_ud)
    inp = _maybe_fliplr(inp, flip_lr)
    out = _maybe_fliplr(out, flip_lr)
    return inp, out


def _slot_requirements(
    locals_by_slot: dict[int, tuple[np.ndarray, np.ndarray]]
) -> tuple[list[int], list[int]]:
    row_heights = [EMPTY_SLOT_SIZE, EMPTY_SLOT_SIZE]
    col_widths = [EMPTY_SLOT_SIZE, EMPTY_SLOT_SIZE]
    for slot, (inp, _out) in locals_by_slot.items():
        r = slot // 2
        c = slot % 2
        h, w = inp.shape
        row_heights[r] = max(row_heights[r], h + 2 * SLOT_MARGIN)
        col_widths[c] = max(col_widths[c], w + 2 * SLOT_MARGIN)
    return row_heights, col_widths


def _add_slack(base0: int, base1: int) -> tuple[int, int]:
    slack = random.randint(0, MAX_GRID_SIZE - (base0 + base1))
    extra0 = random.randint(0, slack)
    extra1 = slack - extra0
    return base0 + extra0, base1 + extra1


def _place_local(canvas: np.ndarray, local: np.ndarray, top: int, left: int) -> None:
    h, w = local.shape
    mask = local != BG
    window = canvas[top : top + h, left : left + w]
    assert not np.any(window[mask] != BG)
    window[mask] = local[mask]
    canvas[top : top + h, left : left + w] = window


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    colors = random.sample(list(range(1, 10)), pair_count)

    start_horizontal_flags = [True, False]
    while len(start_horizontal_flags) < pair_count:
        start_horizontal_flags.append(bool(random.getrandbits(1)))
    random.shuffle(start_horizontal_flags)

    slots = random.sample(range(4), pair_count)
    locals_by_slot: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for slot, color, start_horizontal in zip(slots, colors, start_horizontal_flags):
        locals_by_slot[slot] = _make_pair_arrays(color, start_horizontal)

    row_heights, col_widths = _slot_requirements(locals_by_slot)
    if sum(row_heights) > MAX_GRID_SIZE or sum(col_widths) > MAX_GRID_SIZE:
        return None

    row_heights[0], row_heights[1] = _add_slack(row_heights[0], row_heights[1])
    col_widths[0], col_widths[1] = _add_slack(col_widths[0], col_widths[1])
    h, w = sum(row_heights), sum(col_widths)
    if h > MAX_GRID_SIZE or w > MAX_GRID_SIZE:
        return None

    row_starts = [0, row_heights[0]]
    col_starts = [0, col_widths[0]]

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    for slot, (local_inp, local_out) in locals_by_slot.items():
        slot_r = slot // 2
        slot_c = slot % 2
        slot_top = row_starts[slot_r]
        slot_left = col_starts[slot_c]
        slot_h = row_heights[slot_r]
        slot_w = col_widths[slot_c]
        local_h, local_w = local_inp.shape

        if slot_h < local_h + 2 * SLOT_MARGIN or slot_w < local_w + 2 * SLOT_MARGIN:
            return None

        top = slot_top + random.randint(SLOT_MARGIN, slot_h - local_h - SLOT_MARGIN)
        left = slot_left + random.randint(SLOT_MARGIN, slot_w - local_w - SLOT_MARGIN)
        _place_local(inp, local_inp, top, left)
        _place_local(out, local_out, top, left)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
