from __future__ import annotations

from functools import lru_cache
import math
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _bg_color(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(values))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shape = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    h, w = _shape_size(shape)
    return shape, h, w


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            cell_set = set(cells)
            shape, sh, sw = _normalize_shape(cell_set)
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "cells": tuple(sorted(cells)),
                    "area": len(cells),
                    "shape": shape,
                    "shape_h": sh,
                    "shape_w": sw,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


@lru_cache(maxsize=None)
def _placements(shape: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[frozenset[tuple[int, int]], ...]:
    sh, sw = _shape_size(shape)
    if sh > h or sw > w:
        return ()
    placements: list[frozenset[tuple[int, int]]] = []
    for tr in range(h - sh + 1):
        for tc in range(w - sw + 1):
            placements.append(frozenset((tr + r, tc + c) for r, c in shape))
    return tuple(placements)


@lru_cache(maxsize=None)
def _packing_dims_cached(
    shape_a: tuple[tuple[int, int], ...], shape_b: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    area = len(shape_a) + len(shape_b)
    dims: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        placements_a = _placements(shape_a, h, w)
        placements_b = _placements(shape_b, h, w)
        if not placements_a or not placements_b:
            continue
        for pa in placements_a:
            if any(pa.isdisjoint(pb) for pb in placements_b):
                dims.append((h, w))
                break
    return tuple(dims)


def _packing_dims(
    shape_a: tuple[tuple[int, int], ...], shape_b: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    if shape_b < shape_a:
        shape_a, shape_b = shape_b, shape_a
    return _packing_dims_cached(shape_a, shape_b)


def _infer_anchor_and_dims(components: list[dict]) -> tuple[tuple[int, int], tuple[int, int]]:
    assert len(components) == 3, "each color must have two fragments and one anchor"
    singleton_ids = [idx for idx, comp in enumerate(components) if comp["area"] == 1]
    assert singleton_ids, "missing single-pixel anchor"

    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for anchor_idx in singleton_ids:
        frag_ids = [idx for idx in range(3) if idx != anchor_idx]
        dims = _packing_dims(components[frag_ids[0]]["shape"], components[frag_ids[1]]["shape"])
        if len(dims) == 1:
            anchor = components[anchor_idx]["cells"][0]
            candidates.append((anchor, dims[0]))

    assert len(candidates) == 1, "anchor or rectangle size is ambiguous"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments by complementary shape and move the completed rectangle to the color-matched anchor.

    Input structure
    - Black background with several colors of scattered fragments.
    - For each foreground color there are exactly two fragments and one single-pixel anchor.
    - The two fragments can be translated, without rotation, to tile one solid rectangle.

    Rule
    - Group components by color.
    - For each color, infer the unique rectangle size that the two fragments can tile.
    - Paint that solid rectangle with its top-left corner at the anchor pixel.
    - Remove the original fragments; the anchor becomes covered by the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one fragment color"

    for color in colors:
        components = _find_components(g == color)
        anchor, (rect_h, rect_w) = _infer_anchor_and_dims(components)
        ar, ac = anchor
        assert ar + rect_h <= h and ac + rect_w <= w, "anchor rectangle exceeds grid bounds"
        patch = out[ar : ar + rect_h, ac : ac + rect_w]
        assert np.all(patch == bg), "completed rectangles must not overlap"
        out[ar : ar + rect_h, ac : ac + rect_w] = color

    return out


BG_COLOR = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 28
PAIR_COUNT_RANGE = (1, 4)
RECT_DIMS = tuple(
    (rh, rw)
    for rh in range(2, 6)
    for rw in range(2, 6)
    if 6 <= rh * rw <= 16 and max(rh, rw) >= 3
)
MIN_BORDER = 1
PARTITION_RETRIES = 250
GENERATION_RETRIES = 400


def _is_box(shape: tuple[tuple[int, int], ...]) -> bool:
    sh, sw = _shape_size(shape)
    return len(shape) == sh * sw


def _rect_cells(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _random_connected_subset(h: int, w: int, target: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    chosen = {start}
    frontier = set()
    for dr, dc in DIR4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(chosen) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for dr, dc in DIR4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                frontier.add((nr, nc))
    return chosen


def _make_fragment_pair(rect_h: int, rect_w: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] | None:
    all_cells = _rect_cells(rect_h, rect_w)
    area = rect_h * rect_w
    lo = max(2, area // 3)
    hi = min(area - 2, area - area // 3)
    for _ in range(PARTITION_RETRIES):
        target = random.randint(lo, hi)
        first = _random_connected_subset(rect_h, rect_w, target)
        second = all_cells - first
        if not _is_connected(first) or not _is_connected(second):
            continue
        shape_a, _, _ = _normalize_shape(first)
        shape_b, _, _ = _normalize_shape(second)
        if _is_box(shape_a) or _is_box(shape_b):
            continue
        if random.random() < 0.5:
            shape_a, shape_b = shape_b, shape_a
        if _packing_dims(shape_a, shape_b) == ((rect_h, rect_w),):
            return shape_a, shape_b
    return None


def _translate(shape: tuple[tuple[int, int], ...], top: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = top
    return [(tr + r, tc + c) for r, c in shape]


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _find_placement(
    shape: tuple[tuple[int, int], ...], blocked: np.ndarray, h: int, w: int
) -> tuple[int, int] | None:
    sh, sw = _shape_size(shape)
    row_lo = MIN_BORDER
    row_hi = h - sh - MIN_BORDER
    col_lo = MIN_BORDER
    col_hi = w - sw - MIN_BORDER
    if row_hi < row_lo or col_hi < col_lo:
        return None

    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(positions)
    for top in positions:
        cells = _translate(shape, top)
        if all(not blocked[r, c] for r, c in cells):
            return top
    return None


def _expected_output(specs: list[dict], h: int, w: int) -> np.ndarray:
    out = np.full((h, w), BG_COLOR, dtype=int)
    for spec in specs:
        ar, ac = spec["anchor"]
        rect_h, rect_w = spec["rect"]
        out[ar : ar + rect_h, ac : ac + rect_w] = spec["color"]
    return out


def _build_input(specs: list[dict], h: int, w: int) -> np.ndarray:
    inp = np.full((h, w), BG_COLOR, dtype=int)
    for spec in specs:
        ar, ac = spec["anchor"]
        inp[ar, ac] = spec["color"]
        for cells in spec["fragment_cells"]:
            for r, c in cells:
                inp[r, c] = spec["color"]
    return inp


def _choose_grid_shape(specs: list[dict]) -> tuple[int, int] | None:
    total_area = sum(spec["rect"][0] * spec["rect"][1] for spec in specs)
    max_h = max(spec["rect"][0] for spec in specs) + 2 * MIN_BORDER
    max_w = max(spec["rect"][1] for spec in specs) + 2 * MIN_BORDER
    side_floor = int(math.ceil(math.sqrt(total_area * 5.0)))
    lo = max(GRID_SIDE_MIN, max_h, max_w, side_floor)
    if lo > GRID_SIDE_MAX:
        return None
    return random.randint(lo, GRID_SIDE_MAX), random.randint(lo, GRID_SIDE_MAX)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, pair_count)

    specs: list[dict] = []
    for color in colors:
        rect_h, rect_w = random.choice(RECT_DIMS)
        fragments = _make_fragment_pair(rect_h, rect_w)
        if fragments is None:
            return None
        specs.append({"color": color, "rect": (rect_h, rect_w), "fragments": fragments, "fragment_cells": []})

    grid_shape = _choose_grid_shape(specs)
    if grid_shape is None:
        return None
    h, w = grid_shape
    blocked = np.zeros((h, w), dtype=bool)

    for spec in sorted(specs, key=lambda item: item["rect"][0] * item["rect"][1], reverse=True):
        rect_h, rect_w = spec["rect"]
        rect_shape = tuple((r, c) for r in range(rect_h) for c in range(rect_w))
        top = _find_placement(rect_shape, blocked, h, w)
        if top is None:
            return None
        spec["anchor"] = top
        _mark_blocked(blocked, _translate(rect_shape, top))

    fragment_jobs: list[tuple[int, float, dict, tuple[tuple[int, int], ...]]] = []
    for spec in specs:
        for shape in spec["fragments"]:
            fragment_jobs.append((len(shape), random.random(), spec, shape))
    fragment_jobs.sort(key=lambda item: (-item[0], item[1]))

    for _, _, spec, shape in fragment_jobs:
        top = _find_placement(shape, blocked, h, w)
        if top is None:
            return None
        cells = _translate(shape, top)
        spec["fragment_cells"].append(cells)
        _mark_blocked(blocked, cells)

    inp = _build_input(specs, h, w)
    expected = _expected_output(specs, h, w)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment input")
