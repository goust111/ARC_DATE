from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
MARKER = 6

GRID_SIZES = (12, 14, 16, 18)
KEEP_COUNT_RANGE = (2, 4)
TOTAL_COUNT_RANGE = (4, 8)
OBJECT_SIZE_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _half_bbox(shape: tuple[int, int], mode: str, side: int) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    if mode == "lr":
        half = w // 2
        return 0, side * half, h - 1, (side + 1) * half - 1
    half = h // 2
    return side * half, 0, (side + 1) * half - 1, w - 1


def _find_kept_half(grid: np.ndarray) -> tuple[str, int, tuple[int, int, int, int]]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape == (2, 2), "expected exactly two marker pixels"
    (r0, c0), (r1, c1) = [tuple(map(int, coord)) for coord in coords]
    h, w = map(int, grid.shape)
    if r0 == r1:
        assert w % 2 == 0, "left/right split requires even width"
        half = w // 2
        left = c0 < half and c1 < half
        right = c0 >= half and c1 >= half
        assert left ^ right, "row-aligned markers must lie in one horizontal half"
        side = 0 if left else 1
        return "lr", side, _half_bbox(grid.shape, "lr", side)
    if c0 == c1:
        assert h % 2 == 0, "top/bottom split requires even height"
        half = h // 2
        top = r0 < half and r1 < half
        bottom = r0 >= half and r1 >= half
        assert top ^ bottom, "column-aligned markers must lie in one vertical half"
        side = 0 if top else 1
        return "tb", side, _half_bbox(grid.shape, "tb", side)
    raise AssertionError("markers must share one row or one column")


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "this task uses black background"

    _mode, _side, (r0, c0, r1, c1) = _find_kept_half(g)
    out = np.full_like(g, BG)
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    out[out == MARKER] = BG

    comps = components_mask(out == BLUE)
    assert comps, "kept half must contain blue components"
    step = len(comps)
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    for comp in comps:
        centroid_r = _round_half_up(comp[:, 0].mean())
        centroid_c = _round_half_up(comp[:, 1].mean())
        dr = _sign(centroid_r - center_r)
        dc = _sign(centroid_c - center_c)
        for r, c in comp:
            nr = int(r + step * dr)
            nc = int(c + step * dc)
            if in_bounds(g.shape, nr, nc):
                out[nr, nc] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the marker-selected half, count its blue components, and paint green diagonal copies shifted by that count.

    Input structure:
    - The grid has black background, blue 4-connected foreground objects, and exactly two magenta marker pixels.
    - The two marker pixels share one row or one column and both lie in the same left/right or top/bottom half.
    - Blue objects may appear in both halves, but the markers themselves are not part of any object.

    Rule:
    - Use the aligned marker pair to choose the kept half: same-row markers select the left or right half, and same-column markers select the top or bottom half.
    - Erase the markers, clear the discarded half to black, and count the blue 4-connected components in the kept half; let that count be N.
    - For each kept component, round its centroid to the nearest cell, take the sign of its row/column offset from the kept-half center, and paint a green copy shifted by N cells along that signed row/column direction while keeping the original blue pixels.

    Key insight:
    - The same kept-half components serve twice: first as the objects being counted, then as the objects whose shared count determines every copy's diagonal shift distance.
    """
    return _transform(grid)


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _random_blob(height: int, width: int, size: int, rng: random.Random) -> set[tuple[int, int]]:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return set()
        cells.add(rng.choice(frontier))

    arr = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    normalized = {(int(r - r0), int(c - c0)) for r, c in cells}
    if len(normalized) != size:
        return set()
    return normalized


def _axis_band(low: int, high: int, center: int, step: int, delta: int, limit: int) -> tuple[int, int] | None:
    if delta < 0:
        lo = max(low, step)
        hi = min(high, center - 1)
    else:
        lo = max(low, center + 1)
        hi = min(high, limit - 1 - step)
    if lo > hi:
        return None
    return int(lo), int(hi)


def _feasible_dirs(shape: tuple[int, int], bbox: tuple[int, int, int, int], step: int) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    row_bands = {dr: _axis_band(r0, r1, center_r, step, dr, h) for dr in (-1, 1)}
    col_bands = {dc: _axis_band(c0, c1, center_c, step, dc, w) for dc in (-1, 1)}
    dirs: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for dr, rband in row_bands.items():
        for dc, cband in col_bands.items():
            if rband is not None and cband is not None:
                dirs[(dr, dc)] = (rband, cband)
    return dirs


def _place_component(
    shape: tuple[int, int],
    band_rows: tuple[int, int],
    band_cols: tuple[int, int],
    step: int,
    direction: tuple[int, int],
    source_block: set[tuple[int, int]],
    source_cells: set[tuple[int, int]],
    target_block: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    dr, dc = direction
    row_lo, row_hi = band_rows
    col_lo, col_hi = band_cols
    for _ in range(150):
        max_h = row_hi - row_lo + 1
        max_w = col_hi - col_lo + 1
        patch_h = rng.randint(1, min(4, max_h))
        patch_w = rng.randint(1, min(4, max_w))
        area_cap = min(OBJECT_SIZE_RANGE[1], patch_h * patch_w)
        if area_cap < OBJECT_SIZE_RANGE[0]:
            continue
        size = rng.randint(OBJECT_SIZE_RANGE[0], area_cap)
        blob = _random_blob(patch_h, patch_w, size, rng)
        if not blob:
            continue

        arr = np.array(sorted(blob), dtype=int)
        _, _, br1, bc1 = bbox_coords(arr)
        blob_h = br1 + 1
        blob_w = bc1 + 1
        if row_lo + blob_h - 1 > row_hi or col_lo + blob_w - 1 > col_hi:
            continue

        top = rng.randint(row_lo, row_hi - blob_h + 1)
        left = rng.randint(col_lo, col_hi - blob_w + 1)
        cells = {(top + r, left + c) for r, c in blob}
        if cells & source_block:
            continue

        translated = {(r + step * dr, c + step * dc) for r, c in cells}
        if not all(in_bounds(shape, r, c) for r, c in translated):
            continue
        if translated & source_cells:
            continue
        if translated & target_block:
            continue
        if translated & cells:
            continue
        return cells, translated
    return None


def _choose_directions(available: list[tuple[int, int]], count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count <= len(available):
        return rng.sample(available, count)
    directions = available[:]
    while len(directions) < count:
        directions.append(rng.choice(available))
    rng.shuffle(directions)
    return directions


def _place_other_component(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    for _ in range(150):
        max_h = r1 - r0 + 1
        max_w = c1 - c0 + 1
        patch_h = rng.randint(1, min(4, max_h))
        patch_w = rng.randint(1, min(4, max_w))
        area_cap = min(OBJECT_SIZE_RANGE[1], patch_h * patch_w)
        if area_cap < OBJECT_SIZE_RANGE[0]:
            continue
        size = rng.randint(OBJECT_SIZE_RANGE[0], area_cap)
        blob = _random_blob(patch_h, patch_w, size, rng)
        if not blob:
            continue
        arr = np.array(sorted(blob), dtype=int)
        _, _, br1, bc1 = bbox_coords(arr)
        blob_h = br1 + 1
        blob_w = bc1 + 1
        if r0 + blob_h - 1 > r1 or c0 + blob_w - 1 > c1:
            continue
        top = rng.randint(r0, r1 - blob_h + 1)
        left = rng.randint(c0, c1 - blob_w + 1)
        cells = {(top + r, left + c) for r, c in blob}
        if cells & forbidden:
            continue
        return cells
    return None


def _place_markers(
    shape: tuple[int, int],
    mode: str,
    bbox: tuple[int, int, int, int],
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    if mode == "lr":
        row = rng.choice([r0, r1])
        candidates = [c for c in range(c0, c1 + 1) if (row, c) not in blocked]
        pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1 :] if abs(a - b) >= 2]
        if not pairs:
            return None
        c_a, c_b = rng.choice(pairs)
        return (row, c_a), (row, c_b)

    col = rng.choice([c0, c1])
    candidates = [r for r in range(r0, r1 + 1) if (r, col) not in blocked]
    pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1 :] if abs(a - b) >= 2]
    if not pairs:
        return None
    r_a, r_b = rng.choice(pairs)
    return (r_a, col), (r_b, col)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.choice(GRID_SIZES), rng.choice(GRID_SIZES))
    mode = rng.choice(["lr", "tb"])
    side = rng.choice([0, 1])
    keep_bbox = _half_bbox(shape, mode, side)
    drop_bbox = _half_bbox(shape, mode, 1 - side)

    keep_count = rng.randint(*KEEP_COUNT_RANGE)
    feasible = _feasible_dirs(shape, keep_bbox, keep_count)
    if not feasible:
        return None

    source_cells: set[tuple[int, int]] = set()
    source_block: set[tuple[int, int]] = set()
    target_cells: set[tuple[int, int]] = set()
    target_block: set[tuple[int, int]] = set()
    kept_components: list[set[tuple[int, int]]] = []

    directions = _choose_directions(list(feasible), keep_count, rng)
    for direction in directions:
        placed = _place_component(
            shape,
            feasible[direction][0],
            feasible[direction][1],
            keep_count,
            direction,
            source_block,
            source_cells,
            target_block,
            rng,
        )
        if placed is None:
            return None
        cells, translated = placed
        kept_components.append(cells)
        source_cells |= cells
        source_block |= _expanded(cells, shape)
        target_cells |= translated
        target_block |= _expanded(translated, shape)

    total_min = max(TOTAL_COUNT_RANGE[0], keep_count + 1)
    total_max = min(TOTAL_COUNT_RANGE[1], keep_count + 4)
    if total_min > total_max:
        return None
    total_count = rng.randint(total_min, total_max)
    other_count = total_count - keep_count

    other_components: list[set[tuple[int, int]]] = []
    other_forbidden = set(source_block) | set(target_block)
    for _ in range(other_count):
        cells = _place_other_component(shape, drop_bbox, other_forbidden, rng)
        if cells is None:
            return None
        other_components.append(cells)
        source_cells |= cells
        source_block |= _expanded(cells, shape)
        other_forbidden |= _expanded(cells, shape)

    marker_block = set(source_block) | set(target_block)
    marker_pair = _place_markers(shape, mode, keep_bbox, marker_block, rng)
    if marker_pair is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    for cells in kept_components + other_components:
        for r, c in cells:
            grid[r, c] = BLUE
    for r, c in marker_pair:
        if grid[r, c] != BG:
            return None
        grid[r, c] = MARKER

    try:
        out = _transform(grid)
    except AssertionError:
        return None

    kept_only = np.zeros_like(grid, dtype=bool)
    kr0, kc0, kr1, kc1 = keep_bbox
    kept_only[kr0 : kr1 + 1, kc0 : kc1 + 1] = True
    blue_kept = components_mask((grid == BLUE) & kept_only)
    if len(blue_kept) != keep_count:
        return None
    if not np.any(out == GREEN):
        return None
    if np.any((grid == MARKER) & (out == GREEN)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
