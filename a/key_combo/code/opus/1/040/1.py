from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2

MIN_BLUE_PIXELS = 10
MAX_BLUE_PIXELS = 20
MAX_SIDE = 24
MAX_GENERATION_ATTEMPTS = 400

# Keep row/column counts different so the spacing source is solver-visible.
COUNT_PAIRS = [(3, 2), (4, 2), (4, 3)]


def _empty_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    empty_rows = [r for r in range(h) if np.all(grid[r, :] == BG)]
    empty_cols = [c for c in range(w) if np.all(grid[:, c] == BG)]
    return empty_rows, empty_cols


def _ring_coords(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    h, w = shape
    r, c = center
    top = r - radius
    bottom = r + radius
    left = c - radius
    right = c + radius
    if top > h - 1 or bottom < 0 or left > w - 1 or right < 0:
        return []

    coords: list[tuple[int, int]] = []
    for cc in range(left, right + 1):
        if 0 <= top < h and 0 <= cc < w:
            coords.append((top, cc))
        if bottom != top and 0 <= bottom < h and 0 <= cc < w:
            coords.append((bottom, cc))
    for rr in range(top + 1, bottom):
        if 0 <= rr < h and 0 <= left < w:
            coords.append((rr, left))
        if right != left and 0 <= rr < h and 0 <= right < w:
            coords.append((rr, right))
    return coords


def _solve(
    grid: np.ndarray,
    *,
    spacing_override: int | None = None,
    block_existing: bool = True,
    return_trace: bool = False,
):
    g = np.asarray(grid, dtype=int)
    empty_rows, empty_cols = _empty_lines(g)
    spacing = len(empty_rows) if spacing_override is None else int(spacing_override)
    assert spacing >= 1, "spacing must be positive"

    blockers = g != BG if block_existing else np.zeros_like(g, dtype=bool)
    out = g.copy()
    traces = []

    for seed in [(r, c) for r in empty_rows for c in empty_cols]:
        before_seed_red = out == RED
        layers = []
        layer_index = 1
        while True:
            coords = _ring_coords(g.shape, seed, layer_index * spacing)
            drawn = 0
            blocked = 0
            for rr, cc in coords:
                if blockers[rr, cc]:
                    blocked += 1
                    continue
                out[rr, cc] = RED
                drawn += 1

            layers.append(
                {
                    "distance": layer_index * spacing,
                    "in_bounds": len(coords),
                    "blocked": blocked,
                    "drawn": drawn,
                }
            )
            if drawn == 0:
                break
            layer_index += 1

        if return_trace:
            traces.append(
                {
                    "seed": seed,
                    "changed_mask": (out == RED) & (~before_seed_red),
                    "layers": layers,
                }
            )

    if return_trace:
        return out, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red concentric square borders from every empty-row/empty-column intersection.

    Input structure:
    - A black(0) background grid contains sparse blue(1) obstacle pixels.
    - A few rows and a few columns are completely black.
    - Every intersection of an empty row and an empty column is a black seed location.

    Rule:
    - Count the completely black rows; call this spacing S.
    - Use every empty-row/empty-column intersection as a seed.
    - For each seed in reading order, draw red(2) square borders centered on the seed with half-side lengths S, 2S, 3S, ...
    - Draw only on black cells: existing foreground pixels stay unchanged and block those border cells.
    - Stop a seed when the next border would add no red pixel because every border cell is out of bounds or blocked.

    Key insight:
    - The same empty-line detection provides both hidden controls at once: row/column intersections say where to grow, and the number of empty rows says by how far each square layer expands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    empty_rows, empty_cols = _empty_lines(g)
    assert empty_rows and empty_cols, "expected at least one empty row and one empty column"
    return _solve(g)


def _choose_lines(length: int, count: int, spacing: int, anchor: int) -> list[int] | None:
    candidates = list(range(spacing, length - spacing))
    if anchor not in candidates:
        return None

    for _ in range(120):
        chosen = [anchor]
        pool = candidates[:]
        random.shuffle(pool)
        for pos in pool:
            if pos == anchor:
                continue
            if all(abs(pos - prev) >= 2 for prev in chosen):
                chosen.append(pos)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _base_cover(allowed_rows: list[int], allowed_cols: list[int]) -> set[tuple[int, int]]:
    nr = len(allowed_rows)
    nc = len(allowed_cols)
    if nr >= nc:
        row_seq = random.sample(allowed_rows, nr)
        col_seq: list[int] = []
        while len(col_seq) < nr:
            col_seq.extend(random.sample(allowed_cols, nc))
        col_seq = col_seq[:nr]
    else:
        col_seq = random.sample(allowed_cols, nc)
        row_seq = []
        while len(row_seq) < nc:
            row_seq.extend(random.sample(allowed_rows, nr))
        row_seq = row_seq[:nc]
    return {(int(r), int(c)) for r, c in zip(row_seq, col_seq)}


def _all_ring_positions(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
    spacing: int,
    allowed_rows: set[int],
    allowed_cols: set[int],
    max_layers: int,
) -> list[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for seed in seeds:
        for layer in range(1, max_layers + 1):
            for rr, cc in _ring_coords(shape, seed, layer * spacing):
                if rr in allowed_rows and cc in allowed_cols:
                    positions.add((rr, cc))
    return list(positions)


def _semantic_ok(grid: np.ndarray, expected_rows: list[int], expected_cols: list[int]) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    empty_rows, empty_cols = _empty_lines(g)
    if empty_rows != expected_rows or empty_cols != expected_cols:
        return False
    if len(empty_rows) == len(empty_cols):
        return False

    blue_count = int(np.count_nonzero(g == BLUE))
    if not (MIN_BLUE_PIXELS <= blue_count <= MAX_BLUE_PIXELS):
        return False
    if np.count_nonzero((g != BG) & (g != BLUE)) != 0:
        return False

    spacing = len(empty_rows)
    seeds = [(r, c) for r in empty_rows for c in empty_cols]
    if not seeds:
        return False
    if any(g[r, c] != BG for r, c in seeds):
        return False
    if any(min(r, h - 1 - r, c, w - 1 - c) < spacing for r, c in seeds):
        return False

    out, traces = _solve(g, return_trace=True)
    if not np.array_equal(out, main(g)):
        return False
    if np.any(out[g == BLUE] != BLUE):
        return False

    if any(not trace["changed_mask"].any() for trace in traces):
        return False

    if max(sum(layer["drawn"] > 0 for layer in trace["layers"]) for trace in traces) < 2:
        return False

    if not any(layer["blocked"] > 0 for trace in traces for layer in trace["layers"]):
        return False

    if np.array_equal(out, _solve(g, spacing_override=len(empty_cols))):
        return False

    if np.array_equal(out, _solve(g, block_existing=False)):
        return False

    if np.count_nonzero(out == RED) / float(g.size) > 0.82:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    empty_row_count, empty_col_count = random.choice(COUNT_PAIRS)

    min_h = max(4 * empty_row_count + 1, empty_row_count + MIN_BLUE_PIXELS + 2)
    min_w = max(4 * empty_row_count + 1, empty_col_count + MIN_BLUE_PIXELS + 2)
    max_h = min(MAX_SIDE, empty_row_count + MAX_BLUE_PIXELS)
    max_w = min(MAX_SIDE, empty_col_count + MAX_BLUE_PIXELS)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    anchor_row = random.randint(2 * empty_row_count, h - 1 - 2 * empty_row_count)
    anchor_col = random.randint(2 * empty_row_count, w - 1 - 2 * empty_row_count)
    empty_rows = _choose_lines(h, empty_row_count, empty_row_count, anchor_row)
    empty_cols = _choose_lines(w, empty_col_count, empty_row_count, anchor_col)
    if empty_rows is None or empty_cols is None:
        return None

    allowed_rows = [r for r in range(h) if r not in set(empty_rows)]
    allowed_cols = [c for c in range(w) if c not in set(empty_cols)]

    pixels = _base_cover(allowed_rows, allowed_cols)
    seeds = [(r, c) for r in empty_rows for c in empty_cols]
    allowed_row_set = set(allowed_rows)
    allowed_col_set = set(allowed_cols)

    central_seed = (anchor_row, anchor_col)
    central_layer_two = [
        pos
        for pos in _ring_coords((h, w), central_seed, 2 * empty_row_count)
        if pos[0] in allowed_row_set and pos[1] in allowed_col_set
    ]
    any_ring = _all_ring_positions(
        (h, w),
        seeds,
        empty_row_count,
        allowed_row_set,
        allowed_col_set,
        max_layers=2,
    )

    if not any_ring:
        return None

    forced_pool = [pos for pos in central_layer_two if pos not in pixels]
    if not forced_pool:
        forced_pool = [pos for pos in any_ring if pos not in pixels]
    if forced_pool:
        pixels.add(random.choice(forced_pool))

    max_extra = min(4, MAX_BLUE_PIXELS - len(pixels))
    extra_count = 0 if max_extra <= 0 else random.randint(0, max_extra)
    ring_pool = [pos for pos in any_ring if pos not in pixels]
    random.shuffle(ring_pool)

    for pos in ring_pool[:extra_count]:
        pixels.add(pos)

    if len(pixels) < MIN_BLUE_PIXELS or len(pixels) > MAX_BLUE_PIXELS:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in pixels:
        grid[r, c] = BLUE

    if not _semantic_ok(grid, empty_rows, empty_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-growth ARC input")
