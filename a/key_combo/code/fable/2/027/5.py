from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
PERIOD_RANGE = (2, 6)
REPEAT_RANGE = (3, 5)
PALETTE_SIZE_RANGE = (3, 5)
ANOMALY_MODES = ("single_tile", "cross_row", "cross_col", "cross_both")
GEN_RETRIES = 500
TILE_RETRIES = 300
SHAPE_RETRIES = 200


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return len(seen) == int(mask.sum())


def _exact_axis_period(tile: np.ndarray, axis: int) -> int:
    size = tile.shape[axis]
    for step in range(1, size + 1):
        if np.array_equal(tile, np.roll(tile, -step, axis=axis)):
            return int(step)
    return int(size)


def _axis_mismatch_rates(grid: np.ndarray, axis: int) -> dict[int, float]:
    g = np.asarray(grid, dtype=int)
    size = g.shape[axis]
    rates: dict[int, float] = {}
    for step in range(1, size):
        if axis == 0:
            lhs, rhs = g[:-step, :], g[step:, :]
        else:
            lhs, rhs = g[:, :-step], g[:, step:]
        rates[int(step)] = float(np.count_nonzero(lhs != rhs)) / float(lhs.size)
    return rates


def _low_shift_cluster(rates: dict[int, float]) -> list[int]:
    ranked = sorted((rate, step) for step, rate in rates.items())
    if len(ranked) == 1:
        return [int(ranked[0][1])]

    search_limit = min(len(ranked) - 1, 7)
    split = 1
    best_gap = -1.0
    for i in range(search_limit):
        gap = ranked[i + 1][0] - ranked[i][0]
        if gap > best_gap + 1e-12:
            best_gap = gap
            split = i + 1

    left_rate = ranked[split - 1][0]
    right_rate = ranked[split][0] if split < len(ranked) else left_rate
    threshold = (left_rate + right_rate) / 2.0 if split < len(ranked) else left_rate + 1e-9
    return sorted(int(step) for step, rate in rates.items() if rate <= threshold + 1e-12)


def _gcd_list(values: list[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, int(value))
    return int(g)


def _detect_axis_period(grid: np.ndarray, axis: int) -> int:
    rates = _axis_mismatch_rates(grid, axis)
    return _gcd_list(_low_shift_cluster(rates))


def _anomaly_mask(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros((h, w), dtype=bool)
    for rr in range(row_period):
        for cc in range(col_period):
            orbit = g[rr::row_period, cc::col_period]
            values, counts = np.unique(orbit, return_counts=True)
            idx = int(np.argmax(counts))
            assert int(counts[idx]) * 2 > orbit.size, "expected strict orbit majority"
            majority = int(values[idx])
            out[rr::row_period, cc::col_period] = orbit != majority
    return out


def _expected_output(grid: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(grid).shape, BLACK, dtype=int)
    out[anomaly] = np.asarray(grid, dtype=int)[anomaly]
    box = _bbox(anomaly)
    assert box is not None, "expected at least one anomaly"
    return _crop_bbox(out, box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the hidden tampered figure inside a repeated 2D pattern by orbit-majority voting, then crop it out.

    Input structure:
    - The whole grid is filled by a bidirectionally translational color pattern with no background region.
    - A small connected set of cells has been recolored to other colors that already belong to the pattern palette.
    - The true row and column periods still remain visible because each symmetry orbit keeps a strict correct-color majority.

    Rule:
    - Infer the row period and column period from the repeated pattern while ignoring the few local inconsistencies.
    - For every orbit of cells sharing the same coordinates modulo those periods, find the orbit's majority color.
    - Mark exactly the cells whose input color disagrees with their orbit majority as anomalous.
    - Output the minimum bounding rectangle of all anomalous cells, keeping anomalous colors and painting every other cell black.

    Key insight:
    - The repeated pattern is only a reference frame: translational orbits are used to vote on what belongs, and the answer is the connected intruder shape itself rather than the repaired wallpaper.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2

    row_period = _detect_axis_period(g, axis=0)
    col_period = _detect_axis_period(g, axis=1)
    assert 1 <= row_period < g.shape[0]
    assert 1 <= col_period < g.shape[1]

    anomaly = _anomaly_mask(g, row_period, col_period)
    assert anomaly.any(), "expected at least one anomalous cell"
    return _expected_output(g, anomaly)


def _primitive_tile(row_period: int, col_period: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(TILE_RETRIES):
        tile = np.array(
            [[random.choice(palette) for _ in range(col_period)] for _ in range(row_period)],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < min(3, len(palette)):
            continue
        if _exact_axis_period(tile, axis=0) != row_period:
            continue
        if _exact_axis_period(tile, axis=1) != col_period:
            continue
        return tile
    return None


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, a)
    tr, tc = map(int, b)
    path = [(r, c)]
    if random.random() < 0.5:
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
    return path


def _build_local_shape(
    height: int,
    width: int,
    mode: str,
    row_boundary: int | None,
    col_boundary: int | None,
    orbit_size: int,
) -> np.ndarray | None:
    max_area = min(height * width, max(3, orbit_size - 1), 12)
    if max_area < 3:
        return None

    for _ in range(SHAPE_RETRIES):
        if mode == "single_tile":
            a = (random.randrange(height), random.randrange(width))
            b = (random.randrange(height), random.randrange(width))
            if a == b:
                continue
        elif mode == "cross_row":
            assert row_boundary is not None
            if not (0 < row_boundary < height):
                return None
            a = (random.randrange(row_boundary), random.randrange(width))
            b = (random.randrange(row_boundary, height), random.randrange(width))
        elif mode == "cross_col":
            assert col_boundary is not None
            if not (0 < col_boundary < width):
                return None
            a = (random.randrange(height), random.randrange(col_boundary))
            b = (random.randrange(height), random.randrange(col_boundary, width))
        else:
            assert row_boundary is not None and col_boundary is not None
            if not (0 < row_boundary < height and 0 < col_boundary < width):
                return None
            if random.random() < 0.5:
                a = (random.randrange(row_boundary), random.randrange(col_boundary))
                b = (random.randrange(row_boundary, height), random.randrange(col_boundary, width))
            else:
                a = (random.randrange(row_boundary), random.randrange(col_boundary, width))
                b = (random.randrange(row_boundary, height), random.randrange(col_boundary))

        cells = set(_manhattan_path(a, b))
        min_area = max(3, len(cells))
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r0 == r1 or c0 == c1:
            continue
        if mode in {"cross_row", "cross_both"} and not (r0 < row_boundary <= r1):
            continue
        if mode in {"cross_col", "cross_both"} and not (c0 < col_boundary <= c1):
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _sample_window(
    row_period: int,
    col_period: int,
    row_repeats: int,
    col_repeats: int,
    mode: str,
) -> tuple[int, int, int, int, int | None, int | None]:
    height = row_period * row_repeats
    width = col_period * col_repeats

    if mode == "single_tile":
        box_h = random.randint(2, row_period)
        box_w = random.randint(2, col_period)
        tile_r = random.randrange(row_repeats)
        tile_c = random.randrange(col_repeats)
        r0 = tile_r * row_period + random.randint(0, row_period - box_h)
        c0 = tile_c * col_period + random.randint(0, col_period - box_w)
        return r0, c0, box_h, box_w, None, None

    if mode == "cross_row":
        box_h = random.randint(2, min(row_period + 1, height))
        box_w = random.randint(2, col_period)
        boundary_idx = random.randint(1, row_repeats - 1)
        local_r = random.randint(max(0, row_period - box_h + 1), row_period - 1)
        r0 = (boundary_idx - 1) * row_period + local_r
        tile_c = random.randrange(col_repeats)
        c0 = tile_c * col_period + random.randint(0, col_period - box_w)
        return r0, c0, box_h, box_w, boundary_idx * row_period - r0, None

    if mode == "cross_col":
        box_h = random.randint(2, row_period)
        box_w = random.randint(2, min(col_period + 1, width))
        boundary_idx = random.randint(1, col_repeats - 1)
        local_c = random.randint(max(0, col_period - box_w + 1), col_period - 1)
        c0 = (boundary_idx - 1) * col_period + local_c
        tile_r = random.randrange(row_repeats)
        r0 = tile_r * row_period + random.randint(0, row_period - box_h)
        return r0, c0, box_h, box_w, None, boundary_idx * col_period - c0

    box_h = random.randint(2, min(row_period + 1, height))
    box_w = random.randint(2, min(col_period + 1, width))
    row_boundary_idx = random.randint(1, row_repeats - 1)
    col_boundary_idx = random.randint(1, col_repeats - 1)
    local_r = random.randint(max(0, row_period - box_h + 1), row_period - 1)
    local_c = random.randint(max(0, col_period - box_w + 1), col_period - 1)
    r0 = (row_boundary_idx - 1) * row_period + local_r
    c0 = (col_boundary_idx - 1) * col_period + local_c
    return r0, c0, box_h, box_w, row_boundary_idx * row_period - r0, col_boundary_idx * col_period - c0


def _paint_anomalies(
    base: np.ndarray,
    palette: list[int],
    global_mask: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(base, dtype=int).copy()
    for r, c in np.argwhere(global_mask):
        rr, cc = int(r), int(c)
        base_color = int(base[rr, cc])
        replacement_choices = [color for color in palette if color != base_color]
        grid[rr, cc] = int(random.choice(replacement_choices))
    return grid


def _has_clear_period_gap(rates: dict[int, float], period: int) -> bool:
    aligned = [rate for step, rate in rates.items() if step % period == 0]
    misaligned = [rate for step, rate in rates.items() if step % period != 0]
    if not aligned or not misaligned:
        return False
    return max(aligned) + 0.05 < min(misaligned)


def _validate_candidate(
    grid: np.ndarray,
    base: np.ndarray,
    anomaly: np.ndarray,
    row_period: int,
    col_period: int,
) -> bool:
    if not anomaly.any() or not _is_connected(anomaly):
        return False

    orbit_size = (grid.shape[0] // row_period) * (grid.shape[1] // col_period)
    counts: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(anomaly):
        key = (int(r) % row_period, int(c) % col_period)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] * 2 >= orbit_size + 1:
            return False
        if int(grid[r, c]) == int(base[r, c]):
            return False

    if _detect_axis_period(grid, axis=0) != row_period:
        return False
    if _detect_axis_period(grid, axis=1) != col_period:
        return False
    if not _has_clear_period_gap(_axis_mismatch_rates(grid, axis=0), row_period):
        return False
    if not _has_clear_period_gap(_axis_mismatch_rates(grid, axis=1), col_period):
        return False

    recovered = _anomaly_mask(grid, row_period, col_period)
    if not np.array_equal(recovered, anomaly):
        return False

    expected = _expected_output(grid, anomaly)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    box = _bbox(anomaly)
    assert box is not None
    if box[2] - box[0] + 1 == grid.shape[0] and box[3] - box[1] + 1 == grid.shape[1]:
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    row_period = random.randint(*PERIOD_RANGE)
    col_period = random.randint(*PERIOD_RANGE)
    row_repeats = random.randint(*REPEAT_RANGE)
    col_repeats = random.randint(*REPEAT_RANGE)
    height = row_period * row_repeats
    width = col_period * col_repeats
    if max(height, width) > 30:
        return None

    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(COLOR_POOL, palette_size)
    tile = _primitive_tile(row_period, col_period, palette)
    if tile is None:
        return None
    base = np.tile(tile, (row_repeats, col_repeats))

    mode = random.choice(ANOMALY_MODES)
    r0, c0, box_h, box_w, row_boundary, col_boundary = _sample_window(
        row_period,
        col_period,
        row_repeats,
        col_repeats,
        mode,
    )
    local_shape = _build_local_shape(
        box_h,
        box_w,
        mode,
        row_boundary,
        col_boundary,
        orbit_size=row_repeats * col_repeats,
    )
    if local_shape is None:
        return None

    anomaly = np.zeros_like(base, dtype=bool)
    anomaly[r0 : r0 + box_h, c0 : c0 + box_w] = local_shape
    grid = _paint_anomalies(base, palette, anomaly)

    if not _validate_candidate(grid, base, anomaly, row_period, col_period):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
