from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _trace_snake(mask: np.ndarray) -> list[Coord]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert pts, "expected a non-empty snake path"

    pset = set(pts)
    degree: dict[Coord, int] = {}
    for r, c in pts:
        degree[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    endpoints = [pt for pt in pts if degree[pt] == 1]
    assert len(endpoints) == 2, "snake must have exactly two endpoints"
    assert all(d in (1, 2) for d in degree.values()), "snake must be a simple chain"

    start = min(endpoints)
    order: list[Coord] = []
    prev: Coord | None = None
    cur = start

    while True:
        order.append(cur)
        next_pts = [pt for pt in _neighbors4(*cur) if pt in pset and pt != prev]
        if not next_pts:
            break
        assert len(next_pts) == 1, "snake path must not branch"
        prev, cur = cur, next_pts[0]

    assert len(order) == len(pts), "snake traversal must visit every black pixel exactly once"
    return order


def _detect_axis_period(grid: np.ndarray, axis: int) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visible = np.argwhere(g != 0)
    assert visible.size > 0, "expected visible non-black periodic pattern"

    size = h if axis == 0 else w
    other = w if axis == 0 else h
    strict_candidates: list[int] = []
    fallback_candidates: list[tuple[float, int]] = []

    for period in _divisors(size):
        if period == size:
            continue

        mapping: dict[tuple[int, int], int] = {}
        ok = True
        for r, c in visible:
            rr = int(r)
            cc = int(c)
            key = (rr % period, cc) if axis == 0 else (rr, cc % period)
            color = int(g[rr, cc])
            old = mapping.get(key)
            if old is not None and old != color:
                ok = False
                break
            mapping[key] = color
        if not ok:
            continue

        coverage = len(mapping) / float(period * other)
        if len(mapping) == period * other:
            strict_candidates.append(period)
        fallback_candidates.append((coverage, period))

    if strict_candidates:
        return min(strict_candidates)
    if fallback_candidates:
        best_coverage = max(item[0] for item in fallback_candidates)
        best_periods = [period for coverage, period in fallback_candidates if coverage == best_coverage]
        return min(best_periods)
    raise AssertionError("failed to detect translational period")


def _residue_color_map(grid: np.ndarray, row_period: int, col_period: int) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    residue_colors: dict[tuple[int, int], int] = {}

    for r, c in np.argwhere(g != 0):
        rr = int(r)
        cc = int(c)
        key = (rr % row_period, cc % col_period)
        color = int(g[rr, cc])
        old = residue_colors.get(key)
        if old is not None and old != color:
            raise AssertionError("visible cells conflict under the inferred periodic tiling")
        residue_colors[key] = color

    expected = row_period * col_period
    assert len(residue_colors) == expected, "every tile residue class must remain visible somewhere"
    return residue_colors


def _restore_sequence(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    snake_mask = g == 0
    order = _trace_snake(snake_mask)
    row_period = _detect_axis_period(g, axis=0)
    col_period = _detect_axis_period(g, axis=1)
    residue_colors = _residue_color_map(g, row_period, col_period)
    return [residue_colors[(r % row_period, c % col_period)] for r, c in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the periodic colors hidden under a black snake path, then serialize them in snake order.

    Input structure:
    - The grid contains a non-black translationally periodic tiling built from a small 2x2 or 2x3 tile.
    - Some cells of that tiling are replaced by black, and all black cells form one 4-connected snake path
      with exactly two endpoints.
    - Black cells are occlusions, not background; every masked tile residue class still has at least one
      visible non-black copy elsewhere in the grid.

    Rule:
    - Detect the snake path from the black cells and choose the endpoint with smaller row, then smaller
      column, as the traversal start.
    - Infer the minimal row and column translation periods from the visible non-black cells.
    - Restore each black snake cell by copying the unique color of its periodic residue class.
    - Trace the snake from the chosen start to the other endpoint and output the restored colors as an Nx1 column.

    Key insight:
    - The same snake serves two indispensable roles: it marks which periodic cells must be inpainted and it
      fixes the only valid readout order for the restored color sequence.
    """
    sequence = _restore_sequence(np.asarray(grid, dtype=int))
    return np.array(sequence, dtype=int).reshape((-1, 1))


def _count_turns(path: list[Coord]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(2, len(path)):
        cur_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _sequence_is_informative(seq: list[int]) -> bool:
    if len(seq) < 2:
        return False
    if len(set(seq)) < 2:
        return False
    if seq == list(reversed(seq)):
        return False
    runs = 1 + sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
    return runs >= 3


def _sample_periodic_pattern() -> tuple[np.ndarray, int, int]:
    for _ in range(400):
        row_period = 2
        col_period = random.choice([2, 3])
        row_repeats = random.randint(5, 7)
        col_repeats = random.randint(4, 7 if col_period == 2 else 6)
        h = row_period * row_repeats
        w = col_period * col_repeats

        palette_size = random.choice([2, 3])
        palette = random.sample(range(1, 10), palette_size)
        tile = np.array(
            [[random.choice(palette) for _ in range(col_period)] for _ in range(row_period)],
            dtype=int,
        )
        if len(set(map(int, tile.flatten()))) < 2:
            continue

        full = np.tile(tile, (row_repeats, col_repeats))
        if _detect_axis_period(full, axis=0) != row_period:
            continue
        if _detect_axis_period(full, axis=1) != col_period:
            continue
        return full, row_period, col_period

    raise RuntimeError("failed to sample a non-degenerate periodic pattern")


def _can_extend(path_set: set[Coord], cur: Coord, nxt: Coord, shape: tuple[int, int]) -> bool:
    nr, nc = nxt
    if not _in_bounds(shape, nr, nc):
        return False
    if (nr, nc) in path_set:
        return False
    if nr == 0 or nc == 0 or nr == shape[0] - 1 or nc == shape[1] - 1:
        return False
    for adj in _neighbors4(nr, nc):
        if adj in path_set and adj != cur:
            return False
    return True


def _sample_snake(shape: tuple[int, int], row_period: int, col_period: int, length: int) -> list[Coord] | None:
    h, w = shape
    orbit_size = (h // row_period) * (w // col_period)
    max_per_orbit = orbit_size // 2

    for _ in range(800):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        path = [start]
        path_set = {start}
        orbit_counts: dict[tuple[int, int], int] = {(start[0] % row_period, start[1] % col_period): 1}
        last_dir: tuple[int, int] | None = None
        turns = 0

        while len(path) < length:
            cur = path[-1]
            options: list[tuple[Coord, tuple[int, int]]] = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not _can_extend(path_set, cur, nxt, shape):
                    continue
                orbit_key = (nxt[0] % row_period, nxt[1] % col_period)
                if orbit_counts.get(orbit_key, 0) >= max_per_orbit:
                    continue
                options.append((nxt, (dr, dc)))

            if not options:
                break

            if last_dir is not None:
                turning = [item for item in options if item[1] != last_dir]
                straight = [item for item in options if item[1] == last_dir]
                if turns < 2 and turning:
                    options = turning
                elif turning and straight:
                    options = turning if random.random() < 0.6 else straight

            nxt, cur_dir = random.choice(options)
            if last_dir is not None and cur_dir != last_dir:
                turns += 1
            path.append(nxt)
            path_set.add(nxt)
            orbit_key = (nxt[0] % row_period, nxt[1] % col_period)
            orbit_counts[orbit_key] = orbit_counts.get(orbit_key, 0) + 1
            last_dir = cur_dir

        if len(path) != length:
            continue
        if turns < 2:
            continue
        if len({r for r, _ in path}) < 2 or len({c for _, c in path}) < 2:
            continue
        return path

    return None


def _mask_pattern(pattern: np.ndarray, path: list[Coord]) -> np.ndarray:
    masked = np.array(pattern, copy=True)
    for r, c in path:
        masked[r, c] = 0
    return masked


def _is_semantically_valid(pattern: np.ndarray, masked: np.ndarray, row_period: int, col_period: int) -> bool:
    h, w = masked.shape
    snake_mask = masked == 0
    try:
        order = _trace_snake(snake_mask)
        inferred_row = _detect_axis_period(masked, axis=0)
        inferred_col = _detect_axis_period(masked, axis=1)
        residue_colors = _residue_color_map(masked, inferred_row, inferred_col)
    except AssertionError:
        return False

    if inferred_row != row_period or inferred_col != col_period:
        return False

    seq = [int(pattern[r, c]) for r, c in order]
    if not _sequence_is_informative(seq):
        return False

    orbit_size = (h // row_period) * (w // col_period)
    orbit_counts: dict[tuple[int, int], int] = {}
    for r, c in order:
        key = (r % row_period, c % col_period)
        orbit_counts[key] = orbit_counts.get(key, 0) + 1
        if orbit_counts[key] > orbit_size // 2:
            return False
        if residue_colors.get(key) != int(pattern[r, c]):
            return False

    expected = np.array(seq, dtype=int).reshape((-1, 1))
    actual = main(masked)
    if not np.array_equal(actual, expected):
        return False

    if max(masked.shape) > 30 or len(order) > 30:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        pattern, row_period, col_period = _sample_periodic_pattern()
        path_length = random.randint(8, 15)
        path = _sample_snake(pattern.shape, row_period, col_period, path_length)
        if path is None:
            continue

        masked = _mask_pattern(pattern, path)
        if _is_semantically_valid(pattern, masked, row_period, col_period):
            return masked

    raise RuntimeError("failed to generate a valid snake-masked periodic tiling")
