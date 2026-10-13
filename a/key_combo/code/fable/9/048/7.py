from __future__ import annotations

import random

import numpy as np


BG = 0
TEAL = 8
DEFLECTOR_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

HEIGHT_RANGE = (10, 22)
WIDTH_RANGE = (14, 28)
SOURCE_COUNT_RANGE = (2, 4)
ACTIVE_COUNT_CHOICES = (0, 1, 2, 3)
ACTIVE_COUNT_WEIGHTS = (2, 3, 3, 2)
EXTRA_INACTIVE_RANGE = (1, 3)
MAX_ATTEMPTS = 400


def _parse_sources_and_deflectors(grid: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    h, w = map(int, g.shape)
    if h == 0 or w == 0:
        raise AssertionError("empty grid is invalid")

    sources = [int(r) for r in np.flatnonzero(g[:, 0] == TEAL)]
    if not sources:
        raise AssertionError("expected teal source pixels on the left edge")

    teal_coords = np.argwhere(g == TEAL)
    for r, c in teal_coords:
        if int(c) != 0:
            raise AssertionError("teal may only appear as left-edge source pixels")

    deflectors: dict[int, tuple[int, int]] = {}
    for c in range(1, w):
        rows = np.flatnonzero((g[:, c] != BG) & (g[:, c] != TEAL))
        if rows.size > 1:
            raise AssertionError("each interior column may contain at most one deflector")
        if rows.size == 1:
            r = int(rows[0])
            deflectors[c] = (r, int(g[r, c]))
    return sources, deflectors


def _simulate_streams(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    out = g.copy()
    sources, deflectors = _parse_sources_and_deflectors(g)

    occupied: dict[tuple[int, int], int] = {}
    traces: list[dict[str, object]] = []
    for path_idx, source_row in enumerate(sources):
        row = int(source_row)
        color = TEAL
        active_cols: list[int] = []
        colors_seen: list[int] = []
        drawn_cells: list[tuple[int, int]] = [(row, 0)]
        occupied[(row, 0)] = path_idx

        for c in range(1, w):
            hit = deflectors.get(c)
            if hit is not None and hit[0] == row:
                row += 1
                if row >= h:
                    raise AssertionError("stream left the bottom boundary")
                color = int(hit[1])
                active_cols.append(int(c))
                colors_seen.append(color)

            if g[row, c] != BG:
                raise AssertionError("streams may only paint over background cells")
            cell = (row, c)
            if cell in occupied:
                raise AssertionError("stream paths must not overlap")
            occupied[cell] = path_idx
            out[row, c] = color
            drawn_cells.append(cell)

        traces.append(
            {
                "source_row": source_row,
                "active_cols": active_cols,
                "colors_seen": colors_seen,
                "drawn_cells": drawn_cells,
            }
        )

    return out, traces, occupied


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw rightward streams from left-edge teal sources, lowering and recoloring them at aligned deflectors.

    Input structure:
    - A black grid has several teal source pixels on the left edge.
    - Interior columns may contain single colored deflector pixels; each such pixel is
      non-black and non-teal, and each column has at most one.
    - The layout is arranged so every stream can keep moving right without leaving the
      bottom edge or colliding with another stream.

    Rule:
    - Each teal source emits a stream that starts in teal and paints one cell in every
      column while moving from left to right.
    - Before entering column c, if that column's deflector lies on the stream's current
      row, the stream first moves down by one row and then paints column c.
    - From that deflection column onward, the stream's painted color becomes the
      deflector's color; deflector pixels themselves remain unchanged.

    Key insight:
    - A deflector changes both geometry and state: it lowers the stream by one row and
      also passes its own color to the entire downstream segment.
    """
    out, _traces, _occupied = _simulate_streams(np.asarray(grid, dtype=int))
    return out


def _row_before_column(source_row: int, active_cols: list[int], col: int) -> int:
    return int(source_row + sum(1 for active_col in active_cols if active_col < col))


def _choose_active_counts(n_sources: int) -> list[int]:
    counts = random.choices(ACTIVE_COUNT_CHOICES, weights=ACTIVE_COUNT_WEIGHTS, k=n_sources)
    if max(counts) < 2:
        counts[random.randrange(n_sources)] = 2
    if sum(counts) < 2:
        idx = random.randrange(n_sources)
        counts[idx] = min(3, counts[idx] + 1)
    if n_sources >= 3 and all(count > 0 for count in counts) and random.random() < 0.55:
        counts[random.randrange(n_sources)] = 0
        if max(counts) < 2:
            counts[random.randrange(n_sources)] = 2
    return [int(v) for v in counts]


def _choose_source_rows(active_counts: list[int]) -> tuple[int, list[int]]:
    n_sources = len(active_counts)
    top_margin = random.randint(0, 2)
    rows = [top_margin]
    for i in range(n_sources - 1):
        separation = active_counts[i] + random.randint(1, 3)
        rows.append(rows[-1] + separation)
    min_h = rows[-1] + active_counts[-1] + 1
    max_h = min(HEIGHT_RANGE[1], max(HEIGHT_RANGE[0], min_h + random.randint(1, 6)))
    if min_h > max_h:
        raise AssertionError("failed to fit source rows into the height budget")
    h = random.randint(max(HEIGHT_RANGE[0], min_h), max_h)
    return h, [int(r) for r in rows]


def _pick_active_columns(width: int, active_counts: list[int]) -> list[list[int]] | None:
    usable_cols = list(range(1, width - 1))
    total_active = sum(active_counts)
    if len(usable_cols) < total_active + 1:
        return None

    remaining = usable_cols[:]
    per_path: list[list[int]] = [[] for _ in active_counts]
    order = list(range(len(active_counts)))
    random.shuffle(order)
    for path_idx in order:
        need = active_counts[path_idx]
        if need == 0:
            continue
        if len(remaining) < need:
            return None
        chosen = sorted(random.sample(remaining, need))
        per_path[path_idx] = chosen
        remaining = [c for c in remaining if c not in set(chosen)]
    return per_path


def _pick_colors_for_path(count: int) -> list[int]:
    current = TEAL
    colors: list[int] = []
    for _ in range(count):
        palette = [color for color in DEFLECTOR_COLORS if color != current]
        current = random.choice(palette)
        colors.append(current)
    return colors


def _path_rows_before(rows: list[int], active_cols_per_path: list[list[int]], col: int) -> list[int]:
    return [
        _row_before_column(rows[path_idx], active_cols_per_path[path_idx], col)
        for path_idx in range(len(rows))
    ]


def _build_input_from_plan(
    height: int,
    width: int,
    rows: list[int],
    active_cols_per_path: list[list[int]],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    grid[rows, 0] = TEAL

    used_cols: set[int] = set()
    for path_idx, source_row in enumerate(rows):
        colors = _pick_colors_for_path(len(active_cols_per_path[path_idx]))
        for col, color in zip(active_cols_per_path[path_idx], colors):
            row_before = _row_before_column(source_row, active_cols_per_path[path_idx], col)
            if row_before + 1 >= height:
                return None
            grid[row_before, col] = color
            used_cols.add(col)

    inactive_target = random.randint(*EXTRA_INACTIVE_RANGE)
    candidate_cols = [c for c in range(1, width - 1) if c not in used_cols]
    random.shuffle(candidate_cols)
    inactive_added = 0
    for col in candidate_cols:
        current_rows = _path_rows_before(rows, active_cols_per_path, col)
        forbidden = set(current_rows)
        near_rows = []
        for row in current_rows:
            for delta in (-1, 1):
                rr = row + delta
                if 0 <= rr < height and rr not in forbidden:
                    near_rows.append(rr)
        all_rows = [rr for rr in range(height) if rr not in forbidden]
        if not all_rows:
            continue
        chosen_row = random.choice(near_rows if near_rows and random.random() < 0.8 else all_rows)
        grid[chosen_row, col] = random.choice(DEFLECTOR_COLORS)
        inactive_added += 1
        if inactive_added >= inactive_target:
            break

    if inactive_added == 0:
        return None
    return grid


def _validate_generated_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    if max(h, w) > 30:
        raise AssertionError("ARC grids must stay within 30x30")

    sources, deflectors = _parse_sources_and_deflectors(g)
    if len(sources) < 2:
        raise AssertionError("need multiple source streams")

    active_hits = 0
    inactive_hits = 0
    active_colors: set[int] = set()

    out, traces, occupied = _simulate_streams(g)
    for col, (row, color) in deflectors.items():
        if any(col in trace["active_cols"] for trace in traces):
            active_hits += 1
            active_colors.add(int(color))
            if row + 1 >= h or int(out[row + 1, col]) != int(color):
                raise AssertionError("an active deflector must color the stream below itself")
        else:
            inactive_hits += 1
            if (row, col) in occupied:
                raise AssertionError("inactive deflectors must not sit on any drawn path")

    active_counts = [len(trace["active_cols"]) for trace in traces]
    if active_hits < 2:
        raise AssertionError("need at least two active deflections")
    if inactive_hits < 1:
        raise AssertionError("need at least one inactive distractor deflector")
    if max(active_counts) < 2:
        raise AssertionError("need a stream with multiple recoloring deflections")
    if len(active_colors) < 2:
        raise AssertionError("need multiple active deflector colors")
    if np.array_equal(out, g):
        raise AssertionError("output must differ from input")
    if len({tuple(trace["drawn_cells"]) for trace in traces}) != len(traces):
        raise AssertionError("distinct sources must not create identical paths")


def _generate_attempt() -> np.ndarray | None:
    n_sources = random.randint(*SOURCE_COUNT_RANGE)
    active_counts = _choose_active_counts(n_sources)
    try:
        height, rows = _choose_source_rows(active_counts)
    except AssertionError:
        return None

    min_width = max(WIDTH_RANGE[0], sum(active_counts) + random.randint(5, 9))
    max_width = WIDTH_RANGE[1]
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    active_cols_per_path = _pick_active_columns(width, active_counts)
    if active_cols_per_path is None:
        return None

    grid = _build_input_from_plan(height, width, rows, active_cols_per_path)
    if grid is None:
        return None

    try:
        _validate_generated_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-flow puzzle")
