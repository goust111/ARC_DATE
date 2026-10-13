from __future__ import annotations

import random
from itertools import combinations

import numpy as np


MIN_PATH_RUN = 3
MIN_PATH_LEN = 5
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _non_bg_runs(arr: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr):
        if int(value) != bg:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(arr) - 1))
    return runs


def _parse_legend_line(line: np.ndarray, bg: int) -> list[tuple[int, int, int]] | None:
    entries: list[tuple[int, int, int]] = []
    i = 0
    while i < len(line):
        if int(line[i]) == bg:
            i += 1
            continue
        j = i
        while j < len(line) and int(line[j]) != bg:
            j += 1
        if j - i != 3:
            return None
        a, b, mixed = map(int, line[i:j])
        if a == b or mixed in (a, b):
            return None
        entries.append((a, b, mixed))
        i = j
    return entries or None


def _detect_legend(
    grid: np.ndarray, bg: int
) -> tuple[str, dict[frozenset[int], int], tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[str, list[tuple[int, int, int]]]] = []

    if h >= 2 and np.any(grid[0] != bg) and np.all(grid[1] == bg):
        entries = _parse_legend_line(grid[0], bg)
        if entries is not None:
            candidates.append(("top", entries))
    if h >= 2 and np.any(grid[-1] != bg) and np.all(grid[-2] == bg):
        entries = _parse_legend_line(grid[-1], bg)
        if entries is not None:
            candidates.append(("bottom", entries))
    if w >= 2 and np.any(grid[:, 0] != bg) and np.all(grid[:, 1] == bg):
        entries = _parse_legend_line(grid[:, 0], bg)
        if entries is not None:
            candidates.append(("left", entries))
    if w >= 2 and np.any(grid[:, -1] != bg) and np.all(grid[:, -2] == bg):
        entries = _parse_legend_line(grid[:, -1], bg)
        if entries is not None:
            candidates.append(("right", entries))

    assert candidates, "expected a parsable edge legend"
    edge, entries = max(candidates, key=lambda item: len(item[1]))

    mix_lookup: dict[frozenset[int], int] = {}
    pure_colors: set[int] = set()
    for a, b, mixed in entries:
        key = frozenset((a, b))
        assert key not in mix_lookup, "duplicate legend pair"
        mix_lookup[key] = mixed
        pure_colors.add(a)
        pure_colors.add(b)

    row_lo, row_hi = 0, h - 1
    col_lo, col_hi = 0, w - 1
    if edge == "top":
        row_lo = 2
    elif edge == "bottom":
        row_hi = h - 3
    elif edge == "left":
        col_lo = 2
    else:
        assert edge == "right"
        col_hi = w - 3

    return edge, mix_lookup, (row_lo, row_hi, col_lo, col_hi)


def _dominant_color(arr: np.ndarray, bg: int) -> int:
    vals = arr[arr != bg]
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_horizontal_paths(
    grid: np.ndarray, bg: int, bounds: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    row_lo, row_hi, col_lo, col_hi = bounds
    paths: list[tuple[int, int, int, int]] = []
    for r in range(row_lo, row_hi + 1):
        line = grid[r, col_lo : col_hi + 1]
        runs = [run for run in _non_bg_runs(line, bg) if run[1] - run[0] + 1 >= MIN_PATH_RUN]
        if not runs:
            continue
        start, end = max(runs, key=lambda run: run[1] - run[0])
        color = _dominant_color(line[start : end + 1], bg)
        paths.append((r, col_lo + start, col_lo + end, color))
    return paths


def _find_vertical_paths(
    grid: np.ndarray, bg: int, bounds: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    row_lo, row_hi, col_lo, col_hi = bounds
    paths: list[tuple[int, int, int, int]] = []
    for c in range(col_lo, col_hi + 1):
        line = grid[row_lo : row_hi + 1, c]
        runs = [run for run in _non_bg_runs(line, bg) if run[1] - run[0] + 1 >= MIN_PATH_RUN]
        if not runs:
            continue
        start, end = max(runs, key=lambda run: run[1] - run[0])
        color = _dominant_color(line[start : end + 1], bg)
        paths.append((c, row_lo + start, row_lo + end, color))
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """Blend hidden path crossings according to the edge legend."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    _, mix_lookup, bounds = _detect_legend(g, bg)
    horizontal = _find_horizontal_paths(g, bg, bounds)
    vertical = _find_vertical_paths(g, bg, bounds)
    assert horizontal and vertical, "expected both horizontal and vertical paths"

    out = g.copy()
    for row, c0, c1, h_color in horizontal:
        for col, r0, r1, v_color in vertical:
            if c0 <= col <= c1 and r0 <= row <= r1:
                out[row, col] = mix_lookup[frozenset((h_color, v_color))]
    return out


def _choose_spaced_positions(lo: int, hi: int, count: int, gap: int = 2) -> list[int] | None:
    positions = list(range(lo, hi + 1))
    for _ in range(100):
        chosen = sorted(random.sample(positions, count))
        if all(b - a >= gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _choose_span(lo: int, hi: int, anchors: list[int], min_len: int) -> tuple[int, int] | None:
    left_anchor = random.choice(anchors)
    right_anchor = random.choice(anchors)
    left, right = sorted((left_anchor, right_anchor))
    start_hi = min(left, hi - min_len + 1)
    end_lo = max(right, lo + min_len - 1)
    if lo > start_hi or end_lo > hi:
        return None
    for _ in range(50):
        start = random.randint(lo, start_hi)
        end_min = max(end_lo, start + min_len - 1)
        if end_min > hi:
            continue
        end = random.randint(end_min, hi)
        return start, end
    return None


def _legend_length(entry_count: int) -> int:
    return 4 * entry_count - 1


def _paint_legend(
    grid: np.ndarray,
    edge: str,
    entries: list[tuple[int, int, int]],
    bg: int,
) -> None:
    legend_len = _legend_length(len(entries))
    if edge in ("top", "bottom"):
        row = 0 if edge == "top" else grid.shape[0] - 1
        start = random.randint(0, grid.shape[1] - legend_len)
        cursor = start
        for idx, (a, b, mixed) in enumerate(entries):
            grid[row, cursor : cursor + 3] = [a, b, mixed]
            cursor += 3
            if idx + 1 < len(entries):
                grid[row, cursor] = bg
                cursor += 1
    else:
        col = 0 if edge == "left" else grid.shape[1] - 1
        start = random.randint(0, grid.shape[0] - legend_len)
        cursor = start
        for idx, (a, b, mixed) in enumerate(entries):
            grid[cursor : cursor + 3, col] = [a, b, mixed]
            cursor += 3
            if idx + 1 < len(entries):
                grid[cursor, col] = bg
                cursor += 1


def _scene_bounds(h: int, w: int, edge: str) -> tuple[int, int, int, int]:
    row_lo, row_hi = 1, h - 2
    col_lo, col_hi = 1, w - 2
    if edge == "top":
        row_lo = 2
    elif edge == "bottom":
        row_hi = h - 3
    elif edge == "left":
        col_lo = 2
    else:
        assert edge == "right"
        col_hi = w - 3
    return row_lo, row_hi, col_lo, col_hi


def _build_mix_lookup(pure_colors: list[int]) -> dict[frozenset[int], int]:
    available = [color for color in range(1, 10) if color not in pure_colors]
    random.shuffle(available)
    pairs = list(combinations(pure_colors, 2))
    mix_lookup: dict[frozenset[int], int] = {}
    for idx, pair in enumerate(pairs):
        mix_lookup[frozenset(pair)] = available[idx % len(available)]
    return mix_lookup


def _gen_attempt() -> np.ndarray | None:
    h_count, v_count = random.choice([(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])
    total_paths = h_count + v_count
    pure_colors = random.sample(list(range(1, 10)), total_paths)
    mix_lookup = _build_mix_lookup(pure_colors)
    pair_entries = [
        (a, b, mix_lookup[frozenset((a, b))])
        for a, b in combinations(pure_colors, 2)
    ]
    random.shuffle(pair_entries)
    pair_entries = [
        (b, a, mixed) if random.random() < 0.5 else (a, b, mixed)
        for a, b, mixed in pair_entries
    ]

    edge = random.choice(["top", "bottom", "left", "right"])
    legend_len = _legend_length(len(pair_entries))
    if edge in ("top", "bottom"):
        h = random.randint(12, 18)
        w = random.randint(max(legend_len, 14), max(legend_len, 14) + 8)
    else:
        h = random.randint(max(legend_len, 14), max(legend_len, 14) + 8)
        w = random.randint(12, 18)

    row_lo, row_hi, col_lo, col_hi = _scene_bounds(h, w, edge)
    if row_hi - row_lo + 1 < 2 * h_count - 1 or col_hi - col_lo + 1 < 2 * v_count - 1:
        return None

    h_rows = _choose_spaced_positions(row_lo, row_hi, h_count, gap=2)
    v_cols = _choose_spaced_positions(col_lo, col_hi, v_count, gap=2)
    if h_rows is None or v_cols is None:
        return None

    width = col_hi - col_lo + 1
    height = row_hi - row_lo + 1
    min_h_len = min(MIN_PATH_LEN, width)
    min_v_len = min(MIN_PATH_LEN, height)
    if min_h_len < MIN_PATH_RUN or min_v_len < MIN_PATH_RUN:
        return None

    h_spans: list[tuple[int, int]] = []
    for _ in range(h_count):
        span = _choose_span(col_lo, col_hi, v_cols, min_h_len)
        if span is None:
            return None
        h_spans.append(span)

    v_spans: list[tuple[int, int]] = []
    for _ in range(v_count):
        span = _choose_span(row_lo, row_hi, h_rows, min_v_len)
        if span is None:
            return None
        v_spans.append(span)

    intersections = np.zeros((h_count, v_count), dtype=bool)
    for i, row in enumerate(h_rows):
        c0, c1 = h_spans[i]
        for j, col in enumerate(v_cols):
            r0, r1 = v_spans[j]
            intersections[i, j] = c0 <= col <= c1 and r0 <= row <= r1

    if not intersections.any():
        return None
    if np.any(intersections.sum(axis=1) == 0) or np.any(intersections.sum(axis=0) == 0):
        return None

    bg = 0
    grid = np.full((h, w), bg, dtype=int)
    _paint_legend(grid, edge, pair_entries, bg)

    h_colors = pure_colors[:h_count]
    v_colors = pure_colors[h_count:]
    for row, (c0, c1), color in zip(h_rows, h_spans, h_colors):
        grid[row, c0 : c1 + 1] = color
    for col, (r0, r1), color in zip(v_cols, v_spans, v_colors):
        grid[r0 : r1 + 1, col] = color

    for i, row in enumerate(h_rows):
        for j, col in enumerate(v_cols):
            if intersections[i, j]:
                grid[row, col] = h_colors[i] if random.random() < 0.5 else v_colors[j]

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
