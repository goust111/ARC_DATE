from __future__ import annotations

import numpy as np


BG = 0
TARGET_COLOR = 2
BAR_COLOR = 5
SOURCE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _segments_in_row(row: np.ndarray, color: int) -> list[tuple[int, int]]:
    cols = np.flatnonzero(row == color)
    if cols.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        segments.append((start, prev))
        start = prev = col
    segments.append((start, prev))
    return segments


def _bar_segments(grid: np.ndarray, bar_color: int) -> dict[int, list[tuple[int, int]]]:
    bars: dict[int, list[tuple[int, int]]] = {}
    for row in range(grid.shape[0]):
        segments = _segments_in_row(grid[row], bar_color)
        if segments:
            bars[row] = segments
    return bars


def _trace_source(
    shape: tuple[int, int], bars_by_row: dict[int, list[tuple[int, int]]], source_col: int
) -> dict[str, object]:
    height, _ = shape
    row = 0
    col = int(source_col)
    cells = [(0, col)]
    directions: list[str] = []
    hits: list[tuple[int, int, int]] = []

    ordered_rows = sorted(bars_by_row)
    while True:
        hit = None
        for bar_row in ordered_rows:
            if bar_row <= row:
                continue
            for start, end in bars_by_row[bar_row]:
                if start <= col <= end:
                    hit = (bar_row, start, end)
                    break
            if hit is not None:
                break

        if hit is None:
            for rr in range(row + 1, height):
                cells.append((rr, col))
            break

        bar_row, start, end = hit
        for rr in range(row + 1, bar_row):
            cells.append((rr, col))

        left_dist = col - start
        right_dist = end - col
        assert left_dist != right_dist, "hit must not be centered on a bar"
        if left_dist < right_dist:
            outside_col = start - 1
            step = -1
            directions.append("left")
        else:
            outside_col = end + 1
            step = 1
            directions.append("right")

        assert 0 <= outside_col < shape[1], "detour must stay in bounds"
        for cc in range(col + step, outside_col + step, step):
            cells.append((bar_row - 1, cc))

        hits.append((bar_row, start, end))
        row = bar_row - 1
        col = outside_col

    return {
        "end_col": col,
        "cells": cells,
        "hit_count": len(hits),
        "directions": directions,
        "hits": hits,
    }


def _render_output(
    shape: tuple[int, int],
    bars_by_row: dict[int, list[tuple[int, int]]],
    trace: dict[str, object],
    path_color: int,
    bg: int,
    bar_color: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for row, segments in bars_by_row.items():
        for start, end in segments:
            out[row, start : end + 1] = bar_color
    for rr, cc in trace["cells"]:
        out[rr, cc] = path_color
    return out


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    bg = _most_common_color(g)

    non_bg = g[g != bg]
    assert non_bg.size > 0
    values, counts = np.unique(non_bg, return_counts=True)
    bar_color = int(values[np.argmax(counts)])

    source_cols = [c for c in range(width) if int(g[0, c]) not in (bg, bar_color)]
    assert len(source_cols) >= 2
    source_colors = [int(g[0, c]) for c in source_cols]
    assert len(set(source_colors)) == len(source_colors)

    target_cols = [c for c in range(width) if int(g[height - 1, c]) not in (bg, bar_color)]
    assert len(target_cols) == 1
    target_col = int(target_cols[0])

    bars_by_row = _bar_segments(g, bar_color)
    assert bars_by_row, "expected interior obstacle bars"

    traces = []
    for source_col, source_color in zip(source_cols, source_colors):
        trace = _trace_source(g.shape, bars_by_row, source_col)
        trace["source_col"] = source_col
        trace["source_color"] = source_color
        traces.append(trace)

    winners = [idx for idx, trace in enumerate(traces) if int(trace["end_col"]) == target_col]
    assert len(winners) == 1, "exactly one source must reach the target"

    return {
        "bg": bg,
        "bar_color": bar_color,
        "source_cols": source_cols,
        "source_colors": source_colors,
        "target_col": target_col,
        "bars_by_row": bars_by_row,
        "traces": traces,
        "winner_idx": winners[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only the unique top source whose cascading detours land on the bottom target.

    Input structure:
    - Black background with several distinct-colored source pixels on the top border.
    - The middle rows contain one-cell-thick gray horizontal obstacle bars that do not touch.
    - The bottom border contains a single red target pixel.

    Rule:
    - From each source, imagine water falling straight down in its current column.
    - Whenever the next cell would be a gray bar, the water slides along the row just above that bar toward the nearer bar end, then continues falling from the cell one step outside that end.
    - Exactly one source eventually lands on the bottom target; output only that full path in the winning source color, erase every other source, preserve all gray bars, and overwrite the target with the winning color.

    Key insight:
    - The answer is chosen by reachability, not by color matching, because each detour depends on which end of each struck bar is nearer to the impact column.
    """
    info = _analyze_grid(grid)
    winner = info["traces"][info["winner_idx"]]
    color = int(winner["source_color"])
    return _render_output(grid.shape, info["bars_by_row"], winner, color, info["bg"], info["bar_color"])


def _segment_too_close(segment: tuple[int, int], others: list[tuple[int, int]], min_gap: int = 2) -> bool:
    start, end = segment
    for other_start, other_end in others:
        if not (end + min_gap + 1 <= other_start or start >= other_end + min_gap + 1):
            return True
    return False


def _choose_segment_for_lane(
    lane: int,
    direction: str,
    width: int,
    current_lanes: list[int],
    future_lanes: set[int],
    existing_segments: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int, int] | None:
    params = [(near_dist, extra) for near_dist in range(3) for extra in (1, 2)]
    rng.shuffle(params)
    for near_dist, extra in params:
        if direction == "left":
            start = lane - near_dist
            end = lane + near_dist + extra
            new_lane = start - 1
        else:
            start = lane - near_dist - extra
            end = lane + near_dist
            new_lane = end + 1

        if not (1 <= start <= end <= width - 2):
            continue
        if not (1 <= new_lane <= width - 2):
            continue
        if new_lane in future_lanes:
            continue
        if any(start <= other_lane <= end for other_lane in current_lanes if other_lane != lane):
            continue
        segment = (start, end)
        if _segment_too_close(segment, existing_segments):
            continue
        return start, end, new_lane
    return None


def _place_decoy_segment(
    width: int,
    row_lanes: list[int],
    existing_segments: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for length in range(2, 7):
        for start in range(1, width - length):
            end = start + length - 1
            if any(start <= lane <= end for lane in row_lanes):
                continue
            if _segment_too_close((start, end), existing_segments):
                continue
            candidates.append((start, end))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _construct_attempt(rng: np.random.Generator) -> tuple[np.ndarray, int] | None:
    width = int(rng.integers(16, 25))
    height = int(rng.integers(18, 28))
    source_count = 3

    min_gap = 4
    source_cols: list[int] = []
    low = 2
    for idx in range(source_count):
        remaining = source_count - idx - 1
        high = width - 3 - remaining * min_gap
        if low > high:
            return None
        source_cols.append(int(rng.integers(low, high + 1)))
        low = source_cols[-1] + min_gap

    winner_idx = int(rng.integers(source_count))
    decoy_idx = int(rng.choice([idx for idx in range(source_count) if idx != winner_idx]))

    bar_count = int(rng.integers(4, 7))
    rows: list[int] = []
    row = int(rng.integers(2, 5))
    for _ in range(bar_count):
        if row > height - 3:
            break
        rows.append(row)
        row += int(rng.integers(2, 5))
    if len(rows) < 4:
        return None

    winner_hits = int(rng.integers(2, min(4, len(rows) - 1) + 1))
    hit_rows = sorted(int(r) for r in rng.choice(rows, size=winner_hits, replace=False))
    base_dirs = ["left", "right"]
    while len(base_dirs) < winner_hits:
        base_dirs.append("left" if rng.random() < 0.5 else "right")
    rng.shuffle(base_dirs)
    winner_dir_for_row = {row: direction for row, direction in zip(hit_rows, base_dirs)}

    remaining_rows = [row for row in rows if row not in winner_dir_for_row]
    if not remaining_rows:
        return None
    decoy_hits = int(rng.integers(1, min(3, len(remaining_rows)) + 1))
    decoy_rows = sorted(int(r) for r in rng.choice(remaining_rows, size=decoy_hits, replace=False))
    decoy_dir_for_row = {
        row: ("left" if rng.random() < 0.5 else "right")
        for row in decoy_rows
    }

    lanes = list(source_cols)
    bars_by_row: dict[int, list[tuple[int, int]]] = {}
    lanes_before_row: dict[int, list[int]] = {}
    hit_counts = [0] * source_count
    direction_lists: list[list[str]] = [[] for _ in range(source_count)]

    for row in rows:
        lanes_before_row[row] = list(lanes)
        row_segments: list[tuple[int, int]] = []
        if row in winner_dir_for_row:
            chosen_idx = winner_idx
            directions = [winner_dir_for_row[row]]
        elif row in decoy_dir_for_row:
            chosen_idx = decoy_idx
            directions = [decoy_dir_for_row[row]]
        else:
            chosen_idx = None
            directions = []

        if chosen_idx is not None:
            future_lanes = {lanes[idx] for idx in range(source_count) if idx != chosen_idx}
            placement = None
            lane = lanes[chosen_idx]
            for direction in directions:
                placement = _choose_segment_for_lane(
                    lane=lane,
                    direction=direction,
                    width=width,
                    current_lanes=lanes,
                    future_lanes=future_lanes,
                    existing_segments=row_segments,
                    rng=rng,
                )
                if placement is not None:
                    start, end, new_lane = placement
                    row_segments.append((start, end))
                    lanes[chosen_idx] = new_lane
                    hit_counts[chosen_idx] += 1
                    direction_lists[chosen_idx].append(direction)
                    break
            if placement is None:
                return None

        if not row_segments:
            decoy = _place_decoy_segment(width, lanes_before_row[row], row_segments, rng)
            if decoy is None:
                return None
            row_segments.append(decoy)

        extra_segments = 1 if rng.random() < 0.6 else 0
        for _ in range(extra_segments):
            decoy = _place_decoy_segment(width, lanes_before_row[row], row_segments, rng)
            if decoy is not None:
                row_segments.append(decoy)

        row_segments.sort()
        bars_by_row[row] = list(row_segments)

    if len(set(lanes)) != source_count:
        return None

    if hit_counts[winner_idx] < 2:
        return None
    if set(direction_lists[winner_idx]) != {"left", "right"}:
        return None
    if source_cols[winner_idx] == lanes[winner_idx]:
        return None
    if hit_counts[decoy_idx] == 0:
        return None

    target_col = lanes[winner_idx]

    source_colors = list(rng.choice(np.array(SOURCE_POOL, dtype=int), size=source_count, replace=False))
    if TARGET_COLOR not in source_colors and rng.random() < 0.7:
        source_colors[int(rng.integers(source_count))] = TARGET_COLOR
    if len(set(source_colors)) != source_count:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for row, segments in bars_by_row.items():
        for start, end in segments:
            grid[row, start : end + 1] = BAR_COLOR
    for col, color in zip(source_cols, source_colors):
        grid[0, col] = color
    grid[height - 1, target_col] = TARGET_COLOR

    return grid, winner_idx


def _semantic_checks(grid: np.ndarray, expected_winner: int) -> bool:
    info = _analyze_grid(grid)
    traces = info["traces"]
    winner_idx = int(info["winner_idx"])
    winner = traces[winner_idx]

    if winner_idx != expected_winner:
        return False
    if int(winner["source_col"]) == int(info["target_col"]):
        return False
    if int(winner["hit_count"]) < 2:
        return False
    if set(winner["directions"]) != {"left", "right"}:
        return False
    if len({int(trace["end_col"]) for trace in traces}) != len(traces):
        return False
    if sum(int(trace["hit_count"]) > 0 for idx, trace in enumerate(traces) if idx != winner_idx) == 0:
        return False

    bar_rows = sorted(info["bars_by_row"])
    if len(bar_rows) < 4:
        return False
    if any(bar_rows[idx + 1] - bar_rows[idx] < 2 for idx in range(len(bar_rows) - 1)):
        return False

    g = np.asarray(grid, dtype=int)
    if int(info["bar_color"]) != BAR_COLOR:
        return False
    for row, segments in info["bars_by_row"].items():
        for start, end in segments:
            if row <= 0 or row >= g.shape[0] - 1:
                return False
            if any(g[row, start : end + 1] != BAR_COLOR):
                return False
    for trace in traces:
        lane = int(trace["source_col"])
        seen_cells = set()
        for rr, cc in trace["cells"]:
            if g[rr, cc] == BAR_COLOR:
                return False
            seen_cells.add((int(rr), int(cc)))
        if len(seen_cells) != len(trace["cells"]):
            return False

        for direction, (row, start, end) in zip(trace["directions"], trace["hits"]):
            if not (start <= lane <= end):
                return False
            left_dist = lane - start
            right_dist = end - lane
            if left_dist == right_dist:
                return False
            nearer = "left" if left_dist < right_dist else "right"
            if direction != nearer:
                return False
            lane = start - 1 if direction == "left" else end + 1

    expected = _render_output(
        g.shape,
        info["bars_by_row"],
        traces[winner_idx],
        int(traces[winner_idx]["source_color"]),
        int(info["bg"]),
        int(info["bar_color"]),
    )
    return np.array_equal(main(grid), expected) and not np.array_equal(expected, g)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _construct_attempt(rng)
        if result is None:
            continue
        grid, winner_idx = result
        if _semantic_checks(grid, winner_idx):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
