from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    ranked = counts.most_common()
    assert ranked, "grid must be non-empty"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _wire_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    non_bg = [(count, color) for color, count in counts.items() if color != int(bg)]
    assert non_bg, "expected at least one non-background color"
    non_bg.sort(reverse=True)
    if len(non_bg) > 1:
        assert non_bg[0][0] > non_bg[1][0], "wire color must be the unique largest non-background color"
    return int(non_bg[0][1])


def _parse_legend(grid: np.ndarray, bg: int, grey: int) -> tuple[dict[frozenset[int], int], set[tuple[int, int]]]:
    h, w = grid.shape
    assert w >= 3, "legend requires at least three columns"

    mapping: dict[frozenset[int], int] = {}
    legend_cells: set[tuple[int, int]] = set()
    row = 0
    while row < h:
        triple = [int(v) for v in grid[row, :3]]
        if all(color not in (bg, grey) for color in triple):
            a, b, c = triple
            assert a != b, "legend mixes two distinct colors"
            key = frozenset((a, b))
            assert key not in mapping, "legend color pairs must be unique"
            mapping[key] = c
            for col in range(3):
                legend_cells.add((row, col))
            if row + 1 < h:
                assert np.all(grid[row + 1, :3] == bg), "legend rows must be separated by one background row"
            row += 2
            continue
        break

    assert mapping, "expected at least one legend entry in the top-left corner"
    return mapping, legend_cells


def _source_events(
    grid: np.ndarray,
    bg: int,
    grey: int,
    legend_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int, int, int, int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    events: list[tuple[int, int, int, int, int, int]] = []
    sources: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if (r, c) in legend_cells:
                continue
            color = int(grid[r, c])
            if color in (bg, grey):
                continue
            adjacent = [
                (rr, cc)
                for rr, cc in _neighbors4(r, c)
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(grey)
            ]
            assert len(adjacent) == 1, "each source must touch exactly one wire endpoint"
            rr, cc = adjacent[0]
            events.append((1, rr, cc, color, r, c))
            sources.append((r, c))

    assert sources, "expected at least one signal source"
    return events, sources


def _wire_degrees(wire_mask: np.ndarray) -> np.ndarray:
    h, w = wire_mask.shape
    degree = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(wire_mask):
        deg = 0
        for rr, cc in _neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and wire_mask[rr, cc]:
                deg += 1
        degree[int(r), int(c)] = deg
    return degree


def _propagate_colors(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    grey = _wire_color(g, bg)
    mix_map, legend_cells = _parse_legend(g, bg, grey)
    initial_events, sources = _source_events(g, bg, grey, legend_cells)

    wire_mask = g == int(grey)
    degree = _wire_degrees(wire_mask)
    arrival_time = np.full((h, w), -1, dtype=int)
    painted = np.full((h, w), -1, dtype=int)

    heap: list[tuple[int, int, int, int, int, int]] = []
    for event in initial_events:
        heapq.heappush(heap, event)

    while heap:
        time = heap[0][0]
        bucket: list[tuple[int, int, int, int, int, int]] = []
        while heap and heap[0][0] == time:
            bucket.append(heapq.heappop(heap))

        grouped: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
        for _t, r, c, color, pr, pc in bucket:
            if wire_mask[r, c]:
                grouped[(r, c)].append((int(color), (int(pr), int(pc))))

        for (r, c), proposals in grouped.items():
            if arrival_time[r, c] != -1:
                continue

            colors = {color for color, _prev in proposals}
            if len(colors) == 1:
                final_color = next(iter(colors))
            else:
                assert degree[r, c] >= 3, "simultaneous multi-color arrivals must happen at a node"
                assert len(colors) == 2, "nodes mix exactly two colors"
                key = frozenset(int(color) for color in colors)
                assert key in mix_map, "missing legend entry for simultaneous colors"
                final_color = int(mix_map[key])

            arrival_time[r, c] = int(time)
            painted[r, c] = int(final_color)

            blocked = {
                prev
                for _color, prev in proposals
                if 0 <= prev[0] < h and 0 <= prev[1] < w and wire_mask[prev]
            }
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if not wire_mask[rr, cc]:
                    continue
                if (rr, cc) in blocked:
                    continue
                if arrival_time[rr, cc] != -1:
                    continue
                heapq.heappush(heap, (time + 1, rr, cc, final_color, r, c))

    out = np.full_like(g, int(bg))
    visited = arrival_time != -1
    out[wire_mask & ~visited] = int(grey)
    out[visited] = painted[visited]
    return out, wire_mask, legend_cells, sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color a grey wire network by propagating source colors and mixing them at nodes
    according to the top-left legend.

    Input structure:
    - Background is the dominant color.
    - The largest non-background color draws a one-pixel orthogonal wire network.
    - The top-left corner contains 3x1 legend bars [A, B, C] on alternating rows,
      meaning colors A and B mix into C.
    - Other non-background, non-wire pixels are sources attached to wire endpoints.

    Rule:
    - Sources start simultaneously and advance one wire pixel per step.
    - A node reached from one direction keeps that color and emits it through all
      non-arrival directions.
    - A node reached simultaneously by two different colors mixes them using the
      legend and emits only the mixed color through the remaining directions.
    - The output removes the legend and source pixels, keeping only the colored
      wire network on the same canvas.
    """
    out, _wire_mask, _legend_cells, _sources = _propagate_colors(grid)
    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (20, 30)
BRANCH_LENGTH_PASSIVE = (2, 6)
BOTTOM_TAIL_RANGE = (2, 4)
TOP_GAP_RANGE = (2, 4)
NODE_GAP_RANGE = (2, 4)
OUTER_RETRIES = 400


def _sample_modes() -> list[str]:
    patterns = {
        2: [
            ("sync", "passive"),
            ("sync", "late"),
            ("sync", "sync"),
        ],
        3: [
            ("sync", "late", "passive"),
            ("sync", "sync", "passive"),
            ("sync", "late", "sync"),
            ("sync", "passive", "passive"),
        ],
        4: [
            ("sync", "late", "passive", "passive"),
            ("sync", "sync", "late", "passive"),
            ("sync", "sync", "passive", "passive"),
            ("sync", "late", "sync", "passive"),
        ],
    }
    size = random.choices([2, 3, 4], weights=[2, 5, 2])[0]
    return list(random.choice(patterns[size]))


def _choose_sides(lengths: list[int]) -> tuple[list[str], int, int, int] | None:
    candidates: list[tuple[int, int, int, float, list[str]]] = []
    n = len(lengths)
    for mask in range(1 << n):
        sides = ["left" if (mask >> idx) & 1 else "right" for idx in range(n)]
        left_max = max((lengths[idx] for idx, side in enumerate(sides) if side == "left"), default=0)
        right_max = max((lengths[idx] for idx, side in enumerate(sides) if side == "right"), default=0)
        min_width = max(GRID_W_RANGE[0], left_max + right_max + 5)
        if min_width > GRID_W_RANGE[1]:
            continue
        used_sides = len(set(sides))
        balance = abs(left_max - right_max)
        candidates.append((-used_sides, balance, min_width, random.random(), sides))

    if not candidates:
        return None

    candidates.sort()
    _score_a, _score_b, min_width, _tie, sides = random.choice(candidates[: min(6, len(candidates))])
    left_max = max((lengths[idx] for idx, side in enumerate(sides) if side == "left"), default=0)
    right_max = max((lengths[idx] for idx, side in enumerate(sides) if side == "right"), default=0)
    return sides, left_max, right_max, int(min_width)


def _attempt_generate() -> np.ndarray | None:
    modes = _sample_modes()
    sync_count = sum(mode == "sync" for mode in modes)
    legend_height = 2 * sync_count - 1
    top_wire_row = legend_height + random.randint(*TOP_GAP_RANGE)

    rows: list[int] = []
    row = top_wire_row
    for _mode in modes:
        row += random.randint(*NODE_GAP_RANGE)
        rows.append(int(row))

    bottom_row = rows[-1] + random.randint(*BOTTOM_TAIL_RANGE)
    height = bottom_row + random.randint(2, 5)
    if not (GRID_H_RANGE[0] <= height <= GRID_H_RANGE[1]):
        return None

    branch_lengths: list[int] = []
    trunk_time = 1
    prev_row = top_wire_row
    for row, mode in zip(rows, modes):
        trunk_time += row - prev_row
        if mode == "sync":
            branch_lengths.append(int(trunk_time))
        elif mode == "late":
            branch_lengths.append(int(trunk_time + random.choice((1, 3))))
        else:
            branch_lengths.append(random.randint(*BRANCH_LENGTH_PASSIVE))
        prev_row = row

    side_choice = _choose_sides(branch_lengths)
    if side_choice is None:
        return None
    sides, left_max, right_max, min_width = side_choice

    width = random.randint(min_width, GRID_W_RANGE[1])
    trunk_col_min = max(6, left_max + 4)
    trunk_col_max = min(width - 3, width - right_max - 1)
    if trunk_col_min > trunk_col_max:
        return None
    trunk_col = random.randint(trunk_col_min, trunk_col_max)

    bg, grey = random.sample(range(10), 2)
    extra_needed = 1 + sum(mode in ("sync", "late") for mode in modes) + sync_count
    palette = [color for color in range(10) if color not in (bg, grey)]
    if len(palette) < extra_needed:
        return None
    random.shuffle(palette)

    top_source_color = int(palette.pop())
    current_trunk_color = int(top_source_color)
    legend_entries: list[tuple[int, int, int]] = []
    branch_source_colors: list[int | None] = []
    for mode in modes:
        if mode == "sync":
            side_color = int(palette.pop())
            mix_color = int(palette.pop())
            legend_entries.append((int(current_trunk_color), int(side_color), int(mix_color)))
            branch_source_colors.append(int(side_color))
            current_trunk_color = int(mix_color)
        elif mode == "late":
            side_color = int(palette.pop())
            branch_source_colors.append(int(side_color))
        else:
            branch_source_colors.append(None)

    grid = np.full((height, width), int(bg), dtype=int)

    for idx, (a, b, c) in enumerate(legend_entries):
        row = 2 * idx
        grid[row, 0:3] = np.array([a, b, c], dtype=int)

    grid[top_wire_row:bottom_row + 1, trunk_col] = int(grey)
    grid[top_wire_row - 1, trunk_col] = int(top_source_color)

    source_cells = [(top_wire_row - 1, trunk_col)]
    for row, mode, length, side, source_color in zip(rows, modes, branch_lengths, sides, branch_source_colors):
        if side == "left":
            endpoint = trunk_col - length + 1
            if endpoint < 5:
                return None
            grid[row, endpoint:trunk_col + 1] = int(grey)
            if mode in ("sync", "late"):
                source_col = endpoint - 1
                if source_col < 4:
                    return None
                grid[row, source_col] = int(source_color)
                source_cells.append((row, source_col))
        else:
            endpoint = trunk_col + length - 1
            if endpoint > width - 2:
                return None
            grid[row, trunk_col:endpoint + 1] = int(grey)
            if mode in ("sync", "late"):
                source_col = endpoint + 1
                if source_col >= width:
                    return None
                grid[row, source_col] = int(source_color)
                source_cells.append((row, source_col))

    out = main(grid)
    wire_mask = grid == int(grey)
    if not np.all(out[wire_mask] != int(bg)):
        return None

    for row in range(0, 2 * len(legend_entries), 2):
        if not np.all(out[row, 0:3] == int(bg)):
            return None
    for r, c in source_cells:
        if int(out[r, c]) != int(bg):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wire-mixing puzzle")
