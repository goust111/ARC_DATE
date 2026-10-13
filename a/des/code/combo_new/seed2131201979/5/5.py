from __future__ import annotations

import random
from collections import defaultdict
from typing import NamedTuple

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_ATTEMPTS = 400


class PathSpec(NamedTuple):
    color: int
    orientation: str
    fixed: int
    start: int
    end: int


def _path_cells(spec: PathSpec) -> list[tuple[int, int]]:
    if spec.orientation == "h":
        return [(spec.fixed, c) for c in range(spec.start, spec.end + 1)]
    return [(r, spec.fixed) for r in range(spec.start, spec.end + 1)]


def _extract_priority(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    legend_rows = np.flatnonzero(grid[:, 0] != BG)
    assert len(legend_rows) >= 2, "expected at least two legend colors in the left column"
    priority = [int(grid[r, 0]) for r in legend_rows]
    assert len(priority) == len(set(priority)), "legend colors must be distinct"
    return priority, {color: idx for idx, color in enumerate(priority)}


def _extract_paths(grid: np.ndarray, priority: list[int]) -> dict[int, PathSpec]:
    specs: dict[int, PathSpec] = {}
    path_colors = set()
    for color in priority:
        coords = np.argwhere((grid == color) & (np.indices(grid.shape)[1] != 0))
        assert len(coords) >= 2, f"path color {color} must have at least two visible cells"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1 and len(cols) >= 2:
            row = int(rows[0])
            specs[color] = PathSpec(color, "h", row, int(cols.min()), int(cols.max()))
        elif len(cols) == 1 and len(rows) >= 2:
            col = int(cols[0])
            specs[color] = PathSpec(color, "v", col, int(rows.min()), int(rows.max()))
        else:
            raise AssertionError(f"path color {color} is not a single straight segment")
        path_colors.add(color)

    visible_colors = {
        int(color)
        for color in np.unique(grid[:, 1:])
        if int(color) != BG
    }
    assert visible_colors == path_colors, "all non-legend colors must be path colors"
    return specs


def _build_cell_index(specs: dict[int, PathSpec]) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for spec in specs.values():
        for cell in _path_cells(spec):
            owners[cell].append(spec.color)

    for colors in owners.values():
        assert len(colors) <= 2, "only simple pairwise crossings are supported"
    for cell, colors in owners.items():
        if len(colors) == 2:
            a, b = (specs[colors[0]], specs[colors[1]])
            assert a.orientation != b.orientation, f"same-orientation overlap at {cell}"
    return owners


def _solve_from_specs(
    shape: tuple[int, int],
    specs: dict[int, PathSpec],
    priority_order: list[int],
    legend: np.ndarray,
) -> np.ndarray:
    priority_rank = {color: idx for idx, color in enumerate(priority_order)}
    owners = _build_cell_index(specs)

    out = np.full(shape, BG, dtype=int)
    out[:, 0] = legend
    for spec in specs.values():
        for r, c in _path_cells(spec):
            out[r, c] = spec.color

    higher_hits: dict[int, list[tuple[int, int]]] = {color: [] for color in specs}
    for cell, colors in owners.items():
        if len(colors) <= 1:
            continue
        winner = min(colors, key=lambda color: priority_rank[color])
        for color in colors:
            if color != winner:
                pos = cell[1] if specs[color].orientation == "h" else cell[0]
                higher_hits[color].append((pos, winner))

    for color, hits in higher_hits.items():
        if not hits:
            continue
        spec = specs[color]
        hits.sort(key=lambda item: item[0])
        first_pos, first_winner = hits[0]
        last_pos, last_winner = hits[-1]
        if spec.orientation == "h":
            out[spec.fixed, spec.start : first_pos + 1] = first_winner
            out[spec.fixed, last_pos : spec.end + 1] = last_winner
        else:
            out[spec.start : first_pos + 1, spec.fixed] = first_winner
            out[last_pos : spec.end + 1, spec.fixed] = last_winner

    for cell, colors in owners.items():
        if len(colors) > 1:
            out[cell] = min(colors, key=lambda color: priority_rank[color])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor straight path segments according to the left-column crossing priority order.

    Input structure:
    - The background is black, and the leftmost column contains one singleton pixel
      for each path color; their top-to-bottom order defines the priority order.
    - Every non-legend foreground color forms one 1-cell-thick straight horizontal
      or vertical path segment.
    - Paths may cross orthogonally, and the visible input color at a crossing may be
      either of the two path colors.

    Rule:
    - Read the priority order from the non-black pixels in column 0, with higher rows
      meaning higher priority.
    - Reconstruct each colored path as its full straight span between its visible
      endpoints, then identify every orthogonal crossing.
    - At each crossing, the highest-priority path color present there wins the crossing
      cell. For any lower-priority path, the endpoint-side outer segment cut off by its
      first/last higher-priority crossings is recolored to the winning higher color.
    - Keep the left-column priority legend unchanged.

    Key insight:
    - The left column is a priority legend rather than part of the path network, and
      crossings do not only change one pixel: higher-priority paths take over the
      lower-priority path's outer endpoint branches beyond the crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.max(g) <= 9 and np.min(g) >= 0
    assert np.all(g[:, 0] >= 0)

    priority, _priority_rank = _extract_priority(g)
    specs = _extract_paths(g, priority)
    return _solve_from_specs(g.shape, specs, priority, g[:, 0].copy())


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    for _ in range(100):
        picks = sorted(random.sample(range(lo, hi + 1), count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _choose_free_interval(lo: int, hi: int, min_len: int) -> tuple[int, int] | None:
    if hi - lo + 1 < min_len:
        return None
    length = random.randint(min_len, hi - lo + 1)
    start = random.randint(lo, hi - length + 1)
    return start, start + length - 1


def _horizontal_span(cols: list[int], width: int, cross_range: tuple[int, int] | None) -> tuple[int, int] | None:
    if cross_range is None:
        gaps: list[tuple[int, int]] = []
        left = 2
        for col in cols:
            if col - left >= 3:
                gaps.append((left, col - 1))
            left = col + 1
        if width - 2 - left + 1 >= 3:
            gaps.append((left, width - 2))
        if not gaps:
            return None
        lo, hi = random.choice(gaps)
        return _choose_free_interval(lo, hi, 3)

    left_idx, right_idx = cross_range
    left_limit = 2 if left_idx == 0 else cols[left_idx - 1] + 1
    right_limit = width - 2 if right_idx == len(cols) - 1 else cols[right_idx + 1] - 1
    start_hi = cols[left_idx] - 1
    end_lo = cols[right_idx] + 1
    if left_limit > start_hi or end_lo > right_limit:
        return None
    start = random.randint(left_limit, start_hi)
    end = random.randint(end_lo, right_limit)
    if end - start + 1 < 4:
        return None
    return start, end


def _vertical_span(rows: list[int], height: int, cross_range: tuple[int, int] | None) -> tuple[int, int] | None:
    if cross_range is None:
        gaps: list[tuple[int, int]] = []
        top = 1
        for row in rows:
            if row - top >= 3:
                gaps.append((top, row - 1))
            top = row + 1
        if height - 2 - top + 1 >= 3:
            gaps.append((top, height - 2))
        if not gaps:
            return None
        lo, hi = random.choice(gaps)
        return _choose_free_interval(lo, hi, 3)

    top_idx, bottom_idx = cross_range
    top_limit = 1 if top_idx == 0 else rows[top_idx - 1] + 1
    bottom_limit = height - 2 if bottom_idx == len(rows) - 1 else rows[bottom_idx + 1] - 1
    start_hi = rows[top_idx] - 1
    end_lo = rows[bottom_idx] + 1
    if top_limit > start_hi or end_lo > bottom_limit:
        return None
    start = random.randint(top_limit, start_hi)
    end = random.randint(end_lo, bottom_limit)
    if end - start + 1 < 4:
        return None
    return start, end


def _count_higher_hits(
    specs: dict[int, PathSpec],
    priority: list[int],
) -> dict[int, int]:
    ranks = {color: idx for idx, color in enumerate(priority)}
    owners = _build_cell_index(specs)
    counts = {color: 0 for color in specs}
    for colors in owners.values():
        if len(colors) <= 1:
            continue
        winner = min(colors, key=lambda color: ranks[color])
        for color in colors:
            if color != winner:
                counts[color] += 1
    return counts


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 24)
    width = random.randint(12, 24)

    h_count = random.randint(2, 4)
    v_count = random.randint(2, 4)
    path_count = h_count + v_count
    if path_count > len(COLOR_POOL):
        return None

    legend_start = random.randint(0, height - path_count - 1)
    rows = _sample_spaced_positions(1, height - 2, h_count, 2)
    cols = _sample_spaced_positions(2, width - 2, v_count, 2)
    if rows is None or cols is None:
        return None

    horizontal_specs: list[tuple[int, int, int]] = []
    vertical_specs: list[tuple[int, int, int]] = []

    for row in rows:
        mode = random.choice(["free", "cross", "cross", "cross"])
        if mode == "free":
            span = _horizontal_span(cols, width, None)
        else:
            left_idx = random.randint(0, v_count - 1)
            right_idx = random.randint(left_idx, min(v_count - 1, left_idx + 1))
            span = _horizontal_span(cols, width, (left_idx, right_idx))
        if span is None:
            return None
        horizontal_specs.append((row, span[0], span[1]))

    for col in cols:
        mode = random.choice(["free", "cross", "cross", "cross"])
        if mode == "free":
            span = _vertical_span(rows, height, None)
        else:
            top_idx = random.randint(0, h_count - 1)
            bottom_idx = random.randint(top_idx, min(h_count - 1, top_idx + 1))
            span = _vertical_span(rows, height, (top_idx, bottom_idx))
        if span is None:
            return None
        vertical_specs.append((col, span[0], span[1]))

    colors = random.sample(COLOR_POOL, path_count)
    random.shuffle(colors)
    h_colors = colors[:h_count]
    v_colors = colors[h_count:]

    specs: dict[int, PathSpec] = {}
    for color, (row, start, end) in zip(h_colors, horizontal_specs):
        specs[color] = PathSpec(color, "h", row, start, end)
    for color, (col, start, end) in zip(v_colors, vertical_specs):
        specs[color] = PathSpec(color, "v", col, start, end)

    owners = _build_cell_index(specs)
    degrees = {color: 0 for color in specs}
    for colors_here in owners.values():
        if len(colors_here) == 2:
            for color in colors_here:
                degrees[color] += 1

    if not any(degrees.values()):
        return None
    if max(degrees.values()) > 2:
        return None
    if any((spec.end - spec.start + 1) - degrees[spec.color] < 3 for spec in specs.values()):
        return None

    want_double = random.choice([False, True])
    candidate_priorities = [random.sample(list(specs.keys()), len(specs)) for _ in range(80)]
    chosen_priority = None
    chosen_counts = None
    for priority in candidate_priorities:
        higher_counts = _count_higher_hits(specs, priority)
        if not any(count > 0 for count in higher_counts.values()):
            continue
        if not any(count == 0 for count in higher_counts.values()):
            continue
        if want_double:
            if not any(count == 2 for count in higher_counts.values()):
                continue
        else:
            if any(count == 2 for count in higher_counts.values()):
                continue
            if not any(count == 1 for count in higher_counts.values()):
                continue
        chosen_priority = priority
        chosen_counts = higher_counts
        break

    if chosen_priority is None or chosen_counts is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for offset, color in enumerate(chosen_priority):
        grid[legend_start + offset, 0] = color

    for spec in specs.values():
        for r, c in _path_cells(spec):
            if c == 0:
                return None
            grid[r, c] = spec.color

    for cell, colors_here in owners.items():
        if len(colors_here) == 2:
            if random.random() < 0.5:
                ranks = {color: idx for idx, color in enumerate(chosen_priority)}
                grid[cell] = min(colors_here, key=lambda color: ranks[color])
            else:
                grid[cell] = random.choice(colors_here)

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_ATTEMPTS} attempts")
