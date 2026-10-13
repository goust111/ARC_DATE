from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
INPUT_COLORS = (1, 2, 3)
OUTPUT_HIGHLIGHT = 4
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (10, 15)
GRID_W_RANGE = (12, 16)
SEG_LEN_RANGE = (3, 8)
GROUP_COUNT_RANGE = (4, 7)
GROUP_MAX_HEIGHT = 4
GEN_ATTEMPTS = 400


def _extract_row_segment(row: np.ndarray) -> tuple[int, int, int]:
    cols = np.flatnonzero(row != BG)
    assert cols.size > 0, "each row must contain one foreground segment"

    start = int(cols[0])
    end = int(cols[-1])
    color = int(row[start])

    assert color in INPUT_COLORS, "unexpected segment color"
    assert cols.size == end - start + 1, "segment must be contiguous"
    assert np.all(row[start : end + 1] == color), "segment must be monochromatic"
    assert np.all(row[:start] == BG) and np.all(row[end + 1 :] == BG), "row must contain exactly one segment"
    return color, start, end


def _left_align_rows(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    info: list[tuple[int, int, int]] = []
    for r in range(g.shape[0]):
        color, start, end = _extract_row_segment(g[r])
        length = end - start + 1
        out[r, :length] = color
        info.append((color, start, length))
    return out, info


def _components_of_color(grid: np.ndarray, color: int) -> list[tuple[list[tuple[int, int]], set[int]]]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[list[tuple[int, int]], set[int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            rows: set[int] = set()
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                rows.add(rr)
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append((coords, rows))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Left-align every row segment, then recolor emergent multi-row monochrome components.

    Input structure:
    - The grid has black background cells and exactly one contiguous non-black horizontal segment in each row.
    - Each segment is monochromatic and uses one of the foreground colors {blue(1), red(2), green(3)}.
    - Segment lengths vary by row, and their starting columns may differ before alignment.

    Rule:
    - Shift each row's segment horizontally so its left end lands in column 0, preserving its color and length.
    - In that left-aligned grid, detect 4-connected components separately for each non-background color.
    - Recolor every component that spans at least two distinct rows to yellow(4).
    - Leave single-row components in their original color, and keep all remaining cells black(0).

    Key insight:
    - Left alignment changes only horizontal placement, but that new placement can make same-color segments in adjacent rows touch vertically; those newly formed multi-row components are exactly the regions that must be recolored.
    """
    aligned, _ = _left_align_rows(np.asarray(grid, dtype=int))
    out = aligned.copy()
    for color in sorted(int(c) for c in np.unique(aligned) if int(c) != BG):
        for coords, rows in _components_of_color(aligned, color):
            if len(rows) >= 2:
                for r, c in coords:
                    out[r, c] = OUTPUT_HIGHLIGHT
    return out


def _row_groups(row_colors: list[int]) -> list[dict]:
    groups: list[dict] = []
    start = 0
    while start < len(row_colors):
        end = start + 1
        while end < len(row_colors) and row_colors[end] == row_colors[start]:
            end += 1
        groups.append(
            {
                "color": int(row_colors[start]),
                "rows": list(range(start, end)),
            }
        )
        start = end
    return groups


def _intervals_overlap(start_a: int, len_a: int, start_b: int, len_b: int) -> bool:
    end_a = int(start_a + len_a - 1)
    end_b = int(start_b + len_b - 1)
    return not (end_a < start_b or end_b < start_a)


def _input_component_count(rows: list[int], starts: list[int], lengths: list[int]) -> int:
    count = 1
    for upper, lower in zip(rows, rows[1:]):
        if not _intervals_overlap(starts[upper], lengths[upper], starts[lower], lengths[lower]):
            count += 1
    return count


def _choose_group_lengths(height: int) -> list[int] | None:
    max_groups = min(GROUP_COUNT_RANGE[1], height)
    min_groups = min(GROUP_COUNT_RANGE[0], max_groups)
    for _ in range(120):
        n_groups = random.randint(min_groups, max_groups)
        lengths = [1] * n_groups
        remaining = height - n_groups
        while remaining > 0:
            idxs = [idx for idx, value in enumerate(lengths) if value < GROUP_MAX_HEIGHT]
            if not idxs:
                break
            choice = random.choice(idxs)
            lengths[choice] += 1
            remaining -= 1
        if remaining != 0:
            continue
        if sum(value >= 2 for value in lengths) >= 2 and any(value == 1 for value in lengths) and any(value >= 3 for value in lengths):
            return lengths
    return None


def _choose_group_colors(n_groups: int) -> list[int]:
    while True:
        colors = [random.choice(INPUT_COLORS)]
        for _ in range(n_groups - 1):
            choices = [color for color in INPUT_COLORS if color != colors[-1]]
            colors.append(random.choice(choices))
        if len(set(colors)) == len(INPUT_COLORS):
            return colors


def _force_emergent_group(rows: list[int], width: int, starts: list[int], lengths: list[int]) -> None:
    r0, r1 = rows[0], rows[1]
    max_len = min(5, width - 4)
    lengths[r0] = random.randint(3, max_len)
    lengths[r1] = random.randint(3, max_len)
    starts[r0] = 0
    starts[r1] = width - lengths[r1]

    for prev, curr in zip(rows[1:], rows[2:]):
        if random.random() < 0.5:
            starts[curr] = 0
        else:
            starts[curr] = width - lengths[curr]
        if not _intervals_overlap(starts[prev], lengths[prev], starts[curr], lengths[curr]) and random.random() < 0.35:
            overlap_min = max(0, starts[prev] - lengths[curr] + 1)
            overlap_max = min(width - lengths[curr], starts[prev] + lengths[prev] - 1)
            if overlap_min <= overlap_max:
                starts[curr] = random.randint(overlap_min, overlap_max)


def _force_connected_group(rows: list[int], width: int, starts: list[int], lengths: list[int]) -> None:
    anchor = random.randint(1, max(1, width - max(lengths[r] for r in rows) - 1))
    prev_start = anchor
    for idx, r in enumerate(rows):
        low = max(0, prev_start - lengths[r] + 1)
        high = min(width - lengths[r], prev_start + max(1, lengths[r] - 1))
        if idx == 0:
            starts[r] = min(anchor, width - lengths[r])
        else:
            starts[r] = random.randint(low, max(low, high))
        prev_start = starts[r]


def _build_candidate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    group_lengths = _choose_group_lengths(height)
    if group_lengths is None:
        return None
    group_colors = _choose_group_colors(len(group_lengths))

    row_colors: list[int] = []
    for color, count in zip(group_colors, group_lengths):
        row_colors.extend([int(color)] * int(count))

    groups = _row_groups(row_colors)
    multi_groups = [group for group in groups if len(group["rows"]) >= 2]
    if len(multi_groups) < 2:
        return None

    emergent_group = random.choice(multi_groups)
    connected_choices = [group for group in multi_groups if group is not emergent_group]
    connected_group = random.choice(connected_choices)

    lengths = [random.randint(*SEG_LEN_RANGE) for _ in range(height)]
    starts = [0] * height

    _force_emergent_group(emergent_group["rows"], width, starts, lengths)
    _force_connected_group(connected_group["rows"], width, starts, lengths)

    for group in groups:
        if group is emergent_group or group is connected_group:
            continue
        for r in group["rows"]:
            starts[r] = random.randint(0, width - lengths[r])

    if len(set(lengths)) < 3 or len(set(starts)) < 3:
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, color in enumerate(row_colors):
        start = starts[r]
        length = lengths[r]
        grid[r, start : start + length] = int(color)

    if not _validate_candidate(grid):
        return None
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False

        row_colors: list[int] = []
        starts: list[int] = []
        lengths: list[int] = []
        for r in range(g.shape[0]):
            color, start, end = _extract_row_segment(g[r])
            row_colors.append(color)
            starts.append(start)
            lengths.append(end - start + 1)

        groups = _row_groups(row_colors)
        multi_groups = [group for group in groups if len(group["rows"]) >= 2]
        single_groups = [group for group in groups if len(group["rows"]) == 1]
        if len(multi_groups) < 2 or not single_groups:
            return False

        if len(set(row_colors)) != len(INPUT_COLORS):
            return False
        if len(set(lengths)) < 3 or len(set(starts)) < 3:
            return False
        if not any(start == 0 for start in starts) or not any(start > 0 for start in starts):
            return False

        component_counts = [_input_component_count(group["rows"], starts, lengths) for group in multi_groups]
        if not any(count > 1 for count in component_counts):
            return False
        if not any(count == 1 for count in component_counts):
            return False

        aligned, _ = _left_align_rows(g)
        out = main(g)
        if out.shape != g.shape:
            return False

        yellow_components = _components_of_color(out, OUTPUT_HIGHLIGHT)
        if len(yellow_components) != len(multi_groups):
            return False

        yellow_row_sets = sorted(sorted(rows) for _coords, rows in yellow_components)
        expected_row_sets = sorted(sorted(group["rows"]) for group in multi_groups)
        if yellow_row_sets != expected_row_sets:
            return False

        for group in single_groups:
            r = group["rows"][0]
            length = lengths[r]
            color = row_colors[r]
            if not np.all(out[r, :length] == color):
                return False
            if np.any(out[r, length:] != BG):
                return False

        aligned_colors = {int(c) for c in np.unique(aligned)} - {BG}
        if aligned_colors != set(INPUT_COLORS):
            return False
        if OUTPUT_HIGHLIGHT not in np.unique(out):
            return False
        if not any(color in np.unique(out) for color in INPUT_COLORS):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid left-align emergent-connectivity puzzle input")
