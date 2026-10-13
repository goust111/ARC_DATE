from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
RED = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

SPLITS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]
LENGTH_VALUES = [1, 2, 3, 4, 5]
PAD_RANGE = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _first_seen_colors(grid: np.ndarray) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for value in map(int, grid.reshape(-1)):
        if value in (BLACK, RED) or value in seen:
            continue
        seen.add(value)
        order.append(value)
    return order


def _extract_segments(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    segments: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, RED) or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r0 != r1 and c0 != c1:
                raise AssertionError("every subsegment must be horizontal or vertical")

            expected = (c1 - c0 + 1) if r0 == r1 else (r1 - r0 + 1)
            if len(coords) != expected:
                raise AssertionError("every subsegment must be a solid straight run")

            segments.append(
                {
                    "color": color,
                    "coords": coords,
                    "length": len(coords),
                    "first": min(coords),
                }
            )

    segments.sort(key=lambda seg: seg["first"])
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split colored paths at red intersections, then recolor singleton length ranks.

    The ranking is computed over distinct subsegment lengths in ascending order.
    If a length occurs on multiple subsegments, that whole tied group keeps its
    original color. Red intersection pixels stay red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = _first_seen_colors(g)
    segments = _extract_segments(g)
    assert colors, "expected at least one non-black, non-red line color"
    assert segments, "expected at least one subsegment"

    by_length: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        by_length[int(seg["length"])].append(seg)

    distinct_lengths = sorted(by_length)
    assert len(distinct_lengths) <= len(colors), "not enough visible colors for the length ranks"

    out = g.copy()
    for rank, length in enumerate(distinct_lengths, start=1):
        group = by_length[length]
        if len(group) != 1:
            continue
        target_color = int(colors[rank - 1])
        for r, c in group[0]["coords"]:
            out[r, c] = target_color

    return out


def _paint_horizontal(grid: np.ndarray, row: int, col0: int, col1: int, color: int) -> None:
    grid[row, col0 : col1 + 1] = color


def _paint_vertical(grid: np.ndarray, row0: int, row1: int, col: int, color: int) -> None:
    for r in range(row0, row1 + 1):
        grid[r, col] = RED if grid[r, col] != BLACK else color


def _build_candidate() -> np.ndarray:
    n_h, n_v = random.choice(SPLITS)
    line_count = n_h + n_v
    rank_count = random.randint(2, min(line_count, len(LENGTH_VALUES)))
    used_lengths = sorted(random.sample(LENGTH_VALUES, rank_count))
    unique_length = random.choice(used_lengths)
    shared_lengths = [value for value in used_lengths if value != unique_length]
    if not shared_lengths:
        raise ValueError("need at least one shared length")

    lefts = [random.choice(shared_lengths) for _ in range(n_h)]
    rights = [random.choice(shared_lengths) for _ in range(n_h)]
    tops = [random.choice(shared_lengths) for _ in range(n_v)]
    bottoms = [random.choice(shared_lengths) for _ in range(n_v)]
    row_seg_lengths = [random.choice(shared_lengths) for _ in range(n_h - 1)]
    col_seg_lengths = [random.choice(shared_lengths) for _ in range(n_v - 1)]

    endpoint_buckets = [lefts, rights, tops, bottoms]
    bucket = random.choice(endpoint_buckets)
    bucket[random.randrange(len(bucket))] = unique_length

    top_pad = random.randint(*PAD_RANGE)
    bottom_pad = random.randint(*PAD_RANGE)
    left_pad = random.randint(*PAD_RANGE)
    right_pad = random.randint(*PAD_RANGE)

    row0 = top_pad + max(tops)
    rows = [row0]
    for seg_len in row_seg_lengths:
        rows.append(rows[-1] + seg_len + 1)

    col0 = left_pad + max(lefts)
    cols = [col0]
    for seg_len in col_seg_lengths:
        cols.append(cols[-1] + seg_len + 1)

    height = rows[-1] + max(bottoms) + 1 + bottom_pad
    width = cols[-1] + max(rights) + 1 + right_pad

    grid = np.full((height, width), BLACK, dtype=int)
    palette = random.sample(LINE_COLORS, line_count)
    h_colors = palette[:n_h]
    v_colors = palette[n_h:]

    for i, row in enumerate(rows):
        start = cols[0] - lefts[i]
        end = cols[-1] + rights[i]
        _paint_horizontal(grid, row, start, end, h_colors[i])

    for j, col in enumerate(cols):
        start = rows[0] - tops[j]
        end = rows[-1] + bottoms[j]
        _paint_vertical(grid, start, end, col, v_colors[j])

    return grid


def _valid_candidate(grid: np.ndarray) -> bool:
    if np.count_nonzero(grid == RED) == 0:
        return False

    segments = _extract_segments(grid)
    if not segments:
        return False

    by_length: dict[int, int] = defaultdict(int)
    for seg in segments:
        by_length[int(seg["length"])] += 1

    if not any(count == 1 for count in by_length.values()):
        return False
    if not any(count > 1 for count in by_length.values()):
        return False
    if len(by_length) > len(_first_seen_colors(grid)):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _build_candidate()
        except ValueError:
            continue
        if _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
