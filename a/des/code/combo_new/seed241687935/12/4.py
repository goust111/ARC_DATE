from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
GRID_SIDE_RANGE = (8, 24)
MAX_GENERATION_ATTEMPTS = 300


def _extract_segments(line: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous non-black runs on a border line, excluding the corners."""
    arr = np.asarray(line, dtype=int)
    assert arr.ndim == 1 and arr.size >= 3
    assert int(arr[0]) == BG and int(arr[-1]) == BG, "border corners must stay black"

    segments: list[tuple[int, int, int]] = []
    i = 1
    stop = arr.size - 1
    while i < stop:
        if int(arr[i]) == BG:
            i += 1
            continue

        color = int(arr[i])
        start = i
        while i < stop and int(arr[i]) == color:
            i += 1
        end = i - 1
        if i < stop:
            assert int(arr[i]) == BG, "segments must be separated by black pixels"
        segments.append((start, end, color))

    return segments


def _projection_from_segments(length: int, segments: list[tuple[int, int, int]]) -> np.ndarray:
    proj = np.full(length, BG, dtype=int)
    for start, end, color in segments:
        proj[start : end + 1] = int(color)
    return proj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the colored bands on the top and left borders inward and combine their crossings.

    Input structure:
    - The background is black.
    - The top border and the left border contain several contiguous non-black
      color bands, with black gaps between neighboring bands.
    - The interior starts entirely black, and the bottom and right borders stay black.

    Rule:
    - Every top-border band projects straight downward through the interior
      cells of its column range.
    - Every left-border band projects straight rightward through the interior
      cells of its row range.
    - For each interior cell: use the top color if only the top projects there;
      use the left color if only the left projects there; keep that color if
      both projections have the same color; otherwise paint the crossing grey.
      Cells reached by neither projection stay black.
    - Leave the full outer border unchanged.

    Key insight:
    - The grid is solved by decoding two independent one-dimensional border
      signals and then merging them cellwise, with grey marking only
      different-color intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3
    assert bool(np.all(g[-1, :] == BG)) and bool(np.all(g[:, -1] == BG))
    assert bool(np.all(g[1:-1, 1:-1] == BG)), "interior must start empty"

    top_segments = _extract_segments(g[0, :])
    left_segments = _extract_segments(g[:, 0])
    assert top_segments and left_segments, "expected source bands on both borders"

    top_proj = _projection_from_segments(w, top_segments)
    left_proj = _projection_from_segments(h, left_segments)

    out = g.copy()
    for r in range(1, h - 1):
        left_color = int(left_proj[r])
        for c in range(1, w - 1):
            top_color = int(top_proj[c])
            if top_color == BG and left_color == BG:
                continue
            if top_color == BG:
                out[r, c] = left_color
            elif left_color == BG:
                out[r, c] = top_color
            elif top_color == left_color:
                out[r, c] = top_color
            else:
                out[r, c] = GREY

    return out


def _positive_parts(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    values = [1] * parts
    for _ in range(total - parts):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _nonnegative_parts(total: int, parts: int) -> list[int]:
    assert total >= 0 and parts >= 1
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _sample_segments(line_len: int, count: int) -> list[tuple[int, int]]:
    assert line_len >= 2 * count, "need room for segments plus separating gaps"
    cover_total = random.randint(count, line_len - count)
    lengths = _positive_parts(cover_total, count)

    extra_gap_budget = line_len - cover_total - (count - 1)
    gaps = _nonnegative_parts(extra_gap_budget, count + 1)
    for idx in range(1, count):
        gaps[idx] += 1

    segments: list[tuple[int, int]] = []
    pos = gaps[0]
    for idx, length in enumerate(lengths):
        start = pos
        end = start + length - 1
        segments.append((start, end))
        pos = end + 1 + gaps[idx + 1]

    assert pos == line_len
    return segments


def _color_assignment(count: int, shared: int, distinct: int, palette: list[int]) -> list[int]:
    colors = [shared, distinct]
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _paint_top(grid: np.ndarray, segments: list[tuple[int, int]], colors: list[int]) -> None:
    for (start, end), color in zip(segments, colors, strict=True):
        grid[0, start + 1 : end + 2] = int(color)


def _paint_left(grid: np.ndarray, segments: list[tuple[int, int]], colors: list[int]) -> None:
    for (start, end), color in zip(segments, colors, strict=True):
        grid[start + 1 : end + 2, 0] = int(color)


def _branch_signature(grid: np.ndarray) -> set[str]:
    branches: set[str] = set()
    h, w = grid.shape
    for r in range(1, h - 1):
        left_color = int(grid[r, 0])
        for c in range(1, w - 1):
            top_color = int(grid[0, c])
            if top_color != BG and left_color != BG:
                branches.add("same" if top_color == left_color else "diff")
            elif top_color != BG:
                branches.add("top_only")
            elif left_color != BG:
                branches.add("left_only")
            else:
                branches.add("none")
    return branches


def _sample_count(line_len: int) -> int:
    choices = [2]
    if line_len >= 9:
        choices.append(3)
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    inner_h = h - 2
    inner_w = w - 2
    if inner_h < 4 or inner_w < 4:
        return None

    left_count = _sample_count(inner_h)
    top_count = _sample_count(inner_w)
    if inner_h < 2 * left_count or inner_w < 2 * top_count:
        return None

    palette_size = random.randint(3, 5)
    palette = random.sample(SOURCE_COLORS, palette_size)
    shared, top_other, left_other = palette[:3]

    top_segments = _sample_segments(inner_w, top_count)
    left_segments = _sample_segments(inner_h, left_count)
    top_colors = _color_assignment(top_count, shared, top_other, palette)
    left_colors = _color_assignment(left_count, shared, left_other, palette)

    grid = np.zeros((h, w), dtype=int)
    _paint_top(grid, top_segments, top_colors)
    _paint_left(grid, left_segments, left_colors)

    branches = _branch_signature(grid)
    required = {"top_only", "left_only", "same", "diff", "none"}
    if branches != required:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out[1:-1, 1:-1] == GREY):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
