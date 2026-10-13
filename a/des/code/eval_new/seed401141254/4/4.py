from __future__ import annotations

import random

import numpy as np


def _outer_background(grid: np.ndarray) -> int:
    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0],
            grid[1:-1, -1],
        ]
    )
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_mono_columns(grid: np.ndarray, bg: int) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != bg:
            cols.append(int(c))
    return cols


def _full_mono_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r, :])
        if vals.size == 1 and int(vals[0]) != bg:
            rows.append(int(r))
    return rows


def _extract_horizontal_fences(
    grid: np.ndarray, bg: int
) -> list[tuple[int, int, int, int, int]] | None:
    h, w = grid.shape
    if h < 4 or w < 5:
        return None

    separators = _full_mono_columns(grid, bg)
    if len(separators) < 3 or separators[0] != 0 or separators[-1] != w - 1:
        return None

    fences: list[tuple[int, int, int, int, int]] = []
    for left, right in zip(separators, separators[1:], strict=False):
        if right - left <= 1:
            return None
        if np.any(grid[0, left : right + 1] == bg) or np.any(grid[h - 1, left : right + 1] == bg):
            return None
        interior = grid[1 : h - 1, left + 1 : right]
        if interior.size == 0:
            return None
        vals = np.unique(interior)
        if vals.size != 1:
            return None
        fences.append((1, left + 1, h - 2, right - 1, int(vals[0])))

    return fences if len(fences) >= 2 else None


def _extract_vertical_fences(
    grid: np.ndarray, bg: int
) -> list[tuple[int, int, int, int, int]] | None:
    h, w = grid.shape
    if h < 5 or w < 4:
        return None

    separators = _full_mono_rows(grid, bg)
    if len(separators) < 3 or separators[0] != 0 or separators[-1] != h - 1:
        return None

    fences: list[tuple[int, int, int, int, int]] = []
    for top, bottom in zip(separators, separators[1:], strict=False):
        if bottom - top <= 1:
            return None
        if np.any(grid[top : bottom + 1, 0] == bg) or np.any(grid[top : bottom + 1, w - 1] == bg):
            return None
        interior = grid[top + 1 : bottom, 1 : w - 1]
        if interior.size == 0:
            return None
        vals = np.unique(interior)
        if vals.size != 1:
            return None
        fences.append((top + 1, 1, bottom - 1, w - 2, int(vals[0])))

    return fences if len(fences) >= 2 else None


def _shift_fills(fills: list[int], bg: int) -> list[int]:
    result = list(fills)
    empty_positions = [i for i, color in enumerate(fills) if color == bg]
    empty_ptr = 0

    for i, color in enumerate(fills):
        if color == bg:
            continue
        while empty_ptr < len(empty_positions) and empty_positions[empty_ptr] <= i:
            empty_ptr += 1
        if empty_ptr >= len(empty_positions):
            continue
        target = empty_positions[empty_ptr]
        empty_ptr += 1
        result[target] = color
        result[i] = bg

    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each filled fenced region forward into the next empty fenced region in spatial order.

    Input structure:
    - The background is black.
    - A single strip of rectangular fenced regions appears either left-to-right or top-to-bottom.
    - Neighboring regions share one full border line, and each region interior is a solid color or black.

    Rule:
    - Detect whether the fenced regions are arranged horizontally or vertically, then read them in that order.
    - For each originally non-black interior, find the nearest later region whose interior is black; if an immediate neighbor is filled, skip it and continue forward.
    - Move the fill color into that empty region and turn the source interior black; if no later empty region exists, keep the color in place.
    - Fence borders themselves are unchanged.

    Key insight:
    The colored border strip encodes an ordered sequence of enclosed containers, so the task is a stable left-to-right or top-to-bottom assignment of filled interiors into later empty slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _outer_background(g)
    r0, c0, r1, c1 = _bbox(g != bg)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]

    horizontal = _extract_horizontal_fences(sub, bg)
    vertical = _extract_vertical_fences(sub, bg)

    if horizontal is not None and vertical is None:
        fences = horizontal
    elif vertical is not None and horizontal is None:
        fences = vertical
    else:
        assert horizontal is not None or vertical is not None, "could not detect the fenced strip"
        if horizontal is not None and vertical is not None:
            fences = horizontal if sub.shape[1] >= sub.shape[0] else vertical
        else:
            fences = horizontal if horizontal is not None else vertical

    fills = [color for _ir0, _ic0, _ir1, _ic1, color in fences]
    shifted = _shift_fills(fills, bg)

    out = g.copy()
    for (ir0, ic0, ir1, ic1, _old_color), new_color in zip(fences, shifted, strict=False):
        out[r0 + ir0 : r0 + ir1 + 1, c0 + ic0 : c0 + ic1 + 1] = int(new_color)
    return out


_BG = 0
_COLORS = list(range(1, 10))
_MAX_ATTEMPTS = 400


def _has_move(occupied: list[bool]) -> bool:
    return any(occupied[i] and any(not v for v in occupied[i + 1 :]) for i in range(len(occupied)))


def _has_skip(occupied: list[bool]) -> bool:
    return any(
        occupied[i] and occupied[i + 1] and any(not v for v in occupied[i + 2 :])
        for i in range(len(occupied) - 1)
    )


def _has_tail_stay(occupied: list[bool]) -> bool:
    last_empty = max((i for i, v in enumerate(occupied) if not v), default=-1)
    return any(occupied[i] for i in range(last_empty + 1, len(occupied)))


def _sample_occupancy(mode: int) -> list[bool] | None:
    min_count = 4 if mode == 3 else 3
    count = random.randint(min_count, 5)
    for _ in range(200):
        occupied = [bool(random.getrandbits(1)) for _ in range(count)]
        if all(occupied) or not any(occupied):
            continue
        if not _has_move(occupied):
            continue
        has_skip = _has_skip(occupied)
        has_tail = _has_tail_stay(occupied)
        if mode == 0 and not has_skip and not has_tail:
            return occupied
        if mode == 1 and has_skip and not has_tail:
            return occupied
        if mode == 2 and not has_skip and has_tail:
            return occupied
        if mode == 3 and has_skip and has_tail:
            return occupied
    return None


def _margin_pair(span: int) -> tuple[int, int]:
    free = 30 - span
    assert free >= 2, "span must leave room for a border margin"
    choices = [
        (before, after)
        for before in range(1, min(6, free - 1) + 1)
        for after in range(1, min(6, free - before) + 1)
    ]
    return random.choice(choices)


def _separator_positions(widths_or_heights: list[int]) -> list[int]:
    positions = [0]
    cur = 0
    for size in widths_or_heights:
        cur += size + 1
        positions.append(cur)
    return positions


def _render_horizontal(
    heights: int, widths: list[int], fence_colors: list[int], fills: list[int]
) -> np.ndarray:
    bbox_h = heights + 2
    bbox_w = sum(widths) + len(widths) + 1
    grid = np.full((bbox_h, bbox_w), _BG, dtype=int)
    separators = _separator_positions(widths)

    for i, width in enumerate(widths):
        left = separators[i]
        right = separators[i + 1]
        color = int(fence_colors[i])
        if i == 0:
            grid[:, left] = color
        grid[:, right] = color
        start = left if i == 0 else left + 1
        grid[0, start : right + 1] = color
        grid[-1, start : right + 1] = color
        grid[1:-1, left + 1 : right] = int(fills[i])

    return grid


def _render_vertical(
    widths: int, heights: list[int], fence_colors: list[int], fills: list[int]
) -> np.ndarray:
    bbox_h = sum(heights) + len(heights) + 1
    bbox_w = widths + 2
    grid = np.full((bbox_h, bbox_w), _BG, dtype=int)
    separators = _separator_positions(heights)

    for i, height in enumerate(heights):
        top = separators[i]
        bottom = separators[i + 1]
        color = int(fence_colors[i])
        if i == 0:
            grid[top, :] = color
        grid[bottom, :] = color
        start = top if i == 0 else top + 1
        grid[start : bottom + 1, 0] = color
        grid[start : bottom + 1, -1] = color
        grid[top + 1 : bottom, 1:-1] = int(fills[i])

    return grid


def _embed(
    structure: np.ndarray, top: int, bottom: int, left: int, right: int
) -> np.ndarray:
    out = np.full((top + structure.shape[0] + bottom, left + structure.shape[1] + right), _BG, dtype=int)
    out[top : top + structure.shape[0], left : left + structure.shape[1]] = structure
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    mode = random.randint(0, 3)
    occupied = _sample_occupancy(mode)
    if occupied is None:
        return None

    count = len(occupied)
    orientation = random.choice(["horizontal", "vertical"])
    fence_colors = random.sample(_COLORS, count)
    fill_pool = [color for color in _COLORS if color not in fence_colors]
    full_count = sum(occupied)
    if full_count > len(fill_pool):
        return None
    fill_colors = random.sample(fill_pool, full_count)
    random.shuffle(fill_colors)

    fills: list[int] = []
    fill_idx = 0
    for is_full in occupied:
        if is_full:
            fills.append(int(fill_colors[fill_idx]))
            fill_idx += 1
        else:
            fills.append(_BG)

    if orientation == "horizontal":
        inner_height = random.randint(2, 5)
        inner_widths = [random.randint(2, 4) for _ in range(count)]
        structure_in = _render_horizontal(inner_height, inner_widths, fence_colors, fills)
        structure_out = _render_horizontal(inner_height, inner_widths, fence_colors, _shift_fills(fills, _BG))
    else:
        inner_width = random.randint(2, 5)
        inner_heights = [random.randint(2, 4) for _ in range(count)]
        structure_in = _render_vertical(inner_width, inner_heights, fence_colors, fills)
        structure_out = _render_vertical(inner_width, inner_heights, fence_colors, _shift_fills(fills, _BG))

    top, bottom = _margin_pair(structure_in.shape[0])
    left, right = _margin_pair(structure_in.shape[1])
    inp = _embed(structure_in, top, bottom, left, right)
    out = _embed(structure_out, top, bottom, left, right)

    if inp.shape != out.shape:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if np.array_equal(inp, expected):
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid fenced-transfer task")
