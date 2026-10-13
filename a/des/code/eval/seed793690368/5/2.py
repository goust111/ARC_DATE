from __future__ import annotations

import random

import numpy as np


FOLD_COLOR = 5
SEGMENT_COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]

HALF_DEPTH_BOUNDS = (4, 8)
COLOR_COUNT_BOUNDS = (3, 5)
SPAN_EXTRA_BOUNDS = (0, 4)
PERMUTATION_MODES = ("identity", "partial", "deranged")
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_fold_axis(grid: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = grid.shape

    row_candidates = [
        r
        for r in range(h)
        if 0 < r < h - 1
        and 2 * r + 1 == h
        and int(grid[r, 0]) != bg
        and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c
        for c in range(w)
        if 0 < c < w - 1
        and 2 * c + 1 == w
        and int(grid[0, c]) != bg
        and np.all(grid[:, c] == grid[0, c])
    ]

    assert len(row_candidates) + len(col_candidates) == 1, "expected exactly one centered fold axis"
    if row_candidates:
        return "row", int(row_candidates[0])
    return "col", int(col_candidates[0])


def _resolve_overlap(a: int, b: int, bg: int, counts: dict[int, int]) -> int:
    if a == bg:
        return b
    if b == bg or a == b:
        return a

    count_a = counts[int(a)]
    count_b = counts[int(b)]
    assert count_a != count_b, "different overlapping colors must have different global frequencies"
    return int(a) if count_a < count_b else int(b)


def _overlay(moving: np.ndarray, stationary: np.ndarray, bg: int, counts: dict[int, int]) -> np.ndarray:
    assert moving.shape == stationary.shape
    out = np.full(moving.shape, int(bg), dtype=int)
    for idx in np.ndindex(out.shape):
        out[idx] = _resolve_overlap(int(moving[idx]), int(stationary[idx]), int(bg), counts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the upper or left half onto the opposite half and merge overlapping colored segments.

    Input structure
    - The background is the most common color.
    - A single full grey row or full grey column sits at the exact center and acts as the fold axis.
    - Each half contains straight colored segments perpendicular to the axis, with one endpoint adjacent to it.

    Rule
    - Remove the fold axis.
    - Reflect the upper half downward or the left half rightward.
    - Overlay the reflected half onto the stationary half.
    - Background yields to color; same color stays unchanged; different colors resolve by global rarity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D grid with a central fold axis"

    bg = _most_common_color(g)
    axis_kind, axis_index = _find_fold_axis(g, bg)
    counts = {int(color): int(np.count_nonzero(g == color)) for color in np.unique(g)}

    if axis_kind == "row":
        moving = np.flipud(g[:axis_index, :])
        stationary = g[axis_index + 1 :, :]
    else:
        moving = np.fliplr(g[:, :axis_index])
        stationary = g[:, axis_index + 1 :]

    return _overlay(moving, stationary, bg, counts)


def _sample_spaced_positions(length: int, count: int) -> list[int]:
    assert length >= 2 * count - 1
    bases = sorted(random.sample(range(length - count + 1), count))
    return [int(base + i) for i, base in enumerate(bases)]


def _permute_order(order: list[int], mode: str) -> list[int]:
    if mode == "identity":
        return order.copy()

    for _ in range(200):
        candidate = order.copy()
        random.shuffle(candidate)
        fixed = sum(a == b for a, b in zip(order, candidate, strict=False))
        if mode == "partial" and 0 < fixed < len(order):
            return candidate
        if mode == "deranged" and fixed == 0:
            return candidate
    raise RuntimeError(f"failed to sample permutation mode {mode}")


def _sample_lengths(colors: list[int], half_depth: int) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    totals = random.sample(range(4, 2 * half_depth + 1), len(colors))
    random.shuffle(totals)

    first_lengths: dict[int, int] = {}
    second_lengths: dict[int, int] = {}
    total_counts: dict[int, int] = {}

    for color, total in zip(colors, totals, strict=False):
        lo = max(2, total - half_depth)
        hi = min(half_depth, total - 2)
        first = random.randint(lo, hi)
        second = total - first
        first_lengths[int(color)] = int(first)
        second_lengths[int(color)] = int(second)
        total_counts[int(color)] = int(total)

    return first_lengths, second_lengths, total_counts


def _render_input(
    axis_kind: str,
    half_depth: int,
    span: int,
    bg: int,
    first_by_pos: dict[int, int],
    second_by_pos: dict[int, int],
    first_lengths: dict[int, int],
    second_lengths: dict[int, int],
) -> np.ndarray:
    if axis_kind == "row":
        grid = np.full((2 * half_depth + 1, span), int(bg), dtype=int)
        mid = half_depth
        grid[mid, :] = FOLD_COLOR

        for pos, color in first_by_pos.items():
            length = first_lengths[int(color)]
            grid[mid - length : mid, int(pos)] = int(color)
        for pos, color in second_by_pos.items():
            length = second_lengths[int(color)]
            grid[mid + 1 : mid + 1 + length, int(pos)] = int(color)
        return grid

    grid = np.full((span, 2 * half_depth + 1), int(bg), dtype=int)
    mid = half_depth
    grid[:, mid] = FOLD_COLOR

    for pos, color in first_by_pos.items():
        length = first_lengths[int(color)]
        grid[int(pos), mid - length : mid] = int(color)
    for pos, color in second_by_pos.items():
        length = second_lengths[int(color)]
        grid[int(pos), mid + 1 : mid + 1 + length] = int(color)
    return grid


def _render_expected(
    axis_kind: str,
    half_depth: int,
    span: int,
    bg: int,
    positions: list[int],
    first_by_pos: dict[int, int],
    second_by_pos: dict[int, int],
    first_lengths: dict[int, int],
    second_lengths: dict[int, int],
    total_counts: dict[int, int],
) -> np.ndarray:
    if axis_kind == "row":
        out = np.full((half_depth, span), int(bg), dtype=int)
    else:
        out = np.full((span, half_depth), int(bg), dtype=int)

    for pos in positions:
        first_color = int(first_by_pos[int(pos)])
        second_color = int(second_by_pos[int(pos)])
        first_len = int(first_lengths[first_color])
        second_len = int(second_lengths[second_color])
        depth = max(first_len, second_len)

        for dist in range(depth):
            first_active = dist < first_len
            second_active = dist < second_len

            if first_active and second_active:
                color = _resolve_overlap(first_color, second_color, bg, total_counts)
            elif first_active:
                color = first_color
            else:
                color = second_color

            if axis_kind == "row":
                out[dist, int(pos)] = int(color)
            else:
                out[int(pos), dist] = int(color)

    return out


def _valid_generated_case(grid: np.ndarray, expected: np.ndarray, bg: int, colors: list[int]) -> bool:
    if _most_common_color(grid) != int(bg):
        return False

    present = set(map(int, np.unique(expected))) - {int(bg)}
    if present != set(map(int, colors)):
        return False

    try:
        got = main(grid)
    except AssertionError:
        return False
    return np.array_equal(got, expected)


def _gen_attempt() -> np.ndarray | None:
    axis_kind = random.choice(("row", "col"))
    half_depth = random.randint(*HALF_DEPTH_BOUNDS)
    color_count = random.randint(*COLOR_COUNT_BOUNDS)
    span = 2 * color_count - 1 + random.randint(*SPAN_EXTRA_BOUNDS)

    bg = random.choice(SEGMENT_COLOR_POOL)
    colors = random.sample([c for c in SEGMENT_COLOR_POOL if c != bg], color_count)
    positions = _sample_spaced_positions(span, color_count)

    first_order = colors.copy()
    random.shuffle(first_order)
    mode = random.choice(PERMUTATION_MODES)
    second_order = _permute_order(first_order, mode)

    first_lengths, second_lengths, total_counts = _sample_lengths(colors, half_depth)
    if all(first_lengths[c] == second_lengths[c] for c in colors):
        return None

    first_by_pos = {int(pos): int(color) for pos, color in zip(positions, first_order, strict=False)}
    second_by_pos = {int(pos): int(color) for pos, color in zip(positions, second_order, strict=False)}

    grid = _render_input(
        axis_kind,
        half_depth,
        span,
        bg,
        first_by_pos,
        second_by_pos,
        first_lengths,
        second_lengths,
    )
    expected = _render_expected(
        axis_kind,
        half_depth,
        span,
        bg,
        positions,
        first_by_pos,
        second_by_pos,
        first_lengths,
        second_lengths,
        total_counts,
    )

    if _valid_generated_case(grid, expected, bg, colors):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-overlay input")
