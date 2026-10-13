from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _palette_order(grid: np.ndarray, bg: int) -> list[int]:
    palette: list[int] = []
    for value in map(int, grid[0]):
        if value == bg:
            continue
        assert value not in palette, "palette colors must be unique"
        palette.append(value)
    assert palette, "top row must contain a non-empty palette"
    return palette


def _find_row_pair(grid: np.ndarray, color: int) -> tuple[int, int]:
    rows = np.flatnonzero(grid[1:, 0] == int(color)) + 1
    assert rows.size == 2, "each palette color needs exactly two left-edge row markers"
    r0, r1 = map(int, rows)
    assert r1 - r0 >= 2, "row markers must bracket a non-empty interval"
    return r0, r1


def _find_col_pair(grid: np.ndarray, color: int) -> tuple[int, int]:
    cols = np.flatnonzero(grid[-1] == int(color))
    assert cols.size == 2, "each palette color needs exactly two bottom-edge column markers"
    c0, c1 = map(int, cols)
    assert c1 - c0 >= 2, "column markers must bracket a non-empty interval"
    return c0, c1


def _extract_window(grid: np.ndarray, color: int) -> np.ndarray:
    r0, r1 = _find_row_pair(grid, color)
    c0, c1 = _find_col_pair(grid, color)
    window = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert window.size > 0, "window must be non-empty"
    return window.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cross-match same-color row and column brackets, then concatenate the cropped windows in top-palette order.

    Input structure:
    - The top row is a palette row: its non-background cells list the relevant colors from left to right.
    - Below it is the main canvas on a dominant background.
    - For each palette color, the left edge column contains exactly two pixels of that color, and the bottom edge row also contains exactly two pixels of that color.
    - Extra edge-marker colors that do not appear in the top palette may also be present as distractors.

    Rule:
    - For each palette color, use its left-edge pair to bracket a row interval and its bottom-edge pair to bracket a column interval.
    - Take the intersection of those two intervals, excluding the marker cells themselves, as that color's window.
    - All palette windows have the same height; concatenate them horizontally in the palette's left-to-right order.
    - Ignore any complete marker pairs whose color is absent from the top palette.

    Key insight:
    - The crop is not explicitly drawn anywhere: one axis comes from the left edge, the other from the bottom edge, so the solver must match the two brackets by shared color and only then obey the palette row for output order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 4, "expected a 2D ARC grid"

    bg = _get_bg(g)
    palette = _palette_order(g, bg)
    windows = [_extract_window(g, color) for color in palette]

    heights = {int(window.shape[0]) for window in windows}
    assert len(heights) == 1, "palette windows must share one common height"
    return np.concatenate(windows, axis=1)


# --------------- generator ---------------

BG = 0
NON_BG = tuple(range(1, 10))
PALETTE_COUNT_RANGE = (3, 4)
DISTRACTOR_COUNT_OPTIONS = (1,)
WINDOW_HEIGHT_RANGE = (2, 4)
WINDOW_WIDTH_RANGE = (2, 4)
GAP_MAX = 1
MAX_ATTEMPTS = 400


def _bounded_composition(
    rng: np.random.Generator, n: int, total: int, lo: int, hi: int
) -> list[int] | None:
    if n <= 0:
        return [] if total == 0 else None
    if total < n * lo or total > n * hi:
        return None

    parts: list[int] = []
    remaining = int(total)
    for i in range(n):
        left = n - i - 1
        min_here = max(lo, remaining - hi * left)
        max_here = min(hi, remaining - lo * left)
        if min_here > max_here:
            return None
        value = int(rng.integers(min_here, max_here + 1))
        parts.append(value)
        remaining -= value
    return parts


def _sample_gaps(rng: np.random.Generator, count: int, budget: int) -> list[int]:
    if count <= 1:
        return []
    max_total = min((count - 1) * GAP_MAX, int(budget))
    total = int(rng.integers(0, max_total + 1))
    gaps = _bounded_composition(rng, count - 1, total, 0, GAP_MAX)
    assert gaps is not None
    return gaps


def _concat_in_order(windows: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    return np.concatenate([windows[color] for color in order], axis=1)


def _make_pattern(
    rng: np.random.Generator, h: int, w: int, fill_pool: list[int]
) -> np.ndarray:
    max_colors = min(3, len(fill_pool))
    for _ in range(120):
        chosen = [int(c) for c in rng.choice(fill_pool, size=int(rng.integers(2, max_colors + 1)), replace=False)]
        patch = np.full((h, w), BG, dtype=int)

        if bool(rng.integers(0, 2)):
            patch[int(rng.integers(0, h)), :] = chosen[0]
        else:
            patch[:, int(rng.integers(0, w))] = chosen[0]

        r0 = int(rng.integers(0, h))
        r1 = int(rng.integers(r0, h))
        c0 = int(rng.integers(0, w))
        c1 = int(rng.integers(c0, w))
        patch[r0 : r1 + 1, c0 : c1 + 1] = chosen[1]

        if len(chosen) >= 3:
            extra_count = int(rng.integers(1, min(3, h * w) + 1))
            idx = rng.choice(h * w, size=extra_count, replace=False)
            patch.reshape(-1)[idx] = chosen[2]

        if h * w >= 4:
            hole_count = int(rng.integers(1, min(2, h * w - 1) + 1))
            hole_idx = rng.choice(h * w, size=hole_count, replace=False)
            patch.reshape(-1)[hole_idx] = BG

        non_bg = patch[patch != BG]
        if non_bg.size < max(2, h):
            continue
        if len(set(map(int, non_bg))) < 2:
            continue
        if len(set(map(int, np.unique(patch)))) < 3:
            continue
        return patch
    raise RuntimeError("failed to generate a non-degenerate window pattern")


def _sample_orders(
    rng: np.random.Generator, palette_colors: list[int], all_colors: list[int]
) -> tuple[list[int], list[int], list[int]] | None:
    palette_set = set(palette_colors)
    for _ in range(200):
        palette_order = [int(c) for c in rng.permutation(palette_colors)]
        row_order = [int(c) for c in rng.permutation(all_colors)]
        col_order = [int(c) for c in rng.permutation(all_colors)]
        row_palette = [color for color in row_order if color in palette_set]
        col_palette = [color for color in col_order if color in palette_set]
        if palette_order == row_palette or palette_order == col_palette:
            continue
        if row_palette == col_palette:
            continue
        return palette_order, row_order, col_order
    return None


def _layout_positions(
    order: list[int], inner_sizes: dict[int, int], gaps: list[int], start: int = 1
) -> dict[int, int]:
    pos: dict[int, int] = {}
    cursor = int(start)
    for i, color in enumerate(order):
        pos[int(color)] = cursor
        cursor += inner_sizes[int(color)] + 2
        if i < len(gaps):
            cursor += int(gaps[i])
    return pos


def _validate_example(
    grid: np.ndarray,
    windows: dict[int, np.ndarray],
    palette_order: list[int],
    row_order: list[int],
    col_order: list[int],
    palette_colors: list[int],
    distractor_colors: list[int],
) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False

    expected = _concat_in_order(windows, palette_order)
    if not np.array_equal(solved, expected):
        return False

    extracted = {color: _extract_window(grid, color) for color in palette_colors}
    if any(not np.array_equal(extracted[color], windows[color]) for color in palette_colors):
        return False

    row_palette = [color for color in row_order if color in set(palette_colors)]
    col_palette = [color for color in col_order if color in set(palette_colors)]
    if np.array_equal(expected, _concat_in_order(windows, row_palette)):
        return False
    if np.array_equal(expected, _concat_in_order(windows, col_palette)):
        return False

    for perm in itertools.permutations(palette_order):
        if list(perm) == palette_order:
            continue
        if np.array_equal(expected, _concat_in_order(windows, list(perm))):
            return False

    signatures = {(windows[color].shape, windows[color].tobytes()) for color in palette_colors}
    if len(signatures) != len(palette_colors):
        return False

    palette_set = set(palette_colors)
    for color in distractor_colors:
        if color in palette_set:
            return False
        try:
            _ = _extract_window(grid, color)
        except AssertionError:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    palette_count = int(rng.integers(PALETTE_COUNT_RANGE[0], PALETTE_COUNT_RANGE[1] + 1))
    distractor_count = int(rng.choice(DISTRACTOR_COUNT_OPTIONS))
    total_count = palette_count + distractor_count

    max_h = min(
        WINDOW_HEIGHT_RANGE[1],
        (28 // total_count) - 2,
    )
    if max_h < WINDOW_HEIGHT_RANGE[0]:
        return None
    window_h = int(rng.integers(WINDOW_HEIGHT_RANGE[0], max_h + 1))

    row_gap_budget = 28 - total_count * (window_h + 2)
    if row_gap_budget < 0:
        return None
    row_gaps = _sample_gaps(rng, total_count, row_gap_budget)

    col_gap_budget = 29 - total_count * (WINDOW_WIDTH_RANGE[0] + 2)
    if col_gap_budget < 0:
        return None
    col_gaps = _sample_gaps(rng, total_count, col_gap_budget)

    max_total_width = 29 - total_count * 2 - sum(col_gaps)
    width_lo = total_count * WINDOW_WIDTH_RANGE[0]
    width_hi = min(total_count * WINDOW_WIDTH_RANGE[1], max_total_width)
    if width_hi < width_lo:
        return None
    width_total = int(rng.integers(width_lo, width_hi + 1))
    width_parts = _bounded_composition(
        rng,
        total_count,
        width_total,
        WINDOW_WIDTH_RANGE[0],
        WINDOW_WIDTH_RANGE[1],
    )
    if width_parts is None:
        return None

    marker_colors = [int(c) for c in rng.choice(NON_BG, size=total_count, replace=False)]
    palette_colors = marker_colors[:palette_count]
    distractor_colors = marker_colors[palette_count:]

    sampled = _sample_orders(rng, palette_colors, marker_colors)
    if sampled is None:
        return None
    palette_order, row_order, col_order = sampled

    widths = {color: width for color, width in zip(col_order, width_parts)}
    row_starts = _layout_positions(row_order, {color: window_h for color in marker_colors}, row_gaps)
    col_starts = _layout_positions(col_order, widths, col_gaps)

    height = 2 + total_count * (window_h + 2) + sum(row_gaps)
    width = 1 + sum(widths[color] + 2 for color in col_order) + sum(col_gaps)
    if height > 30 or width > 30:
        return None

    fill_pool = [color for color in NON_BG if color not in marker_colors]
    signatures: set[tuple[tuple[int, int], bytes]] = set()
    windows: dict[int, np.ndarray] = {}
    for color in marker_colors:
        local_w = widths[color]
        for _ in range(80):
            patch = _make_pattern(rng, window_h, local_w, fill_pool)
            signature = (patch.shape, patch.tobytes())
            if signature not in signatures:
                signatures.add(signature)
                windows[color] = patch
                break
        else:
            return None

    grid = np.full((height, width), BG, dtype=int)

    palette_span = 2 * palette_count - 1 if width >= 2 * palette_count + 1 else palette_count
    palette_start_max = width - palette_span
    palette_start = int(rng.integers(1, palette_start_max + 1))
    for i, color in enumerate(palette_order):
        c = palette_start + (2 * i if palette_span > palette_count else i)
        grid[0, c] = int(color)

    for color in marker_colors:
        r0 = row_starts[color]
        c0 = col_starts[color]
        local_w = widths[color]
        grid[r0, 0] = color
        grid[r0 + window_h + 1, 0] = color
        grid[-1, c0] = color
        grid[-1, c0 + local_w + 1] = color
        grid[r0 + 1 : r0 + 1 + window_h, c0 + 1 : c0 + 1 + local_w] = windows[color]

    if not _validate_example(
        grid,
        windows,
        palette_order,
        row_order,
        col_order,
        palette_colors,
        distractor_colors,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-bracket palette-order puzzle")
