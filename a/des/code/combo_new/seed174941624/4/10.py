import random

import numpy as np


BG = 0
INTERSECTION = 5
STRIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _non_bg_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != bg)
    assert coords.size, "expected a loom area in the body"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _extract_uniform_color(values, bg: int) -> int:
    palette = {int(v) for v in values if int(v) != bg}
    assert len(palette) == 1, f"expected one stripe color, got {palette}"
    return palette.pop()


def _priority_legend(first_col: np.ndarray, bg: int) -> tuple[list[int], dict[int, int]]:
    legend = [int(v) for v in first_col if int(v) != bg]
    assert legend, "expected a non-empty left-edge priority legend"
    assert len(legend) == len(set(legend)), "legend colors must be unique"
    return legend, {color: idx for idx, color in enumerate(legend)}


def _higher_lower(color_a: int, color_b: int, priority: dict[int, int]) -> tuple[int, int]:
    assert color_a in priority and color_b in priority, "every stripe color must appear in the legend"
    if priority[color_a] < priority[color_b]:
        return color_a, color_b
    return color_b, color_a


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill grey stripe crossings with an alternating woven over-under pattern.

    Input structure:
    - The background is black, and the leftmost column is a priority legend made of single colored pixels ordered from top to bottom.
    - The remaining area contains one rectangular loom region: some rows are solid horizontal stripes, some columns are solid vertical stripes, and every crossing is replaced by the same neutral grey color.
    - Extra cells inside the loom stay black unless they belong to one of the vertical or horizontal stripes.

    Rule:
    - Remove the leftmost legend column in the output.
    - In the loom region, recover each horizontal stripe color H from the non-vertical cells of its row and each vertical stripe color V from the non-horizontal cells of its column.
    - Use the legend order to decide which of H and V has higher priority, then scan each horizontal stripe from left to right: odd-numbered crossings show the higher-priority color and even-numbered crossings show the lower-priority color.
    - Preserve every non-crossing cell unchanged.

    Key insight:
    - The visible color at a crossing depends on two coupled cues: the left-edge legend gives pairwise color priority, while the left-to-right crossing index along each horizontal stripe flips which layer appears on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a legend column"

    bg = _most_common_color(g[:, 0])
    _legend, priority = _priority_legend(g[:, 0], bg)

    body = g[:, 1:].copy()
    r0, c0, r1, c1 = _non_bg_bbox(body, bg)
    loom = body[r0 : r1 + 1, c0 : c1 + 1]
    loom_h, loom_w = loom.shape

    full_rows = [r for r in range(loom_h) if np.all(loom[r] != bg)]
    full_cols = [c for c in range(loom_w) if np.all(loom[:, c] != bg)]
    assert full_rows and full_cols, "expected both horizontal and vertical stripes"
    assert len(full_rows) < loom_h and len(full_cols) < loom_w, "need gap rows and gap columns"

    full_row_set = set(full_rows)
    full_col_set = set(full_cols)

    for r in range(loom_h):
        occupied = {int(c) for c in np.flatnonzero(loom[r] != bg)}
        if r in full_row_set:
            assert len(occupied) == loom_w, "horizontal stripe rows must cover the whole loom width"
        else:
            assert occupied.issubset(full_col_set), "non-horizontal rows may only contain vertical stripes"

    for c in range(loom_w):
        occupied = {int(r) for r in np.flatnonzero(loom[:, c] != bg)}
        if c in full_col_set:
            assert len(occupied) == loom_h, "vertical stripe columns must cover the whole loom height"
        else:
            assert occupied.issubset(full_row_set), "non-vertical columns may only contain horizontal stripes"

    crossings = {int(loom[r, c]) for r in full_rows for c in full_cols}
    assert len(crossings) == 1, "all crossings should share the same neutral placeholder color"

    row_colors: dict[int, int] = {}
    for r in full_rows:
        row_colors[r] = _extract_uniform_color(
            [loom[r, c] for c in range(loom_w) if c not in full_col_set],
            bg,
        )

    col_colors: dict[int, int] = {}
    for c in full_cols:
        col_colors[c] = _extract_uniform_color(
            [loom[r, c] for r in range(loom_h) if r not in full_row_set],
            bg,
        )

    out = body.copy()
    for r in full_rows:
        h_color = row_colors[r]
        for index, c in enumerate(full_cols, start=1):
            v_color = col_colors[c]
            high, low = _higher_lower(h_color, v_color, priority)
            out[r0 + r, c0 + c] = high if index % 2 == 1 else low

    return out


def _sample_spaced_positions(length: int, count: int, rng: random.Random) -> list[int] | None:
    for _ in range(200):
        positions = sorted(rng.sample(range(length), count))
        if all(b - a >= 2 for a, b in zip(positions, positions[1:])):
            return positions
    return None


def _render_loom(
    loom_h: int,
    loom_w: int,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
) -> np.ndarray:
    loom = np.full((loom_h, loom_w), BG, dtype=int)
    row_set = set(row_positions)
    for row, color in zip(row_positions, row_colors):
        loom[row, :] = int(color)
    for col, color in zip(col_positions, col_colors):
        for row in range(loom_h):
            if row not in row_set:
                loom[row, col] = int(color)
    for row in row_positions:
        for col in col_positions:
            loom[row, col] = INTERSECTION
    return loom


def _weave_body(
    body: np.ndarray,
    top: int,
    left: int,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
    priority: dict[int, int],
) -> np.ndarray:
    out = body.copy()
    for row, h_color in zip(row_positions, row_colors):
        for index, (col, v_color) in enumerate(zip(col_positions, col_colors), start=1):
            high, low = _higher_lower(int(h_color), int(v_color), priority)
            out[top + row, left + col] = high if index % 2 == 1 else low
    return out


def _has_both_priority_directions(
    row_colors: list[int],
    col_colors: list[int],
    priority: dict[int, int],
) -> bool:
    row_above = any(priority[h] < priority[v] for h in row_colors for v in col_colors)
    col_above = any(priority[v] < priority[h] for h in row_colors for v in col_colors)
    return row_above and col_above


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(2, 4)
    n_cols = rng.randint(2, 4)
    total_colors = n_rows + n_cols
    if total_colors > len(STRIPE_COLORS):
        return None

    used_colors = rng.sample(list(STRIPE_COLORS), total_colors)
    rng.shuffle(used_colors)
    row_colors = used_colors[:n_rows]
    col_colors = used_colors[n_rows:]

    legend_order = used_colors[:]
    rng.shuffle(legend_order)
    priority = {color: idx for idx, color in enumerate(legend_order)}
    if not _has_both_priority_directions(row_colors, col_colors, priority):
        return None

    loom_h = rng.randint(2 * n_rows - 1, min(12, 2 * n_rows + 4))
    loom_w = rng.randint(2 * n_cols - 1, min(12, 2 * n_cols + 4))
    row_positions = _sample_spaced_positions(loom_h, n_rows, rng)
    col_positions = _sample_spaced_positions(loom_w, n_cols, rng)
    if row_positions is None or col_positions is None:
        return None

    top_margin = rng.randint(1, 4)
    bottom_margin = rng.randint(1, 4)
    left_gap = rng.randint(0, 3)
    right_gap = rng.randint(0, 3)

    grid_h = top_margin + loom_h + bottom_margin
    min_h = len(legend_order) + 2
    if grid_h < min_h:
        bottom_margin += min_h - grid_h
        grid_h = min_h

    body_w = left_gap + loom_w + right_gap
    grid_w = 1 + body_w
    if grid_h > MAX_GRID_SIZE or grid_w > MAX_GRID_SIZE:
        return None

    legend_rows = sorted(rng.sample(range(1, grid_h - 1), len(legend_order)))
    loom = _render_loom(loom_h, loom_w, row_positions, row_colors, col_positions, col_colors)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[top_margin : top_margin + loom_h, 1 + left_gap : 1 + left_gap + loom_w] = loom
    for row, color in zip(legend_rows, legend_order):
        grid[row, 0] = int(color)

    expected = _weave_body(
        grid[:, 1:],
        top_margin,
        left_gap,
        row_positions,
        row_colors,
        col_positions,
        col_colors,
        priority,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-stripe input")
