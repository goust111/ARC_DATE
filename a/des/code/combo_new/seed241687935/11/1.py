from __future__ import annotations

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 300


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _collect_stripes(
    shape: tuple[int, int],
    vertical_specs: list[tuple[int, int]],
    horizontal_specs: list[tuple[int, int]],
) -> dict[tuple[int, int], set[int]]:
    h, w = shape
    painted: dict[tuple[int, int], set[int]] = {}

    for col, color in vertical_specs:
        for row in range(1, h - 1):
            painted.setdefault((row, int(col)), set()).add(int(color))

    for row, color in horizontal_specs:
        for col in range(1, w - 1):
            painted.setdefault((int(row), col), set()).add(int(color))

    return painted


def _paint_stripes(
    grid: np.ndarray,
    vertical_specs: list[tuple[int, int]],
    horizontal_specs: list[tuple[int, int]],
    cross_color: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    painted = _collect_stripes(out.shape, vertical_specs, horizontal_specs)
    for (row, col), colors in painted.items():
        out[row, col] = int(cross_color) if len(colors) >= 2 else int(next(iter(colors)))
    return out


def _extract_structure(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and w >= 5

    interior = g[1:-1, 1:-1]
    bg = _most_common(interior.ravel())

    marker_coords = np.argwhere(interior != bg)
    assert marker_coords.shape == (1, 2), "expected exactly one interior clue pixel"
    marker_r, marker_c = map(int, marker_coords[0] + 1)
    cross_color = int(g[marker_r, marker_c])

    perimeter = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    border_color = _most_common(perimeter)
    assert int(g[0, 0]) == border_color
    assert int(g[0, -1]) == border_color
    assert int(g[-1, 0]) == border_color
    assert int(g[-1, -1]) == border_color

    vertical_specs: list[tuple[int, int]] = []
    horizontal_specs: list[tuple[int, int]] = []

    for col in range(1, w - 1):
        top_color = int(g[0, col])
        bottom_color = int(g[h - 1, col])
        if top_color != border_color:
            assert top_color != bg
            vertical_specs.append((col, top_color))
        if bottom_color != border_color:
            assert bottom_color != bg
            vertical_specs.append((col, bottom_color))

    for row in range(1, h - 1):
        left_color = int(g[row, 0])
        right_color = int(g[row, w - 1])
        if left_color != border_color:
            assert left_color != bg
            horizontal_specs.append((row, left_color))
        if right_color != border_color:
            assert right_color != bg
            horizontal_specs.append((row, right_color))

    assert vertical_specs or horizontal_specs, "expected at least one border gap"
    return bg, border_color, cross_color, vertical_specs, horizontal_specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project colored stripes inward from border gaps and recolor stripe crossings with the interior marker color.

    Input structure:
    - The background is black.
    - The outermost ring is a one-pixel monochrome rectangular frame, except for several single-pixel border gaps recolored with other foreground colors.
    - The frame interior is otherwise black and contains exactly one extra non-black singleton pixel away from the border.

    Rule:
    - Treat each recolored border gap as a launcher: top or bottom gaps project a vertical stripe inward, and left or right gaps project a horizontal stripe inward.
    - Extend each stripe straight through the interior until the cell just before the opposite border, using the gap's own color.
    - If an interior cell is reached by stripes of two or more different colors, repaint that cell with the singleton marker's color.
    - Keep the frame and the border-gap pixels unchanged.

    Key insight:
    - The border gap location determines stripe direction, while the lone interior pixel does not mark a place to copy; it defines the universal color used only for stripe intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _, _, cross_color, vertical_specs, horizontal_specs = _extract_structure(g)
    return _paint_stripes(g, vertical_specs, horizontal_specs, cross_color)


def _render_input(
    h: int,
    w: int,
    border_color: int,
    cross_color: int,
    vertical_gaps: list[tuple[str, int, int]],
    horizontal_gaps: list[tuple[str, int, int]],
    marker: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = border_color
    grid[-1, :] = border_color
    grid[:, 0] = border_color
    grid[:, -1] = border_color

    vertical_specs: list[tuple[int, int]] = []
    horizontal_specs: list[tuple[int, int]] = []

    for side, col, color in vertical_gaps:
        row = 0 if side == "top" else h - 1
        grid[row, col] = int(color)
        vertical_specs.append((int(col), int(color)))

    for side, row, color in horizontal_gaps:
        col = 0 if side == "left" else w - 1
        grid[row, col] = int(color)
        horizontal_specs.append((int(row), int(color)))

    grid[marker] = int(cross_color)
    return grid, vertical_specs, horizontal_specs


def _sample_positions(rng: np.random.Generator, size: int, count: int) -> list[int]:
    return [int(v) for v in sorted(rng.choice(np.arange(1, size - 1), size=count, replace=False))]


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(8, 19))
    w = int(rng.integers(8, 19))
    cross_mode = bool(rng.integers(0, 2))

    if cross_mode:
        max_vertical = min(3, w - 3)
        max_horizontal = min(3, h - 3)
        if max_vertical < 1 or max_horizontal < 1:
            return None
        vertical_count = int(rng.integers(1, max_vertical + 1))
        horizontal_count = int(rng.integers(1, max_horizontal + 1))
    else:
        orientation = "vertical" if bool(rng.integers(0, 2)) else "horizontal"
        vertical_count = 0
        horizontal_count = 0
        if orientation == "vertical":
            max_vertical = min(4, w - 3)
            if max_vertical < 1:
                return None
            vertical_count = int(rng.integers(1, max_vertical + 1))
        else:
            max_horizontal = min(4, h - 3)
            if max_horizontal < 1:
                return None
            horizontal_count = int(rng.integers(1, max_horizontal + 1))

    total_gaps = vertical_count + horizontal_count
    palette = [int(v) for v in rng.choice(np.arange(1, 10), size=total_gaps + 2, replace=False)]
    border_color = palette[0]
    cross_color = palette[1]
    gap_colors = palette[2:]

    vertical_cols = _sample_positions(rng, w, vertical_count) if vertical_count else []
    horizontal_rows = _sample_positions(rng, h, horizontal_count) if horizontal_count else []

    vertical_gaps: list[tuple[str, int, int]] = []
    horizontal_gaps: list[tuple[str, int, int]] = []

    color_idx = 0
    for col in vertical_cols:
        side = "top" if bool(rng.integers(0, 2)) else "bottom"
        vertical_gaps.append((side, int(col), gap_colors[color_idx]))
        color_idx += 1

    for row in horizontal_rows:
        side = "left" if bool(rng.integers(0, 2)) else "right"
        horizontal_gaps.append((side, int(row), gap_colors[color_idx]))
        color_idx += 1

    free_rows = [row for row in range(1, h - 1) if row not in horizontal_rows]
    free_cols = [col for col in range(1, w - 1) if col not in vertical_cols]
    if not free_rows or not free_cols:
        return None
    marker = (int(rng.choice(free_rows)), int(rng.choice(free_cols)))

    inp, vertical_specs, horizontal_specs = _render_input(
        h,
        w,
        border_color,
        cross_color,
        vertical_gaps,
        horizontal_gaps,
        marker,
    )
    expected = _paint_stripes(inp, vertical_specs, horizontal_specs, cross_color)

    if cross_mode:
        if not np.any(expected[1:-1, 1:-1] == cross_color):
            return None
        intersection_count = int(np.count_nonzero(expected[1:-1, 1:-1] == cross_color)) - 1
        if intersection_count < vertical_count * horizontal_count:
            return None
    else:
        if int(np.count_nonzero(expected[1:-1, 1:-1] == cross_color)) != 1:
            return None

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
