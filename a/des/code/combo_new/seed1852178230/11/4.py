import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_OPTIONS = (3, 5)
GEN_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def rotate_offsets(offsets, turns: int):
    turns = int(turns) % 4
    out = {(int(dr), int(dc)) for dr, dc in offsets}
    for _ in range(turns):
        out = {(-dc, dr) for dr, dc in out}
    return frozenset(out)


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]):
            candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected exactly one full-width separator row"
    return candidates[0]


def _find_canvas_ranges(lower: np.ndarray, divider_color: int) -> tuple[list[tuple[int, int]], list[int]]:
    sep_cols = [c for c in range(lower.shape[1]) if np.all(lower[:, c] == divider_color)]
    assert len(sep_cols) == 3, "expected three vertical separator columns"
    ranges = [
        (0, sep_cols[0]),
        (sep_cols[0] + 1, sep_cols[1]),
        (sep_cols[1] + 1, sep_cols[2]),
        (sep_cols[2] + 1, lower.shape[1]),
    ]
    return ranges, sep_cols


def _extract_color_offsets(square: np.ndarray, bg: int, divider_color: int) -> dict[int, frozenset[tuple[int, int]]]:
    n = square.shape[0]
    center = n // 2
    assert square.shape[0] == square.shape[1] and n % 2 == 1
    assert int(square[center, center]) == divider_color

    color_offsets: dict[int, frozenset[tuple[int, int]]] = {}
    for color in sorted(int(c) for c in np.unique(square)):
        if color in (bg, divider_color):
            continue
        coords = np.argwhere(square == color)
        assert coords.size > 0
        offsets = frozenset((int(r - center), int(c - center)) for r, c in coords)
        assert offsets
        color_offsets[color] = offsets

    assert len(color_offsets) == 4, "expected exactly four encoded colors"
    first_offsets = next(iter(color_offsets.values()))
    rotated_family = {rotate_offsets(first_offsets, k) for k in range(4)}
    assert rotated_family == set(color_offsets.values()), "encoded colors must be quarter-turn copies"
    return color_offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a four-way rotational color code and render the requested rotated shape in each target canvas.

    Input structure:
    - The background is black. A single full-width grey row splits the grid into an upper code panel and a lower strip of four canvases.
    - The upper panel has odd height and contains one odd-sized square code window whose center cell is grey; four non-grey colors occupy quarter-turn-related pixel sets around that center.
    - The lower panel has the same height as the code window side length and consists of four equal square canvases separated by grey columns; each canvas is otherwise black except for a one-pixel color marker at its top-left corner.

    Rule:
    - Read the colored pixels in the upper square as four rotated versions of one hidden shape, keyed by color.
    - For each lower canvas, use its top-left marker color to select the matching rotated pixel set from the upper code window.
    - Output only the lower strip: keep the grey separator columns, clear the marker pixels, place the selected color pattern centered in each canvas, and paint the grey center cell of every canvas.

    Key insight:
    - The four colors in the upper panel do not denote four different objects; they are the same offset pattern under 90-degree rotations, so the marker color simply selects which rotated offset set must be copied into a blank target canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    sep_r, divider_color = _find_separator_row(g)
    top = g[:sep_r, :]
    lower = g[sep_r + 1 :, :]
    assert top.shape[0] == lower.shape[0] and top.shape[0] % 2 == 1

    n = int(lower.shape[0])
    radius = n // 2
    canvas_ranges, sep_cols = _find_canvas_ranges(lower, divider_color)
    assert all((c1 - c0) == n for c0, c1 in canvas_ranges), "lower canvases must be square"

    center_positions = np.argwhere(top == divider_color)
    assert center_positions.shape[0] == 1, "expected one grey rotation center in the upper panel"
    cr, cc = map(int, center_positions[0])
    assert cr == radius
    assert radius <= cc < g.shape[1] - radius
    assert np.all(top[:, : cc - radius] == bg)
    assert np.all(top[:, cc + radius + 1 :] == bg)

    square = top[:, cc - radius : cc + radius + 1]
    assert square.shape == (n, n)
    color_offsets = _extract_color_offsets(square, bg, divider_color)

    out = np.full(lower.shape, bg, dtype=int)
    for sep_c in sep_cols:
        out[:, sep_c] = divider_color

    for c0, c1 in canvas_ranges:
        canvas = lower[:, c0:c1]
        marker_color = int(canvas[0, 0])
        assert marker_color not in (bg, divider_color)
        assert marker_color in color_offsets

        blank_check = canvas.copy()
        blank_check[0, 0] = bg
        assert np.all(blank_check == bg), "input canvases should contain only the marker pixel"

        out_canvas = out[:, c0:c1]
        out_canvas[radius, radius] = divider_color
        for dr, dc in color_offsets[marker_color]:
            rr = radius + int(dr)
            cc2 = radius + int(dc)
            assert 0 <= rr < n and 0 <= cc2 < n
            assert out_canvas[rr, cc2] in (bg, marker_color)
            out_canvas[rr, cc2] = marker_color

    return out


def _canonical_positions(radius: int) -> list[tuple[int, int]]:
    pts = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if dr < 0 and dc >= 0:
                pts.append((dr, dc))
    return pts


def _neighbors8(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _random_base_offsets(side: int) -> frozenset[tuple[int, int]]:
    radius = side // 2
    pts = _canonical_positions(radius)
    assert len(pts) >= 2
    size_lo = 1 if side == 3 else 2
    size_hi = min(len(pts), 3 if side == 3 else 6)

    for _ in range(80):
        target = random.randint(size_lo, size_hi)
        shape = {random.choice(pts)}
        while len(shape) < target:
            frontier = [p for p in pts if p not in shape and any(_neighbors8(p, q) for q in shape)]
            if not frontier:
                break
            shape.add(random.choice(frontier))
        if len(shape) < size_lo:
            continue
        return frozenset(shape)

    raise RuntimeError("failed to build a non-degenerate base shape")


def _build_input(
    side: int,
    base_offsets: frozenset[tuple[int, int]],
    color_to_turn: dict[int, int],
    marker_order: list[int],
    center_col: int,
) -> np.ndarray:
    width = 4 * side + 3
    height = 2 * side + 1
    radius = side // 2

    grid = np.full((height, width), BG, dtype=int)
    grid[side, :] = DIVIDER

    for color, turns in color_to_turn.items():
        for dr, dc in rotate_offsets(base_offsets, turns):
            rr = radius + int(dr)
            cc = center_col + int(dc)
            assert 0 <= rr < side and 0 <= cc < width
            assert grid[rr, cc] == BG
            grid[rr, cc] = int(color)
    grid[radius, center_col] = DIVIDER

    sep_cols = [side, 2 * side + 1, 3 * side + 2]
    lower = grid[side + 1 :, :]
    for c in sep_cols:
        lower[:, c] = DIVIDER

    starts = [0, side + 1, 2 * (side + 1), 3 * (side + 1)]
    for color, c0 in zip(marker_order, starts):
        lower[0, c0] = int(color)

    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDE_OPTIONS)
    width = 4 * side + 3
    radius = side // 2

    colors = random.sample(PALETTE, 4)
    turns = [0, 1, 2, 3]
    random.shuffle(turns)
    color_to_turn = {color: turn for color, turn in zip(colors, turns)}
    marker_order = random.sample(colors, len(colors))
    center_col = random.randint(radius, width - radius - 1)
    base_offsets = _random_base_offsets(side)

    grid = _build_input(side, base_offsets, color_to_turn, marker_order, center_col)
    try:
        out = main(grid)
    except AssertionError:
        return None

    expected_shape = (side, width)
    if out.shape != expected_shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
