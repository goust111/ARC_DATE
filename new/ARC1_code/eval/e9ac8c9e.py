from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangle into four equal quadrants using its four diagonal corner markers.

    Input structure
    - The background is black.
    - One or more solid gray rectangles appear on the grid.
    - Each gray rectangle has four non-background, non-gray marker pixels placed
      diagonally just outside its bounding-box corners.

    Rule
    - Remove the four marker pixels for each gray rectangle.
    - Split the rectangle in half horizontally and vertically.
    - Fill the top-left, top-right, bottom-left, and bottom-right quadrants with
      the colors of the corresponding diagonal markers.

    Key insight
    - The marker positions are a local color legend for the surrounded gray
      rectangle; the gray block is the output region and its corner markers name
      the colors of its quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK

    out = g.copy()
    gray_blocks = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
    )
    assert gray_blocks

    marker_positions: set[tuple[int, int]] = set()
    height, width = g.shape
    for block in gray_blocks:
        r0, c0, r1, c1 = block["bbox"]
        block_h = int(r1) - int(r0) + 1
        block_w = int(c1) - int(c0) + 1
        assert block_h % 2 == 0 and block_w % 2 == 0
        assert int(block["area"]) == block_h * block_w
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == Color.GREY)

        corners = [
            (int(r0) - 1, int(c0) - 1),
            (int(r0) - 1, int(c1) + 1),
            (int(r1) + 1, int(c0) - 1),
            (int(r1) + 1, int(c1) + 1),
        ]
        assert all(0 <= r < height and 0 <= c < width for r, c in corners)
        assert not (set(corners) & marker_positions)
        marker_positions.update(corners)

        marker_colors = [int(g[r, c]) for r, c in corners]
        assert all(color not in {bg, Color.GREY} for color in marker_colors)

        for r, c in corners:
            out[r, c] = bg

        r_mid = int(r0) + block_h // 2
        c_mid = int(c0) + block_w // 2
        out[r0:r_mid, c0:c_mid] = marker_colors[0]
        out[r0:r_mid, c_mid : c1 + 1] = marker_colors[1]
        out[r_mid : r1 + 1, c0:c_mid] = marker_colors[2]
        out[r_mid : r1 + 1, c_mid : c1 + 1] = marker_colors[3]

    non_gray_foreground = {
        (int(r), int(c)) for r, c in np.argwhere((g != bg) & (g != Color.GREY))
    }
    assert non_gray_foreground == marker_positions

    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle structure.
CANVAS_MIN_SIZE = 10
CANVAS_MAX_SIZE_EXCLUSIVE = 16
RECT_COUNT_MIN = 1
RECT_COUNT_MAX_EXCLUSIVE = 4
RECT_SIDE_OPTIONS = [2, 4, 6]

# Marker placement, colors, and retry budget.
MARKER_PADDING = 1
MARKER_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
BLACK, GREY = 0, 5
GENERATION_RETRIES = 300


def _gen_expanded_zone(r0: int, c0: int, side: int) -> tuple[slice, slice]:
    end = side + MARKER_PADDING
    return np.s_[r0 - MARKER_PADDING : r0 + end, c0 - MARKER_PADDING : c0 + end]


def _gen_valid_zone_positions(occupied: np.ndarray, side: int) -> list[tuple[int, int]]:
    height, width = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(MARKER_PADDING, height - side):
        for c0 in range(MARKER_PADDING, width - side):
            if not occupied[_gen_expanded_zone(r0, c0, side)].any():
                positions.append((r0, c0))
    return positions


def _gen_place_rectangles(
    height: int, width: int, count: int
) -> list[tuple[int, int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    rects: list[tuple[int, int, int]] = []

    def search(k: int) -> bool:
        if k == count:
            return True

        for side in np.random.permutation(RECT_SIDE_OPTIONS).astype(int).tolist():
            positions = _gen_valid_zone_positions(occupied, side)
            if not positions:
                continue
            for idx in np.random.permutation(len(positions)).astype(int).tolist():
                r0, c0 = positions[idx]
                zone = _gen_expanded_zone(r0, c0, side)
                occupied[zone] = True
                rects.append((r0, c0, side))
                if search(k + 1):
                    return True
                rects.pop()
                occupied[zone] = False
        return False

    return rects if search(0) else None


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(CANVAS_MIN_SIZE, CANVAS_MAX_SIZE_EXCLUSIVE))
    width = height
    count = int(np.random.randint(RECT_COUNT_MIN, RECT_COUNT_MAX_EXCLUSIVE))

    rects = _gen_place_rectangles(height, width, count)
    if rects is None:
        return None

    inp = np.full((height, width), BLACK, dtype=int)
    out = np.zeros_like(inp)

    for r0, c0, side in rects:
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        colors = np.random.choice(MARKER_PALETTE, size=4, replace=False).astype(int).tolist()

        inp[r0 : r1 + 1, c0 : c1 + 1] = GREY
        corners = [
            (r0 - MARKER_PADDING, c0 - MARKER_PADDING),
            (r0 - MARKER_PADDING, c1 + MARKER_PADDING),
            (r1 + MARKER_PADDING, c0 - MARKER_PADDING),
            (r1 + MARKER_PADDING, c1 + MARKER_PADDING),
        ]
        for (r, c), color in zip(corners, colors):
            inp[r, c] = color

        r_mid = r0 + side // 2
        c_mid = c0 + side // 2
        out[r0:r_mid, c0:c_mid] = colors[0]
        out[r0:r_mid, c_mid : c1 + 1] = colors[1]
        out[r_mid : r1 + 1, c0:c_mid] = colors[2]
        out[r_mid : r1 + 1, c_mid : c1 + 1] = colors[3]

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e9ac8c9e input")
