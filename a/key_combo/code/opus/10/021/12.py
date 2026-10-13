from __future__ import annotations

import random

import numpy as np


BG = 0
ROLE_COLORS = (1, 2, 3, 4)
GRID_RANGE = (12, 18)
RECT_HEIGHT_RANGE = (6, 11)
RECT_WIDTH_RANGE = (6, 11)
MAX_GENERATION_ATTEMPTS = 400


def _role_controls(grid: np.ndarray) -> dict[str, tuple[int, int, int]]:
    controls = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != BG)]
    assert len(controls) == 4, "expected exactly four control pixels"
    assert len({color for _, _, color in controls}) == 4, "control colors must be distinct"

    top_row = min(r for r, _, _ in controls)
    bottom_row = max(r for r, _, _ in controls)
    left_col = min(c for _, c, _ in controls)
    right_col = max(c for _, c, _ in controls)

    top = [control for control in controls if control[0] == top_row]
    bottom = [control for control in controls if control[0] == bottom_row]
    left = [control for control in controls if control[1] == left_col]
    right = [control for control in controls if control[1] == right_col]

    assert len(top) == len(bottom) == len(left) == len(right) == 1, "extrema must be unique"
    roles = {
        "top": top[0],
        "bottom": bottom[0],
        "left": left[0],
        "right": right[0],
    }
    assert len({(r, c) for r, c, _ in roles.values()}) == 4, "controls must occupy distinct cells"

    r_top, c_top, _ = roles["top"]
    r_bottom, c_bottom, _ = roles["bottom"]
    r_left, c_left, _ = roles["left"]
    r_right, c_right, _ = roles["right"]

    assert r_top < r_left < r_bottom, "left control must lie strictly between top and bottom"
    assert r_top < r_right < r_bottom, "right control must lie strictly between top and bottom"
    assert c_left < c_top < c_right, "top control must lie strictly between left and right"
    assert c_left < c_bottom < c_right, "bottom control must lie strictly between left and right"
    assert r_bottom - r_top - 1 >= 4, "rectangle interior must be at least 4 cells tall"
    assert c_right - c_left - 1 >= 4, "rectangle interior must be at least 4 cells wide"
    return roles


def _paint_rectangle(
    shape: tuple[int, int],
    r_top: int,
    r_bottom: int,
    c_left: int,
    c_right: int,
    color_a: int,
    color_b: int,
    color_c: int,
    color_d: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r in range(r_top, r_bottom + 1):
        for c in range(c_left, c_right + 1):
            parity_even = (r + c) % 2 == 0
            if r in (r_top, r_bottom) or c in (c_left, c_right):
                out[r, c] = color_c if parity_even else color_d
            else:
                out[r, c] = color_a if parity_even else color_b
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a parity-colored rectangle from four control pixels and fill its interior with a checkerboard.

    Input structure:
    - A black background contains exactly four single-pixel controls and no other foreground cells.
    - The controls use four distinct colors and have unique topmost, bottommost, leftmost, and rightmost positions.
    - None of the controls sits on a rectangle corner, and the rectangle interior is at least 4x4.

    Rule:
    - Use the topmost control's color as checkerboard color A and the leftmost control's color as checkerboard color B.
    - Use the bottommost control's color as border color C and the rightmost control's color as border color D.
    - Let the four extreme control positions define rows r_top/r_bottom and columns c_left/c_right, then draw that full rectangle.
    - Color border cells by parity: even (r+c) gets color C, odd gets color D; color interior cells by the same parity rule using colors A/B.

    Key insight:
    - The same four control pixels supply two different kinds of information: their extreme positions define the rectangle geometry, while their directional roles split their colors into the interior checkerboard pair and the border pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    roles = _role_controls(g)

    r_top, _, color_a = roles["top"]
    r_bottom, _, color_c = roles["bottom"]
    _, c_left, color_b = roles["left"]
    _, c_right, color_d = roles["right"]

    return _paint_rectangle(g.shape, r_top, r_bottom, c_left, c_right, color_a, color_b, color_c, color_d)


def _canonical_controls(
    r_top: int,
    r_bottom: int,
    c_left: int,
    c_right: int,
    colors: tuple[int, int, int, int],
) -> list[tuple[int, int, int]]:
    color_top, color_bottom, color_left, color_right = colors
    return [
        (r_top, c_left + 1, color_top),
        (r_bottom, c_right - 1, color_bottom),
        (r_top + 1, c_left, color_left),
        (r_bottom - 1, c_right, color_right),
    ]


def _reconstruct_generator_input(output: np.ndarray) -> np.ndarray:
    coords = np.argwhere(output != BG)
    assert len(coords) > 0, "output must contain a rectangle"
    r_top = int(coords[:, 0].min())
    r_bottom = int(coords[:, 0].max())
    c_left = int(coords[:, 1].min())
    c_right = int(coords[:, 1].max())

    color_top = int(output[r_top + 1, c_left + 1]) if (r_top + 1 + c_left + 1) % 2 == 0 else int(output[r_top + 1, c_left + 2])
    color_left = int(output[r_top + 1, c_left + 1]) if (r_top + 1 + c_left + 1) % 2 == 1 else int(output[r_top + 1, c_left + 2])
    color_bottom = int(output[r_top, c_left]) if (r_top + c_left) % 2 == 0 else int(output[r_top, c_left + 1])
    color_right = int(output[r_top, c_left]) if (r_top + c_left) % 2 == 1 else int(output[r_top, c_left + 1])

    reconstructed = np.zeros_like(output)
    for r, c, color in _canonical_controls(
        r_top,
        r_bottom,
        c_left,
        c_right,
        (color_top, color_bottom, color_left, color_right),
    ):
        reconstructed[r, c] = color
    return reconstructed


def _semantic_checks(inp: np.ndarray) -> None:
    roles = _role_controls(inp)
    out = main(inp)

    r_top, _, color_top = roles["top"]
    r_bottom, _, color_bottom = roles["bottom"]
    _, c_left, color_left = roles["left"]
    _, c_right, color_right = roles["right"]

    expected = _paint_rectangle(
        inp.shape,
        r_top,
        r_bottom,
        c_left,
        c_right,
        color_top,
        color_left,
        color_bottom,
        color_right,
    )
    assert np.array_equal(out, expected), "main must follow the puzzle semantics"

    for role_name, (r, c, role_color) in roles.items():
        assert out[r, c] != BG, f"{role_name} control position must be overwritten by the output pattern"
        assert np.any(out == role_color), f"{role_name} color must visibly contribute to the output"

    inside = out[r_top + 1 : r_bottom, c_left + 1 : c_right]
    assert inside.shape[0] >= 4 and inside.shape[1] >= 4
    assert set(map(int, np.unique(inside))) == {color_top, color_left}, "interior must use the top/left colors only"

    boundary_mask = np.zeros_like(out, dtype=bool)
    boundary_mask[r_top, c_left : c_right + 1] = True
    boundary_mask[r_bottom, c_left : c_right + 1] = True
    boundary_mask[r_top : r_bottom + 1, c_left] = True
    boundary_mask[r_top : r_bottom + 1, c_right] = True
    boundary_colors = set(map(int, np.unique(out[boundary_mask])))
    assert boundary_colors == {color_bottom, color_right}, "border must use the bottom/right colors only"

    reconstructed = _reconstruct_generator_input(out)
    assert np.array_equal(reconstructed, inp), "generator inputs must be uniquely recoverable from the output"


def _sample_rectangle(rows: int, cols: int) -> tuple[int, int, int, int] | None:
    max_h = min(RECT_HEIGHT_RANGE[1], rows - 2)
    max_w = min(RECT_WIDTH_RANGE[1], cols - 2)
    if max_h < RECT_HEIGHT_RANGE[0] or max_w < RECT_WIDTH_RANGE[0]:
        return None

    rect_h = random.randint(RECT_HEIGHT_RANGE[0], max_h)
    rect_w = random.randint(RECT_WIDTH_RANGE[0], max_w)

    row_margin_limit = rows - rect_h - 1
    col_margin_limit = cols - rect_w - 1
    if row_margin_limit < 1 or col_margin_limit < 1:
        return None

    r_top = random.randint(1, row_margin_limit)
    c_left = random.randint(1, col_margin_limit)
    r_bottom = r_top + rect_h - 1
    c_right = c_left + rect_w - 1
    return r_top, r_bottom, c_left, c_right


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(*GRID_RANGE)
    cols = random.randint(*GRID_RANGE)
    rect = _sample_rectangle(rows, cols)
    if rect is None:
        return None
    r_top, r_bottom, c_left, c_right = rect

    chosen = random.sample(ROLE_COLORS, 4)
    inp = np.zeros((rows, cols), dtype=int)
    for r, c, color in _canonical_controls(r_top, r_bottom, c_left, c_right, tuple(chosen)):
        inp[r, c] = color

    try:
        _semantic_checks(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 12")
