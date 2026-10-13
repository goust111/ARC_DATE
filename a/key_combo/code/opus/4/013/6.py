from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

LEGEND_BY_QUADRANT = {
    "tl": BLUE,
    "tr": RED,
    "bl": YELLOW,
    "br": GREEN,
}

GEN_RETRY_LIMIT = 400


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(map(int, np.asarray(grid).reshape(-1))))


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in _colors_present(grid):
        full_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(full_rows) != 2 or len(full_cols) != 2:
            continue
        top, bottom = map(int, full_rows)
        left, right = map(int, full_cols)
        if bottom - top > 1 and right - left > 1:
            candidates.append((int(color), top, bottom, left, right))
    assert len(candidates) == 1, "expected exactly one framed interior"
    return candidates[0]


def _legend_colors(grid: np.ndarray, top: int, bottom: int, left: int, right: int) -> dict[str, int]:
    h, w = map(int, grid.shape)
    coords = {
        "tl": (top - 1, left - 1),
        "tr": (top - 1, right + 1),
        "bl": (bottom + 1, left - 1),
        "br": (bottom + 1, right + 1),
    }
    out: dict[str, int] = {}
    for key, (r, c) in coords.items():
        assert 0 <= r < h and 0 <= c < w, "legend pixel must sit outside the frame corner"
        out[key] = int(grid[r, c])
    return out


def _quadrant_key(r: int, c: int, h: int, w: int) -> str:
    top_half = int(r) < h // 2
    left_half = int(c) < w // 2
    if top_half and left_half:
        return "tl"
    if top_half and not left_half:
        return "tr"
    if not top_half and left_half:
        return "bl"
    return "br"


def _bridge_sprite(left_color: int, right_color: int) -> np.ndarray:
    return np.array(
        [
            [left_color, left_color, right_color],
            [left_color, BLACK, right_color],
            [left_color, left_color, right_color],
        ],
        dtype=int,
    )


def _paint_up_wedge(out: np.ndarray, rect_top: int, center_c: int, color: int) -> None:
    d = 0
    r = rect_top - 1
    while r >= 0:
        half_span = 1 + d
        c0 = max(0, center_c - half_span)
        c1 = min(out.shape[1] - 1, center_c + half_span)
        out[r, c0 : c1 + 1] = int(color)
        d += 1
        r -= 1


def _paint_down_wedge(out: np.ndarray, rect_bottom: int, center_c: int, color: int) -> None:
    d = 0
    r = rect_bottom + 1
    while r < out.shape[0]:
        width = 3 - 2 * d
        if width <= 0:
            break
        half_span = width // 2
        c0 = max(0, center_c - half_span)
        c1 = min(out.shape[1] - 1, center_c + half_span)
        out[r, c0 : c1 + 1] = int(color)
        d += 1
        r += 1


def _build_output_from_spec(
    interior_shape: tuple[int, int],
    row: int,
    center_c: int,
    half_gap: int,
    left_color: int,
    right_color: int,
) -> np.ndarray:
    h, w = map(int, interior_shape)
    out = np.full((h, w), BLACK, dtype=int)

    left_c = int(center_c - half_gap)
    right_c = int(center_c + half_gap)
    r = int(row)
    out[r, left_c] = int(left_color)
    out[r, right_c] = int(right_color)

    if left_c + 1 <= center_c - 2:
        out[r, left_c + 1 : center_c - 1] = int(left_color)
    if center_c + 2 <= right_c - 1:
        out[r, center_c + 2 : right_c] = int(right_color)

    rect = _bridge_sprite(left_color, right_color)
    out[r - 1 : r + 2, center_c - 1 : center_c + 2] = rect

    _paint_up_wedge(out, r - 1, center_c, left_color)
    _paint_down_wedge(out, r + 1, center_c, right_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the framed interior, recolor the two orange anchors by quadrant legend, then bridge them with split-color wedges.

    Input structure:
    - A black background contains one same-color rectangular frame encoded by two full rows and two full columns.
    - The four outside-corner pixels around that frame are legend colors for the four interior quadrants.
    - Inside the frame are exactly two orange singleton pixels on one row, one in the left half and one in the right half.

    Rule:
    - Crop away the frame and keep only its interior.
    - Split the interior into four equal quadrants and recolor each orange singleton with the legend color of its quadrant.
    - Center a 3x3 hollow rectangle between the two recolored pixels, using the left pixel's color on the rectangle's left side and the right pixel's color on the rectangle's right side.
    - Connect each pixel horizontally to the rectangle, fill an expanding upward wedge in the left color above it, and fill a shrinking downward wedge in the right color below it.

    Key insight:
    - The quadrant legend does not merely repaint the two anchors: those two visible quadrant-derived colors also determine the bridge split and the two wedge fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, top, bottom, left, right = _find_frame(g)
    legends = _legend_colors(g, top, bottom, left, right)
    interior = g[top + 1 : bottom, left + 1 : right].copy()
    h, w = map(int, interior.shape)
    assert h > 0 and w > 0, "frame must enclose a nonempty interior"
    assert h % 2 == 0 and w % 2 == 0, "interior must split into equal quadrants"

    orange_coords = np.argwhere(interior == ORANGE)
    assert orange_coords.shape == (2, 2), "expected exactly two orange anchors"
    rows = np.unique(orange_coords[:, 0])
    assert rows.size == 1, "anchors must lie on the same row"

    anchor_info: list[tuple[int, int, int]] = []
    for r, c in orange_coords:
        rr, cc = int(r), int(c)
        color = legends[_quadrant_key(rr, cc, h, w)]
        anchor_info.append((rr, cc, int(color)))

    anchor_info.sort(key=lambda item: item[1])
    left_r, left_c, left_color = anchor_info[0]
    right_r, right_c, right_color = anchor_info[1]
    assert left_r == right_r
    assert left_c < w // 2 <= right_c, "anchors must occupy different left/right quadrants"

    distance = right_c - left_c
    assert distance >= 6 and distance % 2 == 0, "anchors must leave centered space for the bridge"
    center_c = (left_c + right_c) // 2
    assert 1 <= left_r <= h - 2, "bridge needs one-row clearance inside the crop"

    out = interior
    out[out == ORANGE] = BLACK
    out[left_r, left_c] = int(left_color)
    out[right_r, right_c] = int(right_color)

    if left_c + 1 <= center_c - 2:
        out[left_r, left_c + 1 : center_c - 1] = int(left_color)
    if center_c + 2 <= right_c - 1:
        out[left_r, center_c + 2 : right_c] = int(right_color)

    out[left_r - 1 : left_r + 2, center_c - 1 : center_c + 2] = _bridge_sprite(left_color, right_color)
    _paint_up_wedge(out, left_r - 1, center_c, left_color)
    _paint_down_wedge(out, left_r + 1, center_c, right_color)

    # Preserve the frame only as a detection aid; it should not survive the crop.
    assert frame_color not in out, "output must be the cropped interior only"
    return out


def _choose_even(options: list[int], *, low: int) -> int:
    valid = [value for value in options if value >= low]
    assert valid, "no valid even choice"
    return int(random.choice(valid))


def _branch_for_call() -> str:
    call_index = int(getattr(generate_input, "_call_index", 0))
    generate_input._call_index = call_index + 1
    return "top" if call_index % 2 == 0 else "bottom"


def _generate_case(branch: str) -> tuple[np.ndarray, np.ndarray]:
    interior_h = _choose_even([10, 12, 14, 16, 18, 20], low=10)

    if branch == "top":
        row = random.randint(2, interior_h // 2 - 1)
    else:
        row_lo = interior_h // 2
        row_hi = min(interior_h - 4, interior_h // 2 + 3)
        row = random.randint(row_lo, row_hi)

    interior_w = _choose_even([10, 12, 14, 16, 18, 20, 22, 24, 26], low=max(10, 2 * row))
    center_options = [interior_w // 2 - 1, interior_w // 2]
    center_options = [c for c in center_options if row - 1 <= c <= interior_w - row]
    assert center_options, "wedge needs horizontal clearance"
    center_c = int(random.choice(center_options))

    max_gap = min(center_c, interior_w - 1 - center_c)
    gap_choices = [gap for gap in range(3, max_gap + 1)]
    assert gap_choices, "need enough span for the bridge"
    half_gap = int(random.choice(gap_choices))

    left_c = center_c - half_gap
    right_c = center_c + half_gap
    assert left_c < interior_w // 2 <= right_c

    top_half = branch == "top"
    left_color = BLUE if top_half else YELLOW
    right_color = RED if top_half else GREEN

    inp = np.full((interior_h + 4, interior_w + 4), BLACK, dtype=int)
    inp[1, :] = GREY
    inp[-2, :] = GREY
    inp[:, 1] = GREY
    inp[:, -2] = GREY
    inp[0, 0] = BLUE
    inp[0, -1] = RED
    inp[-1, 0] = YELLOW
    inp[-1, -1] = GREEN
    inp[2 + row, 2 + left_c] = ORANGE
    inp[2 + row, 2 + right_c] = ORANGE

    expected = _build_output_from_spec(
        (interior_h, interior_w),
        row,
        center_c,
        half_gap,
        left_color,
        right_color,
    )
    return inp, expected


def generate_input() -> np.ndarray:
    branch = _branch_for_call()
    for _ in range(GEN_RETRY_LIMIT):
        inp, expected = _generate_case(branch)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
