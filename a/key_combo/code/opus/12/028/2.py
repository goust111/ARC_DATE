from __future__ import annotations

import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
_GEN_COUNTER = 0


def _locate_marker(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert 10 <= h <= 30 and 10 <= w <= 30

    coords = np.argwhere(g != BLACK)
    assert int(coords.shape[0]) == 4, "expected exactly one 2x2 foreground marker block"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 == r0 + 1 and c1 == c0 + 1, "foreground cells must form a 2x2 block"

    block = g[r0 : r0 + 2, c0 : c0 + 2]
    assert bool(np.all(block != BLACK))
    assert len(set(map(int, block.ravel()))) == 4, "marker colors must be distinct"

    top = int(r0)
    left = int(c0)
    bottom = int(h - (r0 + 2))
    right = int(w - (c0 + 2))

    return {
        "grid": g,
        "shape": (h, w),
        "block_top_left": (r0, c0),
        "block": block.copy(),
        "margins": {"top": top, "bottom": bottom, "left": left, "right": right},
        "quadrants": [
            {
                "name": "tl",
                "rows": (0, r0),
                "cols": (0, c0),
                "height": top,
                "width": left,
                "color": int(block[1, 1]),
                "near_row": "bottom",
                "near_col": "right",
            },
            {
                "name": "tr",
                "rows": (0, r0),
                "cols": (c0 + 2, w),
                "height": top,
                "width": right,
                "color": int(block[1, 0]),
                "near_row": "bottom",
                "near_col": "left",
            },
            {
                "name": "bl",
                "rows": (r0 + 2, h),
                "cols": (0, c0),
                "height": bottom,
                "width": left,
                "color": int(block[0, 1]),
                "near_row": "top",
                "near_col": "right",
            },
            {
                "name": "br",
                "rows": (r0 + 2, h),
                "cols": (c0 + 2, w),
                "height": bottom,
                "width": right,
                "color": int(block[0, 0]),
                "near_row": "top",
                "near_col": "left",
            },
        ],
    }


def _quadrant_branch(quadrant: dict) -> str:
    if quadrant["width"] > quadrant["height"]:
        return "horizontal"
    if quadrant["height"] > quadrant["width"]:
        return "vertical"
    return "solid"


def _fill_quadrant(out: np.ndarray, quadrant: dict) -> None:
    r_start, r_end = quadrant["rows"]
    c_start, c_end = quadrant["cols"]
    h = int(quadrant["height"])
    w = int(quadrant["width"])
    color = int(quadrant["color"])
    branch = _quadrant_branch(quadrant)

    if h == 0 or w == 0:
        return

    if branch == "solid":
        out[r_start:r_end, c_start:c_end] = color
        return

    if branch == "horizontal":
        for rr in range(r_start, r_end):
            distance = (r_end - 1 - rr) if quadrant["near_row"] == "bottom" else (rr - r_start)
            if distance % 2 == 0:
                out[rr, c_start:c_end] = color
        return

    for cc in range(c_start, c_end):
        distance = (c_end - 1 - cc) if quadrant["near_col"] == "right" else (cc - c_start)
        if distance % 2 == 0:
            out[r_start:r_end, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each quadrant around a 2x2 marker block with opposite-color stripes whose direction comes from quadrant geometry.

    Input structure:
    - The grid has a black background and exactly one 2x2 foreground marker block.
    - The four marker cells use four distinct non-black colors.
    - The marker block splits the remaining canvas into four rectangular quadrants.

    Rule:
    - Keep the 2x2 marker block unchanged.
    - Use diagonal mapping from marker cell to opposite quadrant:
      top-left -> bottom-right, top-right -> bottom-left, bottom-left -> top-right, bottom-right -> top-left.
    - For each quadrant, compare its width and height. If width > height, draw horizontal alternating stripes starting from the row nearest the marker block. If height > width, draw vertical alternating stripes starting from the column nearest the marker block. If width == height, fill the whole quadrant solid with the assigned color.

    Key insight:
    - The same 2x2 marker block determines both which color each quadrant receives and, through its off-center placement, which stripe direction that quadrant must use.
    """
    info = _locate_marker(grid)
    out = np.zeros(info["shape"], dtype=int)
    r0, c0 = info["block_top_left"]
    out[r0 : r0 + 2, c0 : c0 + 2] = info["block"]
    for quadrant in info["quadrants"]:
        _fill_quadrant(out, quadrant)
    return out


def _spec_output(grid: np.ndarray) -> np.ndarray:
    info = _locate_marker(grid)
    out = np.zeros(info["shape"], dtype=int)
    r0, c0 = info["block_top_left"]
    out[r0 : r0 + 2, c0 : c0 + 2] = info["block"]

    for quadrant in info["quadrants"]:
        r_start, r_end = quadrant["rows"]
        c_start, c_end = quadrant["cols"]
        h = int(quadrant["height"])
        w = int(quadrant["width"])
        color = int(quadrant["color"])
        branch = _quadrant_branch(quadrant)

        if h == 0 or w == 0:
            continue
        if branch == "solid":
            out[r_start:r_end, c_start:c_end] = color
            continue

        if branch == "horizontal":
            row_offsets = np.arange(h - 1, -1, -1, dtype=int) if quadrant["near_row"] == "bottom" else np.arange(h, dtype=int)
            stripe_rows = (row_offsets % 2) == 0
            patch = np.zeros((h, w), dtype=int)
            patch[stripe_rows, :] = color
            out[r_start:r_end, c_start:c_end] = patch
            continue

        col_offsets = np.arange(w - 1, -1, -1, dtype=int) if quadrant["near_col"] == "right" else np.arange(w, dtype=int)
        stripe_cols = (col_offsets % 2) == 0
        patch = np.zeros((h, w), dtype=int)
        patch[:, stripe_cols] = color
        out[r_start:r_end, c_start:c_end] = patch

    return out


def _validate_generated_input(grid: np.ndarray, *, require_square: bool | None) -> bool:
    try:
        info = _locate_marker(grid)
    except AssertionError:
        return False

    h, w = info["shape"]
    margins = info["margins"]
    if h == w:
        return False
    if min(margins.values()) < 3:
        return False
    if margins["top"] == margins["bottom"] and margins["left"] == margins["right"]:
        return False

    branches = [_quadrant_branch(q) for q in info["quadrants"]]
    if "horizontal" not in branches or "vertical" not in branches:
        return False

    has_square = "solid" in branches
    if require_square is True and not has_square:
        return False
    if require_square is False and has_square:
        return False

    try:
        out_main = main(grid)
        out_spec = _spec_output(grid)
    except AssertionError:
        return False

    if not np.array_equal(out_main, out_spec):
        return False
    if np.array_equal(out_main, grid):
        return False
    if len(set(map(int, out_main.ravel()))) < 3:
        return False
    return True


def _sample_square_margins() -> tuple[int, int, int, int]:
    corner = random.choice(("tl", "tr", "bl", "br"))
    side = random.randint(3, 10)
    delta_a, delta_b = random.sample(range(1, 6), 2)

    if corner == "tl":
        top, left = side, side
        right, bottom = side + delta_a, side + delta_b
    elif corner == "tr":
        top, right = side, side
        left, bottom = side + delta_a, side + delta_b
    elif corner == "bl":
        bottom, left = side, side
        top, right = side + delta_a, side + delta_b
    else:
        bottom, right = side, side
        top, left = side + delta_a, side + delta_b

    return int(top), int(bottom), int(left), int(right)


def _sample_between_margins() -> tuple[int, int, int, int]:
    small = random.randint(3, 8)
    large = random.randint(small + 2, 12)
    if random.random() < 0.5:
        top, bottom = small, large
    else:
        top, bottom = large, small

    middle = random.randint(min(top, bottom) + 1, max(top, bottom) - 1)
    total_vertical = top + bottom
    avoid_sum = total_vertical - middle
    other_choices = [
        value
        for value in range(3, 13)
        if value not in {top, bottom} and value != avoid_sum
    ]
    other = random.choice(other_choices)

    if random.random() < 0.5:
        left, right = middle, other
    else:
        left, right = other, middle

    return int(top), int(bottom), int(left), int(right)


def _sample_axis_margins() -> tuple[int, int, int, int]:
    if random.random() < 0.5:
        vertical = random.randint(4, 10)
        top = bottom = vertical
        left = random.randint(3, vertical - 1)
        right_choices = [
            value
            for value in range(vertical + 1, 13)
            if value != (2 * vertical - left)
        ]
        right = random.choice(right_choices)
    else:
        horizontal = random.randint(4, 10)
        left = right = horizontal
        top = random.randint(3, horizontal - 1)
        bottom_choices = [
            value
            for value in range(horizontal + 1, 13)
            if value != (2 * horizontal - top)
        ]
        bottom = random.choice(bottom_choices)

    return int(top), int(bottom), int(left), int(right)


def _build_input(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    h = int(top + 2 + bottom)
    w = int(left + 2 + right)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(NON_BLACK_COLORS, 4)
    block = np.array(colors, dtype=int).reshape(2, 2)
    grid[top : top + 2, left : left + 2] = block
    return grid


def _gen_attempt(mode: int) -> np.ndarray | None:
    if mode == 0:
        margins = _sample_square_margins()
        require_square = True
    elif mode == 1:
        margins = _sample_between_margins()
        require_square = False
    else:
        margins = _sample_axis_margins()
        require_square = False

    grid = _build_input(*margins)
    if _validate_generated_input(grid, require_square=require_square):
        return grid
    return None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    mode = _GEN_COUNTER % 3
    _GEN_COUNTER += 1

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
