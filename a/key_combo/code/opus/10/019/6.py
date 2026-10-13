from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

MARKER_PALETTE = (BLUE, RED, GREEN, YELLOW, PINK)
QUADRANTS = ("tl", "tr", "bl", "br")
QUAD_TO_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
MODE_TO_BASE = {
    "top_split": ("tl", "tr"),
    "bottom_split": ("bl", "br"),
    "left_split": ("tl", "bl"),
    "right_split": ("tr", "br"),
}
MODE_TO_EXTRA = {
    "top_split": ("bl", "br"),
    "bottom_split": ("tl", "tr"),
    "left_split": ("tr", "br"),
    "right_split": ("tl", "bl"),
}
GEN_RETRIES = 400


def _grey_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one grey rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    rect = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(rect == GREY), "grey region must be a solid rectangle"
    assert coords.shape[0] == rect.size, "all grey cells must belong to the rectangle"
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 0 and w % 2 == 0, "rectangle must split evenly into quadrants"
    return r0, c0, r1, c1


def _marker_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }


def _control_hit_quadrant(
    control: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> str:
    r, c = control
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    r_mid = r0 + h // 2
    c_mid = c0 + w // 2

    if r0 <= r <= r1 and c < c0:
        entry_r, entry_c = r, c0
    elif r0 <= r <= r1 and c > c1:
        entry_r, entry_c = r, c1
    elif c0 <= c <= c1 and r < r0:
        entry_r, entry_c = r0, c
    elif c0 <= c <= c1 and r > r1:
        entry_r, entry_c = r1, c
    else:
        raise AssertionError("control must be outside the rectangle and axis-aligned to it")

    vertical = "t" if entry_r < r_mid else "b"
    horizontal = "l" if entry_c < c_mid else "r"
    return vertical + horizontal


def _expected_output_from_spec(
    grid: np.ndarray,
    marker_colors: dict[str, int],
    active_quadrants: set[str],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    r_mid = r0 + h // 2
    c_mid = c0 + w // 2

    out = grid.copy()
    for quad, (mr, mc) in _marker_positions(r0, c0, r1, c1).items():
        out[mr, mc] = BLACK
    out[out == ORANGE] = BLACK

    quad_slices = {
        "tl": (slice(r0, r_mid), slice(c0, c_mid)),
        "tr": (slice(r0, r_mid), slice(c_mid, c1 + 1)),
        "bl": (slice(r_mid, r1 + 1), slice(c0, c_mid)),
        "br": (slice(r_mid, r1 + 1), slice(c_mid, c1 + 1)),
    }
    for quad, region in quad_slices.items():
        fill = marker_colors[quad] if quad in active_quadrants else GREY
        out[region] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the ray-activated quadrants of the gray rectangle with the colors named by diagonal markers.

    Input structure:
    - The background is black.
    - One solid gray rectangle with even height and even width sits away from the grid edges.
    - The four diagonal cells just outside the rectangle corners are colored marker pixels, and 2-3 orange control pixels lie outside the rectangle while sharing either a rectangle row or a rectangle column.

    Rule:
    - Split the gray rectangle into the top-left, top-right, bottom-left, and bottom-right quadrants.
    - Read the four diagonal marker pixels as the colors assigned to those four quadrants.
    - For each orange control pixel, project a straight axis-aligned ray toward the rectangle; the quadrant containing the ray's first rectangle cell becomes active.
    - Erase all markers and control pixels, recolor every active quadrant with its marker color, and leave inactive quadrants gray.

    Key insight:
    - The diagonal markers answer "what color would this quadrant be", while the control rays independently answer "which quadrants are allowed to change".
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    r0, c0, r1, c1 = _grey_bbox(g)
    marker_pos = _marker_positions(r0, c0, r1, c1)
    marker_colors: dict[str, int] = {}
    for quad, (mr, mc) in marker_pos.items():
        assert 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1], "marker must stay in bounds"
        color = int(g[mr, mc])
        assert color in MARKER_PALETTE, "marker colors must come from the visible marker palette"
        marker_colors[quad] = color
    assert len(set(marker_colors.values())) == 4, "all four markers must be distinct"

    controls = [tuple(map(int, rc)) for rc in np.argwhere(g == ORANGE)]
    assert 2 <= len(controls) <= 3, "expected 2-3 control pixels"
    active_quadrants = {_control_hit_quadrant(control, r0, c0, r1, c1) for control in controls}
    assert len(active_quadrants) == len(controls), "different controls must activate different quadrants"
    assert len(active_quadrants) <= 3, "at least one quadrant must remain inactive"

    allowed = set(marker_pos.values()) | set(controls) | {
        (r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
    }
    foreground = {
        (int(r), int(c)) for r, c in np.argwhere(g != BLACK)
    }
    assert foreground == allowed, "input should contain only the rectangle, markers, and controls"

    return _expected_output_from_spec(g, marker_colors, active_quadrants, (r0, c0, r1, c1))


def _build_input(
    height: int,
    width: int,
    bbox: tuple[int, int, int, int],
    marker_colors: dict[str, int],
    controls: list[tuple[int, int]],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    grid = np.full((height, width), BLACK, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for quad, color in marker_colors.items():
        mr, mc = _marker_positions(r0, c0, r1, c1)[quad]
        grid[mr, mc] = color
    for r, c in controls:
        grid[r, c] = ORANGE
    return grid


def _sample_geometry() -> tuple[int, int, tuple[int, int, int, int]]:
    rect_h = random.choice((4, 6, 8))
    rect_w = random.choice((4, 6, 8))
    top = random.randint(3, 7)
    bottom = random.randint(3, 7)
    left = random.randint(3, 7)
    right = random.randint(3, 7)
    height = top + rect_h + bottom
    width = left + rect_w + right
    r0 = top
    c0 = left
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    assert height <= 30 and width <= 30
    return height, width, (r0, c0, r1, c1)


def _half_ranges(
    height: int, width: int, bbox: tuple[int, int, int, int]
) -> dict[str, list[int]]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    r_mid = r0 + h // 2
    c_mid = c0 + w // 2
    return {
        "top_rows": list(range(0, r0 - 1)),
        "bottom_rows": list(range(r1 + 2, height)),
        "left_cols": list(range(0, c0 - 1)),
        "right_cols": list(range(c1 + 2, width)),
        "top_half_rows": list(range(r0, r_mid)),
        "bottom_half_rows": list(range(r_mid, r1 + 1)),
        "left_half_cols": list(range(c0, c_mid)),
        "right_half_cols": list(range(c_mid, c1 + 1)),
    }


def _sample_control(
    quad: str,
    side: str,
    bands: dict[str, list[int]],
    fixed_outer: int | None = None,
) -> tuple[int, int]:
    assert side in QUAD_TO_SIDES[quad]
    if quad == "tl":
        row_band = bands["top_half_rows"]
        col_band = bands["left_half_cols"]
    elif quad == "tr":
        row_band = bands["top_half_rows"]
        col_band = bands["right_half_cols"]
    elif quad == "bl":
        row_band = bands["bottom_half_rows"]
        col_band = bands["left_half_cols"]
    else:
        row_band = bands["bottom_half_rows"]
        col_band = bands["right_half_cols"]

    if side == "top":
        outer = fixed_outer if fixed_outer is not None else random.choice(bands["top_rows"])
        return outer, random.choice(col_band)
    if side == "bottom":
        outer = fixed_outer if fixed_outer is not None else random.choice(bands["bottom_rows"])
        return outer, random.choice(col_band)
    if side == "left":
        outer = fixed_outer if fixed_outer is not None else random.choice(bands["left_cols"])
        return random.choice(row_band), outer
    if side == "right":
        outer = fixed_outer if fixed_outer is not None else random.choice(bands["right_cols"])
        return random.choice(row_band), outer
    raise AssertionError("unknown side")


def _sample_controls(
    height: int, width: int, bbox: tuple[int, int, int, int]
) -> tuple[list[tuple[int, int]], set[str]]:
    bands = _half_ranges(height, width, bbox)
    assert bands["top_rows"] and bands["bottom_rows"] and bands["left_cols"] and bands["right_cols"]

    mode = random.choice(tuple(MODE_TO_BASE))
    active = list(MODE_TO_BASE[mode])
    if random.random() < 0.5:
        active.append(random.choice(MODE_TO_EXTRA[mode]))

    controls: list[tuple[int, int]] = []
    if mode == "top_split":
        shared_row = random.choice(bands["top_rows"])
        controls.append(_sample_control("tl", "top", bands, fixed_outer=shared_row))
        controls.append(_sample_control("tr", "top", bands, fixed_outer=shared_row))
    elif mode == "bottom_split":
        shared_row = random.choice(bands["bottom_rows"])
        controls.append(_sample_control("bl", "bottom", bands, fixed_outer=shared_row))
        controls.append(_sample_control("br", "bottom", bands, fixed_outer=shared_row))
    elif mode == "left_split":
        shared_col = random.choice(bands["left_cols"])
        controls.append(_sample_control("tl", "left", bands, fixed_outer=shared_col))
        controls.append(_sample_control("bl", "left", bands, fixed_outer=shared_col))
    else:
        shared_col = random.choice(bands["right_cols"])
        controls.append(_sample_control("tr", "right", bands, fixed_outer=shared_col))
        controls.append(_sample_control("br", "right", bands, fixed_outer=shared_col))

    if len(active) == 3:
        extra_quad = active[-1]
        extra_side = random.choice(QUAD_TO_SIDES[extra_quad])
        controls.append(_sample_control(extra_quad, extra_side, bands))

    assert len(controls) == len(active)
    assert len(set(controls)) == len(controls)
    return controls, set(active)


def _scene_attempt() -> np.ndarray | None:
    height, width, bbox = _sample_geometry()
    marker_colors = {
        quad: color
        for quad, color in zip(QUADRANTS, random.sample(MARKER_PALETTE, 4))
    }
    controls, active_quadrants = _sample_controls(height, width, bbox)

    grid = _build_input(height, width, bbox, marker_colors, controls)
    hit_quadrants = {_control_hit_quadrant(control, *bbox) for control in controls}
    if hit_quadrants != active_quadrants:
        return None

    expected = _expected_output_from_spec(grid, marker_colors, active_quadrants, bbox)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
