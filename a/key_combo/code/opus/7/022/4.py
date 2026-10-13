from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
import random

import numpy as np

try:
    from arc_utils import bbox_coords, get_bg
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_fallback",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    bbox_coords = _ARC_UTILS.bbox_coords
    get_bg = _ARC_UTILS.get_bg


BACKGROUND = 0
CROSS_COLOR = 1
FRAME_COLOR = 5
GUIDE_PALETTE = (2, 3, 4, 6, 7, 8, 9)
FRAME_SIZES = (9, 11)
MARGIN_RANGE = (1, 4)
GENERATION_RETRIES = 400


def _frame_bbox_from_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "missing frame"
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 5 and w >= 5
    assert h % 2 == 1 and w % 2 == 1, "frame dimensions must be odd"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    assert np.array_equal(sub == int(color), expected), "frame must be a hollow rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _guide_positions(frame_bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    return {
        "ul": (r0 - 1, c0 - 1),
        "ur": (r0 - 1, c1 + 1),
        "ll": (r1 + 1, c0 - 1),
        "lr": (r1 + 1, c1 + 1),
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BACKGROUND, "background must be black"

    frame_bbox = _frame_bbox_from_color(g, FRAME_COLOR)
    r0, c0, r1, c1 = frame_bbox
    H, W = g.shape

    guides = _guide_positions(frame_bbox)
    guide_colors: dict[str, int] = {}
    for key, (r, c) in guides.items():
        assert 0 <= r < H and 0 <= c < W, "guide pixel out of bounds"
        color = int(g[r, c])
        assert color not in (BACKGROUND, CROSS_COLOR, FRAME_COLOR), "invalid guide color"
        guide_colors[key] = color
    assert len(set(guide_colors.values())) == 4, "guide colors must be distinct"

    blue_coords = np.argwhere(g == CROSS_COLOR)
    assert blue_coords.size > 0, "missing cross"
    assert all(r0 < int(r) < r1 and c0 < int(c) < c1 for r, c in blue_coords), "cross must stay inside the frame interior"

    row_counts = Counter(int(r) for r in blue_coords[:, 0])
    col_counts = Counter(int(c) for c in blue_coords[:, 1])
    cross_rows = [r for r, count in row_counts.items() if count > 1]
    cross_cols = [c for c, count in col_counts.items() if count > 1]
    assert len(cross_rows) == 1 and len(cross_cols) == 1, "expected one interior cross"
    cross_r = int(cross_rows[0])
    cross_c = int(cross_cols[0])

    expected_cross = np.zeros_like(g, dtype=bool)
    expected_cross[cross_r, c0 + 1 : c1] = True
    expected_cross[r0 + 1 : r1, cross_c] = True
    assert np.array_equal(g == CROSS_COLOR, expected_cross), "cross must span the full interior row and column"

    allowed_guide_cells = set(guides.values())
    for r, c in np.argwhere(g != BACKGROUND):
        cell = (int(r), int(c))
        value = int(g[r, c])
        if value == FRAME_COLOR:
            continue
        if value == CROSS_COLOR:
            continue
        assert cell in allowed_guide_cells, "unexpected extra foreground pixel"

    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    return {
        "grid": g,
        "frame_bbox": frame_bbox,
        "cross": (cross_r, cross_c),
        "center": (int(center_r), int(center_c)),
        "guide_colors": guide_colors,
    }


def _paint_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, frame_bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _quadrant_slices(
    frame_bbox: tuple[int, int, int, int],
    cross_r: int,
    cross_c: int,
) -> dict[str, tuple[slice, slice]]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    return {
        "ul": (slice(r0 + 1, int(cross_r)), slice(c0 + 1, int(cross_c))),
        "ur": (slice(r0 + 1, int(cross_r)), slice(int(cross_c) + 1, c1)),
        "ll": (slice(int(cross_r) + 1, r1), slice(c0 + 1, int(cross_c))),
        "lr": (slice(int(cross_r) + 1, r1), slice(int(cross_c) + 1, c1)),
    }


def _render_scene(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    cross: tuple[int, int],
    guide_colors: dict[str, int],
    fill_quadrants: bool,
    include_guides: bool,
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    cross_r, cross_c = map(int, cross)
    if fill_quadrants:
        for key, (rs, cs) in _quadrant_slices(frame_bbox, cross_r, cross_c).items():
            color = int(guide_colors[key])
            out[rs, cs] = color

    r0, c0, r1, c1 = map(int, frame_bbox)
    out[cross_r, c0 + 1 : c1] = CROSS_COLOR
    out[r0 + 1 : r1, cross_c] = CROSS_COLOR
    _paint_frame(out, frame_bbox, FRAME_COLOR)

    if include_guides:
        for key, (r, c) in _guide_positions(frame_bbox).items():
            out[r, c] = int(guide_colors[key])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recenter the interior cross to the frame middle, then use the four diagonal guide pixels to fill the resulting quadrants.

    Input structure:
    - The grid has black background, one grey one-pixel-thick hollow rectangle, one blue interior cross, and four singleton guide pixels.
    - Each guide pixel sits exactly one step diagonally outside a different frame corner.
    - The blue cross is confined to the frame interior as one full interior row plus one full interior column, with its intersection away from the frame border.

    Rule:
    - Find the hollow frame and its unique geometric center.
    - Translate the blue cross so its intersection moves to that center while keeping the cross entirely inside the frame interior.
    - Use the colors of the upper-left, upper-right, lower-left, and lower-right guide pixels to fill the corresponding four interior quadrants created by the recentered cross.
    - Keep the frame grey, keep the recentered cross blue, and erase the four guide pixels to black.

    Key insight:
    - The cross is not merely recolored: it must first be recentered, because only the moved cross creates the final four-way partition that determines where the corner-guide colors belong.
    """
    scene = _parse_scene(grid)
    return _render_scene(
        scene["grid"].shape,
        scene["frame_bbox"],
        scene["center"],
        scene["guide_colors"],
        fill_quadrants=True,
        include_guides=False,
    )


def _sample_shifted_cross(center: int, lo: int, hi: int, mode: str) -> int:
    choices = list(range(int(lo), int(hi) + 1))
    assert center in choices
    if mode == "center":
        return int(center)
    choices = [v for v in choices if v != int(center)]
    return int(random.choice(choices))


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    scene = _parse_scene(inp)
    frame_bbox = scene["frame_bbox"]
    guide_colors = scene["guide_colors"]
    center = scene["center"]
    cross = scene["cross"]

    expected = _render_scene(
        inp.shape,
        frame_bbox,
        center,
        guide_colors,
        fill_quadrants=True,
        include_guides=False,
    )
    if not np.array_equal(out, expected):
        return False

    if cross == center:
        return False

    stationary = _render_scene(
        inp.shape,
        frame_bbox,
        cross,
        guide_colors,
        fill_quadrants=True,
        include_guides=False,
    )
    if np.array_equal(out, stationary):
        return False

    for key, (r, c) in _guide_positions(frame_bbox).items():
        if int(out[r, c]) != BACKGROUND:
            return False

    center_r, center_c = center
    r0, c0, r1, c1 = frame_bbox
    if np.any(out[center_r, c0 + 1 : c1] != CROSS_COLOR):
        return False
    if np.any(out[r0 + 1 : r1, center_c] != CROSS_COLOR):
        return False

    for key, (rs, cs) in _quadrant_slices(frame_bbox, center_r, center_c).items():
        block = out[rs, cs]
        if block.size == 0 or not np.all(block == int(guide_colors[key])):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    frame_h = random.choice(FRAME_SIZES)
    frame_w = random.choice(FRAME_SIZES)
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    H = frame_h + top + bottom
    W = frame_w + left + right
    if max(H, W) > 30:
        return None

    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
    r0, c0, r1, c1 = frame_bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    guide_colors = dict(
        zip(
            ("ul", "ur", "ll", "lr"),
            random.sample(GUIDE_PALETTE, 4),
        )
    )

    branch = random.choice(("row_only", "col_only", "both"))
    row_mode = "offset" if branch in ("row_only", "both") else "center"
    col_mode = "offset" if branch in ("col_only", "both") else "center"
    cross_r = _sample_shifted_cross(center_r, r0 + 1, r1 - 1, row_mode)
    cross_c = _sample_shifted_cross(center_c, c0 + 1, c1 - 1, col_mode)

    inp = _render_scene(
        (H, W),
        frame_bbox,
        (cross_r, cross_c),
        guide_colors,
        fill_quadrants=False,
        include_guides=True,
    )
    if (cross_r, cross_c) == (center_r, center_c):
        return None

    out = main(inp)
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
