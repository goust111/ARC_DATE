from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"failed to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds


BACKGROUND = Color.BLACK
BAR_COLOR = Color.RED
MARKER_COLOR = Color.TEAL

HEIGHT_RANGE = (3, 9)
WIDTH_RANGE = (6, 14)
HEIGHT_SLACK_MAX = 8
MARKER_COUNT_CAP = 5
MAX_GENERATION_ATTEMPTS = 400


def _bar_height(obj: dict) -> int:
    r0, _c0, r1, _c1 = obj["bbox"]
    return int(r1 - r0 + 1)


def _extract_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert width >= 4, "grid must have room for edge bars and adjacent marker columns"

    bg = int(get_bg(g))
    red_objects = find_components(
        g,
        bg=bg,
        colors=[BAR_COLOR],
        connectivity=4,
        monochromatic=True,
    )
    assert len(red_objects) == 2, "expected exactly two red components"

    left_bar = None
    right_bar = None
    for obj in red_objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert c0 == c1, "red bars must be one column wide"
        assert int(obj["area"]) == int(r1 - r0 + 1), "red bars must be solid"
        if c0 == 0:
            assert left_bar is None, "expected exactly one left edge bar"
            left_bar = obj
        elif c1 == width - 1:
            assert right_bar is None, "expected exactly one right edge bar"
            right_bar = obj
        else:
            raise AssertionError("red components must touch the left or right edge")

    assert left_bar is not None and right_bar is not None, "missing edge bar"

    left_height = _bar_height(left_bar)
    right_height = _bar_height(right_bar)
    assert right_height % left_height == 0, "right bar height must be an integer multiple of left bar height"
    scale = right_height // left_height
    assert 1 <= scale <= 3, "scale factor must be in 1..3"

    left_r0, _lc0, left_r1, _lc1 = left_bar["bbox"]
    teal_coords = np.argwhere(g == MARKER_COLOR)
    assert teal_coords.size > 0, "expected at least one teal marker"
    marker_rows = []
    for r, c in teal_coords.tolist():
        assert c == 1, "input teal markers must sit next to the left bar"
        assert left_r0 <= r <= left_r1, "input teal markers must align to left bar rows"
        marker_rows.append(int(r))
    marker_rows = sorted(set(marker_rows))
    assert marker_rows, "expected at least one distinct teal marker row"

    right_r0, _rc0, right_r1, _rc1 = right_bar["bbox"]
    return {
        "grid": g,
        "bg": bg,
        "height": height,
        "width": width,
        "left_top": int(left_r0),
        "left_height": int(left_height),
        "right_top": int(right_r0),
        "right_bottom": int(right_r1),
        "right_height": int(right_height),
        "scale": int(scale),
        "marker_rows": marker_rows,
    }


def _paint_correct_segments(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    col = info["width"] - 2
    for row in info["marker_rows"]:
        offset = row - info["left_top"]
        seg_top = info["right_top"] + info["right_height"] - info["scale"] * (offset + 1)
        seg_bottom = seg_top + info["scale"] - 1
        assert info["right_top"] <= seg_top <= seg_bottom <= info["right_bottom"]
        assert all(in_bounds(out.shape, rr, col) for rr in range(seg_top, seg_bottom + 1))
        out[seg_top : seg_bottom + 1, col] = MARKER_COLOR
    return out


def _paint_same_absolute_rows(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    col = info["width"] - 2
    for row in info["marker_rows"]:
        if info["right_top"] <= row <= info["right_bottom"]:
            out[row, col] = MARKER_COLOR
    return out


def _paint_top_scaled(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    col = info["width"] - 2
    for row in info["marker_rows"]:
        offset = row - info["left_top"]
        seg_top = info["right_top"] + info["scale"] * offset
        seg_bottom = seg_top + info["scale"] - 1
        if info["right_top"] <= seg_top <= seg_bottom <= info["right_bottom"]:
            out[seg_top : seg_bottom + 1, col] = MARKER_COLOR
    return out


def _paint_bottom_unscaled(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    col = info["width"] - 2
    for row in info["marker_rows"]:
        offset = row - info["left_top"]
        seg_bottom = info["right_bottom"] - offset
        seg_top = seg_bottom - info["scale"] + 1
        if info["right_top"] <= seg_top <= seg_bottom <= info["right_bottom"]:
            out[seg_top : seg_bottom + 1, col] = MARKER_COLOR
    return out


def _sample_marker_offsets(left_height: int) -> list[int]:
    max_count = min(MARKER_COUNT_CAP, left_height - 1)
    modes = ("top", "bottom", "both", "interior", "mixed")
    for _ in range(80):
        required = set()
        mode = random.choice(modes)
        if mode == "top":
            required.add(0)
        elif mode == "bottom":
            required.add(left_height - 1)
        elif mode == "both":
            required.update({0, left_height - 1})
        elif mode == "interior" and left_height >= 4:
            required.add(random.randint(1, left_height - 2))

        min_count = max(2, len(required))
        count = random.randint(min_count, max_count)
        pool = [d for d in range(left_height) if d not in required]
        offsets = set(required)
        offsets.update(random.sample(pool, count - len(offsets)))
        if len(offsets) < left_height:
            return sorted(offsets)
    raise RuntimeError("failed to sample a non-full marker subset")


def _build_input(height: int, width: int, left_top: int, left_height: int, right_top: int, scale: int, offsets: list[int]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[left_top : left_top + left_height, 0] = BAR_COLOR
    right_height = left_height * scale
    grid[right_top : right_top + right_height, width - 1] = BAR_COLOR
    for offset in offsets:
        grid[left_top + offset, 1] = MARKER_COLOR
    return grid


def _semantic_example_ok(grid: np.ndarray) -> bool:
    try:
        info = _extract_instance(grid)
    except AssertionError:
        return False

    correct = _paint_correct_segments(grid, info)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, correct):
        return False

    if np.array_equal(correct, _paint_same_absolute_rows(grid, info)):
        return False
    if np.array_equal(correct, _paint_top_scaled(grid, info)):
        return False
    if info["scale"] > 1 and np.array_equal(correct, _paint_bottom_unscaled(grid, info)):
        return False

    painted = np.argwhere(correct[:, info["width"] - 2] == MARKER_COLOR)
    expected_count = len(info["marker_rows"]) * info["scale"]
    if painted.shape[0] != expected_count:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge left-side teal markers by the red-bar height ratio and remap them from the opposite end onto the right bar.

    Input structure:
    - The background is black.
    - A one-column red bar touches the left edge, and another one-column red bar touches the right edge.
    - The right bar height is 1 to 3 times the left bar height, and teal marker pixels sit in the column immediately right of the left bar.

    Rule:
    - For each teal marker, measure its row offset from the top of the left bar.
    - Let k be the right-bar height divided by the left-bar height; paint a teal segment of length k in the column immediately left of the right bar.
    - Place that segment on the right bar's scale so the measured offset is counted upward from the right bar's bottom instead of downward from its top, while keeping the original bars and markers unchanged.

    Key insight:
    - The two red edge bars are the same ruler at different scales and opposite orientations: the left bar is read from the top, but the right bar is read from the bottom after stretching offsets by the height ratio.
    """
    info = _extract_instance(grid)
    return _paint_correct_segments(grid, info)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scale = random.randint(1, 3)
        left_height = random.randint(*HEIGHT_RANGE)
        if left_height * scale > 30:
            continue

        right_height = left_height * scale
        height = right_height + random.randint(0, min(HEIGHT_SLACK_MAX, 30 - right_height))
        width = random.randint(*WIDTH_RANGE)

        left_top = random.randint(0, height - left_height)
        right_top = random.randint(0, height - right_height)
        offsets = _sample_marker_offsets(left_height)
        grid = _build_input(height, width, left_top, left_height, right_top, scale, offsets)

        if _semantic_example_ok(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
