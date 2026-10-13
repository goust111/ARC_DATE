from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("shared_arc_utils", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox = _ARC_UTILS.bbox
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg


BG = 0
RECT_COLOR = 1
X_COLOR = 2
STRETCH_MARKER_COLOR = 4

DIRECTIONS = ("up", "down", "left", "right")
VERTICAL_DIRECTIONS = {"up", "down"}

GRID_H_RANGE = (11, 18)
GRID_W_RANGE = (11, 18)
VERTICAL_ORIG_H_RANGE = (3, 5)
VERTICAL_ORIG_W_RANGE = (5, 7)
HORIZONTAL_ORIG_H_RANGE = (5, 7)
HORIZONTAL_ORIG_W_RANGE = (3, 5)
STRETCH_GAP_RANGE = (4, 8)
PLACEMENT_BORDER = 1
GENERATION_RETRIES = 300

_direction_cursor = 0


def _edge_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _locate_original_rectangle(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], tuple[int, int]]:
    bg = int(get_bg(grid))

    marker_coords = np.argwhere(grid == STRETCH_MARKER_COLOR)
    assert marker_coords.shape == (1, 2), "expected one stretch marker"
    marker_r, marker_c = map(int, marker_coords[0])

    rect_mask = (grid != bg) & (grid != STRETCH_MARKER_COLOR)
    components = components_mask(rect_mask, connectivity=4)
    assert len(components) == 1, "expected one rectangle component"

    box = bbox(rect_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    rect = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert rect_mask[r0 : r1 + 1, c0 : c1 + 1].all(), "rectangle bbox must be solid"
    assert np.all((rect == RECT_COLOR) | (rect == X_COLOR)), "rectangle must use blue plus one red marker"

    edge = _edge_mask(*rect.shape)
    assert np.all(rect[edge] == RECT_COLOR), "rectangle border must be blue"

    red_coords = np.argwhere(rect == X_COLOR)
    assert red_coords.shape == (1, 2), "expected one embedded red marker"
    red_r, red_c = map(int, red_coords[0])
    assert 0 < red_r < rect.shape[0] - 1 and 0 < red_c < rect.shape[1] - 1, "red marker must be interior"

    return bg, (r0, c0, r1, c1), (marker_r, marker_c)


def _expanded_box(box: tuple[int, int, int, int], marker: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    marker_r, marker_c = marker

    candidates = []
    if marker_r < r0 and c0 <= marker_c <= c1:
        candidates.append((marker_r, c0, r1, c1))
    if marker_r > r1 and c0 <= marker_c <= c1:
        candidates.append((r0, c0, marker_r, c1))
    if marker_c < c0 and r0 <= marker_r <= r1:
        candidates.append((r0, marker_c, r1, c1))
    if marker_c > c1 and r0 <= marker_r <= r1:
        candidates.append((r0, c0, r1, marker_c))

    assert len(candidates) == 1, "marker must define one stretch direction"
    return candidates[0]


def _draw_corner_ray(
    out: np.ndarray,
    inner_box: tuple[int, int, int, int],
    start: tuple[int, int],
    delta: tuple[int, int],
) -> None:
    ir0, ic0, ir1, ic1 = inner_box
    r, c = start
    dr, dc = delta
    while ir0 <= r <= ir1 and ic0 <= c <= ic1:
        out[r, c] = X_COLOR
        r += dr
        c += dc


def _render_output(shape: tuple[int, int], bg: int, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    out = np.full(shape, bg, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    out[r0, c0 : c1 + 1] = RECT_COLOR
    out[r1, c0 : c1 + 1] = RECT_COLOR
    out[r0 : r1 + 1, c0] = RECT_COLOR
    out[r0 : r1 + 1, c1] = RECT_COLOR

    ir0, ic0 = r0 + 1, c0 + 1
    ir1, ic1 = r1 - 1, c1 - 1
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1
    assert inner_h >= 3 and inner_w >= 3, "expanded rectangle must leave at least a 3x3 interior"

    rays = [
        ((ir0, ic0), (1, 1)),
        ((ir0, ic1), (1, -1)),
        ((ir1, ic0), (-1, 1)),
        ((ir1, ic1), (-1, -1)),
    ]
    for start, delta in rays:
        _draw_corner_ray(out, (ir0, ic0, ir1, ic1), start, delta)

    if inner_h % 2 == 1 or inner_w % 2 == 1:
        out[ir0 + inner_h // 2, ic0 + inner_w // 2] = X_COLOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the marked solid rectangle, hollow it out, and draw a red X inside the expanded frame.

    Input structure:
    - A black background contains one blue solid rectangle.
    - The rectangle has exactly one red marker pixel strictly inside it.
    - One yellow marker pixel sits outside the rectangle, aligned with exactly one side.

    Rule:
    - Use the yellow marker to determine which side of the rectangle stretches and where the new edge lands.
    - Replace the stretched solid rectangle with a one-cell-thick blue hollow frame covering the expanded bbox.
    - Clear the whole frame interior to black, then draw red 45-degree rays from all four interior corners.
    - If the expanded interior has an odd height or width, also color the center cell red so the X stays connected.

    Key insight:
    - The external marker changes the rectangle's final geometry first; only that expanded interior determines the arm lengths of the red X.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, original_box, marker = _locate_original_rectangle(g)
    stretched_box = _expanded_box(original_box, marker)
    return _render_output(g.shape, bg, stretched_box)


def _sample_dim(bounds: tuple[int, int], required: int) -> int | None:
    lo = max(bounds[0], required + 2 * PLACEMENT_BORDER)
    hi = bounds[1]
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _pick_direction() -> str:
    global _direction_cursor
    direction = DIRECTIONS[_direction_cursor % len(DIRECTIONS)]
    _direction_cursor += 1
    return direction


def _sample_specs(direction: str) -> tuple[tuple[int, int], tuple[int, int, int, int], tuple[int, int], tuple[int, int, int, int]] | None:
    if direction in VERTICAL_DIRECTIONS:
        orig_h = random.randint(*VERTICAL_ORIG_H_RANGE)
        orig_w = random.randint(*VERTICAL_ORIG_W_RANGE)
    else:
        orig_h = random.randint(*HORIZONTAL_ORIG_H_RANGE)
        orig_w = random.randint(*HORIZONTAL_ORIG_W_RANGE)

    gap = random.randint(*STRETCH_GAP_RANGE)
    exp_h = orig_h + gap if direction in VERTICAL_DIRECTIONS else orig_h
    exp_w = orig_w if direction in VERTICAL_DIRECTIONS else orig_w + gap

    height = _sample_dim(GRID_H_RANGE, exp_h)
    width = _sample_dim(GRID_W_RANGE, exp_w)
    if height is None or width is None:
        return None

    exp_r0 = random.randint(PLACEMENT_BORDER, height - PLACEMENT_BORDER - exp_h)
    exp_c0 = random.randint(PLACEMENT_BORDER, width - PLACEMENT_BORDER - exp_w)
    expanded_box = (exp_r0, exp_c0, exp_r0 + exp_h - 1, exp_c0 + exp_w - 1)

    if direction == "up":
        original_box = (exp_r0 + gap, exp_c0, exp_r0 + gap + orig_h - 1, exp_c0 + orig_w - 1)
        marker = (exp_r0, random.randint(exp_c0 + 1, exp_c0 + orig_w - 2))
    elif direction == "down":
        original_box = (exp_r0, exp_c0, exp_r0 + orig_h - 1, exp_c0 + orig_w - 1)
        marker = (exp_r0 + exp_h - 1, random.randint(exp_c0 + 1, exp_c0 + orig_w - 2))
    elif direction == "left":
        original_box = (exp_r0, exp_c0 + gap, exp_r0 + orig_h - 1, exp_c0 + gap + orig_w - 1)
        marker = (random.randint(exp_r0 + 1, exp_r0 + orig_h - 2), exp_c0)
    else:
        original_box = (exp_r0, exp_c0, exp_r0 + orig_h - 1, exp_c0 + orig_w - 1)
        marker = (random.randint(exp_r0 + 1, exp_r0 + orig_h - 2), exp_c0 + exp_w - 1)

    return (height, width), original_box, marker, expanded_box


def _render_input(
    shape: tuple[int, int],
    original_box: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = original_box
    out[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR
    red_r = random.randint(r0 + 1, r1 - 1)
    red_c = random.randint(c0 + 1, c1 - 1)
    out[red_r, red_c] = X_COLOR
    out[marker] = STRETCH_MARKER_COLOR
    return out


def _gen_attempt(direction: str) -> np.ndarray | None:
    specs = _sample_specs(direction)
    if specs is None:
        return None

    shape, original_box, marker, expanded_box = specs
    inp = _render_input(shape, original_box, marker)
    expected = _render_output(shape, BG, expanded_box)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    preferred_direction = _pick_direction()
    for attempt in range(GENERATION_RETRIES):
        direction = preferred_direction if attempt < GENERATION_RETRIES // 2 else random.choice(DIRECTIONS)
        grid = _gen_attempt(direction)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stretched-hollow-X input")
