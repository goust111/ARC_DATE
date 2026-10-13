from __future__ import annotations

import importlib.util
import random

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_038_5", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg


GEN_ATTEMPTS = 400
GRID_MIN = 13
GRID_MAX = 29
MAX_OBJECT_H = 13
MAX_OBJECT_W = 11
PALETTE = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.GRAY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
)


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    parts = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    assert boundary.size > 0, "expected a nonempty rectangle boundary"
    colors = np.unique(boundary)
    assert colors.size == 1, "each peeled boundary must be monochrome"
    return int(colors[0])


def _layer_depths(rect: np.ndarray) -> tuple[np.ndarray, list[int]]:
    rect = np.asarray(rect, dtype=int)
    h, w = rect.shape
    depth_map = np.zeros((h, w), dtype=int)
    top, left = 0, 0
    bottom, right = h, w
    depth = 1
    colors: list[int] = []

    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        colors.append(color)
        previous = (top, bottom, left, right)

        while top < bottom and np.all(rect[top, left:right] == color):
            depth_map[top, left:right] = depth
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            depth_map[bottom - 1, left:right] = depth
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            depth_map[top:bottom, left] = depth
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            depth_map[top:bottom, right - 1] = depth
            right -= 1

        assert (top, bottom, left, right) != previous, "peeling must make progress"
        depth += 1

    assert np.all(depth_map > 0), "every rectangle cell must belong to a layer"
    assert len(colors) >= 2, "expected at least two nested layers"
    assert len(colors) == len(set(colors)), "layer colors must be distinct"
    return depth_map, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the left nested rectangle to its point-symmetric position and reverse the copied layer colors.

    Input structure:
    - The grid has odd height and width on a black background.
    - Maroon markers appear at the left edge, center, and right edge of the center row.
    - A yellow divider fills the center column except at the center marker, and one multicolor nested rectangle sits entirely on the left side.

    Rule:
    - Keep the original nested rectangle, the divider, and the three maroon markers unchanged.
    - Read the left rectangle's layer colors from outside to inside by repeatedly peeling each monochrome rectangular boundary.
    - Rotate every source pixel 180 degrees around the center marker to the right side, but recolor layer depth d with the reversed layer order instead of the original color.

    Key insight:
    - The right object is determined by two coupled signals at once: geometry comes from point symmetry, while color comes from reversing the source rectangle's nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must have odd height and width"

    bg = int(get_bg(g))
    cr, cc = h // 2, w // 2
    marker_color = Color.MAROON
    divider_color = Color.YELLOW

    marker_mask = np.zeros_like(g, dtype=bool)
    marker_mask[cr, 0] = True
    marker_mask[cr, cc] = True
    marker_mask[cr, w - 1] = True
    assert np.array_equal(g == marker_color, marker_mask), "unexpected marker layout"

    divider_mask = np.zeros_like(g, dtype=bool)
    divider_mask[:, cc] = True
    divider_mask[cr, cc] = False
    assert np.array_equal(g == divider_color, divider_mask), "unexpected divider layout"

    source_mask = np.zeros_like(g, dtype=bool)
    source_mask[:, :cc] = True
    source_mask &= (g != bg) & (g != marker_color) & (g != divider_color)
    components = components_mask(source_mask, connectivity=4)
    assert len(components) == 1, "expected exactly one source object"

    source_coords = components[0]
    r0, c0, r1, c1 = bbox_coords(source_coords)
    assert c1 < cc, "source object must stay left of the divider"

    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(rect != bg), "source object must fill its bounding box"
    assert np.all(rect != marker_color) and np.all(rect != divider_color), "source colors must exclude markers and divider"

    right_payload = g[:, cc + 1 :].copy()
    right_payload[cr, -1] = bg
    assert not np.any(right_payload != bg), "right half must be empty except for the edge marker"

    depth_map, layer_colors = _layer_depths(rect)
    reversed_colors = list(reversed(layer_colors))

    out = g.copy()
    for sr, sc in source_coords:
        r = int(sr)
        c = int(sc)
        rr = 2 * cr - r
        rc = 2 * cc - c
        assert 0 <= rr < h and 0 <= rc < w, "reflected coordinate must stay on-grid"
        assert out[rr, rc] == bg, "reflected target must be blank before painting"
        depth = int(depth_map[r - r0, c - c0]) - 1
        out[rr, rc] = reversed_colors[depth]

    return out


def _odd_choices(lo: int, hi: int) -> list[int]:
    start = lo if lo % 2 == 1 else lo + 1
    return list(range(start, hi + 1, 2))


def _sample_rect_spec() -> tuple[list[int], int, int]:
    for _ in range(100):
        n_layers = random.randint(2, 4)
        ring_thicknesses = [1 + int(random.random() < 0.35) for _ in range(n_layers - 1)]
        total_ring = sum(ring_thicknesses)

        max_inner_h = MAX_OBJECT_H - 2 * total_ring
        max_inner_w = MAX_OBJECT_W - 2 * total_ring
        if max_inner_h < 1 or max_inner_w < 1:
            continue

        inner_h = random.randint(1, max_inner_h)
        inner_w = random.randint(1, max_inner_w)
        total_h = inner_h + 2 * total_ring
        total_w = inner_w + 2 * total_ring
        if total_h < 5 or total_w < 5:
            continue
        return ring_thicknesses, inner_h, inner_w

    raise RuntimeError("failed to sample a legal nested rectangle")


def _build_rect(colors: list[int], ring_thicknesses: list[int], inner_h: int, inner_w: int) -> np.ndarray:
    total_ring = sum(ring_thicknesses)
    h = inner_h + 2 * total_ring
    w = inner_w + 2 * total_ring
    rect = np.zeros((h, w), dtype=int)
    top, left = 0, 0
    bottom, right = h, w

    for i, color in enumerate(colors):
        rect[top:bottom, left:right] = color
        if i < len(ring_thicknesses):
            thickness = ring_thicknesses[i]
            top += thickness
            bottom -= thickness
            left += thickness
            right -= thickness

    return rect


def _gen_attempt() -> np.ndarray | None:
    ring_thicknesses, inner_h, inner_w = _sample_rect_spec()
    n_layers = len(ring_thicknesses) + 1
    colors = random.sample(PALETTE, n_layers)
    rect = _build_rect(colors, ring_thicknesses, inner_h, inner_w)
    obj_h, obj_w = rect.shape

    min_h = max(GRID_MIN, obj_h + 2)
    min_w = max(GRID_MIN, 2 * obj_w + 5)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h_choices = _odd_choices(min_h, GRID_MAX)
    w_choices = _odd_choices(min_w, GRID_MAX)
    if not h_choices or not w_choices:
        return None

    h = random.choice(h_choices)
    w = random.choice(w_choices)
    cr, cc = h // 2, w // 2

    max_r0 = h - obj_h - 1
    max_c0 = cc - obj_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return None

    r0 = random.randint(1, max_r0)
    c0 = random.randint(1, max_c0)

    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[:, cc] = Color.YELLOW
    grid[cr, cc] = Color.MAROON
    grid[cr, 0] = Color.MAROON
    grid[cr, w - 1] = Color.MAROON
    grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = rect

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out[:, cc + 1 :] != grid[:, cc + 1 :]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
