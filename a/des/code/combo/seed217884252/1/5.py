from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

MIN_LAYERS = 3
MAX_LAYERS = 5

INNER_HEIGHT_RANGE = (3, 8)
INNER_WIDTH_RANGE = (3, 8)
INSET_RANGE = (2, 3)
MARGIN_RANGE = (0, 3)
MAX_GENERATION_ATTEMPTS = 300


BBox = tuple[int, int, int, int]
Layer = tuple[int, BBox]


def _bbox_from_mask(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one colored cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_cell_count(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return 2 * (h + w) - 4


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _paint_corners(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0] = int(color)
    grid[r0, c1] = int(color)
    grid[r1, c0] = int(color)
    grid[r1, c1] = int(color)


def _strictly_contains(outer: BBox, inner: BBox) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _layer_from_color(grid: np.ndarray, color: int) -> Layer:
    mask = grid == int(color)
    bbox = _bbox_from_mask(mask)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "each frame must have a non-empty interior"
    assert int(mask.sum()) == _frame_cell_count(bbox), "each color must form exactly one 1-pixel frame"
    assert np.all(grid[r0, c0 : c1 + 1] == color)
    assert np.all(grid[r1, c0 : c1 + 1] == color)
    assert np.all(grid[r0 : r1 + 1, c0] == color)
    assert np.all(grid[r0 : r1 + 1, c1] == color)
    return int(color), bbox


def _extract_layers(grid: np.ndarray) -> list[Layer]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert MIN_LAYERS <= len(colors) <= MAX_LAYERS, "expected 3 to 5 nested frames"

    layers = [_layer_from_color(grid, color) for color in colors]
    layers.sort(key=lambda layer: _frame_cell_count(layer[1]), reverse=True)

    for (outer_color, outer_bbox), (inner_color, inner_bbox) in zip(layers, layers[1:]):
        assert outer_color != inner_color
        assert _strictly_contains(outer_bbox, inner_bbox), "frames must form one strict nesting chain"

    return layers


def _render_output(shape: tuple[int, int], layers: list[Layer]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx, (color, bbox) in enumerate(layers):
        body_color = BG if idx == len(layers) - 1 else layers[idx + 1][0]
        _draw_frame(out, bbox, body_color)
        _paint_corners(out, bbox, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each frame's inner neighbor color outward while preserving corner markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layers = _extract_layers(g)
    return _render_output(g.shape, layers)


def _expand_bbox(bbox: BBox, top: int, bottom: int, left: int, right: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return r0 - top, c0 - left, r1 + bottom, c1 + right


def _shift_bbox(bbox: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _sample_layout(n_layers: int) -> tuple[tuple[int, int], list[BBox]]:
    inner_h = random.randint(*INNER_HEIGHT_RANGE)
    inner_w = random.randint(*INNER_WIDTH_RANGE)

    bboxes_inner_first = [(0, 0, inner_h - 1, inner_w - 1)]
    current = bboxes_inner_first[0]
    for _ in range(n_layers - 1):
        top = random.randint(*INSET_RANGE)
        bottom = random.randint(*INSET_RANGE)
        left = random.randint(*INSET_RANGE)
        right = random.randint(*INSET_RANGE)
        current = _expand_bbox(current, top, bottom, left, right)
        bboxes_inner_first.append(current)

    outer_bbox = bboxes_inner_first[-1]
    normalize_r = -outer_bbox[0]
    normalize_c = -outer_bbox[1]
    normalized = [_shift_bbox(bbox, normalize_r, normalize_c) for bbox in bboxes_inner_first]

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    shifted = [_shift_bbox(bbox, top_margin, left_margin) for bbox in normalized]
    outer_shifted = shifted[-1]
    height = int(outer_shifted[2] + 1 + bottom_margin)
    width = int(outer_shifted[3] + 1 + right_margin)

    bboxes_outer_first = list(reversed(shifted))
    return (height, width), bboxes_outer_first


def _render_input(shape: tuple[int, int], layers: list[Layer]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, bbox in layers:
        _draw_frame(grid, bbox, color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    shape, bboxes = _sample_layout(n_layers)
    colors = random.sample(PALETTE, n_layers)
    layers = [(int(color), bbox) for color, bbox in zip(colors, bboxes)]

    inp = _render_input(shape, layers)
    expected = _render_output(shape, layers)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
