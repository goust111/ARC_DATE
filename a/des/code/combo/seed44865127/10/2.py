from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    values = values[order]
    counts = counts[order]
    assert len(counts) >= 1
    if len(counts) > 1:
        assert counts[0] > counts[1], "background color must be uniquely dominant"
    return int(values[0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_layer(grid: np.ndarray, color: int, bg: int) -> dict[str, object]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "layer color must be present"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)

    mask = np.zeros((height, width), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    full_mask = np.ones((height, width), dtype=bool)
    perimeter = _frame_mask(height, width)

    if np.array_equal(mask, full_mask):
        kind = "solid"
    else:
        assert height >= 2 and width >= 2, "frame layers cannot be degenerate"
        assert np.array_equal(mask, perimeter), "layer must be a rectangular frame or a solid rectangle"
        kind = "frame"

    sprite = np.full((height, width), bg, dtype=int)
    sprite[mask] = color
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "area": height * width,
        "kind": kind,
        "sprite": sprite,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Explode nested rectangular layers into separate horizontally arranged views."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert 3 <= len(colors) <= 5, "expected 3-5 nested layers"

    layers = [_extract_layer(g, color, bg) for color in colors]
    areas = [int(layer["area"]) for layer in layers]
    assert len(set(areas)) == len(areas), "nested layer areas must be unique"
    layers.sort(key=lambda layer: int(layer["area"]), reverse=True)

    solid_indices = [idx for idx, layer in enumerate(layers) if layer["kind"] == "solid"]
    assert len(solid_indices) <= 1, "at most one solid layer is allowed"
    if solid_indices:
        assert solid_indices[0] == len(layers) - 1, "only the innermost layer may be solid"

    for outer, inner in zip(layers, layers[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1, "layers must be strictly nested"

    out_height = max(int(layer["height"]) for layer in layers)
    out_width = sum(int(layer["width"]) for layer in layers) + len(layers) - 1
    out = np.full((out_height, out_width), bg, dtype=int)

    cursor = 0
    for layer in layers:
        height = int(layer["height"])
        width = int(layer["width"])
        top = (out_height - height) // 2
        sprite = np.asarray(layer["sprite"], dtype=int)
        region = out[top : top + height, cursor : cursor + width]
        region[sprite != bg] = sprite[sprite != bg]
        cursor += width + 1

    return out


# --------------- generator ---------------

_LAYER_COUNT_RANGE = (3, 5)
_INNER_SOLID_PROBABILITY = 0.5
_INNER_FRAME_SIZE_RANGE = (3, 6)
_INNER_SOLID_SIZE_RANGE = (1, 5)
_INSET_RANGE = (1, 3)
_CANVAS_MARGIN_RANGE = (1, 5)
_CANVAS_EXTRA_SLACK_RANGE = (0, 5)
_MAX_OUTER_SIZE = 24
_MAX_CANVAS_SIZE = 30
_MAX_OUTPUT_WIDTH = 30
_MAX_GENERATION_ATTEMPTS = 300
_PALETTE = tuple(range(10))


def _randint(bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(_RNG.integers(lo, hi + 1))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_solid(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _sample_nested_boxes(layer_count: int, inner_solid: bool) -> list[tuple[int, int, int, int]] | None:
    size_bounds = _INNER_SOLID_SIZE_RANGE if inner_solid else _INNER_FRAME_SIZE_RANGE
    inner_h = _randint(size_bounds)
    inner_w = _randint(size_bounds)

    if not inner_solid and (inner_h < 3 or inner_w < 3):
        return None

    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(layer_count - 1):
        r0, c0, r1, c1 = boxes[0]
        top = _randint(_INSET_RANGE)
        bottom = _randint(_INSET_RANGE)
        left = _randint(_INSET_RANGE)
        right = _randint(_INSET_RANGE)
        boxes.insert(0, (r0 - top, c0 - left, r1 + bottom, c1 + right))

    shift_r = -boxes[0][0]
    shift_c = -boxes[0][1]
    normalized = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        for r0, c0, r1, c1 in boxes
    ]
    outer_h = normalized[0][2] + 1
    outer_w = normalized[0][3] + 1
    if outer_h > _MAX_OUTER_SIZE or outer_w > _MAX_OUTER_SIZE:
        return None

    for bbox in normalized[:-1]:
        height = bbox[2] - bbox[0] + 1
        width = bbox[3] - bbox[1] + 1
        if height < 3 or width < 3:
            return None

    return normalized


def _gen_attempt() -> np.ndarray | None:
    layer_count = _randint(_LAYER_COUNT_RANGE)
    inner_solid = bool(_RNG.random() < _INNER_SOLID_PROBABILITY)
    boxes = _sample_nested_boxes(layer_count, inner_solid)
    if boxes is None:
        return None

    outer_h = boxes[0][2] + 1
    outer_w = boxes[0][3] + 1
    predicted_out_w = sum(bbox[3] - bbox[1] + 1 for bbox in boxes) + layer_count - 1
    if predicted_out_w > _MAX_OUTPUT_WIDTH:
        return None
    margin_top = _randint(_CANVAS_MARGIN_RANGE)
    margin_bottom = _randint(_CANVAS_MARGIN_RANGE)
    margin_left = _randint(_CANVAS_MARGIN_RANGE)
    margin_right = _randint(_CANVAS_MARGIN_RANGE)
    extra_h = _randint(_CANVAS_EXTRA_SLACK_RANGE)
    extra_w = _randint(_CANVAS_EXTRA_SLACK_RANGE)
    canvas_h = outer_h + margin_top + margin_bottom + extra_h
    canvas_w = outer_w + margin_left + margin_right + extra_w
    if canvas_h > _MAX_CANVAS_SIZE or canvas_w > _MAX_CANVAS_SIZE:
        return None

    colors = list(_RNG.choice(_PALETTE, size=layer_count + 1, replace=False))
    bg = int(colors[0])
    layer_colors = [int(color) for color in colors[1:]]
    grid = np.full((canvas_h, canvas_w), bg, dtype=int)

    anchor_r = _randint((margin_top, canvas_h - outer_h - margin_bottom))
    anchor_c = _randint((margin_left, canvas_w - outer_w - margin_right))
    translated_boxes = [
        (r0 + anchor_r, c0 + anchor_c, r1 + anchor_r, c1 + anchor_c)
        for r0, c0, r1, c1 in boxes
    ]

    for idx, (bbox, color) in enumerate(zip(translated_boxes, layer_colors)):
        is_last = idx == len(translated_boxes) - 1
        if is_last and inner_solid:
            _draw_solid(grid, bbox, color)
        else:
            _draw_frame(grid, bbox, color)

    counts = Counter(map(int, grid.ravel()))
    bg_count = counts[bg]
    if any(count >= bg_count for color, count in counts.items() if color != bg):
        return None

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
