from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _frame_mask(height: int, width: int) -> np.ndarray:
    assert height >= 3 and width >= 3, "frames must have a nonempty interior"
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    canvas[r0, c0 : c0 + width] = color
    canvas[r0 + height - 1, c0 : c0 + width] = color
    canvas[r0 : r0 + height, c0] = color
    canvas[r0 : r0 + height, c0 + width - 1] = color


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    items: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "foreground color must appear"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == color
        expected = _frame_mask(height, width)
        assert np.array_equal(mask, expected), "each foreground color must form one 1-pixel rectangular frame"
        items.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
                "area": int(height * width),
                "center2": (int(r0 + r1), int(c0 + c1)),
            }
        )

    assert 2 <= len(items) <= 4, "expected 2 to 4 frame layers"
    items.sort(key=lambda item: int(item["area"]), reverse=True)

    center2 = items[0]["center2"]
    for item in items[1:]:
        assert item["center2"] == center2, "frames must be concentric"

    for outer, inner in zip(items, items[1:], strict=False):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert ir0 >= or0 + 2 and ic0 >= oc0 + 2, "inner frame must be separated from outer frame by background"
        assert ir1 <= or1 - 2 and ic1 <= oc1 - 2, "inner frame must be separated from outer frame by background"
        assert inner["height"] < outer["height"] and inner["width"] < outer["width"], "frame sizes must strictly decrease inward"

    return items


def _frame_crop(item: dict, bg: int) -> np.ndarray:
    crop = np.full((int(item["height"]), int(item["width"])), int(bg), dtype=int)
    _draw_frame(crop, 0, 0, int(item["height"]), int(item["width"]), int(item["color"]))
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split concentric rectangular frames into separate crops and lay them out left-to-right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    items = _extract_frames(g, bg)
    crops = [_frame_crop(item, bg) for item in items]

    out_h = max(int(crop.shape[0]) for crop in crops)
    out_w = sum(int(crop.shape[1]) for crop in crops) + len(crops) - 1
    out = np.full((out_h, out_w), int(bg), dtype=int)

    col = 0
    for crop in crops:
        h, w = map(int, crop.shape)
        row = (out_h - h) // 2
        out[row : row + h, col : col + w] = crop
        col += w + 1

    return out


COLOR_VALUES = tuple(range(10))
LAYER_COUNT_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (3, 11)
INNER_WIDTH_RANGE = (3, 11)
INSET_RANGE = (2, 4)
MARGIN_RANGE = (1, 6)
GRID_MAX_SIDE = 30
BACKGROUND_DOMINANCE_RATIO = 2
MAX_GENERATION_ATTEMPTS = 300


def _expected_from_layers(bg: int, layers: list[dict]) -> np.ndarray:
    ordered = sorted(layers, key=lambda item: int(item["height"] * item["width"]), reverse=True)
    crops = []
    for item in ordered:
        crop = np.full((int(item["height"]), int(item["width"])), int(bg), dtype=int)
        _draw_frame(crop, 0, 0, int(item["height"]), int(item["width"]), int(item["color"]))
        crops.append(crop)

    out_h = max(int(crop.shape[0]) for crop in crops)
    out_w = sum(int(crop.shape[1]) for crop in crops) + len(crops) - 1
    out = np.full((out_h, out_w), int(bg), dtype=int)

    c = 0
    for crop in crops:
        h, w = map(int, crop.shape)
        r = (out_h - h) // 2
        out[r : r + h, c : c + w] = crop
        c += w + 1
    return out


def _sample_layers() -> list[dict] | None:
    n_layers = random.randint(*LAYER_COUNT_RANGE)
    insets = [random.randint(*INSET_RANGE) for _ in range(n_layers - 1)]
    total_inset = sum(insets)

    max_inner_h = GRID_MAX_SIDE - 2 * total_inset - 2 * MARGIN_RANGE[0]
    max_inner_w = GRID_MAX_SIDE - 2 * total_inset - 2 * MARGIN_RANGE[0]
    hi_h = min(INNER_HEIGHT_RANGE[1], max_inner_h)
    hi_w = min(INNER_WIDTH_RANGE[1], max_inner_w)
    if hi_h < INNER_HEIGHT_RANGE[0] or hi_w < INNER_WIDTH_RANGE[0]:
        return None

    inner_h = random.randint(INNER_HEIGHT_RANGE[0], hi_h)
    inner_w = random.randint(INNER_WIDTH_RANGE[0], hi_w)

    dims_rev = [(inner_h, inner_w)]
    h, w = inner_h, inner_w
    for inset in insets:
        h += 2 * inset
        w += 2 * inset
        dims_rev.append((h, w))

    dims = list(reversed(dims_rev))
    bg = random.choice(COLOR_VALUES)
    colors = random.sample([c for c in COLOR_VALUES if c != bg], n_layers)
    return [{"height": h0, "width": w0, "color": color} for (h0, w0), color in zip(dims, colors, strict=False)]


def _build_input(layers: list[dict], bg: int) -> np.ndarray | None:
    outer_h = int(layers[0]["height"])
    outer_w = int(layers[0]["width"])
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    grid_h = outer_h + top + bottom
    grid_w = outer_w + left + right
    if grid_h > GRID_MAX_SIDE or grid_w > GRID_MAX_SIDE:
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    r0 = top
    c0 = left
    for item in layers:
        h = int(item["height"])
        w = int(item["width"])
        inset_r = (outer_h - h) // 2
        inset_c = (outer_w - w) // 2
        _draw_frame(grid, r0 + inset_r, c0 + inset_c, h, w, int(item["color"]))

    fg_pixels = int(np.count_nonzero(grid != int(bg)))
    if fg_pixels * BACKGROUND_DOMINANCE_RATIO >= grid.size:
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    layers = _sample_layers()
    if layers is None:
        return None
    bg = random.choice([c for c in COLOR_VALUES if c not in {item["color"] for item in layers}])
    grid = _build_input(layers, bg)
    if grid is None:
        return None

    expected = _expected_from_layers(bg, layers)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-frame ARC input")
