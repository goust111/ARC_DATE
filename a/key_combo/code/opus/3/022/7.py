from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

try:
    from arc_utils import (
        Color,
        bbox,
        bbox_h,
        bbox_w,
        colors_present,
        crop_bbox,
        find_components,
        scale_sprite,
    )
except ModuleNotFoundError:
    _ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _SPEC = importlib.util.spec_from_file_location("arc_utils", _ARC_UTILS_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    _ARC_UTILS = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_ARC_UTILS)
    Color = _ARC_UTILS.Color
    bbox = _ARC_UTILS.bbox
    bbox_h = _ARC_UTILS.bbox_h
    bbox_w = _ARC_UTILS.bbox_w
    colors_present = _ARC_UTILS.colors_present
    crop_bbox = _ARC_UTILS.crop_bbox
    find_components = _ARC_UTILS.find_components
    scale_sprite = _ARC_UTILS.scale_sprite


RING_COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_square_ring(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h != w or h % 2 == 0:
        return False
    if h == 1:
        return bool(mask[0, 0])

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(mask[border]) and not np.any(mask[~border]))


def _extract_pattern_crop(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    orange_components = find_components(
        grid,
        bg=Color.BLACK,
        colors=[Color.ORANGE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(orange_components) == 1, "expected one orange source motif"

    comp = orange_components[0]
    area = int(comp["area"])
    assert 3 <= area <= 6, "orange motif must contain 3..6 cells"

    box = comp["bbox"]
    h = bbox_h(box)
    w = bbox_w(box)
    assert 2 <= h <= 4 and 2 <= w <= 4, "orange motif bbox must be between 2x2 and 4x4"

    crop = crop_bbox(grid, box)
    crop_colors = set(map(int, np.unique(crop)))
    assert crop_colors.issubset({Color.BLACK, Color.ORANGE}), "orange motif bbox must contain only black and orange"
    assert int(np.count_nonzero(crop == Color.ORANGE)) == area
    assert area < h * w, "generator avoids solid rectangles so bbox black cells are visible after scaling"
    return crop, box


def _count_concentric_rings(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    ring_colors = colors_present(grid, exclude={Color.BLACK, Color.ORANGE})
    components = find_components(
        grid,
        bg=Color.BLACK,
        colors=ring_colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 2 <= len(components) <= 4, "expected 2..4 colored square rings"
    assert len(ring_colors) == len(components), "each ring must use a distinct color"

    centers = set()
    boxes: list[tuple[int, int, int, int]] = []
    for comp in components:
        color = int(comp["color"])
        box = comp["bbox"]
        side = bbox_h(box)
        assert side == bbox_w(box), "ring bbox must be square"
        assert side % 2 == 1, "ring side lengths must be odd"

        mask = crop_bbox(grid == color, box)
        assert _is_square_ring(mask), "each colored component must be a complete 1-pixel square ring"
        centers.add(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2))
        boxes.append(box)

    assert len(centers) == 1, "rings must be concentric"
    boxes.sort(key=bbox_h, reverse=True)
    sides = [bbox_h(box) for box in boxes]
    for outer, inner in zip(sides, sides[1:]):
        assert outer - inner == 4, "adjacent rings must be separated by one black row and column"
    assert sides[-1] in {1, 3}, "innermost layer must be 1x1 or a 3x3 ring"
    return len(boxes), boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count concentric square-ring layers, crop the orange motif, and enlarge that crop by the ring count.

    Input structure
    The grid has a black background. On the left there is one concentric structure
    made of 2 to 4 complete 1-pixel-thick square rings in distinct non-orange
    colors, separated by one black row and column between layers; the innermost
    layer is either a 1x1 pixel or a 3x3 ring. On the right there is exactly one
    4-connected orange motif whose tight bounding box is at most 4x4 and contains
    only black and orange cells.

    Rule
    Count how many concentric ring layers appear in the left structure; call that
    number k. Crop the orange motif to its tight bounding box, then replace every
    cell of that crop by a k-by-k block of the same color. Return only this scaled
    crop.

    Key insight
    The concentric rings do not contribute geometry to the output; they encode the
    integer scale factor, while the orange motif supplies the shape to crop and
    enlarge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    pattern_crop, pattern_box = _extract_pattern_crop(g)
    scale, ring_boxes = _count_concentric_rings(g)
    outer_ring = ring_boxes[0]
    assert pattern_box[1] - outer_ring[3] >= 3, "ring encoder and orange motif must be separated by at least two black columns"

    return scale_sprite(pattern_crop, scale)


# --------------- generator ---------------


def _ring_patch(layer_count: int, inner_side: int, colors: tuple[int, ...]) -> np.ndarray:
    outer_side = inner_side + 4 * (layer_count - 1)
    patch = np.zeros((outer_side, outer_side), dtype=int)
    for layer, color in enumerate(colors):
        side = outer_side - 4 * layer
        offset = 2 * layer
        if side == 1:
            patch[offset, offset] = int(color)
            continue
        patch[offset, offset : offset + side] = int(color)
        patch[offset + side - 1, offset : offset + side] = int(color)
        patch[offset : offset + side, offset] = int(color)
        patch[offset : offset + side, offset + side - 1] = int(color)
    return patch


def _random_pattern_patch() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(6, h * w - 1))

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set(_neighbors4(seed[0], seed[1], h, w))

        while int(mask.sum()) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for nb in _neighbors4(cell[0], cell[1], h, w):
                if not mask[nb]:
                    frontier.add(nb)

        if int(mask.sum()) != area:
            continue

        box = bbox(mask)
        assert box is not None
        mask = crop_bbox(mask.astype(int), box).astype(bool)
        hh, ww = mask.shape
        if not (2 <= hh <= 4 and 2 <= ww <= 4):
            continue
        if int(mask.sum()) == hh * ww:
            continue
        return np.where(mask, Color.ORANGE, Color.BLACK).astype(int)

    raise RuntimeError("failed to sample a connected orange motif")


def _compose_input(ring_patch: np.ndarray, pattern_patch: np.ndarray) -> np.ndarray:
    ring_h, ring_w = ring_patch.shape
    pat_h, pat_w = pattern_patch.shape

    gap_cols = random.randint(2, 5)
    min_h = max(ring_h, pat_h)
    max_h = min(30, min_h + 8)
    height = random.randint(min_h, max_h)

    min_w = ring_w + pat_w + gap_cols
    max_w = min(30, min_w + 8)
    width = random.randint(min_w, max_w)

    extra_w = width - min_w
    left_margin = random.randint(0, extra_w)
    extra_w -= left_margin
    middle_slack = random.randint(0, extra_w)

    ring_r = random.randint(0, height - ring_h)
    pat_r = random.randint(0, height - pat_h)
    ring_c = left_margin
    pat_c = ring_c + ring_w + gap_cols + middle_slack

    grid = np.zeros((height, width), dtype=int)
    grid[ring_r : ring_r + ring_h, ring_c : ring_c + ring_w] = ring_patch
    grid[pat_r : pat_r + pat_h, pat_c : pat_c + pat_w] = pattern_patch
    return grid


def _output_sig(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.tobytes()


def _validate_candidate(grid: np.ndarray, layer_count: int, pattern_patch: np.ndarray) -> np.ndarray:
    expected = scale_sprite(pattern_patch, layer_count)
    assert expected.ndim == 2 and max(expected.shape) <= 30

    parsed_crop, parsed_box = _extract_pattern_crop(grid)
    parsed_layers, ring_boxes = _count_concentric_rings(grid)
    assert parsed_layers == layer_count
    assert np.array_equal(parsed_crop, pattern_patch)
    assert parsed_box[1] - ring_boxes[0][3] >= 3

    if layer_count > 2:
        assert not np.array_equal(scale_sprite(pattern_patch, layer_count - 1), expected)
    if layer_count < 4:
        assert not np.array_equal(scale_sprite(pattern_patch, layer_count + 1), expected)

    actual = main(grid)
    assert np.array_equal(actual, expected)
    return expected


def _gen_attempt(forced_layers: int | None, forced_inner_side: int | None) -> np.ndarray | None:
    layer_count = forced_layers if forced_layers is not None else random.randint(2, 4)
    inner_side = forced_inner_side if forced_inner_side is not None else random.choice((1, 3))

    ring_colors = tuple(random.sample(RING_COLOR_POOL, layer_count))
    ring_patch = _ring_patch(layer_count, inner_side, ring_colors)
    pattern_patch = _random_pattern_patch()
    grid = _compose_input(ring_patch, pattern_patch)
    expected = _validate_candidate(grid, layer_count, pattern_patch)

    out_sig = _output_sig(expected)
    if out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    preferred_layers = 2 + (_GEN_CALL_COUNT % 3)
    preferred_inner_side = 1 if ((_GEN_CALL_COUNT // 3) % 2 == 0) else 3
    _GEN_CALL_COUNT += 1

    for attempt in range(GEN_ATTEMPTS):
        forced_layers = preferred_layers if attempt < GEN_ATTEMPTS // 2 else None
        forced_inner_side = preferred_inner_side if attempt < GEN_ATTEMPTS // 3 else None
        grid = _gen_attempt(forced_layers, forced_inner_side)
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid concentric-rings scaling input")
