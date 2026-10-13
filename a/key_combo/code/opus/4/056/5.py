from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_opus4_056_5", path)
    assert spec is not None and spec.loader is not None, "failed to load arc_utils"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arc_utils = _load_arc_utils()
bbox_coords = _arc_utils.bbox_coords
crop_bbox = _arc_utils.crop_bbox
find_components = _arc_utils.find_components


BLACK = 0
BLUE = 1
RED = 2

GRID_MIN = 14
GRID_MAX = 24
GENERATION_ATTEMPTS = 400


def _bbox_hw(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(box[2] - box[0] + 1), int(box[3] - box[1] + 1)


def _boxes_touch_or_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    gap: int = 0,
) -> bool:
    return not (
        int(a[2]) + gap < int(b[0])
        or int(b[2]) + gap < int(a[0])
        or int(a[3]) + gap < int(b[1])
        or int(b[3]) + gap < int(a[1])
    )


def _find_container_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    components = find_components(grid, bg=BLACK, colors=[RED], connectivity=4, monochromatic=True)
    candidates = []
    for component in components:
        box = tuple(map(int, component["bbox"]))
        h, w = _bbox_hw(box)
        if h < 6 or w < 6:
            continue
        if int(component["area"]) != h * w:
            continue
        r0, c0, r1, c1 = box
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RED):
            candidates.append(box)
    assert len(candidates) == 1, "expected a unique solid red rectangle"
    return candidates[0]


def _find_tile_bbox(grid: np.ndarray, container_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    blue_coords = np.argwhere(grid == BLUE)
    assert 2 <= int(blue_coords.shape[0]) <= 4, "expected 2 to 4 blue tile pixels"
    box = tuple(map(int, bbox_coords(blue_coords)))
    h, w = _bbox_hw(box)
    assert (h, w) in {(2, 3), (3, 2)}, "tile bbox must be 2x3 or 3x2"
    assert not _boxes_touch_or_overlap(container_bbox, box, gap=1), "tile must stay outside the rectangle"

    tile = crop_bbox(grid, box)
    assert set(map(int, np.unique(tile))).issubset({BLACK, BLUE}), "tile window must be binary"
    assert np.all(np.any(tile == BLUE, axis=1)), "tile blue pixels must span every row"
    assert np.all(np.any(tile == BLUE, axis=0)), "tile blue pixels must span every column"
    return box


def _invert_binary_tile(tile: np.ndarray) -> np.ndarray:
    assert tile.ndim == 2
    assert set(map(int, np.unique(tile))).issubset({BLACK, BLUE})
    return (BLUE - tile).astype(int)


def _fill_container(
    base: np.ndarray,
    container_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    r0, c0, r1, c1 = map(int, container_bbox)
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_h, inner_w = r1 - r0 - 1, c1 - c0 - 1
    tile_h, tile_w = map(int, tile.shape)
    assert inner_h >= tile_h * 2 and inner_w >= tile_w * 2

    out[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w] = BLACK
    for dr in range(inner_h):
        for dc in range(inner_w):
            out[inner_r0 + dr, inner_c0 + dc] = int(tile[dr % tile_h, dc % tile_w])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Hollow the solid red container, invert the small binary tile, and tile the inverted pattern into the container interior.

    Input structure:
    - The background is black.
    - There is exactly one solid red axis-aligned rectangle whose interior is at
      least two tile-heights tall and two tile-widths wide.
    - Outside that rectangle there is exactly one small binary tile whose bbox is
      2x3 or 3x2 and whose visible blue pixels span every row and column of that bbox.

    Rule:
    - Keep the outer one-pixel red border of the rectangle and clear its strict interior.
    - Read the tile from its tight bbox and invert its binary colors: blue becomes
      black and black becomes blue.
    - Starting at the container interior's top-left cell, repeat that inverted
      tile across the interior and clip any overhang at the bottom or right edge.
    - Leave the original tile in place unchanged.

    Key insight:
    - The red rectangle provides only the fill window, while the separate binary
      tile provides only the repeated content; the answer appears only after
      hollowing one object and inverting the other before combining them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "unexpected colors"

    container_bbox = _find_container_bbox(g)
    tile_bbox = _find_tile_bbox(g, container_bbox)
    tile = crop_bbox(g, tile_bbox)
    inverted = _invert_binary_tile(tile)
    return _fill_container(g, container_bbox, inverted)


def _valid_tile_masks(shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = map(int, shape)
    masks = []
    for bits in range(1 << (h * w)):
        arr = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=int).reshape(h, w)
        count = int(arr.sum())
        if not (2 <= count <= 4):
            continue
        if not np.all(np.any(arr == 1, axis=1)):
            continue
        if not np.all(np.any(arr == 1, axis=0)):
            continue
        row_patterns = {tuple(map(int, row)) for row in arr}
        col_patterns = {tuple(map(int, col)) for col in arr.T}
        if len(row_patterns) < min(2, h):
            continue
        if len(col_patterns) < min(2, w):
            continue
        masks.append(arr.astype(int))
    return masks


VALID_MASKS = {
    (2, 3): _valid_tile_masks((2, 3)),
    (3, 2): _valid_tile_masks((3, 2)),
}


def _visible_clipping(inverted_tile: np.ndarray, inner_h: int, inner_w: int) -> bool:
    tile_h, tile_w = map(int, inverted_tile.shape)
    r_rem = inner_h % tile_h
    c_rem = inner_w % tile_w
    if r_rem and np.any(inverted_tile[:r_rem, :] == BLUE):
        return True
    if c_rem and np.any(inverted_tile[:, :c_rem] == BLUE):
        return True
    return False


def _sample_inner_size(tile_h: int, tile_w: int, inverted_tile: np.ndarray) -> tuple[int, int] | None:
    heights = list(range(tile_h * 2, tile_h * 5 + 1))
    widths = list(range(tile_w * 2, tile_w * 5 + 1))
    random.shuffle(heights)
    random.shuffle(widths)
    for inner_h in heights:
        for inner_w in widths:
            if inner_h % tile_h == 0 and inner_w % tile_w == 0:
                continue
            if _visible_clipping(inverted_tile, inner_h, inner_w):
                return inner_h, inner_w
    return None


def _candidate_tile_positions(
    grid_h: int,
    grid_w: int,
    tile_h: int,
    tile_w: int,
    rect_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(grid_h - tile_h + 1):
        for c0 in range(grid_w - tile_w + 1):
            tile_box = (r0, c0, r0 + tile_h - 1, c0 + tile_w - 1)
            if _boxes_touch_or_overlap(rect_bbox, tile_box, gap=1):
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    rect_bbox = _find_container_bbox(inp)
    tile_bbox = _find_tile_bbox(inp, rect_bbox)
    rect_h, rect_w = _bbox_hw(rect_bbox)
    tile_h, tile_w = _bbox_hw(tile_bbox)
    inner_h, inner_w = rect_h - 2, rect_w - 2
    assert inner_h >= tile_h * 2 and inner_w >= tile_w * 2
    assert inner_h % tile_h != 0 or inner_w % tile_w != 0, "clipping must matter"

    original_tile = crop_bbox(inp, tile_bbox)
    inverted_tile = _invert_binary_tile(original_tile)
    assert np.any(inverted_tile == BLUE) and np.any(inverted_tile == BLACK)
    assert _visible_clipping(inverted_tile, inner_h, inner_w), "partial tile must be visible"

    expected = _fill_container(inp, rect_bbox, inverted_tile)
    assert np.array_equal(out, expected)

    no_invert = _fill_container(inp, rect_bbox, original_tile)
    assert not np.array_equal(out, no_invert), "color inversion must affect the result"

    overlay_only = np.array(inp, copy=True)
    ir0, ic0, ir1, ic1 = rect_bbox[0] + 1, rect_bbox[1] + 1, rect_bbox[2] - 1, rect_bbox[3] - 1
    for dr in range(ir1 - ir0 + 1):
        for dc in range(ic1 - ic0 + 1):
            if inverted_tile[dr % tile_h, dc % tile_w] == BLUE:
                overlay_only[ir0 + dr, ic0 + dc] = BLUE
    assert not np.array_equal(out, overlay_only), "hollowing must affect the result"

    single_stamp = np.array(inp, copy=True)
    single_stamp[ir0 : ir1 + 1, ic0 : ic1 + 1] = BLACK
    h_lim = min(tile_h, ir1 - ir0 + 1)
    w_lim = min(tile_w, ic1 - ic0 + 1)
    single_stamp[ir0 : ir0 + h_lim, ic0 : ic0 + w_lim] = inverted_tile[:h_lim, :w_lim]
    assert not np.array_equal(out, single_stamp), "full tiling must affect the result"

    assert np.array_equal(crop_bbox(inp, tile_bbox), crop_bbox(out, tile_bbox)), "source tile must stay unchanged"


def _generate_attempt() -> np.ndarray | None:
    tile_h, tile_w = random.choice([(2, 3), (3, 2)])
    masks = VALID_MASKS[(tile_h, tile_w)]
    random.shuffle(masks)

    for mask in masks:
        inverted = _invert_binary_tile(mask)
        size = _sample_inner_size(tile_h, tile_w, inverted)
        if size is None:
            continue
        inner_h, inner_w = size
        rect_h, rect_w = inner_h + 2, inner_w + 2

        min_h = max(GRID_MIN, rect_h + tile_h + 3)
        min_w = max(GRID_MIN, rect_w + tile_w + 3)
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue

        grid_h = random.randint(min_h, GRID_MAX)
        grid_w = random.randint(min_w, GRID_MAX)
        rect_r0 = random.randint(0, grid_h - rect_h)
        rect_c0 = random.randint(0, grid_w - rect_w)
        rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

        positions = _candidate_tile_positions(grid_h, grid_w, tile_h, tile_w, rect_bbox)
        if not positions:
            continue
        tile_r0, tile_c0 = positions[0]

        grid = np.zeros((grid_h, grid_w), dtype=int)
        grid[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = RED
        for rr in range(tile_h):
            for cc in range(tile_w):
                if mask[rr, cc] == 1:
                    grid[tile_r0 + rr, tile_c0 + cc] = BLUE

        try:
            out = main(grid)
            _semantic_checks(grid, out)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for opus/4/056/5")
