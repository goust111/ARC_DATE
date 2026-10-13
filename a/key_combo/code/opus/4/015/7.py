from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, first_free_location, paste


BG = Color.BLACK
ANCHOR = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
CORE_NEIGHBORS = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1), (-1, -1), (1, 1)]
AREA_CHOICES = [3, 4, 5, 6, 7, 8, 9]
GRID_MIN = 13
GRID_MAX = 19
PLACEMENT_CANVAS = 19
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 250


def _sprite_infos(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=8, monochromatic=False, return_format="dict")
    infos: list[dict] = []
    for comp in comps:
        comp_bbox = comp["bbox"]
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = map(int, comp_bbox)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        sprite[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
        anchor_coords = np.argwhere(sprite == ANCHOR)
        assert len(anchor_coords) == 1, "each sprite must contain exactly one gray anchor"
        anchor_r, anchor_c = map(int, anchor_coords[0])
        fg_colors = sorted(set(map(int, sprite.ravel())) - {BG, ANCHOR})
        assert len(fg_colors) == 1, "each sprite must use exactly one non-gray foreground color"
        fg_area = int(np.count_nonzero((sprite != BG) & (sprite != ANCHOR)))
        assert fg_area > 0, "each sprite must contain non-gray foreground pixels"
        infos.append(
            {
                "bbox": tuple(map(int, comp_bbox)),
                "sprite": sprite,
                "anchor": (anchor_r, anchor_c),
                "center": (sprite.shape[0] // 2, sprite.shape[1] // 2),
                "fg_area": fg_area,
                "color": int(fg_colors[0]),
            }
        )
    return infos


def _overlay(infos: list[dict], order: list[dict], *, align_key: str) -> np.ndarray:
    top = max(int(info[align_key][0]) for info in infos)
    left = max(int(info[align_key][1]) for info in infos)
    bottom = max(int(info["sprite"].shape[0] - 1 - info[align_key][0]) for info in infos)
    right = max(int(info["sprite"].shape[1] - 1 - info[align_key][1]) for info in infos)

    canvas = np.full((top + bottom + 1, left + right + 1), BG, dtype=int)
    for info in order:
        ar, ac = map(int, info[align_key])
        paste(canvas, info["sprite"], top - ar, left - ac, transparent=BG)

    out_bbox = bbox(canvas != BG)
    assert out_bbox is not None
    return crop_bbox(canvas, out_bbox)


def _sorted_by_area(infos: list[dict]) -> list[dict]:
    return sorted(infos, key=lambda info: (-int(info["fg_area"]), int(info["bbox"][0]), int(info["bbox"][1])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the separated sprites by gray-anchor alignment, with larger areas drawn first and smaller areas drawn last.

    Input structure:
    - The grid has black background and 3-4 separated multicolor sprites.
    - Each sprite contains exactly one gray anchor pixel and pixels of one other non-black color.
    - Every sprite is cropped by its own odd-height, odd-width bounding box, but the gray anchor need not be at the box center.

    Rule:
    - Extract each sprite by cropping its full bounding box, treating black as transparent.
    - Count only the non-black, non-gray pixels of each sprite to get its area.
    - Sort sprites from largest area to smallest area.
    - Paste them in that order onto one canvas so all gray anchors coincide; later sprites overwrite earlier non-black pixels.
    - Crop the result to the smallest bounding box containing all non-black pixels.

    Key insight:
    - Area does not decide position; it decides layer depth, so smaller sprites stay on top after anchor-aligned transparent overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = _sprite_infos(g)
    assert len(infos) >= 2, "expected multiple separated sprites"
    return _overlay(infos, _sorted_by_area(infos), align_key="anchor")


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS8:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _grow_shape(area: int, core: list[tuple[int, int]], preferred: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    total = int(area) + 1
    for _ in range(SHAPE_ATTEMPTS):
        cells: set[tuple[int, int]] = {(0, 0)}
        seed_count = min(len(core), random.choice([1, 2]))
        for coord in random.sample(core, k=seed_count):
            cells.add(coord)

        while len(cells) < total:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in NEIGHBORS8
                if (r + dr, c + dc) not in cells and -3 <= r + dr <= 3 and -3 <= c + dc <= 3
            }
            if not frontier:
                break
            overlap_choices = [coord for coord in frontier if coord in preferred]
            if overlap_choices and random.random() < 0.65:
                choice = random.choice(overlap_choices)
            else:
                choice = random.choice(sorted(frontier))
            cells.add(choice)

        if len(cells) != total:
            continue
        if not _is_connected(cells):
            continue
        r0, c0, r1, c1 = _shape_bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
            continue
        if h > 7 or w > 7:
            continue
        return cells
    return None


def _shape_to_patch(color_cells: set[tuple[int, int]], color: int) -> np.ndarray:
    cells = set(color_cells) | {(0, 0)}
    r0, c0, r1, c1 = _shape_bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in color_cells:
        patch[r - r0, c - c0] = int(color)
    patch[-r0, -c0] = ANCHOR
    return patch


def _pairwise_overlap_stats(shapes: list[set[tuple[int, int]]]) -> tuple[int, set[tuple[int, int]]]:
    pair_count = 0
    shared_coords: set[tuple[int, int]] = set()
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            overlap = shapes[i] & shapes[j]
            if overlap:
                pair_count += 1
                shared_coords |= overlap
    return pair_count, shared_coords


def _embed_random(cropped: np.ndarray) -> np.ndarray | None:
    h, w = cropped.shape
    if h + 2 > GRID_MAX or w + 2 > GRID_MAX:
        return None
    H = random.randint(max(GRID_MIN, h + 2), max(GRID_MIN, min(GRID_MAX, h + 6)))
    W = random.randint(max(GRID_MIN, w + 2), max(GRID_MIN, min(GRID_MAX, w + 6)))
    out = np.full((H, W), BG, dtype=int)
    r0 = random.randint(1, H - h - 1)
    c0 = random.randint(1, W - w - 1)
    paste(out, cropped, r0, c0, transparent=BG)
    return out


def _input_row_major_order(grid: np.ndarray) -> list[dict]:
    infos = _sprite_infos(grid)
    return sorted(infos, key=lambda info: (int(info["bbox"][0]), int(info["bbox"][1])))


def _gen_attempt() -> np.ndarray | None:
    n_objects = 4 if random.random() < 0.8 else 3
    area_pool = [3, 4, 5, 6, 7] if n_objects == 4 else AREA_CHOICES
    areas = random.sample(area_pool, k=n_objects)
    colors = random.sample(PALETTE, k=n_objects)
    core_size = random.choice([2, 3])
    core = random.sample(CORE_NEIGHBORS, k=core_size)

    shapes: list[set[tuple[int, int]]] = []
    preferred: set[tuple[int, int]] = set(core)
    for area in sorted(areas, reverse=True):
        shape = _grow_shape(area, core, preferred)
        if shape is None:
            return None
        color_cells = set(shape) - {(0, 0)}
        shapes.append(color_cells)
        preferred |= color_cells

    pair_count, shared_coords = _pairwise_overlap_stats(shapes)
    if pair_count < 2 or len(shared_coords) < 2:
        return None

    patches = [_shape_to_patch(shape, color) for shape, color in zip(shapes, colors)]
    placement_canvas = np.full((PLACEMENT_CANVAS, PLACEMENT_CANVAS), BG, dtype=int)
    for patch in random.sample(patches, k=len(patches)):
        loc = first_free_location(
            placement_canvas,
            patch,
            transparent=BG,
            border=1,
            padding=2,
            padding_connectivity=8,
        )
        if loc is None:
            return None
        paste(placement_canvas, patch, int(loc[0]), int(loc[1]), transparent=BG)

    placed_bbox = bbox(placement_canvas != BG)
    if placed_bbox is None:
        return None
    cropped_input = crop_bbox(placement_canvas, placed_bbox)
    grid = _embed_random(cropped_input)
    if grid is None:
        return None

    true_infos = _sprite_infos(grid)
    if len(true_infos) != n_objects:
        return None
    true_out = _overlay(true_infos, _sorted_by_area(true_infos), align_key="anchor")
    oracle_out = main(grid)
    if oracle_out.shape != true_out.shape or not np.array_equal(oracle_out, true_out):
        return None

    visible_colors = set(map(int, true_out.ravel())) - {BG, ANCHOR}
    if visible_colors != {int(info["color"]) for info in true_infos}:
        return None

    total_nonblack = sum(int(np.count_nonzero(info["sprite"] != BG)) for info in true_infos)
    union_nonblack = int(np.count_nonzero(true_out != BG))
    overlap_loss = total_nonblack - union_nonblack
    if overlap_loss < 2 or overlap_loss > total_nonblack // 2:
        return None

    center_out = _overlay(true_infos, _sorted_by_area(true_infos), align_key="center")
    if center_out.shape == true_out.shape and np.array_equal(center_out, true_out):
        return None

    ascending = sorted(true_infos, key=lambda info: (int(info["fg_area"]), int(info["bbox"][0]), int(info["bbox"][1])))
    ascending_out = _overlay(true_infos, ascending, align_key="anchor")
    if ascending_out.shape == true_out.shape and np.array_equal(ascending_out, true_out):
        return None

    row_major_out = _overlay(true_infos, _input_row_major_order(grid), align_key="anchor")
    if row_major_out.shape == true_out.shape and np.array_equal(row_major_out, true_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
