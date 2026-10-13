from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, components_mask, crop_bbox, find_components, paste, random_free_location


QUADRANT_TO_COLOR = {
    (-1, -1): Color.BLUE,
    (-1, 1): Color.RED,
    (1, -1): Color.GREEN,
    (1, 1): Color.YELLOW,
}

SHAPE_POOL = (
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}),
)

STRUCTURE_SIZES = ((12, 12), (12, 14), (14, 12), (14, 14))
OUTER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _mask_from_cells(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> np.ndarray:
    normalized = sorted(_normalize_cells(set(cells)))
    h = max(r for r, _ in normalized) + 1
    w = max(c for _, c in normalized) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in normalized:
        mask[r, c] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask.astype(bool))
    assert box is not None
    return crop_bbox(mask.astype(bool), box).astype(bool)


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    shifted = coords - np.array([r0, c0])
    h = int(shifted[:, 0].max()) + 1
    w = int(shifted[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _rotation_masks(mask: np.ndarray) -> list[np.ndarray]:
    return [_normalize_mask(np.rot90(mask, k)) for k in range(4)]


def _extract_holes(grid: np.ndarray, structure_box: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = structure_box
    patch = crop_bbox(grid, structure_box)
    holes = []
    for coords in components_mask(patch == Color.BLACK, connectivity=4):
        touches_border = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == patch.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == patch.shape[1] - 1)
        )
        if touches_border:
            continue
        abs_coords = coords + np.array([r0, c0])
        hole_box = (
            int(abs_coords[:, 0].min()),
            int(abs_coords[:, 1].min()),
            int(abs_coords[:, 0].max()),
            int(abs_coords[:, 1].max()),
        )
        holes.append(
            {
                "coords": abs_coords,
                "bbox": hole_box,
                "mask": _component_mask(abs_coords),
            }
        )
    return holes


def _quadrant_color(hole_coords: np.ndarray, structure_box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = structure_box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    centroid_r = float(hole_coords[:, 0].mean())
    centroid_c = float(hole_coords[:, 1].mean())
    row_sign = -1 if centroid_r < center_r else 1
    col_sign = -1 if centroid_c < center_c else 1
    return QUADRANT_TO_COLOR[(row_sign, col_sign)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outer sprite into its matching hole, recoloring it from the hole's quadrant inside the gray structure.

    Input structure:
    - A black background contains one gray rectangular structure with four internal black holes.
    - The holes are 4-connected shapes, one in each quadrant of the gray structure.
    - Four colored sprites sit outside the structure, and each sprite matches exactly one hole up to rotation.

    Rule:
    - Find the gray structure and extract its internal black hole masks.
    - For each outer sprite, identify the uniquely matching hole by exact binary-shape equality under 0/90/180/270 degree rotations.
    - Classify the matched hole by the sign of its centroid offset from the structure center: upper-left -> blue, upper-right -> red, lower-left -> green, lower-right -> yellow.
    - Erase the original sprite, recolor it with that quadrant color, rotate it into the matching orientation, and paste it into the hole.

    Key insight:
    - Shape determines which hole a sprite fills, while the final color is controlled only by the matched hole's position inside the structure rather than by the sprite's original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    structures = find_components(
        g,
        bg=Color.BLACK,
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(structures) == 1, "expected exactly one gray structure"
    structure_box = structures[0]["bbox"]

    holes = _extract_holes(g, structure_box)
    assert len(holes) == 4, "expected four internal holes"

    filtered = g.copy()
    filtered[filtered == Color.GREY] = Color.BLACK
    sprites = find_components(
        filtered,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(sprites) == 4, "expected four outer sprites"

    out = g.copy()
    used_holes: set[int] = set()
    for sprite in sprites:
        sprite_mask = _component_mask(sprite["coords"])
        candidates: list[tuple[int, int]] = []
        for hole_index, hole in enumerate(holes):
            for rotation_k, rotated in enumerate(_rotation_masks(sprite_mask)):
                if rotated.shape == hole["mask"].shape and np.array_equal(rotated, hole["mask"]):
                    candidates.append((hole_index, rotation_k))

        available_holes = sorted({hole_index for hole_index, _ in candidates if hole_index not in used_holes})
        assert len(available_holes) == 1, "each sprite must match one unused hole"
        hole_index = available_holes[0]
        used_holes.add(hole_index)

        hole = holes[hole_index]
        color = _quadrant_color(hole["coords"], structure_box)

        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = Color.BLACK
        patch = np.zeros(hole["mask"].shape, dtype=int)
        patch[hole["mask"]] = color
        paste(out, patch, hole["bbox"][0], hole["bbox"][1], transparent=Color.BLACK)

    assert len(used_holes) == 4, "all holes must be filled"
    return out


def _random_rotated_mask(mask: np.ndarray) -> np.ndarray:
    return _normalize_mask(np.rot90(mask, random.randrange(4)))


def _quadrant_anchor_ranges(struct_h: int, struct_w: int, hole_h: int, hole_w: int, quadrant: tuple[int, int]) -> tuple[list[int], list[int]]:
    half_h = struct_h // 2
    half_w = struct_w // 2

    if quadrant[0] < 0:
        row_min, row_max = 1, half_h - hole_h - 1
    else:
        row_min, row_max = half_h + 1, struct_h - hole_h - 2

    if quadrant[1] < 0:
        col_min, col_max = 1, half_w - hole_w - 1
    else:
        col_min, col_max = half_w + 1, struct_w - hole_w - 2

    return list(range(row_min, row_max + 1)), list(range(col_min, col_max + 1))


def _place_holes(struct_h: int, struct_w: int) -> tuple[np.ndarray, list[dict]] | None:
    structure = np.full((struct_h, struct_w), Color.GREY, dtype=int)
    chosen_shapes = random.sample(SHAPE_POOL, 4)
    quadrants = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    holes = []

    for quadrant, shape in zip(quadrants, chosen_shapes):
        mask = _random_rotated_mask(_mask_from_cells(shape))
        hole_h, hole_w = mask.shape
        row_choices, col_choices = _quadrant_anchor_ranges(struct_h, struct_w, hole_h, hole_w, quadrant)
        if not row_choices or not col_choices:
            return None
        top = random.choice(row_choices)
        left = random.choice(col_choices)
        structure[top : top + hole_h, left : left + hole_w][mask] = Color.BLACK

        local_coords = np.argwhere(mask) + np.array([top, left])
        holes.append(
            {
                "quadrant": quadrant,
                "mask": mask,
                "bbox": (top, left, top + hole_h - 1, left + hole_w - 1),
                "coords": local_coords,
            }
        )

    return structure, holes


def _sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _absolute_holes(local_holes: list[dict], top: int, left: int) -> list[dict]:
    holes = []
    for hole in local_holes:
        r0, c0, r1, c1 = hole["bbox"]
        holes.append(
            {
                "quadrant": hole["quadrant"],
                "mask": hole["mask"],
                "bbox": (r0 + top, c0 + left, r1 + top, c1 + left),
                "coords": hole["coords"] + np.array([top, left]),
            }
        )
    return holes


def _render_expected(grid: np.ndarray, structure_box: tuple[int, int, int, int], holes: list[dict], sprite_coords: list[np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for coords in sprite_coords:
        out[coords[:, 0], coords[:, 1]] = Color.BLACK
    for hole in holes:
        patch = np.zeros(hole["mask"].shape, dtype=int)
        patch[hole["mask"]] = _quadrant_color(hole["coords"], structure_box)
        paste(out, patch, hole["bbox"][0], hole["bbox"][1], transparent=Color.BLACK)
    return out


def _generate_attempt() -> np.ndarray | None:
    struct_h, struct_w = random.choice(STRUCTURE_SIZES)
    placed = _place_holes(struct_h, struct_w)
    if placed is None:
        return None
    structure_patch, local_holes = placed

    grid_h = struct_h + random.randint(8, 11)
    grid_w = random.randint(max(18, struct_w + 4), min(24, 30))
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    structure_top = random.randint(1, 2)
    left_min = 1
    left_max = grid_w - struct_w - 1
    if left_max < left_min:
        return None
    structure_left = random.randint(left_min, left_max)
    paste(grid, structure_patch, structure_top, structure_left, transparent=Color.BLACK)

    structure_box = (
        structure_top,
        structure_left,
        structure_top + struct_h - 1,
        structure_left + struct_w - 1,
    )
    holes = _absolute_holes(local_holes, structure_top, structure_left)

    band_top = structure_box[2] + 2
    if band_top >= grid_h - 1:
        return None

    colors = random.sample(OUTER_COLORS, 4)
    sprite_order = list(range(4))
    random.shuffle(sprite_order)
    sprite_coords: list[np.ndarray] = []

    for color, hole_index in zip(colors, sprite_order):
        sprite_mask = _random_rotated_mask(holes[hole_index]["mask"])
        patch = _sprite_patch(sprite_mask, color)
        band = grid[band_top:, :]
        try:
            r0, c0 = random_free_location(
                band,
                patch,
                transparent=Color.BLACK,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        abs_r0 = band_top + r0
        paste(grid, patch, abs_r0, c0, transparent=Color.BLACK)
        coords = np.argwhere(patch != Color.BLACK) + np.array([abs_r0, c0])
        sprite_coords.append(coords)

    expected = _render_expected(grid, structure_box, holes, sprite_coords)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
