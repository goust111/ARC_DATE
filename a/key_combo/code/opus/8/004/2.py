from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg, paste, random_free_location


BG = 0
STRUCTURE_COLOR = 5
OBJECT_COLORS = tuple(color for color in range(1, 10) if color != STRUCTURE_COLOR)
GEN_ATTEMPTS = 400


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _mask_to_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    cells = {(int(r), int(c)) for r, c in coords}
    return _normalize_shape(cells)


def _rotate_mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = (rotated.shape, rotated.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(rotated.copy())
    return variants


def _rotate_shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    return [_mask_to_shape(mask) for mask in _rotate_mask_variants(_shape_to_mask(shape))]


def _rotation_key(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_rotate_shape_variants(shape))


def _is_rectangle(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(shape)
    return len(shape) == h * w


@lru_cache(maxsize=1)
def _polyomino_pool() -> list[tuple[tuple[int, int], ...]]:
    shapes = {((0, 0),)}
    pool: list[tuple[tuple[int, int], ...]] = []
    for _size in range(2, 7):
        next_shapes: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes:
            cell_set = set(shape)
            for r, c in shape:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in cell_set:
                        continue
                    next_shapes.add(_normalize_shape(cell_set | {(nr, nc)}))
        shapes = next_shapes
        pool.extend(sorted(shapes))
    return pool


@lru_cache(maxsize=1)
def _hole_shape_pool() -> list[tuple[tuple[int, int], ...]]:
    result = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for shape in _polyomino_pool():
        if len(shape) < 3 or len(shape) > 6:
            continue
        if _is_rectangle(shape):
            continue
        if len(_rotate_shape_variants(shape)) != 4:
            continue
        h, w = _shape_dims(shape)
        if h > 4 or w > 4:
            continue
        key = _rotation_key(shape)
        if key in seen:
            continue
        seen.add(key)
        result.append(shape)
    return result


@lru_cache(maxsize=1)
def _general_shape_pool() -> list[tuple[tuple[int, int], ...]]:
    result = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for shape in _polyomino_pool():
        if len(shape) < 3 or len(shape) > 6:
            continue
        h, w = _shape_dims(shape)
        if h > 4 or w > 4:
            continue
        key = _rotation_key(shape)
        if key in seen:
            continue
        seen.add(key)
        result.append(shape)
    return result


def _component_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[rows - r0, cols - c0] = grid[rows, cols]
    return patch


def _extract_holes(grid: np.ndarray, structure_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = structure_bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for coords in components_mask(patch == BG, connectivity=4):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == patch.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == patch.shape[1] - 1):
            continue
        rr0 = int(coords[:, 0].min())
        cc0 = int(coords[:, 1].min())
        rr1 = int(coords[:, 0].max())
        cc1 = int(coords[:, 1].max())
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        holes.append(
            {
                "mask": mask,
                "bbox": (r0 + rr0, c0 + cc0, r0 + rr1, c0 + cc1),
            }
        )
    return holes


def _choose_source_object(grid: np.ndarray) -> dict:
    objects_grid = grid.copy()
    objects_grid[objects_grid == STRUCTURE_COLOR] = BG
    objects = find_components(objects_grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert objects, "expected colored objects below the structure"

    counts = []
    for obj in objects:
        colors = {int(grid[int(r), int(c)]) for r, c in obj["coords"] if int(grid[int(r), int(c)]) != BG}
        counts.append(len(colors))
    best = max(counts)
    assert counts.count(best) == 1, "expected a unique color-diversity outlier"
    return objects[counts.index(best)]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the unique three-color outlier object into the matching hole of the upper gray structure.

    Input structure:
    - A black background contains one upper gray rectangular structure with several internal black holes.
    - Below it sit separated small colored 4-connected objects; exactly one object uses more distinct colors than the others.
    - The outlier object's binary shape matches exactly one hole after a 0/90/180/270-degree rotation.

    Rule:
    - Identify the uniquely most color-diverse small object as the stamp source.
    - Extract every internal hole from the gray structure and compare each against all rotations of the source object's mask.
    - Paste the uniquely matching rotated source into that hole with its colors preserved, then erase the source from its original location.

    Key insight:
    - Color diversity selects which object to move, while geometric rotation matching independently determines where and how it is embedded.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    structures = find_components(
        g,
        bg=bg,
        colors=[STRUCTURE_COLOR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert structures, "expected a gray structure"
    structure = max(structures, key=lambda comp: int(comp["area"]))
    holes = _extract_holes(g, structure["bbox"])
    assert holes, "expected holes inside the gray structure"

    source = _choose_source_object(g)
    source_patch = _component_patch(g, source["coords"])

    matches: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    seen_rotations: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(source_patch, k)
        key = (rotated.shape, (rotated != bg).tobytes())
        if key in seen_rotations:
            continue
        seen_rotations.add(key)
        rotated_mask = rotated != bg
        for hole in holes:
            if rotated_mask.shape == hole["mask"].shape and np.array_equal(rotated_mask, hole["mask"]):
                matches.append((rotated.copy(), hole["bbox"]))

    assert len(matches) == 1, "expected a unique hole/rotation match"
    rotated_patch, hole_bbox = matches[0]

    out = g.copy()
    for r, c in source["coords"]:
        out[int(r), int(c)] = bg
    hr0, hc0, _hr1, _hc1 = hole_bbox
    paste(out, rotated_patch, hr0, hc0, transparent=bg)
    return out


def _shape_patch(shape: tuple[tuple[int, int], ...], colors: tuple[int, ...]) -> np.ndarray:
    mask = _shape_to_mask(shape)
    coords = np.argwhere(mask)
    patch = np.zeros(mask.shape, dtype=int)
    assignments = list(colors)
    while len(assignments) < len(coords):
        assignments.append(random.choice(colors))
    random.shuffle(assignments)
    for (r, c), color in zip(coords, assignments):
        patch[int(r), int(c)] = int(color)
    return patch


def _match_rotation(source_patch: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(source_patch, k)
        if rotated.shape == hole_mask.shape and np.array_equal(rotated != BG, hole_mask):
            return rotated.copy()
    raise AssertionError("source patch should match the selected hole")


def _build_structure(
    hole_shapes: list[tuple[tuple[int, int], ...]],
) -> tuple[np.ndarray, list[dict]] | None:
    rect_h = random.randint(6, 8)
    rect_w = random.randint(10, 13)
    interior_h = rect_h - 2
    interior_w = rect_w - 2
    interior = np.zeros((interior_h, interior_w), dtype=int)
    hole_infos: list[dict] = []
    for shape in hole_shapes:
        rotated_shape = random.choice(_rotate_shape_variants(shape))
        hole_mask = _shape_to_mask(rotated_shape).astype(int)
        try:
            r0, c0 = random_free_location(
                interior,
                hole_mask,
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(interior, hole_mask, r0, c0, transparent=0)
        hole_infos.append(
            {
                "key": _rotation_key(shape),
                "mask": hole_mask.astype(bool),
                "bbox": (r0 + 1, c0 + 1, r0 + hole_mask.shape[0], c0 + hole_mask.shape[1]),
            }
        )

    structure = np.full((rect_h, rect_w), STRUCTURE_COLOR, dtype=int)
    structure[1:-1, 1:-1][interior.astype(bool)] = BG
    return structure, hole_infos


def _select_shapes() -> tuple[list[tuple[tuple[int, int], ...]], tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]]]:
    hole_pool = _hole_shape_pool()
    general_pool = _general_shape_pool()

    hole_count = random.randint(2, 3)
    matched_shape = random.choice(hole_pool)
    hole_shapes = [matched_shape]
    used_keys = {_rotation_key(matched_shape)}

    other_holes = [shape for shape in hole_pool if _rotation_key(shape) not in used_keys]
    random.shuffle(other_holes)
    for shape in other_holes:
        if len(hole_shapes) == hole_count:
            break
        key = _rotation_key(shape)
        if key in used_keys:
            continue
        used_keys.add(key)
        hole_shapes.append(shape)

    object_count = random.randint(3, 4)
    distractor_needed = object_count - 1
    distractor_pool = [shape for shape in general_pool if _rotation_key(shape) not in used_keys]
    random.shuffle(distractor_pool)
    distractors = distractor_pool[:distractor_needed]
    assert len(hole_shapes) == hole_count and len(distractors) == distractor_needed
    return hole_shapes, matched_shape, distractors


def _expected_output(
    grid: np.ndarray,
    source_coords: np.ndarray,
    rotated_source: np.ndarray,
    hole_bbox_abs: tuple[int, int, int, int],
) -> np.ndarray:
    out = grid.copy()
    for r, c in source_coords:
        out[int(r), int(c)] = BG
    r0, c0, _r1, _c1 = hole_bbox_abs
    paste(out, rotated_source, r0, c0, transparent=BG)
    return out


def _generate_case() -> np.ndarray | None:
    hole_shapes, matched_shape, distractor_shapes = _select_shapes()
    structure_pack = _build_structure(hole_shapes)
    if structure_pack is None:
        return None
    structure, hole_infos_local = structure_pack

    struct_h, struct_w = structure.shape
    top_margin = random.randint(1, 2)
    gap = random.randint(2, 3)
    lower_h = random.randint(7, 10)
    grid_h = top_margin + struct_h + gap + lower_h
    grid_w = random.randint(max(struct_w + 2, 16), 22)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    struct_r = top_margin
    struct_c = random.randint(1, grid_w - struct_w - 1)
    paste(grid, structure, struct_r, struct_c)

    hole_info_by_key = {info["key"]: info for info in hole_infos_local}
    matched_hole = hole_info_by_key[_rotation_key(matched_shape)]
    hole_bbox_abs = (
        struct_r + matched_hole["bbox"][0],
        struct_c + matched_hole["bbox"][1],
        struct_r + matched_hole["bbox"][2],
        struct_c + matched_hole["bbox"][3],
    )

    source_shape = random.choice(_rotate_shape_variants(matched_shape))
    source_colors = tuple(random.sample(OBJECT_COLORS, 3))
    source_patch = _shape_patch(source_shape, source_colors)
    rotated_source = _match_rotation(source_patch, matched_hole["mask"])

    object_patches = [source_patch]
    for shape in distractor_shapes:
        rotated_shape = random.choice(_rotate_shape_variants(shape))
        colors = tuple(random.sample(OBJECT_COLORS, 2))
        object_patches.append(_shape_patch(rotated_shape, colors))
    random.shuffle(object_patches)

    lower_top = struct_r + struct_h + gap
    lower_canvas = grid[lower_top:, :]
    source_coords_abs: np.ndarray | None = None
    for patch in object_patches:
        try:
            r0, c0 = random_free_location(
                lower_canvas,
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(lower_canvas, patch, r0, c0, transparent=BG)
        if np.array_equal(patch, source_patch):
            local_coords = np.argwhere(patch != BG)
            source_coords_abs = local_coords + np.array([lower_top + r0, c0])

    if source_coords_abs is None:
        return None

    expected = _expected_output(grid, source_coords_abs, rotated_source, hole_bbox_abs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
