import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
PALETTE_COLORS = (1, 2, 3, 4)
OBJECT_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> List[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: List[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _bbox_area(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _coords_to_mask(shape: Tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _count_holes(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty object"
    r0, c0, r1, c1 = _bbox(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    holes = 0
    for comp in _components(empty):
        comp_r = comp[:, 0]
        comp_c = comp[:, 1]
        touches_edge = (
            (comp_r == 0).any()
            or (comp_r == sub.shape[0] - 1).any()
            or (comp_c == 0).any()
            or (comp_c == sub.shape[1] - 1).any()
        )
        if not touches_edge:
            holes += 1
    return holes


def _palette_components(grid: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    comps: List[Tuple[int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BG, OBJECT_COLOR)):
        for comp in _components(grid == color):
            comps.append((color, comp))
    assert comps, "expected nested palette rings"
    return comps


def _extract_palette(grid: np.ndarray) -> List[int]:
    comps = _palette_components(grid)
    areas = [_bbox_area(comp) for _color, comp in comps]
    assert len(set(areas)) == len(areas), "palette ring bounding-box areas must be unique"
    comps.sort(key=lambda item: _bbox_area(item[1]), reverse=True)
    return [color for color, _comp in comps]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the nested outline palette, recolor each gray holed object by hole count, then erase the palette.

    Input structure:
    - Black background contains a nested stack of monochromatic outline rings and several disjoint gray objects.
    - Each ring has a distinct non-gray color, is 4-connected, and has a strictly larger bounding box than the ring it encloses.
    - Each gray object is 4-connected and contains one or more enclosed black 4-connected holes.

    Rule:
    - Gather every non-black, non-gray ring and sort them by bounding-box area from largest to smallest to form palette P.
    - For each gray object, count its enclosed holes H and recolor the whole object with P[H-1].
    - Keep the object geometry and black holes unchanged.
    - Erase every palette ring pixel to black.

    Key insight:
    - The hole count does not map to a fixed color: it indexes a runtime palette encoded by the nested ring order in the input itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _extract_palette(g)
    out = g.copy()

    palette_mask = (g != BG) & (g != OBJECT_COLOR)
    out[palette_mask] = BG

    gray_components = _components(g == OBJECT_COLOR)
    assert gray_components, "expected gray objects"
    for comp in gray_components:
        obj_mask = _coords_to_mask(g.shape, comp)
        hole_count = _count_holes(obj_mask)
        assert 1 <= hole_count <= len(palette), "hole count must index the decoded palette"
        out[obj_mask] = palette[hole_count - 1]

    return out


def _ring_from_bbox(height: int, width: int) -> np.ndarray:
    assert height >= 3 and width >= 3
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = 1
    patch[-1, :] = 1
    patch[:, 0] = 1
    patch[:, -1] = 1
    return patch.astype(bool)


def _generate_nested_rings(n_layers: int) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    assert 3 <= n_layers <= 4
    colors = random.sample(list(PALETTE_COLORS), n_layers)

    outer_h = random.randint(2 * n_layers + 1, 12)
    outer_w = random.randint(2 * n_layers + 1, 10)

    boxes: List[Tuple[int, int, int, int]] = [(0, 0, outer_h, outer_w)]
    cur_r, cur_c, cur_h, cur_w = boxes[0]
    for layer_idx in range(1, n_layers):
        remaining_after = n_layers - layer_idx - 1
        min_h = 3 + 2 * remaining_after
        min_w = 3 + 2 * remaining_after

        inner_h_max = cur_h - 2
        inner_w_max = cur_w - 2
        assert min_h <= inner_h_max and min_w <= inner_w_max

        inner_h = random.randint(min_h, inner_h_max)
        inner_w = random.randint(min_w, inner_w_max)
        off_r = random.randint(1, cur_h - inner_h - 1)
        off_c = random.randint(1, cur_w - inner_w - 1)

        cur_r += off_r
        cur_c += off_c
        cur_h = inner_h
        cur_w = inner_w
        boxes.append((cur_r, cur_c, cur_h, cur_w))

    patch = np.zeros((outer_h, outer_w), dtype=int)
    layers: List[Dict[str, object]] = []
    for color, (r0, c0, h, w) in zip(colors, boxes):
        ring = _ring_from_bbox(h, w)
        local_coords = np.argwhere(ring)
        coords = local_coords + np.array([r0, c0])
        patch[coords[:, 0], coords[:, 1]] = color
        layers.append(
            {
                "color": color,
                "coords": coords,
                "bbox_area": h * w,
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
            }
        )

    return patch, layers


def _checkerboard_hole_sites(height: int, width: int) -> List[Tuple[int, int]]:
    sites = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if (r + c) % 2 == 0:
                sites.append((r, c))
    random.shuffle(sites)
    return sites


def _single_hole_shape(height: int, width: int) -> np.ndarray:
    hole = np.zeros((height, width), dtype=bool)
    shapes = []
    center_r = random.randint(1, height - 2)
    center_c = random.randint(1, width - 2)
    shapes.append([(center_r, center_c)])
    if center_c + 1 <= width - 2:
        shapes.append([(center_r, center_c), (center_r, center_c + 1)])
    if center_r + 1 <= height - 2:
        shapes.append([(center_r, center_c), (center_r + 1, center_c)])
    if center_r + 1 <= height - 2 and center_c + 1 <= width - 2:
        shapes.append([(center_r, center_c), (center_r + 1, center_c), (center_r, center_c + 1)])
        shapes.append([(center_r, center_c), (center_r + 1, center_c), (center_r + 1, center_c + 1)])
    coords = random.choice(shapes)
    for r, c in coords:
        hole[r, c] = True
    return hole


def _generate_object_patch(hole_count: int) -> Tuple[np.ndarray, Dict[str, object]]:
    for _ in range(200):
        height = random.randint(5, 7)
        width = random.randint(5, 7)

        obj = np.full((height, width), OBJECT_COLOR, dtype=int)
        holes_mask = np.zeros((height, width), dtype=bool)

        if hole_count == 1:
            holes_mask = _single_hole_shape(height, width)
        else:
            sites = _checkerboard_hole_sites(height, width)
            if len(sites) < hole_count:
                continue
            for r, c in sites[:hole_count]:
                holes_mask[r, c] = True

        obj[holes_mask] = BG

        if _count_holes(obj == OBJECT_COLOR) != hole_count:
            continue

        area = int(np.sum(obj == OBJECT_COLOR))
        if not (15 <= area <= 35):
            continue

        return obj, {"hole_count": hole_count, "area": area}

    raise RuntimeError(f"failed to build object patch with {hole_count} holes")


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    row_bounds: Tuple[int, int],
    col_bounds: Tuple[int, int],
) -> Tuple[int, int] | None:
    ph, pw = patch.shape
    row_start, row_stop = row_bounds
    col_start, col_stop = col_bounds
    if row_stop - row_start < ph or col_stop - col_start < pw:
        return None

    candidates = [
        (r, c)
        for r in range(row_start, row_stop - ph + 1)
        for c in range(col_start, col_stop - pw + 1)
    ]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(grid.shape[0], r0 + ph + 1)
        cc1 = min(grid.shape[1], c0 + pw + 1)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        patch_mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = patch[patch_mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch_mask
        return r0, c0
    return None


def _metadata_output(
    grid: np.ndarray,
    ring_layers: Sequence[Dict[str, object]],
    placed_objects: Sequence[Dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    palette = [int(layer["color"]) for layer in sorted(ring_layers, key=lambda item: int(item["bbox_area"]), reverse=True)]
    for layer in ring_layers:
        coords = np.asarray(layer["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BG
    for obj in placed_objects:
        coords = np.asarray(obj["coords"], dtype=int)
        color = palette[int(obj["hole_count"]) - 1]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _reject_easy_shortcuts(
    palette: Sequence[int],
    placed_objects: Sequence[Dict[str, object]],
) -> bool:
    if list(palette) == sorted(palette) or list(palette) == sorted(palette, reverse=True):
        return True

    if len(placed_objects) >= 3:
        row_major = sorted(
            placed_objects,
            key=lambda item: (
                int(item["bbox"][0]),
                int(item["bbox"][1]),
            ),
        )
        hole_seq = [int(item["hole_count"]) for item in row_major]
        if hole_seq == sorted(hole_seq) or hole_seq == sorted(hole_seq, reverse=True):
            return True

        area_rank_seq = [
            int(item["hole_count"])
            for item in sorted(
                placed_objects,
                key=lambda item: (int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])),
            )
        ]
        if area_rank_seq == sorted(area_rank_seq) or area_rank_seq == sorted(area_rank_seq, reverse=True):
            return True

    return False


def _validate_sample(
    grid: np.ndarray,
    ring_layers: Sequence[Dict[str, object]],
    placed_objects: Sequence[Dict[str, object]],
) -> bool:
    try:
        decoded_palette = _extract_palette(grid)
        intended_palette = [
            int(layer["color"])
            for layer in sorted(ring_layers, key=lambda item: int(item["bbox_area"]), reverse=True)
        ]
        if decoded_palette != intended_palette:
            return False

        gray_comps = _components(grid == OBJECT_COLOR)
        if len(gray_comps) != len(placed_objects):
            return False
        actual_holes = sorted(_count_holes(_coords_to_mask(grid.shape, comp)) for comp in gray_comps)
        intended_holes = sorted(int(item["hole_count"]) for item in placed_objects)
        if actual_holes != intended_holes:
            return False

        expected = _metadata_output(grid, ring_layers, placed_objects)
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False

        if _reject_easy_shortcuts(decoded_palette, placed_objects):
            return False
    except AssertionError:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.choice([3, 4])
    n_objects = 3 if n_layers == 3 else random.choice([3, 4])
    hole_counts = random.sample(list(range(1, n_layers + 1)), n_objects)

    left_patch, ring_layers = _generate_nested_rings(n_layers)
    left_h, left_w = left_patch.shape

    object_specs = []
    for hole_count in hole_counts:
        patch, meta = _generate_object_patch(hole_count)
        object_specs.append((patch, meta))

    max_obj_w = max(patch.shape[1] for patch, _meta in object_specs)
    max_obj_h = max(patch.shape[0] for patch, _meta in object_specs)

    min_h = max(18, left_h + 2, n_objects * (max_obj_h + 1) + 2)
    min_w = max(20, left_w + max_obj_w + 5)
    if min_h > 30 or min_w > 30:
        return None

    max_h = min(30, max(min_h, 26))
    max_w = min(30, max(min_w, 29))
    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    left_row = random.randint(1, height - left_h - 1)
    left_col_max = max(1, width // 2 - left_w)
    left_col = random.randint(1, left_col_max)
    grid[left_row : left_row + left_h, left_col : left_col + left_w][left_patch != BG] = left_patch[left_patch != BG]
    occupied[left_row : left_row + left_h, left_col : left_col + left_w] |= left_patch != BG

    placed_objects: List[Dict[str, object]] = []
    right_start = max(width // 2, left_col + left_w + 2)
    if right_start >= width - max_obj_w:
        return None

    shuffled_specs = object_specs[:]
    random.shuffle(shuffled_specs)
    for patch, meta in shuffled_specs:
        loc = _place_patch(
            grid,
            occupied,
            patch,
            row_bounds=(1, height - 1),
            col_bounds=(right_start, width - 1),
        )
        if loc is None:
            return None
        r0, c0 = loc
        coords = np.argwhere(patch == OBJECT_COLOR) + np.array([r0, c0])
        placed_objects.append(
            {
                "hole_count": int(meta["hole_count"]),
                "area": int(meta["area"]),
                "coords": coords,
                "bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1),
            }
        )

    shifted_layers: List[Dict[str, object]] = []
    for layer in ring_layers:
        coords = np.asarray(layer["coords"], dtype=int) + np.array([left_row, left_col])
        r0, c0, r1, c1 = layer["bbox"]
        shifted_layers.append(
            {
                "color": int(layer["color"]),
                "coords": coords,
                "bbox_area": int(layer["bbox_area"]),
                "bbox": (left_row + r0, left_col + c0, left_row + r1, left_col + c1),
            }
        )

    if not _validate_sample(grid, shifted_layers, placed_objects):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-palette ARC input")
