from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, is_contiguous, paste


QUADRANTS = ("tl", "tr", "bl", "br")
COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
GRID_BOUNDS = (18, 24)
BBOX_BOUNDS = (4, 6)
GAP_CHOICES = (0, 1)
AREA_RETRIES = 80
GENERATION_RETRIES = 300


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r1 - r0 + 1, c1 - c0 + 1


def _quadrant_map(components: list[dict]) -> dict[str, dict]:
    assert len(components) == 4, "expected exactly four sprites"

    row_order = sorted(
        range(4),
        key=lambda i: (
            int(components[i]["bbox"][0]) + int(components[i]["bbox"][2]),
            int(components[i]["bbox"][0]),
            int(components[i]["bbox"][1]),
        ),
    )
    col_order = sorted(
        range(4),
        key=lambda i: (
            int(components[i]["bbox"][1]) + int(components[i]["bbox"][3]),
            int(components[i]["bbox"][1]),
            int(components[i]["bbox"][0]),
        ),
    )

    top_ids = set(row_order[:2])
    bottom_ids = set(row_order[2:])
    left_ids = set(col_order[:2])
    right_ids = set(col_order[2:])

    top_row_keys = [int(components[i]["bbox"][0]) + int(components[i]["bbox"][2]) for i in top_ids]
    bottom_row_keys = [int(components[i]["bbox"][0]) + int(components[i]["bbox"][2]) for i in bottom_ids]
    left_col_keys = [int(components[i]["bbox"][1]) + int(components[i]["bbox"][3]) for i in left_ids]
    right_col_keys = [int(components[i]["bbox"][1]) + int(components[i]["bbox"][3]) for i in right_ids]
    assert max(top_row_keys) < min(bottom_row_keys), "sprite rows must form a clear 2x2 array"
    assert max(left_col_keys) < min(right_col_keys), "sprite cols must form a clear 2x2 array"

    mapped: dict[str, dict] = {}
    for i, comp in enumerate(components):
        row_tag = "t" if i in top_ids else "b"
        col_tag = "l" if i in left_ids else "r"
        key = row_tag + col_tag
        assert key not in mapped, "each quadrant must contain exactly one sprite"
        mapped[key] = comp

    assert set(mapped) == set(QUADRANTS)
    return mapped


def _corner_origin(quadrant: str, canvas_shape: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[int, int]:
    out_h, out_w = map(int, canvas_shape)
    h, w = map(int, patch_shape)
    if quadrant == "tl":
        return 0, 0
    if quadrant == "tr":
        return 0, out_w - w
    if quadrant == "bl":
        return out_h - h, 0
    assert quadrant == "br"
    return out_h - h, out_w - w


def _canvas_shape_from_quadrants(quadrant_map: dict[str, dict]) -> tuple[int, int]:
    tl_h, tl_w = _bbox_size(tuple(map(int, quadrant_map["tl"]["bbox"]))) if "bbox" in quadrant_map["tl"] else quadrant_map["tl"]["mask"].shape
    tr_h, tr_w = _bbox_size(tuple(map(int, quadrant_map["tr"]["bbox"]))) if "bbox" in quadrant_map["tr"] else quadrant_map["tr"]["mask"].shape
    bl_h, bl_w = _bbox_size(tuple(map(int, quadrant_map["bl"]["bbox"]))) if "bbox" in quadrant_map["bl"] else quadrant_map["bl"]["mask"].shape
    br_h, br_w = _bbox_size(tuple(map(int, quadrant_map["br"]["bbox"]))) if "bbox" in quadrant_map["br"] else quadrant_map["br"]["mask"].shape

    top_h = max(tl_h, tr_h)
    bottom_h = max(bl_h, br_h)
    left_w = max(tl_w, bl_w)
    right_w = max(tr_w, br_w)
    return top_h + bottom_h - 1, left_w + right_w - 1


def _render_corner_stack(
    grid: np.ndarray,
    quadrant_map: dict[str, dict],
    order: list[str],
    canvas_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(canvas_shape, bg, dtype=int)
    for quadrant in order:
        comp = quadrant_map[quadrant]
        patch = crop_bbox(grid, tuple(map(int, comp["bbox"])))
        r0, c0 = _corner_origin(quadrant, canvas_shape, patch.shape)
        paste(out, patch, r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move each sprite bbox from a compact 2x2 array to the matching corner of a collapsed corner frame, with smaller sprites drawn on top.

    Input structure
    - The grid has black background and exactly four monochrome 8-connected sprites.
    - The sprite colors are four distinct non-black colors, one sprite per color.
    - The four sprite bounding boxes form a compact 2x2 array, giving top-left, top-right, bottom-left, and bottom-right roles.

    Rule
    - Detect the four sprites and recover their 2x2 array positions from their bounding-box centers.
    - Let the top and bottom row heights be the larger bbox height within each input row, and let the left and right column widths be the larger bbox width within each input column.
    - Build an output canvas of height `top_row_height + bottom_row_height - 1` and width `left_col_width + right_col_width - 1`.
    - Translate each sprite unchanged so that its bbox corner matching its array role lands on the corresponding output corner.
    - Count sprite pixels and draw larger sprites first and smaller sprites later, so smaller sprites overwrite larger ones where the corner-aligned copies overlap.

    Key insight
    - Array position determines the destination corner, while pixel-count ranking only matters after the two bbox rows and columns are collapsed onto one shared corner frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    assert len(components) == 4, "expected exactly four sprites"

    colors = [int(comp["color"]) for comp in components]
    areas = [int(comp["area"]) for comp in components]
    assert len(set(colors)) == 4, "sprite colors must be distinct"
    assert len(set(areas)) == 4, "sprite areas must be distinct"

    quadrant_map = _quadrant_map(components)
    canvas_h, canvas_w = _canvas_shape_from_quadrants(quadrant_map)
    order = [
        quadrant
        for quadrant, _area in sorted(
            ((quadrant, int(comp["area"])) for quadrant, comp in quadrant_map.items()),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return _render_corner_stack(g, quadrant_map, order, (canvas_h, canvas_w), bg)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    cells = [(r, c)]
    while (r, c) != (tr, tc):
        dr = 0 if r == tr else (1 if tr > r else -1)
        dc = 0 if c == tc else (1 if tc > c else -1)
        if dr != 0 and dc != 0 and random.random() < 0.5:
            r += dr
        elif dc != 0:
            c += dc
        else:
            r += dr
        cells.append((r, c))
    return cells


def _seed_cells(h: int, w: int) -> set[tuple[int, int]]:
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    order = anchors[:]
    random.shuffle(order)
    cells = {order[0]}
    for src, dst in zip(order, order[1:]):
        cells.update(_random_path(src, dst))
    return cells


def _shape_mask(h: int, w: int, target_area: int) -> np.ndarray | None:
    for _ in range(AREA_RETRIES):
        cells = _seed_cells(h, w)
        if len(cells) > target_area:
            continue

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for rr, cc in _neighbors8(h, w, r, c):
                    if (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if mask.mean() > 0.72:
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=8):
            continue
        return mask
    return None


def _sample_sprite(existing_areas: set[int]) -> tuple[np.ndarray, int] | None:
    for _ in range(AREA_RETRIES):
        h = random.randint(BBOX_BOUNDS[0], BBOX_BOUNDS[1])
        w = random.randint(BBOX_BOUNDS[0], BBOX_BOUNDS[1])
        min_area = max(h + w - 3, 7)
        max_area = min(h * w - 2, int(0.68 * h * w) + 1)
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        if target_area in existing_areas:
            continue
        mask = _shape_mask(h, w, target_area)
        if mask is not None:
            return mask, target_area
    return None


def _corner_occupancy(specs: dict[str, dict], canvas_shape: tuple[int, int]) -> np.ndarray:
    occ = np.zeros(canvas_shape, dtype=int)
    for quadrant in QUADRANTS:
        mask = specs[quadrant]["mask"]
        r0, c0 = _corner_origin(quadrant, canvas_shape, mask.shape)
        occ[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] += mask.astype(int)
    return occ


def _build_expected_output(specs: dict[str, dict], canvas_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    order = sorted(QUADRANTS, key=lambda quadrant: (-specs[quadrant]["area"], quadrant))
    for quadrant in order:
        mask = specs[quadrant]["mask"]
        color = int(specs[quadrant]["color"])
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        r0, c0 = _corner_origin(quadrant, canvas_shape, mask.shape)
        paste(out, patch, r0, c0, transparent=0)
    return out


def _place_array(specs: dict[str, dict]) -> np.ndarray | None:
    row_heights = [
        max(specs["tl"]["mask"].shape[0], specs["tr"]["mask"].shape[0]),
        max(specs["bl"]["mask"].shape[0], specs["br"]["mask"].shape[0]),
    ]
    col_widths = [
        max(specs["tl"]["mask"].shape[1], specs["bl"]["mask"].shape[1]),
        max(specs["tr"]["mask"].shape[1], specs["br"]["mask"].shape[1]),
    ]
    row_gap = random.choice(GAP_CHOICES)
    col_gap = random.choice(GAP_CHOICES)
    array_h = row_heights[0] + row_gap + row_heights[1]
    array_w = col_widths[0] + col_gap + col_widths[1]

    min_h = max(GRID_BOUNDS[0], array_h + 4)
    min_w = max(GRID_BOUNDS[0], array_w + 4)
    max_h = max(min_h, GRID_BOUNDS[1])
    max_w = max(min_w, GRID_BOUNDS[1])
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)

    max_r0 = grid_h - array_h
    max_c0 = grid_w - array_w
    center_r = max_r0 // 2
    center_c = max_c0 // 2
    origin_r = min(max(random.randint(max(0, center_r - 2), min(max_r0, center_r + 2)), 1), max_r0 - 1)
    origin_c = min(max(random.randint(max(0, center_c - 2), min(max_c0, center_c + 2)), 1), max_c0 - 1)

    positions = {
        "tl": (origin_r, origin_c),
        "tr": (origin_r, origin_c + col_widths[0] + col_gap),
        "bl": (origin_r + row_heights[0] + row_gap, origin_c),
        "br": (origin_r + row_heights[0] + row_gap, origin_c + col_widths[0] + col_gap),
    }

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for quadrant in QUADRANTS:
        mask = specs[quadrant]["mask"]
        color = int(specs[quadrant]["color"])
        cell_r, cell_c = positions[quadrant]
        free_r = row_heights[0] if quadrant.startswith("t") else row_heights[1]
        free_c = col_widths[0] if quadrant.endswith("l") else col_widths[1]
        dr = random.randint(0, free_r - mask.shape[0])
        dc = random.randint(0, free_c - mask.shape[1])
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        paste(grid, patch, cell_r + dr, cell_c + dc, transparent=0)

    return grid


def _semantic_checks(inp: np.ndarray, specs: dict[str, dict]) -> bool:
    canvas_shape = _canvas_shape_from_quadrants(specs)
    correct = _build_expected_output(specs, canvas_shape)
    reverse_order = sorted(QUADRANTS, key=lambda quadrant: (specs[quadrant]["area"], quadrant))
    reverse = np.zeros(canvas_shape, dtype=int)
    for quadrant in reverse_order:
        mask = specs[quadrant]["mask"]
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(specs[quadrant]["color"])
        r0, c0 = _corner_origin(quadrant, canvas_shape, mask.shape)
        paste(reverse, patch, r0, c0, transparent=0)

    swapped = _build_expected_output(
        {
            "tl": specs["tr"],
            "tr": specs["tl"],
            "bl": specs["bl"],
            "br": specs["br"],
        },
        canvas_shape,
    )

    occupancy = _corner_occupancy(specs, canvas_shape)
    overlap_cells = int(np.count_nonzero(occupancy >= 2))
    visible_cells = int(np.count_nonzero(correct))
    if overlap_cells < 1 or overlap_cells > 12:
        return False
    if visible_cells < 8 or visible_cells > canvas_shape[0] * canvas_shape[1] - 2:
        return False
    if np.array_equal(correct, reverse):
        return False
    if np.array_equal(correct, swapped):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, correct)


def _gen_attempt() -> np.ndarray | None:
    colors = COLOR_POOL[:]
    random.shuffle(colors)

    specs: dict[str, dict] = {}
    used_areas: set[int] = set()
    for quadrant, color in zip(QUADRANTS, colors):
        sprite = _sample_sprite(used_areas)
        if sprite is None:
            return None
        mask, area = sprite
        specs[quadrant] = {"mask": mask, "area": area, "color": color}
        used_areas.add(area)

    inp = _place_array(specs)
    if inp is None:
        return None
    if max(inp.shape) > 30:
        return None
    if not _semantic_checks(inp, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
