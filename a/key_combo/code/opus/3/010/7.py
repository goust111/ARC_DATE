from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
colors_present = _ARC_UTILS.colors_present
get_bg = _ARC_UTILS.get_bg


AREAS = (3, 5, 7, 9, 15, 21, 25, 35, 49)
AREA_DIM_OPTIONS = {
    3: ((1, 2), (2, 1)),
    5: ((1, 3), (3, 1)),
    7: ((1, 4), (4, 1)),
    9: ((2, 2),),
    15: ((2, 3), (3, 2)),
    21: ((2, 4), (4, 2)),
    25: ((3, 3),),
    35: ((3, 4), (4, 3)),
    49: ((4, 4),),
}
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _boustrophedon_grid(colors_in_order: list[int] | tuple[int, ...]) -> np.ndarray:
    out = np.zeros((3, 3), dtype=int)
    for idx, color in enumerate(colors_in_order):
        r = idx // 3
        pos = idx % 3
        c = pos if r % 2 == 0 else 2 - pos
        out[r, c] = int(color)
    return out


def _extract_sparse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects: list[dict] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        assert 2 <= len(coords) <= 5, "each sparse sprite must have 2..5 visible pixels"
        r0, c0, r1, c1 = bbox_coords(coords)
        rel = coords - np.array([r0, c0], dtype=int)
        assert np.all(rel % 2 == 0), "sprite pixels must lie on one step-2 lattice"
        dense_h = int(r1 - r0 + 1)
        dense_w = int(c1 - c0 + 1)
        assert dense_h % 2 == 1 and dense_w % 2 == 1
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "dense_h": dense_h,
                "dense_w": dense_w,
                "dense_area": int(dense_h * dense_w),
                "visible_count": int(len(coords)),
            }
        )
    assert len(objects) == 9, "expected exactly 9 sparse sprites"
    areas = [obj["dense_area"] for obj in objects]
    assert len(set(areas)) == 9, "dense areas must be unique"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify sparse same-color sprites, sort them by the densified area, then snake-pack the colors into 3x3.

    Input structure:
    - A black background contains exactly nine sparse monochrome sprites, one per nonzero color.
    - Every sprite uses 2..5 visible pixels that lie on a step-2 lattice inside one color-specific bbox.
    - Different sprites are separated, but their visible pixel counts need not match their densified sizes.

    Rule:
    - For each color, take the bbox of its step-2 lattice pixels and densify that sparse bbox into a solid rectangle.
    - Use the solid rectangle's pixel area as the sprite's true area.
    - Sort the nine colors by this densified area from small to large.
    - Write the sorted colors into a 3x3 output using boustrophedon order: row 0 left-to-right, row 1 right-to-left, row 2 left-to-right.

    Key insight:
    - The ordering key is not the number of visible pixels; it is the area after the sparse lattice bbox is filled into a dense solid shape.
    """
    objects = _extract_sparse_objects(np.asarray(grid, dtype=int))
    ordered = [obj["color"] for obj in sorted(objects, key=lambda obj: obj["dense_area"])]
    return _boustrophedon_grid(ordered)


@lru_cache(maxsize=None)
def _logical_patterns(h: int, w: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    patterns: list[tuple[tuple[int, int], ...]] = []
    max_count = min(5, h * w)
    for count in range(2, max_count + 1):
        for combo in combinations(cells, count):
            arr = np.array(combo, dtype=int)
            if bbox_coords(arr) == (0, 0, h - 1, w - 1):
                patterns.append(tuple(sorted((int(r), int(c)) for r, c in combo)))
    assert patterns, f"no sparse pattern spans logical bbox {(h, w)}"
    return tuple(patterns)


def _choose_pattern(h: int, w: int) -> tuple[tuple[int, int], ...]:
    return random.choice(_logical_patterns(h, w))


def _make_specs() -> list[dict]:
    colors = list(COLORS)
    random.shuffle(colors)

    specs: list[dict] = []
    for area, color in zip(AREAS, colors):
        logical_h, logical_w = random.choice(AREA_DIM_OPTIONS[area])
        pattern = _choose_pattern(logical_h, logical_w)
        dense_h = 2 * logical_h - 1
        dense_w = 2 * logical_w - 1
        specs.append(
            {
                "area": int(area),
                "color": int(color),
                "logical_h": int(logical_h),
                "logical_w": int(logical_w),
                "pattern": pattern,
                "dense_h": int(dense_h),
                "dense_w": int(dense_w),
                "visible_count": int(len(pattern)),
            }
        )
    return specs


def _layout_grid(specs: list[dict]) -> np.ndarray:
    placed = list(specs)
    random.shuffle(placed)

    row_heights = []
    for row in range(3):
        row_specs = placed[row * 3 : (row + 1) * 3]
        row_heights.append(max(spec["dense_h"] for spec in row_specs) + random.randint(0, 1))

    col_widths = []
    for col in range(3):
        col_specs = [placed[row * 3 + col] for row in range(3)]
        col_widths.append(max(spec["dense_w"] for spec in col_specs) + random.randint(0, 1))

    row_gaps = [random.randint(1, 2), random.randint(1, 2)]
    col_gaps = [random.randint(1, 2), random.randint(1, 2)]

    height = sum(row_heights) + sum(row_gaps)
    width = sum(col_widths) + sum(col_gaps)
    assert height <= 30 and width <= 30

    row_starts = [0, row_heights[0] + row_gaps[0], row_heights[0] + row_gaps[0] + row_heights[1] + row_gaps[1]]
    col_starts = [0, col_widths[0] + col_gaps[0], col_widths[0] + col_gaps[0] + col_widths[1] + col_gaps[1]]

    grid = np.zeros((height, width), dtype=int)
    for idx, spec in enumerate(placed):
        row = idx // 3
        col = idx % 3
        slot_r0 = row_starts[row]
        slot_c0 = col_starts[col]
        slack_r = row_heights[row] - spec["dense_h"]
        slack_c = col_widths[col] - spec["dense_w"]
        r0 = slot_r0 + random.randint(0, slack_r)
        c0 = slot_c0 + random.randint(0, slack_c)

        coords = []
        for lr, lc in spec["pattern"]:
            rr = r0 + 2 * int(lr)
            cc = c0 + 2 * int(lc)
            grid[rr, cc] = int(spec["color"])
            coords.append((rr, cc))

        spec["bbox"] = (r0, c0, r0 + spec["dense_h"] - 1, c0 + spec["dense_w"] - 1)
        spec["coords"] = tuple(coords)

    return grid


def _ordered_colors(objects: list[dict], key) -> tuple[int, ...]:
    return tuple(obj["color"] for obj in sorted(objects, key=key))


def _semantic_checks(grid: np.ndarray) -> bool:
    objects = _extract_sparse_objects(grid)
    area_order = _ordered_colors(objects, key=lambda obj: obj["dense_area"])
    count_order = _ordered_colors(
        objects,
        key=lambda obj: (obj["visible_count"], obj["bbox"][0], obj["bbox"][1]),
    )
    pos_order = _ordered_colors(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    height_order = _ordered_colors(objects, key=lambda obj: (obj["dense_h"], obj["bbox"][0], obj["bbox"][1]))
    width_order = _ordered_colors(objects, key=lambda obj: (obj["dense_w"], obj["bbox"][0], obj["bbox"][1]))

    counts_in_area_order = [obj["visible_count"] for obj in sorted(objects, key=lambda obj: obj["dense_area"])]
    has_inversion = any(
        counts_in_area_order[i] > counts_in_area_order[j]
        for i in range(len(counts_in_area_order))
        for j in range(i + 1, len(counts_in_area_order))
    )
    has_repeat = len(set(counts_in_area_order)) < len(counts_in_area_order)
    if not (has_inversion and has_repeat):
        return False

    area_grid = _boustrophedon_grid(area_order)
    if np.array_equal(area_grid, _boustrophedon_grid(count_order)):
        return False
    if np.array_equal(area_grid, _boustrophedon_grid(pos_order)):
        return False
    if np.array_equal(area_grid, _boustrophedon_grid(height_order)):
        return False
    if np.array_equal(area_grid, _boustrophedon_grid(width_order)):
        return False

    return np.array_equal(main(grid), area_grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        specs = _make_specs()
        grid = _layout_grid(specs)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
