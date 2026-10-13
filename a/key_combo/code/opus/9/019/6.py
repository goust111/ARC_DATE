from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components


BG = 0
WALL = 5
FG_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_REGION_SIDE = 5
MAX_GENERATION_ATTEMPTS = 400

_GEN_COUNTER = 0

_ONE_HOLE_BASES = [
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=int,
    ),
]

_TWO_HOLE_BASES = [
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
        ],
        dtype=int,
    ),
]


def _region_components(grid: np.ndarray) -> list[dict]:
    region_grid = (grid != WALL).astype(int)
    regions = find_components(region_grid, bg=0, colors=[1], connectivity=4)
    regions = sorted(regions, key=lambda comp: tuple(map(int, comp["bbox"])))
    for region in regions:
        r0, c0, r1, c1 = map(int, region["bbox"])
        assert int(region["area"]) == int((r1 - r0 + 1) * (c1 - c0 + 1))
    return regions


def _count_holes_from_bg(patch: np.ndarray) -> int:
    zero_grid = (patch == BG).astype(int)
    holes = 0
    for comp in find_components(zero_grid, bg=0, colors=[1], connectivity=4):
        coords = [tuple(map(int, coord)) for coord in comp["coords"]]
        if any(r == 0 or c == 0 or r == patch.shape[0] - 1 or c == patch.shape[1] - 1 for r, c in coords):
            continue
        holes += 1
    return int(holes)


def _single_object(region_patch: np.ndarray) -> dict:
    colors = sorted(int(v) for v in np.unique(region_patch) if int(v) != BG)
    assert len(colors) == 1
    color = colors[0]
    objs = find_components(region_patch, bg=BG, colors=[color], connectivity=4)
    assert len(objs) == 1
    return objs[0]


def _complete_rotational_symmetry(region_patch: np.ndarray) -> np.ndarray:
    obj = _single_object(region_patch)
    color = int(obj["color"])
    r0, c0, r1, c1 = map(int, obj["bbox"])
    patch = crop_bbox(region_patch, obj["bbox"])
    mask = patch == color
    assert mask.shape[0] == mask.shape[1]

    completed = mask.copy()
    for k in (1, 2, 3):
        completed |= np.rot90(mask, k)

    out = region_patch.copy()
    window = out[r0 : r1 + 1, c0 : c1 + 1]
    window[(completed) & (window == BG)] = color
    return out


def _analyze(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    regions = []
    for region in _region_components(g):
        region_patch = crop_bbox(g, region["bbox"])
        obj = _single_object(region_patch)
        holes = _count_holes_from_bg(region_patch)
        regions.append(
            {
                "bbox": tuple(map(int, region["bbox"])),
                "patch": region_patch,
                "holes": int(holes),
                "obj_bbox": tuple(map(int, obj["bbox"])),
                "obj_color": int(obj["color"]),
                "obj_area": int(obj["area"]),
                "completed": None,
            }
        )

    counts = Counter(info["holes"] for info in regions)
    unique_holes = [hole_count for hole_count, freq in counts.items() if freq == 1]
    assert len(unique_holes) == 1
    outlier = next(i for i, info in enumerate(regions) if info["holes"] == unique_holes[0])
    regions[outlier]["completed"] = _complete_rotational_symmetry(regions[outlier]["patch"])
    return regions, int(outlier)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the region whose object has the unique hole count, and complete that object by 90-degree rotation.

    Input structure:
    - Black background with thin grey partition walls that split the canvas into 4 or 5 rectangular regions.
    - Each region contains exactly one 4-connected monochrome foreground object and otherwise only black cells.
    - The object's black connected components that do not touch the region border are its internal holes.

    Rule:
    - Use the grey walls to recover the rectangular regions.
    - Count the number of internal black holes in each region's object.
    - Find the single region whose hole count is unique among all regions.
    - In that region, crop the object's bounding box and union the object with its 90, 180, and 270 degree rotations about the bbox center, filling only black cells.
    - Erase every foreground object in the non-unique regions and keep the grey walls unchanged.

    Key insight:
    - The decisive signal is global: the target is identified only by comparing hole counts across regions, and only that winning region receives the local rotational-completion operation.
    """
    g = np.asarray(grid, dtype=int)
    regions, outlier = _analyze(g)

    out = np.where(g == WALL, WALL, BG)
    target = regions[outlier]
    r0, c0, r1, c1 = target["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = target["completed"]
    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _ring_mask(size: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=int)
    mask[1:-1, 1:-1] = 0
    return mask


def _component_count(mask: np.ndarray) -> int:
    comps = find_components(mask.astype(int), bg=0, colors=[1], connectivity=4)
    return int(len(comps))


def _make_outlier_mask(common_holes: int, rng: random.Random, region_shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    candidates = [size for size in (4, 5) if size <= min(region_shape)]
    if not candidates:
        raise ValueError("region too small for outlier")
    size = rng.choice(candidates)
    full = _ring_mask(size)

    if common_holes == 2:
        orbit = [(0, 0), (0, size - 1), (size - 1, size - 1), (size - 1, 0)]
        expected_holes = 1
    else:
        mid = size // 2
        orbit = [(0, mid), (mid, size - 1), (size - 1, mid), (mid, 0)]
        expected_holes = 0

    subsets: list[tuple[tuple[int, int], ...]] = []
    for remove_count in range(1, 4):
        for subset in itertools.combinations(orbit, remove_count):
            subsets.append(tuple(subset))
    rng.shuffle(subsets)

    for subset in subsets:
        partial = full.copy()
        for r, c in subset:
            partial[r, c] = 0
        if _component_count(partial) != 1:
            continue
        if _count_holes_from_bg(partial) != expected_holes:
            continue
        completed = partial.copy()
        for k in (1, 2, 3):
            completed |= np.rot90(partial, k)
        if np.array_equal(completed, full):
            return partial, int(expected_holes)

    raise ValueError("failed to build connected outlier")


def _mask_library_for_holes(hole_count: int) -> list[np.ndarray]:
    if hole_count == 1:
        return [mask.copy() for mask in _ONE_HOLE_BASES]
    if hole_count == 2:
        return [mask.copy() for mask in _TWO_HOLE_BASES]
    raise ValueError("unsupported hole count")


def _sample_common_mask(hole_count: int, rng: random.Random, region_shape: tuple[int, int]) -> np.ndarray:
    bases = _mask_library_for_holes(hole_count)
    variants = []
    for base in bases:
        for _ in range(8):
            cand = _transform_mask(base, rng)
            cand = _tight_crop(cand)
            if cand.shape[0] <= region_shape[0] and cand.shape[1] <= region_shape[1]:
                variants.append(cand)
    if not variants:
        raise ValueError("no common mask fits region")
    chosen = variants[rng.randrange(len(variants))].copy()
    assert _count_holes_from_bg(chosen) == hole_count
    return chosen


def _placement_options(region_shape: tuple[int, int], patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    rh, rw = map(int, region_shape)
    ph, pw = map(int, patch_shape)
    opts = []
    for margin in (1, 0):
        opts.clear()
        for r in range(rh - ph + 1):
            for c in range(rw - pw + 1):
                if margin == 1:
                    if r == 0 or c == 0 or r + ph == rh or c + pw == rw:
                        continue
                opts.append((r, c))
        if opts:
            return list(opts)
    return []


def _paint_region(grid: np.ndarray, region_bbox: tuple[int, int, int, int], mask: np.ndarray, color: int, rng: random.Random) -> None:
    r0, c0, r1, c1 = map(int, region_bbox)
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    options = _placement_options((region_h, region_w), mask.shape)
    if not options:
        raise ValueError("mask cannot be placed in region")
    off_r, off_c = rng.choice(options)
    rr0 = r0 + off_r
    cc0 = c0 + off_c
    patch = grid[rr0 : rr0 + mask.shape[0], cc0 : cc0 + mask.shape[1]]
    patch[mask.astype(bool)] = int(color)


def _split_regions(shape: tuple[int, int], target_regions: int, rng: random.Random) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]]:
    H, W = map(int, shape)
    rects = [(0, 0, H - 1, W - 1)]
    walls: list[tuple[str, int, int, int]] = []

    while len(rects) < target_regions:
        splittable = []
        for idx, rect in enumerate(rects):
            r0, c0, r1, c1 = rect
            if r1 - r0 + 1 >= 2 * MIN_REGION_SIDE + 1:
                splittable.append((idx, "h"))
            if c1 - c0 + 1 >= 2 * MIN_REGION_SIDE + 1:
                splittable.append((idx, "v"))
        if not splittable:
            raise ValueError("partition became unsplittable")

        idx, axis = rng.choice(splittable)
        r0, c0, r1, c1 = rects.pop(idx)
        if axis == "h":
            wall_r = rng.randint(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE)
            rects.append((r0, c0, wall_r - 1, c1))
            rects.append((wall_r + 1, c0, r1, c1))
            walls.append(("h", wall_r, c0, c1))
        else:
            wall_c = rng.randint(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE)
            rects.append((r0, c0, r1, wall_c - 1))
            rects.append((r0, wall_c + 1, r1, c1))
            walls.append(("v", wall_c, r0, r1))

    rects = sorted(rects)
    return rects, walls


def _inject_counterexamples(mask_specs: list[tuple[str, int]], common_holes: int, outlier_idx: int, rng: random.Random) -> None:
    common_indices = [idx for idx in range(len(mask_specs)) if idx != outlier_idx]
    if common_holes == 1 and common_indices:
        idx = rng.choice(common_indices)
        mask_specs[idx] = ("force_square_ring", 1)
    elif common_holes == 2 and len(common_indices) >= 2:
        picked = rng.sample(common_indices, 2)
        mask_specs[picked[0]] = ("force_two_hole_a", 2)
        mask_specs[picked[1]] = ("force_two_hole_b", 2)


def _grid_shape_for_case(target_regions: int, rng: random.Random) -> tuple[int, int]:
    if target_regions == 4:
        return rng.randint(17, 24), rng.randint(17, 24)
    return rng.randint(20, 28), rng.randint(20, 28)


def _gen_attempt(case_id: int) -> np.ndarray | None:
    rng = random.Random((case_id + 1) * 1009)
    target_regions = 4 if case_id % 3 == 0 else 5
    common_holes = 1 if case_id % 2 == 0 else 2

    H, W = _grid_shape_for_case(target_regions, rng)
    try:
        rects, walls = _split_regions((H, W), target_regions, rng)
    except ValueError:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for axis, fixed, lo, hi in walls:
        if axis == "h":
            grid[int(fixed), int(lo) : int(hi) + 1] = WALL
        else:
            grid[int(lo) : int(hi) + 1, int(fixed)] = WALL

    outlier_idx = rng.randrange(len(rects))
    colors = rng.sample(FG_COLORS, len(rects))
    mask_specs = [("common", common_holes) for _ in rects]
    mask_specs[outlier_idx] = ("outlier", common_holes)
    _inject_counterexamples(mask_specs, common_holes, outlier_idx, rng)

    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect
        region_shape = (r1 - r0 + 1, c1 - c0 + 1)
        spec = mask_specs[idx]
        try:
            if spec[0] == "outlier":
                mask, _holes = _make_outlier_mask(common_holes, rng, region_shape)
            elif spec[0] == "force_square_ring":
                mask = _ring_mask(4)
            elif spec[0] == "force_two_hole_a":
                mask = _TWO_HOLE_BASES[0]
            elif spec[0] == "force_two_hole_b":
                mask = _transform_mask(_TWO_HOLE_BASES[2], rng)
            else:
                mask = _sample_common_mask(common_holes, rng, region_shape)
        except ValueError:
            return None

        mask = _tight_crop(mask)
        if mask.shape[0] > region_shape[0] or mask.shape[1] > region_shape[1]:
            return None
        if spec[0] != "outlier":
            assert _count_holes_from_bg(mask) == common_holes
        _paint_region(grid, rect, mask, colors[idx], rng)

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    regions, outlier = _analyze(grid)
    assert 4 <= len(regions) <= 5

    counts = Counter(info["holes"] for info in regions)
    assert len([hole for hole, freq in counts.items() if freq == 1]) == 1

    out = main(grid)
    assert np.array_equal(out == WALL, grid == WALL)

    added = 0
    for idx, info in enumerate(regions):
        r0, c0, r1, c1 = info["bbox"]
        out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if idx == outlier:
            assert np.array_equal(out_patch, info["completed"])
            added += int(np.count_nonzero(info["completed"])) - int(np.count_nonzero(info["patch"]))
            assert not np.array_equal(info["patch"], info["completed"])
        else:
            assert np.count_nonzero(out_patch) == 0

    assert added > 0


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_GEN_COUNTER)
        _GEN_COUNTER += 1
        if grid is None:
            continue
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
