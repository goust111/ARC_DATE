from __future__ import annotations

import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


BG = 0
GEN_ATTEMPTS = 500
MAX_PATCH_SIDE = 6
COUNT_SETS = [
    counts
    for k in (2, 3, 4)
    for counts in combinations(range(1, 8), k)
    if counts[0] + 2 * sum(counts[1:]) <= 30 and sum(counts) <= 16
]
AREA_BAND_TEMPLATES = {
    2: (
        ((12, 18), (1, 3)),
        ((10, 16), (2, 4)),
    ),
    3: (
        ((12, 18), (6, 10), (1, 3)),
        ((11, 17), (5, 8), (2, 4)),
    ),
    4: (
        ((12, 18), (8, 11), (4, 6), (1, 3)),
        ((11, 16), (7, 10), (3, 5), (1, 2)),
    ),
}
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _component_stats(grid: np.ndarray) -> dict[int, dict[str, object]]:
    bg = int(get_bg(grid))
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    stats: dict[int, dict[str, object]] = {}
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        entry = stats.setdefault(
            color,
            {
                "count": 0,
                "total_pixels": 0,
                "areas": [],
                "max_area": 0,
                "max_bbox_area": 0,
            },
        )
        entry["count"] = int(entry["count"]) + 1
        entry["total_pixels"] = int(entry["total_pixels"]) + area
        entry["areas"].append(area)
        entry["max_area"] = max(int(entry["max_area"]), area)
        entry["max_bbox_area"] = max(int(entry["max_bbox_area"]), bbox_area)
    return stats


def _build_output_from_ordered_counts(ordered_counts: list[tuple[int, int]]) -> np.ndarray:
    side0, color0 = ordered_counts[0]
    out = np.full((side0, side0), color0, dtype=int)
    for side, color in ordered_counts[1:]:
        nxt = np.full((out.shape[0] + 2 * side, out.shape[1] + 2 * side), color, dtype=int)
        paste(nxt, out, side, side)
        out = nxt
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    stats = _component_stats(grid)
    assert stats, "expected at least one foreground color"
    ordered_counts = sorted((int(v["count"]), color) for color, v in stats.items())
    counts_only = [count for count, _color in ordered_counts]
    assert len(counts_only) == len(set(counts_only)), "per-color object counts must be distinct"
    return _build_output_from_ordered_counts(ordered_counts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each color's objects and rebuild them as concentric square layers whose order and thickness equal those counts.

    Input structure:
    - A black-dominant background contains several monochrome connected objects in multiple colors.
    - Objects of the same color may have different shapes, sizes, and positions.
    - Different colors occur in different numbers of connected objects.

    Rule:
    - Count the number of monochrome connected objects for each non-background color.
    - The color with the smallest count becomes a solid central square whose side length equals that count.
    - Wrap the current square outward with one square layer per remaining color, in ascending count order, where each layer's thickness equals that color's count.

    Key insight:
    - Shape, size, and placement are decoys; the only meaningful signal is how many objects each color has, and that same count controls both the nesting order and the layer thickness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _expected_output(g)


def _crop_non_bg(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _feasible_dims(area: int, density_min: float) -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in range(1, MAX_PATCH_SIDE + 1)
        for w in range(1, MAX_PATCH_SIDE + 1)
        if area <= h * w and area / (h * w) >= density_min
    ]


def _make_boxy_mask(area: int) -> np.ndarray | None:
    dims = _feasible_dims(area, 0.45)
    if not dims:
        return None
    h, w = random.choice(dims)
    mask = np.zeros((h, w), dtype=bool)
    full_rows, rem = divmod(area, w)
    if full_rows > 0:
        mask[:full_rows, :] = True
    if rem:
        if random.random() < 0.5:
            mask[full_rows, :rem] = True
        else:
            mask[full_rows, w - rem :] = True
    return _transform_patch(mask)


def _make_worm_mask(area: int) -> np.ndarray | None:
    side = min(MAX_PATCH_SIDE + 1, max(3, area + 1))
    for _ in range(40):
        mask = np.zeros((side, side), dtype=bool)
        start = (random.randrange(side), random.randrange(side))
        used = {start}
        path = [start]
        while len(path) < area:
            r, c = path[-1]
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in used:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            nxt = random.choice(nbrs)
            used.add(nxt)
            path.append(nxt)
        if len(path) == area:
            for r, c in path:
                mask[r, c] = True
            return _transform_patch(mask)
    return None


def _make_blob_mask(area: int) -> np.ndarray | None:
    dims = _feasible_dims(area, 0.35)
    if not dims:
        return None
    for _ in range(40):
        h, w = random.choice(dims)
        mask = np.zeros((h, w), dtype=bool)
        used = {(random.randrange(h), random.randrange(w))}
        while len(used) < area:
            frontier = []
            for r, c in used:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in used:
                            frontier.append((nr, nc))
            if not frontier:
                break
            used.add(random.choice(frontier))
        if len(used) == area:
            for r, c in used:
                mask[r, c] = True
            return _transform_patch(mask)
    return None


def _make_patch(color: int, area: int) -> np.ndarray | None:
    makers = [_make_blob_mask, _make_worm_mask, _make_boxy_mask]
    random.shuffle(makers)
    for maker in makers:
        mask = maker(area)
        if mask is None:
            continue
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = int(color)
        return _crop_non_bg(patch)
    return None


def _sample_area_lists(counts: tuple[int, ...]) -> list[list[int]] | None:
    band_template = random.choice(AREA_BAND_TEMPLATES[len(counts)])
    area_lists: list[list[int]] = []
    varied = False
    for count, (lo, hi) in zip(counts, band_template, strict=True):
        vals = [random.randint(lo, hi) for _ in range(count)]
        if count > 1 and len(set(vals)) > 1:
            varied = True
        area_lists.append(vals)
    if not varied:
        for idx, count in enumerate(counts):
            lo, hi = band_template[idx]
            if count > 1 and lo < hi:
                area_lists[idx][0] = lo
                area_lists[idx][-1] = hi
                varied = True
                break
    return area_lists if varied else None


def _choose_canvas_shape(total_objects: int, total_pixels: int) -> tuple[int, int]:
    side_min = 18
    if total_objects >= 10:
        side_min += 4
    if total_objects >= 14:
        side_min += 2
    if total_pixels >= 90:
        side_min += 2
    side_min = min(side_min, 28)
    return random.randint(side_min, 30), random.randint(side_min, 30)


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    total_pixels = sum(int(np.count_nonzero(patch != BG)) for patch in patches)
    h, w = _choose_canvas_shape(len(patches), total_pixels)
    grid = np.full((h, w), BG, dtype=int)
    ordered = sorted(patches, key=lambda patch: int(np.count_nonzero(patch != BG)), reverse=True)
    try:
        for patch in ordered:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
            paste(grid, patch, r0, c0, transparent=BG)
    except ValueError:
        return None
    return grid


def _orders_from_stats(stats: dict[int, dict[str, object]]) -> dict[str, tuple[int, ...]]:
    return {
        "count": tuple(color for color, info in sorted(stats.items(), key=lambda item: (int(item[1]["count"]), item[0]))),
        "pixels": tuple(
            color for color, info in sorted(stats.items(), key=lambda item: (int(item[1]["total_pixels"]), item[0]))
        ),
        "max_area": tuple(color for color, info in sorted(stats.items(), key=lambda item: (int(item[1]["max_area"]), item[0]))),
    }


def _semantic_ok(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(get_bg(grid)) != BG:
        return False

    stats = _component_stats(grid)
    if not (2 <= len(stats) <= 4):
        return False

    counts = [int(info["count"]) for info in stats.values()]
    if len(counts) != len(set(counts)):
        return False

    total_pixels = [int(info["total_pixels"]) for info in stats.values()]
    max_areas = [int(info["max_area"]) for info in stats.values()]
    if len(total_pixels) != len(set(total_pixels)):
        return False
    if len(max_areas) != len(set(max_areas)):
        return False

    orders = _orders_from_stats(stats)
    if orders["count"] == orders["pixels"]:
        return False
    if orders["count"] == orders["max_area"]:
        return False

    if not any(len(set(info["areas"])) > 1 for info in stats.values() if int(info["count"]) > 1):
        return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False

    sig = (tuple(map(int, out.shape)), out.tobytes())
    if sig in _SEEN_OUTPUTS:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    counts = random.choice(COUNT_SETS)
    colors = random.sample(range(1, 10), len(counts))
    area_lists = _sample_area_lists(counts)
    if area_lists is None:
        return None

    patches: list[np.ndarray] = []
    for color, per_object_areas in zip(colors, area_lists, strict=True):
        for area in per_object_areas:
            patch = _make_patch(color, area)
            if patch is None:
                return None
            patches.append(patch)

    random.shuffle(patches)
    grid = _place_patches(patches)
    if grid is None or not _semantic_ok(grid):
        return None

    out = main(grid)
    _SEEN_OUTPUTS.add((tuple(map(int, out.shape)), out.tobytes()))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quantity-tower input")
