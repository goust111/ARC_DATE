from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, paste, scale_sprite


BLACK = 0
RED = 2

_GEN_PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
_GEN_REGION_OPTIONS = (6, 7, 8)
_GEN_HEIGHT_RANGE = (18, 25)
_GEN_MAX_ATTEMPTS = 400
_GEN_CALL_COUNT = 0

_PAIR_BASES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
)

_UNMATCHED_BASES = (
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
    np.array([[1], [1], [1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
)


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    full = np.zeros(shape, dtype=int)
    full[coords[:, 0], coords[:, 1]] = 1
    return crop_bbox(full, bbox_coords(coords)).astype(bool)


def _extract_region_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    records: list[dict] = []
    for region in find_components(g, bg=RED, connectivity=4, monochromatic=False, return_format="dict"):
        coords = region["coords"]
        vals = g[coords[:, 0], coords[:, 1]]
        colors = sorted({int(v) for v in vals if int(v) not in (BLACK, RED)})
        assert len(colors) == 1, "each divider region must contain exactly one monochrome object"
        color = int(colors[0])
        obj_coords = coords[vals == color]
        obj_mask = _coords_to_mask(obj_coords, g.shape)
        records.append(
            {
                "region_coords": coords,
                "color": color,
                "object_coords": obj_coords,
                "object_mask": obj_mask,
                "area": int(obj_mask.sum()),
            }
        )
    return records


def _scaled_match(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int] | None:
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    if area_a == area_b:
        return None
    if area_a < area_b:
        small_id, big_id = 0, 1
        small, big = mask_a, mask_b
    else:
        small_id, big_id = 1, 0
        small, big = mask_b, mask_a
    if int(big.sum()) != 4 * int(small.sum()):
        return None
    scaled = scale_sprite(small.astype(int), 2).astype(bool)
    if scaled.shape != big.shape:
        return None
    if not np.array_equal(scaled, big):
        return None
    return small_id, big_id


def _find_scaled_pairs(records: list[dict]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            order = _scaled_match(records[i]["object_mask"], records[j]["object_mask"])
            if order is None:
                continue
            if order == (0, 1):
                pairs.append((i, j))
            else:
                pairs.append((j, i))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Flood the smaller region in each 2x shape-matching pair with the larger object's color.

    Input structure:
    - The grid has black background cells split by a single continuous red divider polyline into multiple regions.
    - Every non-red region contains exactly one 4-connected monochrome object floating in black space.
    - Exactly two object pairs share the same shape across scale: one object is the exact 2x enlargement of the other.

    Rule:
    - Treat red cells as walls and identify every connected non-red region.
    - Extract the unique object mask from each region, normalize it by its own bounding box, and compare all object pairs.
    - Whenever a smaller mask becomes identical to a larger mask after 2x replication, recolor every black cell in the smaller object's region to the larger object's color, while leaving both objects and the red divider unchanged.

    Key insight:
    - The fill target and the fill color come from different regions: shape matching finds the cross-region pair, then the smaller partner selects which region to flood and the larger partner supplies the paint color.
    """
    g = np.asarray(grid, dtype=int)
    records = _extract_region_records(g)
    out = g.copy()

    for small_idx, big_idx in _find_scaled_pairs(records):
        fill_color = int(records[big_idx]["color"])
        coords = records[small_idx]["region_coords"]
        vals = g[coords[:, 0], coords[:, 1]]
        bg_coords = coords[vals == BLACK]
        out[bg_coords[:, 0], bg_coords[:, 1]] = fill_color

    return out


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.astype(bool).copy())
    return variants


def _sample_strip_widths(region_count: int) -> list[int] | None:
    base_total = 2 * region_count + (region_count - 1)
    max_extra = 30 - base_total
    min_extra = 6
    if max_extra < min_extra:
        return None

    widths = [2] * region_count
    for idx in random.sample(range(region_count), 2):
        widths[idx] += 3

    extra = random.randint(min_extra, max_extra) - 6
    while extra > 0:
        widths[random.randrange(region_count)] += 1
        extra -= 1

    random.shuffle(widths)
    return widths


def _strip_ranges(widths: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for width in widths:
        spans.append((cursor, cursor + width - 1))
        cursor += width + 1
    return spans


def _draw_vertical_divider(height: int, widths: list[int]) -> np.ndarray:
    width = sum(widths) + len(widths) - 1
    grid = np.zeros((height, width), dtype=int)
    wall_cols = [right + 1 for _left, right in _strip_ranges(widths[:-1])]

    row = 0
    direction = 1
    for i, col in enumerate(wall_cols):
        if direction == 1:
            grid[row:, col] = RED
            row = height - 1
        else:
            grid[: row + 1, col] = RED
            row = 0
        if i + 1 < len(wall_cols):
            next_col = wall_cols[i + 1]
            c0, c1 = sorted((col, next_col))
            grid[row, c0 : c1 + 1] = RED
            direction *= -1
    return grid


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    colored = np.zeros(patch.shape, dtype=int)
    colored[patch] = int(color)
    paste(grid, colored, r0, c0, transparent=BLACK)


def _place_in_strip(height: int, left: int, right: int, patch: np.ndarray) -> tuple[int, int] | None:
    patch_h, patch_w = patch.shape
    strip_w = right - left + 1
    if patch_w > strip_w or patch_h > height - 2:
        return None

    row_lo = 1
    row_hi = height - patch_h - 1
    if row_hi - row_lo >= 2:
        row_lo += 1
        row_hi -= 1

    col_lo = left
    col_hi = right - patch_w + 1
    if col_hi - col_lo >= 2:
        col_lo += 1
        col_hi -= 1

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _pick_pair_shapes(strip_widths: list[int]) -> list[dict] | None:
    chosen_bases = random.sample(list(_PAIR_BASES), 2)
    pair_defs: list[dict] = []
    max_width = max(strip_widths)

    for base in chosen_bases:
        options = [
            variant
            for variant in _dihedral_variants(base)
            if scale_sprite(variant.astype(int), 2).shape[1] <= max_width
        ]
        if not options:
            return None
        small = random.choice(options)
        large = scale_sprite(small.astype(int), 2).astype(bool)
        pair_defs.append({"small_patch": small, "large_patch": large})

    pair_defs.sort(key=lambda item: item["large_patch"].shape[1], reverse=True)
    return pair_defs


def _assign_regions(widths: list[int], pair_defs: list[dict]) -> bool:
    remaining = set(range(len(widths)))

    for pair in pair_defs:
        needed = pair["large_patch"].shape[1]
        choices = [i for i in remaining if widths[i] >= needed]
        if not choices:
            return False
        widest = max(widths[i] for i in choices)
        top_choices = [i for i in choices if widths[i] >= widest - 1]
        large_region = random.choice(top_choices)
        pair["large_region"] = large_region
        remaining.remove(large_region)

    for pair in pair_defs:
        needed = pair["small_patch"].shape[1]
        choices = [i for i in remaining if widths[i] >= needed]
        if not choices:
            return False
        small_region = random.choice(choices)
        pair["small_region"] = small_region
        remaining.remove(small_region)

    unmatched_regions = list(remaining)
    random.shuffle(unmatched_regions)
    unused_bases = random.sample(list(_UNMATCHED_BASES), len(unmatched_regions))
    for region, base in zip(unmatched_regions, unused_bases):
        options = [variant for variant in _dihedral_variants(base) if variant.shape[1] <= widths[region]]
        if not options:
            return False
        pair_defs.append({"small_region": region, "patch": random.choice(options), "role": "unmatched"})

    for pair in pair_defs:
        if "role" not in pair:
            pair["role"] = "pair"
    return True


def _render_layout(height: int, widths: list[int], assignments: list[dict], colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    grid = _draw_vertical_divider(height, widths)
    expected = grid.copy()
    spans = _strip_ranges(widths)

    region_patches: dict[int, np.ndarray] = {}
    region_roles: dict[int, dict] = {}
    color_iter = iter(colors)

    for item in assignments:
        if item["role"] == "pair":
            for key in ("small_region", "large_region"):
                region = item[key]
                patch = item["small_patch"] if key == "small_region" else item["large_patch"]
                region_patches[region] = patch
                region_roles[region] = {"kind": key, "pair": item, "color": next(color_iter)}
        else:
            region = item["small_region"]
            region_patches[region] = item["patch"]
            region_roles[region] = {"kind": "unmatched", "color": next(color_iter)}

    for region, patch in region_patches.items():
        left, right = spans[region]
        placed = _place_in_strip(height, left, right, patch)
        if placed is None:
            return None
        r0, c0 = placed
        info = region_roles[region]
        _paint_patch(grid, patch, r0, c0, info["color"])
        _paint_patch(expected, patch, r0, c0, info["color"])

    for item in assignments:
        if item["role"] != "pair":
            continue
        small_region = item["small_region"]
        large_region = item["large_region"]
        fill_color = region_roles[large_region]["color"]
        left, right = spans[small_region]
        window = expected[:, left : right + 1]
        window[window == BLACK] = int(fill_color)

    return grid, expected


def _apply_grid_transform(grid: np.ndarray, variant: int) -> np.ndarray:
    if variant == 0:
        return grid.copy()
    if variant == 1:
        return np.rot90(grid, 1).copy()
    if variant == 2:
        return np.rot90(grid, 2).copy()
    if variant == 3:
        return np.rot90(grid, 3).copy()
    if variant == 4:
        return np.fliplr(grid).copy()
    if variant == 5:
        return np.flipud(grid).copy()
    if variant == 6:
        return np.transpose(grid).copy()
    return np.fliplr(np.transpose(grid)).copy()


def _valid_generated_pair(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        records = _extract_region_records(inp)
    except AssertionError:
        return False

    if len(records) not in _GEN_REGION_OPTIONS:
        return False
    colors = [record["color"] for record in records]
    if len(colors) != len(set(colors)):
        return False

    pairs = _find_scaled_pairs(records)
    if len(pairs) != 2:
        return False
    used = [idx for pair in pairs for idx in pair]
    if len(set(used)) != 4:
        return False

    if np.array_equal(inp, expected):
        return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt(region_count: int, forced_variant: int | None) -> np.ndarray | None:
    widths = _sample_strip_widths(region_count)
    if widths is None:
        return None
    pair_defs = _pick_pair_shapes(widths)
    if pair_defs is None:
        return None
    if not _assign_regions(widths, pair_defs):
        return None

    height = random.randint(*_GEN_HEIGHT_RANGE)
    colors = random.sample(_GEN_PALETTE, region_count)
    rendered = _render_layout(height, widths, pair_defs, colors)
    if rendered is None:
        return None
    inp, expected = rendered

    variant = forced_variant if forced_variant is not None else random.randrange(8)
    inp = _apply_grid_transform(inp, variant)
    expected = _apply_grid_transform(expected, variant)

    if not _valid_generated_pair(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    region_count = _GEN_REGION_OPTIONS[_GEN_CALL_COUNT % len(_GEN_REGION_OPTIONS)]
    forced_variant = 6 if _GEN_CALL_COUNT % 2 else None
    _GEN_CALL_COUNT += 1

    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(region_count, forced_variant)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-scale flood-fill input")
