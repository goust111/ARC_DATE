from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste, random_free_location


BG = Color.BLACK
OCCLUDER = Color.BLUE
TEMPLATE_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_SIDE_RANGE = (18, 25)
TEMPLATE_SHAPES = ((3, 4), (4, 3))
GENERATION_ATTEMPTS = 400


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _is_filled_rectangle(sub: np.ndarray) -> bool:
    return bool(np.all((sub != BG) & (sub != OCCLUDER)))


def _dihedral_variants(template: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(template, k)
        for reflected, arr in ((False, rotated), (True, np.fliplr(rotated))):
            key = (arr.shape, arr.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "array": arr.copy(),
                    "rot": k,
                    "reflected": reflected,
                    "shape": arr.shape,
                }
            )
    return variants


def _border_signature(arr: np.ndarray) -> tuple:
    h, w = arr.shape
    coords = []
    for c in range(w):
        coords.append((0, c))
    for r in range(1, h - 1):
        coords.append((r, w - 1))
    if h > 1:
        for c in range(w - 1, -1, -1):
            coords.append((h - 1, c))
    if w > 1:
        for r in range(h - 2, 0, -1):
            coords.append((r, 0))
    return (arr.shape, tuple(int(arr[r, c]) for r, c in coords))


def _template_ok(template: np.ndarray) -> bool:
    variants = _dihedral_variants(template)
    if len(variants) != 8:
        return False
    if len({v["shape"] for v in variants}) != 2:
        return False
    if len({_border_signature(v["array"]) for v in variants}) != 8:
        return False
    return True


def _generate_template() -> np.ndarray:
    for _ in range(200):
        h, w = random.choice(TEMPLATE_SHAPES)
        color_count = random.randint(3, 4)
        palette = random.sample(TEMPLATE_COLORS, color_count)
        cells = [palette[i % color_count] for i in range(h * w)]
        random.shuffle(cells)
        template = np.array(cells, dtype=int).reshape(h, w)
        if _template_ok(template):
            return template
    raise RuntimeError("failed to generate a non-degenerate template")


def _find_template_component(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    H, W = grid.shape
    fg_colors = [int(c) for c in np.unique(grid) if c not in (BG, OCCLUDER)]
    comps = find_components(
        grid,
        bg=BG,
        colors=fg_colors,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    matches = []
    for comp in comps:
        b = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = b
        h, w = _bbox_dims(b)
        if int(comp["area"]) != h * w:
            continue
        if not ((r0 == 0 and c0 == 0) or (r1 == H - 1 and c1 == W - 1)):
            continue
        sub = crop_bbox(grid, b)
        if _is_filled_rectangle(sub):
            matches.append((sub, b))
    assert len(matches) == 1, "expected exactly one corner template rectangle"
    return matches[0]


def _blue_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        colors=[OCCLUDER],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        b = tuple(map(int, comp["bbox"]))
        h, w = _bbox_dims(b)
        assert int(comp["area"]) == h * w, "occluders must be solid rectangles"
    return comps


def _match_candidates(
    grid: np.ndarray,
    variants: list[dict],
    blue_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int]]:
    H, W = grid.shape
    br0, bc0, br1, bc1 = blue_bbox
    matches: list[tuple[int, int, int]] = []
    for idx, variant in enumerate(variants):
        arr = variant["array"]
        h, w = arr.shape
        row_lo = max(0, br1 - h + 2)
        row_hi = min(br0 - 1, H - h)
        col_lo = max(0, bc1 - w + 2)
        col_hi = min(bc0 - 1, W - w)
        if row_lo > row_hi or col_lo > col_hi:
            continue
        for r0 in range(row_lo, row_hi + 1):
            for c0 in range(col_lo, col_hi + 1):
                sub = grid[r0 : r0 + h, c0 : c0 + w]
                local_blue = np.zeros((h, w), dtype=bool)
                local_blue[br0 - r0 : br1 - r0 + 1, bc0 - c0 : bc1 - c0 + 1] = True
                if not np.all(sub[local_blue] == OCCLUDER):
                    continue
                if np.any(sub[~local_blue] != arr[~local_blue]):
                    continue
                matches.append((idx, r0, c0))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each blue-masked stamp by inferring which dihedral transform of the corner template fits its visible border.

    Input structure:
    - The background is black.
    - One complete multicolor rectangular template sits isolated in the top-left or bottom-right corner.
    - Elsewhere are non-overlapping copies of that template under various rotations/reflections; 2-3 copies contain solid blue inner rectangles that leave a visible colored border.

    Rule:
    - Detect the corner template and generate all eight dihedral transforms of it.
    - For each blue rectangle, test every transformed template placement whose rectangle strictly contains the blue box.
    - The unique placement whose visible non-blue cells exactly match the exposed border identifies the hidden transform; fill the blue cells with the corresponding template pixels.

    Key insight:
    - The blue block does not tell you the answer by itself; the exposed border around each block is a partial template observation that uniquely selects one dihedral variant.
    """
    g = np.asarray(grid, dtype=int)
    template, _template_bbox = _find_template_component(g)
    variants = _dihedral_variants(template)

    out = g.copy()
    for comp in _blue_components(g):
        blue_bbox = tuple(map(int, comp["bbox"]))
        matches = _match_candidates(g, variants, blue_bbox)
        assert len(matches) == 1, "each occluder must match exactly one transformed template"
        variant_idx, r0, c0 = matches[0]
        arr = variants[variant_idx]["array"]
        br0, bc0, br1, bc1 = blue_bbox
        out[br0 : br1 + 1, bc0 : bc1 + 1] = arr[br0 - r0 : br1 - r0 + 1, bc0 - c0 : bc1 - c0 + 1]
    return out


def _shape_groups(variants: list[dict]) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for idx, variant in enumerate(variants):
        groups.setdefault(tuple(variant["shape"]), []).append(idx)
    return list(groups.values())


def _choose_variant_ids(variants: list[dict]) -> tuple[list[int], list[int]]:
    groups = _shape_groups(variants)
    same_shape_pairs = [
        (a, b)
        for group in groups
        for a, b in combinations(group, 2)
        if variants[a]["reflected"] or variants[b]["reflected"]
    ]
    if not same_shape_pairs:
        raise RuntimeError("template lacks useful same-shape dihedral pairs")

    occluded_ids = list(random.choice(same_shape_pairs))
    if random.random() < 0.6:
        remaining = [i for i in range(len(variants)) if i not in occluded_ids]
        other_shape = [i for i in remaining if variants[i]["shape"] != variants[occluded_ids[0]]["shape"]]
        pool = other_shape or remaining
        third = random.choice(pool)
        occluded_ids.append(third)

    if not any(variants[i]["reflected"] for i in occluded_ids):
        reflected = [i for i in range(len(variants)) if variants[i]["reflected"] and i not in occluded_ids]
        if reflected:
            occluded_ids[-1] = random.choice(reflected)

    unoccluded_count = random.randint(1, 2)
    pool = [i for i in range(len(variants)) if i not in occluded_ids]
    if not pool:
        pool = list(range(len(variants)))
    random.shuffle(pool)
    unoccluded_ids = pool[:unoccluded_count]
    if len(unoccluded_ids) < unoccluded_count:
        while len(unoccluded_ids) < unoccluded_count:
            candidate = random.randrange(len(variants))
            if candidate not in unoccluded_ids:
                unoccluded_ids.append(candidate)
    return occluded_ids, unoccluded_ids


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int]:
    r, c = random_free_location(occupied, np.ones_like(patch), transparent=0, border=1, padding=1, padding_connectivity=8)
    paste(grid, patch, r, c)
    paste(occupied, np.ones_like(patch), r, c, transparent=0)
    return r, c


def _visible_variant_ids(grid: np.ndarray, variants: list[dict], template_bbox: tuple[int, int, int, int]) -> list[int]:
    fg_colors = [int(c) for c in np.unique(grid) if c not in (BG, OCCLUDER)]
    comps = find_components(
        grid,
        bg=BG,
        colors=fg_colors,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    visible_ids: list[int] = []
    for comp in comps:
        b = tuple(map(int, comp["bbox"]))
        if b == template_bbox:
            continue
        h, w = _bbox_dims(b)
        if int(comp["area"]) != h * w:
            continue
        sub = crop_bbox(grid, b)
        ids = [idx for idx, variant in enumerate(variants) if np.array_equal(sub, variant["array"])]
        assert len(ids) == 1, "visible full stamps must map to a unique variant"
        visible_ids.append(ids[0])
    return visible_ids


def _validate_instance(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    template, template_bbox = _find_template_component(grid)
    variants = _dihedral_variants(template)
    blues = _blue_components(grid)
    if len(blues) not in (2, 3):
        return False

    matched_ids: list[int] = []
    for comp in blues:
        matches = _match_candidates(grid, variants, tuple(map(int, comp["bbox"])))
        if len(matches) != 1:
            return False
        matched_ids.append(matches[0][0])

    if len(set(matched_ids)) < 2:
        return False
    if not any(variants[idx]["reflected"] for idx in matched_ids):
        return False
    if not any(
        variants[a]["shape"] == variants[b]["shape"] and a != b
        for a, b in combinations(matched_ids, 2)
    ):
        return False

    visible_ids = _visible_variant_ids(grid, variants, template_bbox)
    if not visible_ids:
        return False
    if not any(idx not in visible_ids for idx in matched_ids):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        template = _generate_template()
        variants = _dihedral_variants(template)
        occluded_ids, unoccluded_ids = _choose_variant_ids(variants)
        total_instances = len(occluded_ids) + len(unoccluded_ids)

        H = random.randint(*GRID_SIDE_RANGE)
        W = random.randint(*GRID_SIDE_RANGE)
        grid = np.zeros((H, W), dtype=int)
        expected = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=int)

        th, tw = template.shape
        corner_top_left = random.random() < 0.5
        template_r = 0 if corner_top_left else H - th
        template_c = 0 if corner_top_left else W - tw
        if template_r < 0 or template_c < 0:
            continue

        paste(grid, template, template_r, template_c)
        paste(expected, template, template_r, template_c)
        paste(occupied, np.ones_like(template), template_r, template_c, transparent=0)

        placements: list[tuple[int, int, int]] = []
        try:
            for variant_id in occluded_ids + unoccluded_ids:
                patch = variants[variant_id]["array"]
                r, c = _place_patch(grid, occupied, patch)
                paste(expected, patch, r, c)
                placements.append((variant_id, r, c))
        except ValueError:
            continue

        for variant_id, r, c in placements[: len(occluded_ids)]:
            patch = variants[variant_id]["array"]
            h, w = patch.shape
            if h < 3 or w < 3:
                break
            grid[r + 1 : r + h - 1, c + 1 : c + w - 1] = OCCLUDER
        else:
            if total_instances < 3:
                continue
            if _validate_instance(grid, expected):
                return grid

    raise RuntimeError("failed to generate a valid edge-inference repair puzzle")
