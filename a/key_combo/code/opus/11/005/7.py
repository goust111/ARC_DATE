import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

SPRITE_LIBRARY = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bool(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_pattern(coords: Sequence[Tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch


def _pattern_signature(patch: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in patch.astype(int))


def _find_value_components(grid: np.ndarray, value: int) -> List[Dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: List[Dict[str, object]] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) != int(value):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == int(value):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": int(value),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                    "pattern": _component_pattern(coords),
                }
            )
    return comps


def _analyse_input(grid: np.ndarray) -> Dict[str, object]:
    g = np.asarray(grid, dtype=int)
    grey_components = _find_value_components(g, GREY)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY))
    sprite_groups: Dict[int, Dict[str, object]] = {}
    sprite_coords: List[Tuple[int, int]] = []
    for color in colors:
        comps = _find_value_components(g, color)
        if not comps:
            continue
        anchor = min((comp["bbox"][0], comp["bbox"][1]) for comp in comps)
        sprite_groups[color] = {
            "color": color,
            "count": len(comps),
            "pattern": comps[0]["pattern"],
            "components": comps,
            "anchor": anchor,
        }
        for comp in comps:
            sprite_coords.extend(comp["coords"])
    return {
        "grey_components": grey_components,
        "sprite_groups": sprite_groups,
        "sprite_coords": sprite_coords,
    }


def _apply_stamp(out: np.ndarray, bbox: Tuple[int, int, int, int], stamp: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    hh = min(bh, stamp.shape[0])
    ww = min(bw, stamp.shape[1])
    cut = stamp[:hh, :ww]
    view = out[r0 : r0 + hh, c0 : c0 + ww]
    view[cut] = BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray regions by matching area rank to same-color sprite-count rank, then punch each region with that color's sprite.

    Input structure:
    - The grid has a black background with 3-4 separated gray 4-connected components.
    - It also has several small non-black, non-gray monochrome sprites; sprites of the same color share one shape.
    - The number of gray components equals the number of sprite colors, and each gray area and sprite-count total is unique.

    Rule:
    - Find all gray 4-connected components and sort them by area from largest to smallest.
    - Group the colored sprites by color, count how many sprites each color has, and sort colors by that count from largest to smallest.
    - Recolor the gray component of rank i with the sprite color of rank i.
    - Inside each recolored component's bounding box, stamp that color's sprite shape at the top-left corner; sprite foreground becomes black and sprite background leaves the component color unchanged, with clipping if needed.
    - Erase every original sprite from the canvas back to black.

    Key insight:
    - The final color assignment is a cross-reference between two independent rankings: gray-component area rank and same-color sprite-count rank.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    analysis = _analyse_input(g)
    grey_components = analysis["grey_components"]
    sprite_groups = analysis["sprite_groups"]

    assert len(grey_components) == len(sprite_groups) and len(grey_components) > 0

    for r, c in analysis["sprite_coords"]:
        out[r, c] = BLACK

    grey_sorted = sorted(grey_components, key=lambda comp: (-int(comp["area"]), comp["bbox"][0], comp["bbox"][1]))
    color_sorted = sorted(
        sprite_groups.values(),
        key=lambda group: (-int(group["count"]), group["anchor"][0], group["anchor"][1], int(group["color"])),
    )

    for comp, group in zip(grey_sorted, color_sorted):
        color = int(group["color"])
        for r, c in comp["coords"]:
            out[r, c] = color
        _apply_stamp(out, comp["bbox"], group["pattern"])

    return out


def _all_symmetries(patch: np.ndarray, *, allow_rotate: bool = True) -> List[np.ndarray]:
    variants: List[np.ndarray] = []
    seen = set()
    rotations = range(4) if allow_rotate else (0,)
    for k in rotations:
        rotated = np.rot90(patch, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = _pattern_signature(candidate)
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


def _stamp_overlap(component_patch: np.ndarray, stamp: np.ndarray) -> int:
    hh = min(component_patch.shape[0], stamp.shape[0])
    ww = min(component_patch.shape[1], stamp.shape[1])
    return int(np.sum(component_patch[:hh, :ww] & stamp[:hh, :ww]))


def _make_slender_component(area: int, axis: str) -> np.ndarray:
    if axis == "height2":
        width = (area + 1) // 2 if area % 2 == 1 else area // 2 + 1
        patch = np.ones((2, width), dtype=bool)
        deficit = int(patch.sum()) - area
        remove_order = [(0, width - 1), (1, 0)]
        for idx in range(deficit):
            rr, cc = remove_order[idx]
            patch[rr, cc] = False
        return patch
    height = (area + 1) // 2 if area % 2 == 1 else area // 2 + 1
    return _make_slender_component(area, "height2").T if axis == "width2" else np.ones((height, 2), dtype=bool)


def _grow_irregular_patch(rng: random.Random, area: int) -> np.ndarray | None:
    candidates = []
    for h in range(3, 7):
        for w in range(3, 7):
            if area < h + w - 1:
                continue
            if area > h * w - 1:
                continue
            candidates.append((h, w))
    rng.shuffle(candidates)
    for h, w in candidates:
        for _ in range(80):
            cells = {(rng.randrange(h), rng.randrange(w))}
            while len(cells) < area:
                frontier = []
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(rng.choice(frontier))
            if len(cells) != area:
                continue
            patch = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                patch[r, c] = True
            cropped = _crop_bool(patch)
            if cropped.shape != (h, w):
                continue
            if bool(np.all(cropped)):
                continue
            return cropped
    return None


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    H, W = occupied.shape
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if bool(np.any(occupied[r1:r2, c1:c2])):
            return False
    return True


def _paint_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    for dr, dc in np.argwhere(patch):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True


def _choose_sprite_patches(
    rng: random.Random, k: int, crop_rank: int | None, crop_axis: str | None
) -> List[np.ndarray] | None:
    chosen: List[np.ndarray] = [None] * k  # type: ignore[assignment]
    used = set()

    if crop_rank is not None and crop_axis is not None:
        special = []
        for base in SPRITE_LIBRARY:
            for variant in _all_symmetries(base, allow_rotate=True):
                if crop_axis == "height2" and variant.shape[0] == 3:
                    special.append(variant)
                if crop_axis == "width2" and variant.shape[1] == 3:
                    special.append(variant)
        rng.shuffle(special)
        for variant in special:
            key = _pattern_signature(variant)
            if key not in used:
                chosen[crop_rank] = variant
                used.add(key)
                break
        if chosen[crop_rank] is None:
            return None

    slots = [idx for idx in range(k) if chosen[idx] is None]
    candidates = []
    for base in SPRITE_LIBRARY:
        candidates.extend(_all_symmetries(base, allow_rotate=True))
    rng.shuffle(candidates)
    for idx in slots:
        found = False
        for variant in candidates:
            key = _pattern_signature(variant)
            if key in used:
                continue
            chosen[idx] = variant
            used.add(key)
            found = True
            break
        if not found:
            return None
    return [patch.copy() for patch in chosen]


def _validate_input_semantics(grid: np.ndarray) -> bool:
    analysis = _analyse_input(grid)
    grey_components = analysis["grey_components"]
    sprite_groups = analysis["sprite_groups"]
    if len(grey_components) not in (3, 4):
        return False
    if len(sprite_groups) != len(grey_components):
        return False
    grey_areas = [int(comp["area"]) for comp in grey_components]
    if len(set(grey_areas)) != len(grey_areas):
        return False
    if not all(8 <= area <= 25 for area in grey_areas):
        return False
    if any(bool(np.all(comp["pattern"])) for comp in grey_components):
        return False

    count_values = []
    signatures = []
    for group in sprite_groups.values():
        count_values.append(int(group["count"]))
        pattern = group["pattern"]
        signatures.append(_pattern_signature(pattern))
        comps = group["components"]
        if not all(3 <= int(comp["area"]) <= 5 for comp in comps):
            return False
        if not all(_pattern_signature(comp["pattern"]) == signatures[-1] for comp in comps):
            return False
    if len(set(count_values)) != len(count_values):
        return False
    if len(set(signatures)) != len(signatures):
        return False
    return True


def _build_expected_from_plan(
    shape: Tuple[int, int],
    components: List[Dict[str, object]],
    colors_ranked: List[int],
    sprite_patches_ranked: List[np.ndarray],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ranked_components = sorted(components, key=lambda comp: (-int(comp["area"]), comp["r"], comp["c"]))
    for rank, comp in enumerate(ranked_components):
        color = int(colors_ranked[rank])
        patch = comp["patch"]
        occupied = np.zeros(shape, dtype=bool)
        _paint_patch(out, occupied, patch, int(comp["r"]), int(comp["c"]), color)
        bbox = (int(comp["r"]), int(comp["c"]), int(comp["r"]) + patch.shape[0] - 1, int(comp["c"]) + patch.shape[1] - 1)
        _apply_stamp(out, bbox, sprite_patches_ranked[rank])
    return out


def _row_major_component_ids(components: List[Dict[str, object]]) -> List[int]:
    return [int(comp["id"]) for comp in sorted(components, key=lambda comp: (int(comp["r"]), int(comp["c"])))]


def _ranked_component_ids(components: List[Dict[str, object]]) -> List[int]:
    return [int(comp["id"]) for comp in sorted(components, key=lambda comp: (-int(comp["area"]), int(comp["r"]), int(comp["c"])))]


def _place_scene(rng: random.Random, H: int, W: int, components: List[Dict[str, object]], sprites: Dict[int, Dict[str, object]]) -> np.ndarray | None:
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    comp_order = list(range(len(components)))
    rng.shuffle(comp_order)
    for idx in comp_order:
        comp = components[idx]
        patch = comp["patch"]
        positions = [(r, c) for r in range(H - patch.shape[0] + 1) for c in range(W - patch.shape[1] + 1)]
        rng.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _can_place(occupied, patch, r0, c0, padding=1):
                comp["r"] = r0
                comp["c"] = c0
                _paint_patch(grid, occupied, patch, r0, c0, GREY)
                placed = True
                break
        if not placed:
            return None

    sprite_positions: Dict[int, List[Tuple[int, int]]] = {}
    for color, info in sprites.items():
        patch = info["patch"]
        positions = [(r, c) for r in range(H - patch.shape[0] + 1) for c in range(W - patch.shape[1] + 1)]
        sprite_positions[color] = []
        for _ in range(int(info["count"])):
            rng.shuffle(positions)
            placed = False
            for r0, c0 in positions:
                if _can_place(occupied, patch, r0, c0, padding=1):
                    sprite_positions[color].append((r0, c0))
                    _paint_patch(grid, occupied, patch, r0, c0, color)
                    placed = True
                    break
            if not placed:
                return None

    for color, placements in sprite_positions.items():
        sprites[color]["positions"] = placements
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    k = rng.choice([3, 4])
    H = rng.randint(20, 25)
    W = rng.randint(20, 25)
    colors_ranked = rng.sample(PALETTE, k)
    if colors_ranked == sorted(colors_ranked) or colors_ranked == sorted(colors_ranked, reverse=True):
        return None

    force_crop = rng.random() < 0.6
    components: List[Dict[str, object]] = []
    used_areas = set()
    slender_axis = None
    slender_id = None

    if force_crop:
        slender_axis = rng.choice(["height2", "width2"])
        slender_area = rng.randint(8, 10)
        components.append({"id": 0, "patch": _make_slender_component(slender_area, slender_axis), "area": slender_area})
        used_areas.add(slender_area)
        slender_id = 0

    while len(components) < k:
        area = rng.randint(8, 23)
        if area in used_areas:
            continue
        patch = _grow_irregular_patch(rng, area)
        if patch is None:
            continue
        components.append({"id": len(components), "patch": patch, "area": area})
        used_areas.add(area)

    ranked_components = sorted(components, key=lambda comp: (-int(comp["area"]), int(comp["id"])))
    crop_rank = None
    crop_axis = None
    if force_crop and slender_id is not None:
        for idx, comp in enumerate(ranked_components):
            if int(comp["id"]) == slender_id:
                crop_rank = idx
                crop_axis = slender_axis
                break

    sprite_patches_ranked = _choose_sprite_patches(rng, k, crop_rank, crop_axis)
    if sprite_patches_ranked is None:
        return None

    for rank, comp in enumerate(ranked_components):
        base_patch = comp["patch"]
        allow_rotate = not (force_crop and crop_rank == rank)
        variants = _all_symmetries(base_patch, allow_rotate=allow_rotate)
        rng.shuffle(variants)
        best_variant = max(variants, key=lambda variant: _stamp_overlap(variant, sprite_patches_ranked[rank]))
        if _stamp_overlap(best_variant, sprite_patches_ranked[rank]) <= 0:
            return None
        comp["patch"] = best_variant

    counts_ranked = sorted(rng.sample(range(1, 6), k), reverse=True)
    sprites: Dict[int, Dict[str, object]] = {}
    for rank, color in enumerate(colors_ranked):
        sprites[color] = {"patch": sprite_patches_ranked[rank], "count": counts_ranked[rank]}

    grid = _place_scene(rng, H, W, components, sprites)
    if grid is None:
        return None

    area_rank_ids = _ranked_component_ids(components)
    row_major_ids = _row_major_component_ids(components)
    if area_rank_ids == row_major_ids:
        return None

    group_anchors = []
    for color in colors_ranked:
        anchor = min(sprites[color]["positions"])
        group_anchors.append((anchor[0], anchor[1], color))
    by_anchor = [color for _, _, color in sorted(group_anchors)]
    if by_anchor == colors_ranked:
        return None

    expected = _build_expected_from_plan((H, W), components, colors_ranked, sprite_patches_ranked)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if not _validate_input_semantics(grid):
        return None

    if force_crop and crop_rank is not None:
        comp = sorted(components, key=lambda item: (-int(item["area"]), int(item["r"]), int(item["c"])))[crop_rank]
        stamp = sprite_patches_ranked[crop_rank]
        hh = min(comp["patch"].shape[0], stamp.shape[0])
        ww = min(comp["patch"].shape[1], stamp.shape[1])
        if hh == stamp.shape[0] and ww == stamp.shape[1]:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.getrandbits(64))
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
