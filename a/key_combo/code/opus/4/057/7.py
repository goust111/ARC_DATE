from __future__ import annotations

import hashlib
import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
bbox_w = _ARC_UTILS.bbox_w
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


Pattern = tuple[tuple[int, int], ...]
SpriteSpec = tuple[int, int, Pattern]

PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)


def _sprite_cells(spec: SpriteSpec) -> set[tuple[int, int]]:
    r0, c0, pattern = spec
    return {(r0 + dr, c0 + dc) for dr, dc in pattern}


def _mirror_cells(cells: set[tuple[int, int]], width: int) -> set[tuple[int, int]]:
    return {(r, width - 1 - c) for r, c in cells}


def _mirrored_sprite_cells(spec: SpriteSpec, width: int) -> set[tuple[int, int]]:
    r0, c0, pattern = spec
    dst_c0 = width - 2 - c0
    return {(r0 + dr, dst_c0 + dc) for dr, dc in pattern}


def _component_records(mask: np.ndarray) -> list[dict]:
    records = []
    for coords in components_mask(mask, connectivity=4):
        coords = coords.astype(int)
        records.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": bbox_coords(coords),
            }
        )
    records.sort(key=lambda rec: (-rec["area"], rec["bbox"][0], rec["bbox"][1]))
    return records


def _selected_component_from_occupied(occupied: np.ndarray) -> dict:
    comps = _component_records(~occupied)
    assert comps, "expected at least one complement component"
    if len(comps) > 1 and comps[0]["area"] == comps[1]["area"]:
        assert comps[0]["bbox"][0] != comps[1]["bbox"][0], "largest-area tie must be broken by top row"
    return comps[0]


def _output_from_component(shape: tuple[int, int], component: dict) -> np.ndarray:
    full = np.zeros(shape, dtype=int)
    for r, c in component["coords"]:
        full[int(r), int(c)] = Color.GREY
    return crop_bbox(full, component["bbox"])


def _occupied_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = map(int, g.shape)
    sprites = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert 4 <= len(sprites) <= 6, "expected 4 to 6 sprites"

    occupied = g != bg
    mirrored = np.zeros_like(g)
    left_limit = (2 * width) // 3
    for sprite in sprites:
        bbox = tuple(map(int, sprite["bbox"]))
        r0, c0, _r1, _c1 = bbox
        assert bbox[2] - bbox[0] + 1 == 2 and bbox[3] - bbox[1] + 1 == 2, "each sprite must use a 2x2 bbox"
        patch = crop_bbox(g, bbox)
        patch_mask = (patch != bg).astype(int)
        area = int(patch_mask.sum())
        assert area in {2, 3}, "each sprite must have 2 or 3 occupied cells"
        assert len({int(v) for v in patch[patch != bg]}) >= 2, "each sprite must be multicolor"
        assert c0 >= 0 and c0 + 1 < left_limit, "sprites must stay in the left two-thirds"
        dst_c0 = width - bbox_w(bbox) - c0
        paste(mirrored, patch_mask, r0, dst_c0, transparent=0)

    return occupied | (mirrored != 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the largest 4-connected background gap after overlaying each sprite with its position-only horizontal mirror.

    Input structure:
    - A black-background rectangular grid contains 4 to 6 separated multicolor sprites.
    - Every sprite occupies a 2x2 bounding box and uses 2 or 3 non-black cells.
    - Sprites lie in the left two-thirds of the board, and only their occupied cells matter.

    Rule:
    - Detect each sprite as a 4-connected multicolor object, then place a second copy at the horizontally mirrored position while preserving the sprite's internal layout and colors.
    - Form an occupancy mask from the union of all original and mirrored occupied cells.
    - Turn every unoccupied cell grey and every occupied cell black, find all grey 4-connected components, then choose the largest one; if the largest area ties, choose the component whose bounding box starts higher.
    - Return the chosen component cropped to its tight bounding box, with grey kept on the component and black elsewhere inside the crop.

    Key insight:
    - The mirror copies are not themselves the answer; they only reshape the occupied mask, and the real output is the largest connected gap in that complemented mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    occupied = _occupied_from_grid(g)
    component = _selected_component_from_occupied(occupied)
    return _output_from_component(g.shape, component)


def _variant_output_signature(out: np.ndarray) -> tuple[tuple[int, int], str]:
    return tuple(map(int, out.shape)), hashlib.md5(out.tobytes()).hexdigest()


def _render_patch(pattern: Pattern, seed: int) -> np.ndarray:
    patch = np.zeros((2, 2), dtype=int)
    cells = sorted(pattern)
    rng = np.random.default_rng(seed)
    distinct = 2 if len(cells) == 2 else (3 if seed % 2 == 0 else 2)
    colors = list(map(int, rng.choice(PALETTE, size=distinct, replace=False).tolist()))
    assigned = colors[:]
    while len(assigned) < len(cells):
        assigned.append(int(colors[int(rng.integers(0, len(colors)))]))
    rng.shuffle(assigned)
    for (r, c), color in zip(cells, assigned):
        patch[r, c] = color
    return patch


def _render_variant(variant: dict) -> np.ndarray:
    grid = np.zeros((variant["height"], variant["width"]), dtype=int)
    variant_seed = int(hashlib.md5(variant["key"].encode()).hexdigest()[:8], 16)
    for i, spec in enumerate(variant["specs"]):
        r0, c0, pattern = spec
        patch = _render_patch(pattern, variant_seed + 97 * (i + 1))
        paste(grid, patch, r0, c0)
    return grid


def _validate_original_sprites(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    sprites = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert 4 <= len(sprites) <= 6, "sprite count must stay within the puzzle range"
    occupied_sets = []
    width = int(grid.shape[1])
    left_limit = (2 * width) // 3
    for sprite in sprites:
        bbox = tuple(map(int, sprite["bbox"]))
        patch = crop_bbox(grid, bbox)
        assert patch.shape == (2, 2), "each sprite must keep a 2x2 bbox"
        assert int(np.count_nonzero(patch)) in {2, 3}, "each sprite must keep 2 or 3 visible cells"
        assert len({int(v) for v in patch[patch != 0]}) >= 2, "each sprite must stay multicolor"
        assert bbox[1] + 1 < left_limit, "sprite must remain inside the left two-thirds"
        occupied = {(int(r), int(c)) for r, c in sprite["coords"]}
        occupied_sets.append(occupied)
    for i, a in enumerate(occupied_sets):
        for b in occupied_sets[i + 1 :]:
            dist = min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)
            assert dist >= 2, "sprites must stay at least Manhattan distance 2 apart"
    return sprites


def _analyze_layout(specs: list[SpriteSpec], height: int, width: int) -> dict | None:
    if max(r + 1 for r, _c, _p in specs) >= height or max(c + 1 for _r, c, _p in specs) >= width:
        return None

    original_sets = [_sprite_cells(spec) for spec in specs]
    for i, a in enumerate(original_sets):
        for b in original_sets[i + 1 :]:
            dist = min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)
            if dist < 2:
                return None

    original = set().union(*original_sets)
    mirrored = set().union(*(_mirrored_sprite_cells(spec, width) for spec in specs))
    if original & mirrored:
        return None

    occupied = np.zeros((height, width), dtype=bool)
    for r, c in original | mirrored:
        if not (0 <= r < height and 0 <= c < width):
            return None
        occupied[r, c] = True

    components = _component_records(~occupied)
    if not components:
        return None

    selected = components[0]
    expected = _output_from_component((height, width), selected)
    return {
        "height": int(height),
        "width": int(width),
        "specs": list(specs),
        "occupied": occupied,
        "components": components,
        "selected": selected,
        "expected": expected,
    }


def _analyze_variant(specs: list[SpriteSpec], height: int, width: int) -> dict | None:
    analyzed = _analyze_layout(specs, height, width)
    if analyzed is None:
        return None
    if len(analyzed["components"]) < 2:
        return None
    sizes = [rec["area"] for rec in analyzed["components"]]
    if len(set(sizes)) < 2:
        return None
    return analyzed


def _all_sprites_matter(base: dict) -> bool:
    target_sig = _variant_output_signature(base["expected"])
    specs = base["specs"]
    for i in range(len(specs)):
        reduced = _analyze_layout(specs[:i] + specs[i + 1 :], base["height"], base["width"])
        if reduced is None:
            return False
        if _variant_output_signature(reduced["expected"]) == target_sig:
            return False
    return True


def _build_variants() -> list[dict]:
    rng = random.Random(5707)
    patterns = [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ]
    variants: list[dict] = []
    seen_outputs: set[tuple[tuple[int, int], str]] = set()
    for attempt in range(20000):
        height = rng.randint(8, 13)
        width = rng.randint(16, 24)
        sprite_count = rng.randint(4, 6)
        left_limit = (2 * width) // 3
        positions = [(r, c) for r in range(height - 1) for c in range(left_limit - 1)]
        rng.shuffle(positions)

        specs: list[SpriteSpec] = []
        occupied_sets: list[set[tuple[int, int]]] = []
        for r0, c0 in positions:
            pattern = rng.choice(patterns)
            cells = _sprite_cells((r0, c0, pattern))
            candidate_specs = specs + [(r0, c0, pattern)]
            all_cells = set().union(*(occupied_sets + [cells]))
            mirrored_cells = set().union(*(_mirrored_sprite_cells(spec, width) for spec in candidate_specs))
            if all_cells & mirrored_cells:
                continue
            too_close = False
            for old in occupied_sets:
                dist = min(abs(ar - br) + abs(ac - bc) for ar, ac in cells for br, bc in old)
                if dist < 2:
                    too_close = True
                    break
            if too_close:
                continue
            specs.append((r0, c0, pattern))
            occupied_sets.append(cells)
            if len(specs) == sprite_count:
                break
        if len(specs) != sprite_count:
            continue

        analyzed = _analyze_variant(specs, height, width)
        if analyzed is None or not _all_sprites_matter(analyzed):
            continue
        sizes = [rec["area"] for rec in analyzed["components"]]
        if sizes[0] <= sizes[1]:
            continue

        out_sig = _variant_output_signature(analyzed["expected"])
        if out_sig in seen_outputs:
            continue
        seen_outputs.add(out_sig)
        analyzed["name"] = "searched"
        analyzed["mode"] = "unique"
        analyzed["key"] = f"searched|{attempt}|{height}|{width}|{sprite_count}"
        variants.append(analyzed)
        if len(variants) >= 40:
            break

    assert len(variants) >= 20, f"expected at least 20 distinct-output variants, got {len(variants)}"
    variants.sort(key=lambda item: item["key"])
    return variants


_VARIANTS = _build_variants()
_ORDER_RNG = np.random.default_rng()
_ORDER = list(range(len(_VARIANTS)))
_ORDER_RNG.shuffle(_ORDER)
_ORDER_INDEX = 0


def _next_variant() -> dict:
    global _ORDER, _ORDER_INDEX
    if _ORDER_INDEX >= len(_ORDER):
        _ORDER = list(range(len(_VARIANTS)))
        _ORDER_RNG.shuffle(_ORDER)
        _ORDER_INDEX = 0
    variant = _VARIANTS[_ORDER[_ORDER_INDEX]]
    _ORDER_INDEX += 1
    return variant


def generate_input() -> np.ndarray:
    variant = _next_variant()
    grid = _render_variant(variant)

    sprites = _validate_original_sprites(grid)
    occupied = _occupied_from_grid(grid)
    selected = _selected_component_from_occupied(occupied)
    expected = _output_from_component(grid.shape, selected)
    assert np.array_equal(expected, variant["expected"]), "rendered grid must preserve the intended geometry"
    assert np.array_equal(main(grid), expected), "main() must match the independently analyzed target"

    if variant["mode"] == "tie":
        occupied_sets = [set(map(tuple, sprite["coords"])) for sprite in sprites]
        assert len(occupied_sets) == 6, "tie branch should still use six sprites"
    return grid
