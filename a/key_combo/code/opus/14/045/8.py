from __future__ import annotations

import random
from typing import Any

import numpy as np


TRANSFORM_NAMES = (
    "identity",
    "hflip",
    "vflip",
    "rot90_cw",
    "rot90_cw_hflip",
    "rot180",
    "rot270_cw",
    "rot270_cw_hflip",
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _tight_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    bbox = _bbox_from_coords(coords)
    return mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1].copy(), bbox


def _apply_transform(mask: np.ndarray, transform_idx: int) -> np.ndarray:
    if transform_idx == 0:
        return mask.copy()
    if transform_idx == 1:
        return np.fliplr(mask)
    if transform_idx == 2:
        return np.flipud(mask)
    if transform_idx == 3:
        return np.rot90(mask, -1)
    if transform_idx == 4:
        return np.fliplr(np.rot90(mask, -1))
    if transform_idx == 5:
        return np.rot90(mask, 2)
    if transform_idx == 6:
        return np.rot90(mask, 1)
    if transform_idx == 7:
        return np.fliplr(np.rot90(mask, 1))
    raise ValueError(f"unknown transform index: {transform_idx}")


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _first_match_transform(source_mask: np.ndarray, target_mask: np.ndarray) -> int | None:
    for idx in range(len(TRANSFORM_NAMES)):
        transformed = _apply_transform(source_mask, idx)
        if transformed.shape == target_mask.shape and np.array_equal(transformed, target_mask):
            return idx
    return None


def _template_or_sprite_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict[str, Any]:
    bbox = _bbox_from_coords(coords)
    crop = _crop(grid, bbox)
    h, w = crop.shape
    black_inside = _components(crop == 0)
    enclosed_black = [comp for comp in black_inside if not _touches_border(comp, crop.shape)]

    is_template = (
        h >= 4
        and w >= 4
        and np.all(crop[0, :] == color)
        and np.all(crop[-1, :] == color)
        and np.all(crop[:, 0] == color)
        and np.all(crop[:, -1] == color)
        and len(enclosed_black) == 1
    )
    if not is_template:
        return {
            "kind": "sprite",
            "color": int(color),
            "coords": coords,
            "bbox": bbox,
            "mask": crop == color,
        }

    hole_coords = enclosed_black[0]
    hole_bbox = _bbox_from_coords(hole_coords)
    hole_crop = crop[hole_bbox[0] : hole_bbox[2] + 1, hole_bbox[1] : hole_bbox[3] + 1]
    target_full = hole_crop == color
    target_components = _components(target_full)
    assert len(target_components) == 1, "template complement inside hole box must be 4-connected"
    target_mask, target_local_bbox = _tight_mask(target_full)

    hole_global_bbox = (
        bbox[0] + hole_bbox[0],
        bbox[1] + hole_bbox[1],
        bbox[0] + hole_bbox[2],
        bbox[1] + hole_bbox[3],
    )
    target_global_bbox = (
        hole_global_bbox[0] + target_local_bbox[0],
        hole_global_bbox[1] + target_local_bbox[1],
        hole_global_bbox[0] + target_local_bbox[2],
        hole_global_bbox[1] + target_local_bbox[3],
    )

    return {
        "kind": "template",
        "color": int(color),
        "coords": coords,
        "bbox": bbox,
        "hole_coords_local": hole_coords,
        "hole_bbox_global": hole_global_bbox,
        "hole_area": int(len(hole_coords)),
        "target_mask": target_mask,
        "target_bbox_global": target_global_bbox,
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    g = np.asarray(grid, dtype=int)
    templates: list[dict[str, Any]] = []
    sprites: list[dict[str, Any]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        for coords in _components(g == color):
            info = _template_or_sprite_info(g, color, coords)
            if info["kind"] == "template":
                templates.append(info)
            else:
                sprites.append(info)

    templates.sort(key=lambda item: item["bbox"])
    sprites.sort(key=lambda item: item["bbox"])
    return templates, sprites


def _solve_pairings(
    templates: list[dict[str, Any]],
    sprites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assert len(templates) == len(sprites), "template count and sprite count must match"
    pairings: list[dict[str, Any]] = []

    for sprite in sprites:
        sprite_mask = sprite["mask"]
        chosen_transform = None
        chosen_templates: list[dict[str, Any]] = []
        chosen_mask = None
        for transform_idx in range(len(TRANSFORM_NAMES)):
            transformed = _apply_transform(sprite_mask, transform_idx)
            matches = [
                template
                for template in templates
                if transformed.shape == template["target_mask"].shape
                and np.array_equal(transformed, template["target_mask"])
            ]
            if matches:
                chosen_transform = transform_idx
                chosen_templates = matches
                chosen_mask = transformed
                break

        assert chosen_transform is not None, "every sprite must match a template complement"
        assert len(chosen_templates) == 1, "the earliest matching transform must identify one template"
        pairings.append(
            {
                "sprite": sprite,
                "template": chosen_templates[0],
                "transform_idx": chosen_transform,
                "transformed_mask": chosen_mask,
            }
        )

    matched_template_bboxes = [pair["template"]["target_bbox_global"] for pair in pairings]
    assert len(set(matched_template_bboxes)) == len(templates), "matching must be one-to-one"
    return pairings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lower sprite to the unique template-hole complement, invert it, and stamp that negative into the hole region.

    Input structure:
    - The upper half contains 2-3 solid monochrome template rectangles on a black background.
    - Each template encloses one black 4-connected hole; inside the hole's tight box, some cells remain template-colored as a complementary mask.
    - The lower half contains 2-3 small monochrome sprites on black, one per template, with distinct colors.

    Rule:
    - For each sprite, extract its tight binary mask.
    - For each template, extract the tight binary mask of the template-colored complementary cells inside the enclosed hole's local box.
    - Compare each sprite against those complementary masks using this ordered transform list: identity, horizontal flip, vertical flip, 90° clockwise, 90° clockwise plus horizontal flip, 180°, 270° clockwise, 270° clockwise plus horizontal flip.
    - For the unique earliest match, invert the transformed sprite inside its own box: sprite pixels become black, box-background pixels become the matched template color.
    - Stamp that inverted patch back into the matched complementary sub-box, erase the original lower sprite, and keep all other template pixels unchanged.

    Key insight:
    - The lower objects do not match the black hole directly; they match the hole's complementary colored remainder, so the moved result becomes a visible negative inside the template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates, sprites = _parse_scene(g)
    assert 2 <= len(templates) <= 3, "expected 2-3 templates"
    assert len(sprites) == len(templates), "expected one sprite per template"

    pairings = _solve_pairings(templates, sprites)
    out = g.copy()

    for pair in pairings:
        sprite = pair["sprite"]
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = 0

    for pair in pairings:
        template = pair["template"]
        patch = np.where(pair["transformed_mask"], 0, template["color"]).astype(int)
        r0, c0, r1, c1 = template["target_bbox_global"]
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


PATTERN_LIBRARY: dict[str, np.ndarray] = {
    "l3": np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "square4": np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "bar3": np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    "rect6": np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
}


def _render_template_patch(
    hole_mask: np.ndarray,
    color: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    hole_h, hole_w = hole_mask.shape
    extra_h = 1 if hole_h == 3 and random.random() < 0.5 else 0
    extra_w = 1 if hole_w == 3 and random.random() < 0.5 else 0
    if hole_h == 2:
        extra_h = 0
    if hole_w >= 4:
        extra_w = 0

    top_margin = 1 + (random.randint(0, extra_h) if extra_h else 0)
    bottom_margin = 1 + (extra_h - (top_margin - 1))
    left_margin = 1 + (random.randint(0, extra_w) if extra_w else 0)
    right_margin = 1 + (extra_w - (left_margin - 1))

    patch_h = top_margin + hole_h + bottom_margin
    patch_w = left_margin + hole_w + right_margin
    patch = np.full((patch_h, patch_w), int(color), dtype=int)
    hole_r0 = top_margin
    hole_c0 = left_margin
    patch[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w][hole_mask] = 0

    hole_bbox = (hole_r0, hole_c0, hole_r0 + hole_h - 1, hole_c0 + hole_w - 1)
    complement_mask = ~hole_mask
    target_mask, target_bbox_local = _tight_mask(complement_mask)
    target_bbox = (
        hole_bbox[0] + target_bbox_local[0],
        hole_bbox[1] + target_bbox_local[1],
        hole_bbox[0] + target_bbox_local[2],
        hole_bbox[1] + target_bbox_local[3],
    )
    return patch, {"hole_bbox": hole_bbox, "target_mask": target_mask, "target_bbox": target_bbox}


def _unique_variants_for_target(target_mask: np.ndarray) -> list[tuple[np.ndarray, int]]:
    variants: dict[tuple[tuple[int, int], bytes], tuple[np.ndarray, int]] = {}
    for idx in range(len(TRANSFORM_NAMES)):
        source_mask = _apply_transform(target_mask, idx)
        sig = _mask_signature(source_mask)
        if sig in variants:
            continue
        earliest = _first_match_transform(source_mask, target_mask)
        assert earliest is not None
        variants[sig] = (source_mask, earliest)
    return list(variants.values())


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return
    rr = coords[:, 0] + int(top)
    cc = coords[:, 1] + int(left)
    grid[rr, cc] = int(color)


def _template_catalog_entry(name: str) -> dict[str, Any]:
    base_hole = PATTERN_LIBRARY[name]
    unique_holes: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for idx in range(len(TRANSFORM_NAMES)):
        hole = _apply_transform(base_hole, idx)
        unique_holes[_mask_signature(hole)] = hole
    hole_mask = random.choice(list(unique_holes.values()))
    target_mask, _ = _tight_mask(~hole_mask)
    patch, patch_info = _render_template_patch(hole_mask, color=1)
    variants = _unique_variants_for_target(target_mask)
    source_mask, earliest = random.choice(variants)
    return {
        "name": name,
        "hole_mask": hole_mask,
        "target_mask": target_mask,
        "source_mask": source_mask,
        "earliest_transform": earliest,
        "template_patch_shape": patch.shape,
        "render_info": patch_info,
    }


def _build_instance() -> np.ndarray | None:
    num_templates = 2 if random.random() < 0.6 else 3
    chosen_names = ["l3"] + random.sample(["square4", "bar3", "rect6"], num_templates - 1)
    random.shuffle(chosen_names)
    catalog = [_template_catalog_entry(name) for name in chosen_names]

    if all(entry["earliest_transform"] == 0 for entry in catalog):
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    template_colors = palette[:num_templates]
    sprite_colors = palette[num_templates : 2 * num_templates]

    upper_patches: list[dict[str, Any]] = []
    for idx, entry in enumerate(catalog):
        patch, render = _render_template_patch(entry["hole_mask"], template_colors[idx])
        upper_patches.append(
            {
                "patch": patch,
                "color": template_colors[idx],
                "source_mask": entry["source_mask"],
                "target_mask": render["target_mask"],
                "hole_bbox_local": render["hole_bbox"],
                "target_bbox_local": render["target_bbox"],
                "earliest_transform": entry["earliest_transform"],
                "sprite_color": sprite_colors[idx],
            }
        )

    template_gaps = [random.randint(2, 3) for _ in range(num_templates - 1)]
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    top_margin = random.randint(0, 1)
    upper_height = max(patch["patch"].shape[0] for patch in upper_patches)
    width = left_margin + right_margin + sum(patch["patch"].shape[1] for patch in upper_patches) + sum(template_gaps)
    width = max(width, 14)

    min_h = max(14, top_margin + upper_height + 6)
    max_h = min(18, top_margin + upper_height + 8)
    if min_h > max_h:
        return None
    grid_h = random.randint(min_h, max_h)
    if width > 24:
        return None

    grid = np.zeros((grid_h, width), dtype=int)
    placed_bboxes: list[tuple[int, int, int, int]] = []

    cur_left = left_margin
    for patch_info in upper_patches:
        patch = patch_info["patch"]
        ph, pw = patch.shape
        bbox = (top_margin, cur_left, top_margin + ph - 1, cur_left + pw - 1)
        if any(not _bbox_gap_ok(bbox, old, pad=1) for old in placed_bboxes):
            return None
        grid[top_margin : top_margin + ph, cur_left : cur_left + pw] = patch
        patch_info["bbox_global"] = bbox
        placed_bboxes.append(bbox)
        cur_left += pw
        if patch_info is not upper_patches[-1]:
            cur_left += template_gaps[len(placed_bboxes) - 1]

    sprite_order = list(range(num_templates))
    random.shuffle(sprite_order)
    sprite_bboxes: list[tuple[int, int, int, int]] = []
    lower_top_min = top_margin + upper_height + 2

    for order_idx, patch_idx in enumerate(sprite_order):
        sprite_mask = upper_patches[patch_idx]["source_mask"]
        sh, sw = sprite_mask.shape
        candidates: list[tuple[int, int]] = []
        for top in range(lower_top_min, grid_h - sh + 1):
            for left in range(0, width - sw + 1):
                bbox = (top, left, top + sh - 1, left + sw - 1)
                if any(not _bbox_gap_ok(bbox, old, pad=1) for old in placed_bboxes + sprite_bboxes):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paint_mask(grid, sprite_mask, top, left, upper_patches[patch_idx]["sprite_color"])
        sprite_bbox = (top, left, top + sh - 1, left + sw - 1)
        sprite_bboxes.append(sprite_bbox)
        upper_patches[patch_idx]["sprite_bbox"] = sprite_bbox

    return grid


def _validate_generated_instance(grid: np.ndarray) -> bool:
    templates, sprites = _parse_scene(grid)
    if len(templates) != len(sprites) or len(templates) not in (2, 3):
        return False
    if len({tpl["color"] for tpl in templates}) != len(templates):
        return False
    if len({spr["color"] for spr in sprites}) != len(sprites):
        return False
    if set(tpl["color"] for tpl in templates) & set(spr["color"] for spr in sprites):
        return False

    try:
        pairings = _solve_pairings(templates, sprites)
    except AssertionError:
        return False

    if not any((~pair["template"]["target_mask"]).any() for pair in pairings):
        return False

    out = main(grid)
    upper_changed = False
    for pair in pairings:
        template = pair["template"]
        sprite = pair["sprite"]
        if template["hole_area"] < 3 or template["hole_area"] > 6:
            return False
        if np.any(out[sprite["coords"][:, 0], sprite["coords"][:, 1]] != 0):
            return False
        r0, c0, r1, c1 = template["target_bbox_global"]
        before = grid[r0 : r1 + 1, c0 : c1 + 1]
        after = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.array_equal(before, after):
            return False
        upper_changed = True

    if not upper_changed:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_instance()
        if grid is None:
            continue
        if _validate_generated_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid complementary negative-hole instance")
