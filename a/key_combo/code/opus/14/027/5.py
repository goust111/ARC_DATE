from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GEN_RETRY_LIMIT = 500
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 24
LARGEST_DIM_RANGE = (6, 8)
SMALLER_DIM_MIN = 4
COLOR_POOL = tuple(range(1, 10))


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components_from_mask(mask)) == 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _group_info(grid: np.ndarray, color: int) -> dict:
    mask = grid == int(color)
    comps = _components_from_mask(mask)
    partials: list[np.ndarray] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        partials.append(local)

    max_h = max(partial.shape[0] for partial in partials)
    max_w = max(partial.shape[1] for partial in partials)
    restored = np.zeros((max_h, max_w), dtype=bool)
    for partial in partials:
        h, w = partial.shape
        restored[:h, :w] |= partial

    return {
        "color": int(color),
        "partials": partials,
        "mask": restored,
        "area": int(restored.sum()),
        "visible": int(mask.sum()),
    }


def _stack_groups(groups: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(groups, key=lambda group: group["area"], reverse=True)
    out_h, out_w = ordered[0]["mask"].shape
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for group in ordered:
        mask = group["mask"]
        h, w = mask.shape
        assert h <= out_h and w <= out_w
        region = out[:h, :w]
        region[mask] = int(group["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each color's incomplete sprite by aligned union, then stack the recovered sprites by restored area.

    Input structure:
    - Black is the background and there are exactly three foreground colors.
    - Each foreground color appears as 2-3 separate 4-connected components.
    - Components of the same color are damaged copies of one sprite, cropped by their own bounding boxes and missing complementary pixels.

    Rule:
    - For each color, crop every same-color component to its bounding box.
    - Pad those crops to that color group's maximum observed height and width, aligned at the upper-left corner, and take their non-black union to recover the full sprite.
    - Compute each recovered sprite's area, use the largest recovered sprite's bounding-box size as the output canvas, and paint the recovered sprites at the upper-left corner from largest area to smallest area.
    - Later, smaller recovered sprites overwrite earlier larger ones wherever they have pixels.

    Key insight:
    - The visible fragments do not reveal the ranking directly: only the aligned union within each color group recovers the true sprite, and that recovered area determines the stack order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert colors

    groups = [_group_info(g, color) for color in colors]
    areas = [group["area"] for group in groups]
    assert len(set(areas)) == len(areas), "recovered sprite areas must be distinct"
    largest = max(groups, key=lambda group: group["area"])
    out_h, out_w = largest["mask"].shape
    for group in groups:
        h, w = group["mask"].shape
        assert h <= out_h and w <= out_w, "every recovered sprite must fit the output canvas"

    return _stack_groups(groups, bg)


def _frontier(mask: np.ndarray) -> list[Coord]:
    h, w = mask.shape
    cells = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((int(nr), int(nc)))
    return list(cells)


def _generate_connected_mask(h: int, w: int, target_area: int) -> np.ndarray | None:
    min_area = h + w + 1
    if target_area < min_area or target_area > h * w:
        return None

    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    steps = ["D"] * (h - 1) + ["R"] * (w - 1)
    random.shuffle(steps)
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    # Ensure the bottom edge and right edge each have a non-corner witness pixel.
    mask[h - 1, w - 2] = True
    mask[h - 2, w - 1] = True

    if int(mask.sum()) > target_area:
        return None

    while int(mask.sum()) < target_area:
        frontier = _frontier(mask)
        if not frontier:
            return None
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True

    if (
        mask[0, 0]
        and mask[-1].any()
        and mask[:, -1].any()
        and mask[-1, :-1].any()
        and mask[:-1, -1].any()
        and _is_connected(mask)
    ):
        return mask
    return None


def _choose_full_masks() -> list[np.ndarray] | None:
    largest_h = random.randint(*LARGEST_DIM_RANGE)
    largest_w = random.randint(*LARGEST_DIM_RANGE)
    bbox_area = largest_h * largest_w

    base0 = largest_h + largest_w + 1
    area0_lo = max(base0, int(np.ceil(0.62 * bbox_area)))
    area0_hi = int(np.floor(0.74 * bbox_area))
    area1_lo = max(base0, int(np.ceil(0.46 * bbox_area)))
    area1_hi = int(np.floor(0.58 * bbox_area))
    if area0_lo > area0_hi or area1_lo > min(area1_hi, area0_hi - 2):
        return None

    area0 = random.randint(area0_lo, area0_hi)
    area1 = random.randint(area1_lo, min(area1_hi, area0 - 2))

    dim2_choices = []
    for h2 in range(SMALLER_DIM_MIN, largest_h + 1):
        for w2 in range(SMALLER_DIM_MIN, largest_w + 1):
            if h2 == largest_h and w2 == largest_w:
                continue
            bbox2 = h2 * w2
            area2_lo = max(h2 + w2 + 1, int(np.ceil(0.48 * bbox2)))
            area2_hi = min(int(np.floor(0.68 * bbox2)), area1 - 2)
            if area2_lo <= area2_hi:
                dim2_choices.append((h2, w2, area2_lo, area2_hi))
    if not dim2_choices:
        return None

    h2, w2, area2_lo, area2_hi = random.choice(dim2_choices)
    area2 = random.randint(area2_lo, area2_hi)

    mask0 = _generate_connected_mask(largest_h, largest_w, area0)
    mask1 = _generate_connected_mask(largest_h, largest_w, area1)
    mask2 = _generate_connected_mask(h2, w2, area2)
    if mask0 is None or mask1 is None or mask2 is None:
        return None

    if np.array_equal(mask0, mask1):
        return None
    return [mask0, mask1, mask2]


def _mandatory_variant(full_mask: np.ndarray, variant: str) -> np.ndarray | None:
    mask = full_mask.copy()
    if variant in {"drop_bottom", "drop_both"}:
        mask[-1, :] = False
    if variant in {"drop_right", "drop_both"}:
        mask[:, -1] = False
    if not _valid_partial(mask):
        return None
    return mask


def _valid_partial(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[:, 0].any() and _is_connected(mask))


def _choose_variants(copy_count: int, strong_damage: bool) -> list[str]:
    if strong_damage and copy_count == 2:
        return random.choice([["drop_bottom", "keep"], ["drop_right", "keep"]])
    if copy_count == 2:
        return random.choice(
            [
                ["drop_bottom", "drop_right"],
                ["drop_both", "keep"],
                ["drop_bottom", "keep"],
                ["drop_right", "keep"],
            ]
        )
    return ["drop_bottom", "drop_right", "keep"]


def _target_area_bounds(full_area: int, copy_count: int, strong_damage: bool) -> tuple[int, int]:
    if strong_damage:
        lo = int(np.ceil(0.64 * full_area))
        hi = int(np.floor(0.80 * full_area))
    elif copy_count == 3:
        lo = int(np.ceil(0.82 * full_area))
        hi = int(np.floor(0.94 * full_area))
    else:
        lo = int(np.ceil(0.70 * full_area))
        hi = int(np.floor(0.84 * full_area))
    return lo, hi


def _build_partials(full_mask: np.ndarray, copy_count: int, strong_damage: bool) -> list[np.ndarray] | None:
    full_area = int(full_mask.sum())
    variants = _choose_variants(copy_count, strong_damage)
    partials: list[np.ndarray] = []
    for variant in variants:
        partial = _mandatory_variant(full_mask, variant)
        if partial is None:
            return None
        partials.append(partial)

    coverage = sum(partial.astype(int) for partial in partials)
    if np.any(coverage[full_mask] == 0):
        return None

    bottom_cells = [(full_mask.shape[0] - 1, int(c)) for c in np.flatnonzero(full_mask[-1])]
    right_cells = [(int(r), full_mask.shape[1] - 1) for r in np.flatnonzero(full_mask[:, -1])]

    can_keep_bottom = [i for i, variant in enumerate(variants) if variant not in {"drop_bottom", "drop_both"}]
    can_keep_right = [i for i, variant in enumerate(variants) if variant not in {"drop_right", "drop_both"}]
    if not can_keep_bottom or not can_keep_right:
        return None

    protect: list[set[Coord]] = [set() for _ in partials]
    for group in protect:
        group.add((0, 0))
    if bottom_cells:
        keep_bottom = random.choice(can_keep_bottom)
        protect[keep_bottom].add(random.choice(bottom_cells))
    if right_cells:
        keep_right = random.choice(can_keep_right)
        protect[keep_right].add(random.choice(right_cells))

    low, high = _target_area_bounds(full_area, copy_count, strong_damage)
    targets = []
    for partial in partials:
        current_area = int(partial.sum())
        lo = min(max(low, 1), current_area - 1)
        hi = min(high, current_area - 1)
        if lo > hi:
            return None
        targets.append(random.randint(lo, hi))

    order = list(range(copy_count))
    random.shuffle(order)
    for idx in order:
        mask = partials[idx]
        target = targets[idx]
        attempts = 0
        while int(mask.sum()) > target and attempts < 200:
            attempts += 1
            candidates = [
                tuple(map(int, rc))
                for rc in np.argwhere(mask)
                if tuple(map(int, rc)) not in protect[idx] and coverage[tuple(map(int, rc))] > 1
            ]
            if not candidates:
                break
            random.shuffle(candidates)
            removed = False
            for cell in candidates:
                trial = mask.copy()
                trial[cell] = False
                if not _valid_partial(trial):
                    continue
                mask = trial
                partials[idx] = mask
                coverage[cell] -= 1
                removed = True
                break
            if not removed:
                break
        if int(partials[idx].sum()) > target:
            return None

    union = np.zeros_like(full_mask, dtype=bool)
    for partial in partials:
        union |= partial
    if not np.array_equal(union, full_mask):
        return None

    if any(int(partial.sum()) >= full_area for partial in partials):
        return None
    if len({partial.tobytes() for partial in partials}) != len(partials):
        return None

    observed_shapes = [_crop_mask(partial).shape for partial in partials]
    if max(h for h, _w in observed_shapes) != full_mask.shape[0]:
        return None
    if max(w for _h, w in observed_shapes) != full_mask.shape[1]:
        return None
    return partials


def _render_stack_masks(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    largest_h, largest_w = masks[0].shape
    padded = []
    for mask in masks:
        canvas = np.zeros((largest_h, largest_w), dtype=bool)
        h, w = mask.shape
        canvas[:h, :w] = mask
        padded.append(canvas)
    visible0 = padded[0] & ~padded[1] & ~padded[2]
    visible1 = padded[1] & ~padded[2]
    visible2 = padded[2]
    return visible0, visible1, visible2


def _place_objects(objects: list[dict]) -> np.ndarray | None:
    total_bbox_area = sum(obj["mask"].shape[0] * obj["mask"].shape[1] for obj in objects)
    for _ in range(60):
        height = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
        width = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
        if total_bbox_area > height * width * 0.55:
            continue
        grid = np.zeros((height, width), dtype=int)
        placed_boxes: list[BBox] = []
        ok = True
        for obj in sorted(objects, key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True):
            mask = obj["mask"]
            h, w = mask.shape
            if h > height or w > width:
                ok = False
                break
            placed = False
            for _ in range(250):
                r0 = random.randint(0, height - h)
                c0 = random.randint(0, width - w)
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(
                    not (
                        box[2] + 1 < other[0]
                        or other[2] + 1 < box[0]
                        or box[3] + 1 < other[1]
                        or other[3] + 1 < box[1]
                    )
                    for other in placed_boxes
                ):
                    continue
                region = grid[r0 : r0 + h, c0 : c0 + w]
                region[mask] = int(obj["color"])
                placed_boxes.append(box)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _validate_instance(inp: np.ndarray, masks: list[np.ndarray], colors: list[int], copy_counts: list[int]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(set(copy_counts)) < 2 or 2 not in copy_counts or 3 not in copy_counts:
        return False

    groups = {_group_info(inp, color)["color"]: _group_info(inp, color) for color in colors}
    if set(groups) != set(colors):
        return False

    full_by_color = {color: mask for color, mask in zip(colors, masks)}
    has_bbox_variation = False
    area_order_expected = colors[:]
    for color, expected_mask, copy_count in zip(colors, masks, copy_counts):
        group = groups[color]
        if len(group["partials"]) != copy_count:
            return False
        if not np.array_equal(group["mask"], expected_mask):
            return False
        if group["area"] != int(expected_mask.sum()):
            return False
        shapes = [partial.shape for partial in group["partials"]]
        if len(set(shapes)) > 1:
            has_bbox_variation = True
        if any(np.array_equal(partial, expected_mask) for partial in group["partials"]):
            return False
    if not has_bbox_variation:
        return False

    ordered_by_area = sorted(colors, key=lambda color: groups[color]["area"], reverse=True)
    ordered_by_visible = sorted(colors, key=lambda color: groups[color]["visible"], reverse=True)
    if ordered_by_area != area_order_expected:
        return False
    if ordered_by_visible == ordered_by_area:
        return False
    if groups[colors[0]]["visible"] != min(groups[color]["visible"] for color in colors):
        return False

    largest_shape = full_by_color[colors[0]].shape
    same_bbox_others = [color for color in colors[1:] if full_by_color[color].shape == largest_shape]
    if not same_bbox_others:
        return False

    visible0, visible1, visible2 = _render_stack_masks(masks)
    if not (visible0.any() and visible1.any() and visible2.any()):
        return False

    out = main(inp)
    expected_groups = [groups[color] for color in colors]
    expected_out = _stack_groups(expected_groups, 0)
    if not np.array_equal(out, expected_out):
        return False

    out_colors = set(map(int, np.unique(out))) - {0}
    if out_colors != set(colors):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    masks = _choose_full_masks()
    if masks is None:
        return None

    copy_counts = [2, 3, 3]

    partial_sets: list[list[np.ndarray]] = []
    for idx, (mask, copy_count) in enumerate(zip(masks, copy_counts)):
        partials = _build_partials(mask, copy_count, strong_damage=(idx == 0))
        if partials is None:
            return None
        partial_sets.append(partials)

    colors = random.sample(list(COLOR_POOL), 3)
    objects = []
    for color, partials in zip(colors, partial_sets):
        for partial in partials:
            objects.append({"color": int(color), "mask": _crop_mask(partial)})

    inp = _place_objects(objects)
    if inp is None:
        return None
    if not _validate_instance(inp, masks, colors, copy_counts):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid ARC instance")
