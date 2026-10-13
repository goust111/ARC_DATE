from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
RNG = np.random.default_rng()
BG = 0
MAX_GENERATION_ATTEMPTS = 600
MAX_PATCH_ATTEMPTS = 200
MAX_SCATTER_ATTEMPTS = 120


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    dst_view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        dst_view[:] = patch
    else:
        mask = patch != int(transparent)
        dst_view[mask] = patch[mask]
    return dst


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert int(connectivity) == 4
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    specs = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
        comps = components_mask(g == int(color), connectivity=4)
        specs.append(
            {
                "color": int(color),
                "count": int(len(comps)),
                "coords": coords,
                "patch": patch,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "pixel_count": int(coords.shape[0]),
                "component_areas": sorted(int(comp.shape[0]) for comp in comps),
            }
        )
    return specs


def _overlay_specs(specs: list[dict], *, order_key=None, out_shape: tuple[int, int] | None = None) -> np.ndarray:
    if not specs:
        return np.zeros((1, 1), dtype=int)
    if out_shape is None:
        out_h = max(int(spec["patch"].shape[0]) for spec in specs)
        out_w = max(int(spec["patch"].shape[1]) for spec in specs)
    else:
        out_h, out_w = map(int, out_shape)
    order = list(specs) if order_key is None else sorted(specs, key=order_key)
    out = np.zeros((out_h, out_w), dtype=int)
    for spec in order:
        patch = spec["patch"]
        ph, pw = map(int, patch.shape)
        top = (out_h - ph) // 2
        left = (out_w - pw) // 2
        paste(out, patch, top, left, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge centered color masks using object-frequency order, while skipping the most frequent color.

    Input structure:
    - The grid has a black (0) background.
    - There are 3 or 4 non-black colors, each split into multiple 4-connected monochrome objects.
    - Objects of the same color are spatially separated; each object is a small irregular 3- to 6-cell component.

    Rule:
    - Count the number of 4-connected objects for each non-black color.
    - Skip the color with the largest object count.
    - For every remaining color, crop the union of all its pixels to that color's overall bounding box to obtain a sparse mask.
    - Make a black output whose height and width are the maximum mask height and width among the remaining colors.
    - Center each remaining mask in that common canvas and overlay them from lower frequency to higher frequency, so later masks overwrite earlier ones.

    Key insight:
    - The same frequency ranking simultaneously decides which color disappears and which surviving color is drawn on top at overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    specs = _extract_specs(g)
    assert 3 <= len(specs) <= 4

    counts = [int(spec["count"]) for spec in specs]
    assert len(set(counts)) == len(counts), "object counts must be unique"
    max_count = max(counts)
    skipped = [spec for spec in specs if int(spec["count"]) == int(max_count)]
    assert len(skipped) == 1

    kept = [spec for spec in specs if int(spec["count"]) != int(max_count)]
    return _overlay_specs(kept, order_key=lambda spec: int(spec["count"]))


def _normalize_shape(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    out = coords - mins
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def _shape_variants() -> dict[int, list[np.ndarray]]:
    base_shapes = [
        np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], dtype=int),
        np.array([(0, 0), (0, 1), (0, 2), (1, 1)], dtype=int),
        np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int),
        np.array([(0, 1), (0, 2), (1, 0), (1, 1)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)], dtype=int),
        np.array([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)], dtype=int),
        np.array([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)], dtype=int),
    ]
    grouped: dict[int, list[np.ndarray]] = {}
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in base_shapes:
        rr = base[:, 0].max() + 1
        cc = base[:, 1].max() + 1
        arr = np.zeros((rr, cc), dtype=int)
        arr[base[:, 0], base[:, 1]] = 1
        variants = []
        for flip in (False, True):
            work = np.fliplr(arr) if flip else arr
            for k in range(4):
                rot = np.rot90(work, k)
                coords = np.argwhere(rot == 1)
                norm = _normalize_shape(coords)
                key = tuple(map(tuple, norm.tolist()))
                if key in seen:
                    continue
                seen.add(key)
                variants.append(norm)
        for variant in variants:
            area = int(variant.shape[0])
            grouped.setdefault(area, []).append(variant)
    return grouped


SHAPES_BY_AREA = _shape_variants()


def _candidate_shapes(area_mode: str) -> list[np.ndarray]:
    if area_mode == "large":
        areas = (5, 6, 4)
    elif area_mode == "medium":
        areas = (4, 5, 3)
    else:
        areas = (3, 4, 5)
    return [shape.copy() for area in areas for shape in SHAPES_BY_AREA.get(area, [])]


def _shape_size(shape: np.ndarray) -> tuple[int, int]:
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _mask_with_padding(mask: np.ndarray) -> np.ndarray:
    padded = mask.copy()
    padded[1:, :] |= mask[:-1, :]
    padded[:-1, :] |= mask[1:, :]
    padded[:, 1:] |= mask[:, :-1]
    padded[:, :-1] |= mask[:, 1:]
    return padded


def _placement_valid(blocked: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    h, w = map(int, blocked.shape)
    sh, sw = _shape_size(shape)
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    rr = top + shape[:, 0]
    cc = left + shape[:, 1]
    return not bool(blocked[rr, cc].any())


def _contains_target(shape: np.ndarray, top: int, left: int, target: Coord | None) -> bool:
    if target is None:
        return True
    tr, tc = map(int, target)
    rr = top + shape[:, 0]
    cc = left + shape[:, 1]
    return bool(np.any((rr == tr) & (cc == tc)))


def _anchor_choices(
    occ: np.ndarray,
    shapes: list[np.ndarray],
    mode: str,
    target: Coord | None,
) -> list[tuple[np.ndarray, int, int]]:
    h, w = map(int, occ.shape)
    blocked = _mask_with_padding(occ)
    choices = []
    for shape in shapes:
        sh, sw = _shape_size(shape)
        if mode == "tl":
            positions = [(0, 0)]
        elif mode == "tr":
            positions = [(0, w - sw)]
        elif mode == "bl":
            positions = [(h - sh, 0)]
        elif mode == "br":
            positions = [(h - sh, w - sw)]
        elif mode == "target":
            positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        elif mode == "tl_target":
            positions = [(0, 0)]
        elif mode == "tr_target":
            positions = [(0, w - sw)]
        elif mode == "bl_target":
            positions = [(h - sh, 0)]
        elif mode == "br_target":
            positions = [(h - sh, w - sw)]
        else:
            raise ValueError(f"unknown anchor mode: {mode}")
        for top, left in positions:
            if not _placement_valid(blocked, shape, int(top), int(left)):
                continue
            if "target" in mode and not _contains_target(shape, int(top), int(left), target):
                continue
            choices.append((shape, int(top), int(left)))
    return choices


def _random_choices(occ: np.ndarray, shapes: list[np.ndarray]) -> list[tuple[np.ndarray, int, int]]:
    h, w = map(int, occ.shape)
    blocked = _mask_with_padding(occ)
    choices = []
    for shape in shapes:
        sh, sw = _shape_size(shape)
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                if _placement_valid(blocked, shape, int(top), int(left)):
                    choices.append((shape, int(top), int(left)))
    return choices


def _place_shape(mask: np.ndarray, shape: np.ndarray, top: int, left: int) -> None:
    mask[top + shape[:, 0], left + shape[:, 1]] = True


def _full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    h, w = map(int, mask.shape)
    return (r0, c0, r1, c1) == (0, 0, h - 1, w - 1)


def _generate_patch(height: int, width: int, count: int, color: int, *, target: Coord | None, area_mode: str) -> np.ndarray | None:
    shapes = _candidate_shapes(area_mode)
    opposite_pairs = [("tl", "br"), ("tr", "bl")]
    target_modes = {
        "tl": "tl_target",
        "tr": "tr_target",
        "bl": "bl_target",
        "br": "br_target",
    }
    for _ in range(MAX_PATCH_ATTEMPTS):
        occ = np.zeros((height, width), dtype=bool)
        plan_options: list[list[str]] = []
        pair = opposite_pairs[int(RNG.integers(len(opposite_pairs)))]
        if target is None:
            plan_options.append(list(pair))
        elif count >= 3:
            plan_options.append([pair[0], pair[1], "target"])
        else:
            plan_options.append([target_modes[pair[0]], pair[1]])
            plan_options.append([pair[0], target_modes[pair[1]]])
            plan_options.append(["target", pair[0]])
            plan_options.append(["target", pair[1]])

        placed = 0
        ok = False
        RNG.shuffle(plan_options)
        for plan in plan_options:
            if len(plan) > count:
                continue
            occ[:] = False
            placed = 0
            ok = True
            for mode in plan:
                choices = _anchor_choices(occ, shapes, mode, target)
                if not choices:
                    ok = False
                    break
                shape, top, left = choices[int(RNG.integers(len(choices)))]
                _place_shape(occ, shape, top, left)
                placed += 1
            if ok:
                break
        if not ok:
            continue

        while placed < count:
            choices = _random_choices(occ, shapes)
            if not choices:
                ok = False
                break
            shape, top, left = choices[int(RNG.integers(len(choices)))]
            _place_shape(occ, shape, top, left)
            placed += 1
        if not ok:
            continue
        if target is not None and not bool(occ[target[0], target[1]]):
            continue
        if not _full_bbox(occ):
            continue
        comps = components_mask(occ, connectivity=4)
        if len(comps) != int(count):
            continue
        if not all(3 <= int(comp.shape[0]) <= 6 for comp in comps):
            continue
        patch = np.zeros((height, width), dtype=int)
        patch[occ] = int(color)
        return patch
    return None


def _capacity_ok(height: int, width: int, count: int) -> bool:
    return ((height + 1) // 3) * ((width + 1) // 3) >= int(count)


def _sample_counts(n_colors: int) -> list[int]:
    options = {
        3: [(4, 5, 6)],
        4: [(3, 4, 5, 6)],
    }
    return list(options[int(n_colors)][int(RNG.integers(len(options[int(n_colors)])))])


TALL_AREA_SHAPES = {
    3: [
        np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
        np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
    ],
    4: [
        np.array([(0, 0), (0, 1), (1, 0), (2, 0)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 1), (2, 1)], dtype=int),
        np.array([(0, 0), (1, 0), (1, 1), (2, 0)], dtype=int),
        np.array([(0, 1), (1, 0), (1, 1), (2, 1)], dtype=int),
    ],
    5: [
        np.array([(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)], dtype=int),
    ],
}
WIDE_AREA_SHAPES = {
    area: [shape[:, [1, 0]].copy() for shape in shapes]
    for area, shapes in TALL_AREA_SHAPES.items()
}
FAMILY_SHAPES = {"tall": TALL_AREA_SHAPES, "wide": WIDE_AREA_SHAPES}


def _candidate_layouts(count: int) -> list[dict]:
    layouts = []
    if count <= 6:
        for h in range(7, 11):
            for w in range(8, 11):
                layouts.append({"family": "tall", "size": (h, w), "count": int(count)})
        for h in range(8, 11):
            for w in range(7, 11):
                layouts.append({"family": "wide", "size": (h, w), "count": int(count)})
    RNG.shuffle(layouts)
    return layouts


def _layout_slots(layout: dict) -> dict:
    h, w = map(int, layout["size"])
    family = layout["family"]
    if family == "tall":
        box_h, box_w = 3, 2
        row_starts = [0, h - box_h]
        col_starts = [0, (w - box_w) // 2, w - box_w]
    else:
        box_h, box_w = 2, 3
        row_starts = [0, (h - box_h) // 2, h - box_h]
        col_starts = [0, w - box_w]
    slots = []
    rows_n = len(row_starts)
    cols_n = len(col_starts)
    for rr, top in enumerate(row_starts):
        for cc, left in enumerate(col_starts):
            slots.append(
                {
                    "id": rr * cols_n + cc,
                    "top": int(top),
                    "left": int(left),
                    "bottom": int(top + box_h - 1),
                    "right": int(left + box_w - 1),
                    "row": rr,
                    "col": cc,
                }
            )
    corners = {
        "tl": 0,
        "tr": cols_n - 1,
        "bl": (rows_n - 1) * cols_n,
        "br": rows_n * cols_n - 1,
    }
    return {"slots": slots, "box": (box_h, box_w), "corners": corners, "rows_n": rows_n, "cols_n": cols_n}


def _slot_union_mask(layout: dict) -> np.ndarray:
    info = _layout_slots(layout)
    h, w = map(int, layout["size"])
    mask = np.zeros((h, w), dtype=bool)
    for slot in info["slots"]:
        mask[slot["top"] : slot["bottom"] + 1, slot["left"] : slot["right"] + 1] = True
    return mask


def _center_offset(out_shape: tuple[int, int], inner_shape: tuple[int, int]) -> tuple[int, int]:
    return (int(out_shape[0]) - int(inner_shape[0])) // 2, (int(out_shape[1]) - int(inner_shape[1])) // 2


def _sample_kept_layouts(kept_counts: list[int]) -> list[dict] | None:
    for _ in range(200):
        layouts = []
        used_sizes: set[tuple[int, int]] = set()
        for count in kept_counts:
            choices = [layout for layout in _candidate_layouts(int(count)) if tuple(layout["size"]) not in used_sizes]
            if not choices:
                break
            chosen = choices[int(RNG.integers(len(choices)))]
            layouts.append(chosen)
            used_sizes.add(tuple(chosen["size"]))
        if len(layouts) != len(kept_counts):
            continue
        out_h = max(int(layout["size"][0]) for layout in layouts)
        out_w = max(int(layout["size"][1]) for layout in layouts)
        if 7 <= out_h <= 10 and 7 <= out_w <= 10:
            return layouts
    return None


def _sample_skipped_layout(skip_count: int, used_sizes: set[tuple[int, int]]) -> dict | None:
    choices = [layout for layout in _candidate_layouts(int(skip_count)) if tuple(layout["size"]) not in used_sizes]
    if not choices:
        return None
    return choices[int(RNG.integers(len(choices)))]


def _choose_overlap_cell(layouts: list[dict], out_shape: tuple[int, int]) -> tuple[int, int] | None:
    combined = np.ones(out_shape, dtype=bool)
    for layout in layouts:
        local = _slot_union_mask(layout)
        centered = np.zeros(out_shape, dtype=bool)
        top, left = _center_offset(out_shape, local.shape)
        centered[top : top + local.shape[0], left : left + local.shape[1]] = local
        combined &= centered
    candidates = np.argwhere(combined)
    if candidates.size == 0:
        return None
    weights = np.abs(candidates[:, 0] - out_shape[0] / 2) + np.abs(candidates[:, 1] - out_shape[1] / 2)
    order = np.argsort(weights)
    best = candidates[order[: max(1, min(12, len(order))) ]]
    choice = best[int(RNG.integers(len(best)))]
    return int(choice[0]), int(choice[1])


def _centered_patch(patch: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(out_shape, dtype=int)
    ph, pw = map(int, patch.shape)
    top = (int(out_shape[0]) - ph) // 2
    left = (int(out_shape[1]) - pw) // 2
    paste(out, patch, top, left, transparent=BG)
    return out


def _rect_patch(shape: tuple[int, int], color: int) -> np.ndarray:
    return np.full(shape, int(color), dtype=int)


def _choose_area_modes(kept_counts: list[int], skip_count: int) -> tuple[list[str], str]:
    if len(kept_counts) == 2:
        ranked_modes = ["large", "small"]
    else:
        ranked_modes = ["small", "large", "small"]
    ordered = sorted(range(len(kept_counts)), key=lambda idx: int(kept_counts[idx]))
    assigned = ["small"] * len(kept_counts)
    for rank, idx in enumerate(ordered):
        assigned[idx] = ranked_modes[min(rank, len(ranked_modes) - 1)]
    return assigned, "small"


def _shape_pool(family: str, area_mode: str) -> list[np.ndarray]:
    if area_mode == "large":
        areas = (5,)
    elif area_mode == "medium":
        areas = (4,)
    else:
        areas = (3,)
    pool: list[np.ndarray] = []
    for area in areas:
        pool.extend(shape.copy() for shape in FAMILY_SHAPES[family][area])
    return pool


def _generate_patch(height: int, width: int, count: int, color: int, *, target: Coord | None, area_mode: str, family: str) -> np.ndarray | None:
    layout = {"family": family, "size": (int(height), int(width)), "count": int(count)}
    info = _layout_slots(layout)
    slots = info["slots"]
    corners = info["corners"]
    corner_pairs = [(corners["tl"], corners["br"]), (corners["tr"], corners["bl"])]
    target_slots = []
    if target is not None:
        tr, tc = map(int, target)
        target_slots = [
            int(slot["id"])
            for slot in slots
            if slot["top"] <= tr <= slot["bottom"] and slot["left"] <= tc <= slot["right"]
        ]
        if not target_slots:
            return None
    else:
        target_slots = [-1]

    slot_ids = [int(slot["id"]) for slot in slots]
    shape_pool = _shape_pool(family, area_mode)
    for _ in range(MAX_PATCH_ATTEMPTS):
        target_slot = int(target_slots[int(RNG.integers(len(target_slots)))])
        valid_pairs = [pair for pair in corner_pairs if target_slot not in pair]
        if not valid_pairs:
            valid_pairs = corner_pairs
        corner_a, corner_b = valid_pairs[int(RNG.integers(len(valid_pairs)))]
        chosen = [corner_a, corner_b]
        if target_slot >= 0 and target_slot not in chosen:
            chosen.append(target_slot)
        remaining = [slot_id for slot_id in slot_ids if slot_id not in chosen]
        RNG.shuffle(remaining)
        chosen.extend(remaining[: int(count) - len(chosen)])
        if len(chosen) != int(count):
            continue

        patch = np.zeros((height, width), dtype=int)
        ok = True
        for slot_id in chosen:
            slot = slots[int(slot_id)]
            local_target = None
            if int(slot_id) == target_slot and target is not None:
                local_target = (int(target[0]) - int(slot["top"]), int(target[1]) - int(slot["left"]))
            choices = []
            for shape in shape_pool:
                if local_target is None or any(int(r) == local_target[0] and int(c) == local_target[1] for r, c in shape):
                    choices.append(shape)
            if not choices:
                ok = False
                break
            shape = choices[int(RNG.integers(len(choices)))]
            patch[int(slot["top"]) + shape[:, 0], int(slot["left"]) + shape[:, 1]] = int(color)
        if not ok:
            continue
        comps = components_mask(patch == int(color), connectivity=4)
        if len(comps) != int(count):
            continue
        if not all(3 <= int(comp.shape[0]) <= 6 for comp in comps):
            continue
        return patch
    return None


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    min_side = max(18, max_h + 6, max_w + 6)
    side_options = [(h, w) for h in range(min_side, 25) for w in range(min_side, 25)]
    RNG.shuffle(side_options)
    for h, w in side_options[:40]:
        for _ in range(MAX_SCATTER_ATTEMPTS):
            grid = np.zeros((h, w), dtype=int)
            occ = np.zeros((h, w), dtype=bool)
            ok = True
            order = list(range(len(patches)))
            RNG.shuffle(order)
            for idx in order:
                patch = patches[idx]
                ph, pw = map(int, patch.shape)
                choices = []
                blocked = _mask_with_padding(occ)
                for top in range(1, h - ph):
                    for left in range(1, w - pw):
                        mask = patch != BG
                        if not bool((blocked[top : top + ph, left : left + pw] & mask).any()):
                            choices.append((top, left))
                if not choices:
                    ok = False
                    break
                top, left = choices[int(RNG.integers(len(choices)))]
                paste(grid, patch, int(top), int(left), transparent=BG)
                occ |= grid != BG
            if ok:
                return grid
    return None


def _sample_colors(n_colors: int) -> list[int]:
    for _ in range(40):
        colors = list(map(int, RNG.choice(np.arange(1, 10, dtype=int), size=n_colors, replace=False)))
        if colors[:-1] != sorted(colors[:-1]):
            return colors
    return list(map(int, RNG.choice(np.arange(1, 10, dtype=int), size=n_colors, replace=False)))


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        specs = _extract_specs(grid)
        counts = [int(spec["count"]) for spec in specs]
        if len(set(counts)) != len(counts):
            return False
        bbox_sizes = [(int(spec["patch"].shape[0]), int(spec["patch"].shape[1])) for spec in specs]
        if len(set(bbox_sizes)) != len(bbox_sizes):
            return False
        if not all(3 <= area <= 6 for spec in specs for area in spec["component_areas"]):
            return False

        max_count = max(counts)
        kept = [spec for spec in specs if int(spec["count"]) != int(max_count)]
        skipped = [spec for spec in specs if int(spec["count"]) == int(max_count)]
        if len(skipped) != 1 or len(kept) < 2:
            return False

        expected = _overlay_specs(kept, order_key=lambda spec: int(spec["count"]))
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        if any(not np.any(actual == int(spec["color"])) for spec in kept):
            return False

        centered = [_centered_patch(spec["patch"], expected.shape) for spec in kept]
        overlap_area = 0
        for a, b in itertools.combinations(centered, 2):
            overlap_area += int(np.sum((a != BG) & (b != BG)))
        if overlap_area == 0:
            return False
        if overlap_area > max(8, int(np.sum(actual != BG) * 0.5)):
            return False

        reverse_out = _overlay_specs(kept, order_key=lambda spec: -int(spec["count"]))
        if np.array_equal(actual, reverse_out):
            return False

        pixel_sorted = _overlay_specs(kept, order_key=lambda spec: (int(spec["pixel_count"]), int(spec["color"])))
        if np.array_equal(actual, pixel_sorted):
            return False

        color_sorted = _overlay_specs(kept, order_key=lambda spec: int(spec["color"]))
        if np.array_equal(actual, color_sorted):
            return False

        rect_specs = [{**spec, "patch": _rect_patch(spec["patch"].shape, spec["color"])} for spec in kept]
        rect_out = _overlay_specs(rect_specs, order_key=lambda spec: int(spec["count"]))
        if np.array_equal(actual, rect_out):
            return False

        include_all = _overlay_specs(specs, order_key=lambda spec: int(spec["count"]))
        if np.array_equal(actual, include_all):
            return False

        count_order = [int(spec["color"]) for spec in sorted(specs, key=lambda spec: int(spec["count"]))]
        pixel_order = [int(spec["color"]) for spec in sorted(specs, key=lambda spec: int(spec["pixel_count"]))]
        if count_order == pixel_order:
            return False

        top_by_count = max(specs, key=lambda spec: int(spec["count"]))
        top_by_pixels = max(specs, key=lambda spec: int(spec["pixel_count"]))
        if int(top_by_count["color"]) == int(top_by_pixels["color"]):
            return False

        return True
    except Exception:
        return False


def _build_example() -> np.ndarray | None:
    n_colors = int(RNG.choice(np.array([3, 4], dtype=int)))
    counts = _sample_counts(n_colors)
    colors = _sample_colors(n_colors)

    kept_counts = counts[:-1]
    skip_count = counts[-1]
    kept_layouts = _sample_kept_layouts(kept_counts)
    if kept_layouts is None:
        return None
    skip_layout = _sample_skipped_layout(skip_count, {tuple(layout["size"]) for layout in kept_layouts})
    if skip_layout is None:
        return None

    out_h = max(int(layout["size"][0]) for layout in kept_layouts)
    out_w = max(int(layout["size"][1]) for layout in kept_layouts)
    overlap_cell = _choose_overlap_cell(kept_layouts, (out_h, out_w))
    if overlap_cell is None:
        return None
    area_modes, skip_area_mode = _choose_area_modes(kept_counts, skip_count)

    specs: list[dict] = []
    for color, count, layout, area_mode in zip(colors[:-1], kept_counts, kept_layouts, area_modes):
        h, w = map(int, layout["size"])
        top, left = _center_offset((out_h, out_w), (h, w))
        local_target = (int(overlap_cell[0]) - top, int(overlap_cell[1]) - left)
        patch = _generate_patch(h, w, int(count), int(color), target=local_target, area_mode=area_mode, family=layout["family"])
        if patch is None:
            return None
        specs.append({"color": int(color), "count": int(count), "patch": patch})

    skip_color = int(colors[-1])
    sh, sw = map(int, skip_layout["size"])
    skip_target = None
    if sh <= out_h and sw <= out_w:
        top, left = _center_offset((out_h, out_w), (sh, sw))
        local_try = (int(overlap_cell[0]) - top, int(overlap_cell[1]) - left)
        if 0 <= local_try[0] < sh and 0 <= local_try[1] < sw and _slot_union_mask(skip_layout)[local_try]:
            skip_target = local_try
    skip_patch = _generate_patch(
        sh,
        sw,
        int(skip_count),
        skip_color,
        target=skip_target,
        area_mode=skip_area_mode,
        family=skip_layout["family"],
    )
    if skip_patch is None:
        return None
    specs.append({"color": skip_color, "count": int(skip_count), "patch": skip_patch})

    patches = [spec["patch"] for spec in specs]
    grid = _scatter_patches(patches)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frequency-overlay ARC input")
