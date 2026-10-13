from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SPRITE_COLORS = (1, 2, 3)
TEMPLATE_COLOR = 8
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _is_frame_component(comp: dict) -> bool:
    if int(comp["color"]) not in SPRITE_COLORS:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border_area = 2 * (h + w) - 4
    coords = comp["coords"]
    if len(coords) != border_area:
        return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _intervals_overlap(a[0], a[2], b[0], b[2]) and _intervals_overlap(a[1], a[3], b[1], b[3])


def _shift_bbox(b: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return b[0] + dr, b[1] + dc, b[2] + dr, b[3] + dc


def _paint_component(grid: np.ndarray, rel_coords: np.ndarray, top: int, left: int, color: int) -> None:
    grid[rel_coords[:, 0] + top, rel_coords[:, 1] + left] = int(color)


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = _same_color_components(np.asarray(grid, dtype=int))
    frames = [comp for comp in comps if _is_frame_component(comp)]
    teals = [comp for comp in comps if int(comp["color"]) == TEMPLATE_COLOR]
    sprites = [comp for comp in comps if int(comp["color"]) in SPRITE_COLORS and not _is_frame_component(comp)]

    assert 3 <= len(frames) <= 5
    assert len(frames) == len(teals) == len(sprites)

    templates: list[dict] = []
    used_teals: set[int] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside: list[tuple[int, dict]] = []
        for idx, teal in enumerate(teals):
            tr0, tc0, tr1, tc1 = teal["bbox"]
            if r0 < tr0 and tr1 < r1 and c0 < tc0 and tc1 < c1:
                inside.append((idx, teal))
        assert len(inside) == 1
        idx, teal = inside[0]
        used_teals.add(idx)
        templates.append(
            {
                "frame_color": int(frame["color"]),
                "frame_bbox": frame["bbox"],
                "shape": _normalize(teal["coords"]),
            }
        )
    assert len(used_teals) == len(teals)

    templates_by_shape = {tpl["shape"]: tpl for tpl in templates}
    assert len(templates_by_shape) == len(templates)

    parsed_sprites: list[dict] = []
    for sprite in sprites:
        shape = _normalize(sprite["coords"])
        assert shape in templates_by_shape
        r0, c0, r1, c1 = sprite["bbox"]
        rel = sprite["coords"] - np.array([r0, c0], dtype=int)
        parsed_sprites.append(
            {
                "color": int(sprite["color"]),
                "bbox": sprite["bbox"],
                "shape": shape,
                "rel_coords": rel,
                "template": templates_by_shape[shape],
            }
        )
    return parsed_sprites, templates


def _movement_plan(sprite: dict) -> dict:
    sr0, sc0, sr1, sc1 = sprite["bbox"]
    tr0, tc0, tr1, tc1 = sprite["template"]["frame_bbox"]
    sprite_h = sr1 - sr0 + 1
    sprite_w = sc1 - sc0 + 1
    sprite_center = ((sr0 + sr1) / 2.0, (sc0 + sc1) / 2.0)
    target_center = ((tr0 + tr1) / 2.0, (tc0 + tc1) / 2.0)
    dr_center = target_center[0] - sprite_center[0]
    dc_center = target_center[1] - sprite_center[1]

    if abs(dc_center) >= abs(dr_center):
        if dc_center >= 0:
            direction = (0, 1)
            target_top = sr0
            target_left = tc0 - sprite_w - 1
        else:
            direction = (0, -1)
            target_top = sr0
            target_left = tc1 + 2
    else:
        if dr_center >= 0:
            direction = (1, 0)
            target_top = tr0 - sprite_h - 1
            target_left = sc0
        else:
            direction = (-1, 0)
            target_top = tr1 + 2
            target_left = sc0

    steps = abs(target_top - sr0) + abs(target_left - sc0)
    assert steps >= 0
    return {
        "bbox": sprite["bbox"],
        "rel_coords": sprite["rel_coords"],
        "color": sprite["color"],
        "direction": direction,
        "target_top": int(target_top),
        "target_left": int(target_left),
        "target_steps": int(steps),
    }


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    sprites, _templates = _parse_scene(g)
    out = g.copy()
    meta: list[dict] = []

    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        rel = sprite["rel_coords"]
        out[rel[:, 0] + r0, rel[:, 1] + c0] = BACKGROUND

    movers = []
    for idx, sprite in enumerate(sprites):
        if sprite["color"] != sprite["template"]["frame_color"]:
            meta.append({"index": idx, "action": "erase"})
            continue
        plan = _movement_plan(sprite)
        plan["index"] = idx
        movers.append(plan)

    movers.sort(
        key=lambda item: (
            item["target_steps"],
            item["target_top"],
            item["target_left"],
            item["bbox"][0],
            item["bbox"][1],
        )
    )

    settled: list[tuple[int, int, int, int]] = []
    for plan in movers:
        current_bbox = plan["bbox"]
        steps_taken = 0
        dr, dc = plan["direction"]
        while steps_taken < plan["target_steps"]:
            next_bbox = _shift_bbox(current_bbox, dr, dc)
            if any(_bboxes_overlap(next_bbox, other_bbox) for other_bbox in settled):
                break
            current_bbox = next_bbox
            steps_taken += 1

        top, left = current_bbox[0], current_bbox[1]
        _paint_component(out, plan["rel_coords"], top, left, plan["color"])
        settled.append(current_bbox)
        meta.append(
            {
                "index": plan["index"],
                "action": "slide",
                "target_steps": plan["target_steps"],
                "actual_steps": steps_taken,
            }
        )

    if return_meta:
        meta.sort(key=lambda item: item["index"])
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite to the template with the same normalized teal shape, but erase it if its color disagrees with the template frame.

    Input structure:
    - A black background contains 3-5 small monochrome 4-connected sprites in blue, red, or green.
    - Separate hollow rectangular frames in the same palette contain one teal shape each on black.
    - Every sprite matches exactly one teal template shape up to translation, and sprites, frames, and teal shapes stay disconnected.

    Rule:
    - For each sprite, find the unique template whose teal component has the same translation-normalized shape.
    - If the sprite color equals that template's frame color, slide the sprite along the dominant cardinal axis between their bbox centers until its bbox is one cell away from the frame bbox.
    - If a farther sliding sprite would run into an already settled nearer sprite on the same axis, stop it at the last non-overlapping position; otherwise, if the colors differ, erase the sprite.
    - Keep every template frame and teal shape unchanged.

    Key insight:
    - Shape identity chooses the destination template, while color agreement separately decides whether that matched sprite moves or disappears.
    """
    return _solve(np.asarray(grid, dtype=int))


def _random_shape_mask() -> np.ndarray:
    for _ in range(400):
        canvas = np.zeros((3, 3), dtype=bool)
        start = (random.randrange(3), random.randrange(3))
        canvas[start] = True
        target_area = random.randint(3, 6)
        while int(canvas.sum()) < target_area:
            frontier = set()
            for r, c in np.argwhere(canvas):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < 3 and 0 <= nc < 3 and not canvas[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            canvas[nr, nc] = True

        coords = np.argwhere(canvas)
        r0, c0, r1, c1 = _bbox(coords)
        cropped = canvas[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] == 1 or cropped.shape[1] == 1:
            continue
        if int(cropped.sum()) == cropped.size:
            continue
        sig = _normalize(np.argwhere(cropped))
        if len(sig) < 3 or len(sig) > 6:
            continue
        if _is_frame_component(
            {
                "color": 1,
                "coords": np.argwhere(cropped),
                "bbox": (0, 0, cropped.shape[0] - 1, cropped.shape[1] - 1),
            }
        ):
            continue
        return cropped
    raise RuntimeError("failed to generate a sprite shape")


def _sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _template_patch(mask: np.ndarray, frame_color: int) -> np.ndarray:
    sh, sw = mask.shape
    patch = np.zeros((sh + 4, sw + 4), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[2 : 2 + sh, 2 : 2 + sw][mask] = TEMPLATE_COLOR
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w][patch != BACKGROUND] = patch[patch != BACKGROUND]


def _build_horizontal_scene() -> tuple[np.ndarray, bool]:
    count = random.randint(3, 5)
    collision_required = random.random() < 0.7

    masks: list[np.ndarray] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    while len(masks) < count:
        mask = _random_shape_mask()
        sig = _normalize(np.argwhere(mask))
        if sig in seen_shapes:
            continue
        seen_shapes.add(sig)
        masks.append(mask)

    sprite_colors = [random.choice(SPRITE_COLORS) for _ in range(count)]
    match_flags = [False] * count
    if collision_required:
        match_flags[0] = True
        match_flags[1] = True
    else:
        match_flags[random.randrange(count)] = True
    mismatch_pool = [idx for idx in range(count) if not match_flags[idx]]
    if mismatch_pool:
        match_flags[random.choice(mismatch_pool)] = False
    if all(match_flags):
        match_flags[random.randrange(2, count)] = False
    if not any(match_flags):
        match_flags[0] = True
    if all(not flag for flag in match_flags):
        match_flags[0] = True
    if all(match_flags):
        match_flags[-1] = False
    if not any(not flag for flag in match_flags):
        match_flags[-1] = False

    frame_colors = []
    for sprite_color, matches in zip(sprite_colors, match_flags):
        if matches:
            frame_colors.append(sprite_color)
        else:
            frame_colors.append(random.choice([color for color in SPRITE_COLORS if color != sprite_color]))

    sprite_patches = [_sprite_patch(mask, color) for mask, color in zip(masks, sprite_colors)]
    template_patches = [_template_patch(mask, color) for mask, color in zip(masks, frame_colors)]

    max_sw = max(patch.shape[1] for patch in sprite_patches)
    max_th = max(patch.shape[0] for patch in template_patches)
    max_tw = max(patch.shape[1] for patch in template_patches)
    template_rows = (count + 1) // 2
    band_h = max_th + 1
    sprite_x = [1, 1 + max_sw + 2]
    left_width = sprite_x[1] + max_sw + 1
    center_gap = random.randint(2, 3)
    template_x0 = left_width + center_gap
    template_x = [template_x0, template_x0 + max_tw + 1]
    height = 1 + template_rows * band_h
    width = template_x[-1] + max_tw + 1
    grid = np.zeros((height, width), dtype=int)
    assert max(grid.shape) <= 30

    all_slots = [(row, col) for row in range(template_rows) for col in range(2)]
    used_slots = set()
    slot_for_idx: dict[int, tuple[int, int]] = {}
    if collision_required:
        slot_for_idx[0] = (0, 0)
        slot_for_idx[1] = (0, 1)
        used_slots.update({(0, 0), (0, 1)})
    remaining_indices = [idx for idx in range(count) if idx not in slot_for_idx]
    remaining_slots = [slot for slot in all_slots if slot not in used_slots]
    random.shuffle(remaining_indices)
    random.shuffle(remaining_slots)
    for idx, slot in zip(remaining_indices, remaining_slots):
        slot_for_idx[idx] = slot

    template_pos: dict[int, tuple[int, int]] = {}
    for idx in range(count):
        row_slot, col_slot = slot_for_idx[idx]
        patch = template_patches[idx]
        band_top = 1 + row_slot * band_h
        top = band_top + random.randint(0, band_h - patch.shape[0])
        left = template_x[col_slot] + random.randint(0, max_tw - patch.shape[1])
        _paste(grid, patch, top, left)
        template_pos[idx] = (top, left)

    for row_slot in range(template_rows):
        idxs = [idx for idx, slot in slot_for_idx.items() if slot[0] == row_slot]
        idxs.sort(key=lambda idx: slot_for_idx[idx][1])
        if not idxs:
            continue
        band_top = 1 + row_slot * band_h
        band_sprite_h = max(sprite_patches[idx].shape[0] for idx in idxs)
        shared_top = band_top + random.randint(0, band_h - band_sprite_h)

        if collision_required and row_slot == 0 and len(idxs) >= 2:
            left_template_idx, right_template_idx = idxs[0], idxs[1]
            sprite_order = [(right_template_idx, 0), (left_template_idx, 1)]
        else:
            slots = [0, 1][: len(idxs)]
            random.shuffle(slots)
            sprite_order = list(zip(idxs, slots))

        for idx, sprite_col_slot in sprite_order:
            patch = sprite_patches[idx]
            top = shared_top
            left = sprite_x[sprite_col_slot]
            _paste(grid, patch, top, left)

    return grid, collision_required


def _valid_generated_scene(grid: np.ndarray, collision_required: bool) -> bool:
    try:
        _sprites, _templates = _parse_scene(grid)
        out, meta = _solve(grid, return_meta=True)
    except Exception:
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if not any(item["action"] == "slide" for item in meta):
        return False
    if not any(item["action"] == "erase" for item in meta):
        return False
    if collision_required:
        if not any(
            item["action"] == "slide" and item["actual_steps"] < item["target_steps"]
            for item in meta
        ):
            return False
    return not np.array_equal(grid, out)


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, collision_required = _build_horizontal_scene()
        if random.random() < 0.5:
            grid = grid.T.copy()
        if _valid_generated_scene(grid, collision_required):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
