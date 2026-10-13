from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
TEMPLATE = 6
OBJECT_COLORS = (1, 2, 3, 4)
ROTATION_TO_BORDER = {
    0: "keep",
    1: "object",
    2: "erase",
    3: "separator",
}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
            out.append(np.array(sorted(coords), dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    return (int(mask.shape[0]), int(mask.shape[1])), tuple(sorted((int(r), int(c)) for r, c in coords))


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _hole_border(panel: np.ndarray, coords: np.ndarray, template_color: int) -> np.ndarray:
    border: set[tuple[int, int]] = set()
    h, w = panel.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(panel[nr, nc]) == template_color:
                border.add((nr, nc))
    return np.array(sorted(border), dtype=int)


def _object_components(panel: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components(panel != BACKGROUND):
        colors = {int(panel[r, c]) for r, c in coords}
        assert len(colors) == 1, "right-panel objects must be monochromatic"
        color = next(iter(colors))
        assert color in OBJECT_COLORS, "unexpected object color"
        objects.append({"coords": coords, "color": int(color), "mask": _tight_mask(coords)})
    return objects


def _shape_catalog() -> list[np.ndarray]:
    classes: dict[tuple, np.ndarray] = {}
    for bits in range(1, 1 << 9):
        area = bits.bit_count()
        if area < 3 or area > 5:
            continue
        raw = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if not raw.any():
            continue
        coords = np.argwhere(raw)
        mask = _tight_mask(coords)
        if len(_components(mask)) != 1:
            continue
        if int(mask.sum()) == mask.size:
            continue
        rot_keys = [_mask_key(np.rot90(mask, k)) for k in range(4)]
        if len(set(rot_keys)) != 4:
            continue
        class_key = min(rot_keys)
        if class_key not in classes:
            classes[class_key] = mask
    return list(classes.values())


SHAPE_CATALOG = _shape_catalog()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match rotated right-panel objects to left-panel holes, then fill each hole and alter its border according to the required rotation.

    Input structure:
    - A full-height grey separator column splits the grid into equal-width left and right panels.
    - The left panel contains a single connected magenta template on black background with 3 or 4 enclosed black hole-regions.
    - The right panel contains one separated monochromatic object per hole, each object being a rotated copy of one hole shape.

    Rule:
    - Extract every enclosed hole from the left panel and every monochromatic object from the right panel.
    - For each hole, find the unique object and the unique rotation k in {0,1,2,3} such that rotating the object mask by `np.rot90(mask, k)` matches the hole mask.
    - Fill the hole with the matched object's color, then recolor the hole's magenta 4-neighbor border by rotation: 0 keeps magenta, 1 uses the object color, 2 erases to black, and 3 recolors to grey.
    - Return only the transformed left panel.

    Key insight:
    - The rotation needed for shape matching is not just an alignment aid; it is the hidden control signal that determines how the matched hole's border must be treated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    assert len(sep_cols) == 1, "expected exactly one full-height separator column"
    sep = sep_cols[0]
    assert sep * 2 + 1 == g.shape[1], "panels must have equal width"

    left = g[:, :sep].copy()
    right = g[:, sep + 1 :].copy()

    left_colors = set(map(int, np.unique(left))) - {BACKGROUND}
    assert len(left_colors) == 1, "left panel should contain one non-background template color"
    template_color = next(iter(left_colors))

    hole_records = []
    for coords in _components(left == BACKGROUND):
        if _touches_border(coords, left.shape):
            continue
        hole_records.append(
            {
                "coords": coords,
                "mask": _tight_mask(coords),
                "border": _hole_border(left, coords, template_color),
            }
        )
    assert 3 <= len(hole_records) <= 4, "expected 3 or 4 enclosed holes"

    objects = _object_components(right)
    assert len(objects) == len(hole_records), "object count must equal hole count"

    unmatched = list(range(len(objects)))
    out = left.copy()
    for hole in hole_records:
        candidates: list[tuple[int, int]] = []
        for obj_idx in unmatched:
            obj_mask = objects[obj_idx]["mask"]
            for k in range(4):
                if _mask_key(np.rot90(obj_mask, k)) == _mask_key(hole["mask"]):
                    candidates.append((obj_idx, k))
        assert len(candidates) == 1, "each hole must have one unique rotated object match"

        obj_idx, rotation_k = candidates[0]
        unmatched.remove(obj_idx)
        color = objects[obj_idx]["color"]
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = color

        border_coords = hole["border"]
        mode = ROTATION_TO_BORDER[rotation_k]
        if mode == "object":
            out[border_coords[:, 0], border_coords[:, 1]] = color
        elif mode == "erase":
            out[border_coords[:, 0], border_coords[:, 1]] = BACKGROUND
        elif mode == "separator":
            out[border_coords[:, 0], border_coords[:, 1]] = SEPARATOR

    assert not unmatched, "every right-panel object must be used"
    return out


def _expand4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _apply_generated_rule(
    left: np.ndarray,
    placements: list[dict],
) -> np.ndarray:
    out = left.copy()
    for record in placements:
        coords = record["coords"]
        color = int(record["color"])
        rotation_k = int(record["rotation"])
        out[coords[:, 0], coords[:, 1]] = color
        border = record["border"]
        if rotation_k == 1:
            out[border[:, 0], border[:, 1]] = color
        elif rotation_k == 2:
            out[border[:, 0], border[:, 1]] = BACKGROUND
        elif rotation_k == 3:
            out[border[:, 0], border[:, 1]] = SEPARATOR
    return out


def _template_origin(panel_h: int, panel_w: int) -> tuple[int, int]:
    return random.randint(1, panel_h - 13), random.randint(1, panel_w - 13)


def _choose_holes() -> list[np.ndarray]:
    n_holes = 4 if random.random() < 0.75 else 3
    return [mask.copy() for mask in random.sample(SHAPE_CATALOG, n_holes)]


def _rotation_choices(n_holes: int) -> list[int]:
    if n_holes == 4:
        rotations = [0, 1, 2, 3]
    else:
        rotations = random.sample([0, 1, 2, 3], 3)
    random.shuffle(rotations)
    return rotations


def _hole_slots(mask: np.ndarray, slot_row: str, slot_col: str) -> tuple[int, int]:
    h, w = mask.shape
    if slot_row == "top":
        row_choices = [2, 3] if h == 2 else [2]
    else:
        row_choices = [7, 8] if h == 2 else [7]
    if slot_col == "left":
        col_choices = [2, 3] if w == 2 else [2]
    else:
        col_choices = [7, 8] if w == 2 else [7]
    return random.choice(row_choices), random.choice(col_choices)


def _build_left_panel(panel_h: int, panel_w: int, masks: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    left = np.zeros((panel_h, panel_w), dtype=int)
    top, left0 = _template_origin(panel_h, panel_w)
    left[top : top + 12, left0 : left0 + 12] = TEMPLATE

    slot_names = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    random.shuffle(slot_names)

    placements: list[dict] = []
    for mask, (slot_row, slot_col) in zip(masks, slot_names):
        r0, c0 = _hole_slots(mask, slot_row, slot_col)
        hole_coords = np.argwhere(mask) + np.array([top + r0, left0 + c0])
        left[hole_coords[:, 0], hole_coords[:, 1]] = BACKGROUND
        placements.append({"coords": hole_coords})

    template_crop = left[:, :]
    hole_comps = [
        coords
        for coords in _components(template_crop == BACKGROUND)
        if not _touches_border(coords, left.shape)
    ]
    assert len(hole_comps) == len(masks), "generated holes must stay separated"

    key_to_border = {}
    for coords in hole_comps:
        key_to_border[_mask_key(_tight_mask(coords))] = (coords, _hole_border(left, coords, TEMPLATE))

    for placement in placements:
        coords = placement["coords"]
        key = _mask_key(_tight_mask(coords))
        matched_coords, border = key_to_border[key]
        placement["coords"] = matched_coords
        placement["border"] = border

    return left, placements


def _candidate_object_positions(
    panel_shape: tuple[int, int],
    mask: np.ndarray,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    panel_h, panel_w = panel_shape
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    coords = np.argwhere(mask)
    for r0 in range(1, panel_h - h):
        for c0 in range(1, panel_w - w):
            cells = coords + np.array([r0, c0])
            ok = True
            for rr_raw, cc_raw in cells:
                rr = int(rr_raw)
                cc = int(cc_raw)
                for nr in range(rr - 1, rr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if 0 <= nr < panel_h and 0 <= nc < panel_w and occupied[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    return candidates


def _build_right_panel(
    panel_h: int,
    panel_w: int,
    masks: list[np.ndarray],
    rotations: list[int],
    colors: list[int],
) -> np.ndarray:
    right = np.zeros((panel_h, panel_w), dtype=int)
    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    order = list(range(len(masks)))
    random.shuffle(order)

    for idx in order:
        color = colors[idx]
        obj_mask = np.rot90(masks[idx], -rotations[idx])
        candidates = _candidate_object_positions((panel_h, panel_w), obj_mask, occupied)
        if not candidates:
            raise ValueError("no room for right-panel object")
        r0, c0 = random.choice(candidates)
        coords = np.argwhere(obj_mask) + np.array([r0, c0])
        right[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True
    return right


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.choice([13, 14])
    panel_w = random.choice([13, 14])

    masks = _choose_holes()
    rotations = _rotation_choices(len(masks))
    colors = random.sample(list(OBJECT_COLORS), len(masks))

    try:
        left, placements = _build_left_panel(panel_h, panel_w, masks)
        right = _build_right_panel(panel_h, panel_w, masks, rotations, colors)
    except (AssertionError, ValueError):
        return None

    for placement, color, rotation in zip(placements, colors, rotations):
        placement["color"] = int(color)
        placement["rotation"] = int(rotation)

    expected = _apply_generated_rule(left, placements)
    grid = np.concatenate(
        [left, np.full((panel_h, 1), SEPARATOR, dtype=int), right],
        axis=1,
    )

    try:
        observed = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
