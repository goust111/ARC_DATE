from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
PALETTE = (1, 2, 3, 6, 7)
SLOT = 6
SLOT_GAP = 1
LAYER_GAP = 1

GEN_MAX_ATTEMPTS = 400
GEN_COUNT_CHOICES = (4, 5)
GEN_MIN_SIDE_BY_COUNT = {4: 22, 5: 25}
GEN_MAX_SIDE = 30
GEN_OBJECT_GAP = 3
GEN_BORDER = 1


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _find_objects(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != BLACK)
    assert comps, "expected at least one foreground object"
    objects = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        colors = set(int(v) for v in patch.flatten() if int(v) != BLACK)
        assert YELLOW in colors, "every object must contain a yellow axis"
        fg_colors = colors - {YELLOW}
        assert len(fg_colors) == 1, "each object must use exactly one non-yellow foreground color"
        objects.append(
            {
                "bbox": bbox,
                "patch": patch,
                "fg": int(next(iter(fg_colors))),
            }
        )
    return objects


def _detect_axis(patch: np.ndarray) -> tuple[str, int]:
    h, w = patch.shape
    row_hits = [r for r in range(h) if np.all(patch[r, :] == YELLOW)]
    col_hits = [c for c in range(w) if np.all(patch[:, c] == YELLOW)]
    if row_hits:
        assert len(row_hits) == 1 and not col_hits, "horizontal axis must be unique"
        return "H", int(row_hits[0])
    assert len(col_hits) == 1, "vertical axis must be unique"
    return "V", int(col_hits[0])


def _control_side(patch: np.ndarray, orient: str, axis: int) -> int:
    yellow_coords = np.argwhere(patch == YELLOW)
    signs = set()
    if orient == "H":
        for r, _c in yellow_coords:
            if int(r) != axis:
                signs.add(-1 if int(r) < axis else 1)
    else:
        for _r, c in yellow_coords:
            if int(c) != axis:
                signs.add(-1 if int(c) < axis else 1)
    assert len(signs) == 1, "off-axis yellow controls must lie on exactly one side"
    return int(next(iter(signs)))


def _complete_patch(patch: np.ndarray, orient: str, axis: int) -> np.ndarray:
    out = patch.copy()
    source_side = _control_side(patch, orient, axis)
    h, w = patch.shape

    if orient == "H":
        rows = range(axis) if source_side < 0 else range(axis + 1, h)
        for src_r in rows:
            dst_r = 2 * axis - src_r
            assert 0 <= dst_r < h, "horizontal reflection must stay inside the bbox"
            out[dst_r, :] = patch[src_r, :]
    else:
        cols = range(axis) if source_side < 0 else range(axis + 1, w)
        for src_c in cols:
            dst_c = 2 * axis - src_c
            assert 0 <= dst_c < w, "vertical reflection must stay inside the bbox"
            out[:, dst_c] = patch[:, src_c]
    return out


def _sprite_from_completed(completed: np.ndarray, orient: str, axis: int, fg: int) -> np.ndarray:
    sprite = completed.copy()
    if orient == "H":
        axis_mask = sprite[axis, :] == YELLOW
        sprite[axis, axis_mask] = fg
    else:
        axis_mask = sprite[:, axis] == YELLOW
        sprite[axis_mask, axis] = fg
    return sprite


def _place_in_slots(sprites: list[np.ndarray]) -> np.ndarray:
    if not sprites:
        return np.zeros((SLOT, 0), dtype=int)
    width = len(sprites) * SLOT + (len(sprites) - 1) * SLOT_GAP
    layer = np.zeros((SLOT, width), dtype=int)
    for i, sprite in enumerate(sprites):
        h, w = sprite.shape
        assert h <= SLOT and w <= SLOT, "each cropped sprite must fit inside a 6x6 slot"
        c0 = i * (SLOT + SLOT_GAP)
        layer[:h, c0 : c0 + w] = sprite
    return layer


def _build_atlas(entries: list[dict]) -> np.ndarray:
    h_group = sorted((e for e in entries if e["orient"] == "H"), key=lambda e: e["origin"])
    v_group = sorted((e for e in entries if e["orient"] == "V"), key=lambda e: e["origin"])

    h_layer = _place_in_slots([e["sprite"] for e in h_group])
    v_layer = _place_in_slots([e["sprite"] for e in v_group])

    if h_group and v_group:
        out = np.zeros((SLOT + LAYER_GAP + SLOT, max(h_layer.shape[1], v_layer.shape[1])), dtype=int)
        out[:SLOT, : h_layer.shape[1]] = h_layer
        out[SLOT + LAYER_GAP :, : v_layer.shape[1]] = v_layer
        return out
    return h_layer if h_group else v_layer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every yellow-axis object by symmetry and pack the resulting sprites into axis-based atlas layers.

    Input structure:
    - The background is black.
    - Each separated object contains one non-yellow foreground color plus yellow pixels.
    - The yellow pixels form one full horizontal or vertical axis and one or more
      off-axis yellow control marks on exactly one side of that axis.

    Rule:
    - For each object, detect whether the yellow axis is horizontal or vertical.
    - Use the side containing the off-axis yellow controls as the trusted source and
      mirror that entire rectangular half across the axis, copying both foreground
      and background cells onto the opposite half.
    - Crop the completed object to its bbox, replace only the yellow axis cells with
      the object's foreground color, then place horizontal-axis sprites in the top
      atlas layer and vertical-axis sprites in the bottom layer.
    - Within each layer, order sprites by the original input bbox top-left position
      and paste each crop into a left-aligned 6x6 slot with one-cell gaps.

    Key insight:
    - The same yellow axis that determines how an object is completed also determines
      where that completed sprite belongs in the final packed output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    entries = []
    for obj in _find_objects(g):
        orient, axis = _detect_axis(obj["patch"])
        completed = _complete_patch(obj["patch"], orient, axis)
        sprite = _sprite_from_completed(completed, orient, axis, obj["fg"])
        entries.append(
            {
                "origin": (int(obj["bbox"][0]), int(obj["bbox"][1])),
                "orient": orient,
                "sprite": sprite,
            }
        )

    atlas = _build_atlas(entries)
    assert atlas.ndim == 2 and max(atlas.shape) <= 30
    return atlas


def _grow_connected_mask(shape: tuple[int, int], required: list[tuple[int, int]], area: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    frontier = set()
    for cell in required:
        r, c = cell
        mask[r, c] = True
    while int(mask.sum()) < area:
        frontier.clear()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        pick = random.choice(tuple(frontier))
        mask[pick] = True
    return mask


def _random_source_mask(shape: tuple[int, int], attach_side: str) -> np.ndarray:
    h, w = shape
    total = h * w
    low = max(2, int(np.ceil(total * 0.40)))
    high = max(low, int(np.ceil(total * 0.75)))
    if attach_side == "top":
        required = [(0, random.randrange(w))]
    elif attach_side == "bottom":
        required = [(h - 1, random.randrange(w))]
    elif attach_side == "left":
        required = [(random.randrange(h), 0)]
    else:
        required = [(random.randrange(h), w - 1)]
    area = random.randint(low, min(high, total))
    return _grow_connected_mask(shape, required, area)


def _target_mask(shape: tuple[int, int], orient: str) -> np.ndarray:
    h, w = shape
    if orient == "H":
        anchor = random.randrange(w)
        required = [(0, anchor), (h - 1, anchor)]
    else:
        anchor = random.randrange(h)
        required = [(anchor, 0), (anchor, w - 1)]
    total = h * w
    area = random.randint(len(required), total)
    return _grow_connected_mask(shape, required, area)


def _add_controls(patch: np.ndarray, orient: str, source_side: int) -> None:
    if orient == "H":
        row = 1 if source_side < 0 else 3
        cols = list(range(patch.shape[1]))
        random.shuffle(cols)
        n_controls = 1 if random.random() < 0.65 else min(2, len(cols))
        for c in cols[:n_controls]:
            patch[row, c] = YELLOW
    else:
        col = 1 if source_side < 0 else 3
        rows = list(range(patch.shape[0]))
        random.shuffle(rows)
        n_controls = 1 if random.random() < 0.65 else min(2, len(rows))
        for r in rows[:n_controls]:
            patch[r, col] = YELLOW


def _make_horizontal_object(fg: int) -> dict:
    width = random.randint(4, 6)
    source_side = random.choice((-1, 1))
    source_mask = _random_source_mask((2, width), "bottom" if source_side < 0 else "top")

    full = np.zeros((5, width), dtype=int)
    full[2, :] = YELLOW
    if source_side < 0:
        top = full[:2, :]
        top[source_mask] = fg
        _add_controls(full, "H", source_side)
        full[3:, :] = full[:2, :][::-1, :]
        target_mask = _target_mask((2, width), "H")
        full[3:, :] = np.where(target_mask, full[3:, :] * 0 + fg, BLACK)
    else:
        bottom = full[3:, :]
        bottom[source_mask] = fg
        _add_controls(full, "H", source_side)
        full[:2, :] = full[3:, :][::-1, :]
        target_mask = _target_mask((2, width), "H")
        full[:2, :] = np.where(target_mask, full[:2, :] * 0 + fg, BLACK)

    completed = _complete_patch(full, "H", 2)
    sprite = _sprite_from_completed(completed, "H", 2, fg)
    return {"input_patch": full, "completed": completed, "sprite": sprite, "orient": "H"}


def _make_vertical_object(fg: int) -> dict:
    height = random.randint(4, 6)
    source_side = random.choice((-1, 1))
    source_mask = _random_source_mask((height, 2), "right" if source_side < 0 else "left")

    full = np.zeros((height, 5), dtype=int)
    full[:, 2] = YELLOW
    if source_side < 0:
        left = full[:, :2]
        left[source_mask] = fg
        _add_controls(full, "V", source_side)
        full[:, 3:] = full[:, :2][:, ::-1]
        target_mask = _target_mask((height, 2), "V")
        full[:, 3:] = np.where(target_mask, full[:, 3:] * 0 + fg, BLACK)
    else:
        right = full[:, 3:]
        right[source_mask] = fg
        _add_controls(full, "V", source_side)
        full[:, :2] = full[:, 3:][:, ::-1]
        target_mask = _target_mask((height, 2), "V")
        full[:, :2] = np.where(target_mask, full[:, :2] * 0 + fg, BLACK)

    completed = _complete_patch(full, "V", 2)
    sprite = _sprite_from_completed(completed, "V", 2, fg)
    return {"input_patch": full, "completed": completed, "sprite": sprite, "orient": "V"}


def _separated(existing: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, r1, c1 = bbox
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _place_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    placed: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(objects)), key=lambda i: objects[i]["input_patch"].shape[0] * objects[i]["input_patch"].shape[1], reverse=True)
    for idx in order:
        patch = objects[idx]["input_patch"]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(GEN_BORDER, h - ph - GEN_BORDER + 1):
            for c0 in range(GEN_BORDER, w - pw - GEN_BORDER + 1):
                bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _separated(boxes, bbox, GEN_OBJECT_GAP):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
        placed.append(
            {
                **objects[idx],
                "origin": (r0, c0),
            }
        )
    return placed


def _render_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in placed:
        r0, c0 = obj["origin"]
        patch = obj["input_patch"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return grid


def _expected_output(placed: list[dict]) -> np.ndarray:
    entries = []
    for obj in placed:
        entries.append(
            {
                "origin": obj["origin"],
                "orient": obj["orient"],
                "sprite": obj["sprite"],
            }
        )
    return _build_atlas(entries)


def _sample_objects() -> list[dict]:
    count = random.choice(GEN_COUNT_CHOICES)
    colors = random.sample(PALETTE, count)
    h_count = random.randint(1, count - 1)
    orientations = ["H"] * h_count + ["V"] * (count - h_count)
    random.shuffle(orientations)

    objects = []
    for orient, fg in zip(orientations, colors):
        if orient == "H":
            objects.append(_make_horizontal_object(fg))
        else:
            objects.append(_make_vertical_object(fg))
    return objects


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        objects = _sample_objects()
        count = len(objects)
        h = random.randint(GEN_MIN_SIDE_BY_COUNT[count], GEN_MAX_SIDE)
        w = random.randint(GEN_MIN_SIDE_BY_COUNT[count], GEN_MAX_SIDE)
        placed = _place_objects(objects, (h, w))
        if placed is None:
            continue
        inp = _render_input((h, w), placed)
        expected = _expected_output(placed)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle-6 input")
