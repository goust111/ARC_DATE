from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
LABEL_COLORS = (1, 2, 3, 4)
FRAME_COLOR = 5
SPRITE_COLOR = 7

MAX_GENERATION_ATTEMPTS = 400
SLOT_SIZES = (9, 10)
FRAME_SIZES = (5, 7)
GAPS = (1, 2)

BASE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 < br0:
        dr = br0 - ar1 - 1
    elif br1 < ar0:
        dr = ar0 - br1 - 1
    else:
        dr = 0
    if ac1 < bc0:
        dc = bc0 - ac1 - 1
    elif bc1 < ac0:
        dc = ac0 - bc1 - 1
    else:
        dc = 0
    return int(dr + dc)


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _shape_variants(shape) -> list[tuple[tuple[int, int], ...]]:
    pts = [(int(r), int(c)) for r, c in shape]
    variants = set()
    for k in range(8):
        out = []
        for r, c in pts:
            if k == 0:
                rr, cc = r, c
            elif k == 1:
                rr, cc = r, -c
            elif k == 2:
                rr, cc = -r, c
            elif k == 3:
                rr, cc = -r, -c
            elif k == 4:
                rr, cc = c, r
            elif k == 5:
                rr, cc = c, -r
            elif k == 6:
                rr, cc = -c, r
            else:
                rr, cc = -c, -r
            out.append((rr, cc))
        variants.add(_normalize(out))
    return sorted(variants)


def _frame_border_coords(bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    coords = []
    for r in range(r0, r1 + 1):
        coords.append((r, c0))
        coords.append((r, c1))
    for c in range(c0 + 1, c1):
        coords.append((r0, c))
        coords.append((r1, c))
    return np.array(sorted(set(coords)), dtype=int)


def _is_frame(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(
        np.all(patch[border] == FRAME_COLOR)
        and np.all(patch[~border] == BACKGROUND)
        and len(coords) == int(border.sum())
    )


def _reflect_into_frame(
    sprite_coords: np.ndarray,
    sprite_bbox: tuple[int, int, int, int],
    frame_bbox: tuple[int, int, int, int],
    axis: str,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    rc = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    out = []
    if axis == "vertical":
        if sprite_bbox[3] < c0:
            anchor = sprite_bbox[3]
            for r, c in sprite_coords:
                out.append((int(r), int(cc + (anchor - c))))
        elif sprite_bbox[1] > c1:
            anchor = sprite_bbox[1]
            for r, c in sprite_coords:
                out.append((int(r), int(cc - (c - anchor))))
        else:
            raise AssertionError("sprite must lie entirely left or right of the frame")
    else:
        if sprite_bbox[2] < r0:
            anchor = sprite_bbox[2]
            for r, c in sprite_coords:
                out.append((int(rc + (anchor - r)), int(c)))
        elif sprite_bbox[0] > r1:
            anchor = sprite_bbox[0]
            for r, c in sprite_coords:
                out.append((int(rc - (r - anchor)), int(c)))
        else:
            raise AssertionError("sprite must lie entirely above or below the frame")

    reflected = np.array(out, dtype=int)
    assert np.all((r0 < reflected[:, 0]) & (reflected[:, 0] < r1))
    assert np.all((c0 < reflected[:, 1]) & (reflected[:, 1] < c1))
    return reflected


def _analyze_input(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 8 and g.shape[1] >= 8
    assert np.all(np.isin(g[:-1, -1], LABEL_COLORS))
    assert np.all(np.isin(g[-1, :-1], LABEL_COLORS))
    assert int(g[-1, -1]) in LABEL_COLORS

    interior = g[:-1, :-1]
    frame_components = _components(interior == FRAME_COLOR)
    sprite_components = _components(interior == SPRITE_COLOR)
    assert 3 <= len(frame_components) <= 4
    assert len(sprite_components) == len(frame_components)

    frames = []
    for coords_local in frame_components:
        coords = coords_local.copy()
        bbox = _bbox(coords)
        assert _is_frame(interior, coords)
        rc = (bbox[0] + bbox[2]) // 2
        cc = (bbox[1] + bbox[3]) // 2
        row_label = int(g[rc, -1])
        col_label = int(g[-1, cc])
        frames.append(
            {
                "coords": coords,
                "bbox": bbox,
                "center": (rc, cc),
                "row_label": row_label,
                "col_label": col_label,
            }
        )

    sprites = []
    for coords_local in sprite_components:
        coords = coords_local.copy()
        bbox = _bbox(coords)
        sprites.append({"coords": coords, "bbox": bbox})

    frames.sort(key=lambda item: (item["center"][0], item["center"][1]))
    assigned = set()
    for frame in frames:
        distances = [_rect_distance(frame["bbox"], sprite["bbox"]) for sprite in sprites]
        best = min(distances)
        indices = [i for i, d in enumerate(distances) if d == best]
        assert len(indices) == 1, "nearest sprite must be unique for every frame"
        idx = indices[0]
        assigned.add(idx)
        sprite = sprites[idx]
        match = frame["row_label"] == frame["col_label"]
        axis = "vertical" if match else "horizontal"
        reflected = _reflect_into_frame(sprite["coords"], sprite["bbox"], frame["bbox"], axis)
        frame["sprite_index"] = idx
        frame["sprite_coords"] = sprite["coords"]
        frame["sprite_bbox"] = sprite["bbox"]
        frame["axis"] = axis
        frame["match"] = match
        frame["reflected_coords"] = reflected

    assert len(assigned) == len(frames), "sprite ownership must be one-to-one"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect each orange sprite into the nearest gray frame, with label equality choosing the axis and recolor.

    Input structure:
    - The last row is a column-label strip and the last column is a row-label strip, both using colors 1-4.
    - The interior contains 3-4 gray hollow rectangular frames on a black background.
    - Each frame has one nearby orange 4-connected sprite outside it, and each sprite is nearest to exactly one frame.

    Rule:
    - For every frame, read the row label at the frame center row from the right edge and the column label at the frame center column from the bottom edge.
    - Assign the unique nearest orange sprite by bbox Manhattan distance.
    - If the two labels match, fold the sprite left-right into the frame with the frame center column as the reference and recolor it to that label color; otherwise fold it up-down into the frame with the frame center row as the reference and keep it orange.
    - Erase the original outside sprites, and keep the gray frames plus both label strips unchanged.

    Key insight:
    - The row/column label comparison controls two coupled properties at once: which reflection axis is used and whether the reflected copy is recolored.
    """
    g = np.asarray(grid, dtype=int)
    frames = _analyze_input(g)
    out = g.copy()
    out[out == SPRITE_COLOR] = BACKGROUND
    for frame in frames:
        color = frame["row_label"] if frame["match"] else SPRITE_COLOR
        rr = frame["reflected_coords"][:, 0]
        cc = frame["reflected_coords"][:, 1]
        out[rr, cc] = int(color)
    return out


def _pick_branches(count: int) -> list[bool]:
    while True:
        branches = [bool(random.getrandbits(1)) for _ in range(count)]
        if any(branches) and not all(branches):
            return branches


def _pick_shape_config(
    same_branch: bool,
    side: str,
    frame_size: int,
    slot_h: int,
    slot_w: int,
    used_shape_ids: set[int],
) -> tuple[int, tuple[tuple[int, int], ...], int] | None:
    configs = []
    capacity = (frame_size - 1) // 2
    for shape_id, shape in enumerate(BASE_SHAPES):
        if shape_id in used_shape_ids:
            continue
        for variant in _shape_variants(shape):
            rows = [r for r, _ in variant]
            cols = [c for _, c in variant]
            h = max(rows) + 1
            w = max(cols) + 1
            for gap in GAPS:
                if same_branch:
                    if w > capacity or h > frame_size - 2:
                        continue
                    if gap + w > slot_w - frame_size:
                        continue
                else:
                    if h > capacity or w > frame_size - 2:
                        continue
                    if gap + h > slot_h - frame_size:
                        continue
                configs.append((shape_id, variant, gap))
    if not configs:
        return None
    return random.choice(configs)


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _make_frame_and_sprite(
    slot: tuple[int, int, int, int],
    same_branch: bool,
    used_shape_ids: set[int],
) -> dict | None:
    slot_r0, slot_c0, slot_r1, slot_c1 = slot
    slot_h = slot_r1 - slot_r0
    slot_w = slot_c1 - slot_c0
    side = random.choice(("left", "right")) if same_branch else random.choice(("top", "bottom"))
    frame_size = random.choice(FRAME_SIZES)
    shape_cfg = _pick_shape_config(same_branch, side, frame_size, slot_h, slot_w, used_shape_ids)
    if shape_cfg is None:
        return None
    shape_id, variant, gap = shape_cfg
    rows = [r for r, _ in variant]
    cols = [c for _, c in variant]
    sh = max(rows) + 1
    sw = max(cols) + 1

    if same_branch:
        r0 = random.randint(slot_r0, slot_r1 - frame_size)
        if side == "left":
            c0_min = slot_c0 + gap + sw
            c0_max = slot_c1 - frame_size
            if c0_min > c0_max:
                return None
            c0 = random.randint(c0_min, c0_max)
        else:
            c0_min = slot_c0
            c0_max = slot_c1 - frame_size - gap - sw
            if c0_min > c0_max:
                return None
            c0 = random.randint(c0_min, c0_max)
        row_start = random.randint(r0 + 1, r0 + frame_size - 1 - sh)
        if side == "left":
            col_start = c0 - gap - sw
        else:
            col_start = c0 + frame_size + gap
    else:
        c0 = random.randint(slot_c0, slot_c1 - frame_size)
        if side == "top":
            r0_min = slot_r0 + gap + sh
            r0_max = slot_r1 - frame_size
            if r0_min > r0_max:
                return None
            r0 = random.randint(r0_min, r0_max)
        else:
            r0_min = slot_r0
            r0_max = slot_r1 - frame_size - gap - sh
            if r0_min > r0_max:
                return None
            r0 = random.randint(r0_min, r0_max)
        col_start = random.randint(c0 + 1, c0 + frame_size - 1 - sw)
        if side == "top":
            row_start = r0 - gap - sh
        else:
            row_start = r0 + frame_size + gap

    sprite_coords = np.array([(row_start + r, col_start + c) for r, c in variant], dtype=int)
    frame_bbox = (r0, c0, r0 + frame_size - 1, c0 + frame_size - 1)
    sprite_bbox = _bbox(sprite_coords)
    reflected = _reflect_into_frame(
        sprite_coords,
        sprite_bbox,
        frame_bbox,
        "vertical" if same_branch else "horizontal",
    )
    used_shape_ids.add(shape_id)
    return {
        "frame_bbox": frame_bbox,
        "sprite_coords": sprite_coords,
        "sprite_bbox": sprite_bbox,
        "reflected_coords": reflected,
        "same_branch": same_branch,
    }


def _random_labels(length: int) -> np.ndarray:
    return np.array([random.choice(LABEL_COLORS) for _ in range(length)], dtype=int)


def _construct_instance() -> np.ndarray | None:
    slot_h = random.choice(SLOT_SIZES)
    slot_w = random.choice(SLOT_SIZES)
    gutter = random.choice((1, 2))
    pad_top = random.choice((0, 1))
    pad_left = random.choice((0, 1))
    h = pad_top + slot_h * 2 + gutter + 1
    w = pad_left + slot_w * 2 + gutter + 1

    grid = np.full((h, w), BACKGROUND, dtype=int)
    slots = [
        (pad_top, pad_left, pad_top + slot_h, pad_left + slot_w),
        (pad_top, pad_left + slot_w + gutter, pad_top + slot_h, pad_left + 2 * slot_w + gutter),
        (pad_top + slot_h + gutter, pad_left, pad_top + 2 * slot_h + gutter, pad_left + slot_w),
        (
            pad_top + slot_h + gutter,
            pad_left + slot_w + gutter,
            pad_top + 2 * slot_h + gutter,
            pad_left + 2 * slot_w + gutter,
        ),
    ]
    count = random.choice((3, 4))
    chosen_slots = random.sample(slots, count)
    branches = _pick_branches(count)

    frame_specs = []
    used_shape_ids: set[int] = set()
    for slot, same_branch in zip(chosen_slots, branches):
        spec = _make_frame_and_sprite(slot, same_branch, used_shape_ids)
        if spec is None:
            return None
        frame_specs.append(spec)

    centers = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in (spec["frame_bbox"] for spec in frame_specs)]
    if len({r for r, _ in centers}) != len(centers) or len({c for _, c in centers}) != len(centers):
        return None

    for spec in frame_specs:
        border = _frame_border_coords(spec["frame_bbox"])
        _paint_coords(grid, border, FRAME_COLOR)
        _paint_coords(grid, spec["sprite_coords"], SPRITE_COLOR)

    row_labels = _random_labels(h - 1)
    col_labels = _random_labels(w - 1)
    for spec in frame_specs:
        r, c = ((spec["frame_bbox"][0] + spec["frame_bbox"][2]) // 2, (spec["frame_bbox"][1] + spec["frame_bbox"][3]) // 2)
        if spec["same_branch"]:
            color = random.choice(LABEL_COLORS)
            row_labels[r] = color
            col_labels[c] = color
        else:
            row_color = random.choice(LABEL_COLORS)
            col_color_choices = [color for color in LABEL_COLORS if color != row_color]
            row_labels[r] = row_color
            col_labels[c] = random.choice(col_color_choices)

    grid[:-1, -1] = row_labels
    grid[-1, :-1] = col_labels
    grid[-1, -1] = random.choice(LABEL_COLORS)
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frames = _analyze_input(grid)
        out = main(grid)
    except Exception:
        return False

    if not np.array_equal(out[:-1, -1], grid[:-1, -1]):
        return False
    if not np.array_equal(out[-1, :-1], grid[-1, :-1]):
        return False
    if int(out[-1, -1]) != int(grid[-1, -1]):
        return False

    same_count = 0
    diff_count = 0
    for frame in frames:
        same_count += int(frame["match"])
        diff_count += int(not frame["match"])
        border = _frame_border_coords(frame["bbox"])
        if not np.all(out[border[:, 0], border[:, 1]] == FRAME_COLOR):
            return False
        if not np.all(out[frame["sprite_coords"][:, 0], frame["sprite_coords"][:, 1]] == BACKGROUND):
            return False
        expected_color = frame["row_label"] if frame["match"] else SPRITE_COLOR
        rr = frame["reflected_coords"][:, 0]
        cc = frame["reflected_coords"][:, 1]
        if not np.all(out[rr, cc] == expected_color):
            return False
        if frame["match"] and expected_color == SPRITE_COLOR:
            return False

    if same_count == 0 or diff_count == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _construct_instance()
        if grid is None:
            continue
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
