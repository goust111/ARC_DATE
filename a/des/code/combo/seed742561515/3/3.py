from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

PALETTE_LENGTH_RANGE = (3, 5)
OBJECT_COUNT_RANGE = (3, 5)
GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 30
OBJECT_MIN_COL = 3

OBJECT_ATTEMPTS = 120
SCENE_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    neigh = _neighbors4 if int(connectivity) == 4 else _neighbors8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neigh(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_true(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    bg = ~mask
    count = 0
    for comp in _components(bg, connectivity=4):
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1)
        )
        if not bool(touches_border):
            count += 1
    return int(count)


def _parse_palette(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in np.flatnonzero(grid[:, 0] != BLACK)]
    assert rows, "palette must occupy the leftmost column"
    colors = [int(grid[r, 0]) for r in rows]
    return rows, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by the number of black holes it encloses.

    The leftmost column contains a top-to-bottom palette with one black gap
    between successive colors. The k-th palette color represents objects with
    exactly k holes. Grey object pixels are recolored, holes stay black, and
    the palette is erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_rows, palette_colors = _parse_palette(g)
    out = g.copy()
    out[palette_rows, 0] = BLACK

    for comp in _components(g == GREY, connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            submask[int(r - r0), int(c - c0)] = True
        holes = _hole_count(submask)
        assert 0 <= holes < len(palette_colors), "hole count must index into the palette"
        new_color = int(palette_colors[holes])
        out[comp[:, 0], comp[:, 1]] = new_color

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _add_boundary_bumps(mask: np.ndarray, count: int) -> None:
    for _ in range(int(count)):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return
        r0, c0, r1, c1 = _bbox(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        side = random.choice(("top", "bottom", "left", "right"))
        depth = random.randint(1, 2)

        if side in ("top", "bottom"):
            span = random.randint(1, max(1, width - 1))
            start = random.randint(c0, c1 - span + 1)
            if side == "top":
                rr0 = max(0, r0 - depth)
                if rr0 < r0:
                    mask[rr0:r0, start : start + span] = True
            else:
                rr1 = min(mask.shape[0], r1 + 1 + depth)
                if rr1 > r1 + 1:
                    mask[r1 + 1 : rr1, start : start + span] = True
        else:
            span = random.randint(1, max(1, height - 1))
            start = random.randint(r0, r1 - span + 1)
            if side == "left":
                cc0 = max(0, c0 - depth)
                if cc0 < c0:
                    mask[start : start + span, cc0:c0] = True
            else:
                cc1 = min(mask.shape[1], c1 + 1 + depth)
                if cc1 > c1 + 1:
                    mask[start : start + span, c1 + 1 : cc1] = True


def _carve_outer_notches(mask: np.ndarray, count: int) -> None:
    for _ in range(int(count)):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return
        r0, c0, r1, c1 = _bbox(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        side = random.choice(("top", "bottom", "left", "right"))

        if side in ("top", "bottom") and width >= 4:
            span = random.randint(1, width - 2)
            start = random.randint(c0 + 1, c1 - span)
            row = r0 if side == "top" else r1
            mask[row, start : start + span] = False
        elif side in ("left", "right") and height >= 4:
            span = random.randint(1, height - 2)
            start = random.randint(r0 + 1, r1 - span)
            col = c0 if side == "left" else c1
            mask[start : start + span, col] = False


def _hole_layout_candidates(hole_count: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for rows in range(1, 4):
        for cols in range(1, 4):
            slots = rows * cols
            if slots >= hole_count and slots <= max(4, hole_count + 2):
                candidates.append((rows, cols))
    return candidates


def _generate_object_mask(hole_count: int) -> np.ndarray:
    for _ in range(OBJECT_ATTEMPTS):
        if hole_count == 0:
            base_h = random.randint(3, 7)
            base_w = random.randint(4, 8)
            canvas = np.zeros((base_h + 8, base_w + 8), dtype=bool)
            anchor = 4
            canvas[anchor : anchor + base_h, anchor : anchor + base_w] = True
            _add_boundary_bumps(canvas, random.randint(1, 4))
            if random.random() < 0.8:
                _carve_outer_notches(canvas, random.randint(0, 2))
            if random.random() < 0.5:
                _add_boundary_bumps(canvas, random.randint(0, 2))
        else:
            rows, cols = random.choice(_hole_layout_candidates(hole_count))
            slot_h = random.randint(2, 3)
            slot_w = random.randint(2, 3)
            base_h = rows * slot_h + rows + 1
            base_w = cols * slot_w + cols + 1
            canvas = np.zeros((base_h + 8, base_w + 8), dtype=bool)
            anchor = 4
            canvas[anchor : anchor + base_h, anchor : anchor + base_w] = True
            _add_boundary_bumps(canvas, random.randint(1, 4))

            slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
            chosen_slots = random.sample(slots, hole_count)
            for slot_r, slot_c in chosen_slots:
                cell_r = anchor + 1 + slot_r * (slot_h + 1)
                cell_c = anchor + 1 + slot_c * (slot_w + 1)
                hole_h = random.randint(1, slot_h)
                hole_w = random.randint(1, slot_w)
                shift_r = random.randint(0, slot_h - hole_h)
                shift_c = random.randint(0, slot_w - hole_w)
                canvas[
                    cell_r + shift_r : cell_r + shift_r + hole_h,
                    cell_c + shift_c : cell_c + shift_c + hole_w,
                ] = False

            if random.random() < 0.6:
                _add_boundary_bumps(canvas, random.randint(0, 2))

        cropped = _crop_true(canvas)
        if cropped is None:
            continue
        cropped = _random_transform(cropped)
        if len(_components(cropped, connectivity=8)) != 1:
            continue
        if _hole_count(cropped) != hole_count:
            continue
        h, w = map(int, cropped.shape)
        area = int(cropped.sum())
        if not (3 <= h <= 13 and 3 <= w <= 13):
            continue
        if not (8 <= area <= 90):
            continue
        return cropped

    raise RuntimeError(f"failed to generate an object with {hole_count} holes")


def _sample_hole_counts(palette_len: int, object_count: int) -> list[int]:
    distinct_target = random.randint(2, min(palette_len, object_count, 4))
    distinct = {0, random.randint(1, palette_len - 1)}
    if palette_len >= 3 and object_count >= 3 and random.random() < 0.65:
        distinct.add(palette_len - 1)
    while len(distinct) < distinct_target:
        distinct.add(random.randint(0, palette_len - 1))

    counts = list(distinct)
    while len(counts) < object_count:
        if random.random() < 0.7:
            counts.append(random.choice(tuple(distinct)))
        else:
            counts.append(random.randint(0, palette_len - 1))
    random.shuffle(counts)
    return counts


def _sample_grid_shape(masks: list[np.ndarray], palette_len: int) -> tuple[int, int] | None:
    max_h = max(int(mask.shape[0]) for mask in masks)
    max_w = max(int(mask.shape[1]) for mask in masks)
    reserve_area = sum((int(mask.shape[0]) + 2) * (int(mask.shape[1]) + 2) for mask in masks)

    min_h = max(GRID_SIDE_MIN, 2 * palette_len + 1, max_h + 2)
    min_w = max(GRID_SIDE_MIN, OBJECT_MIN_COL + max_w + 1)
    candidates: list[tuple[int, int]] = []

    for h in range(min_h, GRID_SIDE_MAX + 1):
        for w in range(min_w, GRID_SIDE_MAX + 1):
            usable_area = h * (w - OBJECT_MIN_COL)
            if usable_area >= reserve_area + 20:
                candidates.append((int(h), int(w)))

    if not candidates:
        return None

    compact = [
        shape
        for shape in candidates
        if shape[0] <= min_h + 6 and shape[1] <= min_w + 8
    ]
    pool = compact if compact and random.random() < 0.7 else candidates
    return random.choice(pool)


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(blocked.shape[0], top + height + 1)
    c1 = min(blocked.shape[1], left + width + 1)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(blocked.shape[0], top + height + 1)
    c1 = min(blocked.shape[1], left + width + 1)
    blocked[r0:r1, c0:c1] = True


def _place_objects(
    grid: np.ndarray,
    objects: list[dict[str, object]],
) -> list[tuple[int, int]] | None:
    h, w = map(int, grid.shape)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, :OBJECT_MIN_COL] = True

    placements: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda idx: -int(np.sum(objects[idx]["mask"])))

    for idx in order:
        mask = objects[idx]["mask"]
        obj_h, obj_w = map(int, mask.shape)
        if obj_h > h or obj_w > w - OBJECT_MIN_COL:
            return None

        row_choices = list(range(1, h - obj_h)) if h - obj_h >= 2 else list(range(0, h - obj_h + 1))
        col_choices = (
            list(range(OBJECT_MIN_COL, w - obj_w))
            if w - obj_w - OBJECT_MIN_COL >= 1
            else list(range(OBJECT_MIN_COL, w - obj_w + 1))
        )
        positions = [(top, left) for top in row_choices for left in col_choices]
        random.shuffle(positions)

        placed = False
        for top, left in positions:
            if not _can_place(blocked, int(top), int(left), obj_h, obj_w):
                continue
            rr, cc = np.where(mask)
            grid[top + rr, left + cc] = GREY
            _mark_blocked(blocked, int(top), int(left), obj_h, obj_w)
            placements[idx] = (int(top), int(left))
            placed = True
            break

        if not placed:
            return None

    return [p for p in placements if p is not None]


def _build_expected(
    inp: np.ndarray,
    palette_rows: list[int],
    palette_colors: list[int],
    objects: list[dict[str, object]],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    out[palette_rows, 0] = BLACK
    for obj, (top, left) in zip(objects, placements, strict=False):
        mask = obj["mask"]
        hole_count = int(obj["holes"])
        rr, cc = np.where(mask)
        out[top + rr, left + cc] = int(palette_colors[hole_count])
    return out


def _generate_scene() -> np.ndarray:
    palette_len = random.randint(*PALETTE_LENGTH_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    hole_counts = _sample_hole_counts(palette_len, object_count)
    palette_colors = random.sample(PALETTE_COLOR_POOL, palette_len)

    objects = [
        {"holes": int(hole_count), "mask": _generate_object_mask(int(hole_count))}
        for hole_count in hole_counts
    ]

    shape = _sample_grid_shape([obj["mask"] for obj in objects], palette_len)
    if shape is None:
        raise RuntimeError("no grid shape fits the sampled objects")

    h, w = shape
    grid = np.zeros((h, w), dtype=int)

    max_palette_start = h - (2 * palette_len - 1)
    palette_start = random.randint(0, max_palette_start)
    palette_rows = [palette_start + 2 * i for i in range(palette_len)]
    for r, color in zip(palette_rows, palette_colors, strict=False):
        grid[int(r), 0] = int(color)

    placements = _place_objects(grid, objects)
    if placements is None:
        raise RuntimeError("failed to place all objects")

    expected = _build_expected(grid, palette_rows, palette_colors, objects, placements)

    assert len(_components(grid == GREY, connectivity=8)) == len(objects)
    assert [int(grid[r, 0]) for r in palette_rows] == palette_colors
    assert np.array_equal(main(grid), expected)

    return grid


def generate_input() -> np.ndarray:
    """
    Generate a random ARC-style scene for the hole-count palette task.
    """
    for _ in range(SCENE_ATTEMPTS):
        try:
            return _generate_scene()
        except RuntimeError:
            continue
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input grid")
