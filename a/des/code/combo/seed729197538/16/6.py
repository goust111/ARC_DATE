from __future__ import annotations

import random

import numpy as np


DIR_BY_HOLES = {
    1: (0, -1),
    2: (-1, 0),
    3: (1, 0),
    4: (0, 1),
}

AREA_SHAPES = {
    1: [(1, 1)],
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}

ALL_COLORS = list(range(10))
OBJECT_COUNT_OPTIONS = (2, 3, 4)
INPUT_MARGIN = 2
SLOT_SIZE = 3
GEN_ATTEMPTS = 400
GRID_RANGES_BY_COUNT = {
    2: ((26, 33), (26, 33)),
    3: ((30, 37), (30, 37)),
    4: ((34, 41), (34, 41)),
}


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _corner_markers(grid: np.ndarray) -> tuple[set[int], np.ndarray]:
    h, w = grid.shape
    assert h >= 4 and w >= 4

    blocks = [
        grid[:2, :2],
        grid[:2, w - 2 : w],
        grid[h - 2 : h, :2],
        grid[h - 2 : h, w - 2 : w],
    ]
    colors: list[int] = []
    marker_mask = np.zeros_like(grid, dtype=bool)
    marker_mask[:2, :2] = True
    marker_mask[:2, w - 2 : w] = True
    marker_mask[h - 2 : h, :2] = True
    marker_mask[h - 2 : h, w - 2 : w] = True
    for block in blocks:
        color = int(block[0, 0])
        assert np.all(block == color), "corner markers must be solid 2x2 blocks"
        colors.append(color)
    assert len(set(colors)) == 4, "corner marker colors must be distinct"
    return set(colors), marker_mask


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_objects(
    grid: np.ndarray,
    bg: int,
    marker_colors: set[int],
    marker_mask: np.ndarray,
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg or marker_mask[r, c] or color in marker_colors:
                seen[r, c] = True
                continue

            seen[r, c] = True
            stack = [(r, c)]
            coords = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc]:
                        continue
                    if marker_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
                    coords.append((nr, nc))

            arr = np.array(coords, dtype=int)
            objects.append({"color": color, "coords": arr, "bbox": _bbox_from_coords(arr)})

    assert objects, "expected at least one foreground object"
    objects.sort(key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))
    return objects


def _hole_summary_from_bool(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    area_sum = 0

    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            area = 0
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                area += 1
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
                area_sum += area

    return holes, area_sum


def _hole_summary_in_grid(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros(sub.shape, dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    bg_mask = sub == bg

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    area_sum = 0

    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            area = 0
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                area += 1
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
                area_sum += area

    return holes, area_sum


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome object according to its number of enclosed holes and
    the total hole area. The four 2x2 corner blocks act as a fixed direction
    legend: upper-left -> left, upper-right -> up, lower-left -> down,
    lower-right -> right.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    marker_colors, marker_mask = _corner_markers(g)
    objects = _find_objects(g, bg, marker_colors, marker_mask)

    out = g.copy()
    moved: list[tuple[int, np.ndarray]] = []
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        color = int(obj["color"])
        holes, area_sum = _hole_summary_in_grid(g, coords, bg)
        assert holes in DIR_BY_HOLES, "each object must have between 1 and 4 holes"
        dr, dc = DIR_BY_HOLES[holes]
        shifted = coords + np.array([dr * area_sum, dc * area_sum], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
        moved.append((color, shifted))

    h, w = out.shape
    for color, coords in moved:
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if 0 <= rr < h and 0 <= cc < w and not marker_mask[rr, cc]:
                out[rr, cc] = color

    return out


def _random_partition(total: int, parts: int, cap: int) -> list[int]:
    remaining = total
    out: list[int] = []
    for i in range(parts):
        slots_left = parts - i - 1
        lo = max(1, remaining - cap * slots_left)
        hi = min(cap, remaining - slots_left)
        value = random.randint(lo, hi)
        out.append(value)
        remaining -= value
    random.shuffle(out)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _make_object_mask(hole_count: int) -> tuple[np.ndarray, int]:
    top = random.randint(1, 2)
    mid_r = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    mid_c = random.randint(1, 2)
    right = random.randint(1, 2)

    base_h = top + SLOT_SIZE + mid_r + SLOT_SIZE + bottom
    base_w = left + SLOT_SIZE + mid_c + SLOT_SIZE + right
    base = np.ones((base_h, base_w), dtype=bool)
    row_starts = [top, top + SLOT_SIZE + mid_r]
    col_starts = [left, left + SLOT_SIZE + mid_c]

    total_hole_area = random.randint(hole_count, min(hole_count + 4, 4 * hole_count))
    areas = _random_partition(total_hole_area, hole_count, 4)
    slots = random.sample(range(4), hole_count)
    for slot, area in zip(slots, areas):
        slot_r = slot // 2
        slot_c = slot % 2
        hole_h, hole_w = random.choice(AREA_SHAPES[area])
        off_r = random.randint(0, SLOT_SIZE - hole_h)
        off_c = random.randint(0, SLOT_SIZE - hole_w)
        r0 = row_starts[slot_r] + off_r
        c0 = col_starts[slot_c] + off_c
        base[r0 : r0 + hole_h, c0 : c0 + hole_w] = False

    pad_t = random.randint(0, 1)
    pad_b = random.randint(0, 1)
    pad_l = random.randint(0, 1)
    pad_r = random.randint(0, 1)
    canvas = np.zeros((base_h + pad_t + pad_b, base_w + pad_l + pad_r), dtype=bool)
    roff = pad_t
    coff = pad_l
    canvas[roff : roff + base_h, coff : coff + base_w] = base

    if pad_t and random.random() < 0.7:
        depth = random.randint(1, pad_t)
        length = random.randint(max(2, base_w // 3), base_w)
        start = random.randint(0, base_w - length)
        canvas[roff - depth : roff, coff + start : coff + start + length] = True
    if pad_b and random.random() < 0.7:
        depth = random.randint(1, pad_b)
        length = random.randint(max(2, base_w // 3), base_w)
        start = random.randint(0, base_w - length)
        canvas[roff + base_h : roff + base_h + depth, coff + start : coff + start + length] = True
    if pad_l and random.random() < 0.7:
        depth = random.randint(1, pad_l)
        length = random.randint(max(2, base_h // 3), base_h)
        start = random.randint(0, base_h - length)
        canvas[roff + start : roff + start + length, coff - depth : coff] = True
    if pad_r and random.random() < 0.7:
        depth = random.randint(1, pad_r)
        length = random.randint(max(2, base_h // 3), base_h)
        start = random.randint(0, base_h - length)
        canvas[roff + start : roff + start + length, coff + base_w : coff + base_w + depth] = True

    mask = _crop_mask(canvas)
    holes, area_sum = _hole_summary_from_bool(mask)
    assert holes == hole_count
    assert area_sum == total_hole_area
    return mask, area_sum


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    for r, c in coords:
        blocked[int(r), int(c)] = True


def _build_expected(
    grid: np.ndarray,
    marker_mask: np.ndarray,
    placements: list[dict[str, object]],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for item in placements:
        coords = np.asarray(item["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
    for item in placements:
        color = int(item["color"])
        coords = np.asarray(item["coords"], dtype=int)
        dr, dc = DIR_BY_HOLES[int(item["holes"])]
        shift = int(item["shift"])
        moved = coords + np.array([dr * shift, dc * shift], dtype=int)
        for r, c in moved:
            rr = int(r)
            cc = int(c)
            if 0 <= rr < h and 0 <= cc < w and not marker_mask[rr, cc]:
                out[rr, cc] = color
    return out


def _place_objects(
    grid_h: int,
    grid_w: int,
    marker_mask: np.ndarray,
    objects: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    blocked_in = marker_mask.copy()
    blocked_out = marker_mask.copy()
    placed: list[dict[str, object]] = []

    order = sorted(
        objects,
        key=lambda obj: (
            int(obj["shift"]) + max(int(obj["mask"].shape[0]), int(obj["mask"].shape[1])),
            int(obj["holes"]),
        ),
        reverse=True,
    )

    for obj in order:
        mask = np.asarray(obj["mask"], dtype=bool)
        hole_count = int(obj["holes"])
        shift = int(obj["shift"])
        dr, dc = DIR_BY_HOLES[hole_count]
        rel = np.argwhere(mask).astype(int)
        obj_h, obj_w = mask.shape

        r_lo = INPUT_MARGIN
        c_lo = INPUT_MARGIN
        r_hi = grid_h - obj_h - INPUT_MARGIN
        c_hi = grid_w - obj_w - INPUT_MARGIN
        if dr < 0:
            r_lo = max(r_lo, shift)
        if dr > 0:
            r_hi = min(r_hi, grid_h - obj_h - INPUT_MARGIN - shift)
        if dc < 0:
            c_lo = max(c_lo, shift)
        if dc > 0:
            c_hi = min(c_hi, grid_w - obj_w - INPUT_MARGIN - shift)
        if r_lo > r_hi or c_lo > c_hi:
            return None

        anchors = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        random.shuffle(anchors)

        placed_here = False
        for anchor_r, anchor_c in anchors:
            coords = rel + np.array([anchor_r, anchor_c], dtype=int)
            moved = coords + np.array([dr * shift, dc * shift], dtype=int)
            if np.any(blocked_in[coords[:, 0], coords[:, 1]]):
                continue
            if np.any(blocked_out[moved[:, 0], moved[:, 1]]):
                continue

            _mark_blocked(blocked_in, coords)
            _mark_blocked(blocked_out, moved)
            placed.append(
                {
                    "color": int(obj["color"]),
                    "holes": hole_count,
                    "shift": shift,
                    "mask": mask,
                    "coords": coords,
                }
            )
            placed_here = True
            break

        if not placed_here:
            return None

    return placed


def _generate_attempt() -> np.ndarray | None:
    object_count = random.choice(OBJECT_COUNT_OPTIONS)
    (grid_h_lo, grid_h_hi), (grid_w_lo, grid_w_hi) = GRID_RANGES_BY_COUNT[object_count]
    grid_h = random.randint(grid_h_lo, grid_h_hi)
    grid_w = random.randint(grid_w_lo, grid_w_hi)
    bg = random.choice(ALL_COLORS)
    hole_counts = random.sample([1, 2, 3, 4], object_count)

    needed_colors = 4 + object_count
    usable = [color for color in ALL_COLORS if color != bg]
    if len(usable) < needed_colors:
        return None
    chosen = random.sample(usable, needed_colors)
    marker_colors = chosen[:4]
    object_colors = chosen[4:]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[:2, :2] = marker_colors[0]
    grid[:2, grid_w - 2 : grid_w] = marker_colors[1]
    grid[grid_h - 2 : grid_h, :2] = marker_colors[2]
    grid[grid_h - 2 : grid_h, grid_w - 2 : grid_w] = marker_colors[3]
    _, marker_mask = _corner_markers(grid)

    objects: list[dict[str, object]] = []
    for hole_count, color in zip(hole_counts, object_colors):
        mask, shift = _make_object_mask(hole_count)
        objects.append({"holes": hole_count, "shift": shift, "mask": mask, "color": color})

    placed = _place_objects(grid_h, grid_w, marker_mask, objects)
    if placed is None:
        return None

    for item in placed:
        color = int(item["color"])
        coords = np.asarray(item["coords"], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = color

    expected = _build_expected(grid, marker_mask, placed, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
