from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
SIDE_ORDER = ("top", "bottom", "left", "right")


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_slot_side(grid: np.ndarray, bg: int) -> tuple[str, list[tuple[int, int, int]]]:
    h, w = grid.shape
    edge_points = {
        "top": [(0, c, int(grid[0, c])) for c in range(w) if int(grid[0, c]) != bg],
        "bottom": [(h - 1, c, int(grid[h - 1, c])) for c in range(w) if int(grid[h - 1, c]) != bg],
        "left": [(r, 0, int(grid[r, 0])) for r in range(h) if int(grid[r, 0]) != bg],
        "right": [(r, w - 1, int(grid[r, w - 1])) for r in range(h) if int(grid[r, w - 1]) != bg],
    }
    present = [(side, pts) for side, pts in edge_points.items() if pts]
    assert len(present) == 1, "expected slots on exactly one grid edge"
    side, slots = present[0]
    if side in ("top", "bottom"):
        slots = sorted(slots, key=lambda item: item[1])
    else:
        slots = sorted(slots, key=lambda item: item[0])
    return side, slots


def _find_objects(grid: np.ndarray, bg: int, slot_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    fg_mask = (grid != bg) & (~slot_mask)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not fg_mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.asarray(coords, dtype=int)
            colors = np.unique(grid[coords_arr[:, 0], coords_arr[:, 1]])
            assert len(colors) == 1, "every foreground object must be monochromatic"
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            local = coords_arr - np.array([[r0, c0]], dtype=int)
            objects.append(
                {
                    "color": int(colors[0]),
                    "coords": local,
                    "area": int(local.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                }
            )

    return objects


def _can_place(shape: dict, anchor_r: int, anchor_c: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    coords = shape["coords"] + np.array([[anchor_r, anchor_c]], dtype=int)
    if coords[:, 0].min() < 0 or coords[:, 1].min() < 0:
        return False
    if coords[:, 0].max() >= h or coords[:, 1].max() >= w:
        return False
    return not occupied[coords[:, 0], coords[:, 1]].any()


def _place_ranked_objects(
    shape_order: list[dict],
    slots: list[tuple[int, int, int]],
    side: str,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict]]:
    h, w = canvas_shape
    out = np.full((h, w), BACKGROUND, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for r, c, color in slots:
        out[int(r), int(c)] = int(color)

    placements: list[dict] = []
    for shape, (slot_r, slot_c, slot_color) in zip(shape_order, slots, strict=True):
        if side == "bottom":
            anchor_r = 0
            anchor_c = int(slot_c)
            assert anchor_c + shape["w"] <= w
            while _can_place(shape, anchor_r + 1, anchor_c, occupied):
                anchor_r += 1
        elif side == "top":
            anchor_r = h - shape["h"]
            anchor_c = int(slot_c)
            assert anchor_c + shape["w"] <= w
            while _can_place(shape, anchor_r - 1, anchor_c, occupied):
                anchor_r -= 1
        elif side == "left":
            anchor_r = int(slot_r)
            anchor_c = w - shape["w"]
            assert anchor_r + shape["h"] <= h
            while _can_place(shape, anchor_r, anchor_c - 1, occupied):
                anchor_c -= 1
        else:
            anchor_r = int(slot_r)
            anchor_c = 0
            assert anchor_r + shape["h"] <= h
            while _can_place(shape, anchor_r, anchor_c + 1, occupied):
                anchor_c += 1

        coords = shape["coords"] + np.array([[anchor_r, anchor_c]], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(slot_color)
        occupied[coords[:, 0], coords[:, 1]] = True

        supported_by_object = False
        if side == "bottom":
            supported_by_object = anchor_r != h - shape["h"]
        elif side == "top":
            supported_by_object = anchor_r != 0
        elif side == "left":
            supported_by_object = anchor_c != 0
        else:
            supported_by_object = anchor_c != w - shape["w"]

        placements.append(
            {
                "coords": coords,
                "anchor": (int(anchor_r), int(anchor_c)),
                "slot": (int(slot_r), int(slot_c), int(slot_color)),
                "supported_by_object": bool(supported_by_object),
            }
        )

    return out, placements


def _place_paired_objects(
    pairs: list[tuple[dict, tuple[int, int, int]]],
    side: str,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict]]:
    shapes = [shape for shape, _slot in pairs]
    slots = [slot for _shape, slot in pairs]
    return _place_ranked_objects(shapes, slots, side, canvas_shape)


def main(grid: np.ndarray) -> np.ndarray:
    """Drop area-ranked monochrome objects into edge slots, recoloring them by slot color as earlier landings become supports.

    Input structure:
    - Exactly one grid edge contains a spaced row of colored single-cell slots.
    - Away from that edge are the same number of monochrome 4-connected objects,
      each with a distinct area and arbitrary input color.
    - Some object colors may match slot colors, but those matches are distractors.

    Rule:
    - The slot edge determines compression direction: top->up, bottom->down,
      left->left, right->right.
    - Rank slots by scan order along their edge, and rank objects by descending
      area; pair rank i with rank i.
    - For each pair in rank order, align the object's bbox side nearest the scan
      start with the slot coordinate (left edge for top/bottom slots, top edge
      for left/right slots), then move the whole shape along the compression
      direction until the grid boundary or an already-landed object stops it.
    - Recolor the landed object to the slot color, preserve the shape, and keep
      the slot cells visible.

    Key insight:
    - Neither the objects' original positions nor their input colors define the
      pairing; only area rank and edge slot order do, and the landing order
      itself changes later resting heights through chained stacking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    side, slots = _find_slot_side(g, bg)
    slot_mask = np.zeros_like(g, dtype=bool)
    for r, c, _color in slots:
        slot_mask[int(r), int(c)] = True

    objects = _find_objects(g, bg, slot_mask)
    assert len(objects) == len(slots), "slot/object counts must match"
    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "object areas must be distinct"

    ranked_objects = sorted(objects, key=lambda obj: obj["area"], reverse=True)
    out, _placements = _place_ranked_objects(ranked_objects, slots, side, g.shape)
    return out


def _shape_from_cells(cells: set[tuple[int, int]]) -> dict:
    coords = np.asarray(sorted(cells), dtype=int)
    r0, c0 = coords.min(axis=0)
    coords = coords - np.array([[r0, c0]], dtype=int)
    r1, c1 = coords.max(axis=0)
    return {
        "coords": coords,
        "area": int(coords.shape[0]),
        "h": int(r1 + 1),
        "w": int(c1 + 1),
    }


def _generate_shape(h: int, w: int, target_area: int) -> dict:
    cells = {(h - 1, c) for c in range(w)}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return _shape_from_cells(cells)


def _sample_shapes(count: int) -> list[dict] | None:
    shapes: list[dict] = []
    used_areas: set[int] = set()
    for _ in range(count):
        for _shape_try in range(80):
            h = random.randint(2, 4)
            w = random.randint(2, 5)
            min_area = w + 1
            max_area = h * w
            if min_area > max_area:
                continue
            area = random.randint(min_area, max_area)
            if area in used_areas:
                continue
            shape = _generate_shape(h, w, area)
            used_areas.add(area)
            shapes.append(shape)
            break
        else:
            return None
    if sum(shape["w"] > 1 for shape in shapes) < 2:
        return None
    return shapes


def _sample_slot_columns(widths: list[int], grid_w: int) -> list[int] | None:
    cols: list[int] = []

    def rec(i: int) -> bool:
        if i == len(widths):
            return True
        lo = 1 if i == 0 else cols[-1] + 2
        hi = grid_w - widths[i]
        candidates = []
        for col in range(lo, hi + 1):
            feasible = True
            prev = col
            for j in range(i + 1, len(widths)):
                prev += 2
                if prev > grid_w - widths[j]:
                    feasible = False
                    break
            if feasible:
                candidates.append(col)
        random.shuffle(candidates)
        for col in candidates:
            cols.append(int(col))
            if rec(i + 1):
                return True
            cols.pop()
        return False

    if not rec(0):
        return None
    return cols


def _permute_slot_colors(slot_colors: list[int]) -> list[int]:
    n = len(slot_colors)
    shift = random.randint(1, n - 1)
    return [slot_colors[(i + shift) % n] for i in range(n)]


def _scatter_objects(grid_shape: tuple[int, int], objects: list[dict], colors: list[int]) -> np.ndarray | None:
    h, w = grid_shape
    occupancy = np.zeros((h, w), dtype=bool)
    placements: list[tuple[np.ndarray, int]] = []

    def candidates_for(shape: dict) -> list[tuple[int, int]]:
        shape_mask = np.zeros((shape["h"], shape["w"]), dtype=bool)
        shape_mask[shape["coords"][:, 0], shape["coords"][:, 1]] = True
        cands = []
        for r0 in range(1, h - shape["h"] - 1):
            for c0 in range(1, w - shape["w"] - 1):
                coords = shape["coords"] + np.array([[r0, c0]], dtype=int)
                if occupancy[coords[:, 0], coords[:, 1]].any():
                    continue
                blocked = False
                for rr, cc in coords:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = int(rr + dr), int(cc + dc)
                            if 0 <= nr < h and 0 <= nc < w and occupancy[nr, nc]:
                                blocked = True
                                break
                        if blocked:
                            break
                    if blocked:
                        break
                if not blocked:
                    cands.append((r0, c0))
        random.shuffle(cands)
        return cands

    order = list(range(len(objects)))
    order.sort(key=lambda idx: (objects[idx]["h"] * objects[idx]["w"], objects[idx]["area"]), reverse=True)

    def rec(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        shape = objects[idx]
        for r0, c0 in candidates_for(shape):
            coords = shape["coords"] + np.array([[r0, c0]], dtype=int)
            occupancy[coords[:, 0], coords[:, 1]] = True
            placements.append((coords, colors[idx]))
            if rec(k + 1):
                return True
            placements.pop()
            occupancy[coords[:, 0], coords[:, 1]] = False
        return False

    if not rec(0):
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for coords, color in placements:
        grid[coords[:, 0], coords[:, 1]] = int(color)
    return grid


def _simulate_color_matching(
    objects_in_input_order: list[dict],
    input_colors: list[int],
    slots: list[tuple[int, int, int]],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    by_color = {int(color): obj for color, obj in zip(input_colors, objects_in_input_order, strict=True)}
    matched = [by_color[int(slot_color)] for _r, _c, slot_color in slots]
    out, _placements = _place_ranked_objects(matched, slots, "bottom", canvas_shape)
    return out


def _transform_from_bottom(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "bottom":
        return grid.copy()
    if side == "top":
        return np.flipud(grid)
    if side == "left":
        return np.rot90(grid, k=3)
    if side == "right":
        return grid.T.copy()
    raise ValueError(f"unknown side: {side}")


def _safe_output(fn) -> np.ndarray | None:
    try:
        out, _placements = fn()
        return out
    except AssertionError:
        return None


def _build_canonical_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    count = random.randint(3, 4)

    shapes = _sample_shapes(count)
    if shapes is None:
        return None
    ranked_shapes = sorted(shapes, key=lambda shape: shape["area"], reverse=True)
    widths = [shape["w"] for shape in ranked_shapes]
    slot_cols = _sample_slot_columns(widths, w)
    if slot_cols is None:
        return None

    overlap_pairs = [
        j
        for j in range(1, count)
        for i in range(j)
        if slot_cols[j] < slot_cols[i] + ranked_shapes[i]["w"]
    ]
    if not overlap_pairs:
        return None

    slot_colors = random.sample(FOREGROUND_COLORS, count)
    slots = [(h - 1, slot_cols[i], slot_colors[i]) for i in range(count)]
    expected, placements = _place_ranked_objects(ranked_shapes, slots, "bottom", (h, w))
    if not any(item["supported_by_object"] for item in placements[1:]):
        return None

    reverse_pairs = list(reversed(list(zip(ranked_shapes, slots, strict=True))))
    reverse_expected = _safe_output(lambda: _place_paired_objects(reverse_pairs, "bottom", (h, w)))
    if reverse_expected is not None and np.array_equal(reverse_expected, expected):
        return None

    ascending_expected = _safe_output(
        lambda: _place_ranked_objects(
            sorted(ranked_shapes, key=lambda shape: shape["area"]),
            slots,
            "bottom",
            (h, w),
        )
    )
    if ascending_expected is not None and np.array_equal(ascending_expected, expected):
        return None

    input_colors = _permute_slot_colors(slot_colors)
    color_match_expected = _safe_output(
        lambda: (_simulate_color_matching(ranked_shapes, input_colors, slots, (h, w)), [])
    )
    if color_match_expected is not None and np.array_equal(color_match_expected, expected):
        return None

    scattered = _scatter_objects((h, w), ranked_shapes, input_colors)
    if scattered is None:
        return None

    inp = scattered.copy()
    for _r, c, color in slots:
        inp[h - 1, int(c)] = int(color)

    if np.count_nonzero(inp[0, :]) != 0 or np.count_nonzero(inp[:, 0]) != 0 or np.count_nonzero(inp[:, -1]) != 0:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_canonical_case()
        if built is None:
            continue
        inp, expected = built
        side = random.choice(SIDE_ORDER)
        rotated_inp = _transform_from_bottom(inp, side)
        rotated_expected = _transform_from_bottom(expected, side)
        try:
            solved = main(rotated_inp)
        except Exception:
            continue
        if np.array_equal(solved, rotated_expected):
            return rotated_inp
    raise RuntimeError("failed to generate a valid stacking-slot puzzle")
