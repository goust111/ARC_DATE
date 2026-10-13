from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW = 2
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
NON_SPECIAL_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

# --------------- solver helpers ---------------


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_arrow(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(comp) != 4:
        return None
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    rel = {(r - r0, c - c0) for r, c in comp}
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1

    if (hh, ww) == (2, 3):
        row_counts = [sum(1 for r, _ in rel if r == rr) for rr in range(2)]
        if sorted(row_counts) != [1, 3]:
            return None
        shaft_row_rel = row_counts.index(3)
        tip_row_rel = 1 - shaft_row_rel
        if {(shaft_row_rel, 0), (shaft_row_rel, 1), (shaft_row_rel, 2)} - rel:
            return None
        tip_cols = [c for r, c in rel if r == tip_row_rel]
        if len(tip_cols) != 1:
            return None
        tip_col_rel = tip_cols[0]
        if tip_col_rel == 0:
            direction = "left"
            tail = (r0 + shaft_row_rel, c1)
        elif tip_col_rel == 2:
            direction = "right"
            tail = (r0 + shaft_row_rel, c0)
        else:
            return None
        return {
            "cells": set(comp),
            "direction": direction,
            "vector": DIRS[direction],
            "tail": tail,
            "axis_kind": "row",
            "axis_index": r0 + shaft_row_rel,
        }

    if (hh, ww) == (3, 2):
        col_counts = [sum(1 for _, c in rel if c == cc) for cc in range(2)]
        if sorted(col_counts) != [1, 3]:
            return None
        shaft_col_rel = col_counts.index(3)
        tip_col_rel = 1 - shaft_col_rel
        if {(0, shaft_col_rel), (1, shaft_col_rel), (2, shaft_col_rel)} - rel:
            return None
        tip_rows = [r for r, c in rel if c == tip_col_rel]
        if len(tip_rows) != 1:
            return None
        tip_row_rel = tip_rows[0]
        if tip_row_rel == 0:
            direction = "up"
            tail = (r1, c0 + shaft_col_rel)
        elif tip_row_rel == 2:
            direction = "down"
            tail = (r0, c0 + shaft_col_rel)
        else:
            return None
        return {
            "cells": set(comp),
            "direction": direction,
            "vector": DIRS[direction],
            "tail": tail,
            "axis_kind": "col",
            "axis_index": c0 + shaft_col_rel,
        }

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each arrow-tagged object by a distance encoded in its aligned row or column.

    Input structure:
    - The background is black.
    - Non-red foreground objects are monochrome 4-connected components.
    - Some objects have a nearby red arrow marker made from a length-3 line plus one attached tip pixel, pointing up, down, left, or right.

    Rule:
    - Identify every red arrow and infer its direction, tail cell, and shaft row or column from the arrow geometry.
    - Match the arrow to the unique object touching the tail side opposite the pointing direction.
    - If the arrow points left or right, count how many pixels of that object lie on the arrow's shaft row; if it points up or down, count how many lie on the shaft column.
    - Remove the arrow, clear the object's original cells, and repaint the whole object translated by that count in the arrow direction, clipping any translated cells that leave the grid.

    Key insight:
    - The arrow gives the motion direction, while the movement distance is hidden in the object's own projection onto the arrow's row or column.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    objects: list[dict[str, object]] = []
    cell_to_object: dict[tuple[int, int], int] = {}
    for color in sorted(int(v) for v in np.unique(g)):
        if color in (BG, ARROW):
            continue
        for comp in _components(g == color):
            oid = len(objects)
            cells = [tuple(map(int, cell)) for cell in comp]
            objects.append({"color": color, "cells": cells})
            for cell in cells:
                cell_to_object[cell] = oid

    moves: list[tuple[int, int, int]] = []
    seen_objects: set[int] = set()
    for comp in _components(g == ARROW):
        arrow = _parse_arrow(comp)
        assert arrow is not None, "every red component must be a valid arrow"
        tail_r, tail_c = arrow["tail"]
        dr, dc = arrow["vector"]

        primary = (tail_r - dr, tail_c - dc)
        oid = cell_to_object.get(primary)
        if oid is None:
            neighbor_ids = {
                cell_to_object[(nr, nc)]
                for nr, nc in _neighbors4(tail_r, tail_c, h, w)
                if (nr, nc) in cell_to_object
            }
            assert len(neighbor_ids) == 1, "tail must touch exactly one object"
            oid = next(iter(neighbor_ids))
        assert oid not in seen_objects, "each object should be controlled by at most one arrow"
        seen_objects.add(oid)

        obj = objects[oid]
        cells = obj["cells"]
        if arrow["axis_kind"] == "row":
            axis = int(arrow["axis_index"])
            count = sum(1 for r, _ in cells if r == axis)
        else:
            axis = int(arrow["axis_index"])
            count = sum(1 for _, c in cells if c == axis)
        assert count > 0, "matched object must project onto the arrow axis"
        moves.append((oid, dr, dc, count))

        for r, c in arrow["cells"]:
            out[r, c] = BG

    for oid, _, _, _ in moves:
        for r, c in objects[oid]["cells"]:
            out[r, c] = BG

    for oid, dr, dc, count in moves:
        color = int(objects[oid]["color"])
        for r, c in objects[oid]["cells"]:
            nr = r + dr * count
            nc = c + dc * count
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color

    return out


# --------------- generator helpers ---------------

GRID_H_RANGE = (12, 28)
GRID_W_RANGE = (12, 28)
MOVABLE_COUNT_RANGE = (2, 5)
STATIC_COUNT_RANGE = (0, 2)
OBJ_H_RANGE = (2, 5)
OBJ_W_RANGE = (2, 5)
EXTRA_CELL_MAX = 5
PLACEMENT_GAP = 1
CROP_PROB = 0.5
GEN_MAX_ATTEMPTS = 300


def _buffered(cells: set[tuple[int, int]], h: int, w: int, radius: int = PLACEMENT_GAP) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - radius), min(h, r + radius + 1)):
            for nc in range(max(0, c - radius), min(w, c + radius + 1)):
                out.add((nr, nc))
    return out


def _touches(cells: set[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    return any(cell in blocked for cell in cells)


def _build_movable_shape(direction: str):
    if direction in ("left", "right"):
        hh = random.randint(*OBJ_H_RANGE)
        ww = random.randint(*OBJ_W_RANGE)
        anchor = random.randrange(hh)
        width = random.randint(1, ww)
        if direction == "left":
            cells = {(anchor, c) for c in range(width)}
        else:
            cells = {(anchor, c) for c in range(ww - width, ww)}
        max_area = min(hh * ww, width + EXTRA_CELL_MAX)
        min_area = max(3, width + (1 if hh * ww > width else 0))
        target_area = random.randint(min_area, max_area)
        while len(cells) < target_area:
            frontier = [
                (r, c)
                for r in range(hh)
                for c in range(ww)
                if r != anchor
                and (r, c) not in cells
                and any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells)
            ]
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) < min_area:
            return None
        return cells, hh, ww, anchor, width

    hh = random.randint(*OBJ_H_RANGE)
    ww = random.randint(*OBJ_W_RANGE)
    anchor = random.randrange(ww)
    height = random.randint(1, hh)
    if direction == "up":
        cells = {(r, anchor) for r in range(height)}
    else:
        cells = {(r, anchor) for r in range(hh - height, hh)}
    max_area = min(hh * ww, height + EXTRA_CELL_MAX)
    min_area = max(3, height + (1 if hh * ww > height else 0))
    target_area = random.randint(min_area, max_area)
    while len(cells) < target_area:
        frontier = [
            (r, c)
            for r in range(hh)
            for c in range(ww)
            if c != anchor
            and (r, c) not in cells
            and any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells)
        ]
        if not frontier:
            break
        cells.add(random.choice(frontier))
    if len(cells) < min_area:
        return None
    return cells, hh, ww, anchor, height


def _build_static_shape() -> set[tuple[int, int]] | None:
    hh = random.randint(2, 4)
    ww = random.randint(2, 4)
    target_area = random.randint(3, min(hh * ww, 6))
    start = (random.randrange(hh), random.randrange(ww))
    cells = {start}
    while len(cells) < target_area:
        frontier = [
            (r, c)
            for r in range(hh)
            for c in range(ww)
            if (r, c) not in cells and any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells)
        ]
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells if len(cells) == target_area else None


def _relative_arrow_cells(direction: str, hh: int, ww: int, anchor: int, tip_side: int) -> set[tuple[int, int]]:
    if direction == "right":
        return {(anchor, ww + d) for d in range(3)} | {(anchor + tip_side, ww + 2)}
    if direction == "left":
        return {(anchor, -1 - d) for d in range(3)} | {(anchor + tip_side, -3)}
    if direction == "down":
        return {(hh + d, anchor) for d in range(3)} | {(hh + 2, anchor + tip_side)}
    return {(-1 - d, anchor) for d in range(3)} | {(-3, anchor + tip_side)}


def _shifted(cells: set[tuple[int, int]], dr: int, dc: int, amount: int) -> set[tuple[int, int]]:
    return {(r + dr * amount, c + dc * amount) for r, c in cells}


def _in_bounds_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w}


def _place_cells(rel_cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in rel_cells}


def _origin_interval(
    rel_obj: set[tuple[int, int]],
    rel_arrow: set[tuple[int, int]],
    direction: str,
    amount: int,
    h: int,
    w: int,
    crop_required: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    all_rel = rel_obj | rel_arrow
    min_r = min(r for r, _ in all_rel)
    max_r = max(r for r, _ in all_rel)
    min_c = min(c for _, c in all_rel)
    max_c = max(c for _, c in all_rel)
    top_lo, top_hi = -min_r, h - 1 - max_r
    left_lo, left_hi = -min_c, w - 1 - max_c
    if top_lo > top_hi or left_lo > left_hi:
        return None

    shifted_rows = [r + DIRS[direction][0] * amount for r, _ in rel_obj]
    shifted_cols = [c + DIRS[direction][1] * amount for _, c in rel_obj]

    if direction == "right":
        full_hi = w - 1 - max(shifted_cols)
        some_hi = w - 1 - min(shifted_cols)
        if crop_required:
            left_lo = max(left_lo, full_hi + 1)
            left_hi = min(left_hi, some_hi)
        else:
            left_hi = min(left_hi, full_hi)
    elif direction == "left":
        full_lo = -min(shifted_cols)
        some_lo = -max(shifted_cols)
        if crop_required:
            left_lo = max(left_lo, some_lo)
            left_hi = min(left_hi, full_lo - 1)
        else:
            left_lo = max(left_lo, full_lo)
    elif direction == "down":
        full_hi = h - 1 - max(shifted_rows)
        some_hi = h - 1 - min(shifted_rows)
        if crop_required:
            top_lo = max(top_lo, full_hi + 1)
            top_hi = min(top_hi, some_hi)
        else:
            top_hi = min(top_hi, full_hi)
    else:
        full_lo = -min(shifted_rows)
        some_lo = -max(shifted_rows)
        if crop_required:
            top_lo = max(top_lo, some_lo)
            top_hi = min(top_hi, full_lo - 1)
        else:
            top_lo = max(top_lo, full_lo)

    if top_lo > top_hi or left_lo > left_hi:
        return None
    return (top_lo, top_hi), (left_lo, left_hi)


def _expected_output(
    grid: np.ndarray,
    movable_parts: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for part in movable_parts:
        for r, c in part["arrow_cells"]:
            out[r, c] = BG
        for r, c in part["object_cells"]:
            out[r, c] = BG
    for part in movable_parts:
        color = int(part["color"])
        for r, c in part["moved_cells"]:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    movable_count = random.randint(*MOVABLE_COUNT_RANGE)
    static_count = random.randint(*STATIC_COUNT_RANGE)
    total_objects = movable_count + static_count
    if total_objects > len(NON_SPECIAL_COLORS):
        return None
    colors = random.sample(NON_SPECIAL_COLORS, total_objects)

    grid = np.full((h, w), BG, dtype=int)
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()
    movable_parts: list[dict[str, object]] = []

    directions = list(DIRS)
    random.shuffle(directions)
    directions = directions[: min(movable_count, 4)]
    while len(directions) < movable_count:
        directions.append(random.choice(list(DIRS)))
    random.shuffle(directions)
    crop_flags = [True] * (movable_count // 2) + [False] * (movable_count // 2)
    if movable_count % 2:
        crop_flags.append(random.choice([True, False]))
    random.shuffle(crop_flags)
    for idx in range(movable_count):
        color = colors[idx]
        direction = directions[idx]
        dr, dc = DIRS[direction]
        crop_required = crop_flags[idx]
        placed = False
        for _ in range(180):
            built = _build_movable_shape(direction)
            if built is None:
                continue
            rel_obj, hh, ww, anchor, amount = built
            tip_sides = random.sample([-1, 1], 2)
            for tip_side in tip_sides:
                rel_arrow = _relative_arrow_cells(direction, hh, ww, anchor, tip_side)
                intervals = _origin_interval(rel_obj, rel_arrow, direction, amount, h, w, crop_required)
                if intervals is None:
                    continue
                (top_lo, top_hi), (left_lo, left_hi) = intervals
                for _ in range(12):
                    top = random.randint(top_lo, top_hi)
                    left = random.randint(left_lo, left_hi)
                    object_cells = _place_cells(rel_obj, top, left)
                    arrow_cells = _place_cells(rel_arrow, top, left)
                    union_cells = object_cells | arrow_cells
                    if _touches(union_cells, input_blocked):
                        continue

                    shifted_all = _shifted(object_cells, dr, dc, amount)
                    moved_cells = _in_bounds_cells(shifted_all, h, w)
                    if not moved_cells:
                        continue
                    cropped = len(moved_cells) != len(object_cells)
                    if cropped != crop_required:
                        continue
                    if _touches(moved_cells, output_blocked):
                        continue

                    movable_parts.append(
                        {
                            "color": color,
                            "object_cells": object_cells,
                            "arrow_cells": arrow_cells,
                            "moved_cells": moved_cells,
                        }
                    )
                    input_blocked |= _buffered(union_cells, h, w)
                    output_blocked |= _buffered(moved_cells, h, w)
                    for r, c in object_cells:
                        grid[r, c] = color
                    for r, c in arrow_cells:
                        grid[r, c] = ARROW
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    for idx in range(static_count):
        color = colors[movable_count + idx]
        placed = False
        for _ in range(180):
            rel_obj = _build_static_shape()
            if rel_obj is None:
                continue
            hh = max(r for r, _ in rel_obj) + 1
            ww = max(c for _, c in rel_obj) + 1
            top = random.randint(0, h - hh)
            left = random.randint(0, w - ww)
            object_cells = _place_cells(rel_obj, top, left)
            if _touches(object_cells, input_blocked):
                continue
            if _touches(object_cells, output_blocked):
                continue
            input_blocked |= _buffered(object_cells, h, w)
            output_blocked |= _buffered(object_cells, h, w)
            for r, c in object_cells:
                grid[r, c] = color
            placed = True
            break
        if not placed:
            return None

    if not movable_parts:
        return None

    expected = _expected_output(grid, movable_parts)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-translation puzzle")
