from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
GREY = 5
TEAL = 8
SIDES = ("left", "right", "up", "down")


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_anchor_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = components_of_color(grid, TEAL)
    valid = []
    for comp in comps:
        if len(comp) != 9:
            continue
        r0, c0, r1, c1 = bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TEAL):
            valid.append((r0, c0, r1, c1))
    assert len(valid) == 1, "expected one solid teal 3x3 anchor"
    return valid[0]


def analyze_red_object(
    coords: np.ndarray, anchor_bbox: tuple[int, int, int, int]
) -> dict[str, object]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    anchor_rows = {ar0, ar0 + 1, ar1}
    anchor_cols = {ac0, ac0 + 1, ac1}
    cell_set = {(int(r), int(c)) for r, c in coords}
    rows = {int(r) for r, _ in cell_set}
    cols = {int(c) for _, c in cell_set}
    row_overlap = sorted(rows & anchor_rows)
    col_overlap = sorted(cols & anchor_cols)

    horizontal = bool(row_overlap)
    vertical = bool(col_overlap)
    assert horizontal ^ vertical, "each red object must align with anchor on exactly one axis"

    if horizontal:
        if max(cols) < ac0:
            side = "left"
            shift = ac0 - 1 - max(c for r, c in cell_set if r in row_overlap)
            dr, dc = 0, 1
        else:
            assert min(cols) > ac1, "horizontal-aligned object must lie strictly left or right of anchor"
            side = "right"
            shift = min(c for r, c in cell_set if r in row_overlap) - (ac1 + 1)
            dr, dc = 0, -1
    else:
        if max(rows) < ar0:
            side = "up"
            shift = ar0 - 1 - max(r for r, c in cell_set if c in col_overlap)
            dr, dc = 1, 0
        else:
            assert min(rows) > ar1, "vertical-aligned object must lie strictly above or below anchor"
            side = "down"
            shift = min(r for r, c in cell_set if c in col_overlap) - (ar1 + 1)
            dr, dc = -1, 0

    shifted = {(r + shift * dr, c + shift * dc) for r, c in cell_set}
    anchor = {(r, c) for r in range(ar0, ar1 + 1) for c in range(ac0, ac1 + 1)}
    contact = sum(any((nr, nc) in anchor for nr, nc in neighbors4(r, c)) for r, c in shifted)

    return {
        "side": side,
        "shift": int(shift),
        "dr": int(dr),
        "dc": int(dc),
        "shifted": shifted,
        "contact": int(contact),
        "coords": cell_set,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each red object straight toward the teal anchor, then encode the total contact area as a red bar.

    Input structure:
    - Black background with one solid teal 3x3 square acting as the unique anchor.
    - Three to five separate red 4-connected objects lie strictly above, below, left, or right of the anchor.
    - One or two grey singleton pixels may appear as distractors and are ignored.

    Rule:
    - Ignore grey pixels and locate each red connected component.
    - For each red object, determine the unique cardinal direction that moves it toward the teal square while preserving its row or column overlap with the anchor.
    - Translate the object by the shortest amount until one or more of its pixels first become 4-neighbor-adjacent to the anchor.
    - Count, for each slid object, how many of its pixels are 4-neighbor-adjacent to the anchor after the slide; sum these contact counts to get C.
    - Return a 1x9 grid whose leftmost C cells are red and whose remaining cells are black.

    Key insight:
    - The output length depends on boundary contact after minimal sliding, not on object count, raw area, or initial distance.
    """
    g = np.asarray(grid, dtype=int)
    anchor_bbox = find_anchor_bbox(g)
    total = 0
    for comp in components_of_color(g, RED):
        total += int(analyze_red_object(comp, anchor_bbox)["contact"])
    out = np.zeros((1, 9), dtype=int)
    out[0, :total] = RED
    return out


def _cells_touch(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]], diag: bool) -> bool:
    if cells_a & cells_b:
        return True
    neigh = neighbors8 if diag else neighbors4
    for r, c in cells_a:
        for nr, nc in neigh(r, c):
            if (nr, nc) in cells_b:
                return True
    return False


def _make_contact_cells(
    anchor_bbox: tuple[int, int, int, int], side: str, start: int, k: int
) -> set[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    if side == "left":
        return {(ar0 + start + i, ac0 - 1) for i in range(k)}
    if side == "right":
        return {(ar0 + start + i, ac1 + 1) for i in range(k)}
    if side == "up":
        return {(ar0 - 1, ac0 + start + i) for i in range(k)}
    return {(ar1 + 1, ac0 + start + i) for i in range(k)}


def _allowed_cell(
    cell: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
    side: str,
    contact_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = cell
    h, w = shape
    ar0, ac0, ar1, ac1 = anchor_bbox
    if not (0 <= r < h and 0 <= c < w):
        return False
    if ar0 <= r <= ar1 and ac0 <= c <= ac1:
        return False
    if side == "left":
        if c > ac0 - 1 or ac0 - 1 - c > 3 or r < ar0 - 2 or r > ar1 + 2:
            return False
        return c != ac0 - 1 or (r, c) in contact_cells
    if side == "right":
        if c < ac1 + 1 or c - (ac1 + 1) > 3 or r < ar0 - 2 or r > ar1 + 2:
            return False
        return c != ac1 + 1 or (r, c) in contact_cells
    if side == "up":
        if r > ar0 - 1 or ar0 - 1 - r > 3 or c < ac0 - 2 or c > ac1 + 2:
            return False
        return r != ar0 - 1 or (r, c) in contact_cells
    if r < ar1 + 1 or r - (ar1 + 1) > 3 or c < ac0 - 2 or c > ac1 + 2:
        return False
    return r != ar1 + 1 or (r, c) in contact_cells


def _shape_is_irregular(cells: set[tuple[int, int]]) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1 or len(cols) == 1:
        return False
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
    return area != len(cells)


def _generate_settled_shape(
    shape: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
    side: str,
    start: int,
    k: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    target_min = max(3, k + 1, 4 if k == 3 else 3)
    for _ in range(200):
        target_size = random.randint(target_min, 6)
        contact = _make_contact_cells(anchor_bbox, side, start, k)
        cells = set(contact)
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    cell = (nr, nc)
                    if cell in cells:
                        continue
                    if _allowed_cell(cell, anchor_bbox, side, contact, shape):
                        frontier.append(cell)
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            if side == "left":
                frontier.sort(key=lambda p: (p[1], abs(p[0] - sorted(r for r, _ in contact)[0])))
            elif side == "right":
                frontier.sort(key=lambda p: (-p[1], abs(p[0] - sorted(r for r, _ in contact)[0])))
            elif side == "up":
                frontier.sort(key=lambda p: (p[0], abs(p[1] - sorted(c for _, c in contact)[0])))
            else:
                frontier.sort(key=lambda p: (-p[0], abs(p[1] - sorted(c for _, c in contact)[0])))
            chosen = random.choice(frontier[: min(5, len(frontier))])
            cells.add(chosen)
        if len(cells) != target_size:
            continue
        if not _shape_is_irregular(cells):
            continue
        if _cells_touch(cells, occupied, diag=False):
            continue
        if _cells_touch(cells, occupied, diag=True):
            continue
        analysis = analyze_red_object(np.asarray(sorted(cells), dtype=int), anchor_bbox)
        if int(analysis["contact"]) != k or int(analysis["shift"]) != 0:
            continue
        return cells
    return None


def _assign_sides(contacts: list[int]) -> list[str] | None:
    groups = {side: [] for side in SIDES}
    order = sorted(range(len(contacts)), key=lambda i: (-contacts[i], random.random()))

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        k = contacts[idx]
        sides = list(SIDES)
        random.shuffle(sides)
        for side in sides:
            current = groups[side]
            if k > 1 and current:
                continue
            if k == 1 and len(current) >= 2:
                continue
            if k == 1 and current and any(v != 1 for v in current):
                continue
            if k > 1 and current:
                continue
            current.append(k)
            if rec(pos + 1):
                return True
            current.pop()
        return False

    if not rec(0):
        return None
    per_side = {side: [] for side in SIDES}
    for idx, k in enumerate(contacts):
        for side in SIDES:
            if len(per_side[side]) < len(groups[side]) and groups[side][len(per_side[side])] == k:
                per_side[side].append(idx)
                break

    result = [None] * len(contacts)
    for side, idxs in per_side.items():
        for idx in idxs:
            result[idx] = side
    if any(side is None for side in result):
        return None
    return [str(side) for side in result]


def _segment_starts(sides: list[str], contacts: list[int]) -> list[int]:
    by_side: dict[str, list[int]] = defaultdict(list)
    for idx, side in enumerate(sides):
        by_side[side].append(idx)
    starts = [0] * len(sides)
    for side, idxs in by_side.items():
        if len(idxs) == 2:
            assert all(contacts[i] == 1 for i in idxs)
            pair = [0, 2]
            random.shuffle(pair)
            for idx, start in zip(idxs, pair):
                starts[idx] = start
        else:
            idx = idxs[0]
            starts[idx] = random.randint(0, 3 - contacts[idx])
    return starts


def _sample_contacts(n_objects: int) -> list[int] | None:
    for _ in range(200):
        total = random.randint(n_objects + 1, 9)
        vals = [1] * n_objects
        extra = total - n_objects
        while extra > 0:
            candidates = [i for i, v in enumerate(vals) if v < 3]
            if not candidates:
                break
            idx = random.choice(candidates)
            vals[idx] += 1
            extra -= 1
        if sum(vals) != total:
            continue
        if vals.count(1) < max(0, n_objects - 4):
            continue
        if len(set(vals)) == 1 and vals[0] == 1:
            continue
        sides = _assign_sides(vals)
        if sides is not None and len(set(sides)) >= 2:
            order = list(range(n_objects))
            random.shuffle(order)
            vals = [vals[i] for i in order]
            return vals
    return None


def _translate_cells(cells: set[tuple[int, int]], dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    return {(r + dr * steps, c + dc * steps) for r, c in cells}


def _sample_gaps(
    shape: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
    settled: list[dict[str, object]],
) -> list[int] | None:
    h, w = shape
    for _ in range(300):
        gaps = []
        placed: list[set[tuple[int, int]]] = []
        good = True
        order = list(range(len(settled)))
        random.shuffle(order)
        gap_vals = [0] * len(settled)
        for idx in order:
            obj = settled[idx]
            cells = obj["cells"]
            side = obj["side"]
            if side == "left":
                max_gap = min(c for _, c in cells)
                dr, dc = 0, -1
            elif side == "right":
                max_gap = w - 1 - max(c for _, c in cells)
                dr, dc = 0, 1
            elif side == "up":
                max_gap = min(r for r, _ in cells)
                dr, dc = -1, 0
            else:
                max_gap = h - 1 - max(r for r, _ in cells)
                dr, dc = 1, 0
            if max_gap <= 0:
                good = False
                break
            candidates = list(range(1, min(6, max_gap) + 1))
            random.shuffle(candidates)
            found = False
            for gap in candidates:
                moved = _translate_cells(cells, dr, dc, gap)
                if _cells_touch(moved, set().union(*placed) if placed else set(), diag=True):
                    continue
                if _cells_touch(moved, {(r, c) for r in range(anchor_bbox[0], anchor_bbox[2] + 1) for c in range(anchor_bbox[1], anchor_bbox[3] + 1)}, diag=False):
                    continue
                gap_vals[idx] = gap
                placed.append(moved)
                found = True
                break
            if not found:
                good = False
                break
        if not good:
            continue
        moved_all = [
            _translate_cells(
                obj["cells"],
                0 if obj["side"] in ("left", "right") else (-1 if obj["side"] == "up" else 1),
                (-1 if obj["side"] == "left" else 1) if obj["side"] in ("left", "right") else 0,
                gap_vals[i],
            )
            for i, obj in enumerate(settled)
        ]
        others = [set().union(*(moved_all[:i] + moved_all[i + 1 :])) for i in range(len(moved_all))]
        valid = True
        for i, obj in enumerate(settled):
            step_dr = int(obj["dr"])
            step_dc = int(obj["dc"])
            gap = gap_vals[i]
            current = moved_all[i]
            for step in range(1, gap + 1):
                mid = _translate_cells(current, step_dr, step_dc, step)
                if mid & others[i]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return gap_vals
    return None


def _place_grey_noise(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    target = random.randint(1, 2)
    placed = 0
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for r, c in cells:
        if placed == target:
            break
        if (r, c) in forbidden or int(grid[r, c]) != BG:
            continue
        if any((nr, nc) in forbidden or (0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG) for nr, nc in neighbors8(r, c)):
            continue
        grid[r, c] = GREY
        forbidden.add((r, c))
        placed += 1
    assert target == placed


def _generate_one() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    ar0 = random.randint(4, h - 7)
    ac0 = random.randint(4, w - 7)
    anchor_bbox = (ar0, ac0, ar0 + 2, ac0 + 2)

    n_objects = random.randint(3, 5)
    contacts = _sample_contacts(n_objects)
    if contacts is None:
        return None
    sides = _assign_sides(contacts)
    if sides is None:
        return None
    starts = _segment_starts(sides, contacts)

    settled: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    for side, start, k in sorted(zip(sides, starts, contacts), key=lambda x: (-x[2], x[0])):
        cells = _generate_settled_shape((h, w), anchor_bbox, side, start, k, occupied)
        if cells is None:
            return None
        analysis = analyze_red_object(np.asarray(sorted(cells), dtype=int), anchor_bbox)
        settled.append(
            {
                "side": side,
                "start": start,
                "k": k,
                "cells": cells,
                "dr": -int(analysis["dr"]),
                "dc": -int(analysis["dc"]),
            }
        )
        occupied |= cells

    gaps = _sample_gaps((h, w), anchor_bbox, settled)
    if gaps is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[ar0 : ar0 + 3, ac0 : ac0 + 3] = TEAL
    red_cells: set[tuple[int, int]] = set()
    for obj, gap in zip(settled, gaps):
        cells = _translate_cells(obj["cells"], int(obj["dr"]), int(obj["dc"]), gap)
        for r, c in cells:
            grid[r, c] = RED
        red_cells |= cells

    forbidden = red_cells | {(r, c) for r in range(ar0, ar0 + 3) for c in range(ac0, ac0 + 3)}
    for obj, gap in zip(settled, gaps):
        side = str(obj["side"])
        toward_dr = -int(obj["dr"])
        toward_dc = -int(obj["dc"])
        for step in range(1, gap + 1):
            forbidden |= _translate_cells(_translate_cells(obj["cells"], int(obj["dr"]), int(obj["dc"]), gap), toward_dr, toward_dc, step)
    _place_grey_noise(grid, forbidden)

    if not _validate_generated_grid(grid):
        return None
    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        anchor_bbox = find_anchor_bbox(g)
        anchor = {(r, c) for r in range(anchor_bbox[0], anchor_bbox[2] + 1) for c in range(anchor_bbox[1], anchor_bbox[3] + 1)}
        red_comps = components_of_color(g, RED)
        grey_comps = components_of_color(g, GREY)
        if not (3 <= len(red_comps) <= 5):
            return False
        if not (1 <= len(grey_comps) <= 2 and all(len(comp) == 1 for comp in grey_comps)):
            return False
        analyses = [analyze_red_object(comp, anchor_bbox) for comp in red_comps]
        if not all(3 <= len(comp) <= 6 for comp in red_comps):
            return False
        if not all(int(a["shift"]) > 0 for a in analyses):
            return False
        if not all(1 <= int(a["contact"]) <= 3 for a in analyses):
            return False
        if len({str(a["side"]) for a in analyses}) < 2:
            return False
        total = sum(int(a["contact"]) for a in analyses)
        if total > 9 or total <= len(red_comps):
            return False
        if not any(int(a["contact"]) > 1 for a in analyses):
            return False
        moved_sets = [set(a["shifted"]) for a in analyses]
        if any(_cells_touch(moved_sets[i], moved_sets[j], diag=True) for i in range(len(moved_sets)) for j in range(i + 1, len(moved_sets))):
            return False
        if any(ms & anchor for ms in moved_sets):
            return False
        red_sets = [set(map(tuple, comp.tolist())) for comp in red_comps]
        for i, a in enumerate(analyses):
            others = set().union(*(red_sets[:i] + red_sets[i + 1 :])) if len(red_sets) > 1 else set()
            for step in range(1, int(a["shift"]) + 1):
                mid = _translate_cells(red_sets[i], int(a["dr"]), int(a["dc"]), step)
                if step < int(a["shift"]) and any(any((nr, nc) in anchor for nr, nc in neighbors4(r, c)) for r, c in mid):
                    return False
                if mid & others:
                    return False
        initial_adjacent = 0
        for r, c in set().union(*red_sets):
            if any((nr, nc) in anchor for nr, nc in neighbors4(r, c)):
                initial_adjacent += 1
        if initial_adjacent != 0:
            return False
        out = main(g)
        if out.shape != (1, 9):
            return False
        if int((out == RED).sum()) != total:
            return False
        if not np.all(out[0, :total] == RED) or not np.all(out[0, total:] == BG):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
