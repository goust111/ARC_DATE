from collections import defaultdict, deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
ORANGE = 7
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid, accept):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or not accept(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if int(grid[nr, nc]) == color and accept(color):
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _hole_regions(cells):
    if not cells:
        return []
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    solid = np.zeros((hh, ww), dtype=bool)
    for r, c in cells:
        solid[r - r0, c - c0] = True

    blocked = np.zeros((hh + 2, ww + 2), dtype=bool)
    blocked[1:-1, 1:-1] = solid
    seen = np.zeros_like(blocked)

    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < hh + 2 and 0 <= nc < ww + 2:
                if not blocked[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

    holes = []
    for sr in range(1, hh + 1):
        for sc in range(1, ww + 1):
            if blocked[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            region = []
            while q:
                r, c = q.popleft()
                if 1 <= r <= hh and 1 <= c <= ww:
                    region.append((r0 + r - 1, c0 + c - 1))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < hh + 2 and 0 <= nc < ww + 2:
                        if not blocked[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            holes.append(region)
    return holes


def _center(cells):
    n = len(cells)
    return (
        sum(r for r, _ in cells) / n,
        sum(c for _, c in cells) / n,
    )


def _boundary(cells):
    cell_set = set(cells)
    out = []
    for r, c in cells:
        if any((nr, nc) not in cell_set for nr, nc in _neighbors4(r, c)):
            out.append((r, c))
    return out


def _nearest_boundary_pixel(cells, target):
    tr, tc = target
    return min(
        _boundary(cells),
        key=lambda p: ((p[0] - tr) ** 2 + (p[1] - tc) ** 2, p[0], p[1]),
    )


def _inclusive(a, b):
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _l_path(a, b):
    ar, ac = a
    br, bc = b
    cells = [(ar, c) for c in _inclusive(ac, bc)]
    cells.extend((r, bc) for r in _inclusive(ar, br))
    return cells


def _draw_l(out, a, b, color):
    h, w = out.shape
    for r, c in _l_path(a, b):
        if 0 <= r < h and 0 <= c < w and int(out[r, c]) == BLACK:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Group orange closed figures by hole count and connect each group with legend colors.

    Input structure:
        The background is black. The upper area contains non-orange colored
        solid square legend samples, each with 0, 1, 2, or 3 enclosed black
        holes. The lower area contains orange 4-connected closed figures whose
        enclosed black holes provide the same kind of count key.
    Rule:
        Count enclosed holes in each legend sample to decode hole-count to
        drawing-color mappings. Count enclosed holes in each orange figure and
        group figures by that count. Groups with multiple figures are ordered
        in reading order, then consecutive figures are connected using the
        decoded color by a horizontal-then-vertical L shaped path between the
        boundary pixel of each figure nearest the other figure's center. A
        group with exactly one figure is not connected; instead all enclosed
        hole pixels in that figure are filled with the decoded color.
    Key insight:
        The legend's own holes are the visible class labels: count them first,
        then use the matching legend color for orange figures with the same
        hole count.
    """
    out = np.array(grid, copy=True)

    legend = {}
    legend_comps = _components(grid, lambda v: v not in (BLACK, ORANGE))
    for comp in legend_comps:
        count = len(_hole_regions(comp["cells"]))
        if count not in legend:
            legend[count] = comp["color"]

    groups = defaultdict(list)
    for comp in _components(grid, lambda v: v == ORANGE):
        holes = _hole_regions(comp["cells"])
        count = len(holes)
        if count in legend:
            groups[count].append(
                {
                    "cells": comp["cells"],
                    "center": _center(comp["cells"]),
                    "holes": holes,
                }
            )

    for count, objs in groups.items():
        color = legend[count]
        if len(objs) == 1:
            for region in objs[0]["holes"]:
                for r, c in region:
                    if int(out[r, c]) == BLACK:
                        out[r, c] = color
            continue

        objs.sort(key=lambda item: (item["center"][0], item["center"][1]))
        for first, second in zip(objs, objs[1:]):
            a = _nearest_boundary_pixel(first["cells"], second["center"])
            b = _nearest_boundary_pixel(second["cells"], first["center"])
            _draw_l(out, a, b, color)

    return out


def _pick_holes(rng, mask, count):
    h, w = mask.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                candidates.append((r, c))
    rng.shuffle(candidates)
    holes = []
    for p in candidates:
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1 for q in holes):
            holes.append(p)
            if len(holes) == count:
                return holes
    return holes


def _paint_sample(grid, top, left, size, color, count, rng):
    grid[top : top + size, left : left + size] = color
    mask = np.ones((size, size), dtype=bool)
    for r, c in _pick_holes(rng, mask, count):
        grid[top + r, left + c] = BLACK


def _object_mask(rng, count):
    for _ in range(20):
        h = rng.randint(5, 6)
        w = rng.randint(5, 6)
        mask = np.ones((h, w), dtype=bool)
        for corner in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
            if rng.random() < 0.22:
                mask[corner] = False
        holes = _pick_holes(rng, mask, count)
        if len(holes) == count:
            for r, c in holes:
                mask[r, c] = False
            return mask
    mask = np.ones((5, 5), dtype=bool)
    for r, c in _pick_holes(rng, mask, count):
        mask[r, c] = False
    return mask


def _connected_slots(slots):
    slots = set(slots)
    start = next(iter(slots))
    q = deque([start])
    seen = {start}
    while q:
        s = q.popleft()
        r, c = divmod(s, 4)
        for nr, nc in _neighbors4(r, c):
            ns = nr * 4 + nc
            if 0 <= nr < 2 and 0 <= nc < 4 and ns in slots and ns not in seen:
                seen.add(ns)
                q.append(ns)
    return len(seen) == len(slots)


def _slot_candidates(available, count):
    return [
        combo
        for combo in combinations(sorted(available), count)
        if _connected_slots(combo)
    ]


def _assign_slots(rng, counts):
    for _ in range(100):
        available = set(range(8))
        group_slots = {}
        multi = [k for k, count in counts.items() if count > 1]
        rng.shuffle(multi)
        multi.sort(key=lambda k: -counts[k])
        ok = True
        for k in multi:
            candidates = _slot_candidates(available, counts[k])
            if not candidates:
                ok = False
                break
            chosen = rng.choice(candidates)
            group_slots[k] = list(chosen)
            available.difference_update(chosen)
        if not ok:
            continue
        singles = [k for k, count in counts.items() if count == 1]
        rng.shuffle(singles)
        free = list(available)
        rng.shuffle(free)
        if len(free) < len(singles):
            continue
        for k, slot in zip(singles, free):
            group_slots[k] = [slot]
        return group_slots
    return None


def _place_in_slot(rng, grid, lower_top, slot, k):
    h, w = grid.shape
    row, col = divmod(slot, 4)
    r0 = lower_top + row * (h - lower_top) // 2
    r1 = lower_top + (row + 1) * (h - lower_top) // 2
    c0 = col * w // 4
    c1 = (col + 1) * w // 4
    slot_h = r1 - r0
    slot_w = c1 - c0
    max_h = min(6, max(5, slot_h - 1))
    max_w = min(6, max(5, slot_w - 1))
    mask = _object_mask(rng, k)
    while mask.shape[0] > max_h or mask.shape[1] > max_w:
        mask = _object_mask(rng, k)
    mh, mw = mask.shape
    top_lo = r0 + 1 if slot_h - mh >= 2 else r0
    top_hi = r1 - mh - 1 if slot_h - mh >= 2 else r1 - mh
    left_lo = c0 + 1 if slot_w - mw >= 2 else c0
    left_hi = c1 - mw - 1 if slot_w - mw >= 2 else c1 - mw
    top = rng.randint(top_lo, top_hi)
    left = rng.randint(left_lo, left_hi)
    cells = []
    for rr in range(mh):
        for cc in range(mw):
            if mask[rr, cc]:
                grid[top + rr, left + cc] = ORANGE
                cells.append((top + rr, left + cc))
    return {"k": k, "cells": cells, "center": _center(cells)}


def _layout_clear(grid, placed):
    groups = defaultdict(list)
    for obj in placed:
        groups[obj["k"]].append(obj)
    used = set()
    for objs in groups.values():
        if len(objs) < 2:
            continue
        objs = sorted(objs, key=lambda item: (item["center"][0], item["center"][1]))
        for first, second in zip(objs, objs[1:]):
            a = _nearest_boundary_pixel(first["cells"], second["center"])
            b = _nearest_boundary_pixel(second["cells"], first["center"])
            endpoints = set(first["cells"]) | set(second["cells"])
            line = set()
            for cell in _l_path(a, b):
                r, c = cell
                if int(grid[r, c]) == ORANGE and cell not in endpoints:
                    return False
                if int(grid[r, c]) == BLACK:
                    line.add(cell)
            if used & line:
                return False
            used |= line
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(25, 30)
    w = rng.randint(27, 30)
    grid = np.zeros((h, w), dtype=int)

    colors = PALETTE[:]
    rng.shuffle(colors)
    color_for_count = {k: colors[k] for k in range(4)}

    size = 5
    gap = rng.choice([1, 2])
    total_w = 4 * size + 3 * gap
    top = rng.randint(1, 2)
    left = rng.randint(0, w - total_w)
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    for i, count in enumerate(order):
        x = left + i * (size + gap)
        _paint_sample(grid, top, x, size, color_for_count[count], count, rng)

    lower_top = top + size + rng.randint(2, 3)
    patterns = [
        [1, 1, 1, 2],
        [1, 1, 2, 2],
        [1, 2, 2, 2],
        [1, 1, 1, 3],
        [1, 1, 2, 3],
    ]
    pattern = rng.choice(patterns)[:]
    rng.shuffle(pattern)
    counts = {k: pattern[k] for k in range(4)}

    for _ in range(100):
        trial = grid.copy()
        slots = _assign_slots(rng, counts)
        if slots is None:
            continue
        placed = []
        items = []
        for k, group in slots.items():
            for slot in group:
                items.append((k, slot))
        rng.shuffle(items)
        for k, slot in items:
            placed.append(_place_in_slot(rng, trial, lower_top, slot, k))
        if _layout_clear(trial, placed):
            return trial

    return trial
