import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator tuning.
GEN_ATTEMPTS = 400
MAX_GROUPS = 3
MAX_TOTAL_OBJECTS = 8
OUTER_MARGIN_RANGE = (1, 3)
INTRA_GROUP_GAPS = (2, 3)
INTER_GROUP_GAP_RANGE = (2, 4)
GRID_MIN_SIZE = 10
GRID_MAX_SIZE = 30


def _normalize_cells(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _cells_connected(cells):
    cells = set(cells)
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _hole_free(cells, h, w):
    filled = set(cells)
    start = (-1, -1)
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (-1 <= nr <= h and -1 <= nc <= w):
                continue
            if (nr, nc) in seen:
                continue
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in filled:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    for r in range(h):
        for c in range(w):
            if (r, c) not in filled and (r, c) not in seen:
                return False
    return True


def _shape_families(h, w):
    families = []

    families.append({(r, c) for r in range(h) for c in range(w)})
    families.extend(
        [
            {(0, c) for c in range(w)} | {(r, 0) for r in range(h)},
            {(0, c) for c in range(w)} | {(r, w - 1) for r in range(h)},
            {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)},
            {(h - 1, c) for c in range(w)} | {(r, w - 1) for r in range(h)},
        ]
    )

    if h >= 3 and w >= 3:
        mr = h // 2
        mc = w // 2
        families.extend(
            [
                {(0, c) for c in range(w)} | {(r, mc) for r in range(h)},
                {(h - 1, c) for c in range(w)} | {(r, mc) for r in range(h)},
                {(mr, c) for c in range(w)} | {(r, 0) for r in range(h)},
                {(mr, c) for c in range(w)} | {(r, w - 1) for r in range(h)},
            ]
        )

        for flip_r in (False, True):
            for flip_c in (False, True):
                cells = set()
                for r in range(h):
                    rr = h - 1 - r if flip_r else r
                    extent = (rr * (w - 1)) // (h - 1)
                    for c in range(extent + 1):
                        cc = w - 1 - c if flip_c else c
                        cells.add((r, cc))
                families.append(cells)

        band_h = max(1, h // 2)
        band_w = max(1, w // 2)
        families.extend(
            [
                {(r, c) for r in range(h) for c in range(w) if r < band_h or c < band_w},
                {(r, c) for r in range(h) for c in range(w) if r < band_h or c >= w - band_w},
                {(r, c) for r in range(h) for c in range(w) if r >= h - band_h or c < band_w},
                {(r, c) for r in range(h) for c in range(w) if r >= h - band_h or c >= w - band_w},
            ]
        )

    if h % 2 == 1 and w % 2 == 1 and h >= 3 and w >= 3:
        mr = h // 2
        mc = w // 2
        families.append({(mr, c) for c in range(w)} | {(r, mc) for r in range(h)})

    uniq = []
    seen = set()
    for cells in families:
        key = _normalize_cells(cells)
        if key in seen:
            continue
        if len(key) < 3:
            continue
        if not _cells_connected(key):
            continue
        if not _hole_free(key, h, w):
            continue
        if max(r for r, _ in key) + 1 != h or max(c for _, c in key) + 1 != w:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _build_shape_catalog():
    shapes = []
    by_parity = defaultdict(list)
    for h in range(2, 5):
        for w in range(2, 5):
            for coords in _shape_families(h, w):
                info = {
                    "coords": coords,
                    "h": h,
                    "w": w,
                    "area": len(coords),
                    "parity": (h % 2, w % 2),
                }
                shapes.append(info)
                by_parity[info["parity"]].append(info)
    shapes.sort(key=lambda s: (s["area"], s["h"], s["w"], s["coords"]))
    for parity in by_parity:
        by_parity[parity].sort(key=lambda s: (s["area"], s["h"], s["w"], s["coords"]))
    return shapes, dict(by_parity)


SHAPES, SHAPES_BY_PARITY = _build_shape_catalog()


class _DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _find_objects(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "coord_set": set(coords),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                    "shape": tuple(sorted((rr - r0, cc - c0) for rr, cc in coords)),
                }
            )
    return objects


def _intervals_overlap(a0, a1, b0, b1):
    return max(a0, b0) <= min(a1, b1)


def _projection_groups(objects):
    dsu = _DSU(len(objects))
    for i in range(len(objects)):
        r0a, c0a, r1a, c1a = objects[i]["bbox"]
        for j in range(i + 1, len(objects)):
            r0b, c0b, r1b, c1b = objects[j]["bbox"]
            same_rows = _intervals_overlap(r0a, r1a, r0b, r1b)
            same_cols = _intervals_overlap(c0a, c1a, c0b, c1b)
            if same_rows or same_cols:
                dsu.union(i, j)
    groups = defaultdict(list)
    for i, obj in enumerate(objects):
        groups[dsu.find(i)].append(obj)
    ordered = []
    for members in groups.values():
        members.sort(key=lambda obj: obj["bbox"])
        ordered.append(members)
    ordered.sort(key=lambda grp: grp[0]["bbox"])
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by row/column projection overlap, then replace each object
    by the largest shape in its group while preserving each object's color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _find_objects(g)
    for group in _projection_groups(objects):
        template = max(group, key=lambda obj: obj["area"])
        th = template["bbox"][2] - template["bbox"][0] + 1
        tw = template["bbox"][3] - template["bbox"][1] + 1
        rel = template["shape"]

        for obj in group:
            if obj is template:
                continue
            rows = obj["coords"][:, 0]
            cols = obj["coords"][:, 1]
            out[rows, cols] = BG

        for obj in group:
            if obj is template:
                continue
            r0, c0, r1, c1 = obj["bbox"]
            nr0_num = r0 + r1 - (th - 1)
            nc0_num = c0 + c1 - (tw - 1)
            assert nr0_num % 2 == 0 and nc0_num % 2 == 0
            nr0 = nr0_num // 2
            nc0 = nc0_num // 2
            for rr, cc in rel:
                out[nr0 + rr, nc0 + cc] = obj["color"]
    return out


def _smaller_shapes(template):
    parity = template["parity"]
    return [
        shape
        for shape in SHAPES_BY_PARITY[parity]
        if shape["area"] < template["area"] and shape["h"] <= template["h"] and shape["w"] <= template["w"]
    ]


def _choose_group_specs(rng):
    target_groups = rng.randint(1, MAX_GROUPS)
    specs = []
    total = 0

    first_kind = rng.choice(("horizontal", "vertical", "mixed"))
    if first_kind == "mixed":
        first_count = rng.randint(3, min(4, MAX_TOTAL_OBJECTS))
    else:
        first_count = rng.randint(2, min(3, MAX_TOTAL_OBJECTS))
    specs.append((first_kind, first_count))
    total += first_count

    for _ in range(target_groups - 1):
        remaining = MAX_TOTAL_OBJECTS - total
        if remaining <= 0:
            break
        options = ["isolated"]
        if remaining >= 2:
            options.extend(["horizontal", "vertical"])
        if remaining >= 3:
            options.append("mixed")
        kind = rng.choice(options)
        if kind == "isolated":
            count = 1
        elif kind == "mixed":
            count = rng.randint(3, min(4, remaining))
        else:
            count = rng.randint(2, min(3, remaining))
        specs.append((kind, count))
        total += count
    return specs


def _choose_template(rng, kind):
    candidates = []
    for shape in SHAPES:
        if kind != "isolated" and not _smaller_shapes(shape):
            continue
        if kind == "mixed" and max(shape["h"], shape["w"]) > 4:
            continue
        candidates.append(shape)
    return rng.choice(candidates)


def _group_slots(kind, count, template_h, template_w, gap, rng):
    step_r = template_h + gap
    step_c = template_w + gap
    if kind == "horizontal":
        slots = [(0, i * step_c) for i in range(count)]
    elif kind == "vertical":
        slots = [(i * step_r, 0) for i in range(count)]
    elif kind == "mixed":
        base = [(0, 0), (step_r, 0), (step_r, step_c), (2 * step_r, step_c)]
        slots = base[:count]
    else:
        slots = [(0, 0)]

    block_h = max(r for r, _ in slots) + template_h
    block_w = max(c for _, c in slots) + template_w

    if rng.random() < 0.5:
        slots = [(block_h - template_h - r, c) for r, c in slots]
    if rng.random() < 0.5:
        slots = [(r, block_w - template_w - c) for r, c in slots]

    slots.sort()
    return slots, block_h, block_w


def _shape_cells(shape, top, left):
    return {(top + r, left + c) for r, c in shape["coords"]}


def _objects_have_bg_gap(objects):
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            for r, c in objects[i]["coord_set"]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (r + dr, c + dc) in objects[j]["coord_set"]:
                            return False
    return True


def _validate_generated(grid, expected_group_sizes):
    objects = _find_objects(grid)
    if len(objects) != sum(expected_group_sizes):
        return False
    if not _objects_have_bg_gap(objects):
        return False

    groups = _projection_groups(objects)
    actual_sizes = sorted(len(group) for group in groups)
    if actual_sizes != sorted(expected_group_sizes):
        return False
    if max(actual_sizes) < 2:
        return False

    for group in groups:
        areas = [obj["area"] for obj in group]
        if len(group) > 1 and areas.count(max(areas)) != 1:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False

    out_objects = _find_objects(out)
    if len(out_objects) != len(objects):
        return False
    if not _objects_have_bg_gap(out_objects):
        return False
    return True


def _gen_attempt(rng):
    group_specs = _choose_group_specs(rng)
    total_objects = sum(count for _, count in group_specs)
    colors = rng.sample(PALETTE, total_objects)
    color_idx = 0

    placements = []
    row_cursor = 0
    col_cursor = 0
    expected_group_sizes = []

    for kind, count in group_specs:
        template = _choose_template(rng, kind)
        smaller = _smaller_shapes(template)
        gap = rng.choice(INTRA_GROUP_GAPS)
        slots, block_h, block_w = _group_slots(kind, count, template["h"], template["w"], gap, rng)
        template_idx = rng.randrange(count)

        chosen_shapes = []
        if count == 1:
            chosen_shapes = [template]
        else:
            pool = smaller[:]
            rng.shuffle(pool)
            for idx in range(count):
                if idx == template_idx:
                    chosen_shapes.append(template)
                else:
                    if pool:
                        chosen_shapes.append(pool.pop())
                    else:
                        chosen_shapes.append(rng.choice(smaller))

        for idx, ((slot_r, slot_c), shape) in enumerate(zip(slots, chosen_shapes)):
            top = row_cursor + slot_r + (template["h"] - shape["h"]) // 2
            left = col_cursor + slot_c + (template["w"] - shape["w"]) // 2
            placements.append(
                {
                    "color": colors[color_idx],
                    "cells": _shape_cells(shape, top, left),
                    "bbox": (top, left, top + shape["h"] - 1, left + shape["w"] - 1),
                }
            )
            color_idx += 1

        expected_group_sizes.append(count)
        row_cursor += block_h + rng.randint(*INTER_GROUP_GAP_RANGE)
        col_cursor += block_w + rng.randint(*INTER_GROUP_GAP_RANGE)

    max_r = max(r for item in placements for r, _ in item["cells"])
    max_c = max(c for item in placements for _, c in item["cells"])
    top_margin = rng.randint(*OUTER_MARGIN_RANGE)
    left_margin = rng.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = rng.randint(*OUTER_MARGIN_RANGE)
    right_margin = rng.randint(*OUTER_MARGIN_RANGE)
    h = max_r + 1 + top_margin + bottom_margin
    w = max_c + 1 + left_margin + right_margin
    if not (GRID_MIN_SIZE <= h <= GRID_MAX_SIZE and GRID_MIN_SIZE <= w <= GRID_MAX_SIZE):
        return None

    grid = np.zeros((h, w), dtype=int)
    for item in placements:
        shifted = {(r + top_margin, c + left_margin) for r, c in item["cells"]}
        for r, c in shifted:
            if grid[r, c] != BG:
                return None
            grid[r, c] = item["color"]

    if not _validate_generated(grid, expected_group_sizes):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random projection-grouping puzzle input."""
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
