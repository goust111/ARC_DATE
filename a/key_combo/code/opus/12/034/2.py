import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
SEP_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 7)
MAX_ATTEMPTS = 400
OVERLAP_ROW_PROB = 0.6
SAME_AREA_PROB = 0.5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _anchor_row(coords):
    row_counts = Counter(r for r, _ in coords)
    best_count = max(row_counts.values())
    return min(r for r, cnt in row_counts.items() if cnt == best_count)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected one full-height monochrome separator column"
    return candidates[0]


def _components_left_of_separator(grid: np.ndarray, sep_col: int, bg: int, sep_color: int):
    h, _ = grid.shape
    seen = set()
    comps = []
    for r in range(h):
        for c in range(sep_col):
            color = int(grid[r, c])
            if color in (bg, sep_color) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < sep_col and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                    "row": _anchor_row(coords),
                    "left": c0,
                    "dist": sep_col - c0,
                }
            )
    return comps


def _render_output(shape: tuple[int, int], sep_col: int, sep_color: int, comps):
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[:, sep_col] = sep_color

    for comp in sorted(comps, key=lambda item: (item["dist"], item["row"], item["left"], item["color"])):
        row = int(comp["row"])
        length = int(comp["area"])
        color = int(comp["color"])

        left_start = max(0, sep_col - length)
        for c in range(left_start, sep_col):
            if out[row, c] == BG:
                out[row, c] = color

        start = sep_col + 1
        step = 2 * length
        while start < w:
            end = min(start + length, w)
            for c in range(start, end):
                if out[row, c] == BG:
                    out[row, c] = color
            start += step

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each irregular region, straighten it into a bar, then extend that bar periodically past a separator.

    Input structure:
    - The grid has black background and one full-height monochrome separator column.
    - All other foreground objects lie on one side of the separator as 4-connected monochrome regions.
    - The opposite side of the separator is blank target space.

    Rule:
    - For each foreground region, keep its color and count its pixels to get a bar length N.
    - Pick the region's representative row as the row containing the most pixels from that region.
    - Erase the original shapes, redraw each as a right-aligned 1xN bar flush against the separator,
      and then repeat 1xN segments on the far side in an N-on / N-off rhythm.
    - If two regions would claim the same cell while straightening or extending, the region whose
      original left edge was closer to the separator keeps that cell.

    Key insight:
    - The shape geometry itself is discarded: area becomes bar length and period, while proximity to
      the separator resolves collisions between competing periodic fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    sep_col, sep_color = _find_separator(g, bg)
    comps = _components_left_of_separator(g, sep_col, bg, sep_color)
    assert comps, "expected at least one foreground region"
    return _render_output(g.shape, sep_col, sep_color, comps)


SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1)),
]


def _normalize_shape(coords):
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    anchor = _anchor_row(rel)
    return {
        "coords": rel,
        "area": len(rel),
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "anchor_offset": anchor,
    }


SHAPE_BANK = [_normalize_shape(shape) for shape in SHAPES]


def _shape_cells(shape, anchor_row: int, left_col: int):
    top = anchor_row - shape["anchor_offset"]
    return [(top + r, left_col + c) for r, c in shape["coords"]]


def _expanded_bbox(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r0 - 1, c0 - 1, r1 + 1, c1 + 1


def _boxes_intersect(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_shapes(rng: random.Random, count: int, max_len: int, same_area_branch: bool):
    pool = [shape for shape in SHAPE_BANK if shape["area"] <= max_len]
    by_area = defaultdict(list)
    for shape in pool:
        by_area[shape["area"]].append(shape)

    chosen = []
    if same_area_branch:
        area_options = [area for area, items in by_area.items() if len(items) >= 2]
        if area_options:
            area = rng.choice(area_options)
            chosen.extend(rng.sample(by_area[area], 2))

    remaining_pool = [shape for shape in pool if shape not in chosen]
    while len(chosen) < count:
        chosen.append(rng.choice(remaining_pool))

    rng.shuffle(chosen)
    if len({shape["area"] for shape in chosen}) < 2:
        return None
    return chosen


def _choose_rows(rng: random.Random, h: int, shapes, overlap_branch: bool):
    valid = []
    for shape in shapes:
        lo = shape["anchor_offset"]
        hi = h - 1 - (shape["h"] - 1 - shape["anchor_offset"])
        valid.append(list(range(lo, hi + 1)))
    if any(not rows for rows in valid):
        return None, None

    overlap_pair = None
    rows = [None] * len(shapes)
    if overlap_branch and len(shapes) >= 2:
        pair = rng.sample(range(len(shapes)), 2)
        common = sorted(set(valid[pair[0]]) & set(valid[pair[1]]))
        if common:
            overlap_pair = tuple(pair)
            shared = rng.choice(common)
            rows[pair[0]] = shared
            rows[pair[1]] = shared

    used_unique = {row for row in rows if row is not None}
    remaining = [idx for idx, row in enumerate(rows) if row is None]
    rng.shuffle(remaining)
    for idx in remaining:
        candidates = valid[idx][:]
        rng.shuffle(candidates)
        placed = False
        for row in candidates:
            if row not in used_unique:
                rows[idx] = row
                used_unique.add(row)
                placed = True
                break
        if not placed:
            return None, None
    return rows, overlap_pair


def _place_shapes(rng: random.Random, h: int, sep_col: int, shapes, rows):
    placements = []
    used_left_edges = set()
    available_cols = list(range(sep_col))
    order = list(range(len(shapes)))
    rng.shuffle(order)
    for idx in order:
        shape = shapes[idx]
        candidates = []
        for left in available_cols:
            if left in used_left_edges or left + shape["w"] > sep_col:
                continue
            cells = _shape_cells(shape, rows[idx], left)
            if any(not (0 <= r < h and 0 <= c < sep_col) for r, c in cells):
                continue
            box = _expanded_bbox(cells)
            if any(_boxes_intersect(box, placed["box"]) for placed in placements):
                continue
            candidates.append((left, cells, box))
        if not candidates:
            return None
        left, cells, box = rng.choice(candidates)
        placements.append({"idx": idx, "left": left, "cells": cells, "box": box})
        used_left_edges.add(left)
    placements.sort(key=lambda item: item["idx"])
    return placements


def _claims(shape: tuple[int, int], sep_col: int, comps):
    h, w = shape
    claim_counts = np.zeros((h, w), dtype=int)
    for comp in comps:
        row = int(comp["row"])
        length = int(comp["area"])
        left_start = max(0, sep_col - length)
        claim_counts[row, left_start:sep_col] += 1
        start = sep_col + 1
        while start < w:
            claim_counts[row, start : min(start + length, w)] += 1
            start += 2 * length
    return claim_counts


def _build_example(rng: random.Random):
    h = rng.randint(10, 14)
    w = rng.randint(20, 26)
    sep_col = rng.randint(7, min(10, w - 8))
    right_width = w - sep_col - 1
    count = rng.choice([3, 4])
    overlap_branch = rng.random() < OVERLAP_ROW_PROB
    same_area_branch = rng.random() < SAME_AREA_PROB

    shapes = _choose_shapes(rng, count, min(sep_col, right_width), same_area_branch)
    if shapes is None:
        return None

    rows, overlap_pair = _choose_rows(rng, h, shapes, overlap_branch)
    if rows is None:
        return None

    placements = _place_shapes(rng, h, sep_col, shapes, rows)
    if placements is None:
        return None

    colors = rng.sample(OBJECT_COLORS, count)
    grid = np.zeros((h, w), dtype=int)
    grid[:, sep_col] = SEP_COLOR

    comps = []
    for idx, place in enumerate(placements):
        color = colors[idx]
        for r, c in place["cells"]:
            grid[r, c] = color
        coords = place["cells"]
        r0, c0, r1, c1 = _bbox(coords)
        comps.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": len(coords),
                "row": _anchor_row(coords),
                "left": c0,
                "dist": sep_col - c0,
                "shape_id": tuple(sorted((r - r0, c - c0) for r, c in coords)),
            }
        )

    if not any((comp["bbox"][2] - comp["bbox"][0] + 1) > 1 for comp in comps):
        return None

    if len({comp["left"] for comp in comps}) != len(comps):
        return None

    if overlap_branch and overlap_pair is not None:
        pair_rows = [comps[overlap_pair[0]]["row"], comps[overlap_pair[1]]["row"]]
        if pair_rows[0] != pair_rows[1]:
            return None

    if same_area_branch:
        same_area_pairs = [
            (i, j)
            for i in range(len(comps))
            for j in range(i + 1, len(comps))
            if comps[i]["area"] == comps[j]["area"] and comps[i]["shape_id"] != comps[j]["shape_id"]
        ]
        if not same_area_pairs:
            return None

    if np.any(grid[:, sep_col + 1 :] != BG):
        return None

    claim_counts = _claims(grid.shape, sep_col, comps)
    if overlap_branch and not np.any(claim_counts > 1):
        return None

    expected = _render_output(grid.shape, sep_col, SEP_COLOR, comps)
    parsed = _components_left_of_separator(grid, sep_col, BG, SEP_COLOR)
    if len(parsed) != count:
        return None
    if sorted(comp["area"] for comp in parsed) != sorted(comp["area"] for comp in comps):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_example(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
