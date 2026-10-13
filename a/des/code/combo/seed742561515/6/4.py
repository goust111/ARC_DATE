from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_id = 0

    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or int(labels[r0, c0]) != 0:
                continue
            next_id += 1
            stack = [(int(r0), int(c0))]
            labels[r0, c0] = next_id
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and int(labels[rr, cc]) == 0:
                        labels[rr, cc] = next_id
                        stack.append((rr, cc))
            comps.append(comp)

    return labels, comps


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _holes_by_region(grid: np.ndarray, bg: int) -> tuple[list[int], dict[int, list[dict[str, object]]]]:
    h, w = grid.shape
    palette = [int(grid[r, 0]) for r in range(h) if int(grid[r, 0]) != bg]

    region_mask = grid != bg
    region_mask[:, 0] = False
    region_labels, region_comps = _label_components(region_mask)
    holes: dict[int, list[dict[str, object]]] = {idx: [] for idx in range(1, len(region_comps) + 1)}

    _bg_labels, bg_comps = _label_components(grid == bg)
    for comp in bg_comps:
        if _touches_edge(comp, h, w):
            continue
        neighbor_ids: set[int] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    rid = int(region_labels[rr, cc])
                    if rid:
                        neighbor_ids.add(rid)
        if len(neighbor_ids) != 1:
            continue
        rid = next(iter(neighbor_ids))
        anchor = min(comp)
        holes[rid].append({"area": len(comp), "anchor": anchor, "cells": comp})

    return palette, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed hole inside a foreground region using the left-edge color sequence.

    The non-background pixels in column 0 define an ordered palette from top to bottom.
    In every foreground connected component (excluding column 0), enclosed background
    holes are sorted by area and filled with the corresponding palette colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 2, "expected a left palette column plus drawing area"

    bg = _most_common_color(g)
    palette, holes = _holes_by_region(g, bg)
    assert palette, "expected at least one palette color on the left edge"

    out = g.copy()
    for region_id, region_holes in holes.items():
        ordered = sorted(
            region_holes,
            key=lambda hole: (
                int(hole["area"]),
                int(hole["anchor"][0]),
                int(hole["anchor"][1]),
            ),
        )
        assert len(ordered) <= len(palette), f"region {region_id} has more holes than palette colors"
        for idx, hole in enumerate(ordered):
            fill = int(palette[idx])
            for r, c in hole["cells"]:
                out[r, c] = fill

    return out


# --------------- generator ---------------

BG = 0
PALETTE_K_RANGE = (2, 5)
REGION_COUNT_RANGE = (1, 3)
PALETTE_GAP_RANGE = (2, 4)
REGION_GAP_RANGE = (2, 4)
SHELF_GAP_RANGE = (2, 4)
OUTER_MARGIN_RANGE = (1, 3)
MAX_GEN_ATTEMPTS = 300

HOLE_SHAPES_BY_AREA = {
    1: [(1, 1)],
    2: [(1, 2)],
    3: [(1, 3)],
    4: [(2, 2), (1, 4)],
    5: [(1, 5)],
    6: [(2, 3)],
    8: [(2, 4)],
    9: [(3, 3)],
    10: [(2, 5)],
}


def _choose_palette_and_foreground(k: int) -> tuple[int, list[int]]:
    fg = random.choice([c for c in range(1, 10)])
    palette_pool = [c for c in range(1, 10) if c != fg]
    palette = random.sample(palette_pool, k)
    return fg, palette


def _build_region(hole_count: int) -> dict[str, object]:
    areas = random.sample(sorted(HOLE_SHAPES_BY_AREA), hole_count)
    shapes = []
    for area in areas:
        hh, ww = random.choice(HOLE_SHAPES_BY_AREA[area])
        if hh != ww and random.random() < 0.5:
            hh, ww = ww, hh
        shapes.append({"area": int(area), "h": int(hh), "w": int(ww)})
    random.shuffle(shapes)

    row_count = 1 if hole_count == 1 else random.randint(1, min(2, hole_count))
    rows: list[list[dict[str, int]]] = [[] for _ in range(row_count)]
    for idx in range(row_count):
        rows[idx].append(shapes[idx])
    for shape in shapes[row_count:]:
        rows[random.randrange(row_count)].append(shape)
    for row in rows:
        random.shuffle(row)

    row_heights = [max(int(shape["h"]) for shape in row) for row in rows]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    row_gaps = [random.randint(1, 2) for _ in range(row_count - 1)]

    row_layouts = []
    for row in rows:
        left_pad = random.randint(1, 2)
        right_pad = random.randint(1, 2)
        gaps = [random.randint(1, 2) for _ in range(len(row) - 1)]
        width = left_pad + right_pad + sum(int(shape["w"]) for shape in row) + sum(gaps)
        row_layouts.append({"left_pad": left_pad, "gaps": gaps, "width": int(width)})

    height = top_margin + bottom_margin + sum(row_heights) + sum(row_gaps)
    width = max(int(layout["width"]) for layout in row_layouts)
    mask = np.ones((height, width), dtype=bool)
    holes = []

    cur_r = top_margin
    for row_idx, row in enumerate(rows):
        row_h = int(row_heights[row_idx])
        layout = row_layouts[row_idx]
        offset = random.randint(0, width - int(layout["width"]))
        cur_c = offset + int(layout["left_pad"])
        for shape_idx, shape in enumerate(row):
            hh = int(shape["h"])
            ww = int(shape["w"])
            dy = random.randint(0, row_h - hh)
            r0 = cur_r + dy
            c0 = cur_c
            mask[r0 : r0 + hh, c0 : c0 + ww] = False
            cells = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]
            holes.append({"area": int(shape["area"]), "anchor": (int(r0), int(c0)), "cells": cells})
            cur_c += ww
            if shape_idx < len(layout["gaps"]):
                cur_c += int(layout["gaps"][shape_idx])
        cur_r += row_h
        if row_idx < len(row_gaps):
            cur_r += int(row_gaps[row_idx])

    return {"mask": mask, "holes": holes, "h": int(height), "w": int(width)}


def _choose_hole_counts(region_count: int, k: int) -> list[int]:
    use_full_palette = random.random() < 0.5
    counts = []
    if use_full_palette:
        counts.append(k)
    else:
        counts.append(random.randint(2, max(2, k - 1)))

    for _ in range(region_count - 1):
        upper = k if random.random() < 0.6 else max(1, k - 1)
        counts.append(random.randint(1, upper))

    random.shuffle(counts)
    return counts


def _render_case(k: int, fg: int, palette: list[int], regions: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    region_count = len(regions)
    order = list(range(region_count))
    random.shuffle(order)

    if region_count == 1 or random.random() < 0.45:
        shelves = [order]
    else:
        cut = random.randint(1, region_count - 1)
        shelves = [order[:cut], order[cut:]]

    shelf_gaps = []
    shelf_heights = []
    shelf_widths = []
    for shelf in shelves:
        gaps = [random.randint(*REGION_GAP_RANGE) for _ in range(len(shelf) - 1)]
        shelf_gaps.append(gaps)
        shelf_heights.append(max(int(regions[idx]["h"]) for idx in shelf))
        shelf_widths.append(sum(int(regions[idx]["w"]) for idx in shelf) + sum(gaps))

    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)
    palette_gap = random.randint(*PALETTE_GAP_RANGE)
    between_shelves = [random.randint(*SHELF_GAP_RANGE) for _ in range(len(shelves) - 1)]

    inner_width = max(shelf_widths)
    height = top_margin + bottom_margin + sum(shelf_heights) + sum(between_shelves)
    width = 1 + palette_gap + inner_width + right_margin

    fg_area = sum(int(np.count_nonzero(region["mask"])) for region in regions)
    min_total = 2 * fg_area + k + 1
    if height * width < min_total:
        extra_rows = (min_total - height * width + width - 1) // width
        height += extra_rows
    height = max(height, k + 2) + random.randint(0, 2)
    width += random.randint(0, 2)

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    placements: list[dict[str, object]] = []

    palette_rows = sorted(random.sample(range(height), k))
    for row, color in zip(palette_rows, palette, strict=False):
        grid[int(row), 0] = int(color)
        expected[int(row), 0] = int(color)

    base_c = 1 + palette_gap
    cur_r = top_margin
    for shelf_idx, shelf in enumerate(shelves):
        shelf_h = int(shelf_heights[shelf_idx])
        shelf_w = int(shelf_widths[shelf_idx])
        shelf_offset = random.randint(0, inner_width - shelf_w)
        cur_c = base_c + shelf_offset
        for item_idx, region_idx in enumerate(shelf):
            region = regions[region_idx]
            mask = region["mask"]
            reg_h = int(region["h"])
            reg_w = int(region["w"])
            place_r = cur_r + random.randint(0, shelf_h - reg_h)
            place_c = cur_c

            rr, cc = np.where(mask)
            grid[place_r + rr, place_c + cc] = fg
            expected[place_r + rr, place_c + cc] = fg

            placed_holes = []
            for hole in region["holes"]:
                global_cells = [(place_r + r, place_c + c) for r, c in hole["cells"]]
                anchor_r = place_r + int(hole["anchor"][0])
                anchor_c = place_c + int(hole["anchor"][1])
                placed_holes.append(
                    {"area": int(hole["area"]), "anchor": (anchor_r, anchor_c), "cells": global_cells}
                )
            placements.append({"holes": placed_holes})

            cur_c += reg_w
            if item_idx < len(shelf_gaps[shelf_idx]):
                cur_c += int(shelf_gaps[shelf_idx][item_idx])

        cur_r += shelf_h
        if shelf_idx < len(between_shelves):
            cur_r += int(between_shelves[shelf_idx])

    for placed in placements:
        ordered = sorted(
            placed["holes"],
            key=lambda hole: (
                int(hole["area"]),
                int(hole["anchor"][0]),
                int(hole["anchor"][1]),
            ),
        )
        for idx, hole in enumerate(ordered):
            fill = int(palette[idx])
            for r, c in hole["cells"]:
                expected[r, c] = fill

    return grid, expected


def _is_valid(grid: np.ndarray, expected: np.ndarray, k: int, region_count: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[int(cnts.argmax())]) != BG:
        return False

    palette_rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != BG]
    palette_colors = [int(grid[r, 0]) for r in palette_rows]
    if len(palette_rows) != k or len(set(palette_colors)) != k:
        return False

    region_mask = grid != BG
    region_mask[:, 0] = False
    _labels, comps = _label_components(region_mask)
    if len(comps) != region_count:
        return False

    _palette, holes = _holes_by_region(grid, BG)
    if not any(len(region_holes) >= 2 for region_holes in holes.values()):
        return False
    if any(len(region_holes) > k for region_holes in holes.values()):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        k = random.randint(*PALETTE_K_RANGE)
        region_count = random.randint(*REGION_COUNT_RANGE)
        hole_counts = _choose_hole_counts(region_count, k)
        fg, palette = _choose_palette_and_foreground(k)
        regions = [_build_region(hole_count) for hole_count in hole_counts]
        grid, expected = _render_case(k, fg, palette, regions)
        if _is_valid(grid, expected, k, region_count):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
