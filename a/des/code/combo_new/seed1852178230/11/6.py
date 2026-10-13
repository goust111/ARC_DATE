from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
BULB = 4  # Light stand-in for the white bulb described by the puzzle.
WIRE = 5
KEEP = 7
ROTATE = 8

SOURCE_COLORS = (BLUE, RED, GREEN)
PATH_COLORS = (WIRE, KEEP, ROTATE)
COLOR_CYCLE = {RED: GREEN, GREEN: BLUE, BLUE: RED}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MAX_GENERATION_ATTEMPTS = 300
MIN_REGION = 10
OUTER_MARGIN = 1
REGION_GAP = 1


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _solid_blocks(grid: np.ndarray, color: int) -> list[dict]:
    blocks: list[dict] = []
    for comp in _components(grid == color):
        if len(comp) != 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        if not bool((grid[r0 : r1 + 1, c0 : c1 + 1] == color).all()):
            continue
        blocks.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cells": set(comp),
            }
        )
    return blocks


def _lookup_from_blocks(blocks: list[dict]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, block in enumerate(blocks):
        for cell in block["cells"]:
            lookup[cell] = idx
    return lookup


def _adjacent_ids(
    cell: tuple[int, int],
    lookup: dict[tuple[int, int], int],
    height: int,
    width: int,
) -> set[int]:
    ids: set[int] = set()
    r, c = cell
    for nr, nc in _neighbors4(r, c, height, width):
        idx = lookup.get((nr, nc))
        if idx is not None:
            ids.add(int(idx))
    return ids


def _ordered_path(
    start: tuple[int, int],
    end: tuple[int, int],
    cells: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        nxts = [nb for nb in _neighbors4(cur[0], cur[1], height, width) if nb in cells and nb != prev]
        assert len(nxts) == 1, "wire components must be simple paths"
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each bulb by tracing its wire from the colored source and applying the embedded filters in order.

    Input structure:
    - The background is black.
    - Each circuit contains a 2x2 solid red, green, or blue source block, a one-pixel orthogonal gray wire, optional orange or azure single-pixel filters inside the wire, and a 2x2 light bulb block at the other end.
    - Every wire is a separate simple path that touches exactly one source and one bulb.

    Rule:
    - For each wire, start with the source color and walk from the source-adjacent wire endpoint to the bulb-adjacent endpoint.
    - Gray wire pixels do nothing, orange filter pixels keep the current color, and azure filter pixels rotate the color one step along red to green to blue to red.
    - Replace the bulb block with the final color while leaving the sources, wires, and filters unchanged.

    Key insight:
    - The answer depends on both connectivity and traversal order: the solver must pair each source with its bulb through the wire topology, then accumulate filter effects along that ordered path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = int(Counter(map(int, g.reshape(-1))).most_common(1)[0][0])
    assert bg == BG, "expected black background"

    source_blocks: list[dict] = []
    for color in SOURCE_COLORS:
        source_blocks.extend(_solid_blocks(g, color))
    bulb_blocks = _solid_blocks(g, BULB)
    path_components = _components(np.isin(g, PATH_COLORS))

    assert source_blocks, "expected source blocks"
    assert bulb_blocks, "expected bulb blocks"
    assert len(source_blocks) == len(bulb_blocks) == len(path_components), "every circuit must have one source, one bulb, and one wire"

    source_lookup = _lookup_from_blocks(source_blocks)
    bulb_lookup = _lookup_from_blocks(bulb_blocks)

    out = g.copy()
    used_sources: set[int] = set()
    used_bulbs: set[int] = set()

    for comp in path_components:
        cell_set = set(comp)
        endpoints = [
            cell
            for cell in comp
            if sum((nb in cell_set) for nb in _neighbors4(cell[0], cell[1], height, width)) == 1
        ]
        assert len(endpoints) == 2, "each wire must have exactly two endpoints"

        start_cell: tuple[int, int] | None = None
        end_cell: tuple[int, int] | None = None
        source_idx: int | None = None
        bulb_idx: int | None = None

        for endpoint in endpoints:
            src_ids = _adjacent_ids(endpoint, source_lookup, height, width)
            bulb_ids = _adjacent_ids(endpoint, bulb_lookup, height, width)
            if len(src_ids) == 1 and not bulb_ids:
                start_cell = endpoint
                source_idx = next(iter(src_ids))
            elif len(bulb_ids) == 1 and not src_ids:
                end_cell = endpoint
                bulb_idx = next(iter(bulb_ids))

        assert start_cell is not None and end_cell is not None, "wire endpoints must touch a source and a bulb"
        assert source_idx is not None and bulb_idx is not None
        assert source_idx not in used_sources and bulb_idx not in used_bulbs, "circuits must be disjoint"
        used_sources.add(int(source_idx))
        used_bulbs.add(int(bulb_idx))

        path = _ordered_path(start_cell, end_cell, cell_set, height, width)
        final_color = int(source_blocks[source_idx]["color"])
        for r, c in path:
            if int(g[r, c]) == ROTATE:
                final_color = int(COLOR_CYCLE[final_color])

        r0, c0, r1, c1 = map(int, bulb_blocks[bulb_idx]["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(final_color)

    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be orthogonal")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx in range(len(points) - 1):
        seg = _segment_cells(points[idx], points[idx + 1])
        if idx:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _pick_mid(lo: int, hi: int, exclude: set[int]) -> int:
    candidates = [v for v in range(lo, hi + 1) if v not in exclude]
    if candidates:
        return int(random.choice(candidates))
    return int(random.randint(lo, hi))


def _build_horizontal_ltr(
    start: tuple[int, int],
    end: tuple[int, int],
    row_lo: int,
    row_hi: int,
    prefer_long: bool,
) -> list[tuple[int, int]] | None:
    rs, cs = start
    re, ce = end
    assert cs < ce

    modes = ["detour", "dogleg", "straight"]
    if not prefer_long:
        random.shuffle(modes)

    for mode in modes:
        if mode == "straight":
            if rs != re:
                continue
            points = [start, end]
        elif mode == "dogleg":
            if ce - cs < 2:
                continue
            pivot = random.randint(cs + 1, ce - 1)
            points = [start, (rs, pivot), (re, pivot), end]
        else:
            if ce - cs < 3:
                continue
            col1 = random.randint(cs + 1, ce - 2)
            col2 = random.randint(col1 + 1, ce - 1)
            mid_row = _pick_mid(row_lo, row_hi, {rs, re})
            points = [start, (rs, col1), (mid_row, col1), (mid_row, col2), (re, col2), end]

        cells = _polyline_cells(points)
        if len(cells) == len(set(cells)):
            return cells
    return None


def _build_horizontal_path(
    start: tuple[int, int],
    end: tuple[int, int],
    row_lo: int,
    row_hi: int,
    prefer_long: bool,
) -> list[tuple[int, int]] | None:
    if start[1] < end[1]:
        return _build_horizontal_ltr(start, end, row_lo, row_hi, prefer_long)
    if start[1] > end[1]:
        cells = _build_horizontal_ltr(end, start, row_lo, row_hi, prefer_long)
        return None if cells is None else list(reversed(cells))
    return None


def _build_vertical_ttb(
    start: tuple[int, int],
    end: tuple[int, int],
    col_lo: int,
    col_hi: int,
    prefer_long: bool,
) -> list[tuple[int, int]] | None:
    rs, cs = start
    re, ce = end
    assert rs < re

    modes = ["detour", "dogleg", "straight"]
    if not prefer_long:
        random.shuffle(modes)

    for mode in modes:
        if mode == "straight":
            if cs != ce:
                continue
            points = [start, end]
        elif mode == "dogleg":
            if re - rs < 2:
                continue
            pivot = random.randint(rs + 1, re - 1)
            points = [start, (pivot, cs), (pivot, ce), end]
        else:
            if re - rs < 3:
                continue
            row1 = random.randint(rs + 1, re - 2)
            row2 = random.randint(row1 + 1, re - 1)
            mid_col = _pick_mid(col_lo, col_hi, {cs, ce})
            points = [start, (row1, cs), (row1, mid_col), (row2, mid_col), (row2, ce), end]

        cells = _polyline_cells(points)
        if len(cells) == len(set(cells)):
            return cells
    return None


def _build_vertical_path(
    start: tuple[int, int],
    end: tuple[int, int],
    col_lo: int,
    col_hi: int,
    prefer_long: bool,
) -> list[tuple[int, int]] | None:
    if start[0] < end[0]:
        return _build_vertical_ttb(start, end, col_lo, col_hi, prefer_long)
    if start[0] > end[0]:
        cells = _build_vertical_ttb(end, start, col_lo, col_hi, prefer_long)
        return None if cells is None else list(reversed(cells))
    return None


def _split_spans(total: int, parts: int, min_size: int) -> list[tuple[int, int]]:
    usable = total - 2 * OUTER_MARGIN - (parts - 1) * REGION_GAP
    assert usable >= parts * min_size
    sizes = [min_size] * parts
    extra = usable - parts * min_size
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1

    spans: list[tuple[int, int]] = []
    cur = OUTER_MARGIN
    for size in sizes:
        spans.append((int(cur), int(cur + size)))
        cur += size + REGION_GAP
    return spans


def _filter_sequences(count: int) -> list[tuple[int, ...]]:
    counts = list(range(3))
    random.shuffle(counts)
    chosen = counts[: min(3, count)]
    while len(chosen) < count:
        chosen.append(random.choice([0, 1, 2]))
    random.shuffle(chosen)

    seqs: list[tuple[int, ...]] = []
    for n_filters in chosen:
        if n_filters == 0:
            seqs.append(())
        elif n_filters == 1:
            seqs.append((random.choice([KEEP, ROTATE]),))
        else:
            seqs.append(tuple(random.choice([KEEP, ROTATE]) for _ in range(2)))
    random.shuffle(seqs)
    return seqs


def _paint_block(grid: np.ndarray, top: int, left: int, color: int) -> set[tuple[int, int]]:
    grid[top : top + 2, left : left + 2] = int(color)
    return {(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)}


def _draw_circuit(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    source_color: int,
    filters: tuple[int, ...],
    orientation: str,
) -> bool:
    r0, r1, c0, c1 = map(int, region)

    for _ in range(80):
        if orientation == "h":
            left_slot = c0 + random.choice([1, 2])
            right_slot = c1 - 3 - random.choice([0, 1])
            if right_slot - left_slot < 4:
                continue
            src_row = random.randint(r0 + 1, r1 - 3)
            bulb_row = random.randint(r0 + 1, r1 - 3)
            src_port_row = src_row + random.randint(0, 1)
            bulb_port_row = bulb_row + random.randint(0, 1)
            if random.choice([True, False]):
                src_top, src_left = src_row, left_slot
                bulb_top, bulb_left = bulb_row, right_slot
                start = (src_port_row, src_left + 2)
                end = (bulb_port_row, bulb_left - 1)
            else:
                src_top, src_left = src_row, right_slot
                bulb_top, bulb_left = bulb_row, left_slot
                start = (src_port_row, src_left - 1)
                end = (bulb_port_row, bulb_left + 2)
            path = _build_horizontal_path(start, end, r0, r1 - 1, len(filters) == 2)
        else:
            top_slot = r0 + random.choice([1, 2])
            bottom_slot = r1 - 3 - random.choice([0, 1])
            if bottom_slot - top_slot < 4:
                continue
            src_col = random.randint(c0 + 1, c1 - 3)
            bulb_col = random.randint(c0 + 1, c1 - 3)
            src_port_col = src_col + random.randint(0, 1)
            bulb_port_col = bulb_col + random.randint(0, 1)
            if random.choice([True, False]):
                src_top, src_left = top_slot, src_col
                bulb_top, bulb_left = bottom_slot, bulb_col
                start = (src_top + 2, src_port_col)
                end = (bulb_top - 1, bulb_port_col)
            else:
                src_top, src_left = bottom_slot, src_col
                bulb_top, bulb_left = top_slot, bulb_col
                start = (src_top - 1, src_port_col)
                end = (bulb_top + 2, bulb_port_col)
            path = _build_vertical_path(start, end, c0, c1 - 1, len(filters) == 2)

        if path is None or len(path) - 2 < len(filters):
            continue

        source_cells = {(src_top, src_left), (src_top, src_left + 1), (src_top + 1, src_left), (src_top + 1, src_left + 1)}
        bulb_cells = {(bulb_top, bulb_left), (bulb_top, bulb_left + 1), (bulb_top + 1, bulb_left), (bulb_top + 1, bulb_left + 1)}

        if source_cells & bulb_cells:
            continue
        if any(grid[r, c] != BG for r, c in source_cells | bulb_cells):
            continue
        if len(path) != len(set(path)):
            continue
        if any(not (r0 <= r < r1 and c0 <= c < c1) for r, c in path):
            continue
        if any((r, c) in source_cells or (r, c) in bulb_cells for r, c in path):
            continue
        if any(grid[r, c] != BG for r, c in path):
            continue

        ok = True
        for idx, (r, c) in enumerate(path):
            adj_source = any((nr, nc) in source_cells for nr, nc in _neighbors4(r, c, grid.shape[0], grid.shape[1]))
            adj_bulb = any((nr, nc) in bulb_cells for nr, nc in _neighbors4(r, c, grid.shape[0], grid.shape[1]))
            if idx == 0:
                ok = adj_source and not adj_bulb
            elif idx == len(path) - 1:
                ok = adj_bulb and not adj_source
            else:
                ok = not adj_source and not adj_bulb
            if not ok:
                break
        if not ok:
            continue

        _paint_block(grid, src_top, src_left, source_color)
        _paint_block(grid, bulb_top, bulb_left, BULB)

        colors_on_path = [WIRE] * len(path)
        if filters:
            filter_slots = sorted(random.sample(range(1, len(path) - 1), len(filters)))
            for slot, filt in zip(filter_slots, filters):
                colors_on_path[slot] = int(filt)

        for (r, c), color in zip(path, colors_on_path):
            grid[r, c] = int(color)
        return True

    return False


def _grid_for_layout(circuit_count: int, rows_div: int, cols_div: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    min_h = 2 * OUTER_MARGIN + rows_div * MIN_REGION + (rows_div - 1) * REGION_GAP
    min_w = 2 * OUTER_MARGIN + cols_div * MIN_REGION + (cols_div - 1) * REGION_GAP
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    row_spans = _split_spans(height, rows_div, MIN_REGION)
    col_spans = _split_spans(width, cols_div, MIN_REGION)

    all_regions = [(rr0, rr1, cc0, cc1) for rr0, rr1 in row_spans for cc0, cc1 in col_spans]
    chosen_regions = random.sample(all_regions, circuit_count)
    grid = np.full((height, width), BG, dtype=int)
    return grid, chosen_regions


def _gen_attempt() -> np.ndarray | None:
    circuit_count = random.choice([2, 3, 4])
    if circuit_count == 2:
        rows_div, cols_div = random.choice([(1, 2), (2, 1)])
    else:
        rows_div, cols_div = 2, 2

    grid, regions = _grid_for_layout(circuit_count, rows_div, cols_div)
    seqs = _filter_sequences(circuit_count)
    random.shuffle(regions)

    for idx, region in enumerate(regions):
        source_color = random.choice(SOURCE_COLORS)
        orientation = random.choice(["h", "v"])
        if not _draw_circuit(grid, region, source_color, seqs[idx], orientation):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if not bool((out != grid).any()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
