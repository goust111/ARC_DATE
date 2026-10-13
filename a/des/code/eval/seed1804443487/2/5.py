from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
REGION_SIDES = ("top", "bottom", "left", "right")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _unique_majority(
    grid: np.ndarray, cells: set[tuple[int, int]]
) -> tuple[int, Counter[int]] | None:
    counts = Counter(int(grid[r, c]) for r, c in cells)
    if len(counts) not in (2, 3):
        return None
    ranked = counts.most_common()
    if len(ranked) < 2 or ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0]), counts


def _parse_rule_tiles(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    grey_comps = _components(grid == GREY)
    mapping: dict[int, int] = {}
    legend_mask = np.zeros(grid.shape, dtype=bool)
    tiles: list[tuple[int, int, int, int]] = []

    for comp in grey_comps:
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 + 1 != 4 or c1 - c0 + 1 != 3:
            continue

        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border != bool(grid[r, c] == GREY):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        src = int(grid[r0 + 1, c0 + 1])
        dst = int(grid[r0 + 2, c0 + 1])
        if src in (BG, GREY) or dst in (BG, GREY):
            continue

        tiles.append((c0, r0, src, dst))

    assert tiles, "expected at least one rule tile"
    for _c0, r0, src, dst in sorted(tiles):
        assert src not in mapping, "rule tile sources must be unique"
        mapping[src] = dst
        legend_mask[r0 : r0 + 4, _c0 : _c0 + 3] = True

    return mapping, legend_mask


def _split_region_and_indicator(
    grid: np.ndarray, comp: list[tuple[int, int]]
) -> tuple[set[tuple[int, int]], tuple[int, int], int]:
    cells = set(comp)
    candidates: list[tuple[int, tuple[int, int], set[tuple[int, int]], int]] = []
    full_r0, full_r1, full_c0, full_c1 = _bbox(comp)

    for cell in comp:
        color = int(grid[cell])
        same_neighbors = sum(
            1 for nr, nc in _neighbors4(*cell) if (nr, nc) in cells and int(grid[nr, nc]) == color
        )
        if same_neighbors != 0:
            continue

        remaining = cells - {cell}
        if not _is_connected(remaining):
            continue

        majority = _unique_majority(grid, remaining)
        if majority is None:
            continue
        main_color, _counts = majority
        if color != main_color:
            continue

        degree = sum(1 for nb in _neighbors4(*cell) if nb in remaining)
        if degree == 0:
            continue

        on_outer_edge = cell[0] in (full_r0, full_r1) or cell[1] in (full_c0, full_c1)
        edge_penalty = 0 if on_outer_edge else 1
        candidates.append((degree * 10 + edge_penalty, cell, remaining, main_color))

    assert candidates, "expected a removable indicator pixel"
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "indicator pixel must be unique"
    _, indicator, region_cells, main_color = best[0]
    return region_cells, indicator, int(main_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top rule tiles as source->target color replacements.

    Each lower multi-color object is a region plus one adjacent indicator pixel.
    The region's dominant color selects the tile, then every dominant-color cell
    in that region and its indicator pixel are recolored to the tile's target.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _parse_rule_tiles(g)

    foreground = (g != BG) & (~legend_mask)
    comps = _components(foreground)
    assert comps, "expected at least one lower region"

    out = g.copy()
    for comp in comps:
        region_cells, indicator, main_color = _split_region_and_indicator(g, comp)
        assert main_color in mapping, "every region main color must have a rule tile"
        target = int(mapping[main_color])

        for r, c in region_cells:
            if int(g[r, c]) == main_color:
                out[r, c] = target
        out[indicator] = target

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 31)
RULE_COUNT_RANGE = (3, 5)
REGION_COUNT_RANGE = (2, 4)
LEGEND_TOP_OPTIONS = (0, 1)
LEGEND_LEFT_RANGE = (1, 4)
LEGEND_GAP_OPTIONS = (1, 2)
REGION_START_GAP = 2

BACKBONE_H_RANGE = (3, 6)
BACKBONE_W_RANGE = (3, 7)
APPENDAGE_DEPTH_RANGE = (1, 3)
APPENDAGE_SPAN_MAX = 2

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 400


def _rule_tile(src: int, dst: int) -> np.ndarray:
    tile = np.full((4, 3), GREY, dtype=int)
    tile[1, 1] = src
    tile[2, 1] = dst
    return tile


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _appendage_cells(side: str, bh: int, bw: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    if side in ("top", "bottom"):
        interior = bw - 2
        span = random.randint(1, min(APPENDAGE_SPAN_MAX, interior))
        start = random.randint(1, bw - 1 - span)
        depth = random.randint(*APPENDAGE_DEPTH_RANGE)
        if side == "top":
            cells = _rect_cells(-depth, start, depth, span)
            outer = [(-depth, c) for c in range(start, start + span)]
            step = (-1, 0)
        else:
            cells = _rect_cells(bh, start, depth, span)
            outer = [(bh + depth - 1, c) for c in range(start, start + span)]
            step = (1, 0)
    else:
        interior = bh - 2
        span = random.randint(1, min(APPENDAGE_SPAN_MAX, interior))
        start = random.randint(1, bh - 1 - span)
        depth = random.randint(*APPENDAGE_DEPTH_RANGE)
        if side == "left":
            cells = _rect_cells(start, -depth, span, depth)
            outer = [(r, -depth) for r in range(start, start + span)]
            step = (0, -1)
        else:
            cells = _rect_cells(start, bw, span, depth)
            outer = [(r, bw + depth - 1) for r in range(start, start + span)]
            step = (0, 1)
    return cells, outer, step


def _normalize_cells(
    color_cells: dict[int, set[tuple[int, int]]], indicator: tuple[int, int]
) -> tuple[dict[int, set[tuple[int, int]]], tuple[int, int]]:
    all_cells = set().union(*color_cells.values(), {indicator})
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)

    shifted: dict[int, set[tuple[int, int]]] = {}
    for color, cells in color_cells.items():
        shifted[color] = {(r - min_r, c - min_c) for r, c in cells}
    return shifted, (indicator[0] - min_r, indicator[1] - min_c)


def _build_region_patch(main_color: int, target_color: int) -> tuple[np.ndarray, dict[str, object]]:
    bh = random.randint(*BACKBONE_H_RANGE)
    bw = random.randint(*BACKBONE_W_RANGE)
    backbone = _rect_cells(0, 0, bh, bw)

    n_secondary = 1 if random.random() < 0.5 else 2
    secondary_pool = [c for c in PALETTE if c not in (main_color, target_color)]
    secondary_colors = random.sample(secondary_pool, n_secondary)
    sides = random.sample(REGION_SIDES, n_secondary)

    color_cells: dict[int, set[tuple[int, int]]] = {main_color: set(backbone)}
    appendages: list[tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]] = []
    for side, color in zip(sides, secondary_colors):
        cells, outer, step = _appendage_cells(side, bh, bw)
        color_cells[color] = set(cells)
        appendages.append((cells, outer, step))

    _host_cells, host_outer, host_step = random.choice(appendages)
    anchor = random.choice(host_outer)
    indicator = (anchor[0] + host_step[0], anchor[1] + host_step[1])

    all_region_cells = set().union(*color_cells.values())
    assert indicator not in all_region_cells

    color_cells, indicator = _normalize_cells(color_cells, indicator)
    all_cells = set().union(*color_cells.values(), {indicator})
    h = 1 + max(r for r, _ in all_cells)
    w = 1 + max(c for _, c in all_cells)

    patch = np.full((h, w), BG, dtype=int)
    region_cells: set[tuple[int, int]] = set()
    for color, cells in color_cells.items():
        for r, c in cells:
            patch[r, c] = color
            region_cells.add((r, c))
    patch[indicator] = main_color

    main_cells = {cell for cell in region_cells if int(patch[cell]) == main_color}
    assert all(
        any(nb in main_cells for nb in _neighbors4(*cell))
        for cell in main_cells
    ), "region main-color cells must not look like indicators"

    meta = {
        "region_cells": region_cells,
        "indicator": indicator,
        "main_color": main_color,
        "target_color": target_color,
    }
    return patch, meta


def _can_place(
    occupied: np.ndarray, patch: np.ndarray, top: int, left: int, row_min: int
) -> bool:
    if top < row_min:
        return False
    h, w = occupied.shape
    ph, pw = patch.shape
    if top + ph > h or left + pw > w:
        return False

    for r, c in np.argwhere(patch != BG):
        rr = top + int(r)
        cc = left + int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    coords = np.argwhere(patch != BG)
    for r, c in coords:
        dst[top + int(r), left + int(c)] = int(patch[int(r), int(c)])


def _expected_output(inp: np.ndarray, placed_regions: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for info in placed_regions:
        top = int(info["top"])
        left = int(info["left"])
        main_color = int(info["main_color"])
        target_color = int(info["target_color"])
        region_cells = info["region_cells"]
        indicator = info["indicator"]

        for r, c in region_cells:
            rr = top + r
            cc = left + c
            if int(out[rr, cc]) == main_color:
                out[rr, cc] = target_color
        out[top + indicator[0], left + indicator[1]] = target_color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.randint(*REGION_COUNT_RANGE)
    n_rules = random.randint(max(RULE_COUNT_RANGE[0], n_regions), RULE_COUNT_RANGE[1])

    source_colors = random.sample(PALETTE, n_rules)
    mapping = {
        src: random.choice([c for c in PALETTE if c != src])
        for src in source_colors
    }

    top_margin = random.choice(LEGEND_TOP_OPTIONS)
    left_margin = random.randint(*LEGEND_LEFT_RANGE)
    gaps = [random.choice(LEGEND_GAP_OPTIONS) for _ in range(n_rules - 1)]
    legend_width = n_rules * 3 + sum(gaps)
    min_width = left_margin + legend_width + 1
    if min_width > GRID_W_RANGE[1]:
        return None

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(max(GRID_W_RANGE[0], min_width), GRID_W_RANGE[1])
    grid = np.full((h, w), BG, dtype=int)

    x = left_margin
    for i, src in enumerate(source_colors):
        tile = _rule_tile(src, mapping[src])
        _paste(grid, tile, top_margin, x)
        x += tile.shape[1]
        if i < len(gaps):
            x += gaps[i]

    occupied = grid != BG
    row_min = top_margin + 4 + REGION_START_GAP
    if row_min >= h - 2:
        return None

    region_main_colors = random.sample(source_colors, n_regions)
    random.shuffle(region_main_colors)
    placed_regions: list[dict[str, object]] = []

    for main_color in region_main_colors:
        patch, meta = _build_region_patch(main_color, mapping[main_color])
        ph, pw = patch.shape
        if h - ph < row_min or w - pw < 0:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(row_min, h - ph)
            left = random.randint(0, w - pw)
            if not _can_place(occupied, patch, top, left, row_min):
                continue
            _paste(grid, patch, top, left)
            occupied[top : top + ph, left : left + pw] |= patch != BG
            placed_regions.append(
                {
                    "top": top,
                    "left": left,
                    **meta,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, placed_regions)
    actual = main(grid)
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
