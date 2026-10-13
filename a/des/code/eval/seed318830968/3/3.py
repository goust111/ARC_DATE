import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = tuple(c for c in range(10) if c not in (BG, WALL))

GEN_MAX_ATTEMPTS = 1000
GRID_H_RANGE = (24, 34)
GRID_W_RANGE = (24, 34)
LEGEND_LENGTHS = (2, 3, 4)
REGION_COUNT_CHOICES = (2, 3, 4)
SHAPE_TRIES = 200
PLACEMENT_TRIES = 250


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _distance_layers(region_cells: list[tuple[int, int]], wall_mask: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = wall_mask.shape
    region_set = set(region_cells)
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int]] = deque()
    for r, c in region_cells:
        if any(0 <= nr < h and 0 <= nc < w and wall_mask[nr, nc] for nr, nc in neighbors4(r, c)):
            dist[(r, c)] = 0
            q.append((r, c))
    assert q, "region must have an inner layer adjacent to a wall"
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in region_set and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                q.append(nxt)
    assert len(dist) == len(region_cells), "region interior must be connected"
    return dist


def _detect_legend(grid: np.ndarray) -> tuple[set[tuple[int, int]], list[int]]:
    h, w = grid.shape
    colored = (grid != BG) & (grid != WALL)
    candidates = []
    for comp in _components(colored):
        if not _touches_border(comp, h, w):
            continue
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) != 1:
            continue
        if len(cols) < 2 or len(cols) > 4:
            continue
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        row = rows[0]
        if row not in (0, h - 1):
            continue
        if cols[0] != 0 and cols[-1] != w - 1:
            continue
        colors = [int(grid[row, c]) for c in cols]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((set(comp), colors))
    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    legend_cells, palette = _detect_legend(g)
    work = g.copy()
    for r, c in legend_cells:
        work[r, c] = BG

    wall_mask = work == WALL
    out = work.copy()
    for comp in _components(~wall_mask):
        if _touches_border(comp, h, w):
            continue
        seeds = [(r, c) for r, c in comp if int(work[r, c]) != BG]
        assert len(seeds) == 1, "each closed region must contain exactly one seed pixel"
        seed = seeds[0]
        seed_color = int(work[seed])
        dists = _distance_layers(comp, wall_mask)
        seed_depth = dists[seed]
        for cell, depth in dists.items():
            out[cell] = seed_color if depth >= seed_depth else palette[depth % len(palette)]
    return out


def _has_hole(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    for comp in _components(~mask):
        if not _touches_border(comp, h, w):
            return True
    return False


def _boundary_of(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                boundary.add((nr, nc))
    return boundary


def _crop_shape(
    cells: set[tuple[int, int]], boundary: set[tuple[int, int]]
) -> tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]:
    all_cells = cells | boundary
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    new_cells = {(r - r0, c - c0) for r, c in cells}
    new_boundary = {(r - r0, c - c0) for r, c in boundary}
    return r1 - r0 + 1, c1 - c0 + 1, new_cells, new_boundary


def _choose_seed_cell(
    rng: random.Random,
    dists: dict[tuple[int, int], int],
    palette_len: int,
    category: str,
) -> tuple[int, int]:
    max_depth = max(dists.values())
    if category == "deep":
        candidates = [cell for cell, depth in dists.items() if depth >= palette_len]
        if not candidates:
            candidates = [cell for cell, depth in dists.items() if depth == max_depth]
    elif category == "shallow":
        candidates = [cell for cell, depth in dists.items() if depth <= 1]
    else:
        candidates = [cell for cell, depth in dists.items() if 1 <= depth < max_depth]
        if not candidates:
            middle = max_depth // 2
            candidates = [cell for cell, depth in dists.items() if depth == middle]
        if not candidates:
            candidates = [cell for cell, depth in dists.items() if depth == 0]
    return rng.choice(candidates)


def _make_region_shape(
    rng: random.Random, min_depth: int, palette_len: int, category: str
) -> dict | None:
    for _ in range(SHAPE_TRIES):
        core_h = rng.randint(2 * min_depth + 1, 2 * min_depth + 5)
        core_w = rng.randint(2 * min_depth + 1, 2 * min_depth + 5)
        h = core_h + 6
        w = core_w + 6
        top = 2
        left = 2
        cells = {(top + r, left + c) for r in range(core_h) for c in range(core_w)}

        for _ in range(rng.randint(0, 3)):
            side = rng.choice(("T", "B", "L", "R"))
            if side in ("T", "B"):
                depth = rng.randint(1, 2)
                span = rng.randint(2, max(2, core_w - 1))
                start = rng.randint(left, left + core_w - span)
                rows = range(top - depth, top) if side == "T" else range(top + core_h, top + core_h + depth)
                for r in rows:
                    for c in range(start, start + span):
                        if 1 <= r < h - 1 and 1 <= c < w - 1:
                            cells.add((r, c))
            else:
                depth = rng.randint(1, 2)
                span = rng.randint(2, max(2, core_h - 1))
                start = rng.randint(top, top + core_h - span)
                cols = range(left - depth, left) if side == "L" else range(left + core_w, left + core_w + depth)
                for r in range(start, start + span):
                    for c in cols:
                        if 1 <= r < h - 1 and 1 <= c < w - 1:
                            cells.add((r, c))

        for _ in range(rng.randint(1, 4)):
            side = rng.choice(("T", "B", "L", "R", "TL", "TR", "BL", "BR"))
            if side == "T" and core_h >= 4 and core_w >= 4:
                depth = rng.randint(1, min(2, core_h - 2))
                span = rng.randint(1, core_w - 2)
                start = rng.randint(left + 1, left + core_w - span - 1)
                for r in range(top, top + depth):
                    for c in range(start, start + span):
                        cells.discard((r, c))
            elif side == "B" and core_h >= 4 and core_w >= 4:
                depth = rng.randint(1, min(2, core_h - 2))
                span = rng.randint(1, core_w - 2)
                start = rng.randint(left + 1, left + core_w - span - 1)
                for r in range(top + core_h - depth, top + core_h):
                    for c in range(start, start + span):
                        cells.discard((r, c))
            elif side == "L" and core_h >= 4 and core_w >= 4:
                depth = rng.randint(1, min(2, core_w - 2))
                span = rng.randint(1, core_h - 2)
                start = rng.randint(top + 1, top + core_h - span - 1)
                for r in range(start, start + span):
                    for c in range(left, left + depth):
                        cells.discard((r, c))
            elif side == "R" and core_h >= 4 and core_w >= 4:
                depth = rng.randint(1, min(2, core_w - 2))
                span = rng.randint(1, core_h - 2)
                start = rng.randint(top + 1, top + core_h - span - 1)
                for r in range(start, start + span):
                    for c in range(left + core_w - depth, left + core_w):
                        cells.discard((r, c))
            elif side == "TL":
                dh = rng.randint(1, min(2, core_h - 1))
                dw = rng.randint(1, min(2, core_w - 1))
                for r in range(top, top + dh):
                    for c in range(left, left + dw):
                        cells.discard((r, c))
            elif side == "TR":
                dh = rng.randint(1, min(2, core_h - 1))
                dw = rng.randint(1, min(2, core_w - 1))
                for r in range(top, top + dh):
                    for c in range(left + core_w - dw, left + core_w):
                        cells.discard((r, c))
            elif side == "BL":
                dh = rng.randint(1, min(2, core_h - 1))
                dw = rng.randint(1, min(2, core_w - 1))
                for r in range(top + core_h - dh, top + core_h):
                    for c in range(left, left + dw):
                        cells.discard((r, c))
            else:
                dh = rng.randint(1, min(2, core_h - 1))
                dw = rng.randint(1, min(2, core_w - 1))
                for r in range(top + core_h - dh, top + core_h):
                    for c in range(left + core_w - dw, left + core_w):
                        cells.discard((r, c))

        if not cells:
            continue
        cell_mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            cell_mask[r, c] = True
        if len(_components(cell_mask)) != 1:
            continue
        if _has_hole(cells, h, w):
            continue

        boundary = _boundary_of(cells, h, w)
        if not boundary:
            continue
        boundary_mask = np.zeros((h, w), dtype=bool)
        for r, c in boundary:
            boundary_mask[r, c] = True

        dists = _distance_layers(list(cells), boundary_mask)
        max_depth = max(dists.values())
        if max_depth < min_depth:
            continue

        ch, cw, cropped_cells, cropped_boundary = _crop_shape(cells, boundary)
        cropped_wall = np.zeros((ch, cw), dtype=bool)
        for r, c in cropped_boundary:
            cropped_wall[r, c] = True
        cropped_dists = _distance_layers(list(cropped_cells), cropped_wall)
        seed = _choose_seed_cell(rng, cropped_dists, palette_len, category)
        return {
            "height": ch,
            "width": cw,
            "cells": cropped_cells,
            "boundary": cropped_boundary,
            "dists": cropped_dists,
            "seed_local": seed,
            "category": category,
        }
    return None


def _legend_cells(h: int, w: int, corner: str, colors: list[int]) -> list[tuple[int, int]]:
    length = len(colors)
    if corner == "TL":
        row, start = 0, 0
    elif corner == "TR":
        row, start = 0, w - length
    elif corner == "BL":
        row, start = h - 1, 0
    else:
        row, start = h - 1, w - length
    return [(row, start + i) for i in range(length)]


def _mark_forbidden(forbidden: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    forbidden[rr, cc] = True


def _place_region(
    rng: random.Random,
    forbidden: np.ndarray,
    region: dict,
) -> tuple[int, int] | None:
    h, w = forbidden.shape
    rh, rw = region["height"], region["width"]
    row_lo = 1
    col_lo = 1
    row_hi = h - rh - 1
    col_hi = w - rw - 1
    if row_hi < row_lo or col_hi < col_lo:
        return None
    occupied_local = region["cells"] | region["boundary"]
    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(row_lo, row_hi)
        left = rng.randint(col_lo, col_hi)
        occupied_global = {(top + r, left + c) for r, c in occupied_local}
        if any(forbidden[r, c] for r, c in occupied_global):
            continue
        return top, left
    return None


def _render_expected(grid: np.ndarray, legend_cells: list[tuple[int, int]], palette: list[int], regions: list[dict]) -> np.ndarray:
    out = grid.copy()
    for r, c in legend_cells:
        out[r, c] = BG
    for region in regions:
        seed_depth = region["dists"][region["seed_local"]]
        seed_color = region["seed_color"]
        top, left = region["top"], region["left"]
        for (r, c), depth in region["dists"].items():
            gr, gc = top + r, left + c
            out[gr, gc] = seed_color if depth >= seed_depth else palette[depth % len(palette)]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    palette_len = rng.choice(LEGEND_LENGTHS)
    count_choices = (2, 3) if palette_len == 4 else REGION_COUNT_CHOICES
    region_count = rng.choice(count_choices)
    if palette_len + region_count > len(COLOR_POOL):
        return None

    grid = np.full((h, w), BG, dtype=int)
    chosen_colors = rng.sample(COLOR_POOL, palette_len + region_count)
    palette = chosen_colors[:palette_len]
    seed_colors = chosen_colors[palette_len:]

    corner = rng.choice(("TL", "TR", "BL", "BR"))
    legend_pos = _legend_cells(h, w, corner, palette)
    for (r, c), color in zip(legend_pos, palette):
        grid[r, c] = color

    forbidden = np.zeros((h, w), dtype=bool)
    _mark_forbidden(forbidden, set(legend_pos))

    categories = ["deep", "shallow"]
    extra_choices = ["deep", "mid", "shallow"]
    while len(categories) < region_count:
        categories.append(rng.choice(extra_choices))
    rng.shuffle(categories)

    regions = []
    for idx in range(region_count):
        category = categories[idx]
        min_depth = palette_len if category == "deep" else 1
        region = _make_region_shape(rng, min_depth=min_depth, palette_len=palette_len, category=category)
        if region is None:
            return None
        region["seed_color"] = seed_colors[idx]
        placement = _place_region(rng, forbidden, region)
        if placement is None:
            return None
        top, left = placement
        region["top"] = top
        region["left"] = left
        occupied_global = {(top + r, left + c) for r, c in (region["cells"] | region["boundary"])}
        _mark_forbidden(forbidden, occupied_global)
        regions.append(region)

    for region in regions:
        top, left = region["top"], region["left"]
        for r, c in region["boundary"]:
            grid[top + r, left + c] = WALL
        sr, sc = region["seed_local"]
        grid[top + sr, left + sc] = region["seed_color"]

    expected = _render_expected(grid, legend_pos, palette, regions)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(expected, produced):
        return None
    if np.array_equal(grid, produced):
        return None
    if len({region["category"] for region in regions}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-fill puzzle")
