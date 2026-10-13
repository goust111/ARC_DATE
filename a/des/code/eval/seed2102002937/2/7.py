from __future__ import annotations

import random
from collections import Counter

import numpy as np


BOX_CELLS = {(0, 0), (0, 1), (1, 0), (1, 1)}
DIR_BY_MISSING = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 1): "down",
    (1, 0): "left",
}
VEC_BY_DIR = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in [*sorted(separators), size]:
        if start <= sep - 1:
            spans.append((start, sep - 1))
        start = sep + 1
    return spans


def _find_wall_structure(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_colors = Counter(
        int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])
    )
    col_colors = Counter(
        int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])
    )
    candidates = [color for color in row_colors if color in col_colors]
    assert len(candidates) == 1, "expected one wall color shared by full rows and columns"
    wall = candidates[0]
    wall_rows = [r for r in range(h) if np.all(g[r, :] == wall)]
    wall_cols = [c for c in range(w) if np.all(g[:, c] == wall)]
    assert wall_rows and wall_cols, "expected separator rows and separator columns"
    return wall, wall_rows, wall_cols


def _find_pointer(region: np.ndarray) -> tuple[int, list[tuple[int, int]], str]:
    for color in map(int, np.unique(region)):
        pts = np.argwhere(region == color)
        if len(pts) != 3:
            continue
        r0, c0 = np.min(pts, axis=0)
        r1, c1 = np.max(pts, axis=0)
        if int(r1 - r0) != 1 or int(c1 - c0) != 1:
            continue
        norm = {(int(r - r0), int(c - c0)) for r, c in pts}
        if len(norm) != 3:
            continue
        missing = next(cell for cell in BOX_CELLS if cell not in norm)
        direction = DIR_BY_MISSING[missing]
        return int(color), [(int(r), int(c)) for r, c in pts], direction
    raise AssertionError("expected one 3-cell 2x2 L pointer in each region")


def _slot_order(
    slots: list[tuple[int, int]], height: int, width: int
) -> list[tuple[int, int]]:
    rows = {r for r, _ in slots}
    cols = {c for _, c in slots}
    if len(rows) == 1:
        row = next(iter(rows))
        assert row in (0, height - 1), "horizontal slots must lie on a region edge"
        return sorted(slots, key=lambda p: p[1])
    if len(cols) == 1:
        col = next(iter(cols))
        assert col in (0, width - 1), "vertical slots must lie on a region edge"
        return sorted(slots, key=lambda p: p[0])
    raise AssertionError("slots must form a straight row or column")


def _parse_regions(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _wall, wall_rows, wall_cols = _find_wall_structure(g)
    row_spans = _spans(h, wall_rows)
    col_spans = _spans(w, wall_cols)
    assert row_spans and col_spans, "expected at least one region"

    regions: list[dict] = []
    by_index: dict[tuple[int, int], dict] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            pointer_color, pointer_cells, direction = _find_pointer(region)

            counts = Counter(map(int, region.reshape(-1)))
            counts.pop(pointer_color)
            assert counts, "region must contain a base color"
            base_color = max(counts.items(), key=lambda item: item[1])[0]
            extra_colors = [color for color in counts if color != base_color]
            assert len(extra_colors) <= 1, "only the start region may contain slots"

            slots_local: list[tuple[int, int]] = []
            if extra_colors:
                slot_color = extra_colors[0]
                slots_local = [(int(r), int(c)) for r, c in np.argwhere(region == slot_color)]
                slots_local = _slot_order(slots_local, region.shape[0], region.shape[1])

            info = {
                "index": (ri, ci),
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "base_color": int(base_color),
                "pointer_color": int(pointer_color),
                "pointer_cells": pointer_cells,
                "direction": direction,
                "slots_local": slots_local,
            }
            regions.append(info)
            by_index[(ri, ci)] = info
    return regions, by_index


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the inter-region pointer chain and write the visited region colors into
    the unique slotted start region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    regions, by_index = _parse_regions(g)
    start_regions = [region for region in regions if region["slots_local"]]
    assert len(start_regions) == 1, "expected one start region with slots"
    start = start_regions[0]

    total_regions = len(regions)
    path_colors: list[int] = []
    visited = {start["index"]}
    current = start
    for _ in range(total_regions):
        dr, dc = VEC_BY_DIR[current["direction"]]
        nxt_index = (current["index"][0] + dr, current["index"][1] + dc)
        if nxt_index not in by_index:
            break
        assert nxt_index not in visited, "pointer chain must not loop"
        current = by_index[nxt_index]
        visited.add(nxt_index)
        path_colors.append(current["base_color"])

    assert len(visited) == total_regions, "chain must visit every region exactly once"
    assert len(path_colors) == len(start["slots_local"]), "slot count must match chain length"

    for color, (lr, lc) in zip(path_colors, start["slots_local"]):
        out[start["r0"] + lr, start["c0"] + lc] = color
    return out


BG = 0
WALL = 5
POINTER = 9
BASE_POOL = [1, 2, 3, 4, 6, 7, 8]
LAYOUT_OPTIONS = [(2, 2), (2, 3), (3, 2)]
REGION_SIZE_RANGE = (5, 11)
MAX_GENERATION_ATTEMPTS = 400


def _snake_path(rows: int, cols: int, mode: str) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    if mode == "row":
        for r in range(rows):
            cols_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in cols_iter:
                path.append((r, c))
    else:
        for c in range(cols):
            rows_iter = range(rows) if c % 2 == 0 else range(rows - 1, -1, -1)
            for r in rows_iter:
                path.append((r, c))
    return path


def _choose_path(rows: int, cols: int, rng: random.Random) -> list[tuple[int, int]]:
    paths = [_snake_path(rows, cols, "row"), _snake_path(rows, cols, "col")]
    candidates = paths + [list(reversed(path)) for path in paths]
    return list(rng.choice(candidates))


def _step_count_to_length(count: int, step: int) -> int:
    return 1 + (count - 1) * step


def _fits_start_region(
    start_h: int, start_w: int, slot_count: int, orientation: str, step: int
) -> bool:
    line_len = _step_count_to_length(slot_count, step)
    if orientation == "h":
        return start_h >= 5 and start_w >= line_len + 2
    return start_w >= 5 and start_h >= line_len + 2


def _region_bounds(sizes: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    cur = 0
    for size in sizes:
        bounds.append((cur, cur + size - 1))
        cur += size + 1
    return bounds


def _slot_positions(
    height: int,
    width: int,
    count: int,
    orientation: str,
    step: int,
    side: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if orientation == "h":
        max_start = width - 2 - (count - 1) * step
        assert max_start >= 1
        start = rng.randint(1, max_start)
        row = 0 if side == "top" else height - 1
        return [(row, start + i * step) for i in range(count)]

    max_start = height - 2 - (count - 1) * step
    assert max_start >= 1
    start = rng.randint(1, max_start)
    col = 0 if side == "left" else width - 1
    return [(start + i * step, col) for i in range(count)]


def _pointer_offsets(direction: str) -> list[tuple[int, int]]:
    by_missing = {
        "up": (0, 0),
        "right": (0, 1),
        "down": (1, 1),
        "left": (1, 0),
    }
    missing = by_missing[direction]
    return sorted(cell for cell in BOX_CELLS if cell != missing)


def _pointer_candidates(
    height: int,
    width: int,
    direction: str,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    offsets = _pointer_offsets(direction)
    cands: list[tuple[int, int]] = []
    center_col = (width - 2) / 2.0
    center_row = (height - 2) / 2.0
    for r0 in range(height - 1):
        for c0 in range(width - 1):
            cells = {(r0 + dr, c0 + dc) for dr, dc in offsets}
            if cells & blocked:
                continue
            cands.append((r0, c0))
    rng.shuffle(cands)
    if direction == "up":
        cands.sort(key=lambda p: (p[0], abs(p[1] - center_col)))
    elif direction == "down":
        cands.sort(key=lambda p: (-p[0], abs(p[1] - center_col)))
    elif direction == "left":
        cands.sort(key=lambda p: (p[1], abs(p[0] - center_row)))
    else:
        cands.sort(key=lambda p: (-p[1], abs(p[0] - center_row)))
    return cands


def _place_pointer(
    region: np.ndarray,
    direction: str,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    height, width = region.shape
    offsets = _pointer_offsets(direction)
    cands = _pointer_candidates(height, width, direction, blocked, rng)
    if not cands:
        return False
    head = max(1, len(cands) // 3)
    r0, c0 = rng.choice(cands[:head])
    for dr, dc in offsets:
        region[r0 + dr, c0 + dc] = POINTER
    return True


def _delta_to_dir(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for name, (vr, vc) in VEC_BY_DIR.items():
        if (dr, dc) == (vr, vc):
            return name
    raise AssertionError("path cells must be 4-neighbors")


def _terminal_outside_dirs(cell: tuple[int, int], rows: int, cols: int) -> list[str]:
    ri, ci = cell
    dirs: list[str] = []
    if ri == 0:
        dirs.append("up")
    if ri == rows - 1:
        dirs.append("down")
    if ci == 0:
        dirs.append("left")
    if ci == cols - 1:
        dirs.append("right")
    return dirs


def _expected_output(
    grid: np.ndarray,
    start_cell: tuple[int, int],
    path: list[tuple[int, int]],
    base_color_by_cell: dict[tuple[int, int], int],
    slot_positions_by_cell: dict[tuple[int, int], list[tuple[int, int]]],
    row_bounds: list[tuple[int, int]],
    col_bounds: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    slots = slot_positions_by_cell[start_cell]
    colors = [base_color_by_cell[cell] for cell in path[1:]]
    assert len(slots) == len(colors)
    r0 = row_bounds[start_cell[0]][0]
    c0 = col_bounds[start_cell[1]][0]
    for color, (lr, lc) in zip(colors, slots):
        out[r0 + lr, c0 + lc] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(LAYOUT_OPTIONS)
    path = _choose_path(rows, cols, rng)
    start_cell = path[0]
    terminal_cell = path[-1]
    slot_count = rows * cols - 1

    desired_orientation = rng.choice(["h", "v"])
    slot_step = rng.choice([1, 2])
    if desired_orientation == "h":
        slot_side = rng.choice(["top", "bottom"])
    else:
        slot_side = rng.choice(["left", "right"])

    for _ in range(80):
        row_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(rows)]
        col_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(cols)]
        start_h = row_sizes[start_cell[0]]
        start_w = col_sizes[start_cell[1]]
        if _fits_start_region(start_h, start_w, slot_count, desired_orientation, slot_step):
            break
    else:
        return None

    row_bounds = _region_bounds(row_sizes)
    col_bounds = _region_bounds(col_sizes)
    total_h = row_bounds[-1][1] + 1
    total_w = col_bounds[-1][1] + 1
    grid = np.full((total_h, total_w), WALL, dtype=int)

    base_colors = rng.sample(BASE_POOL, rows * cols)
    base_color_by_cell: dict[tuple[int, int], int] = {}
    for index, cell in enumerate((ri, ci) for ri in range(rows) for ci in range(cols)):
        base_color_by_cell[cell] = base_colors[index]
        r0, r1 = row_bounds[cell[0]]
        c0, c1 = col_bounds[cell[1]]
        grid[r0 : r1 + 1, c0 : c1 + 1] = base_colors[index]

    slot_positions_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
    start_slots = _slot_positions(
        row_sizes[start_cell[0]],
        col_sizes[start_cell[1]],
        slot_count,
        desired_orientation,
        slot_step,
        slot_side,
        rng,
    )
    slot_positions_by_cell[start_cell] = start_slots
    r0, r1 = row_bounds[start_cell[0]]
    c0, c1 = col_bounds[start_cell[1]]
    for lr, lc in start_slots:
        grid[r0 + lr, c0 + lc] = BG

    direction_by_cell: dict[tuple[int, int], str] = {}
    for a, b in zip(path, path[1:]):
        direction_by_cell[a] = _delta_to_dir(a, b)
    direction_by_cell[terminal_cell] = rng.choice(_terminal_outside_dirs(terminal_cell, rows, cols))

    for cell in path:
        rr0, rr1 = row_bounds[cell[0]]
        cc0, cc1 = col_bounds[cell[1]]
        region = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
        blocked = set(slot_positions_by_cell.get(cell, []))
        if not _place_pointer(region, direction_by_cell[cell], blocked, rng):
            return None

    expected = _expected_output(
        grid,
        start_cell,
        path,
        base_color_by_cell,
        slot_positions_by_cell,
        row_bounds,
        col_bounds,
    )
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
