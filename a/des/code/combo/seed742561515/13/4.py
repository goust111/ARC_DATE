from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

LEGEND_TEMPLATE = np.array([[RED, GREEN], [BLUE, YELLOW]], dtype=int)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("tl", "tr", "bl", "br")

GEN_TRIES = 400
CELL_ROWS_RANGE = (6, 11)
CELL_COLS_RANGE = (6, 12)


def _legend_slices(shape: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    h, w = shape
    return {
        "tl": (slice(0, 2), slice(0, 2)),
        "tr": (slice(0, 2), slice(w - 2, w)),
        "bl": (slice(h - 2, h), slice(0, 2)),
        "br": (slice(h - 2, h), slice(w - 2, w)),
    }


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray, str]:
    for corner, (rs, cs) in _legend_slices(grid.shape).items():
        patch = grid[rs, cs]
        if patch.shape == (2, 2) and np.array_equal(patch, LEGEND_TEMPLATE):
            mask = np.zeros_like(grid, dtype=bool)
            mask[rs, cs] = True
            mapping = {
                1: int(patch[0, 0]),
                2: int(patch[0, 1]),
                3: int(patch[1, 0]),
                4: int(patch[1, 1]),
            }
            return mapping, mask, corner
    raise AssertionError("expected a 2x2 legend in one grid corner")


def _neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _degree_map(mask: np.ndarray) -> np.ndarray:
    deg = np.zeros(mask.shape, dtype=int)
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        deg[r, c] = len(_neighbors(mask, int(r), int(c)))
    return deg


def _trace_dead_end_depth(
    passage_mask: np.ndarray, degree: np.ndarray, start: tuple[int, int]
) -> int:
    depth = 1
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nbr for nbr in _neighbors(passage_mask, cur[0], cur[1]) if nbr != prev]
        if not nxts:
            return depth

        nxt = nxts[0]
        nxt_degree = int(degree[nxt])
        if nxt_degree >= 3:
            return depth

        depth += 1
        prev, cur = cur, nxt
        if nxt_degree != 2:
            return depth


def _analyze_maze(grid: np.ndarray) -> tuple[dict[int, int], list[tuple[int, int, int]]]:
    mapping, legend_mask, _ = _find_legend(grid)
    passage_mask = grid == BLACK
    degree = _degree_map(passage_mask)

    h, w = grid.shape
    border_mask = np.zeros_like(passage_mask, dtype=bool)
    border_mask[0, :] = True
    border_mask[h - 1, :] = True
    border_mask[:, 0] = True
    border_mask[:, w - 1] = True
    border_passages = {
        (int(r), int(c))
        for r, c in zip(*np.where(passage_mask & border_mask), strict=False)
    }
    assert len(border_passages) == 1, "maze must have exactly one border entrance"

    coords: list[tuple[int, int, int]] = []
    rows, cols = np.where(passage_mask)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        pos = (int(r), int(c))
        if pos in border_passages:
            continue
        if int(degree[pos]) != 1:
            continue
        depth = _trace_dead_end_depth(passage_mask, degree, pos)
        coords.append((pos[0], pos[1], depth))

    return mapping, coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark maze dead ends by their depth to the nearest branch, using the 2x2 corner legend.
    """
    g = np.asarray(grid, dtype=int)
    mapping, dead_ends = _analyze_maze(g)

    out = g.copy()
    for r, c, depth in dead_ends:
        out[r, c] = int(mapping[min(depth, 4)])
    return out


def _reserved_cell(corner: str, cell_rows: int, cell_cols: int) -> tuple[int, int]:
    if corner == "tl":
        return (0, 0)
    if corner == "tr":
        return (0, cell_cols - 1)
    if corner == "bl":
        return (cell_rows - 1, 0)
    return (cell_rows - 1, cell_cols - 1)


def _cell_to_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    return (2 * cell[0] + 1, 2 * cell[1] + 1)


def _carve_maze(
    cell_rows: int, cell_cols: int, legend_corner: str
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    h = 2 * cell_rows + 1
    w = 2 * cell_cols + 1
    grid = np.full((h, w), GREY, dtype=int)

    blocked = _reserved_cell(legend_corner, cell_rows, cell_cols)
    allowed = {
        (r, c)
        for r in range(cell_rows)
        for c in range(cell_cols)
        if (r, c) != blocked
    }
    start = random.choice(tuple(allowed))
    visited = {start}
    stack = [start]
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            nxt = (nr, nc)
            if nxt in allowed and nxt not in visited:
                nbrs.append(nxt)
        if not nbrs:
            stack.pop()
            continue
        nxt = random.choice(nbrs)
        visited.add(nxt)
        stack.append(nxt)
        edges.append(((r, c), nxt))

    assert visited == allowed

    for cell in allowed:
        pr, pc = _cell_to_pixel(cell)
        grid[pr, pc] = BLACK
    for a, b in edges:
        ar, ac = _cell_to_pixel(a)
        br, bc = _cell_to_pixel(b)
        grid[(ar + br) // 2, (ac + bc) // 2] = BLACK

    return grid, allowed


def _entrance_candidates(
    allowed: set[tuple[int, int]], cell_rows: int, cell_cols: int
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in allowed:
        if r == 0:
            candidates.append((0, 2 * c + 1))
        if r == cell_rows - 1:
            candidates.append((2 * r + 1, 2 * cell_cols))
        if c == 0:
            candidates.append((2 * r + 1, 0))
        if c == cell_cols - 1:
            candidates.append((2 * r + 1, 2 * c + 1))
    return candidates


def _add_entrance(
    grid: np.ndarray, allowed: set[tuple[int, int]], cell_rows: int, cell_cols: int
) -> None:
    candidates = []
    for r, c in allowed:
        if r == 0:
            candidates.append((0, 2 * c + 1))
        if r == cell_rows - 1:
            candidates.append((grid.shape[0] - 1, 2 * c + 1))
        if c == 0:
            candidates.append((2 * r + 1, 0))
        if c == cell_cols - 1:
            candidates.append((2 * r + 1, grid.shape[1] - 1))
    er, ec = random.choice(candidates)
    grid[er, ec] = BLACK


def _paint_legend(grid: np.ndarray, corner: str) -> None:
    rs, cs = _legend_slices(grid.shape)[corner]
    grid[rs, cs] = LEGEND_TEMPLATE


def _depth_classes(dead_ends: list[tuple[int, int, int]]) -> set[int]:
    return {min(depth, 4) for _, _, depth in dead_ends}


def _accept_sample(dead_ends: list[tuple[int, int, int]], mode: str) -> bool:
    if len(dead_ends) < 4:
        return False

    classes = _depth_classes(dead_ends)
    max_depth = max(depth for _, _, depth in dead_ends)
    if mode == "shallow":
        return 1 in classes and 2 in classes and len(classes) >= 2
    if mode == "mid":
        return 2 in classes and 3 in classes and len(classes) >= 2
    if mode == "deep":
        return max_depth >= 4 and len(classes) >= 2
    return len(classes) >= 3 and max_depth >= 4


def generate_input() -> np.ndarray:
    mode = random.choice(("shallow", "mid", "deep", "broad"))
    for attempt in range(GEN_TRIES):
        cell_rows = random.randint(*CELL_ROWS_RANGE)
        cell_cols = random.randint(*CELL_COLS_RANGE)
        legend_corner = random.choice(CORNERS)

        grid, allowed = _carve_maze(cell_rows, cell_cols, legend_corner)
        _add_entrance(grid, allowed, cell_rows, cell_cols)
        _paint_legend(grid, legend_corner)

        try:
            _, dead_ends = _analyze_maze(grid)
        except AssertionError:
            continue

        relaxed = attempt >= GEN_TRIES // 2
        if not _accept_sample(dead_ends, mode):
            if not relaxed:
                continue
            if len(dead_ends) < 3 or max(depth for _, _, depth in dead_ends) < 2:
                continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid maze sample")
