from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BRANCHES = ("keep", "mirror", "delete")

# Generator bounds.
GEN_HEIGHT_RANGE = (10, 18)
GEN_WIDTH_OPTIONS = tuple(range(12, 26, 2))
GEN_OBJECT_COUNT_RANGE = (3, 6)
GEN_SHAPE_HEIGHT_RANGE = (1, 5)
GEN_SHAPE_WIDTH_RANGE = (1, 6)
GEN_SHAPE_CELLS_RANGE = (3, 9)
GEN_PALETTE = tuple(range(1, 10))
GEN_SHAPE_RETRIES = 200
GEN_PLACEMENT_RETRIES = 200
GEN_ATTEMPTS = 300


def _components(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return monochrome 4-connected components as (color, coords)."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            objects.append((color, np.array(cells, dtype=int)))

    return objects


def _branch_for_cells(cells: np.ndarray) -> str:
    even = int(np.count_nonzero(cells[:, 1] % 2 == 0))
    odd = int(len(cells) - even)
    if even > odd:
        return "keep"
    if odd > even:
        return "mirror"
    return "delete"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split objects by hidden column parity.

    For each monochrome 4-connected object, count how many pixels lie in even
    columns and odd columns. Even-majority objects stay in place, odd-majority
    objects are mirrored left-right inside their own bounding boxes, and tied
    objects are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    for color, cells in _components(g):
        branch = _branch_for_cells(cells)
        if branch == "delete":
            continue
        if branch == "keep":
            out[cells[:, 0], cells[:, 1]] = color
            continue

        min_r = int(cells[:, 0].min())
        max_r = int(cells[:, 0].max())
        min_c = int(cells[:, 1].min())
        max_c = int(cells[:, 1].max())
        assert min_r <= max_r and min_c <= max_c
        mirrored_cols = min_c + max_c - cells[:, 1]
        out[cells[:, 0], mirrored_cols] = color

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _grow_connected_mask(height: int, width: int, cell_count: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    chosen = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(chosen) < cell_count:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        add_frontier(*cell)

    for r, c in chosen:
        mask[r, c] = True
    return _trim_mask(mask)


def _local_parity_counts(mask: np.ndarray) -> tuple[int, int]:
    cols = np.nonzero(mask)[1]
    even = int(np.count_nonzero(cols % 2 == 0))
    odd = int(len(cols) - even)
    return even, odd


def _shape_for_branch(branch: str) -> np.ndarray | None:
    for _ in range(GEN_SHAPE_RETRIES):
        height = random.randint(*GEN_SHAPE_HEIGHT_RANGE)
        min_width = 2 if branch in {"mirror", "delete"} else GEN_SHAPE_WIDTH_RANGE[0]
        width = random.randint(min_width, GEN_SHAPE_WIDTH_RANGE[1])
        max_cells = min(height * width, GEN_SHAPE_CELLS_RANGE[1])
        min_cells = min(GEN_SHAPE_CELLS_RANGE[0], max_cells)
        if max_cells < min_cells:
            continue

        if branch == "delete":
            counts = [n for n in range(max(2, min_cells), max_cells + 1) if n % 2 == 0]
            if not counts:
                continue
            cell_count = random.choice(counts)
        else:
            cell_count = random.randint(min_cells, max_cells)

        mask = _grow_connected_mask(height, width, cell_count)
        if mask.shape[1] < 2 and branch in {"mirror", "delete"}:
            continue

        even, odd = _local_parity_counts(mask)
        if branch == "delete":
            if even == odd:
                return mask
            continue
        if even == odd:
            continue
        if branch == "mirror" and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    return None


def _placement_column_parity(mask: np.ndarray, branch: str) -> int | None:
    even, odd = _local_parity_counts(mask)
    if even == odd:
        return None
    rel_even_majority = even > odd
    if branch == "keep":
        return 0 if rel_even_majority else 1
    if branch == "mirror":
        return 1 if rel_even_majority else 0
    raise AssertionError(f"unexpected branch: {branch}")


def _find_placement(
    reserved: np.ndarray,
    shape: np.ndarray,
    required_col_parity: int | None,
) -> tuple[int, int] | None:
    grid_h, grid_w = reserved.shape
    shape_h, shape_w = shape.shape
    rows = list(range(1, grid_h - shape_h))
    cols = list(range(1, grid_w - shape_w))
    if required_col_parity is not None:
        cols = [c for c in cols if c % 2 == required_col_parity]
    if not rows or not cols:
        return None

    candidates = [(r, c) for r in rows for c in cols]
    random.shuffle(candidates)
    for r0, c0 in candidates[:GEN_PLACEMENT_RETRIES]:
        if reserved[r0 - 1 : r0 + shape_h + 1, c0 - 1 : c0 + shape_w + 1].any():
            continue
        return r0, c0
    return None


def _mark_reserved(reserved: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> None:
    shape_h, shape_w = shape.shape
    reserved[r0 - 1 : r0 + shape_h + 1, c0 - 1 : c0 + shape_w + 1] = True


def _paint_shape(grid: np.ndarray, shape: np.ndarray, color: int, r0: int, c0: int) -> None:
    rows, cols = np.nonzero(shape)
    grid[r0 + rows, c0 + cols] = color


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.choice(GEN_WIDTH_OPTIONS)
    object_count = random.randint(*GEN_OBJECT_COUNT_RANGE)
    if object_count > len(GEN_PALETTE):
        return None

    branches = ["keep", "mirror", "delete"]
    branches.extend(random.choice(BRANCHES) for _ in range(object_count - len(branches)))
    random.shuffle(branches)
    colors = random.sample(GEN_PALETTE, object_count)

    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for branch, color in zip(branches, colors):
        shape = _shape_for_branch(branch)
        if shape is None:
            return None
        col_parity = _placement_column_parity(shape, branch) if branch != "delete" else None
        placement = _find_placement(reserved, shape, col_parity)
        if placement is None:
            return None
        r0, c0 = placement
        _paint_shape(grid, shape, color, r0, c0)
        _mark_reserved(reserved, shape, r0, c0)

    objects = _components(grid)
    if len(objects) != object_count:
        return None
    actual_branches = {_branch_for_cells(cells) for _color, cells in objects}
    if actual_branches != set(BRANCHES):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
