import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

FILL_BY_ROTATION = {
    0: BLUE,
    1: GREEN,
    2: YELLOW,
    3: ORANGE,
}

RNG = np.random.default_rng()
MAX_RETRIES = 300


def _clockwise_rot(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, -k)


def _find_separator_lines(grid: np.ndarray) -> tuple[list[int], list[int], int]:
    h, w = grid.shape
    row_colors = [int(grid[r, 0]) for r in range(h) if np.all(grid[r] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    shared = set(row_colors) & set(col_colors)
    assert len(shared) == 1, "expected exactly one separator color"
    sep = next(iter(shared))
    rows = [r for r in range(h) if np.all(grid[r] == sep)]
    cols = [c for c in range(w) if np.all(grid[:, c] == sep)]
    assert len(rows) >= 4 and len(cols) >= 4
    assert rows[0] == 0 and rows[-1] == h - 1
    assert cols[0] == 0 and cols[-1] == w - 1
    row_steps = np.diff(rows)
    col_steps = np.diff(cols)
    assert len(set(row_steps.tolist())) == 1
    assert len(set(col_steps.tolist())) == 1
    assert row_steps[0] == col_steps[0], "cells must be square"
    return rows, cols, sep


def _cell_slices(rows: list[int], cols: list[int], i: int, j: int) -> tuple[slice, slice]:
    return slice(rows[i] + 1, rows[i + 1]), slice(cols[j] + 1, cols[j + 1])


def _rotation_index(ref: np.ndarray, target: np.ndarray) -> int:
    for k in range(4):
        if np.array_equal(_clockwise_rot(ref, k), target):
            return k
    raise AssertionError("target sprite is not a rotation of the reference")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rows, cols, sep = _find_separator_lines(g)
    n = len(rows) - 1
    assert n == len(cols) - 1 and n >= 3

    bg = int(np.bincount(g.ravel()).argmax())
    assert bg != sep
    out = g.copy()

    tl_rs, tl_cs = _cell_slices(rows, cols, 0, 0)
    ref_cell = g[tl_rs, tl_cs]
    ref_mask = ref_cell != bg
    assert np.any(ref_mask)

    corner_expectations = {
        (0, 0): 0,
        (0, n - 1): 1,
        (n - 1, n - 1): 2,
        (n - 1, 0): 3,
    }
    for (i, j), rot in corner_expectations.items():
        rs, cs = _cell_slices(rows, cols, i, j)
        cell = g[rs, cs]
        assert np.array_equal(cell != bg, _clockwise_rot(ref_mask, rot))
        colors = {int(v) for v in np.unique(cell) if int(v) != bg}
        assert len(colors) == 1 and next(iter(colors)) != sep

    for i in range(n):
        for j in range(n):
            if (i, j) in corner_expectations:
                continue
            rs, cs = _cell_slices(rows, cols, i, j)
            cell = g[rs, cs]
            sprite_mask = cell != bg
            rot = _rotation_index(ref_mask, sprite_mask)
            filled = cell.copy()
            filled[cell == bg] = FILL_BY_ROTATION[rot]
            out[rs, cs] = filled

    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(pts[0])}
    stack = [tuple(pts[0])]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(pts)


def _random_sprite(side: int) -> np.ndarray | None:
    area_min = max(4, side + 1)
    area_max = side * side - 2
    for _ in range(100):
        mask = np.zeros((side, side), dtype=bool)
        start = (int(RNG.integers(side)), int(RNG.integers(side)))
        mask[start] = True
        target_area = int(RNG.integers(area_min, area_max + 1))
        frontier = {start}
        while mask.sum() < target_area and frontier:
            r, c = list(frontier)[int(RNG.integers(len(frontier)))]
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side and not mask[rr, cc]:
                    nbrs.append((rr, cc))
            if not nbrs:
                frontier.discard((r, c))
                continue
            rr, cc = nbrs[int(RNG.integers(len(nbrs)))]
            mask[rr, cc] = True
            frontier.add((rr, cc))
        if not _connected(mask):
            continue
        if any(np.array_equal(mask, _clockwise_rot(mask, k)) for k in (1, 2, 3)):
            continue
        if np.any(mask[0]) and np.any(mask[-1]) and np.any(mask[:, 0]) and np.any(mask[:, -1]):
            continue
        return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    n = int(RNG.integers(3, 6))
    cell_side = int(RNG.integers(6, 10))
    sprite_side = int(RNG.integers(3, cell_side - 1))
    sprite = _random_sprite(sprite_side)
    if sprite is None:
        return None

    grid_side = n * cell_side + (n + 1)
    grid = np.full((grid_side, grid_side), BG, dtype=int)
    for idx in range(0, grid_side, cell_side + 1):
        grid[idx, :] = GRAY
        grid[:, idx] = GRAY

    top_margin = int(RNG.integers(0, cell_side - sprite_side + 1))
    left_margin = int(RNG.integers(0, cell_side - sprite_side + 1))

    cells = [(i, j) for i in range(n) for j in range(n)]
    interior = [(i, j) for i, j in cells if (i, j) not in {(0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 0)}]
    rotations = [0, 1, 2, 3]
    RNG.shuffle(interior)
    cell_rotation = {
        (0, 0): 0,
        (0, n - 1): 1,
        (n - 1, n - 1): 2,
        (n - 1, 0): 3,
    }
    for idx, cell in enumerate(interior):
        if idx < 4:
            cell_rotation[cell] = rotations[idx]
        else:
            cell_rotation[cell] = int(RNG.integers(0, 4))

    for i, j in cells:
        rs, cs = _cell_slices(list(range(0, grid_side, cell_side + 1)), list(range(0, grid_side, cell_side + 1)), i, j)
        color = RED if (i, j) in {(0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 0)} else GRAY
        rot_sprite = _clockwise_rot(sprite, cell_rotation[(i, j)])
        sub = grid[rs, cs]
        sub[top_margin : top_margin + sprite_side, left_margin : left_margin + sprite_side][rot_sprite] = color
        grid[rs, cs] = sub

    try:
        out = main(grid)
    except AssertionError:
        return None

    used_fills = {int(v) for v in np.unique(out) if int(v) in FILL_BY_ROTATION.values()}
    if len(used_fills) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
