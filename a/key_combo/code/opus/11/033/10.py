from __future__ import annotations

import numpy as np


BG = 0
INPUT_COLORS = (1, 2, 3)
RAY_COLOR = 4

MIN_H = 6
MAX_H = 10
MIN_W = 5
MAX_W = 15
GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _trigger_rows(grid: np.ndarray) -> list[int]:
    """Rows whose left and right edge pixels become adjacent after horizontal tiling."""
    height, width = grid.shape
    return [
        r
        for r in range(height)
        if int(grid[r, 0]) != BG and int(grid[r, width - 1]) != BG
    ]


def _ray_path_in_input(height: int, width: int, row: int, side: str) -> list[tuple[int, int]]:
    """
    Return the input-grid cells copied under one output ray path.

    side == "left": cells hit by the up-left ray from the left-half seam pixel.
    side == "right": cells whose right-half copies are hit by the up-right ray.
    """
    assert side in {"left", "right"}
    coords: list[tuple[int, int]] = []
    for step in range(1, min(row, width - 1) + 1):
        rr = row - step
        cc = width - 1 - step if side == "left" else step
        coords.append((rr, cc))
    return coords


def _independent_expected_output(grid: np.ndarray) -> np.ndarray:
    """Second implementation used by the generator self-checks."""
    left = np.array(grid, dtype=int)
    height, width = left.shape
    out = np.concatenate([left, left], axis=1)

    for row in range(height):
        if int(left[row, 0]) == BG or int(left[row, width - 1]) == BG:
            continue

        rr = row - 1
        cc = width - 2
        while rr >= 0 and cc >= 0:
            if int(out[rr, cc]) == BG:
                out[rr, cc] = RAY_COLOR
            rr -= 1
            cc -= 1

        rr = row - 1
        cc = width + 1
        while rr >= 0 and cc < 2 * width:
            if int(out[rr, cc]) == BG:
                out[rr, cc] = RAY_COLOR
            rr -= 1
            cc += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the full input once to the right, then emit upward diagonal yellow rays from seam-adjacent edge pairs.

    Input structure:
    - The grid is a black background with a sparse set of blue, red, and green pixels.
    - Yellow does not appear in the input.
    - The leftmost and rightmost input columns each contain at least one non-black pixel.
    - Any row whose two edge pixels are both non-black will create a seam pair after tiling.

    Rule:
    - Duplicate the entire grid once to the right, producing two identical horizontal halves.
    - For every row whose leftmost and rightmost input cells are both non-black, treat the left half's right-edge pixel and the right half's left-edge pixel as an adjacent seam pair.
    - From the left seam pixel, trace an up-left diagonal ray; from the right seam pixel, trace an up-right diagonal ray.
    - Paint yellow only on black cells along those rays, never overwriting any existing non-black pixel, and keep all tiled content unchanged.

    Key insight:
    - The trigger is not present in the original grid: horizontal tiling makes the original right edge and the copied left edge touch at the seam, and only those new seam adjacencies activate the diagonal rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 0 < height <= 30 and 0 < width <= 15, "input must fit ARC size limits after doubling"
    assert 2 * width <= 30, "tiled output would exceed ARC width limit"

    out = np.concatenate([g, g], axis=1)

    for row in range(height):
        if int(g[row, 0]) == BG or int(g[row, width - 1]) == BG:
            continue

        rr, cc = row - 1, width - 2
        while rr >= 0 and cc >= 0:
            if int(out[rr, cc]) == BG:
                out[rr, cc] = RAY_COLOR
            rr -= 1
            cc -= 1

        rr, cc = row - 1, width + 1
        while rr >= 0 and cc < 2 * width:
            if int(out[rr, cc]) == BG:
                out[rr, cc] = RAY_COLOR
            rr -= 1
            cc += 1

    return out


def _choose_trigger_rows(height: int, trigger_count: int) -> list[int]:
    rows = list(range(height))
    if trigger_count == 1:
        return [int(_RNG.integers(1, height))]

    if _RNG.random() < 0.35:
        short_row = 1
        other_choices = [r for r in rows if r != short_row]
        other_row = int(_RNG.choice(other_choices))
        return sorted([short_row, other_row])

    chosen = _RNG.choice(rows, size=2, replace=False)
    if np.all(chosen == 0):
        chosen[0] = 1
    return sorted(int(x) for x in chosen)


def _paint_edge_rows(grid: np.ndarray, trigger_rows: list[int], edge_only_rows: list[int]) -> None:
    for row in trigger_rows:
        grid[row, 0] = int(_RNG.choice(INPUT_COLORS))
        grid[row, -1] = int(_RNG.choice(INPUT_COLORS))

    edge_only_sides = _RNG.choice(["left", "right"], size=len(edge_only_rows), replace=True)
    for row, side in zip(edge_only_rows, edge_only_sides):
        color = int(_RNG.choice(INPUT_COLORS))
        if side == "left":
            grid[row, 0] = color
        else:
            grid[row, -1] = color


def _available_blocker_cells(
    grid: np.ndarray, trigger_rows: list[int], side: str
) -> list[tuple[int, int, int]]:
    """Return (row, r, c) triples for possible blocker placements on ray paths."""
    height, width = grid.shape
    assert height > 0
    cells: list[tuple[int, int, int]] = []
    for row in trigger_rows:
        for rr, cc in _ray_path_in_input(height, width, row, side):
            if cc in (0, width - 1):
                continue
            if int(grid[rr, cc]) != BG:
                continue
            cells.append((row, rr, cc))
    return cells


def _insert_blockers(grid: np.ndarray, trigger_rows: list[int], scene: str) -> None:
    block_sides: list[str]
    if scene == "clear":
        return
    if scene == "blocked_single":
        block_sides = [str(_RNG.choice(["left", "right"]))]
    elif scene == "blocked_double":
        block_sides = ["left", "right"]
    else:
        raise AssertionError(f"unknown scene {scene}")

    for side in block_sides:
        candidates = _available_blocker_cells(grid, trigger_rows, side)
        if not candidates:
            continue
        _RNG.shuffle(candidates)
        chosen_count = 1 if len(candidates) == 1 else int(_RNG.integers(1, min(3, len(candidates)) + 1))
        for _row, rr, cc in candidates[:chosen_count]:
            grid[rr, cc] = int(_RNG.choice(INPUT_COLORS))


def _non_ray_interior_cells(grid: np.ndarray, trigger_rows: list[int]) -> list[tuple[int, int]]:
    height, width = grid.shape
    forbidden = set()
    for row in trigger_rows:
        forbidden.update(_ray_path_in_input(height, width, row, "left"))
        forbidden.update(_ray_path_in_input(height, width, row, "right"))

    cells: list[tuple[int, int]] = []
    for rr in range(height):
        for cc in range(1, width - 1):
            if (rr, cc) in forbidden:
                continue
            if int(grid[rr, cc]) != BG:
                continue
            cells.append((rr, cc))
    return cells


def _add_fillers(grid: np.ndarray, trigger_rows: list[int]) -> None:
    candidates = _non_ray_interior_cells(grid, trigger_rows)
    if not candidates:
        return

    _RNG.shuffle(candidates)
    filler_count = min(len(candidates), int(_RNG.integers(1, min(6, len(candidates)) + 1)))
    for rr, cc in candidates[:filler_count]:
        grid[rr, cc] = int(_RNG.choice(INPUT_COLORS))


def _validate_example(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False

    height, width = grid.shape
    if not (MIN_H <= height <= MAX_H and MIN_W <= width <= MAX_W and 2 * width <= 30):
        return False
    if np.any(grid == RAY_COLOR):
        return False
    if np.array_equal(grid, grid[:, ::-1]):
        return False
    if not np.any(grid[:, 0] != BG) or not np.any(grid[:, width - 1] != BG):
        return False

    trigger_rows = _trigger_rows(grid)
    if len(trigger_rows) not in (1, 2):
        return False

    edge_only_rows = [
        r
        for r in range(height)
        if (int(grid[r, 0]) != BG) ^ (int(grid[r, width - 1]) != BG)
    ]
    if not edge_only_rows:
        return False
    if not np.any(grid[:, 1 : width - 1] != BG):
        return False

    if len(set(map(int, grid[grid != BG]))) < 2:
        return False

    expected = _independent_expected_output(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, np.concatenate([grid, grid], axis=1)):
        return False
    if not np.array_equal(actual[:, :width], expected[:, :width]):
        return False
    if not np.array_equal(actual[:, width:], expected[:, width:]):
        return False

    for row in trigger_rows:
        modified = grid.copy()
        modified[row, 0] = BG
        if np.array_equal(main(modified), actual):
            return False
        modified = grid.copy()
        modified[row, width - 1] = BG
        if np.array_equal(main(modified), actual):
            return False

        row_changes = np.argwhere(actual != np.concatenate([grid, grid], axis=1))
        if not np.any(row_changes[:, 0] < row):
            return False

    yellow_mask = actual == RAY_COLOR
    if not np.any(yellow_mask):
        return False

    legal_yellow = np.zeros_like(actual, dtype=bool)
    for row in trigger_rows:
        for rr, cc in _ray_path_in_input(height, width, row, "left"):
            if int(grid[rr, cc]) == BG:
                legal_yellow[rr, cc] = True
        for rr, cc in _ray_path_in_input(height, width, row, "right"):
            if int(grid[rr, cc]) == BG:
                legal_yellow[rr, width + cc] = True
    if not np.array_equal(yellow_mask, legal_yellow):
        return False

    for rr, cc in np.argwhere(yellow_mask):
        if int(np.concatenate([grid, grid], axis=1)[rr, cc]) != BG:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(MIN_H, MAX_H + 1))
    width = int(_RNG.integers(MIN_W, MAX_W + 1))
    grid = np.zeros((height, width), dtype=int)

    scene = str(_RNG.choice(["clear", "blocked_single", "blocked_double"]))
    trigger_count = 1 if _RNG.random() < 0.45 else 2
    trigger_rows = _choose_trigger_rows(height, trigger_count)

    remaining_rows = [r for r in range(height) if r not in trigger_rows]
    if not remaining_rows:
        return None
    edge_only_count = 1 if len(remaining_rows) == 1 else int(_RNG.integers(1, min(3, len(remaining_rows)) + 1))
    edge_only_rows = sorted(int(x) for x in _RNG.choice(remaining_rows, size=edge_only_count, replace=False))

    _paint_edge_rows(grid, trigger_rows, edge_only_rows)
    _insert_blockers(grid, trigger_rows, scene)
    _add_fillers(grid, trigger_rows)

    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
