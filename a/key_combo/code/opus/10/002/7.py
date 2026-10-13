from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(1, 10))
SIZE_CHOICES = (4, 5)
ORIENTATIONS = ("row", "col")
MAX_GENERATION_ATTEMPTS = 800


def _find_guide(grid: np.ndarray) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    row_hits = [r for r in range(n) if int(g[r, 0]) != 0 and np.all(g[r, :] == int(g[r, 0]))]
    col_hits = [c for c in range(n) if int(g[0, c]) != 0 and np.all(g[:, c] == int(g[0, c]))]
    assert len(row_hits) + len(col_hits) == 1, "expected exactly one full non-black guide row or column"
    if row_hits:
        idx = int(row_hits[0])
        return "row", idx, int(g[idx, 0])
    idx = int(col_hits[0])
    return "col", idx, int(g[0, idx])


def _edge_flags(grid: np.ndarray) -> dict[str, bool]:
    g = np.asarray(grid, dtype=int)
    return {
        "top": bool(np.any(g[0, :] != 0)),
        "bottom": bool(np.any(g[-1, :] != 0)),
        "left": bool(np.any(g[:, 0] != 0)),
        "right": bool(np.any(g[:, -1] != 0)),
    }


def _main_band_cells(n: int, orientation: str, index: int) -> set[tuple[int, int]]:
    if orientation == "row":
        return {(index, c) for c in range(n)}
    return {(r, index) for r in range(n)}


def _propagate_cells(
    n: int,
    occupied: set[tuple[int, int]],
    edge_flags: dict[str, bool],
    *,
    one_round: bool,
) -> set[tuple[int, int]]:
    dirs = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    current = set(occupied)
    frontier = set(occupied)
    while frontier:
        new_cells: set[tuple[int, int]] = set()
        for r, c in frontier:
            for name, (dr, dc) in dirs.items():
                if not edge_flags[name]:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in current:
                    new_cells.add((nr, nc))
        if not new_cells:
            break
        current |= new_cells
        if one_round:
            break
        frontier = new_cells
    return current


def _render_blocks(tile: np.ndarray, occupied: set[tuple[int, int]]) -> np.ndarray:
    n = int(tile.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for gr, gc in occupied:
        r0 = gr * n
        c0 = gc * n
        out[r0 : r0 + n, c0 : c0 + n] = tile
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    orientation, index, _guide_color = _find_guide(g)
    occupied = _main_band_cells(n, orientation, index)
    occupied = _propagate_cells(n, occupied, _edge_flags(g), one_round=True)
    return _render_blocks(g, occupied)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the input along its guide band, then add one outward cascade layer wherever the tile has edge foreground.

    Input structure:
    - The input is a square n x n tile on black background, with n in the ARC-safe range where n^2 <= 30.
    - Exactly one full non-black row or one full non-black column is uniform; that line is the guide.
    - Other foreground pixels may appear elsewhere, including on selected tile edges.

    Rule:
    - Build an n x n block canvas of black n x n cells, so the output size is n^2 x n^2.
    - If the guide is row r, copy the whole input tile into every block of block-row r; if the guide is column c, copy it into every block of block-column c.
    - Inspect the input tile's four pixel edges. If its top, bottom, left, or right edge contains any non-black pixel, then each already placed tile tries once to copy itself into the adjacent block in that direction, but only if that adjacent block is empty and inside the canvas.
    - Newly copied blocks are not expanded again; overlapping attempts are harmless because every copied block is identical.

    Key insight:
    - The solid guide line chooses the primary tiled band, while the tile's own edge foreground decides which neighboring block cells receive a single extra copy, so both the guide position and the tile-edge content matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D tile"
    n = int(g.shape[0])
    assert n >= 1 and n * n <= 30, "tile size must stay within ARC limits"

    orientation, index, _guide_color = _find_guide(g)
    occupied = _main_band_cells(n, orientation, index)
    occupied = _propagate_cells(n, occupied, _edge_flags(g), one_round=True)
    return _render_blocks(g, occupied)


def _sample_guide_index(n: int) -> int:
    if random.random() < 0.38:
        return random.choice([0, n - 1])
    return random.randrange(n)


def _sample_active_sides(orientation: str, index: int, n: int) -> set[str]:
    if orientation == "row":
        candidates = [name for name, ok in (("top", index > 0), ("bottom", index < n - 1)) if ok]
    else:
        candidates = [name for name, ok in (("left", index > 0), ("right", index < n - 1)) if ok]
    assert candidates
    if len(candidates) == 1:
        return {candidates[0]}
    options = [{candidates[0]}, {candidates[1]}, set(candidates)]
    return set(random.choice(options))


def _non_guide_mask(n: int, orientation: str, index: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    if orientation == "row":
        mask[index, :] = False
    else:
        mask[:, index] = False
    return mask


def _paint_edge_row(grid: np.ndarray, row: int, colors: tuple[int, ...], min_count: int) -> None:
    n = int(grid.shape[0])
    count = random.randint(min_count, max(min_count, n - 1))
    cols = random.sample(range(n), count)
    for c in cols:
        grid[row, c] = random.choice(colors)


def _paint_edge_col(grid: np.ndarray, col: int, colors: tuple[int, ...], min_count: int) -> None:
    n = int(grid.shape[0])
    count = random.randint(min_count, max(min_count, n - 1))
    rows = random.sample(range(n), count)
    for r in rows:
        grid[r, col] = random.choice(colors)


def _block_values(grid: np.ndarray, n: int) -> set[tuple[int, ...]]:
    vals: set[tuple[int, ...]] = set()
    for gr in range(n):
        for gc in range(n):
            block = grid[gr * n : (gr + 1) * n, gc * n : (gc + 1) * n]
            vals.add(tuple(int(v) for v in block.reshape(-1)))
    return vals


def _validate_grid(grid: np.ndarray, *, require_longer_than_one_round: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        return False
    n = int(g.shape[0])
    if n not in SIZE_CHOICES or n * n > 30:
        return False

    try:
        orientation, index, guide_color = _find_guide(g)
    except AssertionError:
        return False

    if np.any(g < 0) or np.any(g > 9):
        return False
    nonguide = _non_guide_mask(n, orientation, index)
    if np.any(g[nonguide] == guide_color):
        return False
    if not np.any(g[nonguide] != 0):
        return False

    edge_pixels = np.argwhere(
        (g != 0)
        & (
            (np.arange(n)[:, None] == 0)
            | (np.arange(n)[:, None] == n - 1)
            | (np.arange(n)[None, :] == 0)
            | (np.arange(n)[None, :] == n - 1)
        )
    )
    if len(edge_pixels) < 2:
        return False

    flags = _edge_flags(g)
    if orientation == "row":
        has_effective_cascade = (index > 0 and flags["top"]) or (index < n - 1 and flags["bottom"])
    else:
        has_effective_cascade = (index > 0 and flags["left"]) or (index < n - 1 and flags["right"])
    if not has_effective_cascade:
        return False

    core = g[1:-1, 1:-1]
    if core.size and not np.any(core != 0):
        return False

    try:
        out = main(g)
    except AssertionError:
        return False
    if out.shape != (n * n, n * n):
        return False

    main_cells = _main_band_cells(n, orientation, index)
    one_round_cells = _propagate_cells(n, main_cells, flags, one_round=True)
    if len(one_round_cells) <= len(main_cells):
        return False
    if not np.array_equal(out, _render_blocks(g, one_round_cells)):
        return False

    for gr in range(n):
        for gc in range(n):
            block = out[gr * n : (gr + 1) * n, gc * n : (gc + 1) * n]
            if (gr, gc) in one_round_cells:
                if not np.array_equal(block, g):
                    return False
            elif np.any(block != 0):
                return False

    if _block_values(out, n) - {tuple([0] * (n * n)), tuple(int(v) for v in g.reshape(-1))}:
        return False

    full_cells = _propagate_cells(n, main_cells, flags, one_round=False)
    if require_longer_than_one_round and full_cells == one_round_cells:
        return False

    return True


def _attempt(require_longer_than_one_round: bool) -> np.ndarray | None:
    n = random.choice(SIZE_CHOICES)
    orientation = random.choice(ORIENTATIONS)
    guide_index = _sample_guide_index(n)

    colors = random.sample(PALETTE, random.randint(3, 4))
    guide_color = colors[0]
    fill_colors = tuple(colors[1:])

    grid = np.zeros((n, n), dtype=int)
    locked = np.zeros((n, n), dtype=bool)

    if orientation == "row":
        grid[guide_index, :] = guide_color
        locked[guide_index, :] = True
    else:
        grid[:, guide_index] = guide_color
        locked[:, guide_index] = True

    active_sides = _sample_active_sides(orientation, guide_index, n)
    if orientation == "row":
        if guide_index != 0:
            if "top" in active_sides:
                _paint_edge_row(grid, 0, fill_colors, min_count=2 if active_sides == {"top"} else 1)
            locked[0, :] = True
        if guide_index != n - 1:
            if "bottom" in active_sides:
                _paint_edge_row(grid, n - 1, fill_colors, min_count=2 if active_sides == {"bottom"} else 1)
            locked[n - 1, :] = True
    else:
        if guide_index != 0:
            if "left" in active_sides:
                _paint_edge_col(grid, 0, fill_colors, min_count=2 if active_sides == {"left"} else 1)
            locked[:, 0] = True
        if guide_index != n - 1:
            if "right" in active_sides:
                _paint_edge_col(grid, n - 1, fill_colors, min_count=2 if active_sides == {"right"} else 1)
            locked[:, n - 1] = True

    if orientation == "row":
        extra_border_cols = [0, n - 1]
        for col in extra_border_cols:
            for r in range(n):
                if locked[r, col]:
                    continue
                if random.random() < 0.35:
                    grid[r, col] = random.choice(fill_colors)
    else:
        extra_border_rows = [0, n - 1]
        for row in extra_border_rows:
            for c in range(n):
                if locked[row, c]:
                    continue
                if random.random() < 0.35:
                    grid[row, c] = random.choice(fill_colors)

    for r in range(n):
        for c in range(n):
            if locked[r, c]:
                continue
            if random.random() < 0.42:
                grid[r, c] = random.choice(fill_colors)

    interior_candidates = [
        (r, c)
        for r in range(1, n - 1)
        for c in range(1, n - 1)
        if not locked[r, c]
    ]
    if interior_candidates and not np.any(grid[1:-1, 1:-1] != 0):
        r, c = random.choice(interior_candidates)
        grid[r, c] = random.choice(fill_colors)

    if _validate_grid(grid, require_longer_than_one_round=require_longer_than_one_round):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        require_longer = random.random() < 0.6
        grid = _attempt(require_longer_than_one_round=require_longer)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-cascade tiling input")
