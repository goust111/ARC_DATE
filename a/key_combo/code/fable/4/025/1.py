from __future__ import annotations

import numpy as np


GEN_RETRIES = 300


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _try_candidate(grid: np.ndarray, ph: int, pw: int):
    h, w = map(int, grid.shape)
    tile_rows = h // ph
    tile_cols = w // pw

    base = np.empty((ph, pw), dtype=int)
    defect_color: int | None = None

    for lr in range(ph):
        for lc in range(pw):
            values = grid[lr:h:ph, lc:w:pw].reshape(-1)
            colors, counts = np.unique(values, return_counts=True)
            if len(colors) > 2:
                return None

            if len(colors) == 1:
                base[lr, lc] = int(colors[0])
                continue

            order = np.argsort(-counts)
            if int(counts[order[0]]) == int(counts[order[1]]):
                return None

            majority = int(colors[order[0]])
            minority = int(colors[order[1]])
            if defect_color is None:
                defect_color = minority
            elif minority != defect_color:
                return None
            base[lr, lc] = majority

    if defect_color is None:
        return None
    if defect_color in set(map(int, base.reshape(-1))):
        return None

    out = np.zeros((tile_rows, tile_cols), dtype=int)
    seen_local_positions: set[tuple[int, int]] = set()
    defect_tiles = 0

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            block = grid[tr * ph : (tr + 1) * ph, tc * pw : (tc + 1) * pw]
            mismatch = np.argwhere(block != base)
            if mismatch.size == 0:
                continue
            if int(mismatch.shape[0]) != 1:
                return None

            lr, lc = map(int, mismatch[0])
            if int(block[lr, lc]) != defect_color:
                return None
            if (lr, lc) in seen_local_positions:
                return None

            seen_local_positions.add((lr, lc))
            out[tr, tc] = int(base[lr, lc])
            defect_tiles += 1

    if defect_tiles == 0 or defect_tiles == tile_rows * tile_cols:
        return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the repeating tile, find corrupted tile instances, and draw their hidden original colors on the tile lattice.

    Input structure:
    - The full grid is an exact rectangular tiling of one unknown base tile whose height and width divide the grid size.
    - A small number of tile instances each contain exactly one overwritten pixel, all changed to the same defect color.
    - The defect color does not belong to the true base tile, and different corrupted tiles use different local defect coordinates.

    Rule:
    - Infer the smallest valid tile period by checking divisor-sized candidates and taking the majority color at each congruence class.
    - For every corrupted tile instance, locate its one defect pixel, look up the original color from the inferred base tile at that local coordinate, and place that color on the coarse tile-position grid.
    - Fill every clean tile position with black.

    Key insight:
    - The visible defect color is only a shared eraser; the answer comes from projecting defective tile positions onto the coarser tile grid and restoring the hidden color from the repeated motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for ph in _divisors(h):
        for pw in _divisors(w):
            out = _try_candidate(g, ph, pw)
            if out is not None:
                candidates.append((ph * pw, ph, pw, out))

    assert candidates, "failed to infer a valid repeating tile"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _has_smaller_period(tile: np.ndarray) -> bool:
    ph, pw = map(int, tile.shape)
    for h2 in _divisors(ph):
        for w2 in _divisors(pw):
            if h2 == ph and w2 == pw:
                continue
            if ph % h2 != 0 or pw % w2 != 0:
                continue
            pattern = tile[:h2, :w2]
            if np.array_equal(tile, np.tile(pattern, (ph // h2, pw // w2))):
                return True
    return False


def _build_base_tile(ph: int, pw: int, rng: np.random.Generator) -> np.ndarray | None:
    colors = list(map(int, rng.permutation(np.arange(1, 10))))
    tile = np.full((ph, pw), colors[0], dtype=int)

    stripe_row = int(rng.integers(0, ph))
    stripe_col = int(rng.integers(0, pw))
    tile[stripe_row, :] = colors[1]
    tile[:, stripe_col] = colors[2]

    if ph * pw >= 6:
        accent_r = int(rng.integers(0, ph))
        accent_c = int(rng.integers(0, pw))
        tile[accent_r, accent_c] = colors[3]

    if ph >= 3 and pw >= 3:
        r0 = int(rng.integers(0, ph - 1))
        c0 = int(rng.integers(0, pw - 1))
        if rng.random() < 0.5:
            tile[r0 : r0 + 2, c0] = colors[4]
        else:
            tile[r0, c0 : c0 + 2] = colors[4]

    if len(set(map(int, tile.reshape(-1)))) < 3:
        return None
    if _has_smaller_period(tile):
        return None
    return tile


def _choose_layout(rng: np.random.Generator):
    for _ in range(100):
        ph = int(rng.integers(2, 6))
        pw = int(rng.integers(2, 6))
        max_tile_rows = min(6, 30 // ph)
        max_tile_cols = min(6, 30 // pw)
        if max_tile_rows < 3 or max_tile_cols < 3:
            continue
        tile_rows = int(rng.integers(3, max_tile_rows + 1))
        tile_cols = int(rng.integers(3, max_tile_cols + 1))
        if tile_rows * tile_cols >= 6 and ph * pw >= 4:
            return ph, pw, tile_rows, tile_cols
    return None


def _neighbors4(rows: int, cols: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _connected_coarse_cells(rows: int, cols: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    start = (int(rng.integers(0, rows)), int(rng.integers(0, cols)))
    chosen = {start}
    while len(chosen) < count:
        frontier = set()
        for r, c in chosen:
            frontier.update(_neighbors4(rows, cols, r, c))
        frontier.difference_update(chosen)
        options = sorted(frontier)
        pick = options[int(rng.integers(0, len(options)))]
        chosen.add(pick)
    cells = list(chosen)
    rng.shuffle(cells)
    return cells


def _sample_coarse_cells(rows: int, cols: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if rng.random() < 0.7:
        return _connected_coarse_cells(rows, cols, count, rng)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    picks = rng.choice(len(all_cells), size=count, replace=False)
    return [all_cells[int(i)] for i in picks]


def _sample_local_positions(tile: np.ndarray, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    ph, pw = map(int, tile.shape)
    all_cells = [(r, c) for r in range(ph) for c in range(pw)]
    tile_colors = {int(tile[r, c]) for r, c in all_cells}
    needed_colors = 2 if len(tile_colors) >= 2 and count >= 2 else 1

    for _ in range(100):
        picks = rng.choice(len(all_cells), size=count, replace=False)
        cells = [all_cells[int(i)] for i in picks]
        colors = {int(tile[r, c]) for r, c in cells}
        if len(colors) >= needed_colors:
            return cells
    return None


def _generate_one(rng: np.random.Generator) -> np.ndarray | None:
    layout = _choose_layout(rng)
    if layout is None:
        return None
    ph, pw, tile_rows, tile_cols = layout

    tile = _build_base_tile(ph, pw, rng)
    if tile is None:
        return None

    total_tiles = tile_rows * tile_cols
    max_defects = min(ph * pw, total_tiles - 1, 7)
    min_defects = 3 if max_defects >= 3 else 2
    if max_defects < min_defects:
        return None
    defect_count = int(rng.integers(min_defects, max_defects + 1))

    coarse_cells = _sample_coarse_cells(tile_rows, tile_cols, defect_count, rng)
    local_cells = _sample_local_positions(tile, defect_count, rng)
    if local_cells is None:
        return None

    defect_palette = [color for color in range(1, 10) if color not in set(map(int, tile.reshape(-1)))]
    if not defect_palette:
        return None
    defect_color = int(rng.choice(defect_palette))

    grid = np.tile(tile, (tile_rows, tile_cols))
    expected = np.zeros((tile_rows, tile_cols), dtype=int)
    for (tr, tc), (lr, lc) in zip(coarse_cells, local_cells):
        grid[tr * ph + lr, tc * pw + lc] = defect_color
        expected[tr, tc] = int(tile[lr, lc])

    if np.count_nonzero(expected) < 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_one(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
