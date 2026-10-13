import numpy as np


BG = 0
WALL = 7
PALETTE = np.array([1, 2, 3, 4, 5, 6, 8, 9], dtype=int)
MAX_GRID = 30
MAX_RETRIES = 400


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        cur = int(value)
        if cur == prev + 1:
            prev = cur
            continue
        out.append((start, prev))
        start = prev = cur
    out.append((start, prev))
    return out


def _separator_candidates(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]]:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        row_mask = np.all(grid == color, axis=1)
        col_mask = np.all(grid == color, axis=0)
        if not row_mask.any() or not col_mask.any():
            continue

        row_segs = _segments(row_mask)
        col_segs = _segments(col_mask)
        if any(r1 - r0 != 0 for r0, r1 in row_segs):
            continue
        if any(c1 - c0 != 0 for c0, c1 in col_segs):
            continue

        row_bands = _segments(~row_mask)
        col_bands = _segments(~col_mask)
        if not row_bands or not col_bands:
            continue
        if len(row_bands) * len(col_bands) <= 1:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_bands}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_bands}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        if len(row_segs) < len(row_bands) - 1 or len(row_segs) > len(row_bands) + 1:
            continue
        if len(col_segs) < len(col_bands) - 1 or len(col_segs) > len(col_bands) + 1:
            continue

        ok = True
        for r0, r1 in row_bands:
            for c0, c1 in col_bands:
                patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                fg = patch[(patch != BG) & (patch != color)]
                if fg.size == 0:
                    ok = False
                    break
                if np.unique(fg).size != 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((color, row_bands, col_bands))
    return candidates


def _partition_info(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates = _separator_candidates(grid)
    assert candidates, "failed to detect a valid separator lattice"
    candidates.sort(key=lambda item: (item[0] != WALL, -len(item[1]) * len(item[2])))
    return candidates[0]


def _serpentine_order(h: int, w: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    transpose = bool(rng.integers(0, 2))
    reverse_outer = bool(rng.integers(0, 2))
    reverse_first = bool(rng.integers(0, 2))

    outer = list(range(w if transpose else h))
    if reverse_outer:
        outer.reverse()

    order: list[tuple[int, int]] = []
    for i, major in enumerate(outer):
        inner = list(range(h if transpose else w))
        if (i % 2 == 0) == reverse_first:
            inner.reverse()
        for minor in inner:
            cell = (minor, major) if transpose else (major, minor)
            order.append(cell)
    return order


def _growth_mask(h: int, w: int, count: int, rng: np.random.Generator, lifo: bool) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    frontier = [start]
    seen = {start}
    order: list[tuple[int, int]] = []

    while frontier:
        cell = frontier.pop(-1 if lifo else 0)
        order.append(cell)
        r, c = cell
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            nbr = (rr, cc)
            if 0 <= rr < h and 0 <= cc < w and nbr not in seen:
                seen.add(nbr)
                nbrs.append(nbr)
        if nbrs:
            nbrs = [nbrs[int(i)] for i in rng.permutation(len(nbrs))]
            frontier.extend(nbrs)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in order[:count]:
        mask[r, c] = True
    return mask


def _rectangular_mask(h: int, w: int, count: int, rng: np.random.Generator) -> np.ndarray:
    candidates = []
    for rh in range(1, h + 1):
        for rw in range(1, w + 1):
            area = rh * rw
            if area <= count:
                candidates.append((area, rh, rw))
    areas = np.array([area for area, _rh, _rw in candidates], dtype=float)
    pick = candidates[int(rng.choice(len(candidates), p=areas / areas.sum()))]
    rect_area, rh, rw = pick
    r0 = int(rng.integers(0, h - rh + 1))
    c0 = int(rng.integers(0, w - rw + 1))

    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + rh, c0 : c0 + rw] = True

    frontier = set()
    for r in range(r0, r0 + rh):
        for c in range(c0, c0 + rw):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.add((rr, cc))

    remaining = count - rect_area
    while remaining > 0:
        frontier_list = list(frontier)
        cell = frontier_list[int(rng.integers(0, len(frontier_list)))]
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        remaining -= 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                frontier.add((rr, cc))

    return mask


def _make_mask(h: int, w: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count == h * w:
        return np.ones((h, w), dtype=bool)

    mode = int(rng.integers(0, 4))
    if mode == 0:
        return _growth_mask(h, w, count, rng, lifo=True)
    if mode == 1:
        return _growth_mask(h, w, count, rng, lifo=False)
    if mode == 2:
        mask = np.zeros((h, w), dtype=bool)
        for r, c in _serpentine_order(h, w, rng)[:count]:
            mask[r, c] = True
        return mask
    return _rectangular_mask(h, w, count, rng)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the partitioned subregions so denser single-color motifs move to earlier lattice slots.

    Input structure:
    - The background is Black.
    - One-pixel-thick Orange separator lines form a full row/column lattice that splits the grid into equal rectangular subregions.
    - Each subregion contains a non-empty motif drawn in exactly one non-Black, non-Orange color.

    Rule:
    - Detect the separator lattice and extract every subregion in row-major order.
    - For each subregion, count its density as the number of non-Black cells that are not separator cells.
    - Sort the subregions from highest density to lowest density, breaking ties by their original row-major order, and place them back row by row from the top-left.
    - Keep every subregion's internal pixels unchanged and preserve the separator-line layout.

    Key insight:
    - The Orange grid turns the puzzle into a collection of movable rectangular tiles; the only property that determines their destination is the visible amount of colored material inside each tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= MAX_GRID

    wall_color, row_bands, col_bands = _partition_info(g)
    positions = [(rb, cb) for rb in row_bands for cb in col_bands]

    tiles: list[tuple[int, np.ndarray]] = []
    for (r0, r1), (c0, c1) in positions:
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        density = int(np.count_nonzero((patch != BG) & (patch != wall_color)))
        tiles.append((density, patch))

    order = sorted(range(len(tiles)), key=lambda idx: (-tiles[idx][0], idx))
    out = g.copy()
    for dst_idx, src_idx in enumerate(order):
        (r0, r1), (c0, c1) = positions[dst_idx]
        out[r0 : r1 + 1, c0 : c1 + 1] = tiles[src_idx][1]
    return out


def _layout_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, bool] | None:
    nr = int(rng.integers(2, 5))
    nc = int(rng.integers(2, 5))
    bordered = bool(rng.integers(0, 2))

    sep_rows = nr + 1 if bordered else nr - 1
    sep_cols = nc + 1 if bordered else nc - 1
    max_h = (MAX_GRID - sep_rows) // nr
    max_w = (MAX_GRID - sep_cols) // nc

    candidates = [
        (h, w)
        for h in range(3, max_h + 1)
        for w in range(3, max_w + 1)
        if h * w >= nr * nc
    ]
    if not candidates:
        return None
    h, w = candidates[int(rng.integers(0, len(candidates)))]
    return nr, nc, h, w, bordered


def _build_lattice(
    nr: int,
    nc: int,
    cell_h: int,
    cell_w: int,
    bordered: bool,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = nr * cell_h + (nr + 1 if bordered else nr - 1)
    w = nc * cell_w + (nc + 1 if bordered else nc - 1)
    grid = np.full((h, w), BG, dtype=int)

    if bordered:
        row_starts = [1 + i * (cell_h + 1) for i in range(nr)]
        col_starts = [1 + j * (cell_w + 1) for j in range(nc)]
        wall_rows = [i * (cell_h + 1) for i in range(nr + 1)]
        wall_cols = [j * (cell_w + 1) for j in range(nc + 1)]
    else:
        row_starts = [i * (cell_h + 1) for i in range(nr)]
        col_starts = [j * (cell_w + 1) for j in range(nc)]
        wall_rows = [cell_h + i * (cell_h + 1) for i in range(nr - 1)]
        wall_cols = [cell_w + j * (cell_w + 1) for j in range(nc - 1)]

    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    regions: list[tuple[int, int, int, int]] = []
    for r0 in row_starts:
        for c0 in col_starts:
            regions.append((r0, c0, r0 + cell_h - 1, c0 + cell_w - 1))
    return grid, regions


def _sample_densities(area: int, count: int, rng: np.random.Generator) -> list[int]:
    densities = rng.choice(np.arange(1, area + 1), size=count, replace=False).tolist()
    sorted_desc = sorted((int(v) for v in densities), reverse=True)
    for _ in range(20):
        shuffled = [sorted_desc[int(i)] for i in rng.permutation(count)]
        if shuffled != sorted_desc:
            return shuffled
    shuffled = sorted_desc[:]
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    return shuffled


def _paint_regions(
    grid: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> None:
    area = (regions[0][2] - regions[0][0] + 1) * (regions[0][3] - regions[0][1] + 1)
    densities = _sample_densities(area, len(regions), rng)

    palette_size = int(rng.integers(2, min(6, len(PALETTE), len(regions)) + 1))
    local_palette = rng.choice(PALETTE, size=palette_size, replace=False)
    colors = rng.choice(local_palette, size=len(regions), replace=True)

    for idx, (r0, c0, r1, c1) in enumerate(regions):
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        patch[:] = BG
        mask = _make_mask(patch.shape[0], patch.shape[1], int(densities[idx]), rng)
        patch[mask] = int(colors[idx])


def _is_sorted_input(grid: np.ndarray) -> bool:
    wall_color, row_bands, col_bands = _partition_info(grid)
    densities = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            densities.append(int(np.count_nonzero((patch != BG) & (patch != wall_color))))
    return densities == sorted(densities, reverse=True)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    params = _layout_params(rng)
    if params is None:
        return None

    nr, nc, cell_h, cell_w, bordered = params
    grid, regions = _build_lattice(nr, nc, cell_h, cell_w, bordered)
    _paint_regions(grid, regions, rng)

    if _is_sorted_input(grid):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
