import numpy as np


MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _structure_degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    return sum(1 for rr, cc in _neighbors4(r, c, h, w) if bool(mask[rr, cc]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each branch by its endpoint marker color and extend a same-color ray
    beyond the marker by the branch's original grey length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    non_bg = g[g != bg]
    if non_bg.size == 0:
        return g.copy()

    values, counts = np.unique(non_bg, return_counts=True)
    structure_color = int(values[np.argmax(counts)])
    structure_mask = g == structure_color

    out = g.copy()
    marker_coords = np.argwhere((g != bg) & (g != structure_color))

    for mr, mc in marker_coords:
        mr = int(mr)
        mc = int(mc)
        marker_color = int(g[mr, mc])
        adjacent_structure = [
            (rr, cc)
            for rr, cc in _neighbors4(mr, mc, h, w)
            if bool(structure_mask[rr, cc])
        ]
        assert len(adjacent_structure) == 1, "each marker must touch exactly one branch pixel"

        br, bc = adjacent_structure[0]
        dr_out = mr - br
        dc_out = mc - bc
        dr_in = -dr_out
        dc_in = -dc_out

        branch_pixels: list[tuple[int, int]] = []
        cur_r, cur_c = br, bc

        while True:
            branch_pixels.append((int(cur_r), int(cur_c)))
            next_r = int(cur_r + dr_in)
            next_c = int(cur_c + dc_in)
            assert 0 <= next_r < h and 0 <= next_c < w, "branch must reconnect to trunk"
            assert bool(structure_mask[next_r, next_c]), "branch must stay on the grey tree"
            if _structure_degree(structure_mask, next_r, next_c) >= 3:
                break
            cur_r, cur_c = next_r, next_c

        for rr, cc in branch_pixels:
            out[rr, cc] = marker_color

        ray_r, ray_c = mr, mc
        for _ in range(len(branch_pixels)):
            assert 0 <= ray_r < h and 0 <= ray_c < w, "ray must stay in bounds"
            out[ray_r, ray_c] = marker_color
            ray_r += dr_out
            ray_c += dc_out

    return out


def _choose_colors(branch_count: int) -> tuple[int, int, list[int]]:
    colors = list(RNG.choice(PALETTE, size=branch_count + 2, replace=False))
    bg = int(colors[0])
    structure = int(colors[1])
    markers = [int(c) for c in colors[2:]]
    return bg, structure, markers


def _balanced_directions(branch_count: int) -> list[int]:
    dirs = [-1, 1]
    while len(dirs) < branch_count:
        dirs.append(int(RNG.choice([-1, 1])))
    RNG.shuffle(dirs)
    return dirs


def _gen_vertical() -> np.ndarray | None:
    h = int(RNG.integers(10, 19))
    w = int(RNG.integers(10, 19))
    trunk_len = int(RNG.integers(5, h))
    start_r = int(RNG.integers(0, h - trunk_len + 1))
    trunk_rows = np.arange(start_r, start_r + trunk_len)
    interior_rows = trunk_rows[1:-1]
    max_branches = min(5, (len(interior_rows) + 1) // 2)
    if max_branches < 2:
        return None
    branch_count = int(RNG.integers(2, max_branches + 1))
    bg, structure, marker_colors = _choose_colors(branch_count)
    grid = np.full((h, w), bg, dtype=int)

    trunk_c = int(RNG.integers(2, w - 2))
    left_cap = trunk_c // 2
    right_cap = (w - 1 - trunk_c) // 2
    if left_cap < 1 or right_cap < 1:
        return None

    row_options = [interior_rows[::2], interior_rows[1::2]]
    row_options = [rows for rows in row_options if len(rows) >= branch_count]
    if not row_options:
        return None
    chosen_pool = row_options[int(RNG.integers(0, len(row_options)))]
    chosen_rows = sorted(int(x) for x in RNG.choice(chosen_pool, size=branch_count, replace=False))
    directions = _balanced_directions(branch_count)

    grid[trunk_rows, trunk_c] = structure

    for row, direction, marker_color in zip(chosen_rows, directions, marker_colors):
        cap = left_cap if direction < 0 else right_cap
        if cap < 1:
            return None
        length = int(RNG.integers(1, cap + 1))
        for offset in range(1, length + 1):
            grid[row, trunk_c + direction * offset] = structure
        grid[row, trunk_c + direction * (length + 1)] = marker_color

    return grid


def _gen_horizontal() -> np.ndarray | None:
    h = int(RNG.integers(10, 19))
    w = int(RNG.integers(10, 19))
    trunk_len = int(RNG.integers(5, w))
    start_c = int(RNG.integers(0, w - trunk_len + 1))
    trunk_cols = np.arange(start_c, start_c + trunk_len)
    interior_cols = trunk_cols[1:-1]
    max_branches = min(5, (len(interior_cols) + 1) // 2)
    if max_branches < 2:
        return None
    branch_count = int(RNG.integers(2, max_branches + 1))
    bg, structure, marker_colors = _choose_colors(branch_count)
    grid = np.full((h, w), bg, dtype=int)

    trunk_r = int(RNG.integers(2, h - 2))
    up_cap = trunk_r // 2
    down_cap = (h - 1 - trunk_r) // 2
    if up_cap < 1 or down_cap < 1:
        return None

    col_options = [interior_cols[::2], interior_cols[1::2]]
    col_options = [cols for cols in col_options if len(cols) >= branch_count]
    if not col_options:
        return None
    chosen_pool = col_options[int(RNG.integers(0, len(col_options)))]
    chosen_cols = sorted(int(x) for x in RNG.choice(chosen_pool, size=branch_count, replace=False))
    directions = _balanced_directions(branch_count)

    grid[trunk_r, trunk_cols] = structure

    for col, direction, marker_color in zip(chosen_cols, directions, marker_colors):
        cap = up_cap if direction < 0 else down_cap
        if cap < 1:
            return None
        length = int(RNG.integers(1, cap + 1))
        for offset in range(1, length + 1):
            grid[trunk_r + direction * offset, col] = structure
        grid[trunk_r + direction * (length + 1), col] = marker_color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        if bool(RNG.integers(0, 2)):
            grid = _gen_vertical()
        else:
            grid = _gen_horizontal()
        if grid is None:
            continue

        bg = _get_bg(grid)
        out = main(grid)
        assert isinstance(out, np.ndarray)
        if np.array_equal(out, grid):
            continue
        if int(np.count_nonzero(grid != bg)) < 6:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
