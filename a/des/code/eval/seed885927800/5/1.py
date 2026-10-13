import random
import numpy as np


BG = 0
DIVIDER = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
BAND_DEPTH = 2
SEED_OFFSET = 3
GEN_RETRIES = 400
QUADRANTS = ("TL", "TR", "BL", "BR")


def _find_dividers(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != BG]
    cols = [c for c in range(w) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != BG]
    assert len(rows) == 1 and len(cols) == 1
    row = int(rows[0])
    col = int(cols[0])
    divider = int(grid[row, 0])
    assert np.all(grid[:, col] == divider)
    return row, col, divider


def _quadrant_bounds(
    h: int, w: int, split_r: int, split_c: int, quadrant: str
) -> tuple[int, int, int, int]:
    if quadrant == "TL":
        return 0, split_r - 1, 0, split_c - 1
    if quadrant == "TR":
        return 0, split_r - 1, split_c + 1, w - 1
    if quadrant == "BL":
        return split_r + 1, h - 1, 0, split_c - 1
    if quadrant == "BR":
        return split_r + 1, h - 1, split_c + 1, w - 1
    raise AssertionError(quadrant)


def _band_strip_mask(shape: tuple[int, int], split_r: int, split_c: int, quadrant: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, r1, c0, c1 = _quadrant_bounds(h, w, split_r, split_c, quadrant)

    if quadrant[0] == "T":
        rows = range(max(r0, split_r - BAND_DEPTH), r1 + 1)
    else:
        rows = range(r0, min(r1, split_r + BAND_DEPTH) + 1)

    if quadrant[1] == "L":
        cols = range(max(c0, split_c - BAND_DEPTH), c1 + 1)
    else:
        cols = range(c0, min(c1, split_c + BAND_DEPTH) + 1)

    for r in rows:
        mask[r, c0 : c1 + 1] = True
    for c in cols:
        mask[r0 : r1 + 1, c] = True
    return mask


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    cells: list[tuple[int, int]] = []
    h, w = mask.shape
    while stack:
        cur = stack.pop()
        cells.append(cur)
        for nr, nc in _neighbors4(cur):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    assert len(seen) == int(mask.sum())
    return cells


def _first_two_colors_by_scan(arr: np.ndarray, bg: int) -> tuple[int, int]:
    seen: list[int] = []
    for value in arr.ravel():
        color = int(value)
        if color == bg or color in seen:
            continue
        seen.append(color)
        if len(seen) == 2:
            return seen[0], seen[1]
    raise AssertionError("expected two colors")


def _extract_seed(grid: np.ndarray, split_r: int, split_c: int, divider: int) -> np.ndarray:
    h, w = grid.shape
    colored = (grid != BG) & (grid != divider)
    strip = np.zeros((h, w), dtype=bool)
    for quadrant in QUADRANTS:
        strip |= _band_strip_mask((h, w), split_r, split_c, quadrant)
    seed_mask = colored & (~strip)
    seed_cells = np.argwhere(seed_mask)
    assert seed_cells.size > 0
    rows = sorted({int(r) for r, _ in seed_cells})
    cols = sorted({int(c) for _, c in seed_cells})
    assert 3 <= len(rows) <= 5 and 3 <= len(cols) <= 5
    seed = np.full((len(rows), len(cols)), BG, dtype=int)
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}
    for r, c in seed_cells:
        seed[row_index[int(r)], col_index[int(c)]] = int(grid[int(r), int(c)])
    assert len({int(v) for v in np.unique(seed) if int(v) != BG}) == 2
    assert all(np.any(seed[r, :] != BG) for r in range(seed.shape[0]))
    assert all(np.any(seed[:, c] != BG) for c in range(seed.shape[1]))
    return seed


def _walk_path(
    adjacency: dict[tuple[int, int], list[tuple[int, int]]], start: tuple[int, int]
) -> list[tuple[int, int]]:
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in adjacency[cur] if n != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _analyze_band(
    grid: np.ndarray, split_r: int, split_c: int, divider: int, quadrant: str
) -> tuple[int, tuple[int, int]]:
    strip = _band_strip_mask(grid.shape, split_r, split_c, quadrant)
    mask = strip & (grid != BG) & (grid != divider)
    cells = _component_from_mask(mask)
    cell_set = set(cells)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        adjacency[cell] = [n for n in _neighbors4(cell) if n in cell_set]
    endpoints = [cell for cell, neigh in adjacency.items() if len(neigh) == 1]
    assert len(endpoints) == 2

    same_row = len({r for r, _ in cells}) == 1
    same_col = len({c for _, c in cells}) == 1

    if same_row or same_col:
        if same_row:
            start = min(endpoints, key=lambda p: (p[1], p[0]))
        else:
            start = min(endpoints, key=lambda p: (p[0], p[1]))
        path = _walk_path(adjacency, start)
        return 2, _first_two_colors_by_scan(np.array([grid[r, c] for r, c in path]), BG)

    bends = []
    for cell, neigh in adjacency.items():
        if len(neigh) != 2:
            continue
        horizontal = any(nr == cell[0] for nr, _ in neigh)
        vertical = any(nc == cell[1] for _, nc in neigh)
        if horizontal and vertical:
            bends.append(cell)
    assert len(bends) == 1
    bend = bends[0]

    horizontal_endpoints = [p for p in endpoints if p[0] == bend[0]]
    vertical_endpoints = [p for p in endpoints if p[1] == bend[1]]
    assert len(horizontal_endpoints) == 1 and len(vertical_endpoints) == 1
    start = horizontal_endpoints[0]
    end = vertical_endpoints[0]
    path = _walk_path(adjacency, start)
    assert path[-1] == end

    d1 = (int(np.sign(bend[0] - start[0])), int(np.sign(bend[1] - start[1])))
    d2 = (int(np.sign(end[0] - bend[0])), int(np.sign(end[1] - bend[1])))
    turn = d1[0] * d2[1] - d1[1] * d2[0]
    assert turn != 0
    rotation = 1 if turn > 0 else 3
    return rotation, _first_two_colors_by_scan(np.array([grid[r, c] for r, c in path]), BG)


def _paste_centered(out: np.ndarray, sprite: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    r0, r1, c0, c1 = bounds
    qh = r1 - r0 + 1
    qw = c1 - c0 + 1
    sh, sw = sprite.shape
    top = r0 + (qh - sh) // 2
    left = c0 + (qw - sw) // 2
    assert top >= r0 and left >= c0
    assert top + sh - 1 <= r1 and left + sw - 1 <= c1
    for r in range(sh):
        for c in range(sw):
            color = int(sprite[r, c])
            if color == BG:
                continue
            out[top + r, left + c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate and recolor the central two-color seed into each quadrant according to the adjacent band."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    split_r, split_c, divider = _find_dividers(g)
    seed = _extract_seed(g, split_r, split_c, divider)
    seed_colors = _first_two_colors_by_scan(seed, BG)

    out = g.copy()
    for quadrant in QUADRANTS:
        rotation, band_colors = _analyze_band(g, split_r, split_c, divider, quadrant)
        sprite = np.rot90(seed, rotation)
        recolored = sprite.copy()
        recolored[sprite == seed_colors[0]] = band_colors[0]
        recolored[sprite == seed_colors[1]] = band_colors[1]
        _paste_centered(out, recolored, _quadrant_bounds(h, w, split_r, split_c, quadrant))
    return out


def _seed_slot_positions(split: int, count_a: int, count_b: int) -> tuple[list[int], list[int]]:
    top_or_left = [split - (count_a + SEED_OFFSET) + i for i in range(count_a)]
    bottom_or_right = [split + SEED_OFFSET + i for i in range(count_b)]
    return top_or_left, bottom_or_right


def _valid_seed(seed: np.ndarray) -> bool:
    colors = {int(v) for v in np.unique(seed) if int(v) != BG}
    if len(colors) != 2:
        return False
    if any(not np.any(seed[r, :] != BG) for r in range(seed.shape[0])):
        return False
    if any(not np.any(seed[:, c] != BG) for c in range(seed.shape[1])):
        return False
    if np.count_nonzero(seed != BG) < max(5, seed.shape[0] + seed.shape[1] - 2):
        return False
    for k in (1, 2, 3):
        rot = np.rot90(seed, k)
        if rot.shape == seed.shape and np.array_equal(seed, rot):
            return False
    return True


def _sample_seed(rng: random.Random) -> np.ndarray:
    while True:
        sh = rng.randint(3, 5)
        sw = rng.randint(3, 5)
        c1, c2 = rng.sample(COLOR_POOL, 2)
        seed = np.full((sh, sw), BG, dtype=int)

        chosen: set[tuple[int, int]] = set()
        for r in range(sh):
            chosen.add((r, rng.randrange(sw)))
        for c in range(sw):
            if all(col != c for _, col in chosen):
                chosen.add((rng.randrange(sh), c))

        target = rng.randint(max(sh + sw - 1, (sh * sw) // 2), sh * sw - 1)
        extras = [(r, c) for r in range(sh) for c in range(sw) if (r, c) not in chosen]
        rng.shuffle(extras)
        for cell in extras[: max(0, target - len(chosen))]:
            chosen.add(cell)

        ordered = sorted(chosen)
        seed[ordered[0]] = c1
        seed[ordered[1]] = c2
        for cell in ordered[2:]:
            seed[cell] = c1 if rng.random() < 0.5 else c2

        if _valid_seed(seed):
            return seed


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, int]) -> None:
    a, b = colors
    for i, (r, c) in enumerate(path):
        grid[r, c] = a if i % 2 == 0 else b


def _build_l_band(
    rng: random.Random,
    grid: np.ndarray,
    h: int,
    w: int,
    split_r: int,
    split_c: int,
    quadrant: str,
    branch: str,
    colors: tuple[int, int],
) -> None:
    top = quadrant[0] == "T"
    left = quadrant[1] == "L"
    r0, r1, c0, c1 = _quadrant_bounds(h, w, split_r, split_c, quadrant)

    row_offset = rng.choice([1, 2])
    bend_r = split_r - row_offset if top else split_r + row_offset

    start_west = (branch == "ccw") if top else (branch == "cw")
    if left:
        col_offset = 2 if not start_west else rng.choice([1, 2])
        bend_c = split_c - col_offset
    else:
        col_offset = 2 if start_west else rng.choice([1, 2])
        bend_c = split_c + col_offset

    if top:
        end_r = rng.randint(max(r0, bend_r - 4), bend_r - 1)
    else:
        end_r = rng.randint(bend_r + 1, min(r1, bend_r + 4))

    if start_west:
        start_c = rng.randint(max(c0, bend_c - 4), bend_c - 1)
    else:
        start_c = rng.randint(bend_c + 1, min(c1, bend_c + 4))

    path: list[tuple[int, int]] = []
    step_c = 1 if start_c < bend_c else -1
    for c in range(start_c, bend_c + step_c, step_c):
        path.append((bend_r, c))
    step_r = -1 if top else 1
    for r in range(bend_r + step_r, end_r + step_r, step_r):
        path.append((r, bend_c))

    _paint_path(grid, path, colors)


def _build_straight_band(
    rng: random.Random,
    grid: np.ndarray,
    h: int,
    w: int,
    split_r: int,
    split_c: int,
    quadrant: str,
    colors: tuple[int, int],
) -> None:
    r0, r1, c0, c1 = _quadrant_bounds(h, w, split_r, split_c, quadrant)
    top = quadrant[0] == "T"
    left = quadrant[1] == "L"
    horizontal = rng.random() < 0.5

    path: list[tuple[int, int]] = []
    if horizontal:
        row_offset = rng.choice([1, 2])
        r = split_r - row_offset if top else split_r + row_offset
        length = rng.randint(3, min(6, c1 - c0 + 1))
        start_c = rng.randint(c0, c1 - length + 1)
        path = [(r, c) for c in range(start_c, start_c + length)]
    else:
        col_offset = rng.choice([1, 2])
        c = split_c - col_offset if left else split_c + col_offset
        length = rng.randint(3, min(6, r1 - r0 + 1))
        start_r = rng.randint(r0, r1 - length + 1)
        path = [(r, c) for r in range(start_r, start_r + length)]

    _paint_path(grid, path, colors)


def _expected_output(
    grid: np.ndarray, seed: np.ndarray, instructions: dict[str, tuple[int, tuple[int, int]]]
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    split_r, split_c, _divider = _find_dividers(grid)
    seed_colors = _first_two_colors_by_scan(seed, BG)
    for quadrant in QUADRANTS:
        rotation, band_colors = instructions[quadrant]
        sprite = np.rot90(seed, rotation)
        recolored = sprite.copy()
        recolored[sprite == seed_colors[0]] = band_colors[0]
        recolored[sprite == seed_colors[1]] = band_colors[1]
        _paste_centered(out, recolored, _quadrant_bounds(h, w, split_r, split_c, quadrant))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    qh = rng.randint(9, 12)
    qw = rng.randint(9, 12)
    h = 2 * qh + 1
    w = 2 * qw + 1
    split_r = qh
    split_c = qw

    grid = np.full((h, w), BG, dtype=int)
    grid[split_r, :] = DIVIDER
    grid[:, split_c] = DIVIDER

    seed = _sample_seed(rng)
    sh, sw = seed.shape
    top_rows_n = rng.randint(1, sh - 1)
    left_cols_n = rng.randint(1, sw - 1)
    top_rows, bottom_rows = _seed_slot_positions(split_r, top_rows_n, sh - top_rows_n)
    left_cols, right_cols = _seed_slot_positions(split_c, left_cols_n, sw - left_cols_n)
    physical_rows = top_rows + bottom_rows
    physical_cols = left_cols + right_cols

    if physical_rows[0] < 0 or physical_rows[-1] >= h or physical_cols[0] < 0 or physical_cols[-1] >= w:
        return None

    for rr, r in enumerate(physical_rows):
        for cc, c in enumerate(physical_cols):
            color = int(seed[rr, cc])
            if color != BG:
                grid[r, c] = color

    instructions: dict[str, tuple[int, tuple[int, int]]] = {}
    branches = {"cw": 3, "ccw": 1, "half": 2}
    free_colors = [c for c in COLOR_POOL if c not in {int(v) for v in np.unique(seed) if int(v) != BG}]

    for quadrant in QUADRANTS:
        branch = rng.choice(["cw", "ccw", "half"])
        band_colors = tuple(rng.sample(free_colors, 2))
        if branch == "half":
            _build_straight_band(rng, grid, h, w, split_r, split_c, quadrant, band_colors)
        else:
            _build_l_band(rng, grid, h, w, split_r, split_c, quadrant, branch, band_colors)
        instructions[quadrant] = (branches[branch], band_colors)

    expected = _expected_output(grid, seed, instructions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
