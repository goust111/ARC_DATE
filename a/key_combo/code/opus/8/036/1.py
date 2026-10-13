import random
from collections import Counter

import numpy as np


BG = 0
WALL = 1
YELLOW = 4
LATTICE = 5
SOURCE = 8

CELL = 4
STRIDE = CELL + 1
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_lattice_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    grey = grid == LATTICE
    H, W = grid.shape
    rows = [int(r) for r in np.where(grey.sum(axis=1) >= W - 2)[0]]
    cols = [int(c) for c in np.where(grey.sum(axis=0) >= H - 2)[0]]
    assert len(rows) >= 2 and len(cols) >= 2, "expected regular lattice lines"
    row_steps = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
    col_steps = {cols[i + 1] - cols[i] for i in range(len(cols) - 1)}
    assert len(row_steps) == 1 and len(col_steps) == 1, "lattice must be regular"
    return rows, cols


def _find_source_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(grid == SOURCE)
    assert box is not None, "expected source"
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "source must be 2x2"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == SOURCE), "source must be solid"
    assert np.count_nonzero(grid == SOURCE) == 4, "expected one teal source"
    return r0, c0, r1, c1


def _find_hole_crop(grid: np.ndarray, rows: list[int], cols: list[int]) -> tuple[int, int, int, int]:
    holes = [(r, c) for r in rows for c in cols if int(grid[r, c]) == BG]
    assert len(holes) == 4, "expected exactly four lattice holes"
    hole_rows = sorted({r for r, _ in holes})
    hole_cols = sorted({c for _, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2, "holes must be rectangle corners"
    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return r0, c0, r1, c1


def _paint_rays(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = _find_source_bbox(grid)
    H, W = grid.shape

    for r in range(r0, r1 + 1):
        for dc, start in [(-1, c0 - 1), (1, c1 + 1)]:
            c = start
            while 0 <= c < W:
                val = int(grid[r, c])
                if val in (LATTICE, WALL):
                    break
                if val == BG:
                    out[r, c] = YELLOW
                c += dc

    for c in range(c0, c1 + 1):
        for dr, start in [(-1, r0 - 1), (1, r1 + 1)]:
            r = start
            while 0 <= r < H:
                val = int(grid[r, c])
                if val in (LATTICE, WALL):
                    break
                if val == BG:
                    out[r, c] = YELLOW
                r += dr

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint channel-confined rays from the teal source, then crop to the rectangle marked by four missing lattice intersections.

    Input structure:
    - A black background is overlaid with a regular one-pixel grey lattice whose intersections are almost all intact.
    - Exactly one solid teal 2x2 block sits inside a lattice cell, and several blue 4-connected objects sit inside other cell interiors as walls.
    - Exactly four lattice intersections are black holes instead of grey pixels, and those four holes are the corners of one axis-aligned rectangle.

    Rule:
    - From each of the teal block's two rows, cast rays left and right through black channel pixels only; from each of its two columns, cast rays up and down the same way.
    - A ray stops before any blue wall pixel or grey lattice pixel, so it never crosses a lattice line; every traversed black pixel becomes yellow while teal, blue, and grey stay unchanged.
    - After all ray painting is done, locate the four missing lattice intersections and return the inclusive crop whose corners are those holes.

    Key insight:
    - The lattice has two independent roles: it blocks ray propagation into other cells, and its four missing intersections separately define the crop window.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    rows, cols = _find_lattice_lines(g)
    painted = _paint_rays(g)
    r0, c0, r1, c1 = _find_hole_crop(g, rows, cols)
    return painted[r0 : r1 + 1, c0 : c1 + 1].copy()


def _line_positions(num_bands: int) -> list[int]:
    return [CELL + i * STRIDE for i in range(num_bands - 1)]


def _cell_origin(br: int, bc: int) -> tuple[int, int]:
    return br * STRIDE, bc * STRIDE


def _build_lattice(num_band_rows: int, num_band_cols: int) -> tuple[np.ndarray, list[int], list[int]]:
    H = num_band_rows * CELL + (num_band_rows - 1)
    W = num_band_cols * CELL + (num_band_cols - 1)
    grid = np.full((H, W), BG, dtype=int)
    lattice_rows = _line_positions(num_band_rows)
    lattice_cols = _line_positions(num_band_cols)
    for r in lattice_rows:
        grid[r, :] = LATTICE
    for c in lattice_cols:
        grid[:, c] = LATTICE
    return grid, lattice_rows, lattice_cols


def _local_source_cells(src_top: int, src_left: int) -> set[tuple[int, int]]:
    return {(src_top + dr, src_left + dc) for dr in range(2) for dc in range(2)}


def _safe_mask_for_source_blocker(src_cells: set[tuple[int, int]]) -> np.ndarray:
    allowed = np.ones((CELL, CELL), dtype=bool)
    forbidden = set(src_cells)
    for r, c in src_cells:
        forbidden.update(_neighbors4(r, c))
    for r, c in forbidden:
        if 0 <= r < CELL and 0 <= c < CELL:
            allowed[r, c] = False
    return allowed


def _random_connected_shape(
    rng: random.Random,
    allowed: np.ndarray,
    min_size: int = 2,
    max_size: int = 5,
    required: set[tuple[int, int]] | None = None,
    forbid_straight_big: bool = False,
) -> set[tuple[int, int]] | None:
    required = set() if required is None else {tuple(map(int, p)) for p in required}
    allowed_set = {(r, c) for r in range(CELL) for c in range(CELL) if bool(allowed[r, c])}
    if not required:
        if not allowed_set:
            return None
        required = {rng.choice(sorted(allowed_set))}
    if not required.issubset(allowed_set):
        return None

    target = rng.randint(max(min_size, len(required)), max_size)
    shape = set(required)
    while True:
        frontier = {
            (nr, nc)
            for r, c in shape
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) in allowed_set and (nr, nc) not in shape
        }
        if len(shape) >= target or not frontier:
            break
        shape.add(rng.choice(sorted(frontier)))

    if len(shape) < max(min_size, len(required)):
        return None
    if forbid_straight_big and len(shape) >= 3:
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if len(rows) == 1 or len(cols) == 1:
            return None
    return shape


def _paint_shape(grid: np.ndarray, br: int, bc: int, shape: set[tuple[int, int]], color: int) -> None:
    r_base, c_base = _cell_origin(br, bc)
    for r, c in shape:
        grid[r_base + r, c_base + c] = color


def _choose_hole_span(rng: random.Random, num_bands: int) -> tuple[int, int]:
    indices = list(range(num_bands - 1))
    choices = [(a, b) for a in indices for b in indices if b - a >= 2]
    return rng.choice(choices)


def _cell_in_crop(br: int, bc: int, top_i: int, bottom_i: int, left_i: int, right_i: int) -> bool:
    return (top_i + 1) <= br <= bottom_i and (left_i + 1) <= bc <= right_i


def _all_cells(num_band_rows: int, num_band_cols: int) -> list[tuple[int, int]]:
    return [(br, bc) for br in range(num_band_rows) for bc in range(num_band_cols)]


def _place_general_wall(
    rng: random.Random,
    grid: np.ndarray,
    br: int,
    bc: int,
    required: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    allowed = np.ones((CELL, CELL), dtype=bool)
    for _ in range(60):
        shape = _random_connected_shape(
            rng,
            allowed,
            min_size=2,
            max_size=5,
            required=required,
            forbid_straight_big=True,
        )
        if shape is not None:
            _paint_shape(grid, br, bc, shape, WALL)
            return shape
    return None


def _oracle_from_metadata(
    grid: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    painted = _paint_rays(grid)
    r0, c0, r1, c1 = crop_box
    return painted[r0 : r1 + 1, c0 : c1 + 1].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_band_rows = rng.choice([4, 5])
    num_band_cols = rng.choice([4, 5])
    grid, lattice_rows, lattice_cols = _build_lattice(num_band_rows, num_band_cols)

    top_i, bottom_i = _choose_hole_span(rng, num_band_rows)
    left_i, right_i = _choose_hole_span(rng, num_band_cols)
    crop_box = (
        lattice_rows[top_i],
        lattice_cols[left_i],
        lattice_rows[bottom_i],
        lattice_cols[right_i],
    )

    src_br = rng.randint(top_i + 1, bottom_i)
    src_bc = rng.randint(left_i + 1, right_i)
    src_top, src_left = rng.choice([(0, 0), (0, 2), (2, 0), (2, 2)])
    src_cells = _local_source_cells(src_top, src_left)
    r_base, c_base = _cell_origin(src_br, src_bc)
    for r, c in src_cells:
        grid[r_base + r, c_base + c] = SOURCE

    allowed = _safe_mask_for_source_blocker(src_cells)
    source_rows = [src_top, src_top + 1]
    source_cols = [src_left, src_left + 1]
    block_axis = rng.choice(["h", "v"])
    if block_axis == "h":
        block_col = 3 if src_left == 0 else 0
        required = {(r, block_col) for r in source_rows}
    else:
        block_row = 3 if src_top == 0 else 0
        required = {(block_row, c) for c in source_cols}
    source_wall = _random_connected_shape(rng, allowed, min_size=2, max_size=4, required=required)
    if source_wall is None:
        return None
    _paint_shape(grid, src_br, src_bc, source_wall, WALL)

    cells = _all_cells(num_band_rows, num_band_cols)
    inside_cells = [
        (br, bc)
        for br, bc in cells
        if (br, bc) != (src_br, src_bc) and _cell_in_crop(br, bc, top_i, bottom_i, left_i, right_i)
    ]
    outside_cells = [
        (br, bc)
        for br, bc in cells
        if (br, bc) != (src_br, src_bc) and not _cell_in_crop(br, bc, top_i, bottom_i, left_i, right_i)
    ]
    if not inside_cells or not outside_cells:
        return None

    placed_cells: set[tuple[int, int]] = {(src_br, src_bc)}
    aligned_found = False
    inside_wall_found = False
    outside_wall_found = False

    decoy_mode = rng.choice(["row", "col"])
    decoy_candidates = []
    if decoy_mode == "row":
        decoy_candidates = [(src_br, bc) for bc in range(num_band_cols) if bc != src_bc]
        required_local = {(rng.choice(source_rows), rng.randrange(CELL))}
    else:
        decoy_candidates = [(br, src_bc) for br in range(num_band_rows) if br != src_br]
        required_local = {(rng.randrange(CELL), rng.choice(source_cols))}
    rng.shuffle(decoy_candidates)
    decoy_cell = None
    for br, bc in decoy_candidates:
        if (br, bc) in placed_cells:
            continue
        shape = _place_general_wall(rng, grid, br, bc, required=required_local)
        if shape is None:
            continue
        placed_cells.add((br, bc))
        decoy_cell = (br, bc)
        aligned_found = True
        inside_wall_found |= _cell_in_crop(br, bc, top_i, bottom_i, left_i, right_i)
        outside_wall_found |= not _cell_in_crop(br, bc, top_i, bottom_i, left_i, right_i)
        break
    if decoy_cell is None:
        return None

    remaining_inside = [cell for cell in inside_cells if cell not in placed_cells]
    remaining_outside = [cell for cell in outside_cells if cell not in placed_cells]
    if not inside_wall_found:
        if not remaining_inside:
            return None
        br, bc = rng.choice(remaining_inside)
        if _place_general_wall(rng, grid, br, bc) is None:
            return None
        placed_cells.add((br, bc))
        inside_wall_found = True
    if not outside_wall_found:
        if not remaining_outside:
            return None
        br, bc = rng.choice(remaining_outside)
        if _place_general_wall(rng, grid, br, bc) is None:
            return None
        placed_cells.add((br, bc))
        outside_wall_found = True

    extra_candidates = [cell for cell in cells if cell not in placed_cells]
    rng.shuffle(extra_candidates)
    for br, bc in extra_candidates[: rng.randint(1, 4)]:
        _place_general_wall(rng, grid, br, bc)

    holes = {
        (lattice_rows[top_i], lattice_cols[left_i]),
        (lattice_rows[top_i], lattice_cols[right_i]),
        (lattice_rows[bottom_i], lattice_cols[left_i]),
        (lattice_rows[bottom_i], lattice_cols[right_i]),
    }
    for r, c in holes:
        grid[r, c] = BG

    if np.any(grid == YELLOW):
        return None

    out = _oracle_from_metadata(grid, crop_box)
    crop_inp = grid[crop_box[0] : crop_box[2] + 1, crop_box[1] : crop_box[3] + 1]
    if np.array_equal(out, crop_inp):
        return None
    if not np.any(out == YELLOW):
        return None

    without_source_wall = grid.copy()
    for r, c in source_wall:
        without_source_wall[r_base + r, c_base + c] = BG
    if np.count_nonzero(_paint_rays(without_source_wall) == YELLOW) <= np.count_nonzero(
        _paint_rays(grid) == YELLOW
    ):
        return None

    rows, cols = _find_lattice_lines(grid)
    if _find_hole_crop(grid, rows, cols) != crop_box:
        return None
    if not inside_wall_found or not outside_wall_found or not aligned_found:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid lattice-ray crop input")
