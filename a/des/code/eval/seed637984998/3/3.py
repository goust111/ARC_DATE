import random

import numpy as np


BG = 0
POINTER = 2
GRID = 7
SPRITE_COLORS = (1, 3, 4, 5, 6, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
BASE_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
)
GEN_RETRIES = 400


def _line_positions(grid: np.ndarray, axis: int) -> list[int]:
    line_mask = np.all(grid == GRID, axis=axis)
    return [int(i) for i in np.flatnonzero(line_mask)]


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]:
    assert grid.ndim == 2, "expected a 2D grid"
    row_lines = _line_positions(grid, axis=1)
    col_lines = _line_positions(grid, axis=0)
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "expected a complete orange cell grid"
    assert row_lines[0] == 0 and row_lines[-1] == grid.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == grid.shape[1] - 1
    assert all(b - a > 1 for a, b in zip(row_lines, row_lines[1:])), "grid rows must be 1-pixel thick"
    assert all(b - a > 1 for a, b in zip(col_lines, col_lines[1:])), "grid cols must be 1-pixel thick"

    row_ranges = [(a + 1, b) for a, b in zip(row_lines, row_lines[1:])]
    col_ranges = [(a + 1, b) for a, b in zip(col_lines, col_lines[1:])]
    cell_h = row_ranges[0][1] - row_ranges[0][0]
    cell_w = col_ranges[0][1] - col_ranges[0][0]
    assert cell_h >= 2 and cell_w >= 2
    assert all(r1 - r0 == cell_h for r0, r1 in row_ranges), "cell heights must match"
    assert all(c1 - c0 == cell_w for c0, c1 in col_ranges), "cell widths must match"
    return row_ranges, col_ranges, cell_h, cell_w


def _sprite_mask(interior: np.ndarray) -> np.ndarray:
    return (interior != BG) & (interior != GRID) & (interior != POINTER)


def _extract_sprite(interior: np.ndarray) -> np.ndarray | None:
    mask = _sprite_mask(interior)
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    submask = mask[r0 : r1 + 1, c0 : c1 + 1]
    patch[submask] = interior[r0 : r1 + 1, c0 : c1 + 1][submask]
    return patch


def _extract_pointer_direction(interior: np.ndarray) -> str | None:
    coords = np.argwhere(interior == POINTER)
    if coords.size == 0:
        return None
    assert len(coords) == 3, "pointer must contain exactly three red pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert r1 - r0 == 1 and c1 - c0 == 1, "pointer must occupy a 2x2 box"
    box = np.zeros((2, 2), dtype=bool)
    box[coords[:, 0] - r0, coords[:, 1] - c0] = True
    missing = np.argwhere(~box)
    assert len(missing) == 1, "pointer box must have one missing corner"
    mr, mc = map(int, missing[0])

    # If the gap touches the cell's top/bottom edge, use vertical movement.
    # Otherwise the 2x2 L gap is diagonal, so the prompt's horizontal priority applies.
    if mr == 0 and r0 == 0:
        return "up"
    if mr == 1 and r1 == interior.shape[0] - 1:
        return "down"
    if mc == 0:
        return "left"
    return "right"


def _parse_state(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int, dict[tuple[int, int], dict[str, np.ndarray | str | None]]]:
    row_ranges, col_ranges, cell_h, cell_w = _parse_layout(grid)
    cells: dict[tuple[int, int], dict[str, np.ndarray | str | None]] = {}
    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            interior = grid[r0:r1, c0:c1]
            cells[(rr, cc)] = {
                "direction": _extract_pointer_direction(interior),
                "sprite": _extract_sprite(interior),
            }
    return row_ranges, col_ranges, cell_h, cell_w, cells


def _paste_centered(cell: np.ndarray, patch: np.ndarray) -> None:
    ph, pw = patch.shape
    top = (cell.shape[0] - ph) // 2
    left = (cell.shape[1] - pw) // 2
    block = cell[top : top + ph, left : left + pw]
    mask = patch != BG
    block[mask] = patch[mask]


def _apply_pushes(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    cells: dict[tuple[int, int], dict[str, np.ndarray | str | None]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[out == POINTER] = BG
    nrows = len(row_ranges)
    ncols = len(col_ranges)
    pushes: list[tuple[int, int, np.ndarray]] = []
    for (rr, cc), info in cells.items():
        direction = info["direction"]
        patch = info["sprite"]
        if direction is None or patch is None:
            continue
        dr, dc = DIRS[direction]
        tr, tc = rr + dr, cc + dc
        if 0 <= tr < nrows and 0 <= tc < ncols:
            pushes.append((tr, tc, patch))

    for tr, tc, patch in pushes:
        r0, r1 = row_ranges[tr]
        c0, c1 = col_ranges[tc]
        interior = out[r0:r1, c0:c1]
        interior[_sprite_mask(interior)] = BG
        _paste_centered(interior, patch)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the orange cell grid, decode each red 3-pixel L pointer, and simultaneously
    copy each pointed cell's sprite into the adjacent destination cell. The pushed
    sprite is centered in the target cell, any prior target sprite is replaced, and
    all red pointer pixels are erased at the end.
    """
    g = np.asarray(grid, dtype=int)
    row_ranges, col_ranges, _cell_h, _cell_w, cells = _parse_state(g)
    return _apply_pushes(g, row_ranges, col_ranges, cells)


def _transform_shape(shape: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(shape, dtype=bool, copy=True)
    out = np.rot90(out, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return out


def _random_sprite_patch(rng: random.Random) -> np.ndarray:
    shape = _transform_shape(random.choice(BASE_SHAPES), rng)
    color = rng.choice(SPRITE_COLORS)
    patch = np.full(shape.shape, BG, dtype=int)
    patch[shape] = color
    return patch


def _pointer_patch(direction: str, rng: random.Random) -> np.ndarray:
    missing_by_direction = {
        "left": ((0, 0), (1, 0)),
        "right": ((0, 1), (1, 1)),
        "up": ((0, 0), (0, 1)),
        "down": ((1, 0), (1, 1)),
    }
    patch = np.full((2, 2), POINTER, dtype=int)
    mr, mc = rng.choice(missing_by_direction[direction])
    patch[mr, mc] = BG
    return patch


def _pointer_anchor(direction: str, cell_h: int, cell_w: int, rng: random.Random) -> tuple[int, int]:
    if direction == "up":
        return 0, rng.randint(0, cell_w - 2)
    if direction == "down":
        return cell_h - 2, rng.randint(0, cell_w - 2)
    return rng.randint(1, cell_h - 3), rng.randint(0, cell_w - 2)


def _candidate_edges(nrows: int, ncols: int, direction: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = DIRS[direction]
    out = []
    for rr in range(nrows):
        for cc in range(ncols):
            tr, tc = rr + dr, cc + dc
            if 0 <= tr < nrows and 0 <= tc < ncols:
                out.append(((rr, cc), (tr, tc)))
    return out


def _choose_edges(nrows: int, ncols: int, rng: random.Random) -> dict[tuple[int, int], str]:
    num_cells = nrows * ncols
    max_edges = min(6, num_cells - 1)
    edge_count = rng.randint(2, max_edges)
    direction_order = list(DIRS)
    rng.shuffle(direction_order)
    desired = [direction_order[i % 4] for i in range(edge_count)]
    rng.shuffle(desired)

    chosen: dict[tuple[int, int], str] = {}
    used_targets: set[tuple[int, int]] = set()
    for direction in desired:
        candidates = [
            (src, dst)
            for src, dst in _candidate_edges(nrows, ncols, direction)
            if src not in chosen and dst not in used_targets
        ]
        if not candidates:
            return {}
        src, dst = rng.choice(candidates)
        chosen[src] = direction
        used_targets.add(dst)
    return chosen


def _valid_sprite_positions(occupied: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    cell_h, cell_w = occupied.shape
    pos = []
    mask = patch != BG
    for r0 in range(cell_h - ph + 1):
        for c0 in range(cell_w - pw + 1):
            if not np.any(occupied[r0 : r0 + ph, c0 : c0 + pw] & mask):
                pos.append((r0, c0))
    return pos


def _draw_patch(canvas: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    block = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != BG
    block[mask] = patch[mask]


def _cell_slice(rr: int, cc: int, cell_h: int, cell_w: int) -> tuple[slice, slice]:
    r0 = 1 + rr * (cell_h + 1)
    c0 = 1 + cc * (cell_w + 1)
    return slice(r0, r0 + cell_h), slice(c0, c0 + cell_w)


def _expected_from_metadata(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    sprite_patches: dict[tuple[int, int], np.ndarray],
    edges: dict[tuple[int, int], str],
    sprite_sources: set[tuple[int, int]],
) -> np.ndarray:
    cells = {}
    for rr in range(len(row_ranges)):
        for cc in range(len(col_ranges)):
            cells[(rr, cc)] = {
                "direction": edges.get((rr, cc)),
                "sprite": sprite_patches.get((rr, cc)),
            }
    return _apply_pushes(grid, row_ranges, col_ranges, cells)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    nrows = rng.randint(3, 5)
    ncols = rng.randint(3, 5)
    cell_h = rng.randint(5, 6)
    cell_w = rng.randint(5, 6)
    edges = _choose_edges(nrows, ncols, rng)
    if not edges:
        return None

    H = nrows * (cell_h + 1) + 1
    W = ncols * (cell_w + 1) + 1
    grid = np.full((H, W), BG, dtype=int)
    grid[:: cell_h + 1, :] = GRID
    grid[:, :: cell_w + 1] = GRID

    edge_cells = list(edges)
    pointer_only = set(rng.sample(edge_cells, rng.randint(0, max(0, len(edge_cells) - 1))))
    sprite_sources = set(edge_cells) - pointer_only
    if not sprite_sources:
        sprite_sources.add(rng.choice(edge_cells))
        pointer_only.discard(next(iter(sprite_sources)))

    sprite_cells = set(sprite_sources)
    overwrite_target = rng.choice(
        [(src[0] + DIRS[edges[src]][0], src[1] + DIRS[edges[src]][1]) for src in sprite_sources]
    )
    sprite_cells.add(overwrite_target)

    free_non_pointer = [(rr, cc) for rr in range(nrows) for cc in range(ncols) if (rr, cc) not in edges]
    if free_non_pointer and rng.random() < 0.7:
        extra_count = rng.randint(1, min(3, len(free_non_pointer)))
        sprite_cells.update(rng.sample(free_non_pointer, extra_count))

    sprite_patches: dict[tuple[int, int], np.ndarray] = {}
    intended_dirs = dict(edges)
    for cell in sprite_cells:
        sprite_patches[cell] = _random_sprite_patch(rng)

    for rr in range(nrows):
        for cc in range(ncols):
            rs, cs = _cell_slice(rr, cc, cell_h, cell_w)
            interior = grid[rs, cs]
            occupied = np.zeros((cell_h, cell_w), dtype=bool)
            if (rr, cc) in edges:
                direction = edges[(rr, cc)]
                pr, pc = _pointer_anchor(direction, cell_h, cell_w, rng)
                patch = _pointer_patch(direction, rng)
                _draw_patch(interior, pr, pc, patch)
                occupied[pr : pr + 2, pc : pc + 2] |= patch != BG
            if (rr, cc) in sprite_patches:
                patch = sprite_patches[(rr, cc)]
                positions = _valid_sprite_positions(occupied, patch)
                if not positions:
                    return None
                sr, sc = rng.choice(positions)
                _draw_patch(interior, sr, sc, patch)

    row_ranges, col_ranges, _h, _w, parsed = _parse_state(grid)
    for rr in range(nrows):
        for cc in range(ncols):
            expected_dir = intended_dirs.get((rr, cc))
            parsed_dir = parsed[(rr, cc)]["direction"]
            if parsed_dir != expected_dir:
                return None
            expected_patch = sprite_patches.get((rr, cc))
            parsed_patch = parsed[(rr, cc)]["sprite"]
            if expected_patch is None:
                if parsed_patch is not None:
                    return None
            elif parsed_patch is None or not np.array_equal(parsed_patch, expected_patch):
                return None

    expected = _expected_from_metadata(grid, row_ranges, col_ranges, sprite_patches, edges, sprite_sources)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
