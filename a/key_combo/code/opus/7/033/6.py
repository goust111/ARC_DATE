from __future__ import annotations

import random

import numpy as np


BG = 0
BASE = 5
MARKER = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
NON_BASE_PALETTE = tuple(color for color in PALETTE if color != BASE)
ANGLE_TO_CORNER = {
    0: (0, 0),    # 0 degrees
    3: (0, -1),   # 90 degrees clockwise
    2: (-1, -1),  # 180 degrees
    1: (-1, 0),   # 270 degrees clockwise
}
CORNER_TO_K = {
    "tl": 0,
    "tr": 3,
    "br": 2,
    "bl": 1,
}
GEN_RETRIES = 300


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if bool(value):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _slot_positions(lattice_h: int, lattice_w: int) -> list[tuple[int, int]]:
    return [(1 + 2 * r, 1 + 2 * c) for r in range(lattice_h) for c in range(lattice_w)]


def _decode_corner(tile: np.ndarray) -> int:
    height, width = tile.shape
    corner_values = {
        "tl": int(tile[0, 0]),
        "tr": int(tile[0, width - 1]),
        "br": int(tile[height - 1, width - 1]),
        "bl": int(tile[height - 1, 0]),
    }
    marked = [name for name, value in corner_values.items() if value == MARKER]
    assert len(marked) == 1, "expected exactly one corner marker per tile"
    return CORNER_TO_K[marked[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one motif's slot colors and every motif's corner rotation marker, then rebuild the lattice with per-cell rotated monochrome motifs.

    Input structure:
    - A black background surrounds a rectangular lattice of same-sized motif tiles separated by one black row and one black column.
    - Each tile has size (2R+1) x (2C+1), where R x C is the lattice size.
    - One legend tile carries the R x C slot colors at odd-row/odd-column positions, and every tile has exactly one maroon corner marker that encodes its rotation.

    Rule:
    - Locate the occupied row bands and column bands to recover the lattice size and the tile size.
    - Find the unique legend tile whose odd-row/odd-column slots are not all grey, and read those slots as the color map for the whole lattice.
    - Recover the shared motif mask by taking all non-background, non-maroon cells of a tile, then rotate that mask independently for each lattice position according to its maroon corner.
    - Paint each rotated mask with the decoded slot color for that lattice position on an otherwise black canvas of the same size as the input.

    Key insight:
    - The odd slots and the corner marker encode two different control signals from the same repeated tile family: slot colors choose each output color, while the marked corner chooses each tile's rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    occupied = g != BG
    row_runs = _runs(occupied.any(axis=1))
    col_runs = _runs(occupied.any(axis=0))
    assert row_runs and col_runs, "expected a nonempty motif lattice"

    tile_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "all tiles must share one size"
    assert all(row_runs[i + 1][0] - row_runs[i][1] == 2 for i in range(len(row_runs) - 1)), "expected one blank row between tile rows"
    assert all(col_runs[i + 1][0] - col_runs[i][1] == 2 for i in range(len(col_runs) - 1)), "expected one blank column between tile columns"

    lattice_h = len(row_runs)
    lattice_w = len(col_runs)
    tile_h = next(iter(tile_heights))
    tile_w = next(iter(tile_widths))
    assert tile_h == 2 * lattice_h + 1, "tile height must encode lattice height"
    assert tile_w == 2 * lattice_w + 1, "tile width must encode lattice width"
    assert tile_h == tile_w, "per-tile 90-degree rotations require square tiles"

    tiles: list[list[np.ndarray]] = []
    legend_positions: list[tuple[int, int]] = []
    shared_mask: np.ndarray | None = None
    slots = _slot_positions(lattice_h, lattice_w)

    for lr, (r0, r1) in enumerate(row_runs):
        tile_row: list[np.ndarray] = []
        for lc, (c0, c1) in enumerate(col_runs):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            tile_row.append(tile)

            motif_mask = (tile != BG) & (tile != MARKER)
            if shared_mask is None:
                shared_mask = motif_mask
            else:
                assert np.array_equal(motif_mask, shared_mask), "all tiles must share the same motif mask"

            slot_values = []
            for sr, sc in slots:
                value = int(tile[sr, sc])
                assert value not in (BG, MARKER), "legend slots must be motif-colored pixels"
                slot_values.append(value)
            if any(value != BASE for value in slot_values):
                legend_positions.append((lr, lc))
        tiles.append(tile_row)

    assert shared_mask is not None
    assert len(legend_positions) == 1, "expected exactly one legend tile"
    legend_r, legend_c = legend_positions[0]
    legend_tile = tiles[legend_r][legend_c]

    color_map = np.empty((lattice_h, lattice_w), dtype=int)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            sr, sc = 1 + 2 * lr, 1 + 2 * lc
            color_map[lr, lc] = int(legend_tile[sr, sc])

    out = np.full_like(g, BG)
    for lr, (r0, r1) in enumerate(row_runs):
        for lc, (c0, c1) in enumerate(col_runs):
            k = _decode_corner(tiles[lr][lc])
            rotated_mask = np.rot90(shared_mask, k=k)
            out[r0 : r1 + 1, c0 : c1 + 1][rotated_mask] = int(color_map[lr, lc])

    return out


def _choose_pads(inner: int) -> tuple[int, int]:
    remaining = 30 - inner
    total = random.randint(2, min(10, remaining))
    first = random.randint(1, total - 1)
    return first, total - first


def _build_motif_mask(tile_h: int, tile_w: int) -> np.ndarray:
    mask = np.zeros((tile_h, tile_w), dtype=bool)
    odd_rows = list(range(1, tile_h, 2))
    odd_cols = list(range(1, tile_w, 2))

    for row in odd_rows:
        mask[row, 1:-1] = True

    spine_col = random.choice(odd_cols)
    mask[:, spine_col] = True

    right_row = random.choice(odd_rows[:1] if len(odd_rows) == 1 else odd_rows[: max(1, len(odd_rows) // 2)])
    left_row = random.choice(odd_rows[-1:] if len(odd_rows) == 1 else odd_rows[(len(odd_rows) + 1) // 2 :])
    mask[right_row, tile_w - 1] = True
    mask[left_row, 0] = True

    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        candidates: list[tuple[int, int]] = []
        for r in range(tile_h):
            for c in range(tile_w):
                if mask[r, c] or (r, c) in {(0, 0), (0, tile_w - 1), (tile_h - 1, 0), (tile_h - 1, tile_w - 1)}:
                    continue
                if any(0 <= nr < tile_h and 0 <= nc < tile_w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                    candidates.append((r, c))
        if not candidates:
            break
        rr, cc = random.choice(candidates)
        mask[rr, cc] = True

    assert np.all(mask[0, [0, tile_w - 1]] == False)
    assert np.all(mask[tile_h - 1, [0, tile_w - 1]] == False)
    assert all(mask[sr, sc] for sr, sc in _slot_positions((tile_h - 1) // 2, (tile_w - 1) // 2))
    assert _is_connected(mask)
    assert np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)), "each local row and column must be occupied"
    if tile_h == tile_w:
        assert not np.array_equal(mask, np.rot90(mask, k=1)), "motif must not have 90-degree rotational symmetry"
    return mask


def _sample_color_map(lattice_h: int, lattice_w: int) -> np.ndarray:
    total = lattice_h * lattice_w
    while True:
        values = []
        for _ in range(total):
            if random.random() < 0.3:
                values.append(BASE)
            else:
                values.append(random.choice(NON_BASE_PALETTE))
        non_base = [value for value in values if value != BASE]
        if not non_base:
            continue
        if len(non_base) > total - 1:
            continue
        if BASE not in values:
            continue
        if len(set(non_base)) < min(3, len(non_base)):
            continue
        return np.array(values, dtype=int).reshape(lattice_h, lattice_w)


def _sample_angle_map(lattice_h: int, lattice_w: int) -> np.ndarray:
    total = lattice_h * lattice_w
    values = [0, 1, 2, 3]
    while len(values) < total:
        values.append(random.randrange(4))
    random.shuffle(values)
    return np.array(values[:total], dtype=int).reshape(lattice_h, lattice_w)


def _place_tile(canvas: np.ndarray, tile: np.ndarray, r0: int, c0: int) -> None:
    h, w = tile.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = tile


def _render_input(
    motif_mask: np.ndarray,
    color_map: np.ndarray,
    angle_map: np.ndarray,
    legend_pos: tuple[int, int],
    pads: tuple[int, int, int, int],
) -> np.ndarray:
    lattice_h, lattice_w = color_map.shape
    tile_h, tile_w = motif_mask.shape
    inner_h = lattice_h * tile_h + (lattice_h - 1)
    inner_w = lattice_w * tile_w + (lattice_w - 1)
    top, bottom, left, right = pads

    grid = np.full((top + inner_h + bottom, left + inner_w + right), BG, dtype=int)
    slots = _slot_positions(lattice_h, lattice_w)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            tile = np.full((tile_h, tile_w), BG, dtype=int)
            tile[motif_mask] = BASE
            if (lr, lc) == legend_pos:
                for sr, sc in slots:
                    tile[sr, sc] = int(color_map[(sr - 1) // 2, (sc - 1) // 2])
            corner_r, corner_c = ANGLE_TO_CORNER[int(angle_map[lr, lc])]
            tile[corner_r, corner_c] = MARKER
            r0 = top + lr * (tile_h + 1)
            c0 = left + lc * (tile_w + 1)
            _place_tile(grid, tile, r0, c0)
    return grid


def _render_output(
    motif_mask: np.ndarray,
    color_map: np.ndarray,
    angle_map: np.ndarray,
    pads: tuple[int, int, int, int],
) -> np.ndarray:
    lattice_h, lattice_w = color_map.shape
    tile_h, tile_w = motif_mask.shape
    inner_h = lattice_h * tile_h + (lattice_h - 1)
    inner_w = lattice_w * tile_w + (lattice_w - 1)
    top, bottom, left, right = pads

    out = np.full((top + inner_h + bottom, left + inner_w + right), BG, dtype=int)
    for lr in range(lattice_h):
        for lc in range(lattice_w):
            r0 = top + lr * (tile_h + 1)
            c0 = left + lc * (tile_w + 1)
            rotated_mask = np.rot90(motif_mask, k=int(angle_map[lr, lc]))
            out[r0 : r0 + tile_h, c0 : c0 + tile_w][rotated_mask] = int(color_map[lr, lc])
    return out


def _gen_attempt() -> np.ndarray | None:
    lattice_h = random.randint(2, 3)
    lattice_w = lattice_h
    tile_h = 2 * lattice_h + 1
    tile_w = 2 * lattice_w + 1
    motif_mask = _build_motif_mask(tile_h, tile_w)
    color_map = _sample_color_map(lattice_h, lattice_w)
    angle_map = _sample_angle_map(lattice_h, lattice_w)

    inner_h = lattice_h * tile_h + (lattice_h - 1)
    inner_w = lattice_w * tile_w + (lattice_w - 1)
    top, bottom = _choose_pads(inner_h)
    left, right = _choose_pads(inner_w)
    legend_pos = (random.randrange(lattice_h), random.randrange(lattice_w))
    pads = (top, bottom, left, right)

    inp = _render_input(motif_mask, color_map, angle_map, legend_pos, pads)
    expected = _render_output(motif_mask, color_map, angle_map, pads)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
