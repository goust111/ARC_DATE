import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a repeated tile pattern from fragments aligned to a grey guide grid.

    The input contains a regular grey line grid over a black background. Inside
    each cell sits a small colored patch. Those cell patches repeat with some
    rectangular period measured in whole cells, but most tile instances are
    partially erased back to black. At least one tile instance is complete.

    The solver identifies the cell lattice from the grey rows/columns, tests
    cell-period candidates, finds the unique candidate that has a complete tile
    instance, and fills black holes from the recovered master tile.
    """
    g = np.asarray(grid, dtype=int)
    grid_color, row_lines, col_lines = _find_grid_lines(g)
    bg = _dominant_non_grid_color(g, grid_color)
    patches = _extract_cell_patches(g, row_lines, col_lines)

    tile_h, tile_w, master = _infer_master_tile(patches, bg)

    out = g.copy()
    cell_h = int(patches.shape[2])
    cell_w = int(patches.shape[3])
    for cell_r in range(patches.shape[0]):
        r0 = int(row_lines[cell_r] + 1)
        r1 = r0 + cell_h
        for cell_c in range(patches.shape[1]):
            c0 = int(col_lines[cell_c] + 1)
            c1 = c0 + cell_w
            patch = out[r0:r1, c0:c1]
            phase = master[cell_r % tile_h, cell_c % tile_w]
            fill_mask = (patch == bg) & (phase != bg)
            patch[fill_mask] = phase[fill_mask]

    return out


def _find_grid_lines(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    h, w = grid.shape
    best = None
    for color in np.unique(grid):
        rows = np.flatnonzero(np.all(grid == int(color), axis=1))
        cols = np.flatnonzero(np.all(grid == int(color), axis=0))
        if len(rows) < 3 or len(cols) < 3:
            continue
        row_steps = np.diff(rows)
        col_steps = np.diff(cols)
        if len(row_steps) == 0 or len(col_steps) == 0:
            continue
        if not np.all(row_steps == row_steps[0]):
            continue
        if not np.all(col_steps == col_steps[0]):
            continue
        if int(row_steps[0]) <= 1 or int(col_steps[0]) <= 1:
            continue
        if int(rows[0]) != 0 or int(cols[0]) != 0:
            continue
        if int(rows[-1]) != h - 1 or int(cols[-1]) != w - 1:
            continue
        score = len(rows) * len(cols)
        candidate = (score, int(color), rows.astype(int), cols.astype(int))
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, "failed to identify the guide grid"
    return best[1], best[2], best[3]


def _dominant_non_grid_color(grid: np.ndarray, grid_color: int) -> int:
    values = grid[grid != grid_color]
    assert values.size > 0
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _extract_cell_patches(
    grid: np.ndarray, row_lines: np.ndarray, col_lines: np.ndarray
) -> np.ndarray:
    cell_rows = len(row_lines) - 1
    cell_cols = len(col_lines) - 1
    cell_h = int(row_lines[1] - row_lines[0] - 1)
    cell_w = int(col_lines[1] - col_lines[0] - 1)
    patches = np.empty((cell_rows, cell_cols, cell_h, cell_w), dtype=int)
    for cell_r in range(cell_rows):
        r0 = int(row_lines[cell_r] + 1)
        r1 = int(row_lines[cell_r + 1])
        for cell_c in range(cell_cols):
            c0 = int(col_lines[cell_c] + 1)
            c1 = int(col_lines[cell_c + 1])
            patches[cell_r, cell_c] = grid[r0:r1, c0:c1]
    return patches


def _infer_master_tile(patches: np.ndarray, bg: int) -> tuple[int, int, np.ndarray]:
    cell_rows, cell_cols = map(int, patches.shape[:2])
    candidates: list[tuple[int, int]] = []
    for tile_h in range(1, cell_rows + 1):
        if cell_rows % tile_h != 0:
            continue
        for tile_w in range(1, cell_cols + 1):
            if cell_cols % tile_w != 0:
                continue
            tile_count = (cell_rows // tile_h) * (cell_cols // tile_w)
            if tile_count < 2:
                continue
            candidates.append((tile_h, tile_w))

    candidates.sort(key=lambda hw: (hw[0] * hw[1], hw[0], hw[1]))
    for tile_h, tile_w in candidates:
        info = _candidate_master(patches, bg, tile_h, tile_w)
        if info is not None:
            return tile_h, tile_w, info

    raise AssertionError("failed to infer a valid tile period")


def _candidate_master(
    patches: np.ndarray, bg: int, tile_h: int, tile_w: int
) -> np.ndarray | None:
    cell_rows, cell_cols, cell_h, cell_w = map(int, patches.shape)
    master = np.full((tile_h, tile_w, cell_h, cell_w), bg, dtype=int)

    for cell_r in range(cell_rows):
        for cell_c in range(cell_cols):
            patch = patches[cell_r, cell_c]
            phase = master[cell_r % tile_h, cell_c % tile_w]
            mask = patch != bg
            conflict = mask & (phase != bg) & (phase != patch)
            if np.any(conflict):
                return None
            phase[mask] = patch[mask]

    if np.any(np.all(master == bg, axis=(2, 3))):
        return None

    master_fg = master != bg
    total_fg = int(np.count_nonzero(master_fg))
    if total_fg == 0:
        return None

    tile_rows = cell_rows // tile_h
    tile_cols = cell_cols // tile_w
    full_instances = 0
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            visible = 0
            for local_r in range(tile_h):
                for local_c in range(tile_w):
                    patch = patches[tile_r * tile_h + local_r, tile_c * tile_w + local_c]
                    phase = master[local_r, local_c]
                    mismatch = (patch != bg) & (patch != phase)
                    if np.any(mismatch):
                        return None
                    visible += int(np.count_nonzero((phase != bg) & (patch == phase)))
            if visible == total_fg:
                full_instances += 1

    if full_instances < 1:
        return None
    return master


# --------------- generator ---------------

_BG = 0
_GRID = 5
_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
_CELL_SIZE_OPTIONS = (2, 3, 4)
_CELL_COUNT_OPTIONS = (4, 6, 8, 9)
_ERASE_MODES = ("cell", "pixel", "mixed")
_MAX_CANVAS = 40
_MAX_GENERATE_ATTEMPTS = 500


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_GENERATE_ATTEMPTS):
        sample = _gen_attempt(rng)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid sample")


def _gen_attempt(
    rng: random.Random, return_output: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    cell_size = rng.choice(_CELL_SIZE_OPTIONS)
    cell_rows = rng.choice(_CELL_COUNT_OPTIONS)
    cell_cols = rng.choice(_CELL_COUNT_OPTIONS)

    tile_h_choices = [d for d in range(2, min(4, cell_rows) + 1) if cell_rows % d == 0 and cell_rows // d >= 2]
    tile_w_choices = [d for d in range(2, min(4, cell_cols) + 1) if cell_cols % d == 0 and cell_cols // d >= 2]
    if not tile_h_choices or not tile_w_choices:
        return None

    tile_h = rng.choice(tile_h_choices)
    tile_w = rng.choice(tile_w_choices)
    grid_h = cell_rows * (cell_size + 1) + 1
    grid_w = cell_cols * (cell_size + 1) + 1
    if grid_h > _MAX_CANVAS or grid_w > _MAX_CANVAS:
        return None

    palette_size = rng.randint(2, min(4, len(_PALETTE)))
    palette = rng.sample(_PALETTE, palette_size)
    master = _make_master_tile(tile_h, tile_w, cell_size, palette, rng)
    total_fg = int(np.count_nonzero(master != _BG))
    if total_fg < tile_h * tile_w:
        return None

    tile_rows = cell_rows // tile_h
    tile_cols = cell_cols // tile_w
    if tile_rows * tile_cols < 4:
        return None

    full_tile = (rng.randrange(tile_rows), rng.randrange(tile_cols))
    erase_mode = rng.choice(_ERASE_MODES)

    inp = _blank_grid(cell_rows, cell_cols, cell_size)
    out = _blank_grid(cell_rows, cell_cols, cell_size)

    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            if (tile_r, tile_c) == full_tile:
                tile_patch = master.copy()
            else:
                tile_patch = _erase_tile(master, erase_mode, rng)
                if tile_patch is None:
                    return None

            for local_r in range(tile_h):
                cell_r = tile_r * tile_h + local_r
                for local_c in range(tile_w):
                    cell_c = tile_c * tile_w + local_c
                    _paste_cell(inp, cell_r, cell_c, tile_patch[local_r, local_c], cell_size)
                    _paste_cell(out, cell_r, cell_c, master[local_r, local_c], cell_size)

    non_grid = inp[inp != _GRID]
    counts = np.bincount(non_grid, minlength=10)
    if int(np.argmax(counts)) != _BG:
        return None

    if np.array_equal(inp, out):
        return None

    if return_output:
        return inp, out
    return inp


def _make_master_tile(
    tile_h: int, tile_w: int, cell_size: int, palette: list[int], rng: random.Random
) -> np.ndarray:
    master = np.full((tile_h, tile_w, cell_size, cell_size), _BG, dtype=int)
    used: set[tuple[int, ...]] = set()
    for local_r in range(tile_h):
        for local_c in range(tile_w):
            for _ in range(200):
                patch = _make_cell_patch(cell_size, palette, rng)
                sig = tuple(int(x) for x in patch.ravel())
                if sig not in used:
                    master[local_r, local_c] = patch
                    used.add(sig)
                    break
            else:
                raise RuntimeError("failed to build a unique master tile")
    return master


def _make_cell_patch(cell_size: int, palette: list[int], rng: random.Random) -> np.ndarray:
    patch = np.full((cell_size, cell_size), _BG, dtype=int)
    coords = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    max_fg = max(1, (cell_size * cell_size) // 2)
    min_fg = 1 if cell_size == 2 else 2
    fg_count = rng.randint(min_fg, max_fg)
    chosen = rng.sample(coords, fg_count)
    local_palette = rng.sample(palette, rng.randint(1, min(3, len(palette))))
    for r, c in chosen:
        patch[r, c] = rng.choice(local_palette)
    return patch


def _erase_tile(master: np.ndarray, mode: str, rng: random.Random) -> np.ndarray | None:
    tile = master.copy()
    fg_coords = [tuple(coord) for coord in np.argwhere(tile != _BG)]
    if len(fg_coords) < 2:
        return None

    cell_coords = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]
    if mode == "cell":
        erase_count = rng.randint(1, max(1, len(cell_coords) - 1))
        for local_r, local_c in rng.sample(cell_coords, erase_count):
            cell = tile[local_r, local_c]
            cell[cell != _BG] = _BG
    elif mode == "pixel":
        erase_count = rng.randint(1, len(fg_coords) - 1)
        for local_r, local_c, rr, cc in rng.sample(fg_coords, erase_count):
            tile[local_r, local_c, rr, cc] = _BG
    else:
        max_cell_erase = max(1, len(cell_coords) // 2)
        erase_cells = rng.randint(1, max_cell_erase)
        for local_r, local_c in rng.sample(cell_coords, erase_cells):
            cell = tile[local_r, local_c]
            cell[cell != _BG] = _BG
        remaining = [tuple(coord) for coord in np.argwhere(tile != _BG)]
        if not remaining:
            return None
        extra = rng.randint(1, len(remaining))
        for local_r, local_c, rr, cc in rng.sample(remaining, extra):
            tile[local_r, local_c, rr, cc] = _BG

    visible = int(np.count_nonzero(tile != _BG))
    total = int(np.count_nonzero(master != _BG))
    if visible == 0 or visible >= total:
        return None
    return tile


def _blank_grid(cell_rows: int, cell_cols: int, cell_size: int) -> np.ndarray:
    grid = np.full(
        (cell_rows * (cell_size + 1) + 1, cell_cols * (cell_size + 1) + 1),
        _GRID,
        dtype=int,
    )
    for cell_r in range(cell_rows):
        r0 = cell_r * (cell_size + 1) + 1
        for cell_c in range(cell_cols):
            c0 = cell_c * (cell_size + 1) + 1
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = _BG
    return grid


def _paste_cell(grid: np.ndarray, cell_r: int, cell_c: int, patch: np.ndarray, cell_size: int) -> None:
    r0 = cell_r * (cell_size + 1) + 1
    c0 = cell_c * (cell_size + 1) + 1
    grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch
