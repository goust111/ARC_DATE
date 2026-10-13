import numpy as np
import random
from arc_utils import paste


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the compact table of solid cells hidden inside a separator lattice.

    Input structure
    - The grid has full-width and full-height bars of one separator color, dividing it into rectangular cells.
    - Most cells contain a repeating multicolor pattern, while a contiguous rectangle of cells is solid monochrome.

    Rule
    - Split the input into lattice cells using the full separator rows and columns.
    - Keep the cells whose full interior is one color, crop to their occupied rectangle, and output those colors.

    Key insight
    - The periodic pattern is noise; only completely solid lattice cells encode the reduced output grid.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_colors = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_colors = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "there must be exactly one row/column separator color"
    sep = next(iter(sep_colors))
    assert row_colors == {sep}, "all full rows must be separator rows"
    assert col_colors == {sep}, "all full columns must be separator columns"

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "separator lattice must have rows and columns"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)

    solid_cells = {}
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(cell == cell[0, 0]):
                solid_cells[(ri, ci)] = int(cell[0, 0])
    assert solid_cells, "must contain at least one solid lattice cell"

    solid_rows = [ri for ri, _ in solid_cells]
    solid_cols = [ci for _, ci in solid_cells]
    r0, r1 = min(solid_rows), max(solid_rows)
    c0, c1 = min(solid_cols), max(solid_cols)

    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for ri in range(r0, r1 + 1):
        for ci in range(c0, c1 + 1):
            assert (ri, ci) in solid_cells, "solid lattice cells must form a rectangle"
            out[ri - r0, ci - c0] = solid_cells[(ri, ci)]
    return out

# --------------- generator (att2, o5x) ---------------

# Lattice dimensions: valid (L, cell_size) pairs produce 27-30 square grids.
VALID_LC = [(4, 6), (5, 5), (6, 4)]
COLOR_COUNT = 10
SEPARATOR_WIDTH = 1
# Solid rectangle: output block size in lattice coordinates.
SOLID_SIZE_RANGE = (2, 3)
# Pattern cells: diagonal variants are multicolor and solver-visible as non-solid.
PATTERN_TILE_MIN_COLORS = 5
PATTERN_CELL_MIN_COLORS = 2
DIAGONAL_ID_BITS = 4
# Retry budget: regenerate on rare validation failures.
MAX_GENERATION_ATTEMPTS = 300


def _pattern_tiles(csize: int, lattice: int, sep: int) -> dict[int, np.ndarray]:
    """Build one structured tile per lattice diagonal, or {} if validation fails."""
    colors = [c for c in range(COLOR_COUNT) if c != sep]
    random.shuffle(colors)
    palette = colors[: random.randint(PATTERN_TILE_MIN_COLORS, len(colors))]
    palette_arr = np.asarray(palette, dtype=int)
    k = len(palette)

    rr, cc = np.indices((csize, csize))
    base = (rr + cc + random.randrange(k)) % k
    if random.choice([False, True]):
        base = base.T
    if random.choice([False, True]):
        base = np.fliplr(base)
    if random.choice([False, True]):
        base = np.flipud(base)
    tiles = {}
    for diag in range(2 * lattice - 1):
        tile_idx = (base + random.randrange(k)) % k
        tile = palette_arr[tile_idx]
        # Encode the diagonal id in four cells so variants are guaranteed distinct.
        for bit in range(DIAGONAL_ID_BITS):
            tile.flat[bit] = palette[(diag >> bit) & 1]
        if len(np.unique(tile)) < PATTERN_TILE_MIN_COLORS:
            return {}
        tiles[diag] = tile
    return tiles


def _sample_solid_block(lattice, colors):
    """Return ((row, col, height, width), output colors) for the solid block."""
    solid_h = random.randint(*SOLID_SIZE_RANGE)
    solid_w = random.randint(*SOLID_SIZE_RANGE)
    solid_r0 = random.randint(0, lattice - solid_h)
    solid_c0 = random.randint(0, lattice - solid_w)
    flat = np.random.choice(colors, size=solid_h * solid_w).astype(int)
    if len(set(map(int, flat))) == 1:
        change_at = random.randrange(len(flat))
        flat[change_at] = random.choice([c for c in colors if c != int(flat[change_at])])
    return (solid_r0, solid_c0, solid_h, solid_w), flat.reshape((solid_h, solid_w))


def _render_grid(lattice, csize, sep, block, out, tiles):
    """Assemble the separator lattice from solid cells and pattern tiles."""
    n = lattice * csize + (lattice - 1)
    grid = np.full((n, n), sep, dtype=int)
    solid_r0, solid_c0, solid_h, solid_w = block
    stride = csize + SEPARATOR_WIDTH
    for ri in range(lattice):
        for ci in range(lattice):
            r0, c0 = ri * stride, ci * stride
            if solid_r0 <= ri < solid_r0 + solid_h and solid_c0 <= ci < solid_c0 + solid_w:
                cell = np.full((csize, csize), int(out[ri - solid_r0, ci - solid_c0]), dtype=int)
            else:
                cell = tiles[(ri - ci) % (2 * lattice - 1)]
            paste(grid, cell, r0, c0)
    return grid


def _valid_generated_grid(grid, lattice, csize, sep, block):
    """Check separator lines and non-solid cells before the solver oracle."""
    stride = csize + SEPARATOR_WIDTH
    sep_lines = [csize + i * stride for i in range(lattice - 1)]
    if not all(np.all(grid[r, :] == sep) and np.all(grid[:, r] == sep) for r in sep_lines):
        return False
    solid_r0, solid_c0, solid_h, solid_w = block
    for ri in range(lattice):
        for ci in range(lattice):
            if solid_r0 <= ri < solid_r0 + solid_h and solid_c0 <= ci < solid_c0 + solid_w:
                continue
            r0, c0 = ri * stride, ci * stride
            if len(np.unique(grid[r0 : r0 + csize, c0 : c0 + csize])) < PATTERN_CELL_MIN_COLORS:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    lattice, csize = random.choice(VALID_LC)
    sep = random.randrange(COLOR_COUNT)
    colors = [c for c in range(COLOR_COUNT) if c != sep]
    block, out = _sample_solid_block(lattice, colors)
    tiles = _pattern_tiles(csize, lattice, sep)
    if not tiles:
        return None
    grid = _render_grid(lattice, csize, sep, block, out, tiles)
    if not _valid_generated_grid(grid, lattice, csize, sep, block):
        return None
    try:
        if not np.array_equal(main(grid), out):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 458e3a53 input")
