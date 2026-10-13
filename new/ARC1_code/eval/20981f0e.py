import numpy as np

from arc_utils import colors_present, find_components, get_bg


def _regular_lattice(
    grid: np.ndarray, bg: int
) -> tuple[int, list[int], list[int], int, int]:
    candidates: list[tuple[int, list[int], list[int], int, int]] = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == int(color))
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        if len(rows) < 2 or len(cols) < 2:
            continue
        if int(coords.shape[0]) != len(rows) * len(cols):
            continue
        if any(int(grid[r, c]) != int(color) for r in rows for c in cols):
            continue

        row_diffs = np.diff(rows)
        col_diffs = np.diff(cols)
        if len(set(map(int, row_diffs))) != 1 or len(set(map(int, col_diffs))) != 1:
            continue
        row_pitch = int(row_diffs[0])
        col_pitch = int(col_diffs[0])
        if row_pitch >= 3 and col_pitch >= 3:
            candidates.append((int(color), rows, cols, row_pitch, col_pitch))

    assert len(candidates) == 1, "expected exactly one regular marker lattice"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each small sprite to the centered slot of its marker-lattice cell.

    Input structure
    - The background is the dominant color.
    - One foreground color forms a complete regular Cartesian lattice of isolated markers.
    - The other foreground color forms one 4-connected 2x2-bbox sprite in each occupied lattice cell.

    Rule
    - Keep the marker lattice fixed and clear the original sprite pixels.
    - For each sprite, find the lattice marker immediately above-left of its cell.
    - Repaint the same sprite shape with its bbox upper-left at half a lattice pitch down and right from that marker.

    Key insight
    - The markers define identical cells; the sprite's shape matters, but its off-center position inside the cell is normalized to the cell center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected marker and sprite foreground colors"

    marker_color, marker_rows, marker_cols, row_pitch, col_pitch = _regular_lattice(g, bg)
    assert row_pitch % 2 == 1 and col_pitch % 2 == 1, "expected odd lattice pitch"

    sprite_colors = [color for color in colors if color != marker_color]
    assert len(sprite_colors) == 1, "expected one sprite color"
    sprite_color = int(sprite_colors[0])

    sprites = find_components(
        g,
        bg=bg,
        colors=[sprite_color],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert sprites, "expected at least one sprite"

    out = g.copy()
    out[g == sprite_color] = bg

    used_cells: set[tuple[int, int]] = set()
    for sprite in sprites:
        coords = sprite["coords"]
        r0, c0, r1, c1 = map(int, sprite["bbox"])
        assert r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2, "expected a 2x2 sprite bbox"

        cell_rows = [r for r in marker_rows if r < r0 and r1 < r + row_pitch]
        cell_cols = [c for c in marker_cols if c < c0 and c1 < c + col_pitch]
        assert len(cell_rows) == 1 and len(cell_cols) == 1, "expected sprite inside one lattice cell"
        cell = (cell_rows[0], cell_cols[0])
        assert cell not in used_cells, "expected at most one sprite per cell"
        used_cells.add(cell)

        target_r = cell[0] + row_pitch // 2
        target_c = cell[1] + col_pitch // 2
        rel = coords - np.array([[r0, c0]])
        target_coords = rel + np.array([[target_r, target_c]])
        assert np.all((0 <= target_coords[:, 0]) & (target_coords[:, 0] < height))
        assert np.all((0 <= target_coords[:, 1]) & (target_coords[:, 1] < width))
        assert np.all(out[target_coords[:, 0], target_coords[:, 1]] == bg), "target slot must be empty"
        out[target_coords[:, 0], target_coords[:, 1]] = sprite_color

    return out

# --------------- generator (o5x) ---------------
# Canvas and marker lattice.
PITCH, GRID_MIN, GRID_MAX = 5, 10, 19
FIRST_MARKER_STOP, BOUNDARY_PAD_MAX = 3, 3
CELL_LAYOUTS, FORCED_ROW_BOUNDARY_ROWS = ((2, 3), (3, 2)), 2
# Colors and sprite placement.
BG_COLOR, SPRITE_COLOR, MARKER_COLOR = 0, 1, 2
SPRITE_SIZE, CENTER_OFFSET = 2, PITCH // 2
INPUT_OFFSET_START, INPUT_OFFSET_STOP = 1, 4
INPUT_OFFSETS = tuple(
    (dr, dc) for dr in range(INPUT_OFFSET_START, INPUT_OFFSET_STOP)
    for dc in range(INPUT_OFFSET_START, INPUT_OFFSET_STOP) if (dr, dc) != (CENTER_OFFSET, CENTER_OFFSET)
)
# Sampling limits.
SQUARE_COUNT_STOP, BINARY_STOP, GEN_RETRY_LIMIT = 3, 2, 300


def _gen_axis(first: int, cell_count: int, has_boundary: bool) -> tuple[list[int], int]:
    """Return marker coordinates and canvas size for one lattice axis."""
    marker_count = cell_count + int(has_boundary)
    markers = [first + PITCH * i for i in range(marker_count)]
    if not has_boundary:
        return markers, first + PITCH * cell_count
    boundary = markers[-1]
    max_pad = min(BOUNDARY_PAD_MAX, GRID_MAX - 1 - boundary)
    if max_pad < 0:
        return markers, -1
    size = boundary + 1 + int(np.random.randint(0, max_pad + 1))
    return markers, size


def _gen_sprite(square: bool) -> np.ndarray:
    """Return a 2x2 square or one-missing-pixel triomino mask."""
    sprite = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    if not square:
        missing = int(np.random.randint(0, SPRITE_SIZE * SPRITE_SIZE))
        sprite[missing // SPRITE_SIZE, missing % SPRITE_SIZE] = 0
    return sprite


def _paste_sprite(grid: np.ndarray, r: int, c: int, sprite: np.ndarray, color: int) -> None:
    """Paint nonzero sprite cells at the given top-left corner."""
    grid[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE][sprite.astype(bool)] = color


def _build_output(marker_rows, marker_cols, cell_rows, cell_cols, shape):
    """Render markers and centered sprites, returning sprite metadata."""
    out = np.zeros(shape, dtype=int)
    out[np.ix_(marker_rows, marker_cols)] = MARKER_COLOR

    cell_count = cell_rows * cell_cols
    square_count = int(np.random.randint(0, SQUARE_COUNT_STOP))
    square_idxs = set()
    if square_count:
        square_idxs = set(map(int, np.random.choice(cell_count, size=square_count, replace=False)))

    sprites: list[tuple[int, int, np.ndarray]] = []
    for i in range(cell_count):
        r = marker_rows[i // cell_cols]
        c = marker_cols[i % cell_cols]
        sprite = _gen_sprite(i in square_idxs)
        _paste_sprite(out, r + CENTER_OFFSET, c + CENTER_OFFSET, sprite, SPRITE_COLOR)
        sprites.append((r, c, sprite))
    return out, sprites


def _build_input(out: np.ndarray, sprites: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    """Move centered sprites to sampled non-center offsets."""
    inp = out.copy()
    for r, c, sprite in sprites:
        _paste_sprite(inp, r + CENTER_OFFSET, c + CENTER_OFFSET, sprite, BG_COLOR)
    for r, c, sprite in sprites:
        dr, dc = INPUT_OFFSETS[int(np.random.randint(0, len(INPUT_OFFSETS)))]
        _paste_sprite(inp, r + dr, c + dc, sprite, SPRITE_COLOR)
    return inp


def _gen_attempt() -> np.ndarray | None:
    cell_rows, cell_cols = CELL_LAYOUTS[int(np.random.randint(0, len(CELL_LAYOUTS)))]
    first_r = int(np.random.randint(0, FIRST_MARKER_STOP))
    first_c = int(np.random.randint(0, FIRST_MARKER_STOP))
    extra_row = cell_rows == FORCED_ROW_BOUNDARY_ROWS or bool(np.random.randint(0, BINARY_STOP))
    extra_col = bool(np.random.randint(0, BINARY_STOP))

    marker_rows, height = _gen_axis(first_r, cell_rows, extra_row)
    marker_cols, width = _gen_axis(first_c, cell_cols, extra_col)
    if not (GRID_MIN <= height <= GRID_MAX and GRID_MIN <= width <= GRID_MAX):
        return None

    out, sprites = _build_output(marker_rows, marker_cols, cell_rows, cell_cols, (height, width))
    inp = _build_input(out, sprites)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 20981f0e input")
