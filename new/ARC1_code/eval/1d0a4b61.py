import numpy as np

from arc_utils import Color
import random


def _infer_minimal_tile(g: np.ndarray) -> np.ndarray:
    height, width = map(int, g.shape)
    known = g != Color.BLACK
    assert np.any(~known), "expected black missing pixels"

    candidates: list[tuple[int, np.ndarray]] = []
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            tile = np.full((tile_h, tile_w), -1, dtype=int)
            valid = True
            for r, c in np.argwhere(known):
                tr, tc = int(r) % tile_h, int(c) % tile_w
                value = int(g[r, c])
                if tile[tr, tc] == -1:
                    tile[tr, tc] = value
                elif tile[tr, tc] != value:
                    valid = False
                    break
            if valid and np.all(tile != -1):
                candidates.append((tile_h * tile_w, tile))

    assert candidates, "expected a fully observed repeating tile"
    best_area = min(area for area, _ in candidates)
    best = [tile for area, tile in candidates if area == best_area]
    assert len(best) == 1, "expected a unique minimal repeating tile"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the black holes by continuing the smallest repeating non-black wallpaper pattern.

    Input structure:
    - The grid is a rectangular wallpaper pattern with some cells replaced by black.
    - Every phase of the true repeating tile is visible at least once in a non-black cell.
    - The smallest fully observed tile consistent with the non-black cells is unique.

    Rule:
    - Ignore black pixels while inferring the repeating tile.
    - Find the unique minimal-area tile whose row/column phases agree with all non-black cells.
    - Replace every grid cell by the color at its corresponding tile phase.

    Key insight:
    - Black is missing data, not part of the wallpaper; the observed colored cells determine the latent periodic tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile = _infer_minimal_tile(g)
    tile_h, tile_w = map(int, tile.shape)
    rows = np.arange(g.shape[0])[:, None] % tile_h
    cols = np.arange(g.shape[1])[None, :] % tile_w
    out = tile[rows, cols]
    assert np.all(out[g != Color.BLACK] == g[g != Color.BLACK]), "tile must preserve all known cells"
    return out

# --------------- generator (o5x) ---------------

# Canvas and latent tile bounds.
CANVAS_SIZE = 25
TILE_HEIGHT_RANGE = (4, 9)
TILE_WIDTH_RANGE = (2, 9)
GRID_LINE_COLOR = Color.BLUE
OUTER_GRID_LINE_INDEX = 0
MISSING_COLOR = Color.BLACK

# Interior grid-line and color sampling.
LINE_MIN_LENGTH = 6
LINE_PROBABILITY = 0.35
LINE_STEP_RANGE = (3, 5)
INNER_COLOR_COUNT_RANGE = (2, 3)
NON_BLUE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]

# Black-hole blob sampling and placement.
BLOB_COUNT_RANGE = (3, 4)
BLACK_TOTAL_RANGE = (50, 81)
BLOB_SIZE_RANGE = (4, 55)
BLOB_HEIGHT_RANGE = (2, 11)
BLOB_WIDTH_RANGE = (2, 9)
BLOB_MIN_FILL_RATIO = 0.56
RECT_BLOB_PROBABILITY = 0.35
BORDER_MARGIN = 1
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Outer retry budget for rejected tile/placement attempts.
GEN_RETRY_LIMIT = 300


def _gen_line_positions(length: int) -> set[int]:
    lines = {OUTER_GRID_LINE_INDEX}
    if length >= LINE_MIN_LENGTH and random.random() < LINE_PROBABILITY:
        step = random.randint(LINE_STEP_RANGE[0], min(LINE_STEP_RANGE[1], length - 1))
        lines.update(range(step, length, step))
    return lines


def _gen_neighbors(r: int, c: int, height: int, width: int):
    """Yield in-bounds 4-neighbors in the legacy order."""
    for dr, dc in NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield int(nr), int(nc)


def _gen_inner_orbits(rows: list[int], cols: list[int]) -> list[list[tuple[int, int]]]:
    """Group inner tile cells into LR+UD symmetry orbits."""
    orbits = []
    for i in range((len(rows) + 1) // 2):
        paired_rows = {rows[i], rows[-1 - i]}
        for j in range((len(cols) + 1) // 2):
            paired_cols = {cols[j], cols[-1 - j]}
            orbits.append([(r, c) for r in paired_rows for c in paired_cols])
    return orbits


def _gen_min_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for th in range(1, h + 1):
        for tw in range(1, w + 1):
            if np.array_equal(tile, tile[rows % th, cols % tw]):
                return th, tw
    return h, w


def _gen_build_tile() -> np.ndarray | None:
    """Sample a full-period symmetric tile, or reject it."""
    tile_h = random.randint(*TILE_HEIGHT_RANGE)
    tile_w = random.randint(*TILE_WIDTH_RANGE)
    row_lines = _gen_line_positions(tile_h)
    col_lines = _gen_line_positions(tile_w)
    inner_rows = [r for r in range(tile_h) if r not in row_lines]
    inner_cols = [c for c in range(tile_w) if c not in col_lines]
    orbits = _gen_inner_orbits(inner_rows, inner_cols)
    if len(orbits) < 2:
        return None

    n_colors = random.randint(INNER_COLOR_COUNT_RANGE[0], min(INNER_COLOR_COUNT_RANGE[1], len(orbits)))
    colors = random.sample(NON_BLUE_COLORS, n_colors)
    values = colors + [random.choice(colors) for _ in range(len(orbits) - n_colors)]
    random.shuffle(values)

    tile = np.full((tile_h, tile_w), GRID_LINE_COLOR, dtype=int)
    for cells, color in zip(orbits, values):
        for r, c in cells:
            tile[r, c] = color

    if _gen_min_period(tile) != (tile_h, tile_w):
        return None
    return tile


def _gen_wallpaper(tile: np.ndarray) -> np.ndarray:
    h, w = tile.shape
    rows = np.arange(CANVAS_SIZE)[:, None] % h
    cols = np.arange(CANVAS_SIZE)[None, :] % w
    return tile[rows, cols]


def _gen_blob_sizes(count: int, total: int) -> list[int]:
    sizes = []
    remaining = total
    for i in range(count):
        left = count - i - 1
        lo = max(BLOB_SIZE_RANGE[0], remaining - BLOB_SIZE_RANGE[1] * left)
        hi = min(BLOB_SIZE_RANGE[1], remaining - BLOB_SIZE_RANGE[0] * left)
        size = random.randint(lo, hi)
        sizes.append(size)
        remaining -= size
    random.shuffle(sizes)
    return sizes


def _gen_blob_shape(size: int) -> np.ndarray | None:
    """Sample a connected blob mask with the requested area."""
    dims = [
        (h, w)
        for h in range(BLOB_HEIGHT_RANGE[0], BLOB_HEIGHT_RANGE[1] + 1)
        for w in range(BLOB_WIDTH_RANGE[0], BLOB_WIDTH_RANGE[1] + 1)
        if h + w - 1 <= size <= h * w and size / (h * w) >= BLOB_MIN_FILL_RATIO
    ]
    if not dims:
        return None

    rect_dims = [(h, w) for h, w in dims if h * w == size]
    if rect_dims and random.random() < RECT_BLOB_PROBABILITY:
        h, w = random.choice(rect_dims)
        return np.ones((h, w), dtype=bool)

    h, w = random.choice(dims)
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    steps = [(1, 0)] * (h - 1) + [(0, 1)] * (w - 1)
    random.shuffle(steps)
    for dr, dc in steps:
        r += dr
        c += dc
        mask[r, c] = True

    while int(mask.sum()) < size:
        frontier = []
        for rr, cc in np.argwhere(mask):
            for nr, nc in _gen_neighbors(rr, cc, h, w):
                if not mask[nr, nc]:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _gen_touches_existing(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        if mask[r, c]:
            return True
        for nr, nc in _gen_neighbors(r, c, CANVAS_SIZE, CANVAS_SIZE):
            if mask[nr, nc]:
                return True
    return False


def _gen_phase_counts(tile_h: int, tile_w: int) -> np.ndarray:
    rows = np.arange(CANVAS_SIZE) % tile_h
    cols = np.arange(CANVAS_SIZE) % tile_w
    counts = np.zeros((tile_h, tile_w), dtype=int)
    np.add.at(counts, (rows[:, None], cols[None, :]), 1)
    return counts


def _gen_place_blob(
    black: np.ndarray, shape: np.ndarray, phase_counts: np.ndarray, tile_h: int, tile_w: int
) -> bool:
    """Place one blob without touching existing blobs or hiding a tile phase."""
    h, w = shape.shape
    offsets = [(int(r), int(c)) for r, c in np.argwhere(shape)]
    choices = []
    for r0 in range(BORDER_MARGIN, CANVAS_SIZE - h):
        for c0 in range(BORDER_MARGIN, CANVAS_SIZE - w):
            cells = [(r0 + r, c0 + c) for r, c in offsets]
            if _gen_touches_existing(black, cells):
                continue
            next_counts = phase_counts.copy()
            for r, c in cells:
                next_counts[r % tile_h, c % tile_w] -= 1
            if np.all(next_counts > 0):
                choices.append((r0, c0, next_counts))
    if not choices:
        return False

    r0, c0, next_counts = random.choice(choices)
    for r, c in offsets:
        black[r0 + r, c0 + c] = True
    phase_counts[:, :] = next_counts
    return True


def _gen_attempt() -> np.ndarray | None:
    """Run one output-first generation attempt and oracle-check it."""
    tile = _gen_build_tile()
    if tile is None:
        return None

    out = _gen_wallpaper(tile)
    tile_h, tile_w = tile.shape
    blob_count = random.randint(*BLOB_COUNT_RANGE)
    total_black = random.randint(*BLACK_TOTAL_RANGE)
    sizes = _gen_blob_sizes(blob_count, total_black)
    black = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    phase_counts = _gen_phase_counts(tile_h, tile_w)

    for size in sizes:
        shape = _gen_blob_shape(size)
        if shape is None or not _gen_place_blob(black, shape, phase_counts, tile_h, tile_w):
            return None

    inp = out.copy()
    inp[black] = MISSING_COLOR
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
