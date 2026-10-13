import numpy as np

from arc_utils import components_mask, get_bg, is_contiguous


def _foreground_tile_for_period(
    grid: np.ndarray, bg: int, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg:
                continue
            key = (r % row_period, c % col_period)
            if key in tile and tile[key] != color:
                return None
            tile[key] = color
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the erased cells of a periodic wallpaper by inferring its smallest repeating tile.

    Input structure
    A rectangular grid contains a multi-color repeating pattern. The background
    color is also a valid tile color, and some non-background pattern pixels
    have been erased to that background color.

    Rule
    Ignore background-colored cells and find the unique smallest row/column
    period whose residue classes assign every visible foreground pixel one
    consistent color. Repaint each foreground residue class across the whole
    grid, leaving residue classes with no foreground evidence as background.

    Key insight
    The visible foreground pixels are enough to recover the wallpaper's period;
    erased pixels are exactly the background cells in foreground residue classes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height > 0 and width > 0

    values, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1
    bg = get_bg(g)
    assert int(counts[values == bg][0]) == max_count

    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            tile = _foreground_tile_for_period(g, bg, row_period, col_period)
            if tile is not None:
                candidates.append((row_period, col_period, tile))

    assert candidates
    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [
        candidate
        for candidate in candidates
        if candidate[0] * candidate[1] == min_area
    ]
    assert len(best) == 1
    row_period, col_period, tile = best[0]
    assert tile

    out = np.full_like(g, bg)
    for (tile_r, tile_c), color in tile.items():
        out[tile_r::row_period, tile_c::col_period] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and period bounds.
GEN_GRID_HEIGHT = 22
GEN_GRID_WIDTH = 25
GEN_BACKGROUND = 1
GEN_K_MIN = 5
GEN_K_MAX = 8
GEN_COL_PERIOD_MIN = 4

# Offset sampling limits.
GEN_OFFSET_TRIES = 80
GEN_ZERO_OFFSET_MIN = 2
GEN_MUTATION_DIVISOR = 2

# Erasure rectangle and component constraints.
GEN_RECT_HEIGHT_MIN = 2
GEN_RECT_HEIGHT_MAX = 5
GEN_RECT_WIDTH_MIN = 2
GEN_RECT_WIDTH_MAX = 5
GEN_BLOB_COUNT_MIN = 3
GEN_BLOB_COUNT_MAX = 6
GEN_BLOB_SIZE_MIN = 5
GEN_BLOB_SIZE_MAX = 25
GEN_RECT_TRIES = 40
GEN_RECT_PADDING = 1
GEN_ERASED_FRACTION_MIN = 0.08
GEN_ERASED_FRACTION_MAX = 0.15
GEN_MASK_BACKGROUND = 0
GEN_CONNECTIVITY = 4

# Top-level retry limit.
GEN_ATTEMPT_TRIES = 300


def _gen_has_subperiod(values: list[int]) -> bool:
    n = len(values)
    for p in range(1, n):
        if n % p == 0 and all(values[i] == values[i % p] for i in range(n)):
            return True
    return False


def _gen_wallpaper(row_period: int, col_period: int, offsets: list[int]) -> np.ndarray:
    rows = np.arange(GEN_GRID_HEIGHT)[:, None]
    cols = np.arange(GEN_GRID_WIDTH)[None, :]
    offset_arr = np.asarray(offsets, dtype=int)
    return ((rows - offset_arr[cols % col_period]) % row_period + GEN_BACKGROUND).astype(int)


def _gen_bg_is_unique_max(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    bg_count = int(counts[values == GEN_BACKGROUND][0])
    return bg_count == max_count and int(np.count_nonzero(counts == max_count)) == 1


def _gen_cover_offsets(row_period: int) -> list[int]:
    high_residues = set(range(GEN_GRID_HEIGHT % row_period))
    uncovered = set(range(1, row_period))
    chosen: list[int] = []
    while uncovered:
        candidates = list(high_residues)
        np.random.shuffle(candidates)
        best = max(
            candidates,
            key=lambda a: sum((a + t) % row_period not in high_residues for t in uncovered),
        )
        chosen.append(int(best))
        uncovered = {
            t for t in uncovered if (best + t) % row_period in high_residues
        }
    return chosen


def _gen_offsets(row_period: int, col_period: int) -> list[int] | None:
    extra_rows = GEN_GRID_HEIGHT % row_period
    for _ in range(GEN_OFFSET_TRIES):
        if extra_rows == 1:
            zero_count = int(np.random.randint(GEN_ZERO_OFFSET_MIN, col_period + 1))
            offsets = np.random.randint(1, row_period, size=col_period).astype(int).tolist()
            for idx in np.random.choice(col_period, size=zero_count, replace=False):
                offsets[int(idx)] = 0
        else:
            offsets = _gen_cover_offsets(row_period)
            while len(offsets) < col_period:
                offsets.append(int(np.random.randint(0, extra_rows)))
            np.random.shuffle(offsets)
            for _mut in range(int(np.random.randint(0, col_period // GEN_MUTATION_DIVISOR + 1))):
                idx = int(np.random.randint(0, col_period))
                old = offsets[idx]
                offsets[idx] = int(np.random.randint(0, row_period))
                if not _gen_bg_is_unique_max(_gen_wallpaper(row_period, col_period, offsets)):
                    offsets[idx] = old

        if _gen_has_subperiod(offsets):
            continue
        if _gen_bg_is_unique_max(_gen_wallpaper(row_period, col_period, offsets)):
            return offsets
    return None


def _gen_rect_candidates(foreground: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for height in range(GEN_RECT_HEIGHT_MIN, GEN_RECT_HEIGHT_MAX + 1):
        for width in range(GEN_RECT_WIDTH_MIN, GEN_RECT_WIDTH_MAX + 1):
            for r in range(0, GEN_GRID_HEIGHT - height + 1):
                for c in range(0, GEN_GRID_WIDTH - width + 1):
                    size = int(np.count_nonzero(foreground[r : r + height, c : c + width]))
                    if GEN_BLOB_SIZE_MIN <= size <= GEN_BLOB_SIZE_MAX:
                        candidates.append((r, c, height, width, size))
    return candidates


def _gen_choose_rectangles(
    output: np.ndarray, min_erased: int, max_erased: int
) -> list[tuple[int, int, int, int]] | None:
    foreground = output != GEN_BACKGROUND
    candidates = _gen_rect_candidates(foreground)
    for _ in range(GEN_RECT_TRIES):
        blob_count = int(np.random.randint(GEN_BLOB_COUNT_MIN, GEN_BLOB_COUNT_MAX + 1))
        blocked = np.zeros(output.shape, dtype=bool)
        chosen: list[tuple[int, int, int, int]] = []
        total = 0
        failed = False
        for i in range(blob_count):
            remaining = blob_count - i - 1
            low = max(GEN_BLOB_SIZE_MIN, min_erased - total - remaining * GEN_BLOB_SIZE_MAX)
            high = min(GEN_BLOB_SIZE_MAX, max_erased - total - remaining * GEN_BLOB_SIZE_MIN)
            if low > high:
                failed = True
                break

            choice = None
            for idx in np.random.permutation(len(candidates)):
                r, c, height, width, size = candidates[int(idx)]
                if size < low or size > high:
                    continue
                if blocked[r : r + height, c : c + width].any():
                    continue
                patch = foreground[r : r + height, c : c + width]
                if is_contiguous(patch, background=GEN_MASK_BACKGROUND, connectivity=GEN_CONNECTIVITY):
                    choice = (r, c, height, width, size)
                    break

            if choice is None:
                failed = True
                break

            r, c, height, width, size = choice
            chosen.append((r, c, height, width))
            total += size
            r0 = max(0, r - GEN_RECT_PADDING)
            c0 = max(0, c - GEN_RECT_PADDING)
            r1 = min(GEN_GRID_HEIGHT, r + height + GEN_RECT_PADDING)
            c1 = min(GEN_GRID_WIDTH, c + width + GEN_RECT_PADDING)
            blocked[r0:r1, c0:c1] = True

        if not failed and min_erased <= total <= max_erased:
            return chosen
    return None


def _gen_all_foreground_residues_visible(
    inp: np.ndarray, output: np.ndarray, row_period: int, col_period: int
) -> bool:
    for r in range(row_period):
        for c in range(col_period):
            if int(output[r, c]) == GEN_BACKGROUND:
                continue
            if not np.any(inp[r::row_period, c::col_period] != GEN_BACKGROUND):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = int(np.random.randint(GEN_K_MIN, GEN_K_MAX + 1))
    row_period = k + 1
    col_period = int(np.random.randint(GEN_COL_PERIOD_MIN, row_period + 1))
    offsets = _gen_offsets(row_period, col_period)
    if offsets is None:
        return None

    output = _gen_wallpaper(row_period, col_period, offsets)
    if not _gen_bg_is_unique_max(output):
        return None

    foreground_total = int(np.count_nonzero(output != GEN_BACKGROUND))
    min_erased = int(np.ceil(foreground_total * GEN_ERASED_FRACTION_MIN))
    max_erased = int(np.floor(foreground_total * GEN_ERASED_FRACTION_MAX))
    rectangles = _gen_choose_rectangles(output, min_erased, max_erased)
    if rectangles is None:
        return None

    inp = output.copy()
    for r, c, height, width in rectangles:
        patch_mask = output[r : r + height, c : c + width] != GEN_BACKGROUND
        inp[r : r + height, c : c + width][patch_mask] = GEN_BACKGROUND

    erased = inp != output
    erased_total = int(np.count_nonzero(erased))
    if not (min_erased <= erased_total <= max_erased):
        return None
    components = components_mask(erased, connectivity=GEN_CONNECTIVITY)
    if not (GEN_BLOB_COUNT_MIN <= len(components) <= GEN_BLOB_COUNT_MAX):
        return None
    if any(len(component) < GEN_BLOB_SIZE_MIN or len(component) > GEN_BLOB_SIZE_MAX for component in components):
        return None
    if not _gen_all_foreground_residues_visible(inp, output, row_period, col_period):
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ea959feb input")
