import numpy as np

from arc_utils import Color, components_mask, get_bg


PALETTE_ORDER = (Color.GREY, Color.RED, Color.TEAL, Color.MAROON, Color.PINK)


def main(grid: np.ndarray) -> np.ndarray:
    """Convert scattered foreground objects into bottom-aligned count bars in a fixed five-color order.

    Input structure
    The grid has a dominant background color and foreground objects in the task
    palette: grey, red, teal, maroon, and pink. Objects are monochromatic
    8-connected components; some palette colors may be absent.

    Rule
    For each palette color in order grey, red, teal, maroon, pink, count its
    8-connected objects. Create a background grid with five columns and height
    equal to the largest count, then fill each column upward from the bottom
    with its palette color for that count.

    Key insight
    The output does not preserve object shapes or positions; it is a histogram
    of per-color 8-connected object counts, using the learned palette order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    palette_set = set(PALETTE_ORDER)
    assert bg not in palette_set, "background must not be a histogram color"
    assert set(map(int, g.ravel())) <= palette_set | {bg}

    object_counts = [len(components_mask(g == color, connectivity=8)) for color in PALETTE_ORDER]
    max_count = max(object_counts)
    assert max_count > 0, "expected at least one foreground object"

    out = np.full((max_count, len(PALETTE_ORDER)), bg, dtype=int)
    for c, (color, count) in enumerate(zip(PALETTE_ORDER, object_counts)):
        if count:
            out[-count:, c] = color
    return out

# --------------- generator (att1) ---------------

# Canvas and palette.
GRID_SHAPE = (10, 10)
BACKGROUND_COLOR = Color.ORANGE
N_COLORS = len(PALETTE_ORDER)

# Histogram count sampling.
NONZERO_COLOR_RANGE = (3, 6)
MAX_COUNT_RANGE = (2, 7)
TOTAL_OBJECT_RANGE = (4, 12)

# Blob shape sampling.
N_BLOB_SHAPES = 6
LINE_LENGTH_RANGE = (2, 5)
SMALL_DIM_RANGE = (2, 4)
GROWN_BLOB_MIN_SIZE = 3
PLUS_SHAPE_SIZE = 3

# Placement and retry limits.
SAME_COLOR_MIN_CHEBYSHEV = 2
BLOB_PLACE_TRIES = 100
GENERATION_TRIES = 300


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate grid, returning None when placement or validation fails."""
    rng = np.random.default_rng()
    counts = _sample_counts(rng)
    if counts is None:
        return None

    grid = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    if not _place_all_blobs(grid, counts, rng):
        return None
    if not _counts_match_grid(grid, counts):
        return None
    if not _solver_matches_expected(grid, counts):
        return None
    return grid


def _sample_counts(rng) -> list[int] | None:
    """Sample per-color component counts matching the output histogram contract."""
    n_nonzero = int(rng.integers(*NONZERO_COLOR_RANGE))
    max_count = int(rng.integers(*MAX_COUNT_RANGE))

    counts = [0] * N_COLORS
    nonzero_indices = list(rng.choice(N_COLORS, size=n_nonzero, replace=False))
    max_idx = rng.choice(nonzero_indices)
    counts[max_idx] = max_count

    remaining_budget = TOTAL_OBJECT_RANGE[1] - max_count
    for idx in [i for i in nonzero_indices if i != max_idx]:
        upper = min(max_count, remaining_budget)
        if upper < 1:
            counts[idx] = 1
            remaining_budget -= 1
        else:
            counts[idx] = int(rng.integers(1, upper + 1))
            remaining_budget -= counts[idx]

    total = sum(counts)
    if total < TOTAL_OBJECT_RANGE[0] or total > TOTAL_OBJECT_RANGE[1]:
        return None
    return counts


def _place_all_blobs(grid: np.ndarray, counts: list[int], rng) -> bool:
    """Place each color's requested number of same-color-separated blobs."""
    for color_idx, count in enumerate(counts):
        if count == 0:
            continue
        if not _place_color_blobs(grid, PALETTE_ORDER[color_idx], count, rng):
            return False
    return True


def _place_color_blobs(grid: np.ndarray, color: int, count: int, rng) -> bool:
    placed_blobs = []
    for _ in range(count):
        coords = _choose_placeable_blob(grid, placed_blobs, rng)
        if coords is None:
            return False
        _paint_coords(grid, coords, color)
        placed_blobs.append(coords)
    return True


def _choose_placeable_blob(grid: np.ndarray, placed_blobs: list[set], rng) -> set | None:
    for _ in range(BLOB_PLACE_TRIES):
        blob = _random_blob(rng)
        bh, bw = blob.shape
        r0 = int(rng.integers(0, GRID_SHAPE[0] - bh + 1))
        c0 = int(rng.integers(0, GRID_SHAPE[1] - bw + 1))
        coords = _offset_blob_coords(blob, r0, c0)
        if _coords_are_free(grid, coords) and _chebyshev_ok(coords, placed_blobs):
            return coords
    return None


def _offset_blob_coords(blob: np.ndarray, r0: int, c0: int) -> set:
    offsets = np.argwhere(blob)
    return set(map(tuple, offsets + np.array([r0, c0])))


def _coords_are_free(grid: np.ndarray, coords: set) -> bool:
    return all(grid[r, c] == BACKGROUND_COLOR for r, c in coords)


def _paint_coords(grid: np.ndarray, coords: set, color: int) -> None:
    rr, cc = np.array(list(coords)).T
    grid[rr, cc] = color


def _counts_match_grid(grid: np.ndarray, counts: list[int]) -> bool:
    """Recompute 8-connected components to reject accidental same-color merges."""
    for color, expected_count in zip(PALETTE_ORDER, counts):
        mask = grid == color
        if expected_count == 0:
            if mask.any():
                return False
        elif len(components_mask(mask, connectivity=8)) != expected_count:
            return False
    return True


def _solver_matches_expected(grid: np.ndarray, counts: list[int]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return np.array_equal(out, _expected_output(counts))


def _expected_output(counts: list[int]) -> np.ndarray:
    out = np.full((max(counts), N_COLORS), BACKGROUND_COLOR, dtype=int)
    for c_idx, (color, count) in enumerate(zip(PALETTE_ORDER, counts)):
        if count:
            out[-count:, c_idx] = color
    return out


def _random_blob(rng) -> np.ndarray:
    """Generate a random small 8-connected blob mask."""
    shape_type = rng.integers(0, N_BLOB_SHAPES)

    if shape_type == 0:
        return np.array([[True]])
    elif shape_type == 1:
        length = int(rng.integers(*LINE_LENGTH_RANGE))
        return np.ones((1, length), dtype=bool)
    elif shape_type == 2:
        length = int(rng.integers(*LINE_LENGTH_RANGE))
        return np.ones((length, 1), dtype=bool)
    elif shape_type == 3:
        return _random_grown_blob(rng)
    elif shape_type == 4:
        return _plus_blob()
    else:
        h = int(rng.integers(*SMALL_DIM_RANGE))
        w = int(rng.integers(*SMALL_DIM_RANGE))
        return np.ones((h, w), dtype=bool)


def _random_grown_blob(rng) -> np.ndarray:
    """Grow a small mask by repeatedly adding an 8-neighbor of existing pixels."""
    h = int(rng.integers(*SMALL_DIM_RANGE))
    w = int(rng.integers(*SMALL_DIM_RANGE))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    pixels = [(0, 0)]
    target_size = int(rng.integers(GROWN_BLOB_MIN_SIZE, h * w + 1))
    for _ in range(target_size - 1):
        candidates = _growth_candidates(mask, pixels)
        if not candidates:
            break
        pick = list(candidates)[int(rng.integers(0, len(candidates)))]
        mask[pick[0], pick[1]] = True
        pixels.append(pick)
    return mask


def _growth_candidates(mask: np.ndarray, pixels: list[tuple[int, int]]) -> set:
    h, w = mask.shape
    candidates = set()
    for r, c in pixels:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    candidates.add((nr, nc))
    return candidates


def _plus_blob() -> np.ndarray:
    mask = np.zeros((PLUS_SHAPE_SIZE, PLUS_SHAPE_SIZE), dtype=bool)
    center = PLUS_SHAPE_SIZE // 2
    mask[center, :] = True
    mask[:, center] = True
    return mask


def _chebyshev_ok(new_coords: set, existing_blobs: list) -> bool:
    for blob in existing_blobs:
        for r1, c1 in new_coords:
            for r2, c2 in blob:
                if max(abs(r1 - r2), abs(c1 - c2)) < SAME_COLOR_MIN_CHEBYSHEV:
                    return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_TRIES} attempts")
