import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg


def _paired_marker_blocks(g: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    bg = get_bg(g)
    candidates = []

    for color in colors_present(g, exclude={bg}):
        blocks = []
        for comp in find_components(g, bg=bg, colors=[color], connectivity=4):
            b = comp["bbox"]
            if comp["area"] != 4 or bbox_h(b) != 2 or bbox_w(b) != 2:
                blocks = []
                break
            blocks.append(b)

        if not blocks or len(blocks) % 2 != 0:
            continue

        by_row_span: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for b in blocks:
            by_row_span.setdefault((b[0], b[2]), []).append(b)

        pairs = []
        gap_widths = set()
        for row_span in sorted(by_row_span):
            row_blocks = sorted(by_row_span[row_span], key=lambda b: b[1])
            if len(row_blocks) != 2:
                pairs = []
                break

            left, right = row_blocks
            gap_width = right[1] - left[3] - 1
            if gap_width <= 0:
                pairs = []
                break

            pairs.append((left, right))
            gap_widths.add(gap_width)

        if pairs and len(gap_widths) == 1:
            candidates.append(pairs)

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Extract and stack the contents between each row-aligned pair of 2x2 marker blocks.

    Input structure
    - The grid contains noise plus one marker color whose components are all 2x2 blocks.
    - The marker blocks occur as left/right pairs sharing the same two-row span.

    Rule
    - For each paired marker span in top-to-bottom order, copy the rectangle strictly between the two marker blocks.
    - Stack those copied rectangles vertically, discarding the markers and all cells outside the copied rectangles.

    Key insight
    - The extraction markers are identifiable as the unique color made only of 2x2 components arranged in row-aligned pairs.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _paired_marker_blocks(g)

    stripes = []
    widths = set()
    for left, right in pairs:
        assert left[0] == right[0] and left[2] == right[2]
        stripe = g[left[0] : left[2] + 1, left[3] + 1 : right[1]]
        assert stripe.shape[0] == 2 and stripe.shape[1] > 0
        stripes.append(stripe)
        widths.add(stripe.shape[1])

    assert stripes and len(widths) == 1
    return np.vstack(stripes)

# --------------- generator (o5x) ---------------

GEN_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
BACKGROUND_COLOR, MARKER_COLOR, ARC_COLOR_COUNT = 0, 8, 10

# Canvas and paired-marker layout bounds.
NUM_PAIRS_MIN, NUM_PAIRS_HIGH = 1, 5
GAP_WIDTH_MIN, GAP_WIDTH_HIGH = 3, 7
GRID_HEIGHT_MIN, GRID_HEIGHT_HIGH = 12, 16
MIN_GRID_WIDTH, GRID_WIDTH_HIGH = 10, 18
MIN_WIDTH_GAP_PADDING, LEFT_COL_HIGH_MARGIN, MARKER_SIZE = 6, 9, 2
ROW_GAP_MIN, ROW_GAP_HIGH = 1, 4

# Content densities and retry limits.
STRIPE_DENSITY_MIN, STRIPE_DENSITY_MAX = 0.50, 0.75
NON_MARKER_DENSITY_MIN, NON_MARKER_DENSITY_MAX = 0.35, 0.45
MAX_GENERATION_ATTEMPTS = 300

def _gen_row_starts(rng: np.random.Generator, height: int, num_pairs: int) -> list[int] | None:
    """Return top rows for 2-row marker spans with sampled vertical gaps."""
    if num_pairs == 1:
        return [int(rng.integers(0, height - MARKER_SIZE + 1))]
    gaps = rng.integers(ROW_GAP_MIN, ROW_GAP_HIGH, size=num_pairs - 1)
    used_height = MARKER_SIZE * num_pairs + int(gaps.sum())
    extra = height - used_height
    if extra < 0:
        return None

    row = int(rng.integers(0, extra + 1))
    starts = [row]
    for gap in gaps:
        row += MARKER_SIZE + int(gap)
        starts.append(row)
    return starts

def _gen_output(rng: np.random.Generator, num_pairs: int, gap_width: int) -> np.ndarray:
    """Sample the stacked stripe payload later copied between marker pairs."""
    out = np.zeros((MARKER_SIZE * num_pairs, gap_width), dtype=int)
    area = out.size
    density = float(rng.uniform(STRIPE_DENSITY_MIN, STRIPE_DENSITY_MAX))
    count = int(round(area * density))
    count = max(1, min(area, count))
    locs = rng.choice(area, size=count, replace=False)
    out.flat[locs] = rng.choice(GEN_COLORS, size=count, replace=True)
    return out

def _choose_layout(rng: np.random.Generator):
    """Sample canvas dimensions and pair anchors, or return None if impossible."""
    num_pairs = int(rng.integers(NUM_PAIRS_MIN, NUM_PAIRS_HIGH))
    gap_width = int(rng.integers(GAP_WIDTH_MIN, GAP_WIDTH_HIGH))
    height = int(rng.integers(GRID_HEIGHT_MIN, GRID_HEIGHT_HIGH))
    min_width = max(MIN_GRID_WIDTH, gap_width + MIN_WIDTH_GAP_PADDING, num_pairs + LEFT_COL_HIGH_MARGIN)
    if min_width >= GRID_WIDTH_HIGH:
        return None
    width = int(rng.integers(min_width, GRID_WIDTH_HIGH))
    row_starts = _gen_row_starts(rng, height, num_pairs)
    if row_starts is None:
        return None
    left_options = np.arange(0, width - LEFT_COL_HIGH_MARGIN)
    if len(left_options) < num_pairs:
        return None
    left_cols = rng.choice(left_options, size=num_pairs, replace=False)
    return height, width, num_pairs, gap_width, row_starts, left_cols

def _render_marker_pairs(grid: np.ndarray, protected: np.ndarray, expected: np.ndarray, row_starts, left_cols, gap_width: int) -> None:
    """Place 2x2 markers and copied stripe strips, marking cells excluded from noise."""
    for i, (row, col) in enumerate(zip(row_starts, left_cols)):
        strip = expected[MARKER_SIZE * i : MARKER_SIZE * i + MARKER_SIZE]
        right_col = col + MARKER_SIZE + gap_width
        grid[row : row + MARKER_SIZE, col : col + MARKER_SIZE] = MARKER_COLOR
        grid[row : row + MARKER_SIZE, col + MARKER_SIZE : right_col] = strip
        grid[row : row + MARKER_SIZE, right_col : right_col + MARKER_SIZE] = MARKER_COLOR
        protected[row : row + MARKER_SIZE, col : right_col + MARKER_SIZE] = True

def _add_noise(rng: np.random.Generator, grid: np.ndarray, protected: np.ndarray, expected: np.ndarray, num_pairs: int) -> bool:
    """Fill unprotected cells while forcing every non-marker color to appear."""
    total = grid.size
    marker_count = 2 * MARKER_SIZE * MARKER_SIZE * num_pairs
    stripe_non_bg = int(np.count_nonzero(expected))
    min_non_marker = max(int(np.ceil(NON_MARKER_DENSITY_MIN * total)), stripe_non_bg + len(GEN_COLORS))
    max_non_marker = min(int(np.floor(NON_MARKER_DENSITY_MAX * total)), total - 2 * marker_count - 1)
    if min_non_marker > max_non_marker:
        return False

    target_non_marker = int(rng.integers(min_non_marker, max_non_marker + 1))
    noise_count = target_non_marker - stripe_non_bg
    free = np.argwhere(~protected)
    if noise_count < len(GEN_COLORS) or noise_count > len(free):
        return False

    noise_coords = free[rng.choice(len(free), size=noise_count, replace=False)]
    noise_colors = rng.choice(GEN_COLORS, size=noise_count, replace=True)
    forced = GEN_COLORS.copy()
    rng.shuffle(forced)
    noise_colors[: len(forced)] = forced
    order = rng.permutation(noise_count)
    grid[noise_coords[order, 0], noise_coords[order, 1]] = noise_colors[order]
    return True

def _passes_oracle(grid: np.ndarray, expected: np.ndarray) -> bool:
    """Check background dominance and solver output match."""
    counts = np.bincount(grid.ravel(), minlength=ARC_COLOR_COUNT)
    if counts[BACKGROUND_COLOR] <= max(counts[1:]):
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)

def _gen_attempt() -> np.ndarray | None:
    """Run one complete generation attempt; callers handle retries."""
    rng = np.random.default_rng()
    layout = _choose_layout(rng)
    if layout is None:
        return None

    height, width, num_pairs, gap_width, row_starts, left_cols = layout
    expected = _gen_output(rng, num_pairs, gap_width)
    grid = np.zeros((height, width), dtype=int)
    protected = np.zeros((height, width), dtype=bool)
    _render_marker_pairs(grid, protected, expected, row_starts, left_cols, gap_width)

    if not _add_noise(rng, grid, protected, expected, num_pairs):
        return None
    return grid if _passes_oracle(grid, expected) else None

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid aab50785 input")
