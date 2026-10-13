import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each partial row by applying its left-prefix color mapping to the full template row pattern.

    Input structure
    - The background is the dominant color.
    - Exactly one row is fully non-background; this row is the two-color template pattern.
    - Every other nonempty foreground row starts at column 0, has a contiguous
      non-background prefix, and then background cells to the right.

    Rule
    - For each partial row, pair each prefix cell with the template color in the
      same column to decode that row's color mapping.
    - The prefix must define exactly one output color for each of the two template colors.
    - Fill the entire row by replacing each template color with its decoded row color.

    Key insight
    - The complete row supplies only the sequence of roles; each shorter row's
      prefix gives the row-specific colors for those roles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    height, width = map(int, g.shape)

    template_rows = [r for r in range(height) if np.all(g[r, :] != bg)]
    assert len(template_rows) == 1, "expected exactly one full non-background template row"
    template_r = template_rows[0]
    template = g[template_r, :]
    template_colors = set(map(int, template))
    assert len(template_colors) == 2, "expected exactly two template colors"

    out = g.copy()
    filled_rows = 0
    for r in range(height):
        if r == template_r:
            continue

        prefix_cols = np.flatnonzero(g[r, :] != bg)
        if len(prefix_cols) == 0:
            continue

        expected_prefix = np.arange(int(prefix_cols[-1]) + 1)
        assert np.array_equal(prefix_cols, expected_prefix), "foreground row must be a left prefix"
        assert int(prefix_cols[-1]) < width - 1, "only the template row may already be full width"

        mapping = {}
        for c in map(int, prefix_cols):
            role = int(template[c])
            color = int(g[r, c])
            if role in mapping:
                assert mapping[role] == color, "prefix must map each template color consistently"
            else:
                mapping[role] = color

        assert set(mapping) == template_colors, "prefix must define every template color"
        for c in range(width):
            out[r, c] = mapping[int(template[c])]
        filled_rows += 1

    assert filled_rows > 0, "expected at least one partial row to complete"
    return out

# --------------- generator (o5x) ---------------

# Canvas and layout bounds.
MIN_WIDTH, MAX_WIDTH = 8, 10
MIN_HEIGHT, MAX_HEIGHT = 6, 14
TEMPLATE_ROW, HEADER_ROWS = 1, 2
BG_COLOR = 0

# Colors and template-shape constraints.
MIN_COLOR, MAX_COLOR = 1, 9
FOREGROUND_COLORS = np.arange(MIN_COLOR, MAX_COLOR + 1)
UNORDERED_COLOR_PAIRS = [
    (a, b) for a in range(MIN_COLOR, MAX_COLOR + 1) for b in range(a + 1, MAX_COLOR + 1)
]
MIN_TEMPLATE_COUNT, MAX_TEMPLATE_COUNT = 2, 6
TEMPLATE_FRACTION_DIVISOR = 4
MAX_TEMPLATE_RUN = 3
BALANCE_RATIO, BALANCE_SLACK = 3, 1

# Partial rows and retry limits.
MIN_PARTIAL_COUNT, MAX_PARTIAL_COUNT = 1, 4
MIN_TRAILING, MAX_TRAILING = 1, 2
MIN_GAP, MAX_GAP = 1, 3
MAX_GENERATION_ATTEMPTS = 300


def _gen_template_roles(width, rng):
    min_count = max(
        MIN_TEMPLATE_COUNT, (width + TEMPLATE_FRACTION_DIVISOR - 1) // TEMPLATE_FRACTION_DIVISOR
    )
    max_count = min(MAX_TEMPLATE_COUNT, width - min_count)
    valid_counts = [
        c for c in range(min_count, max_count + 1)
        if min_count <= width - c <= MAX_TEMPLATE_COUNT
    ]
    remaining = [int(rng.choice(valid_counts)), 0]
    remaining[1] = width - remaining[0]

    roles = []
    last = -1
    run = 0
    for _ in range(width):
        options = []
        weights = []
        for role in (0, 1):
            if remaining[role] == 0 or (role == last and run >= MAX_TEMPLATE_RUN):
                continue

            trial = remaining[:]
            trial[role] -= 1
            if max(trial) <= BALANCE_RATIO * (min(trial) + BALANCE_SLACK):
                options.append(role)
                weights.append(remaining[role])

        if not options:
            return None

        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        role = int(rng.choice(options, p=weights))
        roles.append(role)
        remaining[role] -= 1
        if role == last:
            run += 1
        else:
            last, run = role, 1

    return roles


def _gen_color_pairs(count, rng):
    chosen = rng.choice(len(UNORDERED_COLOR_PAIRS), size=count, replace=False)
    pairs = []
    for idx in chosen:
        a, b = UNORDERED_COLOR_PAIRS[int(idx)]
        if rng.integers(2):
            a, b = b, a
        pairs.append((a, b))
    return pairs


def _gen_gaps(count, trailing, rng):
    low = max(MIN_GAP * count, MIN_HEIGHT - HEADER_ROWS - count - trailing)
    high = min(MAX_GAP * count, MAX_HEIGHT - HEADER_ROWS - count - trailing)
    total = int(rng.integers(low, high + 1))

    gaps = [MIN_GAP] * count
    remaining = total - MIN_GAP * count
    while remaining:
        expandable = [i for i, gap in enumerate(gaps) if gap < MAX_GAP]
        i = int(rng.choice(expandable))
        gaps[i] += 1
        remaining -= 1
    return gaps


def _render_grids(width, roles, template, min_prefix, gaps, trailing, rng):
    height = HEADER_ROWS + len(gaps) + sum(gaps) + trailing
    inp = np.full((height, width), BG_COLOR, dtype=int)
    out = np.zeros_like(inp)
    inp[TEMPLATE_ROW, :] = out[TEMPLATE_ROW, :] = template

    row = TEMPLATE_ROW
    for pair, gap in zip(_gen_color_pairs(len(gaps), rng), gaps):
        row += gap + 1
        full = np.asarray(pair, dtype=int)[roles]
        inp[row, :min_prefix] = full[:min_prefix]
        out[row, :] = full
    return inp, out


def _gen_attempt():
    rng = np.random.default_rng()
    width = int(rng.integers(MIN_WIDTH, MAX_WIDTH + 1))
    roles = _gen_template_roles(width, rng)
    if roles is None:
        return None

    template_colors = rng.choice(FOREGROUND_COLORS, size=2, replace=False)
    template = np.asarray(template_colors, dtype=int)[roles]
    min_prefix = next(i + 1 for i in range(width) if len(set(roles[:i + 1])) == 2)

    partial_count = int(rng.integers(MIN_PARTIAL_COUNT, MAX_PARTIAL_COUNT + 1))
    trailing = int(rng.integers(MIN_TRAILING, MAX_TRAILING + 1))
    gaps = _gen_gaps(partial_count, trailing, rng)

    inp, out = _render_grids(width, roles, template, min_prefix, gaps, trailing, rng)
    try:
        solved = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 82819916 input")
