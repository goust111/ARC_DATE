import numpy as np


BG = 0
PATTERN_COLOR = 5
MARKER_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

COUNT_CHOICES = (2, 3, 4)
SIZE_RANGE_BY_COUNT = {
    2: (4, 10),
    3: (4, 7),
    4: (4, 5),
}
GAP_RANGE = (1, 2)
PAD_RANGE = (0, 2)
GEN_RETRIES = 300


def _edge_pairs(line: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    pairs: dict[int, list[int]] = {}
    for idx, value in enumerate(line):
        color = int(value)
        if color == bg:
            continue
        pairs.setdefault(color, []).append(idx)

    out: dict[int, tuple[int, int]] = {}
    for color, positions in pairs.items():
        assert len(positions) == 2, "each tick color must appear exactly twice on an edge"
        a, b = sorted(int(x) for x in positions)
        assert a > 0 and a < b, "tick markers must define a non-empty interior span"
        out[color] = (a, b)
    return out


def _rectangles_from_ticks(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 4
    assert int(g[0, 0]) == BG, "top-left corner should stay background"

    top = _edge_pairs(g[0], BG)
    left = _edge_pairs(g[:, 0], BG)
    assert top and set(top) == set(left), "top and left ticks must use the same colors"

    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in top:
        c0, c1 = top[color]
        r0, r1 = left[color]
        rects[color] = (r0, r1, c0, c1)
    return rects


def _crop_non_bg(region: np.ndarray, bg: int) -> np.ndarray | None:
    mask = region != bg
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return region[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tile_patch_into(out: np.ndarray, rect: tuple[int, int, int, int], patch: np.ndarray, bg: int) -> None:
    r0, r1, c0, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    ph, pw = patch.shape
    reps_r = (h + ph - 1) // ph
    reps_c = (w + pw - 1) // pw
    tiled = np.tile(patch, (reps_r, reps_c))[:h, :w]
    target = out[r0 : r1 + 1, c0 : c1 + 1]
    mask = tiled != bg
    target[mask] = tiled[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate row/column ranges from paired edge ticks, then tile the source motif into the other matching rectangles.

    Input structure:
    - Background is black.
    - The top edge and the left edge contain colored tick markers. Each tick color appears exactly twice on the top edge and exactly twice on the left edge.
    - For each color, the top-edge pair defines a column span and the left-edge pair defines a row span; their intersection is one rectangular subregion.
    - Exactly one color's subregion contains a small non-background motif, while every other color's subregion is empty.

    Rule:
    - Read each color's paired top-edge markers as the left/right bounds of a column interval, and its paired left-edge markers as the top/bottom bounds of a row interval.
    - Find the unique subregion that already contains the motif, and crop that motif to the minimal bounding box of its non-background pixels.
    - Copy this cropped motif into every other color's subregion by periodic tiling from that subregion's top-left corner, clipping any overflow at the subregion boundary.
    - Keep the original edge tick markers and the original source motif unchanged.

    Key insight:
    - Matching colors on the top and left edges jointly encode a hidden rectangle, and only one such rectangle reveals the motif that must be repeated across the others.
    """
    g = np.asarray(grid, dtype=int)
    rects = _rectangles_from_ticks(g)
    assert len(rects) >= 2

    source_color = None
    source_patch = None
    for color, (r0, r1, c0, c1) in rects.items():
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        patch = _crop_non_bg(region, BG)
        if patch is None:
            continue
        assert source_color is None, "exactly one source region should be non-empty"
        source_color = color
        source_patch = patch

    assert source_color is not None and source_patch is not None, "a unique source region is required"

    out = g.copy()
    for color, rect in rects.items():
        if color == source_color:
            continue
        r0, r1, c0, c1 = rect
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == BG), "target regions must start empty"
        _tile_patch_into(out, rect, source_patch, BG)
    return out


def _axis_intervals(rng: np.random.Generator, count: int) -> tuple[list[tuple[int, int]], int] | None:
    low, high = SIZE_RANGE_BY_COUNT[count]
    sizes = [int(rng.integers(low, high + 1)) for _ in range(count)]
    gaps = [int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1)) for _ in range(count - 1)]
    lead = int(rng.integers(PAD_RANGE[0], PAD_RANGE[1] + 1))
    tail = int(rng.integers(PAD_RANGE[0], PAD_RANGE[1] + 1))
    total = 1 + lead + sum(sizes) + sum(gaps) + tail
    if total > 30:
        return None

    intervals: list[tuple[int, int]] = []
    cursor = 1 + lead
    for idx, size in enumerate(sizes):
        intervals.append((cursor, cursor + size - 1))
        cursor += size
        if idx < len(gaps):
            cursor += gaps[idx]
    return intervals, total


def _random_pattern(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    for _ in range(200):
        density = float(rng.uniform(0.35, 0.7))
        mask = rng.random((height, width)) < density
        if int(mask.sum()) < 2 or bool(mask.all()):
            continue
        if not bool(mask.any(axis=0).all()) or not bool(mask.any(axis=1).all()):
            continue
        patch = np.full((height, width), BG, dtype=int)
        patch[mask] = PATTERN_COLOR
        return patch
    return None


def _expected_output(
    grid: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
    source_color: int,
    pattern_patch: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    for color, rect in rects.items():
        if color == source_color:
            continue
        _tile_patch_into(out, rect, pattern_patch, BG)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.choice(COUNT_CHOICES))
    row_layout = _axis_intervals(rng, count)
    col_layout = _axis_intervals(rng, count)
    if row_layout is None or col_layout is None:
        return None
    row_intervals, height = row_layout
    col_intervals, width = col_layout

    row_sizes = [r1 - r0 + 1 for r0, r1 in row_intervals]
    col_sizes = [c1 - c0 + 1 for c0, c1 in col_intervals]
    min_row = min(row_sizes)
    min_col = min(col_sizes)
    if min_row < 4 or min_col < 4:
        return None

    max_ph = min(5, min_row - 1)
    max_pw = min(5, min_col - 1)
    if max_ph < 2 or max_pw < 2:
        return None

    pattern_h = int(rng.integers(2, max_ph + 1))
    pattern_w = int(rng.integers(2, max_pw + 1))
    pattern_patch = _random_pattern(rng, pattern_h, pattern_w)
    if pattern_patch is None:
        return None

    colors = [int(c) for c in rng.choice(np.array(MARKER_POOL, dtype=int), size=count, replace=False)]
    row_perm = list(range(count))
    col_perm = list(range(count))
    rng.shuffle(row_perm)
    rng.shuffle(col_perm)

    rects: dict[int, tuple[int, int, int, int]] = {}
    for idx, color in enumerate(colors):
        r0, r1 = row_intervals[row_perm[idx]]
        c0, c1 = col_intervals[col_perm[idx]]
        rects[color] = (r0, r1, c0, c1)

    source_color = int(rng.choice(np.array(colors, dtype=int)))
    sr0, sr1, sc0, sc1 = rects[source_color]
    source_h = sr1 - sr0 + 1
    source_w = sc1 - sc0 + 1
    if not (pattern_h < source_h or pattern_w < source_w):
        return None

    row_offset = int(rng.integers(0, source_h - pattern_h + 1))
    col_offset = int(rng.integers(0, source_w - pattern_w + 1))

    grid = np.full((height, width), BG, dtype=int)
    for color, (r0, r1, c0, c1) in rects.items():
        grid[0, c0] = color
        grid[0, c1] = color
        grid[r0, 0] = color
        grid[r1, 0] = color

    source_region = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
    source_region[row_offset : row_offset + pattern_h, col_offset : col_offset + pattern_w] = pattern_patch

    expected = _expected_output(grid, rects, source_color, pattern_patch)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
