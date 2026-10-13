import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

# --------------- solver ---------------


def _split_ranges(length: int, separators: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(separators):
        assert 0 < sep < length, "separator must be strictly internal"
        assert start < sep, "separators must leave non-empty regions"
        ranges.append((start, sep))
        start = sep + 1
    assert start < length, "final region must be non-empty"
    ranges.append((start, length))
    return ranges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each partition's defective maroon rectangle using the bottom palette.

    The last full-width grey row separates the partitioned area from the palette
    row. Inside each partition, the unique maroon bbox is a rectangle with black
    defect pixels removed. The defect count selects the 1-indexed palette color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 5, "grid is too small"

    full_rows = [r for r in range(h) if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != BLACK]
    assert full_rows, "expected at least one full-width non-background row"
    grey_candidates = {int(g[r, 0]) for r in full_rows}
    assert len(grey_candidates) == 1, "all full-width separator rows must share one color"
    grey = next(iter(grey_candidates))

    palette_sep = max(full_rows)
    assert palette_sep == h - 2, "palette must occupy the single row below the last separator"

    top = g[:palette_sep, :]
    assert top.size > 0, "partitioned area must be non-empty"

    maroon_candidates = set(np.unique(top)) - {BLACK, grey}
    assert len(maroon_candidates) == 1, "expected exactly one object color above the palette"
    maroon = next(iter(maroon_candidates))

    row_separators = [r for r in full_rows if r < palette_sep]
    col_separators = [c for c in range(w) if np.all(top[:, c] == grey)]
    row_ranges = _split_ranges(palette_sep, row_separators)
    col_ranges = _split_ranges(w, col_separators)

    palette = [int(v) for v in g[-1] if int(v) not in {BLACK, grey}]
    assert len(palette) >= 1, "palette row must contain at least one color"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"

    out = g.copy()
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            region = g[r0:r1, c0:c1]
            coords = np.argwhere(region == maroon)
            assert len(coords) > 0, "each partition must contain a maroon object"

            rr0, cc0 = coords.min(axis=0)
            rr1, cc1 = coords.max(axis=0)
            rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
            bbox = region[rr0 : rr1 + 1, cc0 : cc1 + 1]

            assert not np.any(bbox == grey), "separator color must not appear inside an object bbox"
            assert np.all((bbox == BLACK) | (bbox == maroon)), "bbox must be a maroon rectangle with black defects"

            defects = int(np.count_nonzero(bbox == BLACK))
            assert 1 <= defects <= len(palette), "defect count must index the palette"

            out[r0 + rr0 : r0 + rr1 + 1, c0 + cc0 : c0 + cc1 + 1] = palette[defects - 1]

    return out


# --------------- generator ---------------

GEN_MAX_ATTEMPTS = 300
ROW_COUNT_CHOICES = (1, 2, 3)
COL_COUNT_CHOICES = (1, 2, 3)
REGION_SIZE_MIN = 5
REGION_SIZE_MAX = 9
PALETTE_LEN_MIN = 5
PALETTE_LEN_MAX = 7
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DEFECT_COUNT_MIN = 1
DEFECT_COUNT_MAX = 5


def _choose_partition_counts(rng: np.random.Generator) -> tuple[int, int]:
    while True:
        rows = int(rng.choice(ROW_COUNT_CHOICES))
        cols = int(rng.choice(COL_COUNT_CHOICES))
        if rows * cols >= 2:
            return rows, cols


def _draw_rectangle_with_holes(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    defect_count: int,
    rng: np.random.Generator,
) -> None:
    grid[top : top + height, left : left + width] = MAROON
    cells = [(r, c) for r in range(top, top + height) for c in range(left, left + width)]
    chosen = rng.choice(len(cells), size=defect_count, replace=False)
    for idx in np.atleast_1d(chosen):
        r, c = cells[int(idx)]
        grid[r, c] = BLACK


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_count, col_count = _choose_partition_counts(rng)
    region_heights = [int(rng.integers(REGION_SIZE_MIN, REGION_SIZE_MAX + 1)) for _ in range(row_count)]
    region_widths = [int(rng.integers(REGION_SIZE_MIN, REGION_SIZE_MAX + 1)) for _ in range(col_count)]

    top_height = sum(region_heights) + (row_count - 1)
    width = sum(region_widths) + (col_count - 1)
    height = top_height + 2  # one grey separator row, one palette row

    palette_len = int(rng.integers(PALETTE_LEN_MIN, min(PALETTE_LEN_MAX, width) + 1))
    palette = list(map(int, rng.choice(PALETTE_COLORS, size=palette_len, replace=False)))
    palette_start = int(rng.integers(0, width - palette_len + 1))

    grid = np.full((height, width), BLACK, dtype=int)
    palette_sep = height - 2
    grid[palette_sep, :] = GREY
    grid[-1, palette_start : palette_start + palette_len] = palette

    row_ranges: list[tuple[int, int]] = []
    start = 0
    for idx, region_h in enumerate(region_heights):
        row_ranges.append((start, start + region_h))
        start += region_h
        if idx != row_count - 1:
            grid[start, :] = GREY
            start += 1

    col_ranges: list[tuple[int, int]] = []
    start = 0
    for idx, region_w in enumerate(region_widths):
        col_ranges.append((start, start + region_w))
        start += region_w
        if idx != col_count - 1:
            grid[:palette_sep, start] = GREY
            start += 1

    region_specs: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            rh = r1 - r0
            rw = c1 - c0
            rect_h = int(rng.integers(3, rh - 1))
            rect_w = int(rng.integers(3, rw - 1))
            top_margin = int(rng.integers(1, rh - rect_h))
            left_margin = int(rng.integers(1, rw - rect_w))
            region_specs.append((r0 + top_margin, c0 + left_margin, rect_h, rect_w))

    defect_counts = [int(rng.integers(DEFECT_COUNT_MIN, DEFECT_COUNT_MAX + 1)) for _ in region_specs]
    if len(defect_counts) > 1 and len(set(defect_counts)) == 1:
        pivot = int(rng.integers(0, len(defect_counts)))
        choices = [n for n in range(DEFECT_COUNT_MIN, DEFECT_COUNT_MAX + 1) if n != defect_counts[pivot]]
        defect_counts[pivot] = int(rng.choice(choices))

    for (top, left, rect_h, rect_w), defects in zip(region_specs, defect_counts):
        _draw_rectangle_with_holes(grid, top, left, rect_h, rect_w, defects, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(set(defect_counts)) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
