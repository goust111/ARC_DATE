from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np


RED = 2
GREY = 5
BLACK = 0
PATTERN_COLORS = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)

PAIR_COUNT = 3
GAP_WIDTH_RANGE = (3, 5)
HORIZONTAL_MARGIN_RANGE = (0, 3)
TOP_BOTTOM_MARGIN_RANGE = (0, 4)
ROW_GAP_RANGE = (2, 5)
NOISE_COUNT_RANGE = (4, 12)
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_full_2x2(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 4:
        return False
    r0, c0, r1, c1 = _bbox(comp)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return set(comp) == {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _paired_marker_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    red_components = _components(grid == RED)
    blocks = []
    for comp in red_components:
        assert _is_full_2x2(comp), "all red components must be 2x2 marker blocks"
        b = _bbox(comp)
        blocks.append(b)

    assert len(blocks) >= 2 and len(blocks) % 2 == 0, "marker blocks must come in pairs"
    by_row_span: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for block in blocks:
        by_row_span[(block[0], block[2])].append(block)

    pairs: list[dict[str, object]] = []
    for row_span in sorted(by_row_span):
        row_blocks = sorted(by_row_span[row_span], key=lambda b: b[1])
        assert len(row_blocks) == 2, "each marker row span must contain exactly two blocks"
        left, right = row_blocks
        gap_width = right[1] - left[3] - 1
        assert gap_width > 0, "paired markers must have a non-empty interior gap"
        region = grid[left[0] : left[2] + 1, left[3] + 1 : right[1]].copy()
        pairs.append(
            {
                "rows": row_span,
                "left": left,
                "right": right,
                "gap_width": gap_width,
                "region": region,
            }
        )
    return pairs


def _scale_region(region: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(region, np.ones((factor, factor), dtype=int))


def _stack_scaled_regions(regions: list[np.ndarray]) -> np.ndarray:
    scaled = []
    max_width = 0
    for region in regions:
        factor = int(np.count_nonzero(region))
        assert factor > 0, "each extracted region must contain foreground"
        blown = _scale_region(region, factor)
        scaled.append(blown)
        max_width = max(max_width, blown.shape[1])

    out = np.zeros((sum(b.shape[0] for b in scaled), max_width), dtype=int)
    row = 0
    for blown in scaled:
        h, w = blown.shape
        out[row : row + h, :w] = blown
        row += h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the interior of each paired red marker band, scale each interior by its own foreground count, and stack the results.

    Input structure:
    - The background is black, with a few grey noise pixels outside the marked regions.
    - Red pixels form several 2x2 marker blocks, arranged as left/right pairs that share the same two-row span.
    - Between each paired red blocks lies a 2-row colored pattern made of 4-connected non-black pixels.

    Rule:
    - Identify every red 2x2 block and pair the two blocks that occupy the same two-row span.
    - For each pair, crop the rectangle strictly between the inner edges of the markers and count its non-black pixels as K.
    - Replace that cropped region by a K-times pixel enlargement of itself, then stack all enlarged regions from top to bottom.
    - The output width is the widest enlarged region; narrower enlarged regions are left-aligned and padded with black on the right.

    Key insight:
    - Each extracted region determines its own enlargement factor: the same visible pixels both define the picture to copy and the count K that controls how much that picture expands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    pairs = _paired_marker_blocks(g)
    assert pairs, "expected at least one marker pair"
    return _stack_scaled_regions([pair["region"] for pair in pairs])


def _connected_shapes(width: int, area: int) -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(2) for c in range(width)]
    shapes = []
    for choice in itertools.combinations(cells, area):
        mask = np.zeros((2, width), dtype=bool)
        for r, c in choice:
            mask[r, c] = True
        if len(_components(mask)) == 1:
            shapes.append(tuple(sorted(choice)))
    return shapes


SHAPES_BY_WIDTH_AREA = {
    (width, area): _connected_shapes(width, area)
    for width in range(GAP_WIDTH_RANGE[0], GAP_WIDTH_RANGE[1] + 1)
    for area in (2, 3, 4)
}


def _random_region(rng: np.random.Generator, width: int, area: int) -> np.ndarray:
    shape = SHAPES_BY_WIDTH_AREA[(width, area)][int(rng.integers(0, len(SHAPES_BY_WIDTH_AREA[(width, area)])))]
    region = np.zeros((2, width), dtype=int)
    palette_size = int(rng.integers(1, min(4, area) + 1))
    palette = rng.choice(PATTERN_COLORS, size=palette_size, replace=False)
    colors = rng.choice(palette, size=area, replace=True)
    if area >= 3 and palette_size > 1 and len(set(map(int, colors))) == 1:
        colors[0] = int(palette[0])
        colors[-1] = int(palette[-1])
    for (r, c), color in zip(shape, colors):
        region[r, c] = int(color)
    return region


def _place_noise(
    rng: np.random.Generator,
    grid: np.ndarray,
    protected: np.ndarray,
) -> None:
    free = np.argwhere(~protected)
    assert len(free) > 0
    count = int(rng.integers(NOISE_COUNT_RANGE[0], NOISE_COUNT_RANGE[1] + 1))
    count = min(count, len(free))
    chosen = free[rng.choice(len(free), size=count, replace=False)]
    grid[chosen[:, 0], chosen[:, 1]] = GREY


def _uniformly_scaled_stack(regions: list[np.ndarray], factor: int) -> np.ndarray:
    scaled = [_scale_region(region, factor) for region in regions]
    width = max(block.shape[1] for block in scaled)
    out = np.zeros((sum(block.shape[0] for block in scaled), width), dtype=int)
    row = 0
    for block in scaled:
        h, w = block.shape
        out[row : row + h, :w] = block
        row += h
    return out


def _validate_instance(
    grid: np.ndarray,
    expected_regions: list[np.ndarray],
    protected: np.ndarray,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        parsed_pairs = _paired_marker_blocks(grid)
    except AssertionError:
        return False

    if len(parsed_pairs) != PAIR_COUNT:
        return False

    gap_widths = {int(pair["gap_width"]) for pair in parsed_pairs}
    if len(gap_widths) != 1:
        return False

    parsed_regions = [pair["region"] for pair in parsed_pairs]
    for parsed, expected in zip(parsed_regions, expected_regions):
        if not np.array_equal(parsed, expected):
            return False

    counts = [int(np.count_nonzero(region)) for region in parsed_regions]
    if sorted(counts) != [2, 3, 4]:
        return False
    if counts == sorted(counts) or counts == sorted(counts, reverse=True):
        return False

    for region, area in zip(parsed_regions, counts):
        comps = _components(region != 0)
        if len(comps) != 1 or len(comps[0]) != area:
            return False

    outside = grid.copy()
    outside[protected] = 0
    if not set(map(int, np.unique(outside))).issubset({0, GREY}):
        return False

    expected = _stack_scaled_regions(expected_regions)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if max(actual.shape) > 30:
        return False

    if np.array_equal(np.vstack(parsed_regions), expected):
        return False

    for factor in (min(counts), max(counts)):
        if np.array_equal(_uniformly_scaled_stack(parsed_regions, factor), expected):
            return False

    sorted_regions = [region for _, region in sorted(zip(counts, parsed_regions), key=lambda x: x[0])]
    if np.array_equal(_stack_scaled_regions(sorted_regions), expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    gap_width = int(rng.integers(GAP_WIDTH_RANGE[0], GAP_WIDTH_RANGE[1] + 1))

    count_orders = ((2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3))
    counts = count_orders[int(rng.integers(0, len(count_orders)))]

    left_margin = int(rng.integers(HORIZONTAL_MARGIN_RANGE[0], HORIZONTAL_MARGIN_RANGE[1] + 1))
    right_margin = int(rng.integers(HORIZONTAL_MARGIN_RANGE[0], HORIZONTAL_MARGIN_RANGE[1] + 1))
    grid_width = left_margin + 2 + gap_width + 2 + right_margin

    row_gaps = rng.integers(ROW_GAP_RANGE[0], ROW_GAP_RANGE[1] + 1, size=PAIR_COUNT - 1)
    top_margin = int(rng.integers(TOP_BOTTOM_MARGIN_RANGE[0], TOP_BOTTOM_MARGIN_RANGE[1] + 1))
    bottom_margin = int(rng.integers(TOP_BOTTOM_MARGIN_RANGE[0], TOP_BOTTOM_MARGIN_RANGE[1] + 1))
    grid_height = top_margin + 2 + int(row_gaps[0]) + 2 + int(row_gaps[1]) + 2 + bottom_margin
    if max(grid_height, grid_width) > 30:
        return None

    row_starts = [top_margin]
    for gap in row_gaps:
        row_starts.append(row_starts[-1] + 2 + int(gap))

    grid = np.zeros((grid_height, grid_width), dtype=int)
    protected = np.zeros((grid_height, grid_width), dtype=bool)
    expected_regions: list[np.ndarray] = []

    for row_start, area in zip(row_starts, counts):
        left_col = int(rng.integers(0, grid_width - gap_width - 4 + 1))
        right_col = left_col + gap_width + 2

        grid[row_start : row_start + 2, left_col : left_col + 2] = RED
        grid[row_start : row_start + 2, right_col : right_col + 2] = RED

        region = _random_region(rng, gap_width, area)
        grid[row_start : row_start + 2, left_col + 2 : right_col] = region
        protected[row_start : row_start + 2, left_col : right_col + 2] = True
        expected_regions.append(region)

    _place_noise(rng, grid, protected)
    if not _validate_instance(grid, expected_regions, protected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
