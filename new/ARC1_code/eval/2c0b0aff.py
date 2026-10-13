from __future__ import annotations

import numpy as np

from arc_utils import colors_present, components_mask, crop_bbox, find_components, paste
import random


def _boundary_bg(g: np.ndarray) -> int:
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "expected a grid with a boundary"

    boundary = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    colors, counts = np.unique(boundary, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "expected a unique dominant boundary background color"
    return int(winners[0])


def _marker_color(g: np.ndarray, bg: int) -> int:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    counts = {color: int(np.sum(g == color)) for color in colors}
    min_count = min(counts.values())
    markers = [color for color, count in counts.items() if count == min_count]
    assert len(markers) == 1, "expected a unique rarer foreground marker color"
    return int(markers[0])


def _area_five_marker_count(rect: np.ndarray, marker: int) -> int:
    marker_components = components_mask(rect == int(marker), connectivity=4)
    return sum(1 for coords in marker_components if int(coords.shape[0]) == 5)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the rectangle whose marker-color mask contains the most five-pixel connected objects.

    Input structure:
    - A boundary-colored background separates several solid rectangular foreground regions.
    - Each region is made from two foreground colors: a dominant fill color and a rarer marker color.
    - The marker-color pixels form 4-connected objects of several sizes.

    Rule:
    - Infer the background as the unique dominant boundary color and the marker color as the unique rarer non-background color.
    - For each foreground rectangle, count 4-connected marker-color objects with area exactly five pixels.
    - Return the unique rectangle with the largest count, unchanged.

    Key insight:
    - The desired crop is not determined by position or size; it is the region containing the most complete five-pixel marker objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_bg(g)
    marker = _marker_color(g, bg)

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(regions) >= 2, "expected multiple candidate foreground rectangles"

    scored_regions: list[tuple[int, np.ndarray]] = []
    for region in regions:
        rect = crop_bbox(g, region["bbox"])
        assert np.all(rect != bg), "each foreground component must fill its rectangular bbox"
        scored_regions.append((_area_five_marker_count(rect, marker), rect))

    best_score = max(score for score, _rect in scored_regions)
    winners = [rect for score, rect in scored_regions if score == best_score]
    assert len(winners) == 1, "expected a unique rectangle with the most area-five marker objects"
    return winners[0]

# --------------- generator (o5x) ---------------

# Colors used by all generated grids.
BG_COLOR, MARKER_COLOR, FILL_COLOR = 0, 3, 8

# Canvas and rectangle bounds are inclusive.
GRID_HEIGHT_BOUNDS, GRID_WIDTH_BOUNDS = (22, 24), (20, 24)
REGION_HEIGHT_BOUNDS, REGION_WIDTH_BOUNDS = (5, 10), (7, 11)

# Structural counts are sampled independently except for the unique winner.
REGION_COUNT_BOUNDS, WINNER_PLUS_COUNT_BOUNDS = (3, 4), (3, 5)
NON_WINNER_PLUS_MIN = 1
DISTRACTOR_COUNT_BOUNDS = (1, 9)

# Marker component geometry.
CONNECTIVITY = 4
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
PLUS_AREA, PLUS_CENTER_MARGIN, PLUS_MIN_MANHATTAN = 5, 1, 4
NEIGHBOR4_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DISTRACTOR_SIZES, DISTRACTOR_OFFSETS = (1, 2), ((1, 0), (0, 1))
VALID_MARKER_COMPONENT_SIZES = (1, 2, 5)

# Placement and retry policy.
REGION_GAP, LAYOUT_SHUFFLE_ATTEMPTS, GENERATION_RETRIES = 1, 10, 300
ROW_LAYOUT, COL_LAYOUT = "rows", "cols"


def _gen_plus_cells(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for dr, dc in PLUS_OFFSETS)


def _gen_choose_plus_centers(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    """Choose non-overlapping plus centers with the required Manhattan spacing."""
    candidates = [
        (r, c)
        for r in range(PLUS_CENTER_MARGIN, height - PLUS_CENTER_MARGIN)
        for c in range(PLUS_CENTER_MARGIN, width - PLUS_CENTER_MARGIN)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []

    def search(pool: list[tuple[int, int]]) -> bool:
        if len(chosen) == count:
            return True
        if len(pool) < count - len(chosen):
            return False
        random.shuffle(pool)
        for i, pos in enumerate(pool):
            rest = [
                other
                for other in pool[i + 1 :]
                if abs(int(other[0]) - int(pos[0])) + abs(int(other[1]) - int(pos[1])) >= PLUS_MIN_MANHATTAN
            ]
            chosen.append(pos)
            if search(rest):
                return True
            chosen.pop()
        return False

    if search(candidates):
        return chosen.copy()
    return None


def _gen_component_touches_marker(rect: np.ndarray, cells: tuple[tuple[int, int], ...]) -> bool:
    own = set(cells)
    height, width = rect.shape
    for r, c in cells:
        for dr, dc in NEIGHBOR4_OFFSETS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in own:
                continue
            if 0 <= nr < height and 0 <= nc < width and int(rect[nr, nc]) == MARKER_COLOR:
                return True
    return False


def _gen_distractor_candidates(rect: np.ndarray, size: int) -> list[tuple[tuple[int, int], ...]]:
    height, width = rect.shape
    out: list[tuple[tuple[int, int], ...]] = []
    if size == 1:
        for r in range(height):
            for c in range(width):
                cells = ((r, c),)
                if int(rect[r, c]) == FILL_COLOR and not _gen_component_touches_marker(rect, cells):
                    out.append(cells)
    else:
        for r in range(height):
            for c in range(width):
                for dr, dc in DISTRACTOR_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if nr >= height or nc >= width:
                        continue
                    cells = ((r, c), (nr, nc))
                    if (
                        int(rect[r, c]) == FILL_COLOR
                        and int(rect[nr, nc]) == FILL_COLOR
                        and not _gen_component_touches_marker(rect, cells)
                    ):
                        out.append(cells)
    random.shuffle(out)
    return out


def _gen_add_distractors(rect: np.ndarray, count: int) -> bool:
    area = int(rect.size)
    max_marker = (area - 1) // 2
    for placed in range(count):
        remaining_after = count - placed - 1
        marker_now = int(np.sum(rect == MARKER_COLOR))
        allowed_sizes = [s for s in DISTRACTOR_SIZES if marker_now + s + remaining_after <= max_marker]
        random.shuffle(allowed_sizes)
        picked: tuple[tuple[int, int], ...] | None = None
        for size in allowed_sizes:
            candidates = _gen_distractor_candidates(rect, size)
            if candidates:
                picked = random.choice(candidates)
                break
        if picked is None:
            return False
        for r, c in picked:
            rect[r, c] = MARKER_COLOR
    return True


def _gen_marker_is_rarer(grid: np.ndarray) -> bool:
    return int(np.sum(grid == MARKER_COLOR)) < int(np.sum(grid == FILL_COLOR))


def _gen_build_region(plus_count: int, distractor_count: int) -> np.ndarray | None:
    dims = [
        (height, width)
        for height in range(REGION_HEIGHT_BOUNDS[0], REGION_HEIGHT_BOUNDS[1] + 1)
        for width in range(REGION_WIDTH_BOUNDS[0], REGION_WIDTH_BOUNDS[1] + 1)
        if (height * width - 1) // 2 >= PLUS_AREA * plus_count + distractor_count
    ]
    random.shuffle(dims)

    for height, width in dims:
        centers = _gen_choose_plus_centers(height, width, plus_count)
        if centers is None:
            continue
        rect = np.full((height, width), FILL_COLOR, dtype=int)
        for center_r, center_c in centers:
            for r, c in _gen_plus_cells(center_r, center_c):
                rect[r, c] = MARKER_COLOR
        if not _gen_add_distractors(rect, distractor_count):
            continue
        component_sizes = sorted(
            int(comp.shape[0]) for comp in components_mask(rect == MARKER_COLOR, connectivity=CONNECTIVITY)
        )
        if component_sizes.count(PLUS_AREA) != plus_count:
            continue
        if any(size not in VALID_MARKER_COMPONENT_SIZES for size in component_sizes):
            continue
        if not _gen_marker_is_rarer(rect):
            continue
        return rect
    return None


def _gen_region_fits(canvas: np.ndarray, rect: np.ndarray, r0: int, c0: int) -> bool:
    height, width = rect.shape
    grid_h, grid_w = canvas.shape
    rr0 = max(0, r0 - REGION_GAP)
    cc0 = max(0, c0 - REGION_GAP)
    rr1 = min(grid_h, r0 + height + REGION_GAP)
    cc1 = min(grid_w, c0 + width + REGION_GAP)
    return not bool(np.any(canvas[rr0:rr1, cc0:cc1] != BG_COLOR))


def _gen_split_slack(amount: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(max(0, amount)):
        parts[random.randrange(slots)] += 1
    return parts


def _gen_try_row_groups(canvas: np.ndarray, regions: list[np.ndarray], groups: list[list[int]]) -> np.ndarray | None:
    grid_h, grid_w = canvas.shape
    row_heights = [max(int(regions[idx].shape[0]) for idx in group) for group in groups]
    row_widths = [
        sum(int(regions[idx].shape[1]) for idx in group) + REGION_GAP * (len(group) - 1)
        for group in groups
    ]
    total_h = sum(row_heights) + REGION_GAP * (len(groups) - 1)
    if total_h > grid_h or any(width > grid_w for width in row_widths):
        return None

    out = canvas.copy()
    vertical_parts = _gen_split_slack(grid_h - total_h, len(groups) + 1)
    top = vertical_parts[0]
    for group_i, group in enumerate(groups):
        band_h = row_heights[group_i]
        horizontal_parts = _gen_split_slack(grid_w - row_widths[group_i], len(group) + 1)
        left = horizontal_parts[0]
        for item_i, idx in enumerate(group):
            rect = regions[idx]
            rect_h, rect_w = rect.shape
            r = top + random.randint(0, band_h - rect_h)
            c = left
            if not _gen_region_fits(out, rect, r, c):
                return None
            paste(out, rect, r, c)
            left += rect_w + REGION_GAP + horizontal_parts[item_i + 1]
        top += band_h + REGION_GAP + vertical_parts[group_i + 1]
    return out


def _gen_try_col_groups(canvas: np.ndarray, regions: list[np.ndarray], groups: list[list[int]]) -> np.ndarray | None:
    grid_h, grid_w = canvas.shape
    col_widths = [max(int(regions[idx].shape[1]) for idx in group) for group in groups]
    col_heights = [
        sum(int(regions[idx].shape[0]) for idx in group) + REGION_GAP * (len(group) - 1)
        for group in groups
    ]
    total_w = sum(col_widths) + REGION_GAP * (len(groups) - 1)
    if total_w > grid_w or any(height > grid_h for height in col_heights):
        return None

    out = canvas.copy()
    horizontal_parts = _gen_split_slack(grid_w - total_w, len(groups) + 1)
    left = horizontal_parts[0]
    for group_i, group in enumerate(groups):
        band_w = col_widths[group_i]
        vertical_parts = _gen_split_slack(grid_h - col_heights[group_i], len(group) + 1)
        top = vertical_parts[0]
        for item_i, idx in enumerate(group):
            rect = regions[idx]
            rect_h, rect_w = rect.shape
            r = top
            c = left + random.randint(0, band_w - rect_w)
            if not _gen_region_fits(out, rect, r, c):
                return None
            paste(out, rect, r, c)
            top += rect_h + REGION_GAP + vertical_parts[item_i + 1]
        left += band_w + REGION_GAP + horizontal_parts[group_i + 1]
    return out


def _gen_layout_specs(order: list[int]) -> list[tuple[str, list[list[int]]]]:
    """Return the row/column groupings tried for one shuffled region order."""
    if len(order) == 3:
        return [
            (ROW_LAYOUT, [order[:2], order[2:]]),
            (ROW_LAYOUT, [order[:1], order[1:]]),
            (COL_LAYOUT, [order[:2], order[2:]]),
            (COL_LAYOUT, [order[:1], order[1:]]),
        ]
    return [
        (ROW_LAYOUT, [order[:2], order[2:]]),
        (COL_LAYOUT, [order[:2], order[2:]]),
        (ROW_LAYOUT, [order[:3], order[3:]]),
        (ROW_LAYOUT, [order[:1], order[1:]]),
        (COL_LAYOUT, [order[:3], order[3:]]),
        (COL_LAYOUT, [order[:1], order[1:]]),
    ]


def _gen_place_regions(canvas: np.ndarray, regions: list[np.ndarray]) -> np.ndarray | None:
    order = list(range(len(regions)))
    layouts: list[tuple[str, list[list[int]]]] = []

    for _ in range(LAYOUT_SHUFFLE_ATTEMPTS):
        random.shuffle(order)
        layouts.extend(_gen_layout_specs(order))

    random.shuffle(layouts)
    for kind, groups in layouts:
        placed = (
            _gen_try_row_groups(canvas, regions, groups)
            if kind == ROW_LAYOUT
            else _gen_try_col_groups(canvas, regions, groups)
        )
        if placed is not None:
            return placed
    return None


def _gen_plus_counts(region_count: int) -> tuple[list[int], int]:
    winner_idx = random.randrange(region_count)
    winner_count = random.randint(*WINNER_PLUS_COUNT_BOUNDS)
    counts = [random.randint(NON_WINNER_PLUS_MIN, winner_count - 1) for _ in range(region_count)]
    counts[winner_idx] = winner_count
    return counts, winner_idx


def _gen_build_regions(plus_counts: list[int], distractor_counts: list[int]) -> list[np.ndarray] | None:
    """Build every foreground region, returning None if any requested region fails."""
    regions: list[np.ndarray] = []
    for plus_count, distractor_count in zip(plus_counts, distractor_counts):
        region = _gen_build_region(plus_count, distractor_count)
        if region is None:
            return None
        regions.append(region)
    return regions


def _gen_matches_solver(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_BOUNDS)
    width = random.randint(*GRID_WIDTH_BOUNDS)
    region_count = random.randint(*REGION_COUNT_BOUNDS)
    plus_counts, winner_idx = _gen_plus_counts(region_count)
    distractor_counts = [random.randint(*DISTRACTOR_COUNT_BOUNDS) for _ in range(region_count)]

    regions = _gen_build_regions(plus_counts, distractor_counts)
    if regions is None:
        return None

    canvas = np.full((height, width), BG_COLOR, dtype=int)
    placed = _gen_place_regions(canvas, regions)
    if placed is None:
        return None
    if not _gen_marker_is_rarer(placed):
        return None

    expected = regions[winner_idx]
    if not _gen_matches_solver(placed, expected):
        return None
    return placed


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 2c0b0aff input")
