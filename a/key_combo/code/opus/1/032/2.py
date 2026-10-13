from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
PALETTE = tuple(color for color in range(10) if color not in (BG, GRAY))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

MIN_REGIONS = 3
MAX_REGIONS = 5
MIN_BARS = 2
MAX_BARS = 3
MIN_AREA = 3
MAX_AREA = 20
MAX_GRID = 30
MIN_GRID = 22
MAX_ATTEMPTS = 400


def _component_cells(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    offsets = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in offsets:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _first_gray_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(grid == GRAY, axis=1))
    assert rows.size > 0, "expected gray contours"
    return int(rows[0])


def _read_legend(grid: np.ndarray, drawing_start: int) -> list[tuple[int, int]]:
    legend_mask = (grid[:drawing_start] != BG) & (grid[:drawing_start] != GRAY)
    components = _component_cells(legend_mask, connectivity=4)
    assert MIN_BARS <= len(components) <= MAX_BARS, "expected 2-3 legend bars"

    entries: list[tuple[int, int, int]] = []
    for comp in components:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "legend bar must be monochrome"
        color = colors.pop()
        rows = sorted(r for r, _ in comp)
        cols = {c for _, c in comp}
        assert len(cols) == 1, "legend bar must have width 1"
        assert rows[0] == 0, "legend bars must touch row 0"
        assert rows == list(range(rows[-1] + 1)), "legend bar rows must be contiguous"
        height = rows[-1] + 1
        col = next(iter(cols))
        assert np.all(grid[height:drawing_start, col] == BG), "legend bar must end cleanly"
        entries.append((height, color, col))

    heights = sorted(height for height, _color, _col in entries)
    assert heights == list(range(1, len(entries) + 1)), "legend heights must be 1..k"
    return [(height, color) for height, color, _col in sorted(entries)]


def _extract_internal_regions(grid: np.ndarray) -> tuple[int, list[list[tuple[int, int]]]]:
    drawing_start = _first_gray_row(grid)
    zero_components = _component_cells(grid == BG, connectivity=4)
    regions: list[list[tuple[int, int]]] = []

    h, w = grid.shape
    for comp in zero_components:
        touches_edge = any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp)
        touches_legend_rows = any(r < drawing_start for r, _ in comp)
        if touches_edge or touches_legend_rows:
            continue
        regions.append(sorted(comp))

    assert MIN_REGIONS <= len(regions) <= MAX_REGIONS, "expected 3-5 enclosed regions"
    areas = [len(region) for region in regions]
    assert len(set(areas)) == len(areas), "region areas must be unique"
    regions.sort(key=lambda region: -len(region))
    return drawing_start, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed background regions according to legend bars whose heights index the area ranking.

    Input structure:
    - The background is black and gray pixels form irregular contour loops below a top legend area.
    - The top 2-3 rows contain 2-3 colored vertical bars touching row 0, separated from the contours by at least one blank row.
    - The contours enclose 3-5 distinct 4-connected background regions of different areas.

    Rule:
    - Find all 4-connected background regions using non-background pixels as barriers.
    - Discard any background region that touches the grid edge or reaches the legend rows above the first gray contour row.
    - Rank the remaining enclosed regions by area from largest to smallest.
    - Read each legend bar as (color K, height h) and recolor the region with area-rank h to color K.
    - Leave unmatched enclosed regions as background and preserve all contour and legend pixels.

    Key insight:
    - The legend height is not an area value; it is a rank index into the descending area order of the enclosed background regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    drawing_start, ranked_regions = _extract_internal_regions(g)
    legend_entries = _read_legend(g, drawing_start)

    out = g.copy()
    for height, color in legend_entries:
        if height > len(ranked_regions):
            continue
        for r, c in ranked_regions[height - 1]:
            out[r, c] = color
    return out


def _region_mask_from_cells(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _polyomino(area: int) -> list[tuple[int, int]] | None:
    max_dim = 6 if area <= 12 else 7
    for _ in range(200):
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    neighbors = sum((nr + rr, nc + cc) in cells for rr, cc in NEIGHBORS4)
                    weight = 1 + neighbors
                    frontier.extend([(nr, nc)] * weight)
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            trial = set(cells)
            trial.add((nr, nc))
            normalized = _normalize(trial)
            rows = [r for r, _ in normalized]
            cols = [c for _, c in normalized]
            if max(rows) + 1 <= max_dim and max(cols) + 1 <= max_dim:
                cells = trial
        normalized = _normalize(cells)
        if len(normalized) != area:
            continue
        rows = [r for r, _ in normalized]
        cols = [c for _, c in normalized]
        height = max(rows) + 1
        width = max(cols) + 1
        bbox_area = height * width
        if height <= 1 or width <= 1:
            continue
        if bbox_area == area:
            continue
        if bbox_area > area * 2 + 2:
            continue
        return normalized
    return None


def _contour_patch(region_cells: list[tuple[int, int]]) -> dict[str, object]:
    region_mask = _region_mask_from_cells(region_cells)
    h, w = region_mask.shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    region_coords: list[tuple[int, int]] = []
    ring_coords: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if not region_mask[r, c]:
                continue
            pr, pc = r + 1, c + 1
            region_coords.append((pr, pc))
            for dr, dc in NEIGHBORS8:
                ring_coords.add((pr + dr, pc + dc))
            ring_coords.add((pr, pc))

    gray_cells = sorted(cell for cell in ring_coords if cell not in region_coords)
    for r, c in gray_cells:
        patch[r, c] = GRAY

    return {
        "patch": patch,
        "region": sorted(region_coords),
        "area": len(region_coords),
        "bbox_area": h * w,
    }


def _choose_areas(count: int) -> list[int]:
    return random.sample(range(MIN_AREA, MAX_AREA + 1), count)


def _rank_order(values: list[int], reverse: bool) -> list[int]:
    indexed = list(range(len(values)))
    return sorted(indexed, key=lambda idx: values[idx], reverse=reverse)


def _is_monotone_shortcut(regions: list[dict[str, object]]) -> bool:
    area_order = _rank_order([int(region["area"]) for region in regions], reverse=True)
    bbox_order = _rank_order([int(region["bbox_area"]) for region in regions], reverse=True)
    row_order = _rank_order(
        [sum(r for r, _ in region["abs_region"]) / int(region["area"]) for region in regions],
        reverse=False,
    )
    col_order = _rank_order(
        [sum(c for _, c in region["abs_region"]) / int(region["area"]) for region in regions],
        reverse=False,
    )
    mirrors = (row_order, list(reversed(row_order)), col_order, list(reversed(col_order)), bbox_order)
    return any(area_order == order for order in mirrors)


def _place_regions(
    grid: np.ndarray,
    draw_top: int,
    patches: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:draw_top, :] = True
    placed: list[dict[str, object]] = []

    order = patches[:]
    random.shuffle(order)
    for spec in order:
        patch = np.asarray(spec["patch"])
        ph, pw = patch.shape
        if draw_top + ph + 2 >= h or pw + 2 >= w:
            return None
        location = None
        for _ in range(250):
            top = random.randint(draw_top + 1, h - ph - 1)
            left = random.randint(1, w - pw - 1)
            if blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1].any():
                continue
            location = (top, left)
            break
        if location is None:
            return None

        top, left = location
        gray_coords = np.argwhere(patch == GRAY)
        for rr, cc in gray_coords:
            grid[top + int(rr), left + int(cc)] = GRAY
        blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True

        abs_region = [(top + r, left + c) for r, c in spec["region"]]
        placed.append(
            {
                "area": int(spec["area"]),
                "bbox_area": int(spec["bbox_area"]),
                "abs_region": sorted(abs_region),
            }
        )

    return placed


def _place_legend(grid: np.ndarray, bar_count: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    assert bar_count in (2, 3)
    columns = list(range(1, w - 1))
    random.shuffle(columns)
    chosen: list[int] = []
    for col in columns:
        if all(abs(col - prev) >= 2 for prev in chosen):
            chosen.append(col)
            if len(chosen) == bar_count:
                break
    if len(chosen) != bar_count:
        raise AssertionError("failed to place legend bars")
    chosen.sort()

    heights = list(range(1, bar_count + 1))
    for _ in range(40):
        random.shuffle(heights)
        if heights != sorted(heights) and heights != sorted(heights, reverse=True):
            break
    colors = random.sample(PALETTE, bar_count)
    entries = [(height, color, col) for height, color, col in zip(heights, colors, chosen)]
    for height, color, col in entries:
        grid[:height, col] = color
    return [(height, color) for height, color, _col in entries]


def _expected_from_metadata(
    grid: np.ndarray,
    regions: list[dict[str, object]],
    legend_entries: list[tuple[int, int]],
) -> np.ndarray:
    ranked = sorted(regions, key=lambda region: int(region["area"]), reverse=True)
    out = grid.copy()
    for height, color in legend_entries:
        if height > len(ranked):
            continue
        for r, c in ranked[height - 1]["abs_region"]:
            out[r, c] = color
    return out


def _validate_structure(
    grid: np.ndarray,
    regions: list[dict[str, object]],
    legend_entries: list[tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    try:
        drawing_start, detected_regions = _extract_internal_regions(grid)
        detected_entries = _read_legend(grid, drawing_start)
    except AssertionError:
        return False

    metadata_regions = sorted(
        [sorted(region["abs_region"]) for region in regions],
        key=lambda region: (-len(region), region[0]),
    )
    detected_regions_sorted = sorted(
        [sorted(region) for region in detected_regions],
        key=lambda region: (-len(region), region[0]),
    )
    if metadata_regions != detected_regions_sorted:
        return False
    if sorted(legend_entries) != sorted(detected_entries):
        return False
    if _is_monotone_shortcut(regions):
        return False
    return np.array_equal(main(grid), expected)


def _attempt_generate() -> np.ndarray | None:
    region_count = random.randint(MIN_REGIONS, MAX_REGIONS)
    bar_count = random.randint(MIN_BARS, min(MAX_BARS, region_count))
    legend_rows = bar_count
    separator_rows = random.randint(1, 2)
    draw_top = legend_rows + separator_rows

    areas = _choose_areas(region_count)
    patches: list[dict[str, object]] = []
    for area in areas:
        poly = _polyomino(area)
        if poly is None:
            return None
        patches.append(_contour_patch(poly))

    height = random.randint(max(MIN_GRID, draw_top + 14), MAX_GRID)
    width = random.randint(MIN_GRID, MAX_GRID)
    grid = np.full((height, width), BG, dtype=int)

    placed_regions = _place_regions(grid, draw_top, patches)
    if placed_regions is None:
        return None

    legend_entries = _place_legend(grid, bar_count)
    expected = _expected_from_metadata(grid, placed_regions, legend_entries)

    if not _validate_structure(grid, placed_regions, legend_entries, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-ranking puzzle")
