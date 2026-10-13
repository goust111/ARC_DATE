from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


LINE_COLORS = (1, 2, 3)
FILL_COLORS = (4, 5, 7)
BACKGROUND = 0

GRID_H_RANGE = (13, 20)
GRID_W_RANGE = (14, 22)
RECT_SIDE_RANGE = (4, 7)  # outer size including the 1-cell-thick border
REGION_COUNT_CHOICES = (1, 2, 3)
REGION_COUNT_WEIGHTS = (4, 4, 2)
GAP_COUNT_CHOICES = (1, 2, 3)
GAP_COUNT_WEIGHTS = (5, 3, 1)
DISTRACTOR_COUNT_CHOICES = (1, 2)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert w >= 2, "legend requires at least two columns"
    entries: dict[int, int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    r = h - 1
    while r >= 0 and grid[r, 0] != BACKGROUND and grid[r, 1] != BACKGROUND:
        key = int(grid[r, 0])
        value = int(grid[r, 1])
        assert key not in entries, "legend key colors must be unique"
        entries[key] = value
        legend_mask[r, 0] = True
        legend_mask[r, 1] = True
        r -= 1
    assert entries, "expected a bottom-left vertical legend"
    return entries, legend_mask


def _complete_gaps(grid: np.ndarray, legend_mask: np.ndarray | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if legend_mask is None:
        legend_mask = np.zeros_like(g, dtype=bool)

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if g[r, c] != BACKGROUND or legend_mask[r, c]:
                continue
            candidates: list[int] = []
            if 0 < c < w - 1 and g[r, c - 1] == g[r, c + 1] != BACKGROUND:
                candidates.append(int(g[r, c - 1]))
            if 0 < r < h - 1 and g[r - 1, c] == g[r + 1, c] != BACKGROUND:
                candidates.append(int(g[r - 1, c]))
            if not candidates:
                continue
            fill = candidates[0]
            assert all(color == fill for color in candidates), "ambiguous gap completion"
            out[r, c] = fill
    return out


def _outside_background_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if g[r, c] == BACKGROUND and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r, c] == BACKGROUND and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == BACKGROUND and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _enclosed_regions(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    outside = _outside_background_mask(g)
    visited = outside.copy()
    regions: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if g[r, c] != BACKGROUND or visited[r, c]:
                continue
            mask = np.zeros_like(g, dtype=bool)
            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            mask[r, c] = True
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == BACKGROUND and not visited[nr, nc]:
                        visited[nr, nc] = True
                        mask[nr, nc] = True
                        q.append((nr, nc))
            regions.append(mask)
    return regions


def _region_boundary_color(grid: np.ndarray, region: np.ndarray) -> int:
    colors = set()
    for r, c in np.argwhere(region):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not region[nr, nc]:
                color = int(grid[nr, nc])
                if color != BACKGROUND:
                    colors.add(color)
    assert len(colors) == 1, "each enclosed region must have a single boundary color"
    return next(iter(colors))


def _solve_via_helpers(grid: np.ndarray) -> np.ndarray:
    legend, legend_mask = _find_legend(grid)
    completed = _complete_gaps(grid, legend_mask)
    out = completed.copy()
    for region in _enclosed_regions(completed):
        boundary_color = _region_boundary_color(completed, region)
        assert boundary_color in legend, "every region boundary color must appear in the legend"
        out[region] = legend[boundary_color]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete single-pixel gaps in colored line borders, then flood-fill the newly closed regions via a legend.

    Input structure:
    - The background is black (0).
    - Colored 1-pixel-thick horizontal and vertical line segments use colors from {blue(1), red(2), green(3)}.
    - Some line segments contain an internal 1-cell black gap; the bottom-left corner holds a vertical 2-column legend whose left cell is a boundary color key and right cell is the fill color.

    Rule:
    - Fill every black cell that is the unique 1-cell gap inside a horizontal or vertical same-color segment.
    - On the repaired grid, 4-connected flood-fill the black background from the border to mark the exterior.
    - Every remaining black component is an enclosed region; its 4-neighbor boundary pixels all share one line color C.
    - Look up C in the legend and recolor the enclosed region with the legend value, leaving the legend, line pixels, repaired gap pixels, and exterior background unchanged.

    Key insight:
    - The holes do not exist in the raw input: they appear only after line completion, and the final fill color comes indirectly from the legend keyed by the boundary color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve_via_helpers(g)


def _expanded_box(r0: int, c0: int, r1: int, c1: int, pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_legend(grid: np.ndarray, legend_order: list[int], mapping: dict[int, int]) -> None:
    h = grid.shape[0]
    start = h - len(legend_order)
    for i, key in enumerate(legend_order):
        row = start + i
        grid[row, 0] = key
        grid[row, 1] = mapping[key]


def _rectangle_ring(
    r0: int,
    c0: int,
    height: int,
    width: int,
    color: int,
    gap_count: int,
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]], set[tuple[int, int]]]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    side_points = {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }
    gap_candidates = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }

    chosen_sides = random.sample(list(gap_candidates), k=min(gap_count, 4))
    gaps: set[tuple[int, int]] = set()
    for side in chosen_sides:
        gaps.add(random.choice(gap_candidates[side]))

    cells: dict[tuple[int, int], int] = {}
    for pts in side_points.values():
        for pos in pts:
            if pos not in gaps:
                cells[pos] = color

    interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    return cells, gaps, interior


def _try_place_rectangles(
    grid: np.ndarray,
    region_colors: list[int],
    legend_rows: int,
) -> tuple[list[int], set[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    boundary_colors: list[int] = []
    expected_gaps: set[tuple[int, int]] = set()
    region_limit_row = h - legend_rows - 2
    if region_limit_row < 6:
        return None

    for color in region_colors:
        success = False
        for _ in range(160):
            height = random.randint(*RECT_SIDE_RANGE)
            width = random.randint(*RECT_SIDE_RANGE)
            if region_limit_row - height < 1 or w - width < 5:
                continue
            r0 = random.randint(1, region_limit_row - height)
            c0 = random.randint(4, w - width - 1)
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            box = (r0, c0, r1, c1)
            if any(
                _boxes_overlap(
                    _expanded_box(*box, pad=2, h=h, w=w),
                    _expanded_box(*other, pad=2, h=h, w=w),
                )
                for other in placed_boxes
            ):
                continue
            gap_count = random.choices(GAP_COUNT_CHOICES, weights=GAP_COUNT_WEIGHTS, k=1)[0]
            cells, gaps, _ = _rectangle_ring(r0, c0, height, width, color, gap_count)
            if any(grid[r, c] != BACKGROUND for r, c in cells):
                continue
            for (r, c), value in cells.items():
                grid[r, c] = value
            placed_boxes.append(box)
            boundary_colors.append(color)
            expected_gaps |= gaps
            success = True
            break
        if not success:
            return None
    return boundary_colors, expected_gaps, placed_boxes


def _place_distractor_segment(
    grid: np.ndarray,
    color: int,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]] | None:
    h, w = grid.shape
    broken = random.random() < 0.6
    for _ in range(120):
        horizontal = random.random() < 0.5
        length = random.randint(5, 8)
        if horizontal:
            if w - length - 1 < 4 or h < 3:
                continue
            r = random.randint(1, h - 2)
            c0 = random.randint(4, w - length - 1)
            c1 = c0 + length - 1
            box = (r, c0, r, c1)
            cells = {(r, c): color for c in range(c0, c1 + 1)}
            gap = {(r, random.randint(c0 + 1, c1 - 1))} if broken else set()
        else:
            if h - length - 1 < 1 or w < 6:
                continue
            r0 = random.randint(1, h - length - 1)
            c = random.randint(4, w - 2)
            r1 = r0 + length - 1
            box = (r0, c, r1, c)
            cells = {(r, c): color for r in range(r0, r1 + 1)}
            gap = {(random.randint(r0 + 1, r1 - 1), c)} if broken else set()

        expanded = _expanded_box(*box, pad=1, h=h, w=w)
        if any(_boxes_overlap(expanded, _expanded_box(*other, pad=2, h=h, w=w)) for other in boxes):
            continue
        if any(grid[r, c] != BACKGROUND for r, c in cells if (r, c) not in gap):
            continue
        for (r, c), value in cells.items():
            if (r, c) not in gap:
                grid[r, c] = value
        return gap, box
    return None


def _semantic_output(grid: np.ndarray) -> np.ndarray:
    legend, legend_mask = _find_legend(grid)
    completed = _complete_gaps(grid, legend_mask)
    out = completed.copy()
    for region in _enclosed_regions(completed):
        boundary_color = _region_boundary_color(completed, region)
        out[region] = legend[boundary_color]
    return out


def _validate_example(
    grid: np.ndarray,
    expected_gap_positions: set[tuple[int, int]],
    expected_boundary_colors: list[int],
    extra_legend_colors: set[int],
) -> bool:
    legend, legend_mask = _find_legend(grid)
    if set(legend) != set(expected_boundary_colors) | set(extra_legend_colors):
        return False

    before_regions = _enclosed_regions(grid)
    if before_regions:
        return False

    completed = _complete_gaps(grid, legend_mask)
    changed_positions = {tuple(map(int, pos)) for pos in np.argwhere(completed != grid)}
    if changed_positions != expected_gap_positions:
        return False

    after_regions = _enclosed_regions(completed)
    if len(after_regions) != len(expected_boundary_colors):
        return False

    found_boundary_colors = sorted(_region_boundary_color(completed, region) for region in after_regions)
    if found_boundary_colors != sorted(expected_boundary_colors):
        return False

    if extra_legend_colors & set(found_boundary_colors):
        return False

    out = _semantic_output(grid)
    for region in after_regions:
        boundary_color = _region_boundary_color(completed, region)
        fill_color = legend[boundary_color]
        vals = {int(out[r, c]) for r, c in np.argwhere(region)}
        if vals != {fill_color}:
            return False

    if not np.array_equal(main(grid), out):
        return False

    values = Counter(map(int, out.flatten()))
    if sum(color in values for color in FILL_COLORS) == 0:
        return False
    return True


def _generate_one() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    region_count = random.choices(REGION_COUNT_CHOICES, weights=REGION_COUNT_WEIGHTS, k=1)[0]
    region_colors = [random.choice(LINE_COLORS) for _ in range(region_count)]

    remaining_colors = [color for color in LINE_COLORS if color not in set(region_colors)]
    extra_legend_colors: set[int] = set()
    if remaining_colors and random.random() < 0.7:
        extra_legend_colors.add(random.choice(remaining_colors))

    legend_keys = list(set(region_colors) | extra_legend_colors)
    if not legend_keys:
        return None
    random.shuffle(legend_keys)
    fill_values = random.sample(FILL_COLORS, k=len(legend_keys))
    legend_map = dict(zip(legend_keys, fill_values))
    _place_legend(grid, legend_keys, legend_map)

    rect_result = _try_place_rectangles(grid, region_colors, len(legend_keys))
    if rect_result is None:
        return None
    expected_boundary_colors, expected_gaps, boxes = rect_result

    distractor_colors = list(extra_legend_colors)
    if not distractor_colors:
        distractor_colors = [random.choice(legend_keys)]

    distractor_count = random.choice(DISTRACTOR_COUNT_CHOICES)
    boxes_for_distractors = boxes[:]
    for _ in range(distractor_count):
        color = random.choice(distractor_colors + legend_keys)
        result = _place_distractor_segment(grid, color, boxes_for_distractors)
        if result is None:
            continue
        gap, box = result
        expected_gaps |= gap
        boxes_for_distractors.append(box)

    if not expected_gaps:
        return None

    if not _validate_example(grid, expected_gaps, expected_boundary_colors, extra_legend_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
