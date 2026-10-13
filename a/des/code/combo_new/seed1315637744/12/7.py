from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

LEGEND_SEQUENCE = (RED, YELLOW, BLUE)
MAX_GENERATION_ATTEMPTS = 400
PAIR_RETRIES = 250
EXTRA_RECT_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _legend_candidates(h: int, w: int) -> list[list[tuple[int, int]]]:
    return [
        [(0, 0), (1, 0), (2, 0)],
        [(0, w - 1), (1, w - 1), (2, w - 1)],
        [(h - 3, 0), (h - 2, 0), (h - 1, 0)],
        [(h - 3, w - 1), (h - 2, w - 1), (h - 1, w - 1)],
    ]


def _find_legend(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    assert h >= 3 and w >= 1
    for cells in _legend_candidates(h, w):
        if tuple(int(grid[r, c]) for r, c in cells) == LEGEND_SEQUENCE:
            return cells
    raise AssertionError("expected a corner 3x1 legend with red/yellow/blue")


def _count_runs(values: np.ndarray, target: int) -> int:
    runs = 0
    in_run = False
    for value in map(int, values):
        if value == target:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return runs


def _pair_markers(markers: list[tuple[int, int]]) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    remaining = set(markers)
    pairs: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    by_row: dict[int, list[tuple[int, int]]] = {}
    for pt in markers:
        by_row.setdefault(pt[0], []).append(pt)
    for r in sorted(by_row):
        pts = sorted(by_row[r], key=lambda p: p[1])
        if len(pts) == 2:
            assert pts[0] in remaining and pts[1] in remaining, "ambiguous row pair reuse"
            pairs.append(("h", pts[0], pts[1]))
            remaining.remove(pts[0])
            remaining.remove(pts[1])
        else:
            assert len(pts) in (0, 1), "row markers must form a unique pair"

    by_col: dict[int, list[tuple[int, int]]] = {}
    for pt in markers:
        by_col.setdefault(pt[1], []).append(pt)
    for c in sorted(by_col):
        pts = sorted([pt for pt in by_col[c] if pt in remaining], key=lambda p: p[0])
        if len(pts) == 2:
            pairs.append(("v", pts[0], pts[1]))
            remaining.remove(pts[0])
            remaining.remove(pts[1])
        else:
            assert len(pts) in (0, 1), "column markers must form a unique pair"

    assert not remaining, "every non-legend red marker must belong to exactly one aligned pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each aligned red marker pair with a line color chosen by the number of obstacle runs between them.

    Input structure:
    - The background is black.
    - A corner contains a vertical 3x1 legend whose cells are, from top to bottom, red, yellow, and blue.
    - Away from the legend there are several red single-pixel markers, and each marker belongs to a unique horizontal or vertical pair.
    - Solid green rectangles act as obstacles and may lie between paired markers along their shared row or column.

    Rule:
    - Ignore the legend red pixel and pair the remaining red markers by unique shared row or unique shared column.
    - For each pair, inspect only the cells strictly between the two markers along that row or column and count how many separate green runs appear there.
    - Draw a straight connection using the legend colors: red for 0 runs, yellow for 1 run, blue for 2 or more runs.
    - Fill only black path cells; keep red markers red and leave green obstacle cells untouched.

    Key insight:
    - The deciding quantity is the number of independent green segments on the path, not the total number of green pixels or the marker distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    legend_cells = _find_legend(g)
    legend_values = [int(g[r, c]) for r, c in legend_cells]
    marker_color, one_obstacle_color, multi_obstacle_color = legend_values

    palette = set(map(int, np.unique(g)))
    obstacle_candidates = palette - {bg, marker_color, one_obstacle_color, multi_obstacle_color}
    assert len(obstacle_candidates) == 1, "expected exactly one obstacle color distinct from the legend"
    obstacle_color = next(iter(obstacle_candidates))

    legend_set = set(legend_cells)
    markers = [
        (int(r), int(c))
        for r, c in np.argwhere(g == marker_color)
        if (int(r), int(c)) not in legend_set
    ]
    pairs = _pair_markers(markers)

    out = g.copy()
    for orient, a, b in pairs:
        if orient == "h":
            r = a[0]
            c0, c1 = sorted((a[1], b[1]))
            obstacle_runs = _count_runs(g[r, c0 + 1 : c1], obstacle_color)
            line_color = marker_color if obstacle_runs == 0 else one_obstacle_color if obstacle_runs == 1 else multi_obstacle_color
            for c in range(c0 + 1, c1):
                if int(out[r, c]) == bg:
                    out[r, c] = line_color
        else:
            c = a[1]
            r0, r1 = sorted((a[0], b[0]))
            obstacle_runs = _count_runs(g[r0 + 1 : r1, c], obstacle_color)
            line_color = marker_color if obstacle_runs == 0 else one_obstacle_color if obstacle_runs == 1 else multi_obstacle_color
            for r in range(r0 + 1, r1):
                if int(out[r, c]) == bg:
                    out[r, c] = line_color
    return out


def _expanded_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _sample_obstacle_layout(segment_count: int) -> tuple[list[tuple[int, int]], int]:
    if segment_count == 0:
        interior_len = random.randint(4, 10)
        return [], interior_len

    widths = [random.randint(1, 4) for _ in range(segment_count)]
    gaps = [random.randint(1, 3) for _ in range(segment_count + 1)]
    extra = random.randint(0, 4)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    intervals: list[tuple[int, int]] = []
    cursor = gaps[0]
    for idx, width in enumerate(widths):
        intervals.append((cursor, cursor + width - 1))
        cursor += width + gaps[idx + 1]
    interior_len = cursor
    return intervals, interior_len


def _segment_count_for_category(category: int) -> int:
    if category == 0:
        return 0
    if category == 1:
        return 1
    return random.randint(2, 3)


def _legend_cells_for_corner(h: int, w: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, 0), (1, 0), (2, 0)]
    if corner == "tr":
        return [(0, w - 1), (1, w - 1), (2, w - 1)]
    if corner == "bl":
        return [(h - 3, 0), (h - 2, 0), (h - 1, 0)]
    return [(h - 3, w - 1), (h - 2, w - 1), (h - 1, w - 1)]


def _reserve_legend(grid: np.ndarray, occupied: np.ndarray, blocked: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    cells = _legend_cells_for_corner(h, w, corner)
    for (r, c), color in zip(cells, LEGEND_SEQUENCE):
        grid[r, c] = color
        occupied[r, c] = True

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = max(0, min(rows) - 1)
    r1 = min(h - 1, max(rows) + 1)
    c0 = max(0, min(cols) - 1)
    c1 = min(w - 1, max(cols) + 1)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_horizontal_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
    marker_rows: set[int],
    marker_cols: set[int],
    segment_count: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    intervals, interior_len = _sample_obstacle_layout(segment_count)
    total_len = interior_len + 2
    if total_len > w:
        return None

    for _ in range(PAIR_RETRIES):
        r = random.randint(0, h - 1)
        if r in marker_rows:
            continue

        c0 = random.randint(0, w - total_len)
        c1 = c0 + total_len - 1
        if c0 in marker_cols or c1 in marker_cols:
            continue
        if not _expanded_clear(blocked | occupied | reserved, r, c0, r, c1, pad=0):
            continue

        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for rel_start, rel_end in intervals:
            seg_c0 = c0 + 1 + rel_start
            seg_c1 = c0 + 1 + rel_end
            height = random.randint(1, 4)
            top_min = max(0, r - height + 1)
            top_max = min(r, h - height)
            if top_min > top_max:
                ok = False
                break
            tops = list(range(top_min, top_max + 1))
            random.shuffle(tops)
            placed_rect = None
            for top in tops:
                bottom = top + height - 1
                if not _expanded_clear(blocked | occupied | reserved, top, seg_c0, bottom, seg_c1, pad=1):
                    continue
                placed_rect = (top, seg_c0, bottom, seg_c1)
                break
            if placed_rect is None:
                ok = False
                break
            rects.append(placed_rect)

        if not ok:
            continue

        grid[r, c0] = RED
        grid[r, c1] = RED
        occupied[r, c0] = True
        occupied[r, c1] = True
        reserved[r, c0 : c1 + 1] = True
        marker_rows.add(r)
        marker_cols.add(c0)
        marker_cols.add(c1)
        for rr0, cc0, rr1, cc1 in rects:
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = GREEN
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

        return {
            "orientation": "h",
            "a": (r, c0),
            "b": (r, c1),
            "segment_count": segment_count,
        }
    return None


def _place_vertical_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
    marker_rows: set[int],
    marker_cols: set[int],
    segment_count: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    intervals, interior_len = _sample_obstacle_layout(segment_count)
    total_len = interior_len + 2
    if total_len > h:
        return None

    for _ in range(PAIR_RETRIES):
        c = random.randint(0, w - 1)
        if c in marker_cols:
            continue

        r0 = random.randint(0, h - total_len)
        r1 = r0 + total_len - 1
        if r0 in marker_rows or r1 in marker_rows:
            continue
        if not _expanded_clear(blocked | occupied | reserved, r0, c, r1, c, pad=0):
            continue

        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for rel_start, rel_end in intervals:
            seg_r0 = r0 + 1 + rel_start
            seg_r1 = r0 + 1 + rel_end
            width = random.randint(1, 4)
            left_min = max(0, c - width + 1)
            left_max = min(c, w - width)
            if left_min > left_max:
                ok = False
                break
            lefts = list(range(left_min, left_max + 1))
            random.shuffle(lefts)
            placed_rect = None
            for left in lefts:
                right = left + width - 1
                if not _expanded_clear(blocked | occupied | reserved, seg_r0, left, seg_r1, right, pad=1):
                    continue
                placed_rect = (seg_r0, left, seg_r1, right)
                break
            if placed_rect is None:
                ok = False
                break
            rects.append(placed_rect)

        if not ok:
            continue

        grid[r0, c] = RED
        grid[r1, c] = RED
        occupied[r0, c] = True
        occupied[r1, c] = True
        reserved[r0 : r1 + 1, c] = True
        marker_cols.add(c)
        marker_rows.add(r0)
        marker_rows.add(r1)
        for rr0, cc0, rr1, cc1 in rects:
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = GREEN
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

        return {
            "orientation": "v",
            "a": (r0, c),
            "b": (r1, c),
            "segment_count": segment_count,
        }
    return None


def _place_extra_obstacle(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
) -> bool:
    h, w = grid.shape
    for _ in range(EXTRA_RECT_RETRIES):
        height = random.randint(1, 4)
        width = random.randint(1, 4)
        r0 = random.randint(0, h - height)
        c0 = random.randint(0, w - width)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        if not _expanded_clear(blocked | occupied | reserved, r0, c0, r1, c1, pad=1):
            continue
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREEN
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _expected_output(grid: np.ndarray, pair_specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in pair_specs:
        segment_count = int(spec["segment_count"])
        color = RED if segment_count == 0 else YELLOW if segment_count == 1 else BLUE
        a = spec["a"]
        b = spec["b"]
        if spec["orientation"] == "h":
            r = int(a[0])
            c0, c1 = sorted((int(a[1]), int(b[1])))
            for c in range(c0 + 1, c1):
                if int(out[r, c]) == BG:
                    out[r, c] = color
        else:
            c = int(a[1])
            r0, r1 = sorted((int(a[0]), int(b[0])))
            for r in range(r0 + 1, r1):
                if int(out[r, c]) == BG:
                    out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)

    _reserve_legend(grid, occupied, blocked, random.choice(["tl", "tr", "bl", "br"]))

    categories = [0, 1, 2]
    if random.random() < 0.5:
        categories.append(random.choice([0, 1, 2]))
    random.shuffle(categories)

    orientations = ["h", "v"]
    while len(orientations) < len(categories):
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)

    marker_rows: set[int] = set()
    marker_cols: set[int] = set()
    pair_specs: list[dict[str, object]] = []

    for category, orientation in zip(categories, orientations):
        segment_count = _segment_count_for_category(category)
        if orientation == "h":
            spec = _place_horizontal_pair(grid, occupied, blocked, reserved, marker_rows, marker_cols, segment_count)
        else:
            spec = _place_vertical_pair(grid, occupied, blocked, reserved, marker_rows, marker_cols, segment_count)
        if spec is None:
            return None
        pair_specs.append(spec)

    extra_obstacles = random.randint(0, 2)
    for _ in range(extra_obstacles):
        _place_extra_obstacle(grid, occupied, blocked, reserved)

    expected = _expected_output(grid, pair_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
