from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)

    return comps


def _enclosed_regions_for_wall(grid: np.ndarray, wall_color: int) -> list[BBox]:
    h, w = grid.shape
    regions: list[BBox] = []

    for pts in _components(grid != int(wall_color)):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if len(pts) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if not np.all(grid[r0 - 1, c0 : c1 + 1] == int(wall_color)):
            continue
        if not np.all(grid[r1 + 1, c0 : c1 + 1] == int(wall_color)):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 - 1] == int(wall_color)):
            continue
        if not np.all(grid[r0 : r1 + 1, c1 + 1] == int(wall_color)):
            continue

        regions.append((r0, c0, r1, c1))

    regions.sort()
    return regions


def _choose_wall_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = None

    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue

        regions = _enclosed_regions_for_wall(grid, color)
        total_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in regions)
        score = (len(regions), total_area, int(np.count_nonzero(grid == color)))

        if best_score is None or score > best_score:
            best_score = score
            best_color = color

    assert best_color is not None
    return int(best_color)


def _marker_gap(r: int, c: int, region: BBox) -> tuple[int, int] | None:
    r0, c0, r1, c1 = region
    top_wall = r0 - 1
    bottom_wall = r1 + 1
    left_wall = c0 - 1
    right_wall = c1 + 1

    distances = {
        "top": r - top_wall,
        "bottom": bottom_wall - r,
        "left": c - left_wall,
        "right": right_wall - c,
    }
    min_dist = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == min_dist]
    if len(sides) != 1:
        return None

    side = sides[0]
    if side == "top":
        return top_wall, c
    if side == "bottom":
        return bottom_wall, c
    if side == "left":
        return r, left_wall
    return r, right_wall


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the rectangular regions separated by a one-pixel wall network.
    Erase each marker pixel and open a wall gap at the marker's projected
    position on its unique nearest side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    wall = _choose_wall_color(g, bg)
    regions = _enclosed_regions_for_wall(g, wall)

    out = g.copy()
    for region in regions:
        r0, c0, r1, c1 = region
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        markers = np.argwhere((sub != int(bg)) & (sub != int(wall)))

        for dr, dc in markers:
            r = int(r0 + dr)
            c = int(c0 + dc)
            out[r, c] = int(bg)
            gap = _marker_gap(r, c, region)
            assert gap is not None, "marker must have a unique nearest wall"
            out[gap] = int(bg)

    return out


# --------------- generator ---------------

COLOR_IDS = tuple(range(10))
ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
REGION_HEIGHT_RANGE = (4, 8)
REGION_WIDTH_RANGE = (4, 8)
PADDING_RANGE = (0, 2)
ZERO_MARKER_PROB = 0.35
MAX_MARKERS_PER_REGION = 4
GEN_ATTEMPTS = 400


def _wall_positions(start: int, sizes: list[int]) -> list[int]:
    pos = int(start)
    walls = [pos]
    for size in sizes:
        pos += int(size) + 1
        walls.append(pos)
    return walls


def _region_boxes(wall_rows: list[int], wall_cols: list[int]) -> list[BBox]:
    boxes: list[BBox] = []
    for i in range(len(wall_rows) - 1):
        for j in range(len(wall_cols) - 1):
            boxes.append(
                (
                    wall_rows[i] + 1,
                    wall_cols[j] + 1,
                    wall_rows[i + 1] - 1,
                    wall_cols[j + 1] - 1,
                )
            )
    return boxes


def _draw_wall_grid(grid: np.ndarray, wall_color: int, wall_rows: list[int], wall_cols: list[int]) -> None:
    top, bottom = wall_rows[0], wall_rows[-1]
    left, right = wall_cols[0], wall_cols[-1]
    for r in wall_rows:
        grid[r, left : right + 1] = int(wall_color)
    for c in wall_cols:
        grid[top : bottom + 1, c] = int(wall_color)


def _region_gap_options(region: BBox, used_gaps: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = region
    options: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            gap = _marker_gap(r, c, region)
            if gap is None or gap in used_gaps:
                continue
            options.setdefault(gap, []).append((r, c))

    return options


def _sample_marker_count(cap: int) -> int:
    if cap <= 0 or random.random() < ZERO_MARKER_PROB:
        return 0
    return random.randint(1, min(int(cap), MAX_MARKERS_PER_REGION))


def _gen_attempt() -> np.ndarray | None:
    bg, wall = random.sample(COLOR_IDS, 2)
    region_rows = random.randint(*ROW_COUNT_RANGE)
    region_cols = random.randint(*COL_COUNT_RANGE)
    region_heights = [random.randint(*REGION_HEIGHT_RANGE) for _ in range(region_rows)]
    region_widths = [random.randint(*REGION_WIDTH_RANGE) for _ in range(region_cols)]

    pad_top = random.randint(*PADDING_RANGE)
    pad_bottom = random.randint(*PADDING_RANGE)
    pad_left = random.randint(*PADDING_RANGE)
    pad_right = random.randint(*PADDING_RANGE)

    wall_rows = _wall_positions(pad_top, region_heights)
    wall_cols = _wall_positions(pad_left, region_widths)
    height = wall_rows[-1] + 1 + pad_bottom
    width = wall_cols[-1] + 1 + pad_right

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()
    _draw_wall_grid(inp, wall, wall_rows, wall_cols)
    _draw_wall_grid(expected, wall, wall_rows, wall_cols)

    marker_colors = [color for color in COLOR_IDS if color not in (bg, wall)]
    regions = _region_boxes(wall_rows, wall_cols)
    random.shuffle(regions)

    used_gaps: set[tuple[int, int]] = set()
    total_markers = 0
    zero_regions = 0

    for region in regions:
        gap_options = _region_gap_options(region, used_gaps)
        marker_count = _sample_marker_count(len(gap_options))
        if marker_count == 0:
            zero_regions += 1
            continue

        total_markers += marker_count
        color = random.choice(marker_colors)
        for gap in random.sample(list(gap_options), marker_count):
            r, c = random.choice(gap_options[gap])
            inp[r, c] = color
            expected[gap] = bg
            used_gaps.add(gap)

    if total_markers == 0:
        return None
    if len(regions) > 1 and zero_regions == 0 and random.random() < 0.7:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-gap puzzle")
