import math
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERIOD_BY_COLOR = {
    BLUE: 3,
    RED: 2,
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_plus_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    centers: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == GREY):
        r = int(r)
        c = int(c)
        arms = [(r + dr, c + dc) for dr, dc in DIRS.values()]
        if not all(_in_bounds(g.shape, rr, cc) and g[rr, cc] == GREY for rr, cc in arms):
            continue
        valid = True
        for rr, cc in arms:
            grey_neighbors = 0
            for dr, dc in DIRS.values():
                nr, nc = rr + dr, cc + dc
                if _in_bounds(g.shape, nr, nc) and g[nr, nc] == GREY:
                    grey_neighbors += 1
            if grey_neighbors != 1:
                valid = False
                break
        if valid:
            centers.append((r, c))
    return sorted(centers)


def _extract_rays(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rays: list[dict] = []
    for center in _find_plus_centers(g):
        r, c = center
        for arm_name, (dr, dc) in DIRS.items():
            marker = (r + 2 * dr, c + 2 * dc)
            if not _in_bounds(g.shape, *marker):
                continue
            color = int(g[marker])
            if color not in PERIOD_BY_COLOR:
                continue
            rays.append(
                {
                    "center": center,
                    "arm": arm_name,
                    "marker": marker,
                    "direction": (dr, dc),
                    "color": color,
                    "period": PERIOD_BY_COLOR[color],
                }
            )
    return rays


def _planned_coverage(grid: np.ndarray, rays: list[dict] | None = None) -> dict[tuple[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    if rays is None:
        rays = _extract_rays(g)
    planned: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ray_id, ray in enumerate(rays):
        dr, dc = ray["direction"]
        rr = ray["marker"][0] + dr
        cc = ray["marker"][1] + dc
        while _in_bounds(g.shape, rr, cc):
            if g[rr, cc] == BLACK:
                planned[(rr, cc)].append(ray_id)
            rr += dr * ray["period"]
            cc += dc * ray["period"]
    return planned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend color-coded rays outward from gray '+' objects and recolor ray intersections yellow.

    Input structure:
    - The background is black.
    - One or more gray 5-pixel '+' objects appear as a center with four orthogonal arm pixels.
    - Some arm ends have one adjacent marker pixel outside the '+', colored red or blue.

    Rule:
    - For each marker, use the touched arm side to determine the ray direction away from the '+' center.
    - Decode the marker color as the ray period: red means draw every 2 cells, blue means draw every 3 cells.
    - Starting one step beyond the marker, draw one pixel in the marker color, then keep drawing every decoded period until the grid edge.
    - Original gray '+' pixels are never overwritten, and untouched background stays black.
    - Any position planned by two or more different rays is recolored yellow instead of any marker color.

    Key insight:
    - The marker color has a dual role: it is both the ray color and the visible code for that ray's spacing, so the final yellow cells only appear after decoding the spacing and checking overlaps between all rays.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rays = _extract_rays(g)
    planned = _planned_coverage(g, rays)
    for cell, ray_ids in planned.items():
        if len(ray_ids) >= 2:
            out[cell] = YELLOW
        else:
            ray = rays[ray_ids[0]]
            out[cell] = ray["color"]
    return out


def _place_plus(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r, c] = GREY
    for dr, dc in DIRS.values():
        grid[r + dr, c + dc] = GREY


def _ray_plot_cells(marker: tuple[int, int], direction: tuple[int, int], period: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    rr = marker[0] + dr
    cc = marker[1] + dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, rr, cc):
        cells.append((rr, cc))
        rr += dr * period
        cc += dc * period
    return cells


def _validate_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, GREY}):
        return False

    centers = _find_plus_centers(g)
    if len(centers) != 2:
        return False
    if int(np.sum(g == GREY)) != 10:
        return False

    rays = _extract_rays(g)
    if len(rays) < 4:
        return False

    per_plus = defaultdict(int)
    for ray in rays:
        per_plus[ray["center"]] += 1
    if any(per_plus[center] < 2 for center in centers):
        return False

    planned = _planned_coverage(g, rays)
    if not planned:
        return False

    hit_counts = CounterLike()
    unique_color_hits = {BLUE: 0, RED: 0}
    intersection_count = 0
    for cell, ray_ids in planned.items():
        if g[cell] != BLACK:
            return False
        if len(ray_ids) >= 2:
            intersection_count += 1
        else:
            color = rays[ray_ids[0]]["color"]
            unique_color_hits[color] += 1
        for ray_id in ray_ids:
            hit_counts.add(ray_id)

    if intersection_count == 0:
        return False
    if unique_color_hits[BLUE] == 0 or unique_color_hits[RED] == 0:
        return False
    if any(hit_counts.count(ray_id) == 0 for ray_id in range(len(rays))):
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if YELLOW not in set(map(int, np.unique(out))):
        return False
    if BLUE not in set(map(int, np.unique(out))) or RED not in set(map(int, np.unique(out))):
        return False
    return True


class CounterLike:
    def __init__(self) -> None:
        self._data: dict[int, int] = defaultdict(int)

    def add(self, key: int) -> None:
        self._data[int(key)] += 1

    def count(self, key: int) -> int:
        return self._data.get(int(key), 0)


def _random_color(rng: np.random.Generator) -> int:
    return int(rng.choice([BLUE, RED]))


def _mandatory_colors(rng: np.random.Generator) -> tuple[int, int]:
    if int(rng.integers(0, 2)) == 0:
        return RED, BLUE
    return BLUE, RED


def _candidate_extras(base_arms: set[str]) -> list[str]:
    return [name for name in ("up", "left", "right", "down") if name not in base_arms]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    center_margin = 4
    for _ in range(400):
        two_intersections = bool(rng.integers(0, 2))
        a_down_color, b_left_color = _mandatory_colors(rng)
        a_right_color = _random_color(rng)
        b_top_color = _random_color(rng)

        row_lcm = math.lcm(PERIOD_BY_COLOR[a_down_color], PERIOD_BY_COLOR[b_top_color] if two_intersections else 1)
        col_lcm = math.lcm(PERIOD_BY_COLOR[b_left_color], PERIOD_BY_COLOR[a_right_color] if two_intersections else 1)
        row_mult = int(rng.integers(1, 4))
        col_mult = int(rng.integers(1, 4))
        row_gap = row_lcm * row_mult
        col_gap = col_lcm * col_mult

        min_h = max(18, 2 * center_margin + 4 + row_gap)
        min_w = max(18, 2 * center_margin + 4 + col_gap)
        if min_h > 26 or min_w > 26:
            continue

        h = int(rng.integers(min_h, 27))
        w = int(rng.integers(min_w, 27))
        max_r1 = h - center_margin - 1 - (3 + row_gap)
        max_c1 = w - center_margin - 1 - (3 + col_gap)
        if max_r1 < center_margin or max_c1 < center_margin:
            continue

        r1 = int(rng.integers(center_margin, max_r1 + 1))
        c1 = int(rng.integers(center_margin, max_c1 + 1))
        r2 = r1 + 3 + row_gap
        c2 = c1 + 3 + col_gap

        grid = np.zeros((h, w), dtype=int)
        center_a = (r1, c1)
        center_b = (r2, c2)
        _place_plus(grid, center_a)
        _place_plus(grid, center_b)

        ray_colors: dict[tuple[tuple[int, int], str], int] = {
            (center_a, "down"): a_down_color,
            (center_b, "left"): b_left_color,
        }
        if two_intersections:
            ray_colors[(center_a, "right")] = a_right_color
            ray_colors[(center_b, "up")] = b_top_color

        a_base = {arm for (center, arm) in ray_colors if center == center_a}
        b_base = {arm for (center, arm) in ray_colors if center == center_b}
        a_extra_choices = _candidate_extras(a_base)
        b_extra_choices = _candidate_extras(b_base)

        a_need = max(0, 2 - len(a_base))
        b_need = max(0, 2 - len(b_base))
        a_extra_count = a_need + int(rng.integers(0, len(a_extra_choices) - a_need + 1))
        b_extra_count = b_need + int(rng.integers(0, len(b_extra_choices) - b_need + 1))
        if a_extra_count:
            for arm in rng.choice(a_extra_choices, size=a_extra_count, replace=False):
                ray_colors[(center_a, str(arm))] = _random_color(rng)
        if b_extra_count:
            for arm in rng.choice(b_extra_choices, size=b_extra_count, replace=False):
                ray_colors[(center_b, str(arm))] = _random_color(rng)

        valid_markers = True
        for (center, arm), color in ray_colors.items():
            dr, dc = DIRS[arm]
            marker = (center[0] + 2 * dr, center[1] + 2 * dc)
            if not _in_bounds(grid.shape, *marker):
                valid_markers = False
                break
            if grid[marker] != BLACK:
                valid_markers = False
                break
            grid[marker] = color
        if not valid_markers:
            continue

        # Reject layouts where a planned ray cell would hit an existing source object.
        blocked = False
        for (center, arm), color in ray_colors.items():
            marker = (center[0] + 2 * DIRS[arm][0], center[1] + 2 * DIRS[arm][1])
            period = PERIOD_BY_COLOR[color]
            for cell in _ray_plot_cells(marker, DIRS[arm], period, grid.shape):
                if grid[cell] != BLACK:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            continue

        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid ray-spacing puzzle input")
