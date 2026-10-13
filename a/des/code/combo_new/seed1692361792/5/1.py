from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
WALL_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _mode_color(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    return Counter(flat).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _infer_background(grid: np.ndarray) -> int:
    return BACKGROUND if np.any(grid == BACKGROUND) else _mode_color(grid)


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    candidates = [int(color) for color in np.unique(grid) if int(color) != bg]
    best_color = candidates[0]
    best_score = (-1, -1, -1)
    for color in candidates:
        rect_count = 0
        total_rect_area = 0
        for comp in _components(grid != color):
            if _touches_boundary(comp, *grid.shape):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(comp) == area:
                rect_count += 1
                total_rect_area += area
        score = (total_rect_area, rect_count, int(np.count_nonzero(grid == color)))
        if score > best_score:
            best_score = score
            best_color = color
    return best_color


def _tile_template(height: int, width: int, template: np.ndarray) -> np.ndarray:
    rows = np.arange(height)[:, None] % 2
    cols = np.arange(width)[None, :] % 2
    return template[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle by looking up its seed color in the right-side legend of repeating patterns.

    Input structure:
    - The background is black, and grey line segments form several closed rectangular frames.
    - Each non-legend frame encloses a full rectangular interior containing exactly one non-black seed pixel.
    - A legend strip sits on the right side as stacked grey-framed cells; each cell has a marker pixel at its interior top-left and a repeating two-color 2x2 pattern in the rest of the cell.

    Rule:
    - Treat each grey-enclosed interior rectangle as one component separated from the outside background.
    - If such a component contains exactly one non-background pixel, that pixel's color is the seed key for a target region.
    - Otherwise it is a legend cell: read its top-left marker color, extract the 2x2 block immediately to the right of the marker, and store that template.
    - Replace every target region interior with the matching template tiled from that region's own interior top-left corner, overwriting the seed pixel.

    Key insight:
    - The single seed pixel inside a region is only a lookup key; the actual output content comes from the legend cell whose marker has the same color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _infer_background(g)
    wall = _infer_wall_color(g, bg)
    h, w = g.shape

    legend_templates: dict[int, np.ndarray] = {}
    regions: list[tuple[int, int, int, int, int]] = []

    for comp in _components(g != wall):
        if _touches_boundary(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(comp) != area:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        non_bg_positions = np.argwhere(sub != bg)
        if len(non_bg_positions) == 1:
            sr, sc = map(int, non_bg_positions[0])
            regions.append((r0, c0, r1, c1, int(sub[sr, sc])))
        else:
            if sub.shape[0] < 2 or sub.shape[1] < 3:
                continue
            marker = int(sub[0, 0])
            legend_templates[marker] = np.array(sub[:2, 1:3], dtype=int)

    out = g.copy()
    for r0, c0, r1, c1, seed_color in regions:
        template = legend_templates[seed_color]
        out[r0 : r1 + 1, c0 : c1 + 1] = _tile_template(r1 - r0 + 1, c1 - c0 + 1, template)
    return out


def _make_template(rng: random.Random, marker: int) -> np.ndarray:
    color_a, color_b = rng.sample([c for c in COLOR_POOL if c != marker], 2)
    bases = [
        np.array([[color_a, color_b], [color_b, color_a]], dtype=int),
        np.array([[color_a, color_b], [color_a, color_b]], dtype=int),
        np.array([[color_a, color_a], [color_b, color_b]], dtype=int),
        np.array([[color_a, color_b], [color_b, color_b]], dtype=int),
        np.array([[color_a, color_a], [color_a, color_b]], dtype=int),
    ]
    template = bases[rng.randrange(len(bases))].copy()
    for _ in range(rng.randrange(4)):
        template = np.rot90(template)
    if rng.random() < 0.5:
        template = np.fliplr(template)
    return np.array(template, dtype=int)


def _paint_legend_cell(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    marker: int,
    template: np.ndarray,
) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = WALL_COLOR
    grid[bottom, left : right + 1] = WALL_COLOR
    grid[top : bottom + 1, left] = WALL_COLOR
    grid[top : bottom + 1, right] = WALL_COLOR
    inner_top = top + 1
    inner_left = left + 1
    for rr in range(inner_h):
        for cc in range(inner_w):
            if rr == 0 and cc == 0:
                grid[inner_top + rr, inner_left + cc] = marker
            else:
                grid[inner_top + rr, inner_left + cc] = int(template[rr % 2, (cc - 1) % 2])


def _try_place_regions(
    rng: random.Random,
    height: int,
    width: int,
    region_colors: list[int],
) -> list[dict[str, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    specs = []
    max_inner_h = min(6, height - 2)
    max_inner_w = min(6, width - 2)
    if max_inner_h < 3 or max_inner_w < 3:
        return None

    for color in region_colors:
        inner_h = rng.randint(3, max_inner_h)
        inner_w = rng.randint(3, max_inner_w)
        specs.append({"color": color, "inner_h": inner_h, "inner_w": inner_w, "area": inner_h * inner_w})
    specs.sort(key=lambda spec: spec["area"], reverse=True)

    placements: list[dict[str, int]] = []
    for spec in specs:
        outer_h = spec["inner_h"] + 2
        outer_w = spec["inner_w"] + 2
        placed = False
        for _ in range(250):
            top = rng.randint(0, height - outer_h)
            left = rng.randint(0, width - outer_w)
            rs = max(0, top - 1)
            re = min(height, top + outer_h + 1)
            cs = max(0, left - 1)
            ce = min(width, left + outer_w + 1)
            if occupied[rs:re, cs:ce].any():
                continue
            occupied[top : top + outer_h, left : left + outer_w] = True
            placements.append(
                {
                    "color": spec["color"],
                    "top": top,
                    "left": left,
                    "inner_h": spec["inner_h"],
                    "inner_w": spec["inner_w"],
                }
            )
            placed = True
            break
        if not placed:
            return None
    return placements


def _draw_region(grid: np.ndarray, placement: dict[str, int], rng: random.Random) -> None:
    top = placement["top"]
    left = placement["left"]
    inner_h = placement["inner_h"]
    inner_w = placement["inner_w"]
    color = placement["color"]
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = WALL_COLOR
    grid[bottom, left : right + 1] = WALL_COLOR
    grid[top : bottom + 1, left] = WALL_COLOR
    grid[top : bottom + 1, right] = WALL_COLOR
    seed_r = rng.randint(top + 1, bottom - 1)
    seed_c = rng.randint(left + 1, right - 1)
    grid[seed_r, seed_c] = color


def _expected_output(
    grid: np.ndarray,
    placements: list[dict[str, int]],
    templates: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        top = placement["top"] + 1
        left = placement["left"] + 1
        inner_h = placement["inner_h"]
        inner_w = placement["inner_w"]
        color = placement["color"]
        out[top : top + inner_h, left : left + inner_w] = _tile_template(inner_h, inner_w, templates[color])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_keys = rng.randint(2, 5)
    legend_colors = rng.sample(COLOR_POOL, n_keys)
    templates = {color: _make_template(rng, color) for color in legend_colors}

    inner_legend_h_max = min(6, (30 - (n_keys + 1)) // n_keys)
    if inner_legend_h_max < 4:
        return None
    legend_inner_h = rng.randint(4, inner_legend_h_max)
    legend_inner_w = rng.randint(5, 6)
    legend_outer_h = n_keys * legend_inner_h + (n_keys + 1)
    legend_outer_w = legend_inner_w + 2

    gap = rng.randint(1, 3)
    left_zone_w_max = 30 - gap - legend_outer_w
    if left_zone_w_max < 12:
        return None
    left_zone_w = rng.randint(12, min(22, left_zone_w_max))
    width = left_zone_w + gap + legend_outer_w

    min_height = max(legend_outer_h, 16)
    if min_height > 30:
        return None
    height = rng.randint(min_height, 30)

    capacity = min(6, max(n_keys, (height * left_zone_w) // 50))
    if capacity < n_keys:
        return None
    n_regions = rng.randint(n_keys, capacity)
    region_colors = legend_colors[:] + [rng.choice(legend_colors) for _ in range(n_regions - n_keys)]
    rng.shuffle(region_colors)

    placements = _try_place_regions(rng, height, left_zone_w, region_colors)
    if placements is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for placement in placements:
        _draw_region(grid, placement, rng)

    legend_top = rng.randint(0, height - legend_outer_h)
    legend_left = width - legend_outer_w
    for idx, color in enumerate(legend_colors):
        cell_top = legend_top + idx * (legend_inner_h + 1)
        _paint_legend_cell(grid, cell_top, legend_left, legend_inner_h, legend_inner_w, color, templates[color])

    expected = _expected_output(grid, placements, templates)
    if np.array_equal(grid, expected):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
