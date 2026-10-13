import random
from collections import Counter, deque

import numpy as np


BG = 0
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _neighbors4(r, c):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _polyline(points):
    cells = []
    for (r1, c1), (r2, c2) in zip(points, points[1:]):
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            seg = [(r1, cc) for cc in range(c1, c2 + step, step)]
        elif c1 == c2:
            step = 1 if r2 > r1 else -1
            seg = [(rr, c1) for rr in range(r1, r2 + step, step)]
        else:
            raise ValueError("polyline segments must be axis-aligned")
        if cells:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _top_u(h, left, right, bottom):
    return _polyline([(0, left), (bottom, left), (bottom, right), (0, right)])


def _bottom_u(h, left, right, top):
    return _polyline([(h - 1, left), (top, left), (top, right), (h - 1, right)])


def _left_c(top, bottom, inner_col):
    return _polyline([(top, 0), (top, inner_col), (bottom, inner_col), (bottom, 0)])


def _right_c(w, top, bottom, inner_col):
    return _polyline([(top, w - 1), (top, inner_col), (bottom, inner_col), (bottom, w - 1)])


def _stamp_paths(grid, paths):
    for color, cells in paths:
        for r, c in cells:
            grid[r, c] = color


def _analyze_regions(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    seen = np.zeros((h, w), dtype=bool)
    regions = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            touches_border = False
            boundary_colors = set()
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    val = int(g[nr, nc])
                    if val == BG:
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                    elif (nr, nc) not in corners:
                        boundary_colors.add(val)
            regions.append(
                {
                    "pts": pts,
                    "touches_border": touches_border,
                    "boundary_colors": boundary_colors,
                }
            )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed regions according to how many different path colors surround them.

    Input structure:
    - The background is black.
    - Several thin colored rectilinear paths divide the grid into open and closed
      black regions.
    - The four corners contain singleton marker pixels in two diagonal-matching
      colors; these corner colors do not belong to the paths.

    Rule:
    - Keep the colored path pixels unchanged.
    - Find every 4-connected black region. If a region touches the grid edge, keep
      it black.
    - For each closed black region, count how many distinct path colors appear on
      its boundary.
    - A closed region bordered by an odd number of path colors is filled with the
      top-left corner color. A closed region bordered by an even number of path
      colors (at least two) is filled with the top-right corner color.
    - Closed regions bordered by only one path color stay black.

    Key insight:
    - The output depends on the color diversity of each enclosed region's boundary,
      not on geometric area or nesting depth, and the corner markers encode the two
      fill colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    odd_fill = int(g[0, 0])
    even_fill = int(g[0, w - 1])
    out = g.copy()

    for region in _analyze_regions(g):
        if region["touches_border"]:
            continue
        n_colors = len(region["boundary_colors"])
        if n_colors < 2:
            continue
        fill = odd_fill if n_colors % 2 == 1 else even_fill
        for r, c in region["pts"]:
            out[r, c] = fill

    return out


GRID_MIN = 18
GRID_MAX = 30
GEN_RETRIES = 400


def _random_layout(h, w, mode, rng):
    outer_left = rng.randint(2, max(2, w // 6))
    outer_right_margin = rng.randint(2, max(2, w // 6))
    left0 = outer_left
    right0 = w - 1 - outer_right_margin
    if right0 - left0 < 10:
        return None

    bottom0 = rng.randint(h // 2, h - 4)

    inner_gap_left = rng.randint(3, max(3, (right0 - left0) // 4))
    inner_gap_right = rng.randint(3, max(3, (right0 - left0) // 4))
    left1 = left0 + inner_gap_left
    right1 = right0 - inner_gap_right
    if right1 - left1 < 5:
        return None

    top1 = rng.randint(3, bottom0 - 5)

    y_top = rng.randint(top1 + 1, max(top1 + 1, (top1 + bottom0) // 2))
    y_bottom = rng.randint(y_top + 2, bottom0 - 1)

    x_left = rng.randint(left1 + 1, max(left1 + 1, right1 - 2))
    x_right = rng.randint(max(x_left + 1, left1 + 2), right1 - 1)

    return {
        "outer": (left0, right0, bottom0),
        "inner": (left1, right1, top1),
        "left_cut": (y_top, y_bottom, x_left),
        "right_cut": (y_top, y_bottom, x_right),
    }


def _make_grid(h, w, mode, rng):
    layout = _random_layout(h, w, mode, rng)
    if layout is None:
        return None

    n_paths = 2 if mode == "pair" else 3 if mode.startswith("triple") else 4
    palette = rng.sample(range(1, 10), n_paths + 2)
    odd_marker = palette[0]
    even_marker = palette[1]
    path_colors = palette[2:]

    grid = np.zeros((h, w), dtype=int)
    left0, right0, bottom0 = layout["outer"]
    left1, right1, top1 = layout["inner"]
    y_top, y_bottom, x_left = layout["left_cut"]
    _, _, x_right = layout["right_cut"]

    paths = [
        (path_colors[0], _top_u(h, left0, right0, bottom0)),
        (path_colors[1], _bottom_u(h, left1, right1, top1)),
    ]
    if mode == "triple_left":
        paths.append((path_colors[2], _left_c(y_top, y_bottom, x_left)))
    elif mode == "triple_right":
        paths.append((path_colors[2], _right_c(w, y_top, y_bottom, x_right)))
    elif mode == "quad":
        paths.append((path_colors[2], _left_c(y_top, y_bottom, x_left)))
        paths.append((path_colors[3], _right_c(w, y_top, y_bottom, x_right)))

    _stamp_paths(grid, paths)

    grid[0, 0] = odd_marker
    grid[0, w - 1] = even_marker
    grid[h - 1, 0] = even_marker
    grid[h - 1, w - 1] = odd_marker

    k = rng.randrange(4)
    if k:
        grid = np.rot90(grid, k)

    return grid


def _valid_signature(grid, mode):
    counts = Counter(int(v) for v in grid.ravel())
    if counts.get(BG, 0) != max(counts.values()):
        return False

    regions = _analyze_regions(grid)
    closed = [len(reg["boundary_colors"]) for reg in regions if not reg["touches_border"]]
    if not closed or not any(n >= 2 for n in closed):
        return False

    has_even = any(n >= 2 and n % 2 == 0 for n in closed)
    has_odd = any(n >= 3 and n % 2 == 1 for n in closed)

    if mode == "pair":
        return has_even
    return has_even and has_odd


def _gen_attempt(rng):
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    mode = rng.choice(["pair", "triple_left", "triple_right", "quad"])

    grid = _make_grid(h, w, mode, rng)
    if grid is None:
        return None
    if not _valid_signature(grid, mode):
        return None

    out = main(grid)
    _, gw = grid.shape
    odd_fill = int(grid[0, 0])
    even_fill = int(grid[0, gw - 1])
    odd_added = int(((out == odd_fill) & (grid == BG)).sum())
    even_added = int(((out == even_fill) & (grid == BG)).sum())

    if mode == "pair":
        if even_added == 0:
            return None
    else:
        if odd_added == 0 or even_added == 0:
            return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
