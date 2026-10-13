import random

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = tuple(range(1, 9))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _find_scanline(g: np.ndarray) -> tuple[str, int]:
    row_hits = [int(r) for r in np.where(np.all(g == MAROON, axis=1))[0]]
    col_hits = [int(c) for c in np.where(np.all(g == MAROON, axis=0))[0]]
    assert len(row_hits) + len(col_hits) == 1, "expected one full maroon scanline"
    if row_hits:
        return "h", row_hits[0]
    return "v", col_hits[0]


def _foreground_colors(g: np.ndarray) -> list[int]:
    return [int(c) for c in np.unique(g) if int(c) not in (BLACK, MAROON)]


def _side_masks(coords: np.ndarray, orient: str, line: int) -> tuple[np.ndarray, np.ndarray]:
    axis_values = coords[:, 0] if orient == "h" else coords[:, 1]
    side_a = axis_values < line
    side_b = axis_values > line
    assert np.all(side_a | side_b), "foreground pixels may not lie on the scanline"
    return side_a, side_b


def _unique_neighbor_color(g: np.ndarray, color: int, coords: np.ndarray) -> int:
    h, w = g.shape
    adjacent: set[int] = set()
    for r, c in coords:
        r, c = int(r), int(c)
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v not in (BLACK, MAROON, int(color)):
                    adjacent.add(v)
    assert len(adjacent) == 1, "each colored region must have one adjacent region"
    return next(iter(adjacent))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut each colored region by the maroon scanline and recolor its larger side with its paired neighbor color.

    Input structure:
    - The background is black, and exactly one full-width or full-height maroon
      scanline crosses the grid.
    - Every other color appears as one region split into two 4-connected pieces on
      opposite sides of the scanline.
    - The two pieces of each color have unequal area, and each color has exactly
      one 4-adjacent different-color neighbor.

    Rule:
    - For each non-black, non-maroon color, compare its pixel counts on the two
      sides of the scanline.
    - Keep the larger side, delete the smaller side to black, and repaint the
      kept side with the original color of that region's unique adjacent neighbor.
    - Apply all deletions and recolorings simultaneously; keep the scanline
      unchanged.

    Key insight:
    The maroon line is a cutter, not a boundary label: size decides which half
    survives, while the visible 4-neighbor pairing decides the replacement color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    orient, line = _find_scanline(g)

    decisions: list[tuple[np.ndarray, np.ndarray, int]] = []
    for color in _foreground_colors(g):
        coords = np.argwhere(g == color)
        assert coords.size > 0
        assert len(_components(g == color)) == 2, "each color should be split into two pieces"

        side_a, side_b = _side_masks(coords, orient, line)
        area_a = int(np.sum(side_a))
        area_b = int(np.sum(side_b))
        assert area_a > 0 and area_b > 0, "each region must cross the scanline"
        assert area_a != area_b, "scanline sides must have unequal area"

        if area_a > area_b:
            keep, remove = coords[side_a], coords[side_b]
        else:
            keep, remove = coords[side_b], coords[side_a]
        neighbor = _unique_neighbor_color(g, color, coords)
        decisions.append((keep, remove, neighbor))

    out = g.copy()
    for _keep, remove, _neighbor in decisions:
        out[remove[:, 0], remove[:, 1]] = BLACK
    for keep, _remove, neighbor in decisions:
        out[keep[:, 0], keep[:, 1]] = neighbor
    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (18, 30)
PAIR_COUNT_RANGE = (2, 4)
LINE_MARGIN = 4
EDGE_GAP = 2
PAIR_GAP = 3
OBJ_WIDTH_OPTIONS = (1, 2, 2, 3)
PART_HEIGHT_MAX = 5
EXTENSION_PROB = 0.65
EXTENSION_MAX = 2
GEN_ATTEMPTS = 300


def _add_random_gap(extra: int, gaps: list[int]) -> None:
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1


def _pair_layout(width: int, n_pairs: int) -> list[tuple[int, int, int]] | None:
    widths = [(random.choice(OBJ_WIDTH_OPTIONS), random.choice(OBJ_WIDTH_OPTIONS)) for _ in range(n_pairs)]
    min_total = sum(a + b for a, b in widths) + 2 * EDGE_GAP + (n_pairs - 1) * PAIR_GAP
    if min_total > width:
        return None

    gaps = [EDGE_GAP] + [PAIR_GAP] * (n_pairs - 1) + [EDGE_GAP]
    _add_random_gap(width - min_total, gaps)

    layout: list[tuple[int, int, int]] = []
    c = gaps[0]
    for i, (left_w, right_w) in enumerate(widths):
        layout.append((c, left_w, right_w))
        c += left_w + right_w + gaps[i + 1]
    return layout


def _sample_heights(line: int, height: int) -> tuple[int, int] | None:
    top_max = min(PART_HEIGHT_MAX, line)
    bottom_max = min(PART_HEIGHT_MAX, height - line - 1)
    if top_max < 1 or bottom_max < 1:
        return None

    prefer_top = random.random() < 0.5
    options = [
        (top, bottom)
        for top in range(1, top_max + 1)
        for bottom in range(1, bottom_max + 1)
        if (top > bottom if prefer_top else bottom > top)
    ]
    if not options:
        options = [
            (top, bottom)
            for top in range(1, top_max + 1)
            for bottom in range(1, bottom_max + 1)
            if top != bottom
        ]
    return random.choice(options) if options else None


def _paint_part(grid: np.ndarray, color: int, rows: list[int], c0: int, c1: int, outward_dc: int) -> None:
    if not rows:
        return
    grid[rows, c0 : c1 + 1] = int(color)

    ext_col = c0 - 1 if outward_dc < 0 else c1 + 1
    if not (0 <= ext_col < grid.shape[1]) or random.random() >= EXTENSION_PROB:
        return

    count = random.randint(1, min(EXTENSION_MAX, len(rows)))
    for r in random.sample(rows, count):
        if int(grid[r, ext_col]) == BLACK:
            grid[r, ext_col] = int(color)


def _paint_object(
    grid: np.ndarray,
    color: int,
    line: int,
    c0: int,
    width: int,
    outward_dc: int,
) -> bool:
    heights = _sample_heights(line, grid.shape[0])
    if heights is None:
        return False
    top_h, bottom_h = heights
    c1 = c0 + width - 1
    top_rows = list(range(line - top_h, line))
    bottom_rows = list(range(line + 1, line + 1 + bottom_h))
    _paint_part(grid, color, top_rows, c0, c1, outward_dc)
    _paint_part(grid, color, bottom_rows, c0, c1, outward_dc)
    return True


def _render_horizontal_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    line = random.randint(LINE_MARGIN, height - LINE_MARGIN - 1)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    if 2 * n_pairs > len(COLOR_POOL):
        return None

    layout = _pair_layout(width, n_pairs)
    if layout is None:
        return None

    colors = random.sample(COLOR_POOL, 2 * n_pairs)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[line, :] = MAROON

    for pair_i, (start, left_w, right_w) in enumerate(layout):
        left_color = colors[2 * pair_i]
        right_color = colors[2 * pair_i + 1]
        if not _paint_object(grid, left_color, line, start, left_w, -1):
            return None
        if not _paint_object(grid, right_color, line, start + left_w, right_w, 1):
            return None

    return grid


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        out = main(grid)
        orient, line = _find_scanline(grid)
    except Exception:
        return False
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    colors = _foreground_colors(grid)
    if len(colors) < 4 or len(colors) % 2 != 0:
        return False

    counts = {int(c): int(np.sum(grid == c)) for c in np.unique(grid)}
    if counts.get(BLACK, 0) <= max(v for c, v in counts.items() if c != BLACK):
        return False

    neighbor_of: dict[int, int] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        side_a, side_b = _side_masks(coords, orient, line)
        if int(np.sum(side_a)) == int(np.sum(side_b)):
            return False
        try:
            neighbor_of[color] = _unique_neighbor_color(grid, color, coords)
        except AssertionError:
            return False

    return all(neighbor_of.get(neighbor) == color for color, neighbor in neighbor_of.items())


def _gen_attempt() -> np.ndarray | None:
    grid = _render_horizontal_attempt()
    if grid is None:
        return None
    if random.random() < 0.5:
        grid = grid.T.copy()
    if _generated_grid_is_valid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid scanline-pair input")
