import numpy as np

from arc_utils import bbox, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the foreground shape by mirroring it across the 2x2 marker square.

    Input structure
    - A dominant background, one source foreground color, and one marker color.
    - The marker color appears exactly as a solid 2x2 square.
    - Source pixels lie in one or more quadrants around that marker square.

    Rule
    - Use the center of the 2x2 marker square as both a horizontal and vertical
      mirror axis.
    - For every source pixel, paint its vertical reflection, horizontal
      reflection, and 180-degree reflection in the same source color.

    Key insight
    - The green 2x2 square is not copied; it defines the mirror center for a
      four-way reflection closure of the red mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    marker_candidates = []
    for color in colors:
        mask = g == color
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        if (
            int(mask.sum()) == 4
            and r1 - r0 + 1 == 2
            and c1 - c0 + 1 == 2
            and np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
        ):
            marker_candidates.append((color, b))

    assert len(marker_candidates) == 1
    marker_color, marker_box = marker_candidates[0]
    source_colors = [color for color in colors if color != marker_color]
    assert len(source_colors) == 1
    source_color = source_colors[0]

    r0, c0, r1, c1 = marker_box
    out = g.copy()
    height, width = g.shape
    for r, c in np.argwhere(g == source_color):
        r = int(r)
        c = int(c)
        for rr in (r, r0 + r1 - r):
            for cc in (c, c0 + c1 - c):
                assert 0 <= rr < height and 0 <= cc < width
                assert g[rr, cc] != marker_color
                out[rr, cc] = source_color

    return out

# --------------- generator (o5x) ---------------

# Palette.
_GEN_BG_COLOR = 0
_GEN_SOURCE_COLOR = 2
_GEN_MARKER_COLOR = 3
# Canvas and marker bounds.
_GEN_SOURCE_SIZES = (3, 4, 5)
_GEN_MIN_GRID_SIDE = 10
_GEN_MAX_HEIGHT = 20
_GEN_MAX_WIDTH = 30
_GEN_MARKER_SIDE = 2
# Source shape sampling.
_GEN_MIN_SOURCE_PIXELS = 6
_GEN_MAX_SOURCE_PIXELS = 10
_GEN_MIN_SOURCE_DENSITY = 0.40
_GEN_MAX_SOURCE_DENSITY = 0.70
_GEN_ISLAND_PROBABILITY = 0.35
_GEN_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
# Retry budget.
_GEN_MAX_ATTEMPTS = 300


def _gen_neighbors(cell, n):
    r, c = cell
    for dr, dc in _GEN_DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc


def _gen_monotone_path(n):
    moves = [(1, 0)] * (n - 1) + [(0, 1)] * (n - 1)
    random.shuffle(moves)
    r = c = 0
    cells = {(0, 0)}
    for dr, dc in moves:
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _gen_frontier(cells, n, blocked):
    out = set()
    for cell in cells:
        for nb in _gen_neighbors(cell, n):
            if nb not in cells and nb not in blocked:
                out.add(nb)
    return list(out)


def _gen_fill_connected(cells, n, target_count, blocked=frozenset()):
    cells = set(cells)
    while len(cells) < target_count:
        choices = _gen_frontier(cells, n, blocked)
        if not choices:
            return None
        cells.add(random.choice(choices))
    return cells


def _gen_try_island_shape(path, n, target_count):
    if target_count <= len(path) or random.random() >= _GEN_ISLAND_PROBABILITY:
        return None

    island_candidates = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) not in path
        and all(nb not in path for nb in _gen_neighbors((r, c), n))
    ]
    random.shuffle(island_candidates)
    for island in island_candidates:
        blocked = {island, *_gen_neighbors(island, n)}
        main_cells = _gen_fill_connected(path, n, target_count - 1, blocked)
        if main_cells is not None:
            cells = set(main_cells)
            cells.add(island)
            return cells
    return None


def _gen_has_forbidden_symmetry(mask):
    return (
        np.array_equal(mask, np.flipud(mask))
        and np.array_equal(mask, np.fliplr(mask))
    ) or np.array_equal(mask, np.rot90(mask))


def _gen_source_mask(n, target_count):
    path = _gen_monotone_path(n)
    cells = _gen_try_island_shape(path, n, target_count)
    if cells is None:
        cells = _gen_fill_connected(path, n, target_count)
    if cells is None:
        return None

    mask = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if _gen_has_forbidden_symmetry(mask):
        return None
    return mask


def _gen_reflect_cells(mask, marker_r, marker_c):
    reflected = set()
    n = mask.shape[0]
    for sr, sc in np.argwhere(mask):
        r = marker_r - n + int(sr)
        c = marker_c - n + int(sc)
        for rr in (r, 2 * marker_r + _GEN_MARKER_SIDE - 1 - r):
            for cc in (c, 2 * marker_c + _GEN_MARKER_SIDE - 1 - c):
                reflected.add((rr, cc))
    return reflected


def _gen_attempt():
    n = random.choice(_GEN_SOURCE_SIZES)
    min_side = max(_GEN_MIN_GRID_SIDE, 2 * n + _GEN_MARKER_SIDE)
    height = random.randint(min_side, _GEN_MAX_HEIGHT)
    width = random.randint(min_side, _GEN_MAX_WIDTH)

    count_options = [
        k for k in range(_GEN_MIN_SOURCE_PIXELS, _GEN_MAX_SOURCE_PIXELS + 1)
        if _GEN_MIN_SOURCE_DENSITY <= k / (n * n) <= _GEN_MAX_SOURCE_DENSITY
    ]
    target_count = random.choice(count_options)
    mask = _gen_source_mask(n, target_count)
    if mask is None:
        return None

    marker_r = random.randint(n, height - n - _GEN_MARKER_SIDE)
    marker_c = random.randint(n, width - n - _GEN_MARKER_SIDE)

    out = np.full((height, width), _GEN_BG_COLOR, dtype=int)
    marker_r1 = marker_r + _GEN_MARKER_SIDE
    marker_c1 = marker_c + _GEN_MARKER_SIDE
    out[marker_r:marker_r1, marker_c:marker_c1] = _GEN_MARKER_COLOR
    for r, c in _gen_reflect_cells(mask, marker_r, marker_c):
        out[r, c] = _GEN_SOURCE_COLOR

    inp = out.copy()
    inp[out == _GEN_SOURCE_COLOR] = _GEN_BG_COLOR
    source_r0 = marker_r - n
    source_c0 = marker_c - n
    inp[source_r0:marker_r, source_c0:marker_c][mask] = _GEN_SOURCE_COLOR

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 4938f0c2 input")
