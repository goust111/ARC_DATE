import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, colors_present, get_bg, paste
import itertools
import random


def _tile_occupancy(mask: np.ndarray) -> np.ndarray | None:
    b = bbox(mask)
    assert b is not None, "candidate color must be present"
    height = bbox_h(b)
    width = bbox_w(b)
    if height != width or height % 3 != 0:
        return None

    r0, c0, _r1, _c1 = b
    tile_size = height // 3
    if tile_size <= 1:
        return None
    occ = np.zeros((3, 3), dtype=bool)
    for br in range(3):
        for bc in range(3):
            tile = mask[
                r0 + br * tile_size : r0 + (br + 1) * tile_size,
                c0 + bc * tile_size : c0 + (bc + 1) * tile_size,
            ]
            filled = int(tile.sum())
            if filled == tile_size * tile_size:
                occ[br, bc] = True
            elif filled != 0:
                return None

    if not occ.any() or occ.all():
        return None
    return occ


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize the large square-tile mosaic as a 3x3 mask colored by the sparse marker color.

    Input structure:
    - The background is the dominant color.
    - Exactly two foreground colors appear.
    - One foreground color forms equal solid square tiles on a 3x3 lattice; the
      other foreground color is sparse marker noise.

    Rule:
    - Find the unique foreground color whose mask bbox divides into 3x3 equal
      square regions, each either completely filled or empty.
    - Output a 3x3 grid with the other foreground color at filled tile positions
      and background elsewhere.

    Key insight:
    - The noisy marker color is only the output paint color; the large tiled
      object encodes the output pattern by which coarse lattice cells are filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 2, "expected exactly two foreground colors"

    candidates: list[tuple[int, np.ndarray]] = []
    for color in foreground:
        occ = _tile_occupancy(g == color)
        if occ is not None:
            candidates.append((int(color), occ))

    assert len(candidates) == 1, "expected exactly one color to form the 3x3 tile mosaic"
    tile_color, occupancy = candidates[0]
    marker_colors = [int(color) for color in foreground if int(color) != tile_color]
    assert len(marker_colors) == 1, "expected the remaining foreground color to be the marker color"

    out = np.full((3, 3), int(bg), dtype=int)
    out[occupancy] = marker_colors[0]
    return out

# --------------- generator (o5x) ---------------

# Canvas and mosaic size bounds.
BG_COLOR = 0
CANVAS_SIZE_RANGE = (20, 26)
MOSAIC_GRID, MOSAIC_MARGIN = 3, 1
TILE_SIZE_RANGE = (3, 5)
# Coarse lattice and foreground color choices.
FILLED_CELL_RANGE = (4, 6)
FOREGROUND_COLORS = range(1, 10)
# Sparse marker bounds.
MARKER_COUNT_RANGE = (15, 55)
MARKER_DENSITY_RANGE = (0.04, 0.11)
# Rejection sampling cap.
GENERATION_RETRIES = 300


def _gen_occupancy() -> np.ndarray:
    """Sample a 3x3 occupancy mask that spans the full coarse lattice."""
    n_filled = random.randint(*FILLED_CELL_RANGE)
    cells = [(r, c) for r in range(MOSAIC_GRID) for c in range(MOSAIC_GRID)]
    edge = MOSAIC_GRID - 1
    valid = []
    for combo in itertools.combinations(cells, n_filled):
        rows, cols = zip(*combo)
        if min(rows) == 0 and max(rows) == edge and min(cols) == 0 and max(cols) == edge:
            valid.append(combo)

    occ = np.zeros((MOSAIC_GRID, MOSAIC_GRID), dtype=bool)
    for r, c in random.choice(valid):
        occ[r, c] = True
    return occ


def _render_mosaic(grid, occupancy, tile_color, top, left, tile_size):
    """Paint filled mosaic tiles and return the marker-blocked mask."""
    blocked = np.zeros(grid.shape, dtype=bool)
    tile_patch = np.full((tile_size, tile_size), tile_color, dtype=int)
    for br, bc in np.argwhere(occupancy):
        rr, cc = top + int(br) * tile_size, left + int(bc) * tile_size
        paste(grid, tile_patch, rr, cc)
        blocked[rr : rr + tile_size, cc : cc + tile_size] = True
    return blocked


def _scatter_markers(grid, blocked, marker_color, marker_count):
    """Scatter markers outside filled tiles while forcing quadrant coverage."""
    remaining_allowed = ~blocked.copy()
    height, width = remaining_allowed.shape
    hmid, wmid = height // 2, width // 2
    quadrants = ((0, hmid, 0, wmid), (0, hmid, wmid, width), (hmid, height, 0, wmid), (hmid, height, wmid, width))

    chosen_parts = []
    for qr0, qr1, qc0, qc1 in quadrants:
        candidates = np.argwhere(remaining_allowed[qr0:qr1, qc0:qc1])
        if len(candidates) == 0:
            return False
        pick = candidates[random.randrange(len(candidates))]
        rr, cc = int(qr0 + pick[0]), int(qc0 + pick[1])
        chosen_parts.append((rr, cc))
        remaining_allowed[rr, cc] = False

    allowed = np.argwhere(remaining_allowed)
    remaining_count = marker_count - len(chosen_parts)
    if len(allowed) < remaining_count:
        return False
    if remaining_count:
        chosen = allowed[np.random.choice(len(allowed), size=remaining_count, replace=False)]
        grid[chosen[:, 0], chosen[:, 1]] = marker_color
    for rr, cc in chosen_parts:
        grid[rr, cc] = marker_color
    return True


def _gen_attempt() -> np.ndarray | None:
    height, width = random.randint(*CANVAS_SIZE_RANGE), random.randint(*CANVAS_SIZE_RANGE)
    tile_size = random.randint(*TILE_SIZE_RANGE)
    mosaic_side = MOSAIC_GRID * tile_size
    top = random.randint(MOSAIC_MARGIN, height - mosaic_side - MOSAIC_MARGIN)
    left = random.randint(MOSAIC_MARGIN, width - mosaic_side - MOSAIC_MARGIN)

    tile_color, marker_color = random.sample(FOREGROUND_COLORS, 2)
    occupancy = _gen_occupancy()
    expected = np.full((MOSAIC_GRID, MOSAIC_GRID), BG_COLOR, dtype=int)
    expected[occupancy] = marker_color

    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = _render_mosaic(grid, occupancy, tile_color, top, left, tile_size)

    non_mosaic_area = height * width - mosaic_side * mosaic_side
    min_count = max(MARKER_COUNT_RANGE[0], int(np.ceil(MARKER_DENSITY_RANGE[0] * non_mosaic_area)))
    max_count = min(MARKER_COUNT_RANGE[1], int(np.floor(MARKER_DENSITY_RANGE[1] * non_mosaic_area)))
    if min_count > max_count:
        return None
    marker_count = random.randint(min_count, max_count)

    if not _scatter_markers(grid, blocked, marker_color, marker_count):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
