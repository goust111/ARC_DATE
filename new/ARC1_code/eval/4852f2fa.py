import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, paste
import itertools
import random


def _axis_start(lo: int, hi: int, mean: float, limit: int) -> int:
    assert 0 <= lo <= hi < limit
    assert hi - lo + 1 <= 3

    first = max(0, hi - 2)
    last = min(lo, limit - 3)
    starts = list(range(first, last + 1))
    assert starts

    distances = [abs((start + 1) - mean) for start in starts]
    best = min(distances)
    best_starts = [start for start, dist in zip(starts, distances) if dist == best]
    assert len(best_starts) == 1
    return best_starts[0]


def _find_roles(g: np.ndarray, bg: int) -> tuple[int, int, np.ndarray]:
    components = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    by_color: dict[int, list[dict]] = {}
    for comp in components:
        color = int(comp["color"])
        by_color.setdefault(color, []).append(comp)

    assert len(by_color) == 2
    marker_candidates: list[list[dict]] = []
    sprite_candidates: list[dict] = []
    for comps in by_color.values():
        areas = [int(comp["area"]) for comp in comps]
        if all(area == 1 for area in areas):
            marker_candidates.append(comps)
        if len(comps) == 1 and areas[0] > 1:
            sprite_candidates.append(comps[0])

    assert len(marker_candidates) == 1
    assert len(sprite_candidates) == 1

    marker_count = len(marker_candidates[0])
    sprite = sprite_candidates[0]
    assert marker_count > 0
    return int(sprite["color"]), marker_count, sprite["coords"]


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat the teal-like 3x3 sprite once for each singleton marker pixel.

    Input structure
    - The grid has a dominant background, one non-background color forming isolated singleton markers, and one other non-background color forming a single 8-connected sprite.
    - The sprite fits inside a unique centroid-centered 3x3 window.

    Rule
    - Extract that 3x3 sprite window with background in its empty cells.
    - Concatenate one copy of the sprite window horizontally for each singleton marker.

    Key insight
    - The marker pixels encode only the repetition count; the multi-pixel object is the repeated sprite template.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    assert height >= 3 and width >= 3

    bg = get_bg(g)
    sprite_color, marker_count, sprite_coords = _find_roles(g, bg)

    r0, c0 = sprite_coords.min(axis=0)
    r1, c1 = sprite_coords.max(axis=0)
    mean_r, mean_c = sprite_coords.mean(axis=0)
    top = _axis_start(int(r0), int(r1), float(mean_r), height)
    left = _axis_start(int(c0), int(c1), float(mean_c), width)

    tile = np.full((3, 3), bg, dtype=int)
    rr = sprite_coords[:, 0] - top
    cc = sprite_coords[:, 1] - left
    assert bool(np.all((0 <= rr) & (rr < 3) & (0 <= cc) & (cc < 3)))
    tile[rr, cc] = sprite_color

    return np.tile(tile, (1, marker_count))

# --------------- generator (o5x) ---------------

# Grid/tile geometry.
GRID_SIZE, TILE_SIZE = 9, 3
WINDOW_STARTS = GRID_SIZE - TILE_SIZE + 1
TILE_CELL_COUNT = TILE_SIZE * TILE_SIZE
# Fixed colors.
BG_COLOR, SPRITE_COLOR, MARKER_COLOR = 0, 8, 4
# Sampling ranges and spacing.
SPRITE_AREA_RANGE = (4, 6)
MARKER_COUNT_RANGE = (1, 4)
ADJACENCY_RADIUS = 1
# Retry limit.
GEN_RETRY_LIMIT = 300

_GEN_TILE_MASKS_BY_AREA = {}


def _gen_axis_matches_window(values, offset):
    vals = [int(v) + int(offset) for v in values]
    lo = min(vals)
    hi = max(vals)
    mean = sum(vals) / len(vals)
    try:
        return _axis_start(lo, hi, mean, GRID_SIZE) == int(offset)
    except AssertionError:
        return False


def _gen_valid_placements(mask):
    """Return placements where the solver extracts this mask's source window."""
    coords = np.argwhere(mask)
    placements = []
    for top in range(WINDOW_STARTS):
        if not _gen_axis_matches_window(coords[:, 0], top):
            continue
        for left in range(WINDOW_STARTS):
            if _gen_axis_matches_window(coords[:, 1], left):
                placements.append((top, left))
    return placements


def _gen_valid_tile_masks(area):
    """Return cached contiguous 3x3 masks paired with valid grid placements."""
    area = int(area)
    if area in _GEN_TILE_MASKS_BY_AREA:
        return _GEN_TILE_MASKS_BY_AREA[area]

    valid = []
    for cells in itertools.combinations(range(TILE_CELL_COUNT), area):
        mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
        for idx in cells:
            mask[idx // TILE_SIZE, idx % TILE_SIZE] = True
        if not is_contiguous(mask, background=0, connectivity=8):
            continue

        placements = _gen_valid_placements(mask)
        if placements:
            valid.append((mask, placements))

    assert valid
    _GEN_TILE_MASKS_BY_AREA[area] = valid
    return valid


def _gen_apart(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > ADJACENCY_RADIUS


def _gen_marker_candidates(mask, top, left):
    """Return shuffled cells that are not 8-adjacent to the sprite."""
    sprite = [(int(r) + top, int(c) + left) for r, c in np.argwhere(mask)]
    cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if all(max(abs(r - sr), abs(c - sc)) > ADJACENCY_RADIUS for sr, sc in sprite):
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _gen_choose_markers(cells, count):
    """Choose count cells with no 8-adjacent marker pair, or None if impossible."""
    count = int(count)

    def search(chosen, pool):
        if len(chosen) == count:
            return chosen
        need_after_pick = count - len(chosen) - 1
        options = []
        for i, cell in enumerate(pool):
            next_pool = [p for p in pool[i + 1 :] if _gen_apart(cell, p)]
            if len(next_pool) >= need_after_pick:
                options.append((cell, next_pool))
        random.shuffle(options)
        for cell, next_pool in options:
            result = search(chosen + [cell], next_pool)
            if result is not None:
                return result
        return None

    return search([], list(cells))


def _gen_render_grid(tile, top, left, markers):
    grid = np.full((GRID_SIZE, GRID_SIZE), BG_COLOR, dtype=int)
    paste(grid, tile, top, left, transparent=BG_COLOR)
    for r, c in markers:
        grid[r, c] = MARKER_COLOR
    return grid


def _gen_oracle_accepts(grid, expected):
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    area = random.randint(*SPRITE_AREA_RANGE)
    mask, placements = random.choice(_gen_valid_tile_masks(area))
    tile = np.full((TILE_SIZE, TILE_SIZE), BG_COLOR, dtype=int)
    tile[mask] = SPRITE_COLOR

    marker_count = random.randint(*MARKER_COUNT_RANGE)
    expected = np.tile(tile, (1, marker_count))

    top, left = random.choice(placements)
    markers = _gen_choose_markers(_gen_marker_candidates(mask, top, left), marker_count)
    if markers is None:
        return None
    grid = _gen_render_grid(tile, top, left, markers)
    return grid if _gen_oracle_accepts(grid, expected) else None


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
