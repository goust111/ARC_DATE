from __future__ import annotations
from collections import Counter

import numpy as np

from arc_utils import bbox, colors_present, get_bg, label_mask


def _unique_most_common(grid: np.ndarray, colors: list[int]) -> int:
    counts = Counter(map(int, grid.ravel()))
    best_count = max(counts[c] for c in colors)
    best = [c for c in colors if counts[c] == best_count]
    assert len(best) == 1
    return int(best[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each rectangular non-wall cavity with the colored seed pattern it already contains.

    Input structure
    - The dominant color forms solid walls around one or more rectangular cavities.
    - A unique most common non-wall color is the empty cavity color.
    - Each cavity contains exactly one other color, arranged as one copy of a repeat tile.

    Rule
    - For each 4-connected non-wall cavity, take the bbox of its seed color as the tile.
    - Repeat that tile across the entire cavity, preserving the phase set by the seed's current position.
    - Leave wall cells unchanged and leave tile-empty positions as the empty cavity color.

    Key insight
    - The colored pixels are not moved objects; they are one visible instance of a periodic mask that fills its containing rectangle.
    """
    g = np.asarray(grid, dtype=int)
    wall = get_bg(g)
    non_wall_colors = colors_present(g, exclude={wall})
    assert len(non_wall_colors) >= 2
    empty = _unique_most_common(g, non_wall_colors)

    out = g.copy()
    labels, n_regions = label_mask(g != wall, connectivity=4)
    assert n_regions >= 1

    for region_id in range(1, n_regions + 1):
        region_mask = labels == region_id
        region_bbox = bbox(region_mask)
        assert region_bbox is not None
        r0, c0, r1, c1 = region_bbox
        assert region_mask[r0 : r1 + 1, c0 : c1 + 1].all()

        region_colors = set(map(int, g[region_mask]))
        assert empty in region_colors
        seed_colors = region_colors - {empty}
        assert len(seed_colors) == 1
        seed_color = seed_colors.pop()

        seed_bbox = bbox((g == seed_color) & region_mask)
        assert seed_bbox is not None
        sr0, sc0, sr1, sc1 = seed_bbox
        tile = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        assert set(map(int, tile.ravel())) <= {empty, seed_color}

        tile_h, tile_w = tile.shape
        region_h, region_w = r1 - r0 + 1, c1 - c0 + 1
        assert region_h % tile_h == 0
        assert region_w % tile_w == 0
        assert (sr0 - r0) % tile_h == 0
        assert (sc0 - c0) % tile_w == 0

        rr = np.arange(r0, r1 + 1)[:, None]
        cc = np.arange(c0, c1 + 1)[None, :]
        out[r0 : r1 + 1, c0 : c1 + 1] = tile[(rr - sr0) % tile_h, (cc - sc0) % tile_w]

    return out

# --------------- generator (o5x) ---------------
# Colors and palette.
_GEN_WALL = 8
_GEN_EMPTY = 0
_GEN_SEED_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

# Canvas and cavity coverage.
_GEN_GRID_ROWS = range(19, 27)
_GEN_GRID_COLS = range(13, 23)
_GEN_CAVITY_COUNT_RANGE = (1, 3)
_GEN_NON_WALL_PCT_RANGE = (23, 36)
_GEN_WALL_FRACTION_RANGE = (0.64, 0.77)

# Cavity dimensions and wall spacing.
_GEN_CAVITY_HEIGHT_OPTIONS = (6, 8, 9, 10, 12)
_GEN_CAVITY_WIDTH_OPTIONS = (6, 8, 9, 10, 12, 15)
_GEN_CAVITY_DIMS = [(h, w) for h in _GEN_CAVITY_HEIGHT_OPTIONS for w in _GEN_CAVITY_WIDTH_OPTIONS]
_GEN_BORDER_WALL_THICKNESS = 2
_GEN_CAVITY_START_GAP = 3
_GEN_MIN_CAVITY_AREA = min(h * w for h, w in _GEN_CAVITY_DIMS)
_GEN_MAX_CAVITY_AREA = max(h * w for h, w in _GEN_CAVITY_DIMS)
_GEN_CAVITY_WIDTH_WEIGHT_POWER = 2

# Tile dimensions, repetition, and density.
_GEN_TILE_HEIGHT_RANGE = (2, 3)
_GEN_TILE_WIDTH_RANGE = (2, 5)
_GEN_REPS_PER_AXIS_RANGE = (2, 5)
_GEN_TILE_DENSITY_PCT_RANGE = (56, 78)

# Retry limits.
_GEN_MAX_ATTEMPTS = 300
_GEN_PERCENT_DENOM = 100


def _gen_floor_percent(percent: int, value: int) -> int: return (percent * value) // _GEN_PERCENT_DENOM


def _gen_ceil_percent(percent: int, value: int) -> int: return (percent * value + _GEN_PERCENT_DENOM - 1) // _GEN_PERCENT_DENOM


def _gen_factor_options(length: int, lo: int, hi: int) -> list[int]:
    rep_min, rep_max = _GEN_REPS_PER_AXIS_RANGE
    return [d for d in range(lo, hi + 1) if length % d == 0 and rep_min <= length // d <= rep_max]


def _gen_choose_grid(rng: np.random.Generator, count: int) -> tuple[int, int] | None:
    choices = []
    min_cavity_area = _GEN_MIN_CAVITY_AREA * count
    max_cavity_area = _GEN_MAX_CAVITY_AREA * count
    for rows in _GEN_GRID_ROWS:
        for cols in _GEN_GRID_COLS:
            area = rows * cols
            if min_cavity_area <= _gen_floor_percent(_GEN_NON_WALL_PCT_RANGE[1], area) and max_cavity_area >= _gen_ceil_percent(_GEN_NON_WALL_PCT_RANGE[0], area):
                choices.append((rows, cols))
    if not choices:
        return None
    return choices[int(rng.integers(0, len(choices)))]


def _gen_choose_cavity_dims(rng: np.random.Generator, rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    area = rows * cols
    min_area = _gen_ceil_percent(_GEN_NON_WALL_PCT_RANGE[0], area)
    max_area = _gen_floor_percent(_GEN_NON_WALL_PCT_RANGE[1], area)
    max_h = rows - 2 * _GEN_BORDER_WALL_THICKNESS
    max_w = cols - 2 * _GEN_BORDER_WALL_THICKNESS
    options = [(h, w) for h, w in _GEN_CAVITY_DIMS if h <= max_h and w <= max_w]
    combos: list[tuple[tuple[int, int], ...]] = []

    def rec(start: list[tuple[int, int]], area: int) -> None:
        if len(start) == count:
            if min_area <= area <= max_area:
                combos.append(tuple(start))
            return
        remaining = count - len(start)
        if area + remaining * _GEN_MIN_CAVITY_AREA > max_area:
            return
        if area + remaining * _GEN_MAX_CAVITY_AREA < min_area:
            return
        order = rng.permutation(len(options))
        for idx in order:
            h, w = options[int(idx)]
            rec(start + [(h, w)], area + h * w)

    rec([], 0)
    if not combos:
        return None
    weights = np.array([(sum(w for _, w in combo) / count) ** _GEN_CAVITY_WIDTH_WEIGHT_POWER for combo in combos], dtype=float)
    weights /= weights.sum()
    return list(combos[int(rng.choice(len(combos), p=weights))])


def _gen_rect_is_clear(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        separated = r1 + _GEN_CAVITY_START_GAP <= pr0 or pr1 + _GEN_CAVITY_START_GAP <= r0 or c1 + _GEN_CAVITY_START_GAP <= pc0 or pc1 + _GEN_CAVITY_START_GAP <= c0
        if not separated:
            return False
    return True


def _gen_place_cavities(rng: np.random.Generator, rows: int, cols: int, dims: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for h, w in sorted(dims, key=lambda x: x[0] * x[1], reverse=True):
        positions: list[tuple[int, int, int, int]] = []
        for r0 in range(_GEN_BORDER_WALL_THICKNESS, rows - h - _GEN_BORDER_WALL_THICKNESS + 1):
            for c0 in range(_GEN_BORDER_WALL_THICKNESS, cols - w - _GEN_BORDER_WALL_THICKNESS + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if _gen_rect_is_clear(rect, placed):
                    positions.append(rect)
        if not positions:
            return None
        placed.append(positions[int(rng.integers(0, len(positions)))])
    return placed


def _gen_tile_specs(rng: np.random.Generator, cavity_h: int, cavity_w: int) -> tuple[int, int, int] | None:
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for tile_h in _gen_factor_options(cavity_h, *_GEN_TILE_HEIGHT_RANGE):
        for tile_w in _gen_factor_options(cavity_w, *_GEN_TILE_WIDTH_RANGE):
            area = tile_h * tile_w
            min_pixels = _gen_ceil_percent(_GEN_TILE_DENSITY_PCT_RANGE[0], area)
            max_pixels = _gen_floor_percent(_GEN_TILE_DENSITY_PCT_RANGE[1], area)
            for count in range(min_pixels, max_pixels + 1):
                density_key = int(round(_GEN_PERCENT_DENOM * count / area))
                groups.setdefault(density_key, []).append((tile_h, tile_w, count))
    if not groups:
        return None
    density_key = list(groups)[int(rng.integers(0, len(groups)))]
    choices = groups[density_key]
    return choices[int(rng.integers(0, len(choices)))]


def _gen_tile(rng: np.random.Generator, tile_h: int, tile_w: int, color: int, count: int) -> np.ndarray:
    if rng.integers(0, 2) == 0:
        required = {(0, 0), (tile_h - 1, tile_w - 1)}
    else:
        required = {(0, tile_w - 1), (tile_h - 1, 0)}
    coords = [(r, c) for r in range(tile_h) for c in range(tile_w) if (r, c) not in required]
    extra = count - len(required)
    if extra > 0:
        chosen = rng.choice(len(coords), size=extra, replace=False)
        required.update(coords[int(i)] for i in np.atleast_1d(chosen))

    tile = np.full((tile_h, tile_w), _GEN_EMPTY, dtype=int)
    for r, c in required:
        tile[r, c] = color
    return tile


def _gen_design_layout(rng: np.random.Generator) -> tuple[int, int, int, list[tuple[int, int, int, int]]] | None:
    count = int(rng.integers(_GEN_CAVITY_COUNT_RANGE[0], _GEN_CAVITY_COUNT_RANGE[1] + 1))
    grid_dims = _gen_choose_grid(rng, count)
    if grid_dims is None:
        return None
    rows, cols = grid_dims

    dims = _gen_choose_cavity_dims(rng, rows, cols, count)
    if dims is None:
        return None
    rects = _gen_place_cavities(rng, rows, cols, dims)
    if rects is None:
        return None
    return count, rows, cols, rects


def _gen_render_cavity(rng: np.random.Generator, inp: np.ndarray, out: np.ndarray, rect: tuple[int, int, int, int], seed_color: int) -> bool:
    r0, c0, r1, c1 = rect
    cavity_h, cavity_w = r1 - r0 + 1, c1 - c0 + 1
    tile_specs = _gen_tile_specs(rng, cavity_h, cavity_w)
    if tile_specs is None:
        return False
    tile_h, tile_w, seed_count = tile_specs
    tile = _gen_tile(rng, tile_h, tile_w, int(seed_color), seed_count)

    rep_r, rep_c = cavity_h // tile_h, cavity_w // tile_w
    out[r0 : r1 + 1, c0 : c1 + 1] = np.tile(tile, (rep_r, rep_c))
    inp[r0 : r1 + 1, c0 : c1 + 1] = _GEN_EMPTY

    seed_rep_r = int(rng.integers(0, rep_r))
    seed_rep_c = int(rng.integers(0, rep_c))
    sr = r0 + seed_rep_r * tile_h
    sc = c0 + seed_rep_c * tile_w
    inp[sr : sr + tile_h, sc : sc + tile_w] = tile
    return True


def _gen_pair_is_valid(inp: np.ndarray, out: np.ndarray) -> bool:
    wall_fraction = float(np.count_nonzero(inp == _GEN_WALL)) / inp.size
    if not (_GEN_WALL_FRACTION_RANGE[0] <= wall_fraction <= _GEN_WALL_FRACTION_RANGE[1]):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt(rng: np.random.Generator | None = None) -> np.ndarray | None:
    if rng is None:
        rng = np.random.default_rng()

    layout = _gen_design_layout(rng)
    if layout is None:
        return None
    count, rows, cols, rects = layout

    out = np.full((rows, cols), _GEN_WALL, dtype=int)
    inp = out.copy()
    seed_colors = rng.choice(_GEN_SEED_COLORS, size=count, replace=False)
    for rect, seed_color in zip(rects, seed_colors):
        if not _gen_render_cavity(rng, inp, out, rect, int(seed_color)):
            return None

    if not _gen_pair_is_valid(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid a57f2f04 input")
