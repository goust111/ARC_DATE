from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

PATTERN_COLORS = (BLUE, RED)
MISSING_COLORS = {GREEN, PINK}
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

TILE_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
GRID_SIDE_MIN = 9
GRID_SIDE_MAX = 14
GREEN_COUNT_RANGE = (2, 4)
PINK_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400


def _tile_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % tile.shape[0]
    cc = np.arange(width)[None, :] % tile.shape[1]
    return tile[rr, cc].astype(int)


def _has_period(tile: np.ndarray, row_period: int, col_period: int) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                return False
    return True


def _full_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    full_area = h * w
    best = min(
        row_period * col_period
        for row_period in range(1, h + 1)
        for col_period in range(1, w + 1)
        if _has_period(tile, row_period, col_period)
    )
    return best == full_area


def _random_tile() -> np.ndarray:
    for _ in range(500):
        h, w = random.choice(TILE_SHAPES)
        area = h * w
        red_count = random.randint(1, area - 1)
        tile = np.full((h, w), BLUE, dtype=int)
        for idx in random.sample(range(area), red_count):
            tile.flat[idx] = RED
        if _full_period(tile):
            return tile
    raise RuntimeError("failed to sample a full-period tile")


def _candidate_tile(
    grid: np.ndarray, row_period: int, col_period: int
) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    seen = np.zeros((row_period, col_period), dtype=bool)
    for r, c in np.argwhere(~np.isin(grid, list(MISSING_COLORS))):
        rr = int(r) % row_period
        cc = int(c) % col_period
        value = int(grid[r, c])
        if value not in PATTERN_COLORS:
            return None
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
            seen[rr, cc] = True
        elif int(tile[rr, cc]) != value:
            return None
    if not np.all(seen):
        return None
    if set(map(int, tile.ravel())) != set(PATTERN_COLORS):
        return None
    return tile


def _infer_unique_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _candidate_tile(g, row_period, col_period)
            if tile is not None:
                candidates.append((row_period * col_period, row_period, col_period, tile))
    assert candidates, "visible pattern cells must define a periodic tile"
    min_area = min(area for area, _rp, _cp, _tile in candidates)
    minimal = [
        (row_period, col_period, tile)
        for area, row_period, col_period, tile in candidates
        if area == min_area
    ]
    assert len(minimal) == 1, "minimum-area tile must be unique"
    return minimal[0]


def _apply_rule(
    grid: np.ndarray, tile_info: tuple[int, int, np.ndarray] | None = None
) -> np.ndarray:
    row_period, col_period, tile = _infer_unique_tile(grid) if tile_info is None else tile_info
    out = np.asarray(grid, dtype=int).copy()
    green_coords = [(int(r), int(c)) for r, c in np.argwhere(out == GREEN)]

    for r, c in np.argwhere(out == PINK):
        out[int(r), int(c)] = int(tile[int(r) % row_period, int(c) % col_period])

    for r, c in green_coords:
        virtual_color = int(tile[r % row_period, c % col_period])
        steps = DIAGONALS if virtual_color == RED else ORTHOGONALS
        paint = YELLOW if virtual_color == RED else ORANGE
        for dr, dc in steps:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and int(out[nr, nc]) in PATTERN_COLORS:
                out[nr, nc] = paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a periodic red-blue tile through pink/green occlusions, repair pink cells, and emit green-centered neighborhood marks chosen by the hidden tile color.

    Input structure:
    - The grid is a small rectangle tiled by a strict periodic pattern using only blue(1) and red(2), except for pink(6) and green(3) occlusions.
    - Pink and green cells are interior markers that hide the underlying tile value at their coordinates.
    - Ignoring pink and green, every visible red/blue cell is consistent with one unique minimum-area rectangular repeating tile.

    Rule:
    - Treat all pink and green cells as missing values and infer the unique smallest rectangular periodic tile from the remaining blue/red cells.
    - Replace every pink cell with the inferred tile color at the same row/column residue.
    - Keep every green cell green, but use its inferred hidden tile color to choose an expansion: hidden red(2) paints yellow(4) on the four diagonal neighbors, while hidden blue(1) paints orange(7) on the four orthogonal neighbors.
    - Apply expansions only onto in-bounds cells currently holding blue or red after the pink repairs are done.

    Key insight:
    - Green cells have a double role: they are missing-value occlusions during tile inference, then they become triggers whose neighborhood geometry is determined by the recovered hidden tile color underneath them.
    """
    return _apply_rule(grid)


def _margin(coord: tuple[int, int], height: int, width: int) -> int:
    r, c = coord
    return min(r, c, height - 1 - r, width - 1 - c)


def _chebyshev_far(a: tuple[int, int], b: tuple[int, int], gap: int = 3) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= gap


def _pick_reserve_cells(height: int, width: int, row_period: int, col_period: int) -> set[tuple[int, int]]:
    residue_cells: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for r in range(height):
        for c in range(width):
            residue_cells[(r % row_period, c % col_period)].append((r, c))
    reserves = set()
    for coords in residue_cells.values():
        reserves.add(min(coords, key=lambda rc: _margin(rc, height, width)))
    return reserves


def _select_greens(
    base: np.ndarray, reserves: set[tuple[int, int]], target_count: int
) -> list[tuple[int, int]] | None:
    h, w = base.shape
    eligible = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if (r, c) not in reserves
    ]
    by_color = {
        color: [(r, c) for r, c in eligible if int(base[r, c]) == color]
        for color in PATTERN_COLORS
    }
    for color in PATTERN_COLORS:
        random.shuffle(by_color[color])

    greens: list[tuple[int, int]] = []
    for color in (RED, BLUE):
        for coord in by_color[color]:
            if all(_chebyshev_far(coord, other) for other in greens):
                greens.append(coord)
                break
        else:
            return None

    remainder = [coord for coord in eligible if coord not in greens]
    random.shuffle(remainder)
    for coord in remainder:
        if len(greens) >= target_count:
            break
        if all(_chebyshev_far(coord, other) for other in greens):
            greens.append(coord)
    if len(greens) != target_count:
        return None
    return greens


def _green_footprint(base: np.ndarray, green: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = green
    virtual_color = int(base[r, c])
    steps = DIAGONALS if virtual_color == RED else ORTHOGONALS
    footprint = []
    for dr, dc in steps:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < base.shape[0] and 0 <= nc < base.shape[1]:
            footprint.append((nr, nc))
    return footprint


def _select_pinks(
    base: np.ndarray,
    greens: list[tuple[int, int]],
    reserves: set[tuple[int, int]],
    target_count: int,
) -> list[tuple[int, int]] | None:
    h, w = base.shape
    green_set = set(greens)
    footprint_cells = []
    for green in greens:
        for coord in _green_footprint(base, green):
            if (
                1 <= coord[0] < h - 1
                and 1 <= coord[1] < w - 1
                and coord not in green_set
                and coord not in reserves
            ):
                footprint_cells.append(coord)
    footprint_cells = list(dict.fromkeys(footprint_cells))
    if not footprint_cells:
        return None

    non_footprint = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in green_set and (r, c) not in reserves and (r, c) not in footprint_cells
    ]
    if not non_footprint:
        return None

    pinks: list[tuple[int, int]] = [random.choice(footprint_cells)]
    preferred_survivors = [coord for coord in non_footprint if int(base[coord]) == int(base[pinks[0]])]
    if not preferred_survivors:
        preferred_survivors = non_footprint
    pinks.append(random.choice(preferred_survivors))

    remaining = [coord for coord in footprint_cells + non_footprint if coord not in pinks]
    random.shuffle(remaining)
    for coord in remaining:
        if len(pinks) >= target_count:
            break
        pinks.append(coord)
    if len(pinks) != target_count:
        return None
    if set(int(base[r, c]) for r, c in pinks) != set(PATTERN_COLORS):
        return None
    return pinks


def _validate_instance(
    inp: np.ndarray,
    tile_info: tuple[int, int, np.ndarray],
    greens: list[tuple[int, int]],
    pinks: list[tuple[int, int]],
) -> bool:
    row_period, col_period, tile = tile_info
    h, w = inp.shape
    if max(h, w) > 30:
        return False
    if any(_margin(coord, h, w) < 1 for coord in greens + pinks):
        return False
    if any(not _chebyshev_far(a, b) for i, a in enumerate(greens) for b in greens[i + 1 :]):
        return False

    visible = inp[~np.isin(inp, list(MISSING_COLORS))]
    if set(map(int, np.unique(visible))) != set(PATTERN_COLORS):
        return False

    try:
        inferred = _infer_unique_tile(inp)
    except AssertionError:
        return False
    if inferred[0] != row_period or inferred[1] != col_period or not np.array_equal(inferred[2], tile):
        return False

    out = main(inp)
    green_virtuals = {int(tile[r % row_period, c % col_period]) for r, c in greens}
    if green_virtuals != set(PATTERN_COLORS):
        return False

    overwritten_pink = False
    surviving_pink = False
    for r, c in pinks:
        if int(out[r, c]) in (YELLOW, ORANGE):
            overwritten_pink = True
        if int(out[r, c]) in PATTERN_COLORS:
            surviving_pink = True
    if not overwritten_pink or not surviving_pink:
        return False

    for r, c in greens:
        if int(out[r, c]) != GREEN:
            return False
        virtual = int(tile[r % row_period, c % col_period])
        diagonal_targets = {
            (r + dr, c + dc)
            for dr, dc in DIAGONALS
            if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in greens
        }
        orth_targets = {
            (r + dr, c + dc)
            for dr, dc in ORTHOGONALS
            if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in greens
        }
        if virtual == RED:
            if not any(int(out[rr, cc]) == YELLOW for rr, cc in diagonal_targets):
                return False
            if any(int(out[rr, cc]) == ORANGE for rr, cc in orth_targets):
                return False
        else:
            if not any(int(out[rr, cc]) == ORANGE for rr, cc in orth_targets):
                return False
            if any(int(out[rr, cc]) == YELLOW for rr, cc in diagonal_targets):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile = _random_tile()
    row_period, col_period = tile.shape
    height = random.randint(max(GRID_SIDE_MIN, row_period * 3), GRID_SIDE_MAX)
    width = random.randint(max(GRID_SIDE_MIN, col_period * 3), GRID_SIDE_MAX)
    base = _tile_grid(tile, height, width)

    reserves = _pick_reserve_cells(height, width, row_period, col_period)
    green_count = random.randint(*GREEN_COUNT_RANGE)
    greens = _select_greens(base, reserves, green_count)
    if greens is None:
        return None

    pink_count = random.randint(*PINK_COUNT_RANGE)
    pinks = _select_pinks(base, greens, reserves, pink_count)
    if pinks is None:
        return None

    inp = base.copy()
    for r, c in pinks:
        inp[r, c] = PINK
    for r, c in greens:
        inp[r, c] = GREEN

    tile_info = (row_period, col_period, tile)
    if not _validate_instance(inp, tile_info, greens, pinks):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
