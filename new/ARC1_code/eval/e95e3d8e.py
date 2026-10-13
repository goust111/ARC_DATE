import numpy as np
import random
from functools import lru_cache


def _tile_for_period(g: np.ndarray, known: np.ndarray, period: tuple[int, int]) -> np.ndarray | None:
    pr, pc = period
    tile = np.full((pr, pc), -1, dtype=int)

    for r, c in known:
        rr = int(r % pr)
        cc = int(c % pc)
        color = int(g[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = color
        elif tile[rr, cc] != color:
            return None

    if np.any(tile == -1):
        return None
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the black gaps by continuing the smallest rectangular periodic color tile.

    Input structure
    A grid contains a non-black rectangular periodic pattern with some black pixels
    removed from it.

    Rule
    Find the unique smallest-area period rectangle whose residue classes are
    consistent across all non-black pixels and all observed at least once, then
    replace each black pixel by the color at the same row/column residue.

    Key insight
    The black pixels are missing samples from a repeating two-dimensional pattern,
    not objects to preserve.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H > 0 and W > 0

    known = np.argwhere(g != 0)
    holes = np.argwhere(g == 0)
    assert len(known) > 0
    assert len(holes) > 0

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for pr in range(1, H + 1):
        for pc in range(1, W + 1):
            tile = _tile_for_period(g, known, (pr, pc))
            if tile is not None:
                candidates.append((pr * pc, pr, pc, tile))

    assert candidates
    min_area = min(area for area, _pr, _pc, _tile in candidates)
    best = [(pr, pc, tile) for area, pr, pc, tile in candidates if area == min_area]
    assert len(best) == 1

    pr, pc, tile = best[0]
    out = g.copy()
    for r, c in holes:
        out[r, c] = tile[int(r % pr), int(c % pc)]

    return out

# --------------- generator (o5x) ---------------

# Canvas, tile, and palette parameters.
_GEN_H, _GEN_W = 22, 22
_TILE_SIZE_RANGE, _TILE_COLOR_RANGE = (3, 8), (3, 7)
_TILE_COLOR_POOL, _TILE_RETRIES = range(1, 10), 60

# Hole rectangles and total erased area.
_HOLE_COUNT_RANGE, _HOLE_TOTAL_RANGE = (4, 5), (46, 65)
_RECT_H_RANGE, _RECT_W_RANGE, _RECT_AREA_RANGE = (2, 6), (2, 7), (4, 34)

# Placement and validity constraints.
_MIN_OBSERVATIONS_PER_RESIDUE, _HOLE_PADDING = 2, 1
_PLACEMENT_CANDIDATE_LIMIT = 90
_GEN_ATTEMPTS = 300
_UNKNOWN = -1


def _gen_repeated(tile: np.ndarray) -> np.ndarray:
    n = tile.shape[0]
    reps = (_GEN_H + n - 1) // n
    return np.tile(tile, (reps, reps))[:_GEN_H, :_GEN_W].copy()


def _gen_period_valid(g: np.ndarray, known: np.ndarray, pr: int, pc: int) -> bool:
    tile = np.full((pr, pc), _UNKNOWN, dtype=int)
    for r, c in known:
        rr = int(r % pr)
        cc = int(c % pc)
        color = int(g[r, c])
        if tile[rr, cc] == _UNKNOWN:
            tile[rr, cc] = color
        elif tile[rr, cc] != color:
            return False
    return not np.any(tile == _UNKNOWN)


def _gen_has_smaller_period(g: np.ndarray, known_mask: np.ndarray, area: int) -> bool:
    known = np.argwhere(known_mask)
    for pr in range(1, _GEN_H + 1):
        for pc in range(1, _GEN_W + 1):
            if pr * pc < area and _gen_period_valid(g, known, pr, pc):
                return True
    return False


def _gen_fill_symmetric_tile(n: int, colors: list[int]) -> np.ndarray:
    upper_cells = [(r, c) for r in range(n) for c in range(r, n)]
    tile = np.zeros((n, n), dtype=int)
    for r, c in upper_cells:
        color = random.choice(colors)
        tile[r, c] = color
        tile[c, r] = color

    forced = upper_cells[:]
    random.shuffle(forced)
    for color, (r, c) in zip(colors, forced):
        tile[r, c] = color
        tile[c, r] = color
    return tile


def _gen_symmetric_tile(n: int, n_colors: int) -> np.ndarray | None:
    full_known = np.ones((_GEN_H, _GEN_W), dtype=bool)
    for _ in range(_TILE_RETRIES):
        colors = random.sample(_TILE_COLOR_POOL, n_colors)
        tile = _gen_fill_symmetric_tile(n, colors)
        out = _gen_repeated(tile)
        if not _gen_has_smaller_period(out, full_known, n * n):
            return tile
    return None


@lru_cache(None)
def _gen_rect_options() -> list[tuple[int, int, int]]:
    return [
        (h * w, h, w)
        for h in range(_RECT_H_RANGE[0], _RECT_H_RANGE[1] + 1)
        for w in range(_RECT_W_RANGE[0], _RECT_W_RANGE[1] + 1)
        if _RECT_AREA_RANGE[0] <= h * w <= _RECT_AREA_RANGE[1]
    ]


def _gen_choose_rects(count: int, total: int) -> list[tuple[int, int]] | None:
    options = _gen_rect_options()
    areas = tuple(sorted({area for area, _h, _w in options}))

    @lru_cache(None)
    def reachable(slots: int, remaining: int) -> bool:
        if slots == 0:
            return remaining == 0
        return any(
            remaining >= area and reachable(slots - 1, remaining - area)
            for area in areas
        )

    rects: list[tuple[int, int]] = []
    remaining = total
    for i in range(count):
        slots = count - i - 1
        choices = [
            (area, h, w)
            for area, h, w in options
            if remaining >= area and reachable(slots, remaining - area)
        ]
        if not choices:
            return None
        area, h, w = random.choice(choices)
        rects.append((h, w))
        remaining -= area
    random.shuffle(rects)
    return rects


def _gen_residue_counts(n: int) -> np.ndarray:
    counts = np.zeros((n, n), dtype=int)
    for r in range(_GEN_H):
        for c in range(_GEN_W):
            counts[r % n, c % n] += 1
    return counts


def _gen_rect_counts(n: int, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    counts = np.zeros((n, n), dtype=int)
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            counts[r % n, c % n] += 1
    return counts


def _gen_candidate_placements(
    n: int,
    h: int,
    w: int,
    blocked: np.ndarray,
    removable: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    choices: list[tuple[int, int, np.ndarray]] = []
    for r in range(_GEN_H - h + 1):
        for c in range(_GEN_W - w + 1):
            if np.any(blocked[r : r + h, c : c + w]):
                continue
            removed = _gen_rect_counts(n, r, c, h, w)
            if np.all(removed <= removable):
                choices.append((r, c, removed))
    return choices


def _gen_place_rects(n: int, rects: list[tuple[int, int]]) -> np.ndarray | None:
    ordered = sorted(rects, key=lambda hw: hw[0] * hw[1], reverse=True)
    hole_mask = np.zeros((_GEN_H, _GEN_W), dtype=bool)
    blocked = np.zeros((_GEN_H, _GEN_W), dtype=bool)
    removable = _gen_residue_counts(n) - _MIN_OBSERVATIONS_PER_RESIDUE

    def place_one(i: int) -> bool:
        if i == len(ordered):
            return True

        h, w = ordered[i]
        choices = _gen_candidate_placements(n, h, w, blocked, removable)
        random.shuffle(choices)

        for r, c, removed in choices[:_PLACEMENT_CANDIDATE_LIMIT]:
            r0 = max(0, r - _HOLE_PADDING)
            c0 = max(0, c - _HOLE_PADDING)
            r1 = min(_GEN_H, r + h + _HOLE_PADDING)
            c1 = min(_GEN_W, c + w + _HOLE_PADDING)
            pad_rows, pad_cols = slice(r0, r1), slice(c0, c1)
            old_block = blocked[pad_rows, pad_cols].copy()

            hole_mask[r : r + h, c : c + w] = True
            blocked[pad_rows, pad_cols] = True
            removable[:] -= removed

            if place_one(i + 1):
                return True

            removable[:] += removed
            blocked[pad_rows, pad_cols] = old_block
            hole_mask[r : r + h, c : c + w] = False
        return False

    if place_one(0):
        return hole_mask
    return None


def _gen_design_holes(n: int) -> np.ndarray | None:
    hole_count = random.randint(*_HOLE_COUNT_RANGE)
    total_holes = random.randint(*_HOLE_TOTAL_RANGE)
    rects = _gen_choose_rects(hole_count, total_holes)
    if rects is None:
        return None

    holes = _gen_place_rects(n, rects)
    if holes is None or int(np.sum(holes)) != total_holes:
        return None
    return holes


def _gen_has_enough_observations(n: int, holes: np.ndarray) -> bool:
    observed = _gen_residue_counts(n)
    rows, cols = np.where(holes)
    np.add.at(observed, (rows % n, cols % n), -1)
    return not np.any(observed < _MIN_OBSERVATIONS_PER_RESIDUE)


def _gen_oracle_accepts(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_TILE_SIZE_RANGE)
    n_colors = random.randint(*_TILE_COLOR_RANGE)
    tile = _gen_symmetric_tile(n, n_colors)
    if tile is None:
        return None

    out = _gen_repeated(tile)
    holes = _gen_design_holes(n)
    if holes is None:
        return None
    if not _gen_has_enough_observations(n, holes):
        return None

    inp = out.copy()
    inp[holes] = 0
    if not _gen_oracle_accepts(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e95e3d8e input")
