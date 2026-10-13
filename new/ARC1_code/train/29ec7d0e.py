import numpy as np
import random


def _periodic_tile(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    seen = np.argwhere(g != 0)
    assert len(seen) > 0

    for p in range(1, min(h, w) + 1):
        tile = np.full((p, p), -1, dtype=int)
        ok = True
        for r, c in seen:
            rr, cc = r % p, c % p
            v = g[r, c]
            if tile[rr, cc] == -1:
                tile[rr, cc] = v
            elif tile[rr, cc] != v:
                ok = False
                break
        if ok and np.all(tile != -1):
            return tile

    raise AssertionError("no complete repeating tile fits the visible cells")


def main(grid: np.ndarray) -> np.ndarray:
    """Fill zero holes by reconstructing the smallest repeating color table visible in the grid.

    Input structure
    The grid is a rectangular periodic color table with black 0 cells as missing entries; each position in the true repeating tile is visible at least once.

    Rule
    Infer the smallest period p where all nonzero cells agree by (row mod p, column mod p), build that p x p tile, and replace every cell by the corresponding tile color.

    Key insight
    The nonzero pixels are samples of one complete repeating tile, so the repair follows from periodicity rather than from numeric color ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == 0)

    tile = _periodic_tile(g)
    p = tile.shape[0]
    rows = np.arange(g.shape[0])[:, None] % p
    cols = np.arange(g.shape[1])[None, :] % p
    out = tile[rows, cols]
    assert np.all((g == 0) | (out == g))
    return out

# --------------- generator (o5x) ---------------

# Canvas and tile parameters.
GRID_SIZE = 18
PERIOD_MIN, PERIOD_MAX = 5, 9

# Hole rectangle sampling.
RECT_COUNT = 5
RECT_SIDE_MIN, RECT_SIDE_MAX = 1, 6
RECT_AREA_MIN, RECT_AREA_MAX = 4, 16
TOTAL_HOLE_AREA_MIN, TOTAL_HOLE_AREA_MAX = 26, 51

# Rejection sampling budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_tile_29ec7d0e(p: int) -> np.ndarray:
    rows = np.arange(p)[:, None]
    cols = np.arange(p)[None, :]
    tile = (rows * cols) % p + 1
    tile[0, :] = 1
    tile[:, 0] = 1
    return tile.astype(int)


def _gen_full_grid_29ec7d0e(p: int) -> np.ndarray:
    tile = _gen_tile_29ec7d0e(p)
    rows = np.arange(GRID_SIZE)[:, None] % p
    cols = np.arange(GRID_SIZE)[None, :] % p
    return tile[rows, cols].copy()


def _gen_rectangles_29ec7d0e(total_area: int) -> list[tuple[int, int]]:
    """Sample rectangle dimensions that exactly match the requested total area."""
    area_to_dims: dict[int, list[tuple[int, int]]] = {}
    for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
        for w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
            area = h * w
            if RECT_AREA_MIN <= area <= RECT_AREA_MAX:
                area_to_dims.setdefault(area, []).append((h, w))

    rects: list[tuple[int, int]] = []
    remaining = total_area
    for i in range(RECT_COUNT):
        slots_after = RECT_COUNT - 1 - i
        possible = [
            area
            for area in area_to_dims
            if slots_after * RECT_AREA_MIN
            <= remaining - area
            <= slots_after * RECT_AREA_MAX
        ]
        if not possible:
            return []
        area = random.choice(possible)
        rects.append(random.choice(area_to_dims[area]))
        remaining -= area

    if remaining != 0:
        return []
    random.shuffle(rects)
    return sorted(rects, key=lambda hw: hw[0] * hw[1], reverse=True)


def _gen_residue_counts_29ec7d0e(h: int, w: int, r0: int, c0: int, p: int) -> np.ndarray:
    counts = np.zeros((p, p), dtype=int)
    rr = np.arange(r0, r0 + h) % p
    cc = np.arange(c0, c0 + w) % p
    np.add.at(counts, (rr[:, None], cc[None, :]), 1)
    return counts


def _gen_place_holes_29ec7d0e(p: int, rects: list[tuple[int, int]]) -> np.ndarray | None:
    hidden = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    total_counts = _gen_residue_counts_29ec7d0e(GRID_SIZE, GRID_SIZE, 0, 0, p)
    hidden_counts = np.zeros((p, p), dtype=int)

    for h, w in rects:
        choices = []
        for r0 in range(GRID_SIZE + 1 - h):
            for c0 in range(GRID_SIZE + 1 - w):
                if hidden[r0 : r0 + h, c0 : c0 + w].any():
                    continue
                add_counts = _gen_residue_counts_29ec7d0e(h, w, r0, c0, p)
                if np.any(hidden_counts + add_counts >= total_counts):
                    continue
                choices.append((r0, c0, add_counts))
        if not choices:
            return None

        r0, c0, add_counts = random.choice(choices)
        hidden[r0 : r0 + h, c0 : c0 + w] = True
        hidden_counts += add_counts

    if np.any(hidden_counts >= total_counts):
        return None
    return hidden


def _gen_attempt() -> np.ndarray | None:
    p = random.randint(PERIOD_MIN, PERIOD_MAX)
    out = _gen_full_grid_29ec7d0e(p)
    total_area = random.randint(TOTAL_HOLE_AREA_MIN, TOTAL_HOLE_AREA_MAX)
    rects = _gen_rectangles_29ec7d0e(total_area)
    if len(rects) != RECT_COUNT:
        return None

    hidden = _gen_place_holes_29ec7d0e(p, rects)
    if hidden is None:
        return None

    inp = out.copy()
    inp[hidden] = 0
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate 29ec7d0e input")
