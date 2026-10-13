from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PATTERN_COLORS = (1, 2, 3, 4)
TILE_SHAPES = ((2, 2), (3, 2))
GEN_ATTEMPTS = 400


def _infer_minimal_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    known = np.argwhere((g != BLACK) & (g != GREY))
    assert len(known) > 0, "expected at least one colored sample"

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = np.full((pr, pc), -1, dtype=int)
            ok = True
            for r, c in known:
                rr = int(r % pr)
                cc = int(c % pc)
                color = int(g[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = color
                elif tile[rr, cc] != color:
                    ok = False
                    break
            if ok and not np.any(tile == -1):
                candidates.append((pr * pc, pr, pc, tile.copy()))

    assert candidates, "no valid periodic tile found"
    min_area = min(area for area, _pr, _pc, _tile in candidates)
    best = [(pr, pc, tile) for area, pr, pc, tile in candidates if area == min_area]
    assert len(best) == 1, "expected a unique minimal tile"
    pr, pc, tile = best[0]
    return pr, pc, tile


def _shadow_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(g == GREY):
        r = int(r)
        c = int(c)
        rr = r + 1
        while rr < h and int(g[rr, c]) == BLACK:
            blocked[rr, c] = True
            rr += 1
    return blocked


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    th, tw = tile.shape
    h, w = shape
    return np.fromfunction(lambda r, c: tile[r % th, c % tw], (h, w), dtype=int)


def _tile_has_no_smaller_period(tile: np.ndarray) -> bool:
    th, tw = tile.shape
    for pr in range(1, th + 1):
        for pc in range(1, tw + 1):
            if pr == th and pc == tw:
                continue
            fits = True
            for r in range(th):
                for c in range(tw):
                    if int(tile[r, c]) != int(tile[r % pr, c % pc]):
                        fits = False
                        break
                if not fits:
                    break
            if fits:
                return False
    return True


def _random_tile(shape: tuple[int, int]) -> np.ndarray:
    th, tw = shape
    while True:
        colors = random.sample(PATTERN_COLORS, random.choice((2, 3)))
        tile = np.array(
            [[random.choice(colors) for _ in range(tw)] for _ in range(th)],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _tile_has_no_smaller_period(tile):
            return tile


def _expected_output(inp: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocked = _shadow_mask(inp)
    out = np.asarray(inp, dtype=int).copy()
    for r, c in np.argwhere(inp == BLACK):
        r = int(r)
        c = int(c)
        if not blocked[r, c]:
            out[r, c] = int(full[r, c])
    return out, blocked


def main(grid: np.ndarray) -> np.ndarray:
    """Repair black gaps from the smallest periodic tile except where grey obstacles cast downward blocking shadows.

    Input structure:
    - A small grid is mostly filled by a repeating 2x2 or 3x2 color tile using colors 1-4.
    - Some cells are replaced by black (0) missing pixels, and a few non-bottom cells are replaced by grey (5) obstacle pixels.
    - Tile colors, black gaps, and grey obstacles share the same global row/column coordinate system.

    Rule:
    - Infer the unique smallest-area rectangular tile consistent with every non-black, non-grey pixel, with each tile residue observed at least once.
    - For each grey obstacle, cast a shadow straight downward through consecutive black cells only, starting one row below the obstacle and stopping just before the first non-black cell or the grid edge.
    - Fill each black cell with its periodic tile color unless it lies in any shadow; keep grey obstacle cells unchanged.

    Key insight:
    - Period recovery and obstacle shadowing are independent analyses over the same coordinates: a gap is repairable only when the periodic color is known and no obstacle shadow blocks that exact cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pr, pc, tile = _infer_minimal_tile(g)
    blocked = _shadow_mask(g)

    out = g.copy()
    for r, c in np.argwhere(g == BLACK):
        r = int(r)
        c = int(c)
        if not blocked[r, c]:
            out[r, c] = int(tile[r % pr, c % pc])
    return out


def _build_attempt() -> np.ndarray | None:
    h = random.randint(11, 16)
    w = random.randint(11, 16)
    shape = random.choice(TILE_SHAPES)
    tile = _random_tile(shape)
    full = _repeat_tile(tile, (h, w))

    obstacle_count = random.choice((1, 2, 3))
    columns = random.sample(range(w), obstacle_count)

    holes: set[tuple[int, int]] = set()
    obstacles: list[tuple[int, int]] = []

    # Force one stop-before-color branch and one unshadowed hole later in the same column.
    c0 = columns[0]
    r0 = random.randint(1, h - 5)
    lead_len = random.randint(1, min(3, h - r0 - 3))
    stopper_row = r0 + lead_len + 1
    later_hole = random.randint(stopper_row + 1, h - 1)
    obstacles.append((r0, c0))
    for rr in range(r0 + 1, r0 + lead_len + 1):
        holes.add((rr, c0))
    holes.add((later_hole, c0))

    for c in columns[1:]:
        mode = random.choice(("bottom", "stop"))
        if mode == "bottom":
            r = random.randint(max(0, h - 4), h - 2)
            obstacles.append((r, c))
            for rr in range(r + 1, h):
                holes.add((rr, c))
            continue

        r = random.randint(0, h - 4)
        shadow_len = random.randint(1, min(2, h - r - 2))
        obstacles.append((r, c))
        for rr in range(r + 1, r + shadow_len + 1):
            holes.add((rr, c))
        if r + shadow_len + 2 < h and random.random() < 0.6:
            holes.add((random.randint(r + shadow_len + 2, h - 1), c))

    obstacle_set = set(obstacles)
    holes = {pos for pos in holes if pos not in obstacle_set}
    if len(holes) > 10:
        return None

    inp = full.copy()
    for r, c in obstacle_set:
        inp[r, c] = GREY
    for r, c in holes:
        inp[r, c] = BLACK

    blocked = _shadow_mask(inp)
    blocked_holes = {pos for pos in holes if blocked[pos]}
    fillable_holes = {pos for pos in holes if not blocked[pos]}
    if not blocked_holes or not fillable_holes:
        return None

    target_holes = random.randint(5, 10)
    if len(holes) > target_holes:
        return None

    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in holes and (r, c) not in obstacle_set and not blocked[r, c]
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if len(holes) >= target_holes:
            break
        holes.add((r, c))
        inp[r, c] = BLACK

    if not (5 <= len(holes) <= 10):
        return None

    blocked = _shadow_mask(inp)
    blocked_holes = {pos for pos in holes if blocked[pos]}
    fillable_holes = {pos for pos in holes if not blocked[pos]}
    if not blocked_holes or not fillable_holes:
        return None
    if not any(c == c0 and r > stopper_row and not blocked[r, c] for r, c in holes):
        return None

    th, tw = shape
    residue_counts = np.zeros((th, tw), dtype=int)
    for r in range(h):
        for c in range(w):
            if int(inp[r, c]) not in (BLACK, GREY):
                residue_counts[r % th, c % tw] += 1
    if np.any(residue_counts == 0):
        return None

    try:
        pr, pc, inferred = _infer_minimal_tile(inp)
    except AssertionError:
        return None
    if (pr, pc) != shape or not np.array_equal(inferred, tile):
        return None

    expected, blocked = _expected_output(inp, full)
    if not all(int(expected[r, c]) == BLACK for r, c in blocked_holes):
        return None
    if not all(int(expected[r, c]) == int(full[r, c]) for r, c in fillable_holes):
        return None
    if any(int(expected[r, c]) != GREY for r, c in obstacle_set):
        return None
    if np.array_equal(expected, inp):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-shadow periodic repair input")
