import random
from collections import Counter

import numpy as np


BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# --------------- solver ---------------


def _divisors(n: int) -> list[int]:
    vals: list[int] = []
    for d in range(1, n + 1):
        if n % d == 0:
            vals.append(d)
    return vals


def _majority_color(values: np.ndarray) -> int:
    flat = [int(v) for v in values.ravel()]
    counts = Counter(flat)
    best_count = max(counts.values())
    for value in flat:
        if counts[value] == best_count:
            return int(value)
    raise AssertionError("unreachable")


def _reconstruct_from_period(grid: np.ndarray, tile_h: int, tile_w: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    h, w = grid.shape
    tile = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            tile[r, c] = _majority_color(grid[r::tile_h, c::tile_w])

    repeated = np.tile(tile, (h // tile_h, w // tile_w))
    mismatches = repeated != grid

    bad_tiles: list[tuple[int, int]] = []
    for tr in range(h // tile_h):
        for tc in range(w // tile_w):
            block = mismatches[tr * tile_h : (tr + 1) * tile_h, tc * tile_w : (tc + 1) * tile_w]
            if bool(block.any()):
                bad_tiles.append((tr, tc))

    return tile, repeated, bad_tiles


def _valid_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int, int], int, int, np.ndarray, tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int], int, int, np.ndarray, tuple[int, int]]] = []
    for tile_h in _divisors(h):
        for tile_w in _divisors(w):
            n_tiles = (h // tile_h) * (w // tile_w)
            if n_tiles < 3:
                continue

            _tile, repaired, bad_tiles = _reconstruct_from_period(grid, tile_h, tile_w)
            if len(bad_tiles) != 1 or np.array_equal(repaired, grid):
                continue

            score = (tile_h * tile_w, tile_h, tile_w)
            candidates.append((score, tile_h, tile_w, repaired, bad_tiles[0]))

    candidates.sort(key=lambda item: item[0])
    return candidates


def _solve(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    candidates = _valid_candidates(grid)
    if not candidates:
        raise ValueError("no valid periodic tiling found")
    _score, tile_h, tile_w, repaired, _bad_tile = candidates[0]
    return repaired.copy(), (tile_h, tile_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a periodic tiled grid by replacing the unique corrupted tile with the majority pattern.
    """
    arr = np.asarray(grid, dtype=int)
    if arr.ndim != 2:
        raise ValueError("expected a 2D grid")
    repaired, _period = _solve(arr)
    return repaired


def _fundamental_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    best = (h * w, h, w)
    best_dims = (h, w)
    for tile_h in _divisors(h):
        for tile_w in _divisors(w):
            base = tile[:tile_h, :tile_w]
            repeated = np.tile(base, (h // tile_h, w // tile_w))
            if np.array_equal(repeated, tile):
                score = (tile_h * tile_w, tile_h, tile_w)
                if score < best:
                    best = score
                    best_dims = (tile_h, tile_w)
    return best_dims


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
TILE_HEIGHT_RANGE = (1, 6)
TILE_WIDTH_RANGE = (1, 6)
REPEAT_RANGE = (2, 5)
GRID_SIDE_MAX = 30
PATTERN_MODES = ("rectangle", "bars", "cross", "diag", "ring", "walk", "corners", "dots")
ANOMALY_MODES = ("recolor", "erase", "mixed")


def _paint_rectangle(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    rh = random.randint(1, h)
    rw = random.randint(1, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    tile[r0 : r0 + rh, c0 : c0 + rw] = color


def _paint_bars(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    if random.choice((True, False)):
        step = random.choice((1, 2))
        offset = random.randrange(step)
        rows = list(range(offset, h, step)) or [random.randrange(h)]
        c0 = random.randint(0, w - 1)
        c1 = random.randint(c0, w - 1)
        tile[rows, c0 : c1 + 1] = color
    else:
        step = random.choice((1, 2))
        offset = random.randrange(step)
        cols = list(range(offset, w, step)) or [random.randrange(w)]
        r0 = random.randint(0, h - 1)
        r1 = random.randint(r0, h - 1)
        tile[r0 : r1 + 1, cols] = color


def _paint_cross(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    r = random.randrange(h)
    c = random.randrange(w)
    c0 = random.randint(0, c)
    c1 = random.randint(c, w - 1)
    r0 = random.randint(0, r)
    r1 = random.randint(r, h - 1)
    tile[r, c0 : c1 + 1] = color
    tile[r0 : r1 + 1, c] = color


def _paint_diag(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    if h == 1:
        c0 = random.randint(0, w - 1)
        c1 = random.randint(c0, w - 1)
        tile[0, c0 : c1 + 1] = color
        return

    reverse = random.choice((False, True))
    cols = np.linspace(0, w - 1, h).round().astype(int).tolist()
    for r in range(h):
        c = cols[h - 1 - r] if reverse else cols[r]
        tile[r, c] = color
        if random.random() < 0.3 and c + 1 < w:
            tile[r, c + 1] = color
        if random.random() < 0.2 and c - 1 >= 0:
            tile[r, c - 1] = color


def _paint_ring(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    if h < 2 or w < 2:
        _paint_rectangle(tile, color)
        return
    inset_max = min(h, w) // 2 - 1
    inset = random.randint(0, max(0, inset_max))
    r0, r1 = inset, h - 1 - inset
    c0, c1 = inset, w - 1 - inset
    tile[r0, c0 : c1 + 1] = color
    tile[r1, c0 : c1 + 1] = color
    tile[r0 : r1 + 1, c0] = color
    tile[r0 : r1 + 1, c1] = color


def _paint_walk(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    steps = random.randint(max(2, (h * w) // 3), max(2, h * w))
    r = random.randrange(h)
    c = random.randrange(w)
    tile[r, c] = color
    for _ in range(steps):
        dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
        nr = min(max(r + dr, 0), h - 1)
        nc = min(max(c + dc, 0), w - 1)
        r, c = nr, nc
        tile[r, c] = color


def _paint_corners(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    random.shuffle(corners)
    count = random.randint(1, min(4, len(corners)))
    for r, c in corners[:count]:
        rh = random.randint(1, min(2, h))
        rw = random.randint(1, min(2, w))
        r0 = 0 if r == 0 else h - rh
        c0 = 0 if c == 0 else w - rw
        tile[r0 : r0 + rh, c0 : c0 + rw] = color


def _paint_dots(tile: np.ndarray, color: int) -> None:
    h, w = tile.shape
    count = random.randint(1, max(1, (h * w) // 3))
    picks = random.sample(range(h * w), k=min(count, h * w))
    for index in picks:
        r, c = divmod(index, w)
        tile[r, c] = color


def _apply_mode(tile: np.ndarray, mode: str, color: int) -> None:
    if mode == "rectangle":
        _paint_rectangle(tile, color)
    elif mode == "bars":
        _paint_bars(tile, color)
    elif mode == "cross":
        _paint_cross(tile, color)
    elif mode == "diag":
        _paint_diag(tile, color)
    elif mode == "ring":
        _paint_ring(tile, color)
    elif mode == "walk":
        _paint_walk(tile, color)
    elif mode == "corners":
        _paint_corners(tile, color)
    elif mode == "dots":
        _paint_dots(tile, color)
    else:
        raise ValueError(f"unknown mode: {mode}")


def _break_periodicity(tile: np.ndarray, palette: list[int]) -> None:
    h, w = tile.shape
    choices = [BACKGROUND] + palette
    for _ in range(40):
        if _fundamental_period(tile) == tile.shape and len(np.unique(tile)) >= 2:
            return
        if random.random() < 0.7:
            r = random.choice((0, h - 1)) if h > 1 else 0
            c = random.choice((0, w - 1)) if w > 1 else 0
        else:
            r = random.randrange(h)
            c = random.randrange(w)
        current = int(tile[r, c])
        alternatives = [value for value in choices if value != current]
        tile[r, c] = random.choice(alternatives)


def _build_tile(tile_h: int, tile_w: int, palette: list[int]) -> np.ndarray:
    for _ in range(120):
        tile = np.full((tile_h, tile_w), BACKGROUND, dtype=int)
        layer_count = random.randint(2, 4)
        for _layer in range(layer_count):
            mode = random.choice(PATTERN_MODES)
            color = random.choice(palette)
            _apply_mode(tile, mode, color)

        if random.random() < 0.4:
            _paint_rectangle(tile, BACKGROUND)

        _break_periodicity(tile, palette)

        unique = set(map(int, np.unique(tile)))
        density = float(np.mean(tile != BACKGROUND))
        if len(unique) < 2:
            continue
        if density < 0.15 or density > 0.85:
            continue
        if _fundamental_period(tile) != tile.shape:
            continue
        return tile

    raise RuntimeError("failed to build aperiodic tile")


def _mutate_tile(tile: np.ndarray, palette: list[int], mode: str) -> np.ndarray:
    h, w = tile.shape
    changed = tile.copy()
    all_positions = [(r, c) for r in range(h) for c in range(w)]
    non_bg_positions = [(r, c) for r, c in all_positions if int(tile[r, c]) != BACKGROUND]
    max_changes = max(1, min(h * w // 2, 6))

    if mode == "erase" and not non_bg_positions:
        mode = "recolor"

    if mode == "erase":
        count = random.randint(1, min(max_changes, len(non_bg_positions)))
        picks = random.sample(non_bg_positions, count)
        for r, c in picks:
            changed[r, c] = BACKGROUND
        return changed

    color_choices = [BACKGROUND] + palette + [value for value in COLOR_POOL if value not in palette]
    count = random.randint(1, max_changes)
    picks = random.sample(all_positions, count)
    for r, c in picks:
        current = int(changed[r, c])
        if mode == "mixed" and random.random() < 0.5 and current != BACKGROUND:
            changed[r, c] = BACKGROUND
            continue
        alternatives = [value for value in color_choices if value != current]
        changed[r, c] = random.choice(alternatives)

    return changed


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    tile_h = random.randint(*TILE_HEIGHT_RANGE)
    tile_w = random.randint(*TILE_WIDTH_RANGE)
    if tile_h * tile_w == 1:
        return None

    repeat_h = random.randint(*REPEAT_RANGE)
    repeat_w = random.randint(*REPEAT_RANGE)
    if tile_h * repeat_h > GRID_SIDE_MAX or tile_w * repeat_w > GRID_SIDE_MAX:
        return None

    palette_size = random.randint(2, 4)
    palette = random.sample(COLOR_POOL, palette_size)
    tile = _build_tile(tile_h, tile_w, palette)

    clean = np.tile(tile, (repeat_h, repeat_w))
    bad_r = random.randrange(repeat_h)
    bad_c = random.randrange(repeat_w)

    feasible_modes = list(ANOMALY_MODES)
    if not np.any(tile != BACKGROUND) and "erase" in feasible_modes:
        feasible_modes.remove("erase")
    mode = random.choice(feasible_modes)
    corrupted_tile = _mutate_tile(tile, palette, mode)
    if np.array_equal(corrupted_tile, tile):
        return None

    corrupted = clean.copy()
    r0 = bad_r * tile_h
    c0 = bad_c * tile_w
    corrupted[r0 : r0 + tile_h, c0 : c0 + tile_w] = corrupted_tile

    repaired, inferred_period = _solve(corrupted)
    if not np.array_equal(repaired, clean):
        return None
    if inferred_period != (tile_h, tile_w):
        return None

    candidates = _valid_candidates(corrupted)
    if not candidates:
        return None
    best_area = candidates[0][0][0]
    same_area = [(th, tw) for _score, th, tw, _rep, _bad in candidates if _score[0] == best_area]
    if same_area != [(tile_h, tile_w)]:
        return None

    return corrupted, clean, inferred_period


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            grid, _clean, _period = result
            return grid
    raise RuntimeError("failed to generate a valid periodic-tile anomaly input")
