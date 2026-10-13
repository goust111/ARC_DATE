import math
import random

import numpy as np


BG = 0
FG_COLORS = (2, 3)
OPPOSITE = {2: 3, 3: 2}


def _foreground_tile_for_period(
    coords: np.ndarray, colors: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    for (r, c), color in zip(coords, colors, strict=False):
        key = (int(r) % row_period, int(c) % col_period)
        color_int = int(color)
        if key in tile and tile[key] != color_int:
            return None
        tile[key] = color_int
    return tile


def _infer_period_and_tile(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))) <= {0, 2, 3}, "only black/red/green are supported"

    coords = np.argwhere(g != BG)
    assert coords.size > 0, "expected visible foreground evidence"
    colors = g[g != BG]

    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    height, width = g.shape
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            tile = _foreground_tile_for_period(coords, colors, row_period, col_period)
            if tile is not None:
                candidates.append((row_period, col_period, tile))

    assert candidates, "expected at least one consistent period"
    min_area = min(row_period * col_period for row_period, col_period, _tile in candidates)
    best = [
        candidate
        for candidate in candidates
        if candidate[0] * candidate[1] == min_area
    ]
    assert len(best) == 1, "smallest consistent period must be unique"
    return best[0]


def _render_periodic_grid(height: int, width: int, tile: np.ndarray) -> np.ndarray:
    rows = np.arange(height) % tile.shape[0]
    cols = np.arange(width) % tile.shape[1]
    return tile[rows[:, None], cols[None, :]].astype(int)


def _truth_tile_dict(tile: np.ndarray) -> dict[tuple[int, int], int]:
    truth: dict[tuple[int, int], int] = {}
    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            color = int(tile[r, c])
            if color != BG:
                truth[(r, c)] = color
    return truth


def _apply_opposite_fill(full: np.ndarray, erased_mask: np.ndarray) -> np.ndarray:
    expected = full.copy()
    expected[erased_mask & (full == 2)] = 3
    expected[erased_mask & (full == 3)] = 2
    return expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the smallest two-color wallpaper period, then restore erased cells using the opposite foreground color.

    Input structure:
    - A black background grid contains a periodic wallpaper made from red (2) and green (3).
    - Some wallpaper cells have been erased to black (0), while true background residue classes also stay black.
    - Every foreground residue class keeps at least one visible pixel, so the wallpaper period is recoverable from the remaining evidence.

    Rule:
    - Ignore black cells and infer the unique smallest row/column period whose residue classes give every visible foreground pixel one consistent color.
    - Residue classes with visible evidence are foreground classes colored red or green; classes with no evidence are background classes.
    - Keep every already-visible foreground pixel unchanged, but for every erased black cell belonging to a foreground residue class, fill it with the opposite foreground color (red becomes green, green becomes red).

    Key insight:
    - The visible pixels identify where the periodic wallpaper should continue, but restored cells deliberately use the other foreground color, so the task is not ordinary periodic completion.
    """
    g = np.asarray(grid, dtype=int)
    row_period, col_period, tile = _infer_period_and_tile(g)

    height, width = g.shape
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    visible = g != BG

    out = np.zeros_like(g)
    for (tile_r, tile_c), color in tile.items():
        residue_mask = (rows % row_period == tile_r) & (cols % col_period == tile_c)
        if np.any(visible & residue_mask):
            assert np.all(g[visible & residue_mask] == color), "visible colors must agree within a residue class"
        out[residue_mask] = OPPOSITE[color]

    out[visible] = g[visible]
    return out


# --------------- generator ---------------

GRID_SIZE_MIN = 11
GRID_SIZE_MAX = 17
PERIOD_MIN = 3
PERIOD_MAX = 5
ERASE_FRAC_MIN = 0.20
ERASE_FRAC_MAX = 0.40
MAX_GENERATION_ATTEMPTS = 600
REQUIRE_BG_PROB = 0.7
_GEN_CALL_COUNT = 0


def _tile_is_informative(tile: np.ndarray, require_bg: bool) -> bool:
    fg = tile != BG
    fg_count = int(np.count_nonzero(fg))
    count_red = int(np.count_nonzero(tile == 2))
    count_green = int(np.count_nonzero(tile == 3))

    if fg_count < max(6, tile.shape[0] + tile.shape[1] - 1):
        return False
    if count_red < 2 or count_green < 2:
        return False
    if require_bg and not bool(np.any(tile == BG)):
        return False
    if not np.all(np.any(fg, axis=0)) or not np.all(np.any(fg, axis=1)):
        return False

    row_has_both = any({2, 3} <= set(map(int, row)) for row in tile)
    col_has_both = any({2, 3} <= set(map(int, col)) for col in tile.T)
    if not row_has_both or not col_has_both:
        return False

    has_same_adj = False
    has_diff_adj = False
    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            cur = int(tile[r, c])
            if cur == BG:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr = r + dr
                nc = c + dc
                if nr >= tile.shape[0] or nc >= tile.shape[1]:
                    continue
                nxt = int(tile[nr, nc])
                if nxt == BG:
                    continue
                if nxt == cur:
                    has_same_adj = True
                else:
                    has_diff_adj = True
    return has_same_adj and has_diff_adj


def _random_tile(row_period: int, col_period: int, require_bg: bool) -> np.ndarray | None:
    for _ in range(250):
        if require_bg:
            bg_prob = random.uniform(0.18, 0.38)
        else:
            bg_prob = random.uniform(0.00, 0.14)

        tile = np.zeros((row_period, col_period), dtype=int)
        for r in range(row_period):
            for c in range(col_period):
                if random.random() < bg_prob:
                    tile[r, c] = BG
                else:
                    tile[r, c] = random.choice(FG_COLORS)

        if not _tile_is_informative(tile, require_bg):
            continue
        return tile
    return None


def _pick_periods(profile: int) -> tuple[int, int]:
    pairs = [
        (row_period, col_period)
        for row_period in range(PERIOD_MIN, PERIOD_MAX + 1)
        for col_period in range(PERIOD_MIN, PERIOD_MAX + 1)
    ]
    if profile == 0:
        pairs = [pair for pair in pairs if pair[0] != pair[1]]
    return random.choice(pairs)


def _pick_size(row_period: int, col_period: int) -> tuple[int, int]:
    min_height = max(GRID_SIZE_MIN, 2 * row_period + 1)
    min_width = max(GRID_SIZE_MIN, 2 * col_period + 1)
    height = random.randint(min_height, GRID_SIZE_MAX)
    width = random.randint(min_width, GRID_SIZE_MAX)
    return height, width


def _choose_erased_mask(full: np.ndarray, tile: np.ndarray) -> np.ndarray | None:
    positions_by_class: dict[tuple[int, int], list[tuple[int, int]]] = {}
    colors_by_class: dict[tuple[int, int], int] = {}

    for r in range(full.shape[0]):
        for c in range(full.shape[1]):
            color = int(full[r, c])
            if color == BG:
                continue
            key = (r % tile.shape[0], c % tile.shape[1])
            positions_by_class.setdefault(key, []).append((r, c))
            colors_by_class[key] = color

    if not positions_by_class:
        return None

    capacities = {key: len(pos_list) - 1 for key, pos_list in positions_by_class.items()}
    eligible_by_color = {
        color: [key for key, klass_color in colors_by_class.items() if klass_color == color and capacities[key] > 0]
        for color in FG_COLORS
    }
    if not eligible_by_color[2] or not eligible_by_color[3]:
        return None

    total_fg = int(np.count_nonzero(full))
    min_erase = max(2, math.ceil(total_fg * ERASE_FRAC_MIN))
    max_erase = min(sum(capacities.values()), math.floor(total_fg * ERASE_FRAC_MAX))
    if min_erase > max_erase:
        return None

    target_erase = random.randint(min_erase, max_erase)
    erase_counts = {key: 0 for key in positions_by_class}

    for color in FG_COLORS:
        key = random.choice(eligible_by_color[color])
        erase_counts[key] += 1

    remaining = target_erase - len(FG_COLORS)
    class_keys = list(positions_by_class)
    while remaining > 0:
        options = [key for key in class_keys if erase_counts[key] < capacities[key]]
        if not options:
            return None
        key = random.choice(options)
        erase_counts[key] += 1
        remaining -= 1

    erased_mask = np.zeros_like(full, dtype=bool)
    for key, erase_count in erase_counts.items():
        if erase_count == 0:
            continue
        for r, c in random.sample(positions_by_class[key], erase_count):
            erased_mask[r, c] = True
    return erased_mask


def _semantic_checks(
    inp: np.ndarray,
    full: np.ndarray,
    expected: np.ndarray,
    row_period: int,
    col_period: int,
    tile: np.ndarray,
) -> bool:
    if inp.shape != expected.shape or inp.shape != full.shape:
        return False

    erased_mask = (inp == BG) & (full != BG)
    if not bool(np.any(erased_mask & (full == 2))):
        return False
    if not bool(np.any(erased_mask & (full == 3))):
        return False
    if not np.array_equal(inp[~erased_mask], full[~erased_mask]):
        return False
    if not np.array_equal(expected, _apply_opposite_fill(full, erased_mask)):
        return False

    try:
        inferred_row_period, inferred_col_period, inferred_tile = _infer_period_and_tile(inp)
    except AssertionError:
        return False

    if (inferred_row_period, inferred_col_period) != (row_period, col_period):
        return False
    if inferred_tile != _truth_tile_dict(tile):
        return False
    if not np.array_equal(expected[erased_mask & (full == 2)], np.full(int(np.sum(erased_mask & (full == 2))), 3)):
        return False
    if not np.array_equal(expected[erased_mask & (full == 3)], np.full(int(np.sum(erased_mask & (full == 3))), 2)):
        return False
    if np.array_equal(expected, full):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt(profile: int) -> np.ndarray | None:
    require_bg = profile != 1 or random.random() < REQUIRE_BG_PROB
    row_period, col_period = _pick_periods(profile)
    height, width = _pick_size(row_period, col_period)

    tile = _random_tile(row_period, col_period, require_bg)
    if tile is None:
        return None

    full = _render_periodic_grid(height, width, tile)
    try:
        full_row_period, full_col_period, full_tile = _infer_period_and_tile(full)
    except AssertionError:
        return None

    if (full_row_period, full_col_period) != (row_period, col_period):
        return None
    if full_tile != _truth_tile_dict(tile):
        return None

    erased_mask = _choose_erased_mask(full, tile)
    if erased_mask is None:
        return None

    inp = full.copy()
    inp[erased_mask] = BG
    expected = _apply_opposite_fill(full, erased_mask)
    if _semantic_checks(inp, full, expected, row_period, col_period, tile):
        return inp
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    profile = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic opposite-fill wallpaper input")
