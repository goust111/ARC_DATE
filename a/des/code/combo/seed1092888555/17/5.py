import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 8
SEP = 5
DIGIT = 2
PANEL_SIDE = 11
ZONE_SIDE = 5
OBJECT_COLORS = (1, 3, 4, 6, 7, 9)
ODD_DIGITS = (1, 3, 5)
EVEN_DIGITS = (2, 4)
GEN_ATTEMPTS = 300

FONT: Dict[int, np.ndarray] = {
    1: np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    4: np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    5: np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
}


def _intervals_from_separators(seps: Sequence[int], size: int) -> List[Tuple[int, int]]:
    seps = sorted(int(x) for x in seps)
    starts = [0] + [s + 1 for s in seps]
    ends = seps + [size]
    intervals = [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]
    assert intervals
    return intervals


def _glyph_digit(panel: np.ndarray) -> int:
    top = panel[:ZONE_SIDE, :]
    red = np.argwhere(top == DIGIT)
    assert red.size > 0, "digit pixels must exist in the top half"
    r0, c0 = red.min(axis=0)
    r1, c1 = red.max(axis=0)
    glyph = (top[r0 : r1 + 1, c0 : c1 + 1] == DIGIT).astype(int)
    for value, pattern in FONT.items():
        if glyph.shape == pattern.shape and np.array_equal(glyph, pattern):
            return value
    raise AssertionError("unrecognized digit glyph")


def _transform_zone(zone: np.ndarray, value: int) -> np.ndarray:
    if value % 2 == 0:
        return np.fliplr(zone)
    return np.rot90(zone, -1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    row_seps = [r for r in range(h) if np.all(g[r, :] == SEP)]
    col_seps = [c for c in range(w) if np.all(g[:, c] == SEP)]
    assert row_seps and col_seps

    row_intervals = _intervals_from_separators(row_seps, h)
    col_intervals = _intervals_from_separators(col_seps, w)
    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and row_sizes == col_sizes
    side = next(iter(row_sizes))
    assert side >= PANEL_SIDE and side % 2 == 1

    zone_col0 = (side - ZONE_SIDE) // 2
    zone_col1 = zone_col0 + ZONE_SIDE
    out = np.full_like(g, BG)
    if row_seps:
        out[row_seps, :] = SEP
    if col_seps:
        out[:, col_seps] = SEP

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            assert panel.shape == (side, side)
            value = _glyph_digit(panel)
            zone = panel[side - ZONE_SIDE : side, zone_col0:zone_col1]
            out_zone = _transform_zone(zone, value)
            out[r0 + side - ZONE_SIDE : r1, c0 + zone_col0 : c0 + zone_col1] = out_zone

    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        target = rng.randint(4, 9)
        cells = {(rng.randrange(ZONE_SIDE), rng.randrange(ZONE_SIDE))}
        while len(cells) < target:
            r, c = rng.choice(tuple(cells))
            dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            nr, nc = r + dr, c + dc
            if 0 <= nr < ZONE_SIDE and 0 <= nc < ZONE_SIDE:
                cells.add((nr, nc))
        coords = np.array(sorted(cells))
        r_span = int(coords[:, 0].max() - coords[:, 0].min() + 1)
        c_span = int(coords[:, 1].max() - coords[:, 1].min() + 1)
        if r_span < 2 or c_span < 2:
            continue
        mask = np.zeros((ZONE_SIDE, ZONE_SIDE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to build a connected mask")


def _random_zone(rng: random.Random, digit_value: int) -> np.ndarray:
    for _ in range(120):
        mask = _random_connected_mask(rng)
        zone = np.full((ZONE_SIDE, ZONE_SIDE), BG, dtype=int)
        color_count = rng.randint(1, min(3, len(OBJECT_COLORS)))
        colors = rng.sample(OBJECT_COLORS, color_count)
        coords = list(map(tuple, np.argwhere(mask)))
        rng.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            if color_count == 1:
                zone[r, c] = colors[0]
            else:
                zone[r, c] = colors[(idx + r + 2 * c) % color_count]
        if np.array_equal(zone, _transform_zone(zone, digit_value)):
            continue
        return zone
    raise RuntimeError("failed to build a non-degenerate object zone")


def _digit_sequence(count: int, rng: random.Random) -> List[int]:
    parities = []
    odd_turn = rng.choice((True, False))
    while len(parities) < count:
        parities.append(odd_turn)
        odd_turn = not odd_turn
    rng.shuffle(parities)
    digits = []
    for is_odd in parities:
        pool = ODD_DIGITS if is_odd else EVEN_DIGITS
        digits.append(rng.choice(pool))
    return digits


def _draw_panel(value: int, zone: np.ndarray) -> np.ndarray:
    panel = np.full((PANEL_SIDE, PANEL_SIDE), BG, dtype=int)
    glyph = FONT[value]
    glyph_c0 = PANEL_SIDE // 2 - 1
    panel[:ZONE_SIDE, glyph_c0 : glyph_c0 + 3][glyph == 1] = DIGIT
    zone_c0 = (PANEL_SIDE - ZONE_SIDE) // 2
    panel[PANEL_SIDE - ZONE_SIDE :, zone_c0 : zone_c0 + ZONE_SIDE] = zone
    return panel


def _compose_grid(rows: int, cols: int, digits: Sequence[int], zones: Sequence[np.ndarray]) -> np.ndarray:
    height = rows * PANEL_SIDE + (rows - 1)
    width = cols * PANEL_SIDE + (cols - 1)
    grid = np.full((height, width), BG, dtype=int)
    for r in range(1, rows):
        grid[r * PANEL_SIDE + (r - 1), :] = SEP
    for c in range(1, cols):
        grid[:, c * PANEL_SIDE + (c - 1)] = SEP

    idx = 0
    for pr in range(rows):
        for pc in range(cols):
            r0 = pr * (PANEL_SIDE + 1)
            c0 = pc * (PANEL_SIDE + 1)
            grid[r0 : r0 + PANEL_SIDE, c0 : c0 + PANEL_SIDE] = _draw_panel(digits[idx], zones[idx])
            idx += 1
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    count = rows * cols
    digits = _digit_sequence(count, rng)
    if count > 1:
        assert any(d % 2 == 0 for d in digits)
        assert any(d % 2 == 1 for d in digits)
    zones = [_random_zone(rng, value) for value in digits]
    return _compose_grid(rows, cols, digits, zones)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        try:
            out = main(grid)
        except AssertionError:
            continue
        assert isinstance(out, np.ndarray)
        return grid
    raise RuntimeError("failed to generate valid input")
