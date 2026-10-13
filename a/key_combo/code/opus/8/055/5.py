from __future__ import annotations

from collections import Counter
import random

import numpy as np


PLACEHOLDER = 5
SHIFT_BY_MARKER = {
    2: ("right", (0, 1)),
    1: ("down", (1, 0)),
    3: ("left", (0, -1)),
    4: ("up", (-1, 0)),
}
MARKER_COLORS = [2, 1, 3, 4]
TILE_SHAPES = [(3, 3), (3, 4), (4, 3), (4, 4)]
GEN_RETRIES = 400
_GEN_COUNTER = 0


def _placeholder_suffix(grid: np.ndarray) -> tuple[int, int, int]:
    """Return (placeholder_color, visible_row_cut, visible_col_cut) for the L suffix."""
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        rows_all = np.all(grid == color, axis=1)
        cols_all = np.all(grid == color, axis=0)
        if not rows_all[-1] or not cols_all[-1]:
            continue

        r_cut = h
        while r_cut > 0 and rows_all[r_cut - 1]:
            r_cut -= 1
        c_cut = w
        while c_cut > 0 and cols_all[c_cut - 1]:
            c_cut -= 1
        if r_cut == 0 or c_cut == 0:
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[r_cut:, :] = True
        mask[:, c_cut:] = True
        if np.array_equal(grid == color, mask):
            candidates.append((color, r_cut, c_cut))

    assert len(candidates) == 1, "expected exactly one placeholder L suffix"
    return candidates[0]


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0])[:, None] % tile.shape[0]
    cols = np.arange(shape[1])[None, :] % tile.shape[1]
    return tile[rows, cols]


def _exact_periods(block: np.ndarray) -> tuple[int, int]:
    h, w = block.shape

    row_period = None
    for period in range(1, h + 1):
        if all(block[r, c] == block[r % period, c] for r in range(h) for c in range(w)):
            row_period = period
            break

    col_period = None
    for period in range(1, w + 1):
        if all(block[r, c] == block[r, c % period] for r in range(h) for c in range(w)):
            col_period = period
            break

    assert row_period is not None and col_period is not None
    return int(row_period), int(col_period)


def _period_candidate(
    block: np.ndarray, period_r: int, period_c: int
) -> tuple[np.ndarray, tuple[int, int], int] | None:
    h, w = block.shape
    tile = np.empty((period_r, period_c), dtype=int)
    total_mismatches = 0

    for rr in range(period_r):
        for cc in range(period_c):
            values = block[rr:h:period_r, cc:w:period_c].reshape(-1)
            counts = Counter(map(int, values))
            color, freq = counts.most_common(1)[0]
            total_mismatches += len(values) - freq
            if total_mismatches > 1:
                return None
            tile[rr, cc] = int(color)

    if total_mismatches != 1:
        return None

    completed = _tile_grid(tile, block.shape)
    mismatches = np.argwhere(block != completed)
    if mismatches.shape != (1, 2):
        return None
    bad_r, bad_c = map(int, mismatches[0])
    return tile, (bad_r, bad_c), int(completed[bad_r, bad_c])


def _all_period_candidates(
    block: np.ndarray,
) -> list[tuple[int, int, np.ndarray, tuple[int, int], int]]:
    h, w = block.shape
    candidates: list[tuple[int, int, np.ndarray, tuple[int, int], int]] = []
    for period_r in range(1, h):
        for period_c in range(1, w):
            result = _period_candidate(block, period_r, period_c)
            if result is None:
                continue
            tile, marker_pos, expected_color = result
            candidates.append((period_r, period_c, tile, marker_pos, expected_color))
    return candidates


def _infer_tile_and_marker(block: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    candidates = _all_period_candidates(block)
    assert candidates, "expected at least one valid period candidate"
    candidates.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    best_r, best_c, tile, marker_pos, expected_color = candidates[0]
    assert tile.shape == (best_r, best_c)
    return tile, marker_pos, expected_color


def _decode_shift(marker_color: int, marker_pos: tuple[int, int], tile_shape: tuple[int, int]) -> tuple[int, int]:
    row, col = marker_pos
    tile_h, tile_w = tile_shape
    direction, _ = SHIFT_BY_MARKER[marker_color]

    if direction in {"left", "right"}:
        amount = col % tile_w
        if amount == 0:
            amount = tile_w
        return (0, amount if direction == "right" else -amount)

    amount = row % tile_h
    if amount == 0:
        amount = tile_h
    return (amount if direction == "down" else -amount, 0)


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the hidden periodic tile, decode the embedded anomaly, and circularly shift the finished pattern.

    Input structure:
    - A grey placeholder L-suffix covers the bottom rows and right columns.
    - The visible top-left block is a periodic tiling of a small 3x3/3x4/4x3/4x4 motif made of 2-3 non-grey colors.
    - Exactly one visible cell is an off-pattern marker whose color encodes shift direction.

    Rule:
    - Ignore the anomalous marker cell and infer the smallest repeating tile that explains the visible block.
    - Periodically extend that tile over the whole grid, replacing every grey placeholder cell.
    - Decode the marker: red=right, blue=down, green=left, yellow=up; horizontal moves use the marker's tile-column offset, vertical moves use its tile-row offset, and offset 0 means one full tile period.
    - Circularly shift the completed full grid by the decoded amount and return the shifted pattern.

    Key insight:
    - The unique wrong-color pixel is both noise for tile inference and a two-part code: its color gives the direction while its position inside the repeated tile gives the magnitude.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _placeholder, r_cut, c_cut = _placeholder_suffix(g)
    visible = g[:r_cut, :c_cut]
    tile, marker_pos, expected_color = _infer_tile_and_marker(visible)

    marker_r, marker_c = marker_pos
    marker_color = int(visible[marker_r, marker_c])
    assert marker_color in SHIFT_BY_MARKER, "marker color must decode a direction"
    assert marker_color != expected_color, "marker must differ from the periodic color"

    completed = _tile_grid(tile, g.shape)
    shift = _decode_shift(marker_color, marker_pos, tile.shape)
    return np.roll(completed, shift=shift, axis=(0, 1))


def _sample_tile(tile_h: int, tile_w: int, marker_color: int) -> np.ndarray | None:
    palette = [color for color in range(1, 10) if color not in {PLACEHOLDER, marker_color}]
    n_colors = 2 if random.random() < 0.55 else 3
    colors = random.sample(palette, n_colors)

    tile = np.array(
        [[random.choice(colors) for _ in range(tile_w)] for _ in range(tile_h)],
        dtype=int,
    )
    if len(set(map(int, tile.flat))) != n_colors:
        return None
    if len({tuple(map(int, row)) for row in tile}) == 1:
        return None
    if len({tuple(map(int, tile[:, c])) for c in range(tile_w)}) == 1:
        return None
    if _exact_periods(tile) != (tile_h, tile_w):
        return None
    return tile


def _marker_position(
    r_cut: int,
    c_cut: int,
    tile_h: int,
    tile_w: int,
    marker_color: int,
) -> tuple[int, int]:
    direction, _ = SHIFT_BY_MARKER[marker_color]
    if direction in {"left", "right"}:
        offset_c = random.randint(1, tile_w - 1)
        offset_r = random.randint(0, tile_h - 1)
    else:
        offset_r = random.randint(1, tile_h - 1)
        offset_c = random.randint(0, tile_w - 1)

    row_choices = list(range(offset_r, r_cut, tile_h))
    col_choices = list(range(offset_c, c_cut, tile_w))
    return random.choice(row_choices), random.choice(col_choices)


def _minimal_candidate_shape(block: np.ndarray) -> tuple[int, int]:
    candidates = _all_period_candidates(block)
    assert candidates
    best_area = min(period_r * period_c for period_r, period_c, *_ in candidates)
    best_shapes = sorted(
        (period_r, period_c)
        for period_r, period_c, *_ in candidates
        if period_r * period_c == best_area
    )
    assert len(best_shapes) == 1, "minimal noisy period must be unique"
    return best_shapes[0]


def _validate_instance(
    inp: np.ndarray,
    tile: np.ndarray,
    marker_pos: tuple[int, int],
    marker_color: int,
    expected_out: np.ndarray,
) -> bool:
    try:
        placeholder, r_cut, c_cut = _placeholder_suffix(inp)
        assert placeholder == PLACEHOLDER
        assert np.count_nonzero(inp == marker_color) == 1

        completed = _tile_grid(tile, inp.shape)
        visible = inp[:r_cut, :c_cut]
        mismatches = np.argwhere(visible != completed[:r_cut, :c_cut])
        assert mismatches.shape == (1, 2)
        assert tuple(map(int, mismatches[0])) == marker_pos
        assert _exact_periods(tile) == tile.shape
        assert _minimal_candidate_shape(visible) == tile.shape

        shift = _decode_shift(marker_color, marker_pos, tile.shape)
        assert shift != (0, 0)
        assert not np.array_equal(np.roll(completed, shift=shift, axis=(0, 1)), completed)
        assert np.array_equal(expected_out, np.roll(completed, shift=shift, axis=(0, 1)))
        assert np.array_equal(main(inp), expected_out)
        return True
    except AssertionError:
        return False


def _gen_attempt(counter: int) -> np.ndarray | None:
    marker_color = MARKER_COLORS[counter % len(MARKER_COLORS)]
    tile_h, tile_w = TILE_SHAPES[(counter // len(MARKER_COLORS)) % len(TILE_SHAPES)]

    tile = _sample_tile(tile_h, tile_w, marker_color)
    if tile is None:
        return None

    visible_reps_r = random.randint(2, 3)
    visible_reps_c = random.randint(2, 3)
    visible_tail_r = random.randint(1, tile_h - 1)
    visible_tail_c = random.randint(1, tile_w - 1)
    r_cut = visible_reps_r * tile_h + visible_tail_r
    c_cut = visible_reps_c * tile_w + visible_tail_c

    suffix_rows = random.randint(1, tile_h)
    suffix_cols = random.randint(1, tile_w)
    height = r_cut + suffix_rows
    width = c_cut + suffix_cols
    if max(height, width) > 30:
        return None

    completed = _tile_grid(tile, (height, width))
    inp = np.full((height, width), PLACEHOLDER, dtype=int)
    inp[:r_cut, :c_cut] = completed[:r_cut, :c_cut]

    marker_r, marker_c = _marker_position(r_cut, c_cut, tile_h, tile_w, marker_color)
    true_color = int(completed[marker_r, marker_c])
    if true_color == marker_color:
        return None
    inp[marker_r, marker_c] = marker_color

    expected_out = np.roll(
        completed,
        shift=_decode_shift(marker_color, (marker_r, marker_c), tile.shape),
        axis=(0, 1),
    )
    if np.array_equal(expected_out, completed):
        return None
    if not _validate_instance(inp, tile, (marker_r, marker_c), marker_color, expected_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_GEN_COUNTER)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid hidden-marker tile-shift instance")
