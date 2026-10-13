import hashlib
import random

import numpy as np


BG = 0
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

GRID_WIDTH = 15
DIVIDER_COL = 6
LEFT_REGION_END = DIVIDER_COL
RIGHT_REGION_START = DIVIDER_COL + 1
SOURCE_BAND_WIDTH = 2
BAR_POSITION_OPTIONS = [
    (0, 2, 4),
    (0, 2, 5),
    (0, 3, 5),
    (1, 3, 5),
]
HEIGHT_RANGE = (10, 18)
BAR_HEIGHT_MIN = 1
BAR_HEIGHT_MAX = 8
SHIFT_VALUES = [1, 2, 3, 4, 5, 6]
GEN_ATTEMPTS = 400

_GEN_CYCLES: dict[str, list] = {}
_SEEN_INPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _cycled_choice(name: str, options):
    pool = _GEN_CYCLES.setdefault(name, [])
    if not pool:
        pool[:] = list(options)
        random.shuffle(pool)
    return pool.pop()


def _grid_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _find_divider_column(grid: np.ndarray) -> int:
    matches = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(matches) == 1, "expected exactly one full-height grey divider"
    return matches[0]


def _extract_bars(grid: np.ndarray, divider_col: int) -> list[dict]:
    bars = []
    for c in range(divider_col):
        col = grid[:, c]
        rows = np.flatnonzero(col != BG)
        if rows.size == 0:
            continue
        color_values = set(int(v) for v in np.unique(col[rows]))
        assert len(color_values) == 1, "bar must be monochrome"
        color = color_values.pop()
        assert color in (TEAL, RED), "left-side foreground must be teal or red"
        top = int(rows[0])
        assert int(rows[-1]) == grid.shape[0] - 1, "bar must be bottom-aligned"
        assert rows.size == grid.shape[0] - top, "bar must be contiguous"
        bars.append({"col": c, "height": int(rows.size), "color": color})

    assert len(bars) >= 2, "expected multiple operand bars"
    bars.sort(key=lambda bar: bar["col"])
    for left, right in zip(bars, bars[1:]):
        assert right["col"] - left["col"] >= 2, "bars must be separated by black columns"
    return bars


def _compute_shift_from_bars(bars: list[dict]) -> int:
    shift = 0
    for bar in bars:
        if bar["color"] == TEAL:
            shift += bar["height"]
        else:
            shift -= bar["height"]
    return int(shift)


def _source_band(grid: np.ndarray) -> np.ndarray:
    band = np.asarray(grid[:, -SOURCE_BAND_WIDTH:], dtype=int)
    assert set(np.unique(band)).issubset({BG, ORANGE}), "edge band must use only orange on black"
    assert np.any(band == ORANGE), "edge band must contain orange"
    assert np.any(band == BG), "edge band may not be fully filled"
    assert not np.any(grid[:, :-SOURCE_BAND_WIDTH] == ORANGE), "input orange must be confined to the right edge band"
    return band


def _apply_shift(grid: np.ndarray, shift: int) -> np.ndarray:
    assert 1 <= shift <= 6
    out = np.array(grid, copy=True)
    band = _source_band(out)
    orange_rows, orange_cols = np.where(band == ORANGE)
    out[:, -SOURCE_BAND_WIDTH:] = BG
    for r, local_c in zip(orange_rows, orange_cols):
        src_c = out.shape[1] - SOURCE_BAND_WIDTH + int(local_c)
        dst_c = src_c - shift
        assert RIGHT_REGION_START <= dst_c < out.shape[1], "shifted band must stay in the right region"
        assert out[int(r), dst_c] == BG, "shift destination must be empty background"
        out[int(r), dst_c] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use signed bar heights to determine how far the right-edge orange band moves left.

    Input structure:
    - The background is black, with a full-height grey divider separating a left arithmetic zone from a right edge-band zone.
    - In the left zone, several one-cell-wide vertical bars are bottom-aligned; teal bars are positive and red bars are negative.
    - In the rightmost two columns, orange pixels form a sparse two-column vertical pattern band on black.

    Rule:
    - Compute D as the sum of teal bar heights minus the sum of red bar heights.
    - Translate every orange pixel of the right-edge two-column band left by exactly D columns.
    - Erase the original orange band positions while keeping the bars, the divider, and the orange/black band pattern unchanged apart from the translation.

    Key insight:
    - The left bars are only a calculator: their signed height arithmetic produces a distance parameter that controls a spatial movement in a separate region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider_col = _find_divider_column(g)
    bars = _extract_bars(g, divider_col)
    shift = _compute_shift_from_bars(bars)
    assert 1 <= shift <= 6, "signed height result must be a valid left-shift amount"
    return _apply_shift(g, shift)


def _height_candidates(colors: tuple[int, ...], target_shift: int) -> list[tuple[int, ...]]:
    candidates = []
    for h0 in range(BAR_HEIGHT_MIN, BAR_HEIGHT_MAX + 1):
        for h1 in range(BAR_HEIGHT_MIN, BAR_HEIGHT_MAX + 1):
            for h2 in range(BAR_HEIGHT_MIN, BAR_HEIGHT_MAX + 1):
                heights = (h0, h1, h2)
                shift = sum(h if color == TEAL else -h for h, color in zip(heights, colors))
                if shift != target_shift:
                    continue
                teal_heights = [h for h, color in zip(heights, colors) if color == TEAL]
                red_heights = [h for h, color in zip(heights, colors) if color == RED]
                if len(set(heights)) < 2:
                    continue
                if max(teal_heights) - max(red_heights) == target_shift:
                    continue
                if len(teal_heights) - len(red_heights) == target_shift:
                    continue
                candidates.append(heights)
    return candidates


def _choose_bar_specs(target_shift: int) -> list[tuple[int, int, int]]:
    sign_patterns = [
        (TEAL, TEAL, RED),
        (TEAL, RED, TEAL),
        (RED, TEAL, TEAL),
        (TEAL, RED, RED),
        (RED, TEAL, RED),
        (RED, RED, TEAL),
    ]
    random.shuffle(sign_patterns)
    for colors in sign_patterns:
        candidates = _height_candidates(colors, target_shift)
        if candidates:
            heights = random.choice(candidates)
            return [(c, h, color) for c, h, color in zip(_cycled_choice("bar_positions", BAR_POSITION_OPTIONS), heights, colors)]
    raise RuntimeError(f"unable to realize shift {target_shift}")


def _build_band_pattern(height: int) -> np.ndarray:
    required = [
        (ORANGE, BG),
        (BG, ORANGE),
        (ORANGE, ORANGE),
        (BG, BG),
    ]
    rows = list(required)
    while len(rows) < height:
        rows.append(random.choice(required))
    random.shuffle(rows)
    band = np.array(rows[:height], dtype=int)
    assert band.shape == (height, 2)
    return band


def _construct_grid(height: int, bar_specs: list[tuple[int, int, int]], band: np.ndarray) -> np.ndarray:
    grid = np.zeros((height, GRID_WIDTH), dtype=int)
    grid[:, DIVIDER_COL] = GREY
    for col, bar_height, color in bar_specs:
        grid[height - bar_height :, col] = color
    grid[:, -SOURCE_BAND_WIDTH:] = band
    return grid


def _shortcut_shift_bar_count(bars: list[dict]) -> int:
    teal_count = sum(bar["color"] == TEAL for bar in bars)
    red_count = sum(bar["color"] == RED for bar in bars)
    return teal_count - red_count


def _shortcut_shift_teal_only(bars: list[dict]) -> int:
    return sum(bar["height"] for bar in bars if bar["color"] == TEAL)


def _shortcut_shift_extremes(bars: list[dict]) -> int:
    teal_heights = [bar["height"] for bar in bars if bar["color"] == TEAL]
    red_heights = [bar["height"] for bar in bars if bar["color"] == RED]
    return max(teal_heights) - max(red_heights)


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        divider_col = _find_divider_column(grid)
        if divider_col != DIVIDER_COL:
            return False
        bars = _extract_bars(grid, divider_col)
        if len(bars) != 3:
            return False
        shift = _compute_shift_from_bars(bars)
        if not (1 <= shift <= 6):
            return False
        if {bar["color"] for bar in bars} != {TEAL, RED}:
            return False
        if any(not (1 <= bar["height"] <= 8) for bar in bars):
            return False

        band = _source_band(grid)
        row_types = {tuple(map(int, row)) for row in band}
        if row_types != {(ORANGE, BG), (BG, ORANGE), (ORANGE, ORANGE), (BG, BG)}:
            return False

        wrong_shifts = {
            _shortcut_shift_bar_count(bars),
            _shortcut_shift_teal_only(bars),
            _shortcut_shift_extremes(bars),
        }
        if shift in wrong_shifts:
            return False

        out = main(grid)
        target_slice = out[:, GRID_WIDTH - SOURCE_BAND_WIDTH - shift : GRID_WIDTH - shift]
        if not np.array_equal(target_slice, band):
            return False
        expected_orange = set()
        for r, c in np.argwhere(band == ORANGE):
            expected_orange.add((int(r), GRID_WIDTH - SOURCE_BAND_WIDTH + int(c) - shift))
        actual_orange = set(map(tuple, np.argwhere(out == ORANGE)))
        if actual_orange != expected_orange:
            return False
        if not np.array_equal(out[:, DIVIDER_COL], grid[:, DIVIDER_COL]):
            return False
        if not np.array_equal(out[:, :DIVIDER_COL], grid[:, :DIVIDER_COL]):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    target_shift = _cycled_choice("shift", SHIFT_VALUES)
    try:
        bar_specs = _choose_bar_specs(target_shift)
    except RuntimeError:
        return None
    band = _build_band_pattern(height)
    grid = _construct_grid(height, bar_specs, band)
    if not _validate_instance(grid):
        return None

    input_sig = _grid_signature(grid)
    output_sig = _grid_signature(main(grid))
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
