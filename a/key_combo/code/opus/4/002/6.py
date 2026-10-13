from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        RED = 2
        GREY = 5


BG = int(Color.BLACK)
MARKER = int(Color.RED)
ALT = int(Color.GREY)

MIN_HEIGHT = 10
MAX_HEIGHT = 18
MIN_PAIRS = 4
MAX_PAIRS = 6
GEN_ATTEMPTS = 300
_NEXT_EVEN_GAP = bool(random.getrandbits(1))
_LAST_MIN_ROW: int | None = None


def _pair_rows(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return marker rows as (row, left_col, right_col, gap)."""
    g = np.asarray(grid, dtype=int)
    rows: list[tuple[int, int, int, int]] = []
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] == MARKER)
        assert cols.size in (0, 2), "each row must contain either zero or two red markers"
        if cols.size == 2:
            left, right = map(int, cols)
            gap = right - left - 1
            assert gap >= 0
            rows.append((int(r), left, right, gap))
    return rows


def _analyze_grid(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "grid must fit ARC bounds"
    assert set(np.unique(g)).issubset({BG, MARKER}), "input should contain only black and red"

    rows = _pair_rows(g)
    assert MIN_PAIRS <= len(rows) <= MAX_PAIRS, "expected 4 to 6 mirrored marker rows"

    axis_sums = {left + right for _r, left, right, _gap in rows}
    assert axis_sums == {g.shape[1] - 1}, "all marker pairs must mirror around the canvas center axis"

    gaps = [gap for _r, _left, _right, gap in rows]
    assert min(gaps) >= 2, "the smallest gap must leave at least two black cells between markers"
    assert len(set(gaps)) == len(gaps), "pair gaps must be distinct"

    chosen = min(rows, key=lambda item: item[3])
    assert sum(gap == chosen[3] for gap in gaps) == 1, "the minimum gap row must be unique"
    return rows, chosen


def _phase_color(phase: int) -> int:
    return MARKER if int(phase) % 2 == 0 else ALT


def _paint_selected_row(out: np.ndarray, row: int, left: int, right: int) -> None:
    # Step 3: alternating fill across the marker segment, including both markers.
    for offset, c in enumerate(range(left, right + 1)):
        out[row, c] = _phase_color(offset)

    # Step 4: extend leftward from the left marker with the same parity chain.
    for offset, c in enumerate(range(left - 1, -1, -1), start=1):
        out[row, c] = _phase_color(offset)

    # Step 5: continue to the right, starting one step after the right marker.
    right_phase = right - left
    for offset, c in enumerate(range(right + 1, out.shape[1]), start=1):
        out[row, c] = _phase_color(right_phase + offset)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the closest mirrored marker pair row and extend its alternating fill to both edges.

    Input structure:
    - The grid has black background and 4 to 6 rows containing exactly one pair of red pixels.
    - Each occupied row's two red pixels are mirrored across the canvas's vertical center axis.
    - The rows use distinct left-right gaps, and one row has the unique smallest gap.

    Rule:
    - For each occupied row, measure the gap as the number of black cells between its left and right red markers.
    - Choose the unique row with the strictly smallest gap.
    - On that row, fill from the left marker through the right marker with alternating red and grey, starting with red on the left marker itself.
    - Continue the same alternation from the left marker to column 0 and from the right marker to the last column, flipping phase by one step per cell; leave all other rows unchanged.

    Key insight:
    - The solution depends on comparing all mirrored pairs by their within-row spacing, then preserving one continuous red/grey parity chain across the chosen row so the right-side ray start color is determined by the marker-gap parity.
    """
    g = np.asarray(grid, dtype=int)
    _rows, chosen = _analyze_grid(g)
    row, left, right, _gap = chosen

    out = g.copy()
    _paint_selected_row(out, row, left, right)
    return out


def _candidate_widths(n_pairs: int, want_even_gap: bool) -> list[int]:
    widths = []
    for w in range(14, 27):
        if w > 30:
            continue
        gaps = _gap_options(w)
        if len(gaps) >= n_pairs and (w % 2 == 0) == want_even_gap:
            widths.append(w)
    return widths


def _gap_options(width: int) -> list[int]:
    return [gap for gap in range(2, width - 1) if (width + gap) % 2 == 0]


def _gap_to_pair_cols(width: int, gap: int) -> tuple[int, int]:
    left = (width - gap - 2) // 2
    right = left + gap + 1
    assert 0 <= left < right < width
    assert left + right == width - 1
    return int(left), int(right)


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    rows, chosen = _analyze_grid(inp)
    chosen_row, left, right, gap = chosen

    assert inp.shape == out.shape, "this task preserves the grid shape"
    for r, _left, _right, _gap in rows:
        if r != chosen_row:
            assert np.array_equal(inp[r], out[r]), "non-selected rows must stay unchanged"

    for c in range(inp.shape[1]):
        expected = MARKER if (c - left) % 2 == 0 else ALT
        assert int(out[chosen_row, c]) == expected, "selected row parity chain is broken"

    assert np.array_equal(out[:chosen_row], inp[:chosen_row])
    assert np.array_equal(out[chosen_row + 1 :], inp[chosen_row + 1 :])
    assert gap % 2 == (inp.shape[1] % 2), "gap parity should match the center-axis type"
    assert np.any(out == ALT), "the output must introduce grey fill"


def _gen_attempt(want_even_gap: bool) -> tuple[np.ndarray, int]:
    n_pairs = random.randint(MIN_PAIRS, MAX_PAIRS)
    width = random.choice(_candidate_widths(n_pairs, want_even_gap))
    height = random.randint(max(MIN_HEIGHT, n_pairs + 3), MAX_HEIGHT)

    grid = np.full((height, width), BG, dtype=int)
    rows = sorted(random.sample(range(height), n_pairs))
    options = random.sample(_gap_options(width), n_pairs)
    min_gap = min(options)
    min_row_choices = [r for r in rows if r != _LAST_MIN_ROW] or rows
    min_row = random.choice(min_row_choices)
    other_gaps = [gap for gap in options if gap != min_gap]
    random.shuffle(other_gaps)
    gap_by_row = {min_row: min_gap}
    for r, gap in zip((r for r in rows if r != min_row), other_gaps):
        gap_by_row[r] = gap

    for r in rows:
        gap = gap_by_row[r]
        left, right = _gap_to_pair_cols(width, gap)
        grid[r, left] = MARKER
        grid[r, right] = MARKER

    out = main(grid)
    _semantic_checks(grid, out)
    return grid, min_row


def generate_input() -> np.ndarray:
    global _NEXT_EVEN_GAP, _LAST_MIN_ROW
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, min_row = _gen_attempt(_NEXT_EVEN_GAP)
            _NEXT_EVEN_GAP = not _NEXT_EVEN_GAP
            _LAST_MIN_ROW = int(min_row)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid mirrored-pair input")
