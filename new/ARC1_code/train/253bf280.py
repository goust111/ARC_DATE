from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import itertools
import random


def _paint_between_pairs(g: np.ndarray, out: np.ndarray, bg: int, marker: int, fill: int) -> None:
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r, :] == marker)
        assert len(cols) <= 2, "each row may contain at most two markers"
        if len(cols) == 2:
            c0, c1 = map(int, cols)
            assert np.all(g[r, c0 + 1 : c1] == bg), "row segment must cross only background"
            out[r, c0 + 1 : c1] = fill

    for c in range(g.shape[1]):
        rows = np.flatnonzero(g[:, c] == marker)
        assert len(rows) <= 2, "each column may contain at most two markers"
        if len(rows) == 2:
            r0, r1 = map(int, rows)
            assert np.all(g[r0 + 1 : r1, c] == bg), "column segment must cross only background"
            out[r0 + 1 : r1, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect every pair of aligned teal marker pixels by painting the gap between them green.

    Input structure
    - The grid has a dominant background color and exactly one non-background marker color.
    - Marker pixels are isolated endpoints; any row or column contains zero, one, or two markers.
    - Green is absent from the input and is reserved for the added connector segments.

    Rule
    - Preserve the input grid.
    - For each row with exactly two markers, paint the background pixels strictly between them green.
    - For each column with exactly two markers, paint the background pixels strictly between them green.

    Key insight
    - The sparse markers are endpoints of missing axis-aligned bars; only same-row or same-column pairs authorize a fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 1, "expected exactly one marker color"
    assert Color.GREEN not in foreground and bg != Color.GREEN, "green must be absent from the input"
    marker = foreground[0]

    marker_coords = np.argwhere(g == marker)
    assert len(marker_coords) > 0, "expected at least one marker"

    out = g.copy()
    _paint_between_pairs(g, out, bg, marker, Color.GREEN)
    return out

# --------------- generator (o5x) ---------------

# Colors used by the constructed training inputs.
_GEN_BG, _GEN_MARKER, _GEN_FILL = 0, 8, 3

# Canvas bounds.
_GEN_MIN_H, _GEN_MAX_H = 3, 12
_GEN_MIN_W, _GEN_MAX_W = 3, 13
# Marker count and pair count limits.
_GEN_MAX_ROW_PAIRS, _GEN_MAX_COL_PAIRS, _GEN_MAX_TOTAL_PAIRS = 2, 2, 3
_GEN_MAX_UNPAIRED = 2
_GEN_MIN_MARKERS, _GEN_MAX_MARKERS = 1, 7
# Pair gaps, counted as background cells strictly between endpoints.
_GEN_MIN_GAP, _GEN_MAX_GAP = 2, 7
# Rejection sampling limit.
_GEN_MAX_ATTEMPTS = 300


def _gen_axis_pair_choices(length: int, pair_count: int) -> list[tuple[tuple[int, int], ...]]:
    """All ways to pick disjoint aligned endpoint pairs along one axis."""
    if pair_count == 0:
        return [()]
    min_span, max_span = _GEN_MIN_GAP + 1, _GEN_MAX_GAP + 1
    pairs = [
        (a, b)
        for a in range(length)
        for b in range(a + min_span, min(length, a + max_span + 1))
    ]
    if pair_count == 1:
        return [(p,) for p in pairs]

    choices: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(pairs, pair_count):
        used = [x for pair in combo for x in pair]
        if len(set(used)) == len(used):
            choices.append(combo)
    return choices


def _gen_feasible_count_tuples(h: int, w: int) -> list[tuple[int, int, int]]:
    feasible: list[tuple[int, int, int]] = []
    row_pair_possible = {rp: bool(_gen_axis_pair_choices(w, rp)) for rp in range(_GEN_MAX_ROW_PAIRS + 1)}
    col_pair_possible = {cp: bool(_gen_axis_pair_choices(h, cp)) for cp in range(_GEN_MAX_COL_PAIRS + 1)}

    for rp in range(_GEN_MAX_ROW_PAIRS + 1):
        for cp in range(_GEN_MAX_COL_PAIRS + 1):
            if rp + cp > _GEN_MAX_TOTAL_PAIRS:
                continue
            for up in range(_GEN_MAX_UNPAIRED + 1):
                marker_count = 2 * (rp + cp) + up
                if not (_GEN_MIN_MARKERS <= marker_count <= _GEN_MAX_MARKERS):
                    continue
                if not row_pair_possible[rp] or not col_pair_possible[cp]:
                    continue
                if h - 2 * cp < rp + up:
                    continue
                if w - 2 * rp < cp + up:
                    continue
                feasible.append((rp, cp, up))
    return feasible


def _gen_choose_pair_layout(h: int, w: int, rp: int, cp: int, up: int):
    """Choose pair endpoints and their carrier rows/columns."""
    row_col_pairs = list(random.choice(_gen_axis_pair_choices(w, rp)))
    col_row_pairs = list(random.choice(_gen_axis_pair_choices(h, cp)))
    random.shuffle(row_col_pairs)
    random.shuffle(col_row_pairs)
    row_pair_cols = {c for pair in row_col_pairs for c in pair}
    col_pair_rows = {r for pair in col_row_pairs for r in pair}
    row_pool = [r for r in range(h) if r not in col_pair_rows]
    col_pool = [c for c in range(w) if c not in row_pair_cols]
    if len(row_pool) < rp + up or len(col_pool) < cp + up:
        return None
    row_pair_rows = random.sample(row_pool, rp)
    col_pair_cols = random.sample(col_pool, cp)
    return row_pair_rows, row_col_pairs, col_pair_cols, col_row_pairs, row_pair_cols, col_pair_rows


def _gen_choose_unpaired_coords(h, w, up, row_pair_rows, row_pair_cols, col_pair_rows, col_pair_cols):
    """Choose singleton markers that cannot accidentally form a row/column pair."""
    blocked_rows = set(row_pair_rows) | col_pair_rows
    blocked_cols = row_pair_cols | set(col_pair_cols)
    unpaired_row_pool = [r for r in range(h) if r not in blocked_rows]
    unpaired_col_pool = [c for c in range(w) if c not in blocked_cols]
    if len(unpaired_row_pool) < up or len(unpaired_col_pool) < up:
        return None
    unpaired_rows = random.sample(unpaired_row_pool, up)
    unpaired_cols = random.sample(unpaired_col_pool, up)
    random.shuffle(unpaired_cols)
    return list(zip(unpaired_rows, unpaired_cols))


def _gen_is_isolated(coords: list[tuple[int, int]]) -> bool:
    for (r0, c0), (r1, c1) in itertools.combinations(coords, 2):
        if abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1:
            return False
    return True


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for r in range(inp.shape[0]):
        cols = np.flatnonzero(inp[r] == _GEN_MARKER)
        if len(cols) == 2:
            c0, c1 = map(int, cols)
            out[r, c0 + 1 : c1] = _GEN_FILL
    for c in range(inp.shape[1]):
        rows = np.flatnonzero(inp[:, c] == _GEN_MARKER)
        if len(rows) == 2:
            r0, r1 = map(int, rows)
            out[r0 + 1 : r1, c] = _GEN_FILL
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_GEN_MIN_H, _GEN_MAX_H)
    w = random.randint(_GEN_MIN_W, _GEN_MAX_W)

    count_tuples = _gen_feasible_count_tuples(h, w)
    if not count_tuples:
        return None
    rp, cp, up = random.choice(count_tuples)

    layout = _gen_choose_pair_layout(h, w, rp, cp, up)
    if layout is None:
        return None
    row_pair_rows, row_col_pairs, col_pair_cols, col_row_pairs, row_pair_cols, col_pair_rows = layout

    unpaired_coords = _gen_choose_unpaired_coords(h, w, up, row_pair_rows, row_pair_cols, col_pair_rows, col_pair_cols)
    if unpaired_coords is None:
        return None

    coords: list[tuple[int, int]] = []
    for r, (c0, c1) in zip(row_pair_rows, row_col_pairs):
        coords.extend([(r, c0), (r, c1)])
    for c, (r0, r1) in zip(col_pair_cols, col_row_pairs):
        coords.extend([(r0, c), (r1, c)])
    coords.extend(unpaired_coords)

    if not _gen_is_isolated(coords):
        return None

    inp = np.full((h, w), _GEN_BG, dtype=int)
    for r, c in coords:
        inp[r, c] = _GEN_MARKER

    row_counts = np.count_nonzero(inp == _GEN_MARKER, axis=1)
    col_counts = np.count_nonzero(inp == _GEN_MARKER, axis=0)
    if row_counts.max(initial=0) > 2 or col_counts.max(initial=0) > 2:
        return None

    out = _gen_expected_output(inp)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 253bf280 input")
