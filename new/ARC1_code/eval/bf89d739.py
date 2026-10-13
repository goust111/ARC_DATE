from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import Color, colors_present, get_bg


def _find_spine(red_coords: list[tuple[int, int]]) -> tuple[str, int, int, int]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in red_coords:
        rows[r].append(c)
        cols[c].append(r)

    row_spines = [(r, sorted(cs)) for r, cs in rows.items() if len(cs) == 2]
    col_spines = [(c, sorted(rs)) for c, rs in cols.items() if len(rs) == 2]
    assert len(row_spines) + len(col_spines) == 1, "expected exactly one two-red row or column spine"

    if row_spines:
        spine_row, cols_on_spine = row_spines[0]
        return "row", int(spine_row), int(cols_on_spine[0]), int(cols_on_spine[1])

    spine_col, rows_on_spine = col_spines[0]
    return "col", int(spine_col), int(rows_on_spine[0]), int(rows_on_spine[1])


def _paint_connector_cell(out: np.ndarray, bg: int, r: int, c: int) -> None:
    if int(out[r, c]) == Color.RED:
        return
    assert int(out[r, c]) in (bg, Color.GREEN), "connectors may only cross background or prior connectors"
    out[r, c] = Color.GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect all red control pixels by drawing green orthogonal segments to the unique red spine.

    Input structure
    - A black background contains only red control pixels.
    - Exactly two red pixels share one row or one column; this aligned pair defines the spine segment.
    - Every other red pixel lies within the span of that spine along the perpendicular direction.

    Rule
    - Paint the background cells between the aligned red pair green, preserving the red endpoints.
    - If the spine is horizontal, draw a vertical green segment from each other red pixel to the spine.
    - If the spine is vertical, draw a horizontal green segment from each other red pixel to the spine.

    Key insight
    - The only repeated row or column among the red pixels is the connector trunk; all remaining pixels attach to it perpendicularly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BACKGROUND, "expected black background"
    assert set(colors_present(g)) == {Color.BACKGROUND, Color.RED}, "expected only black background and red controls"

    red_coords = [(int(r), int(c)) for r, c in np.argwhere(g == Color.RED)]
    assert len(red_coords) >= 2, "expected red control pixels"

    axis, fixed, lo, hi = _find_spine(red_coords)
    out = g.copy()

    if axis == "row":
        spine_row = fixed
        for c in range(lo, hi + 1):
            _paint_connector_cell(out, bg, spine_row, c)
        for r, c in red_coords:
            if r == spine_row:
                continue
            assert lo <= c <= hi, "off-spine red pixels must project onto the horizontal spine"
            r0, r1 = sorted((r, spine_row))
            for rr in range(r0, r1 + 1):
                _paint_connector_cell(out, bg, rr, c)
        return out

    spine_col = fixed
    for r in range(lo, hi + 1):
        _paint_connector_cell(out, bg, r, spine_col)
    for r, c in red_coords:
        if c == spine_col:
            continue
        assert lo <= r <= hi, "off-spine red pixels must project onto the vertical spine"
        c0, c1 = sorted((c, spine_col))
        for cc in range(c0, c1 + 1):
            _paint_connector_cell(out, bg, r, cc)

    return out

# --------------- generator (o5x) ---------------

# Colors used by the generated input/output pair.
_GEN_BG = Color.BACKGROUND
_GEN_RED = Color.RED
_GEN_GREEN = Color.GREEN
# Canvas bounds; high endpoints are exclusive for np.random.randint.
_GEN_GRID_H_RANGE = (14, 20)
_GEN_GRID_W_RANGE = (10, 18)
# Spine and branch structure.
_GEN_AXES = ("row", "col")
_GEN_BRANCH_COUNT_RANGE = (2, 6)
_GEN_SPINE_LEN_CELLS = (6, 12)
_GEN_SPINE_LEN_FRAC = (0.60, 0.86)
_GEN_FIXED_MARGIN = 2
_GEN_BRANCH_OFFSET_RANGE = (2, 6)
# Retry budget for rejected layouts.
_GEN_MAX_ATTEMPTS = 300


def _gen_choose_spine_length(parallel_dim: int, branch_count: int) -> int | None:
    min_len = max(_GEN_SPINE_LEN_CELLS[0], int(np.ceil(_GEN_SPINE_LEN_FRAC[0] * parallel_dim)), branch_count + 2)
    max_len = min(_GEN_SPINE_LEN_CELLS[1], int(np.floor(_GEN_SPINE_LEN_FRAC[1] * parallel_dim)))
    if min_len > max_len:
        return None
    return int(np.random.randint(min_len, max_len + 1))


def _gen_offset_options(fixed: int, size: int) -> tuple[list[int], list[int]]:
    neg = [d for d in range(*_GEN_BRANCH_OFFSET_RANGE) if fixed - d >= 0]
    pos = [d for d in range(*_GEN_BRANCH_OFFSET_RANGE) if fixed + d < size]
    return neg, pos


def _gen_pick_offsets(fixed: int, size: int, branch_count: int) -> list[int] | None:
    neg, pos = _gen_offset_options(fixed, size)
    lo = max(1, branch_count - len(pos))
    hi = min(len(neg), branch_count - 1)
    if lo > hi:
        return None
    n_neg = int(np.random.randint(lo, hi + 1))
    n_pos = branch_count - n_neg
    neg_pick = list(np.random.choice(neg, size=n_neg, replace=False))
    pos_pick = list(np.random.choice(pos, size=n_pos, replace=False))
    offsets = [-int(d) for d in neg_pick] + [int(d) for d in pos_pick]
    np.random.shuffle(offsets)
    return offsets


def _gen_sample_layout(axis: str, h: int, w: int, branch_count: int):
    """Return spine placement, branch anchors, and offsets, or None if they cannot fit."""
    parallel_dim = w if axis == "row" else h
    perp_dim = h if axis == "row" else w
    spine_len = _gen_choose_spine_length(parallel_dim, branch_count)
    if spine_len is None:
        return None

    fixed = int(np.random.randint(_GEN_FIXED_MARGIN, perp_dim - _GEN_FIXED_MARGIN))
    offsets = _gen_pick_offsets(fixed, perp_dim, branch_count)
    if offsets is None:
        return None

    lo = int(np.random.randint(0, parallel_dim - spine_len + 1))
    hi = lo + spine_len - 1
    attach_positions = list(np.random.choice(np.arange(lo + 1, hi), size=branch_count, replace=False))
    return fixed, lo, hi, attach_positions, offsets


def _gen_cell(axis: str, perp: int, parallel: int, offset: int = 0) -> tuple[int, int]:
    if axis == "row":
        return perp + offset, int(parallel)
    return int(parallel), perp + offset


def _gen_paint_expected(out: np.ndarray, cells) -> None:
    for r, c in cells:
        if int(out[r, c]) == _GEN_BG:
            out[r, c] = _GEN_GREEN


def _gen_render_pair(axis: str, h: int, w: int, fixed: int, lo: int, hi: int, attach_positions, offsets):
    inp = np.zeros((h, w), dtype=int)
    inp[_gen_cell(axis, fixed, lo)] = _GEN_RED
    inp[_gen_cell(axis, fixed, hi)] = _GEN_RED
    for attach, offset in zip(attach_positions, offsets):
        inp[_gen_cell(axis, fixed, attach, offset)] = _GEN_RED

    out = inp.copy()
    _gen_paint_expected(out, [_gen_cell(axis, fixed, p) for p in range(lo, hi + 1)])
    for attach, offset in zip(attach_positions, offsets):
        p0, p1 = sorted((fixed, fixed + offset))
        _gen_paint_expected(out, [_gen_cell(axis, p, attach) for p in range(p0, p1 + 1)])
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*_GEN_GRID_H_RANGE))
    w = int(np.random.randint(*_GEN_GRID_W_RANGE))
    axis = _GEN_AXES[int(np.random.randint(len(_GEN_AXES)))]
    branch_count = int(np.random.randint(*_GEN_BRANCH_COUNT_RANGE))

    layout = _gen_sample_layout(axis, h, w, branch_count)
    if layout is None:
        return None
    fixed, lo, hi, attach_positions, offsets = layout
    inp, out = _gen_render_pair(axis, h, w, fixed, lo, hi, attach_positions, offsets)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bf89d739 input")
