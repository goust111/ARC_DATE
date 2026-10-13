from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg, paste
import itertools
import random


LINE_COLOR = Color.RED
MARKER_COLOR = Color.GREEN
EMIT_COLOR = Color.BLUE


def _marker_centers(g: np.ndarray) -> list[tuple[int, int]]:
    marker = np.array(
        [
            [MARKER_COLOR, MARKER_COLOR, MARKER_COLOR],
            [MARKER_COLOR, LINE_COLOR, MARKER_COLOR],
            [MARKER_COLOR, MARKER_COLOR, MARKER_COLOR],
        ],
        dtype=int,
    )
    centers: list[tuple[int, int]] = []
    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if np.array_equal(g[r - 1 : r + 2, c - 1 : c + 2], marker):
                centers.append((r, c))

    assert centers, "expected at least one green 3x3 marker on a red line"
    return centers


def _red_line_rows_cols(g: np.ndarray) -> tuple[list[int], list[int]]:
    line_colors = {LINE_COLOR, MARKER_COLOR}
    rows: list[int] = []
    cols: list[int] = []

    H, W = g.shape
    for r in range(H):
        vals = set(map(int, g[r, :]))
        if vals <= line_colors and LINE_COLOR in vals:
            rows.append(r)

    for c in range(W):
        vals = set(map(int, g[:, c]))
        if vals <= line_colors and LINE_COLOR in vals:
            cols.append(c)

    assert bool(rows) != bool(cols), "expected exactly one orientation of full red guide lines"
    return rows, cols


def _paint_ring(out: np.ndarray, r: int, c: int) -> None:
    H, W = out.shape
    assert 0 < r < H - 1 and 0 < c < W - 1, "crossing ring must fit inside the grid"
    assert int(out[r, c]) == LINE_COLOR, "crossing center must remain a red line pixel"
    assert not np.any(out[r - 1 : r + 2, c - 1 : c + 2] == MARKER_COLOR), (
        "blue crossing ring must not overlap a green source marker"
    )

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out[r + dr, c + dc] = EMIT_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit blue perpendicular guide lines from each green marker and stamp blue rings where they cross the red guide lines.

    Input structure
    - The background is dominant, with full-span parallel red guide lines.
    - Each source marker is a green 3x3 ring centered on a red guide-line pixel.
    - All guide lines are horizontal or all guide lines are vertical.

    Rule
    - A marker on a horizontal red line emits a vertical blue line through its center column; a marker on a vertical red line emits a horizontal blue line through its center row.
    - The emitted line paints only background cells blue.
    - At each crossing with a red guide line, stamp a blue 3x3 ring with the red center preserved; keep the original green source marker unchanged.

    Key insight
    - The green ring is not copied along the red line; it defines the perpendicular projection, and every red-line crossing on that projection receives the same ring shape in blue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg not in {LINE_COLOR, MARKER_COLOR, EMIT_COLOR}, "expected a distinct background color"

    centers = _marker_centers(g)
    red_rows, red_cols = _red_line_rows_cols(g)
    source_centers = set(centers)
    out = g.copy()

    if red_rows:
        assert all(r in red_rows for r, _c in centers), "all markers must sit on horizontal red lines"
        for _r, c in centers:
            for rr in range(out.shape[0]):
                if int(out[rr, c]) == bg:
                    out[rr, c] = EMIT_COLOR
            for rr in red_rows:
                if (rr, c) not in source_centers:
                    _paint_ring(out, rr, c)
    else:
        assert red_cols, "expected vertical red guide lines"
        assert all(c in red_cols for _r, c in centers), "all markers must sit on vertical red lines"
        for r, _c in centers:
            for cc in range(out.shape[1]):
                if int(out[r, cc]) == bg:
                    out[r, cc] = EMIT_COLOR
            for cc in red_cols:
                if (r, cc) not in source_centers:
                    _paint_ring(out, r, cc)

    return out

# --------------- generator (o5x) ---------------

# Colors and motifs.
_GEN_BG, _GEN_BLUE, _GEN_RED, _GEN_GREEN = 0, 1, 2, 3
_GEN_MARKER = np.array(
    [
        [_GEN_GREEN, _GEN_GREEN, _GEN_GREEN],
        [_GEN_GREEN, _GEN_RED, _GEN_GREEN],
        [_GEN_GREEN, _GEN_GREEN, _GEN_GREEN],
    ],
    dtype=int,
)
_GEN_BLUE_RING = np.array(
    [
        [_GEN_BLUE, _GEN_BLUE, _GEN_BLUE],
        [_GEN_BLUE, _GEN_RED, _GEN_BLUE],
        [_GEN_BLUE, _GEN_BLUE, _GEN_BLUE],
    ],
    dtype=int,
)

# Canvas and coordinate constraints.
_GEN_ROW_BOUNDS, _GEN_COL_BOUNDS = (10, 16), (10, 18)
_GEN_BORDER_CLEARANCE = 1

# Structural sampling.
_GEN_GUIDE_COUNT_BOUNDS = (1, 4)
_GEN_MARKER_COUNT_BOUNDS = (1, 2)
_GEN_SINGLE_GUIDE_MARKERS = 1
_GEN_GUIDE_GAP, _GEN_MARKER_PERP_GAP = 4, 3
_GEN_ORIENTATIONS = (True, False)
_GEN_RETRY_LIMIT = 300


def _gen_spaced_positions(_gen_limit: int, _gen_count: int, _gen_min_gap: int) -> list[int]:
    _gen_options = [
        _gen_combo
        for _gen_combo in itertools.combinations(
            range(_GEN_BORDER_CLEARANCE, _gen_limit - _GEN_BORDER_CLEARANCE),
            _gen_count,
        )
        if all(_gen_b - _gen_a >= _gen_min_gap for _gen_a, _gen_b in zip(_gen_combo, _gen_combo[1:]))
    ]
    if not _gen_options:
        return []
    return list(random.choice(_gen_options))


def _gen_stamp_marker(_gen_grid: np.ndarray, _gen_r: int, _gen_c: int) -> None:
    paste(_gen_grid, _GEN_MARKER, _gen_r - 1, _gen_c - 1)


def _gen_stamp_blue_ring(_gen_grid: np.ndarray, _gen_r: int, _gen_c: int) -> None:
    paste(_gen_grid, _GEN_BLUE_RING, _gen_r - 1, _gen_c - 1)


def _gen_paint_guides(_gen_grid: np.ndarray, _gen_horizontal: bool, _gen_guides: list[int]) -> None:
    if _gen_horizontal:
        _gen_grid[_gen_guides, :] = _GEN_RED
    else:
        _gen_grid[:, _gen_guides] = _GEN_RED


def _gen_sample_geometry(
    _gen_horizontal: bool,
    _gen_guide_count: int,
    _gen_marker_count: int,
) -> tuple[int, int, list[int], list[tuple[int, int]]] | None:
    """Sample canvas dimensions, guide coordinates, and marker centers."""
    _gen_min_guide_span = 2 * _GEN_BORDER_CLEARANCE + 1 + _GEN_GUIDE_GAP * (_gen_guide_count - 1)
    if _gen_horizontal:
        _gen_h = random.randint(max(_GEN_ROW_BOUNDS[0], _gen_min_guide_span), _GEN_ROW_BOUNDS[1])
        _gen_w = random.randint(*_GEN_COL_BOUNDS)
        _gen_guides = _gen_spaced_positions(_gen_h, _gen_guide_count, _GEN_GUIDE_GAP)
        _gen_marker_lines = random.sample(_gen_guides, _gen_marker_count)
        _gen_perps = _gen_spaced_positions(_gen_w, _gen_marker_count, _GEN_MARKER_PERP_GAP)
        _gen_centers = list(zip(_gen_marker_lines, _gen_perps))
    else:
        _gen_h = random.randint(*_GEN_ROW_BOUNDS)
        _gen_w = random.randint(max(_GEN_COL_BOUNDS[0], _gen_min_guide_span), _GEN_COL_BOUNDS[1])
        _gen_guides = _gen_spaced_positions(_gen_w, _gen_guide_count, _GEN_GUIDE_GAP)
        _gen_marker_lines = random.sample(_gen_guides, _gen_marker_count)
        _gen_perps = _gen_spaced_positions(_gen_h, _gen_marker_count, _GEN_MARKER_PERP_GAP)
        _gen_centers = list(zip(_gen_perps, _gen_marker_lines))

    if len(_gen_guides) != _gen_guide_count or len(_gen_perps) != _gen_marker_count:
        return None
    return _gen_h, _gen_w, _gen_guides, _gen_centers


def _gen_build_output(
    _gen_h: int,
    _gen_w: int,
    _gen_horizontal: bool,
    _gen_guides: list[int],
    _gen_centers: list[tuple[int, int]],
) -> np.ndarray:
    _gen_out = np.full((_gen_h, _gen_w), _GEN_BG, dtype=int)
    _gen_paint_guides(_gen_out, _gen_horizontal, _gen_guides)

    for _gen_r, _gen_c in _gen_centers:
        _gen_stamp_marker(_gen_out, _gen_r, _gen_c)

    _gen_sources = set(_gen_centers)
    if _gen_horizontal:
        for _gen_r, _gen_c in _gen_centers:
            _gen_col = _gen_out[:, _gen_c]
            _gen_col[_gen_col == _GEN_BG] = _GEN_BLUE
            for _gen_rr in _gen_guides:
                if (_gen_rr, _gen_c) not in _gen_sources:
                    _gen_stamp_blue_ring(_gen_out, _gen_rr, _gen_c)
    else:
        for _gen_r, _gen_c in _gen_centers:
            _gen_row = _gen_out[_gen_r, :]
            _gen_row[_gen_row == _GEN_BG] = _GEN_BLUE
            for _gen_cc in _gen_guides:
                if (_gen_r, _gen_cc) not in _gen_sources:
                    _gen_stamp_blue_ring(_gen_out, _gen_r, _gen_cc)

    return _gen_out


def _gen_derive_input(
    _gen_out: np.ndarray,
    _gen_horizontal: bool,
    _gen_guides: list[int],
    _gen_centers: list[tuple[int, int]],
) -> np.ndarray:
    _gen_inp = _gen_out.copy()
    _gen_inp[_gen_inp == _GEN_BLUE] = _GEN_BG
    _gen_paint_guides(_gen_inp, _gen_horizontal, _gen_guides)

    for _gen_r, _gen_c in _gen_centers:
        _gen_stamp_marker(_gen_inp, _gen_r, _gen_c)
    return _gen_inp


def _gen_attempt() -> np.ndarray | None:
    _gen_horizontal = random.choice(_GEN_ORIENTATIONS)
    _gen_guide_count = random.randint(*_GEN_GUIDE_COUNT_BOUNDS)
    _gen_marker_count = (
        _GEN_SINGLE_GUIDE_MARKERS
        if _gen_guide_count == 1
        else random.randint(*_GEN_MARKER_COUNT_BOUNDS)
    )
    _gen_geometry = _gen_sample_geometry(_gen_horizontal, _gen_guide_count, _gen_marker_count)
    if _gen_geometry is None:
        return None
    _gen_h, _gen_w, _gen_guides, _gen_centers = _gen_geometry

    _gen_out = _gen_build_output(_gen_h, _gen_w, _gen_horizontal, _gen_guides, _gen_centers)
    _gen_inp = _gen_derive_input(_gen_out, _gen_horizontal, _gen_guides, _gen_centers)
    if not np.array_equal(main(_gen_inp), _gen_out):
        return None
    return _gen_inp


def generate_input() -> np.ndarray:
    for _gen_i in range(_GEN_RETRY_LIMIT):
        _gen_grid = _gen_attempt()
        if _gen_grid is not None:
            return _gen_grid
    raise RuntimeError("failed to generate a valid ac3e2b04 input")
