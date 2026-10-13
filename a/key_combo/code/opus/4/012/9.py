from __future__ import annotations

import random

import numpy as np


GREY = 5
MARKER = 6
PALETTE = (0, 1, 2, 3, 4, 7, 8, 9)
PERIOD_CHOICES = ((2, 2), (3, 3))
GRID_SIDE_RANGE = (12, 18)
RECT_SIDE_RANGE = (4, 6)
EDGE_MARGINS = (1, 2)
GENERATION_ATTEMPTS = 400

TILE_TEMPLATES: dict[tuple[int, int], tuple[np.ndarray, ...]] = {
    (2, 2): (
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [2, 0]], dtype=int),
        np.array([[0, 1], [2, 1]], dtype=int),
    ),
    (3, 3): (
        np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=int),
        np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=int),
        np.array([[0, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=int),
    ),
}


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile_has_period(tile: np.ndarray, row_period: int, col_period: int) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                return False
    return True


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    periods = sorted(
        ((rp, cp) for rp in range(1, h + 1) for cp in range(1, w + 1)),
        key=lambda item: (item[0] * item[1], item[0], item[1]),
    )
    for row_period, col_period in periods:
        if _tile_has_period(tile, row_period, col_period):
            return row_period, col_period
    raise AssertionError("tile must have a valid period")


def _make_tile(period: tuple[int, int]) -> np.ndarray:
    template = random.choice(TILE_TEMPLATES[period]).copy()
    n_colors = int(template.max()) + 1
    colors = np.array(random.sample(PALETTE, n_colors), dtype=int)
    tile = colors[template]

    for _ in range(random.randrange(4)):
        tile = np.rot90(tile)
    if random.random() < 0.5:
        tile = np.fliplr(tile)
    if random.random() < 0.5:
        tile = np.flipud(tile)

    tile = np.asarray(tile, dtype=int)
    assert _minimal_period(tile) == period
    return tile


def _periodic_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    row_phase = random.randrange(tile.shape[0])
    col_phase = random.randrange(tile.shape[1])
    rows = (np.arange(height)[:, None] + row_phase) % tile.shape[0]
    cols = (np.arange(width)[None, :] + col_phase) % tile.shape[1]
    return tile[rows, cols].astype(int, copy=True)


def _period_values(
    grid: np.ndarray,
    visible_mask: np.ndarray,
    row_period: int,
    col_period: int,
) -> dict[tuple[int, int], int] | None:
    values: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(visible_mask):
        key = (int(r) % row_period, int(c) % col_period)
        value = int(grid[r, c])
        if key in values and values[key] != value:
            return None
        values[key] = value
    if len(values) != row_period * col_period:
        return None
    return values


def _reconstruct_bbox(
    values: dict[tuple[int, int], int],
    row_period: int,
    col_period: int,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    patch = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc, c in enumerate(range(c0, c1 + 1)):
            patch[rr, cc] = values[(r % row_period, c % col_period)]
    return patch


def _marker_geometry(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    marker_coords = np.argwhere(grid == MARKER)
    assert marker_coords.shape == (2, 2), "expected exactly two marker pixels"

    bbox = _bbox_coords(marker_coords)
    rows = tuple(sorted({int(r) for r, _c in marker_coords}))
    cols = tuple(sorted({int(c) for _r, c in marker_coords}))
    assert len(rows) == 2 and len(cols) == 2, "markers must define a diagonal rectangle"

    r0, c0, r1, c1 = bbox
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert all((int(r), int(c)) in corners for r, c in marker_coords)

    rect = grid[r0 : r1 + 1, c0 : c1 + 1]
    rect_marker_mask = rect == MARKER
    assert int(rect_marker_mask.sum()) == 2
    assert np.all(rect[~rect_marker_mask] == GREY), "rectangle interior must be grey except markers"
    return bbox, rows, cols


def _solve_periodic_fill(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    divider_rows: tuple[int, int],
    divider_cols: tuple[int, int],
) -> np.ndarray:
    visible_mask = (grid != GREY) & (grid != MARKER)
    assert visible_mask.any(), "expected visible periodic pixels outside the masked rectangle"

    h, w = grid.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in range(1, min(5, h) + 1):
        for col_period in range(1, min(5, w) + 1):
            values = _period_values(grid, visible_mask, row_period, col_period)
            if values is not None:
                candidates.append((row_period, col_period, values))

    assert candidates, "visible pixels must determine a repeating tile"
    min_area = min(row_period * col_period for row_period, col_period, _ in candidates)
    best = [
        (row_period, col_period, values)
        for row_period, col_period, values in candidates
        if row_period * col_period == min_area
    ]

    fills = [
        _reconstruct_bbox(values, row_period, col_period, bbox)
        for row_period, col_period, values in best
    ]
    fill = fills[0]
    assert all(np.array_equal(fill, other) for other in fills), "minimal periods must agree on the repair"

    out = grid.copy()
    r0, c0, r1, c1 = bbox
    target = out[r0 : r1 + 1, c0 : c1 + 1]
    grey_mask = target == GREY
    target[grey_mask] = fill[grey_mask]
    out[list(divider_rows), :] = MARKER
    out[:, list(divider_cols)] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the periodic pattern inside the marker-defined grey rectangle, then keep the marker rows and columns as magenta dividers.

    Input structure:
    - The whole grid is tiled by a small periodic color pattern except for one grey rectangular occlusion.
    - Exactly two magenta marker pixels replace two opposite corners of that grey rectangle.
    - All non-grey, non-magenta cells outside the rectangle still show the original periodic tile.

    Rule:
    - Locate the two magenta markers and use their min/max rows and columns as the rectangle bounds.
    - Infer the smallest row/column period up to 5x5 that matches every visible non-grey, non-magenta cell.
    - Repaint only the grey cells inside the rectangle with the inferred periodic tile at their absolute `(r, c)` residues.
    - Paint both marker rows and both marker columns magenta across the whole grid, leaving those divider lines on top of the repaired pattern.

    Key insight:
    - The same two magenta pixels play two roles: they mark the missing rectangle to reconstruct and they also specify the row/column dividers that must overwrite the recovered pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bbox, divider_rows, divider_cols = _marker_geometry(g)
    return _solve_periodic_fill(g, bbox, divider_rows, divider_cols)


def _all_residues_visible(
    blocked: np.ndarray,
    row_period: int,
    col_period: int,
) -> bool:
    visible = ~blocked
    for rr in range(row_period):
        for cc in range(col_period):
            if not np.any(visible[rr::row_period, cc::col_period]):
                return False
    return True


def _expected_output(base: np.ndarray, divider_rows: tuple[int, int], divider_cols: tuple[int, int]) -> np.ndarray:
    out = base.copy()
    out[list(divider_rows), :] = MARKER
    out[:, list(divider_cols)] = MARKER
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    row_period, col_period = random.choice(PERIOD_CHOICES)
    tile = _make_tile((row_period, col_period))
    base = _periodic_grid(tile, height, width)

    rect_h = random.randint(*RECT_SIDE_RANGE)
    rect_w = random.randint(*RECT_SIDE_RANGE)
    margin = random.choice(EDGE_MARGINS)
    if rect_h + 2 * margin > height or rect_w + 2 * margin > width:
        return None

    r0 = random.randint(margin, height - rect_h - margin)
    c0 = random.randint(margin, width - rect_w - margin)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    blocked = np.zeros((height, width), dtype=bool)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    if not _all_residues_visible(blocked, row_period, col_period):
        return None

    if random.random() < 0.5:
        markers = ((r0, c0), (r1, c1))
    else:
        markers = ((r0, c1), (r1, c0))
    divider_rows = tuple(sorted({markers[0][0], markers[1][0]}))
    divider_cols = tuple(sorted({markers[0][1], markers[1][1]}))

    inp = base.copy()
    inp[blocked] = GREY
    for r, c in markers:
        inp[r, c] = MARKER

    expected = _expected_output(base, divider_rows, divider_cols)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair = _gen_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input for puzzle 9")
