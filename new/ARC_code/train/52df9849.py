from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg
import random


def _bbox_footprint(g: np.ndarray, color: int, bg: int) -> np.ndarray | None:
    coords = np.argwhere(g == int(color))
    assert coords.size, "expected color pixels"
    r0, c0, r1, c1 = bbox_coords(coords)
    footprint = np.array(
        [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)],
        dtype=int,
    )
    values = g[footprint[:, 0], footprint[:, 1]]
    if np.any(values == int(bg)) or not np.any(values != int(color)):
        return None
    return footprint


def _diagonal_footprint(g: np.ndarray, color: int, bg: int) -> np.ndarray | None:
    coords = np.argwhere(g == int(color))
    assert coords.size, "expected color pixels"
    if coords.shape[0] < 2:
        return None

    r0, c0, r1, c1 = bbox_coords(coords)
    candidates: list[np.ndarray] = []
    if (r1 - r0) == (c1 - c0):
        if np.all(coords[:, 0] - coords[:, 1] == coords[0, 0] - coords[0, 1]):
            candidates.append(np.array([(r0 + i, c0 + i) for i in range(r1 - r0 + 1)], dtype=int))
        if np.all(coords[:, 0] + coords[:, 1] == coords[0, 0] + coords[0, 1]):
            candidates.append(np.array([(r0 + i, c1 - i) for i in range(r1 - r0 + 1)], dtype=int))

    if not candidates:
        return None
    assert len(candidates) == 1, "diagonal footprint must be unique"

    footprint = candidates[0]
    values = g[footprint[:, 0], footprint[:, 1]]
    if np.any(values == int(bg)) or not np.any(values != int(color)):
        return None
    return footprint


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each partially hidden simple foreground shape by recoloring the occluding pixels in its footprint.

    Input structure
    - The grid has a dominant background and several solid foreground shapes.
    - Some shapes are partially hidden by other foreground colors, but their full
      footprint is still indicated by a foreground-filled bbox or by a foreground-
      filled 45-degree diagonal line.

    Rule
    - For each foreground color, complete its full rectangular bbox when that bbox
      contains no background and includes other colors.
    - Otherwise, complete its 45-degree diagonal line when all visible pixels lie
      on that line and the hidden line cells are foreground, not background.
    - Recolor only those hidden footprint pixels; leave all other pixels unchanged.

    Key insight
    - The occluded cells are exactly the non-background cells inside a simple
      shape's implied footprint, so completing shapes is a local recoloring problem.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    planned: dict[tuple[int, int], int] = {}
    for color in colors_present(g, exclude={bg}):
        footprint = _bbox_footprint(g, color, bg)
        if footprint is None:
            footprint = _diagonal_footprint(g, color, bg)
        if footprint is None:
            continue

        for r, c in footprint:
            rr, cc = int(r), int(c)
            if int(g[rr, cc]) == int(color):
                continue
            previous = planned.get((rr, cc))
            assert previous is None or previous == int(color), "completed shapes must not conflict"
            planned[(rr, cc)] = int(color)

    assert planned, "expected at least one hidden foreground footprint to complete"

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = int(color)
    return out

# --------------- generator (att1, Claude) ---------------

# Grid and palette.
GRID_HEIGHT = 16
GRID_WIDTH = 16
BG_COLOR = 7
COLOR_IDS = tuple(range(10))

# Role counts.
FG_COLOR_COUNT_MIN = 2
FG_COLOR_COUNT_MAX = 5
COMPLETABLE_COUNT_MIN = 1
COMPLETABLE_COUNT_MAX = 2
OCCLUDER_COUNT_MIN = 1
OCCLUDER_COUNT_MAX = 3

# Completable shape bounds.
COMPLETABLE_SHAPE_TYPES = ("rect", "diag")
RECT_HEIGHT_MIN = 1
RECT_HEIGHT_MAX = 6
RECT_WIDTH_MIN = 1
RECT_WIDTH_MAX = 16
DIAGONAL_TYPES = ("main", "anti")
DIAGONAL_LENGTH_MIN = 2
DIAGONAL_LENGTH_MAX = 15

# Occluder bounds and validity thresholds.
OCCLUDER_HEIGHT_MIN = 1
OCCLUDER_HEIGHT_MAX = 6
OCCLUDER_WIDTH_MIN = 1
OCCLUDER_WIDTH_MAX = 6
HIDDEN_FRACTION_MIN = 0.10
HIDDEN_FRACTION_MAX = 0.57

# Retry limits.
OCCLUDER_PLACEMENT_TRIES = 100
GENERATION_ATTEMPTS = 300


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _choose_role_colors() -> tuple[list[int], list[int]] | None:
    """Sample completable and occluder colors, rejecting unsupported role counts."""
    all_fg = [c for c in COLOR_IDS if c != BG_COLOR]
    n_colors = random.randint(FG_COLOR_COUNT_MIN, FG_COLOR_COUNT_MAX)
    colors = random.sample(all_fg, n_colors)
    random.shuffle(colors)

    max_completable = min(COMPLETABLE_COUNT_MAX, n_colors - 1)
    n_comp = random.randint(COMPLETABLE_COUNT_MIN, max_completable)
    n_occ = n_colors - n_comp
    if n_occ < OCCLUDER_COUNT_MIN or n_occ > OCCLUDER_COUNT_MAX:
        return None
    return colors[:n_comp], colors[n_comp:]


def _make_completable_shape(color: int):
    """Return (color, shape_type, footprint_cells, bbox_or_none)."""
    shape_type = random.choice(COMPLETABLE_SHAPE_TYPES)
    if shape_type == "rect":
        h = random.randint(RECT_HEIGHT_MIN, RECT_HEIGHT_MAX)
        w = random.randint(RECT_WIDTH_MIN, RECT_WIDTH_MAX)
        r0 = random.randint(0, GRID_HEIGHT - h)
        c0 = random.randint(0, GRID_WIDTH - w)
        cells = _rect_cells(r0, c0, h, w)
        return color, "rect", cells, (r0, c0, r0 + h - 1, c0 + w - 1)

    length = random.randint(DIAGONAL_LENGTH_MIN, DIAGONAL_LENGTH_MAX)
    diag_type = random.choice(DIAGONAL_TYPES)
    if diag_type == "main":
        r0 = random.randint(0, GRID_HEIGHT - length)
        c0 = random.randint(0, GRID_WIDTH - length)
        cells = {(r0 + i, c0 + i) for i in range(length)}
    else:
        r0 = random.randint(0, GRID_HEIGHT - length)
        c0 = random.randint(length - 1, GRID_WIDTH - 1)
        cells = {(r0 + i, c0 - i) for i in range(length)}
    return color, "diag", cells, None


def _make_completable_shapes(comp_colors: list[int]):
    shapes = [_make_completable_shape(color) for color in comp_colors]
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i][2] & shapes[j][2]:
                return None
    return shapes


def _sample_overlapping_occluder(color: int, comp_shapes):
    """Sample one filled rectangle that overlaps at least one completable footprint."""
    for _ in range(OCCLUDER_PLACEMENT_TRIES):
        h = random.randint(OCCLUDER_HEIGHT_MIN, OCCLUDER_HEIGHT_MAX)
        w = random.randint(OCCLUDER_WIDTH_MIN, OCCLUDER_WIDTH_MAX)
        r0 = random.randint(0, GRID_HEIGHT - h)
        c0 = random.randint(0, GRID_WIDTH - w)
        cells = _rect_cells(r0, c0, h, w)
        if any(cells & comp_cells for _, _, comp_cells, _ in comp_shapes):
            return color, cells
    return None


def _make_occluder_shapes(occ_colors: list[int], comp_shapes):
    occluders = []
    for color in occ_colors:
        shape = _sample_overlapping_occluder(color, comp_shapes)
        if shape is None:
            return None
        occluders.append(shape)
    return occluders


def _occluder_cells(occ_shapes) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for _, shape_cells in occ_shapes:
        cells |= shape_cells
    return cells


def _completable_overlap_is_valid(comp_shapes, occ_shapes) -> bool:
    all_occ_cells = _occluder_cells(occ_shapes)
    for _, _, cells, _ in comp_shapes:
        overlap = cells & all_occ_cells
        if not overlap:
            return False
        frac = len(overlap) / len(cells)
        if frac < HIDDEN_FRACTION_MIN or frac > HIDDEN_FRACTION_MAX:
            return False
    return True


def _paint_cells(grid: np.ndarray, color: int, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = color


def _render_pair(comp_shapes, occ_shapes) -> tuple[np.ndarray, np.ndarray]:
    """Render (input, output): occluders win in input, completable shapes win in output."""
    output = np.full((GRID_HEIGHT, GRID_WIDTH), BG_COLOR, dtype=int)
    for color, cells in occ_shapes:
        _paint_cells(output, color, cells)
    for color, _, cells, _ in comp_shapes:
        _paint_cells(output, color, cells)

    inp = np.full((GRID_HEIGHT, GRID_WIDTH), BG_COLOR, dtype=int)
    for color, _, cells, _ in comp_shapes:
        _paint_cells(inp, color, cells)
    for color, cells in occ_shapes:
        _paint_cells(inp, color, cells)

    return inp, output


def _occluders_remain_inert(inp: np.ndarray, occ_shapes) -> bool:
    """Reject occluders whose input bbox can be completed as another shape."""
    for color, cells in occ_shapes:
        coords = np.array(list(cells))
        r0, c0, r1, c1 = bbox_coords(coords)
        region = inp[r0 : r1 + 1, c0 : c1 + 1]
        has_bg = bool(np.any(region == BG_COLOR))
        has_other = bool(np.any((region != BG_COLOR) & (region != color)))
        if has_other and not has_bg:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    roles = _choose_role_colors()
    if roles is None:
        return None
    comp_colors, occ_colors = roles

    comp_shapes = _make_completable_shapes(comp_colors)
    if comp_shapes is None:
        return None

    occ_shapes = _make_occluder_shapes(occ_colors, comp_shapes)
    if occ_shapes is None:
        return None

    if not _completable_overlap_is_valid(comp_shapes, occ_shapes):
        return None

    inp, output = _render_pair(comp_shapes, occ_shapes)
    if not _occluders_remain_inert(inp, occ_shapes):
        return None

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
