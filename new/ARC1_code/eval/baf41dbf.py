from __future__ import annotations

import numpy as np

from arc_utils import bbox, components_mask, get_bg
import random


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _line_and_marker_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 2, "expected one line color and one marker color"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    high = max(counts.values())
    low = min(counts.values())
    assert high > low, "line color must have more pixels than marker color"

    line_colors = [color for color, count in counts.items() if count == high]
    marker_colors = [color for color, count in counts.items() if count == low]
    assert len(line_colors) == 1 and len(marker_colors) == 1

    marker_components = components_mask(g == marker_colors[0], connectivity=4)
    assert marker_components, "expected singleton control markers"
    assert all(comp.shape[0] == 1 for comp in marker_components), "markers must be isolated pixels"

    return line_colors[0], marker_colors[0]


def _full_bar_positions(line_mask: np.ndarray, box: BBox) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = box
    patch = line_mask[r0 : r1 + 1, c0 : c1 + 1]
    horizontal = [r0 + i for i in range(patch.shape[0]) if np.all(patch[i, :])]
    vertical = [c0 + j for j in range(patch.shape[1]) if np.all(patch[:, j])]

    assert r0 in horizontal and r1 in horizontal, "line drawing must have top and bottom bars"
    assert c0 in vertical and c1 in vertical, "line drawing must have left and right bars"

    reconstructed = np.zeros_like(patch, dtype=bool)
    reconstructed[[r - r0 for r in horizontal], :] = True
    reconstructed[:, [c - c0 for c in vertical]] = True
    assert np.array_equal(patch, reconstructed), "line drawing must be exactly full rows and columns"

    return horizontal, vertical


def _marker_sides(marker: Coord, box: BBox) -> list[str]:
    r, c = marker
    r0, c0, r1, c1 = box
    sides = []
    if c == c0 - 1 and r0 <= r <= r1:
        sides.append("left")
    if c == c1 + 1 and r0 <= r <= r1:
        sides.append("right")
    if r == r0 - 1 and c0 <= c <= c1:
        sides.append("top")
    if r == r1 + 1 and c0 <= c <= c1:
        sides.append("bottom")
    return sides


def _target_bbox(g: np.ndarray, original: BBox, markers: list[Coord]) -> BBox:
    height, width = map(int, g.shape)
    r0, c0, r1, c1 = original

    candidate_r0 = {r0}
    candidate_c0 = {c0}
    candidate_r1 = {r1}
    candidate_c1 = {c1}
    for r, c in markers:
        if 0 <= r + 1 < height:
            candidate_r0.add(r + 1)
        if 0 <= r - 1 < height:
            candidate_r1.add(r - 1)
        if 0 <= c + 1 < width:
            candidate_c0.add(c + 1)
        if 0 <= c - 1 < width:
            candidate_c1.add(c - 1)

    candidates: list[BBox] = []
    for new_r0 in candidate_r0:
        for new_c0 in candidate_c0:
            for new_r1 in candidate_r1:
                for new_c1 in candidate_c1:
                    contains_original = (
                        0 <= new_r0 <= r0 <= r1 <= new_r1 < height
                        and 0 <= new_c0 <= c0 <= c1 <= new_c1 < width
                    )
                    if not contains_original:
                        continue
                    if all(len(_marker_sides(marker, (new_r0, new_c0, new_r1, new_c1))) == 1 for marker in markers):
                        candidates.append((new_r0, new_c0, new_r1, new_c1))

    assert len(candidates) == 1, "expected markers to define exactly one final rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch a rectilinear foreground line drawing so its outside edges sit next to the control markers.

    Input structure
    - The background is the dominant color.
    - One foreground color forms a connected rectilinear line drawing made exactly from full horizontal and vertical bars.
    - The other foreground color consists of isolated control pixels outside the drawing.
    Rule
    - Find the unique rectangle that contains the original line drawing and has every control pixel one cell outside one of its sides.
    - Move the original outer bars to the corresponding sides of that rectangle, keeping interior bars fixed.
    - Redraw the bars across the new rectangle and preserve the control pixels.
    Key insight
    - The markers do not get connected; they specify the final bounding sides of a stretched line drawing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    line_color, marker_color = _line_and_marker_colors(g, bg)

    line_mask = g == line_color
    original_box = bbox(line_mask)
    assert original_box is not None, "expected a line drawing"
    horizontal_rows, vertical_cols = _full_bar_positions(line_mask, original_box)

    markers = [tuple(map(int, coord)) for coord in np.argwhere(g == marker_color)]
    assert markers, "expected control markers"
    new_r0, new_c0, new_r1, new_c1 = _target_bbox(g, original_box, markers)

    old_r0, old_c0, old_r1, old_c1 = original_box
    out = g.copy()
    out[line_mask] = bg

    new_horizontal_rows = [
        new_r0 if row == old_r0 else new_r1 if row == old_r1 else row
        for row in horizontal_rows
    ]
    new_vertical_cols = [
        new_c0 if col == old_c0 else new_c1 if col == old_c1 else col
        for col in vertical_cols
    ]
    assert len(new_horizontal_rows) == len(set(new_horizontal_rows))
    assert len(new_vertical_cols) == len(set(new_vertical_cols))

    for row in new_horizontal_rows:
        assert new_r0 <= row <= new_r1
        out[row, new_c0 : new_c1 + 1] = line_color
    for col in new_vertical_cols:
        assert new_c0 <= col <= new_c1
        out[new_r0 : new_r1 + 1, col] = line_color

    assert np.all(out[g == marker_color] == marker_color), "control markers must be preserved"
    return out

# --------------- generator (o5x) ---------------

# Canvas/colors: same-size grids with the fixed observed palette.
GRID_HEIGHT_RANGE = (8, 16)
GRID_WIDTH_RANGE = (15, 17)
BG_COLOR = 0
LINE_COLOR = 3
MARKER_COLOR = 6
# Bars: 2-3 full horizontal/vertical bars with 2-3 cell gaps.
BAR_COUNT_RANGE = (2, 3)
TWO_BAR_GAP_RANGE = (2, 3)
THREE_BAR_GAPS = ((2, 2), (2, 3), (3, 2))
# Expansion/markers: 2-3 left/right/bottom controls, no top expansion.
EXPANDABLE_SIDES = ["left", "right", "bottom"]
SIDE_COUNT_RANGE = (2, 3)
EXPANSION_RANGE = (1, 6)
GRID_MARGIN = 1
# Retry budget: reject rare layouts where markers collide or solver check fails.
GENERATION_RETRY_LIMIT = 300


def _gen_bar_offsets(count: int) -> list[int]:
    if count == 2:
        return [0, random.randint(*TWO_BAR_GAP_RANGE)]
    gap_a, gap_b = random.choice(THREE_BAR_GAPS)
    return [0, gap_a, gap_a + gap_b]


def _gen_valid_expansions(sides: set[str], old_h: int, old_w: int, height: int, width: int) -> list[tuple[int, int, int]]:
    """Return expansion triples that fit inside the required grid margin."""
    left_vals = range(EXPANSION_RANGE[0], EXPANSION_RANGE[1] + 1) if "left" in sides else [0]
    right_vals = range(EXPANSION_RANGE[0], EXPANSION_RANGE[1] + 1) if "right" in sides else [0]
    bottom_vals = range(EXPANSION_RANGE[0], EXPANSION_RANGE[1] + 1) if "bottom" in sides else [0]
    max_h = height - 2 * GRID_MARGIN
    max_w = width - 2 * GRID_MARGIN
    valid = []
    for d_left in left_vals:
        for d_right in right_vals:
            for d_bottom in bottom_vals:
                new_h = old_h + d_bottom
                new_w = old_w + d_left + d_right
                if new_h <= max_h and new_w <= max_w:
                    valid.append((d_left, d_right, d_bottom))
    return valid


def _gen_draw_bars(canvas: np.ndarray, row_positions: list[int], col_positions: list[int], r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    canvas[row_positions, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, col_positions] = color


def _stretch_outer_positions(rows: list[int], cols: list[int], old_box: BBox, new_box: BBox) -> tuple[list[int], list[int]]:
    """Move only original boundary bars to the expanded bbox edges."""
    old_r0, old_c0, old_r1, old_c1 = old_box
    new_r0, new_c0, new_r1, new_c1 = new_box
    new_rows = [new_r0 if r == old_r0 else new_r1 if r == old_r1 else r for r in rows]
    new_cols = [new_c0 if c == old_c0 else new_c1 if c == old_c1 else c for c in cols]
    return new_rows, new_cols


def _gen_markers(sides: set[str], old_box: BBox, new_box: BBox) -> list[Coord]:
    """Place one marker per expanded side in fixed left/right/bottom order."""
    old_r0, old_c0, old_r1, old_c1 = old_box
    _, new_c0, new_r1, new_c1 = new_box
    markers = []
    if "left" in sides:
        markers.append((random.randint(old_r0, old_r1), new_c0 - 1))
    if "right" in sides:
        markers.append((random.randint(old_r0, old_r1), new_c1 + 1))
    if "bottom" in sides:
        markers.append((new_r1 + 1, random.randint(old_c0, old_c1)))
    return markers


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    h_offsets = _gen_bar_offsets(random.randint(*BAR_COUNT_RANGE))
    v_offsets = _gen_bar_offsets(random.randint(*BAR_COUNT_RANGE))
    old_h = h_offsets[-1] + 1
    old_w = v_offsets[-1] + 1

    side_count = random.randint(*SIDE_COUNT_RANGE)
    sides = set(random.sample(EXPANDABLE_SIDES, side_count))
    expansions = _gen_valid_expansions(sides, old_h, old_w, height, width)
    if not expansions:
        return None
    d_left, d_right, d_bottom = random.choice(expansions)

    new_h = old_h + d_bottom
    new_w = old_w + d_left + d_right
    new_r0 = random.randint(GRID_MARGIN, height - new_h - GRID_MARGIN)
    new_c0 = random.randint(GRID_MARGIN, width - new_w - GRID_MARGIN)
    new_r1, new_c1 = new_r0 + new_h - 1, new_c0 + new_w - 1
    new_box = (new_r0, new_c0, new_r1, new_c1)

    old_box = (new_r0, new_c0 + d_left, new_r1 - d_bottom, new_c1 - d_right)
    old_r0, old_c0, old_r1, old_c1 = old_box

    old_rows = [old_r0 + off for off in h_offsets]
    old_cols = [old_c0 + off for off in v_offsets]
    new_rows, new_cols = _stretch_outer_positions(old_rows, old_cols, old_box, new_box)

    inp = np.full((height, width), BG_COLOR, dtype=int)
    out = np.full((height, width), BG_COLOR, dtype=int)
    _gen_draw_bars(inp, old_rows, old_cols, old_r0, old_c0, old_r1, old_c1, LINE_COLOR)
    _gen_draw_bars(out, new_rows, new_cols, new_r0, new_c0, new_r1, new_c1, LINE_COLOR)

    for r, c in _gen_markers(sides, old_box, new_box):
        if inp[r, c] != BG_COLOR or out[r, c] != BG_COLOR:
            return None
        inp[r, c] = MARKER_COLOR
        out[r, c] = MARKER_COLOR

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid baf41dbf input")
