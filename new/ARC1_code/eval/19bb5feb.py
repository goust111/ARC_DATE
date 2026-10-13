from collections import Counter

import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, colors_present, paste
import random


def _unique_max_color(counts: dict[int, int], message: str) -> int:
    assert counts, message
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, message
    return int(winners[0])


def _boundary_background(g: np.ndarray) -> int:
    boundary = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    return _unique_max_color(Counter(map(int, boundary)), "expected a unique boundary background color")


def _half_index(marker_lo: int, marker_hi: int, panel_lo: int, panel_hi: int) -> int:
    marker_center2 = marker_lo + marker_hi
    panel_center2 = panel_lo + panel_hi
    assert marker_center2 != panel_center2, "marker must lie clearly in one half of the panel"
    return 0 if marker_center2 < panel_center2 else 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize the colored 2x2 markers inside a filled panel as a 2x2 position map.

    Input structure
    - The boundary has one empty background color.
    - The non-background pixels form one solid rectangular panel, whose dominant
      non-background color is the panel fill.
    - Every other color appears as one solid 2x2 marker inside the panel.

    Rule
    - Output a 2x2 grid initialized to the background color.
    - Split the panel bbox into top/bottom and left/right halves.
    - Put each marker color into the output cell matching the half that contains
      that marker's center.

    Key insight
    - The large panel is an enlarged 2x2 canvas; the small colored squares mark
      which coarse cells should be non-background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _boundary_background(g)
    panel_box = bbox(g != bg)
    assert panel_box is not None, "expected a non-background panel"
    r0, c0, r1, c1 = panel_box
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] != bg), "panel bbox must be filled"

    color_counts = {color: int((g == color).sum()) for color in colors_present(g, exclude={bg})}
    panel_color = _unique_max_color(color_counts, "expected a unique largest panel color")

    out = np.full((2, 2), bg, dtype=int)
    marker_colors = [color for color in color_counts if color != panel_color]
    assert marker_colors, "expected at least one 2x2 marker"

    for color in marker_colors:
        mask = g == color
        marker_box = bbox(mask)
        assert marker_box is not None, "marker color must be present"
        mr0, mc0, mr1, mc1 = marker_box
        assert int(mask.sum()) == 4, "each marker must have area 4"
        assert bbox_h(marker_box) == 2 and bbox_w(marker_box) == 2, "each marker must be 2x2"
        assert np.all(mask[mr0 : mr1 + 1, mc0 : mc1 + 1]), "each marker bbox must be solid"

        out_r = _half_index(mr0, mr1, r0, r1)
        out_c = _half_index(mc0, mc1, c0, c1)
        assert out[out_r, out_c] == bg, "at most one marker may occupy each output cell"
        out[out_r, out_c] = color

    return out

# --------------- generator (o5x) ---------------
# Fixed colors and 2x2 output lattice.
BG_COLOR, PANEL_COLOR = 0, 8
MARKER_PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
OUTPUT_SIDE = 2
OUTPUT_CELLS = [(r, c) for r in range(OUTPUT_SIDE) for c in range(OUTPUT_SIDE)]

# Size/count ranges.
MARKER_COUNT_RANGE = (2, 4)
PANEL_H_RANGE, PANEL_W_RANGE = (8, 12), (8, 13)
GRID_H_RANGE, GRID_W_RANGE = (13, 16), (15, 16)

# Placement and retry constraints.
MARKER_SIZE = 2
MARGIN_RANGE = (1, 5)
MIN_CENTER_DISTANCE2, GENERATE_ATTEMPTS = 4, 300


def _gen_margin_pair(inner: int, outer_min: int, outer_max: int) -> tuple[int, int] | None:
    """Choose before/after margins that fit one sampled outer canvas size."""
    margin_min, margin_max = MARGIN_RANGE
    feasible_outer = [outer for outer in range(outer_min, outer_max + 1) if 2 * margin_min <= outer - inner <= 2 * margin_max]
    if not feasible_outer:
        return None
    outer = random.choice(feasible_outer)
    total = outer - inner
    starts = [lo for lo in range(margin_min, margin_max + 1) if margin_min <= total - lo <= margin_max]
    if not starts:
        return None
    before = random.choice(starts)
    return before, total - before


def _gen_axis_positions(length: int, half: int) -> list[int]:
    """Return legal 2-cell marker starts for one axis half."""
    panel_center2 = length - 1
    positions = []
    margin_min, _ = MARGIN_RANGE
    stop = length - margin_min - MARKER_SIZE + 1
    for start in range(margin_min, stop):
        marker_center2 = 2 * start + MARKER_SIZE - 1
        if (half == 0 and marker_center2 >= panel_center2) or (half == 1 and marker_center2 <= panel_center2):
            continue
        if abs(marker_center2 - panel_center2) < MIN_CENTER_DISTANCE2:
            continue
        positions.append(start)
    return positions


def _choose_markers() -> tuple[list[int], list[tuple[int, int]], np.ndarray]:
    """Sample marker colors/cells and the matching oracle output."""
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    marker_colors = random.sample(MARKER_PALETTE, marker_count)
    cells = random.sample(OUTPUT_CELLS, marker_count)
    expected = np.full((OUTPUT_SIDE, OUTPUT_SIDE), BG_COLOR, dtype=int)
    for color, (out_r, out_c) in zip(marker_colors, cells):
        expected[out_r, out_c] = color
    return marker_colors, cells, expected


def _build_panel(panel_h: int, panel_w: int, marker_colors: list[int], cells: list[tuple[int, int]]) -> np.ndarray | None:
    """Render the filled panel and place each 2x2 marker in its sampled quadrant."""
    panel = np.full((panel_h, panel_w), PANEL_COLOR, dtype=int)
    row_pos_by_half = {half: _gen_axis_positions(panel_h, half) for half in range(OUTPUT_SIDE)}
    col_pos_by_half = {half: _gen_axis_positions(panel_w, half) for half in range(OUTPUT_SIDE)}
    if any(not row_pos_by_half[h] for h in range(OUTPUT_SIDE)) or any(not col_pos_by_half[h] for h in range(OUTPUT_SIDE)):
        return None
    for color, (out_r, out_c) in zip(marker_colors, cells):
        marker_r = random.choice(row_pos_by_half[out_r])
        marker_c = random.choice(col_pos_by_half[out_c])
        panel[marker_r : marker_r + MARKER_SIZE, marker_c : marker_c + MARKER_SIZE] = color
    return panel


def _gen_attempt() -> np.ndarray | None:
    marker_colors, cells, expected = _choose_markers()
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    row_margins = _gen_margin_pair(panel_h, *GRID_H_RANGE)
    col_margins = _gen_margin_pair(panel_w, *GRID_W_RANGE)
    if row_margins is None or col_margins is None:
        return None

    top_margin, bottom_margin = row_margins
    left_margin, right_margin = col_margins

    panel = _build_panel(panel_h, panel_w, marker_colors, cells)
    if panel is None:
        return None

    grid = np.full((top_margin + panel_h + bottom_margin, left_margin + panel_w + right_margin), BG_COLOR, dtype=int)
    paste(grid, panel, top_margin, left_margin, transparent=None)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 19bb5feb input")
