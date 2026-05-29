# Variant of dfadab01
# Dimension: D8 — 字形字典中标记色与字形色的数值关系 (meta)
# Change: 标记色与字形色形成固定配对：2->4、3->1、5->6、8->7，配对关系为全局常量... → 字形色由标记所在象限决定

# Design analysis
# - concretization and behavior trace:
#   The four 4x4 glyph masks stay exactly the same as in the base task, and the
#   marker color still chooses which mask to stamp. The change is only the paint
#   color: the marker's quadrant now decides it. We use one fixed color per
#   quadrant: top-left -> 4, top-right -> 1, bottom-left -> 6, bottom-right -> 7.
#   A concrete trace shows the difference. If marker 2 appears at (1, 1) and at
#   (12, 1), both outputs are still the "frame" mask, but the upper one is color
#   4 while the lower one is color 6. Under the original logic both would have
#   been color 4 because only the marker color mattered.
# - complexity calibration:
#   The variant adds exactly one new latent variable, quadrant, with only four
#   states. That is enough to create a visible change without making the task
#   branchy. The generator forces one marker type to appear in all four quadrants
#   and also includes extra marker types, so the learner must infer both the mask
#   dictionary and the spatial color rule from repeated evidence.
# - implicit assumptions review:
#   The original `GLYPHS` bundled `(glyph_color, mask)`, which assumed a global
#   marker->color pairing. That assumption is now false, so the solver must store
#   only the masks and compute the color from `(row, col)` at paint time. The
#   legend-skip check also changes: a sample glyph must be recognized by the color
#   predicted from the sample's own top-left position, not from the label marker's
#   color. The non-overlap assumption remains valid because generation still uses
#   grid spacings >= 4.
# - placement legality:
#   Inputs use a 20x20 board with regular marker rows and columns separated by
#   4-6 cells, so every real glyph fits fully and never overlaps another glyph.
#   An optional legend is placed only in an empty 5x5 box that stays inside one
#   quadrant, which guarantees that the legend glyph and its below-right label do
#   not collide with real markers and that the solver can safely suppress the
#   label marker.
# - visual observability:
#   Every generated sample contains at least one marker type repeated across all
#   four quadrants. Therefore a human sees the same shape repeated in multiple
#   colors, which makes the quadrant rule directly observable in 2-3 examples.
#   Extra marker types prevent the task from collapsing into "just recolor one
#   shape by region".
# - cleanliness and color inference:
#   Only four output colors are used, each tied to a large spatial region rather
#   than to hidden numeric properties. There is at most one legend, and it is kept
#   away from the regular marker lattice, so the grid stays sparse and readable.

from __future__ import annotations

import math
import random

import numpy as np


GLYPHS: dict[int, np.ndarray] = {
    2: np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    5: np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    8: np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
}

BASE_GLYPH_COLORS: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
QUADRANT_GLYPH_COLORS: dict[tuple[int, int], int] = {
    (0, 0): 4,
    (0, 1): 1,
    (1, 0): 6,
    (1, 1): 7,
}

GRID_SIZE = 20
MARKER_COLORS = tuple(GLYPHS.keys())
OUTPUT_COLORS = tuple(sorted(set(QUADRANT_GLYPH_COLORS.values())))
ALLOWED_INPUT_COLORS = {0, *MARKER_COLORS, *OUTPUT_COLORS}
AXIS_COUNTS = (3, 4, 5)
AXIS_SPACINGS = (4, 5, 6)
LEGEND_PROBABILITY = 0.55
MAX_RETRIES = 400


def _quadrant_key(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    return int(r >= h // 2), int(c >= w // 2)


def _glyph_color_for_position(shape: tuple[int, int], r: int, c: int) -> int:
    return QUADRANT_GLYPH_COLORS[_quadrant_key(shape, int(r), int(c))]


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    color = _glyph_color_for_position(g.shape, r0, c0)
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _render_assignments(
    shape: tuple[int, int],
    assignments: dict[tuple[int, int], int],
    *,
    quadrant_colors: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for (r, c), marker_color in assignments.items():
        if quadrant_colors:
            glyph_color = _glyph_color_for_position(shape, r, c)
        else:
            glyph_color = BASE_GLYPH_COLORS[marker_color]
        _paint_clipped(out, r, c, glyph_color, GLYPHS[marker_color])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each control pixel into its fixed 4x4 glyph, but color the glyph by
    the marker's quadrant instead of by the marker's own color.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one sample 4x4 glyph plus a same-shape label marker
      at the sample's below-right corner; these legend parts may use any quadrant
      color because the sample's position determines its paint color.

    Rule:
    - The marker color still selects the glyph mask: 2=frame, 3=ring, 5=diagonal
      blocks, 8=X-shape.
    - The glyph's paint color is determined only by the marker position: top-left
      quadrant -> 4, top-right -> 1, bottom-left -> 6, bottom-right -> 7.
    - Stamp each glyph with its top-left corner at the marker pixel, clipping at
      the grid boundary if needed.
    - Remove legend glyphs and do not expand their below-right label markers.

    Key insight:
    - Shape identity and paint color are now split across two cues: marker hue
      tells you which 4x4 mask to use, while coarse spatial location tells you
      which color to paint it with.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0
    assert set(np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, mask in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, mask):
                continue
            glyph_color = _glyph_color_for_position(g.shape, rr, cc)
            _paint_clipped(out, rr, cc, glyph_color, mask)

    assert seen_marker
    return out


def _choose_axis_positions(size: int) -> list[int]:
    choices: list[list[int]] = []
    half = size // 2
    for count in AXIS_COUNTS:
        for spacing in AXIS_SPACINGS:
            max_start = size - 4 - (count - 1) * spacing
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                positions = [start + i * spacing for i in range(count)]
                if min(positions) < half <= max(positions):
                    choices.append(positions)
    assert choices
    return list(random.choice(choices))


def _find_legend_slot(
    shape: tuple[int, int], marker_cells: set[tuple[int, int]]
) -> tuple[int, int] | None:
    h, w = shape
    row_bands = (range(0, h // 2 - 4), range(h // 2, h - 4))
    col_bands = (range(0, w // 2 - 4), range(w // 2, w - 4))
    candidates: list[tuple[int, int]] = []
    for rows in row_bands:
        for cols in col_bands:
            for r0 in rows:
                for c0 in cols:
                    occupied = False
                    for rr in range(r0, r0 + 5):
                        for cc in range(c0, c0 + 5):
                            if (rr, cc) in marker_cells:
                                occupied = True
                                break
                        if occupied:
                            break
                    if not occupied:
                        candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_legend(grid: np.ndarray, marker_color: int, r0: int, c0: int) -> None:
    mask = GLYPHS[marker_color]
    glyph_color = _glyph_color_for_position(grid.shape, r0, c0)
    patch = grid[r0 : r0 + 4, c0 : c0 + 4]
    patch[mask] = glyph_color
    grid[r0 + 4, c0 + 4] = marker_color


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rows = _choose_axis_positions(GRID_SIZE)
        cols = _choose_axis_positions(GRID_SIZE)
        all_cells = [(r, c) for r in rows for c in cols]
        quadrant_cells = {q: [] for q in QUADRANT_GLYPH_COLORS}
        for cell in all_cells:
            quadrant_cells[_quadrant_key((GRID_SIZE, GRID_SIZE), *cell)].append(cell)
        assert all(quadrant_cells.values())

        used_markers = random.sample(MARKER_COLORS, random.randint(2, len(MARKER_COLORS)))
        focus_marker = random.choice(used_markers)

        forced_cells: dict[tuple[int, int], int] = {}
        for q, cells in quadrant_cells.items():
            forced_cells[random.choice(cells)] = focus_marker

        total_cells = len(all_cells)
        min_fill = max(8, math.ceil(0.75 * total_cells))
        num_filled = random.randint(min_fill, total_cells)
        remaining = [cell for cell in all_cells if cell not in forced_cells]
        random.shuffle(remaining)
        filled_cells = set(forced_cells)
        filled_cells.update(remaining[: num_filled - len(forced_cells)])

        assignments = dict(forced_cells)
        spare_cells = [cell for cell in filled_cells if cell not in assignments]
        random.shuffle(spare_cells)
        other_markers = [m for m in used_markers if m != focus_marker]
        if len(spare_cells) < len(other_markers):
            continue
        for cell, marker_color in zip(spare_cells, other_markers):
            assignments[cell] = marker_color
        for cell in filled_cells:
            if cell not in assignments:
                assignments[cell] = random.choice(used_markers)

        focus_quadrants = {
            _quadrant_key((GRID_SIZE, GRID_SIZE), r, c)
            for (r, c), marker_color in assignments.items()
            if marker_color == focus_marker
        }
        if len(focus_quadrants) < 4:
            continue

        expected = _render_assignments((GRID_SIZE, GRID_SIZE), assignments, quadrant_colors=True)
        baseline = _render_assignments((GRID_SIZE, GRID_SIZE), assignments, quadrant_colors=False)
        if np.array_equal(expected, baseline):
            continue

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for (r, c), marker_color in assignments.items():
            grid[r, c] = marker_color

        if random.random() < LEGEND_PROBABILITY:
            slot = _find_legend_slot(grid.shape, set(assignments))
            if slot is not None:
                legend_marker = random.choice(list(set(assignments.values())))
                _draw_legend(grid, legend_marker, *slot)

        if np.array_equal(main(grid), expected):
            return grid

    raise RuntimeError("Failed to generate a valid dfadab01 D8 input")
