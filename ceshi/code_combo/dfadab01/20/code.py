# Combo variant of dfadab01 (index 20)
# combo_spec: D1/1+D8/2
# Name: 象限旋转+象限色

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
            [1, 0, 0, 0],
            [1, 1, 1, 0],
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
QUADRANT_ROTATION_K: dict[tuple[int, int], int] = {
    (0, 0): 0,
    (0, 1): 3,
    (1, 0): 1,
    (1, 1): 2,
}

GRID_SIZE = 20
AXIS_COUNTS = (4, 5)
MAX_GENERATION_ATTEMPTS = 500
LEGEND_PROBABILITY = 0.55
MARKER_COLORS = tuple(sorted(GLYPHS))
OUTPUT_COLORS = tuple(sorted(set(QUADRANT_GLYPH_COLORS.values())))
ALLOWED_INPUT_COLORS = {0, *MARKER_COLORS, *OUTPUT_COLORS}
FOCUS_MARKER = 5
SUPPORT_MARKER = 2

# Design analysis for the fused D1/1 + D8/2 variant
# a) Concrete behavior and trace:
#    - The input still contains isolated marker pixels on black plus an optional legend consisting
#      of one pre-drawn 4x4 glyph and a matching label marker at (+4, +4). The marker color still
#      names the glyph family, but it no longer determines either the final orientation or the
#      paint color.
#    - Fusion happens at a single decision point: determine the anchor cell's quadrant, then use
#      that same quadrant both to rotate the canonical 4x4 mask and to choose the output color.
#      Top-left = unrotated + color 4, top-right = 90 degrees clockwise + color 1, bottom-left =
#      270 degrees clockwise + color 6, bottom-right = 180 degrees + color 7.
#    - Small trace: marker 5 at (2, 2) paints the asymmetric canonical shape in color 4. Another
#      marker 5 at (2, 12) paints the same family rotated clockwise in color 1. Another at
#      (12, 2) paints the 270-degree version in color 6. Under the base task, all three would
#      have produced the same orientation/color pair determined only by color 5 itself.
# b) Complexity calibration:
#    - The new latent variable is still only the coarse 2x2 quadrant partition, so the rule stays
#      inferable from 2-4 examples. Using one shared controller for both orientation and color
#      creates true fusion rather than two unrelated edits.
#    - To keep rotation observable, marker 5 uses an asymmetric 8-pixel canonical glyph. To keep
#      the color rule separately observable, the generator also repeats a rotation-symmetric marker
#      (marker 2) across quadrants, so the learner sees "same shape, new color" in addition to
#      "same marker, new orientation and color."
#    - Output colors remain limited to the original four glyph colors, avoiding a combinatorial
#      remapping burden while still making the quadrant-conditioned recoloring visible.
# c) Implicit-assumption review of the original solver:
#    - The base `GLYPHS` map bundled a final color with each mask. That hidden assumption breaks
#      here because paint color now depends on position, so the solver stores canonical masks only
#      and computes color from the anchor cell at render time.
#    - The original expansion loop assumed one fixed orientation per marker family. That also
#      breaks, so masks must be rotated dynamically from the marker position before clipping.
#    - `_has_source_glyph()` in the base code implicitly assumed a canonical legend sample and a
#      marker-color-specific output color. In the combo rule it must test the rotated mask and the
#      quadrant-derived color at the legend sample's own top-left corner.
#    - The marker loop still assumes the label marker is exactly one row and one column beyond a
#      4x4 legend sample. The generator preserves that layout and keeps legend samples fully
#      inside one quadrant so the combined rotation/color cue is unambiguous.
# d) Placement legality:
#    - Real markers are placed on a regular 4x4-or-larger lattice inside a 20x20 board, so 4x4
#      glyph footprints never overlap and all required focus/support examples fit fully in-bounds.
#    - Row and column positions are chosen so each half of the board contains at least two marker
#      anchors. This guarantees enough cells to place both the asymmetric focus marker and the
#      symmetric support marker in every quadrant without collisions.
#    - A legend is only added in an empty 5x5 slot that is zero in both the input and the expected
#      output, preventing accidental overlap with real expansions or false legend matches.
# e) Visual observability:
#    - Every generated instance contains marker 5 in all four quadrants, so the same input color
#      yields four differently oriented and differently colored outputs. That exposes the fused
#      position rule directly.
#    - Every generated instance also contains marker 2 in all four quadrants. Because marker 2 is
#      rotation-symmetric, those examples isolate the color-by-quadrant behavior even when the
#      rotation behavior is already visible through marker 5.
#    - The generator rejects cases where the combo output collapses to any single-dimension
#      baseline, so training examples must genuinely exercise both control signals.
# f) Visual cleanliness:
#    - The canvas stays sparse: one marker per lattice cell, no overlapping glyphs, and at most
#      one legend placed off the lattice. This keeps quadrant structure readable at a glance.
#    - Only two families are mandatory across quadrants, which is enough evidence without filling
#      every cell with changing colors and orientations.
# g) Change inferability:
#    - Color changes are tied to a visible spatial partition rather than to hidden numeric order.
#      The same four output colors recur in all samples with stable quadrant meaning.
#    - Rotation and recoloring share the same triggering condition, so a solver can infer one
#      compact rule: "quadrant controls how the chosen glyph family is expressed."
# Additional review of new solver assumptions:
#    - `main()` assumes the quadrant is computed from the marker anchor, not from the visible
#      clipped remainder of a glyph. `generate_input()` avoids edge clipping for all real markers,
#      so that assumption is never stressed by ambiguous examples.
#    - `main()` assumes any nonzero 4x4 legend sample already obeys the same combo rule at its own
#      position. `_draw_legend()` constructs legends exactly that way.
#    - `main()` assumes even board dimensions so the quadrant split is deterministic. The generator
#      uses a fixed 20x20 board, and the solver asserts even dimensions on arbitrary inputs.


def _quadrant_key(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    return int(r >= h // 2), int(c >= w // 2)


def _rotation_k(shape: tuple[int, int], r: int, c: int) -> int:
    return QUADRANT_ROTATION_K[_quadrant_key(shape, int(r), int(c))]


def _glyph_color_for_position(shape: tuple[int, int], r: int, c: int) -> int:
    return QUADRANT_GLYPH_COLORS[_quadrant_key(shape, int(r), int(c))]


def _rotated_mask(shape: tuple[int, int], r: int, c: int, mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, k=_rotation_k(shape, r, c))


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    rotated = _rotated_mask(g.shape, r0, c0, mask)
    color = _glyph_color_for_position(g.shape, r0, c0)
    return bool(np.all(patch[rotated] == color) and np.all(patch[~rotated] != color))


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


def _extract_real_markers(g: np.ndarray) -> tuple[list[tuple[int, int, int]], bool]:
    markers: list[tuple[int, int, int]] = []
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
            markers.append((rr, cc, marker_color))
    return markers, seen_marker


def _render_markers(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    *,
    use_rotation: bool,
    use_quadrant_colors: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        mask = GLYPHS[marker_color]
        paint_mask = _rotated_mask(shape, r, c, mask) if use_rotation else mask
        glyph_color = (
            _glyph_color_for_position(shape, r, c)
            if use_quadrant_colors
            else BASE_GLYPH_COLORS[marker_color]
        )
        _paint_clipped(out, r, c, glyph_color, paint_mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker into its 4x4 glyph, with both orientation and paint color controlled by quadrant.

    Input structure:
    - Black background with isolated marker pixels in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one legend sample: a 4x4 glyph already painted in its quadrant's
      output color, plus a same-family label marker one row and one column beyond the sample's
      bottom-right corner.

    Rule:
    - Marker color still chooses the canonical glyph family, but the marker's quadrant decides how
      that family is expressed: top-left keeps the canonical form and color 4, top-right rotates
      90 degrees clockwise and uses color 1, bottom-left rotates 270 degrees clockwise and uses
      color 6, bottom-right rotates 180 degrees and uses color 7.
    - The marker cell remains the top-left anchor of the 4x4 placement window, and any cells that
      fall outside the board are clipped away.
    - Legend glyphs and their below-right label markers are removed rather than copied to output.

    Key insight:
    - The marker hue only tells you which glyph family to start from; coarse position supplies the
      shared control signal that determines both the orientation and the final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0
    assert set(np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    markers, seen_marker = _extract_real_markers(g)
    out = _render_markers(
        g.shape,
        markers,
        use_rotation=True,
        use_quadrant_colors=True,
    )

    assert seen_marker
    return out


def _sample_axis_positions(rng: random.Random, size: int, count: int) -> list[int] | None:
    half = size // 2
    for _ in range(240):
        gaps = [rng.randint(4, 6) for _ in range(count - 1)]
        span = sum(gaps)
        max_start = size - 4 - span
        if max_start < 0:
            continue
        start = rng.randint(0, max_start)
        positions = [start]
        for gap in gaps:
            positions.append(positions[-1] + gap)
        top_count = sum(pos < half for pos in positions)
        bottom_count = sum(pos >= half for pos in positions)
        if top_count < 2 or bottom_count < 2:
            continue
        return positions
    return None


def _quadrant_cells(
    shape: tuple[int, int],
    rows: list[int],
    cols: list[int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {key: [] for key in QUADRANT_GLYPH_COLORS}
    for r in rows:
        for c in cols:
            cells[_quadrant_key(shape, r, c)].append((r, c))
    return cells


def _assign_markers(
    rng: random.Random,
    shape: tuple[int, int],
    rows: list[int],
    cols: list[int],
) -> dict[tuple[int, int], int] | None:
    quadrant_cells = _quadrant_cells(shape, rows, cols)
    if any(len(cells) < 2 for cells in quadrant_cells.values()):
        return None

    assignments: dict[tuple[int, int], int] = {}
    filled_cells: set[tuple[int, int]] = set()

    for cells in quadrant_cells.values():
        chosen = rng.sample(cells, 2)
        assignments[chosen[0]] = FOCUS_MARKER
        assignments[chosen[1]] = SUPPORT_MARKER
        filled_cells.update(chosen)

    extra_palette = [marker for marker in MARKER_COLORS if marker not in {FOCUS_MARKER, SUPPORT_MARKER}]
    extra_count = rng.randint(0, len(extra_palette))
    active_markers = [FOCUS_MARKER, SUPPORT_MARKER, *rng.sample(extra_palette, extra_count)]

    all_cells = [(r, c) for r in rows for c in cols]
    min_fill = max(len(filled_cells) + extra_count, math.ceil(0.75 * len(all_cells)))
    fill_count = rng.randint(min_fill, len(all_cells))
    remaining_cells = [cell for cell in all_cells if cell not in filled_cells]
    rng.shuffle(remaining_cells)
    filled_cells.update(remaining_cells[: fill_count - len(filled_cells)])

    unassigned = [cell for cell in filled_cells if cell not in assignments]
    rng.shuffle(unassigned)
    for marker_color in active_markers[2:]:
        if not unassigned:
            return None
        assignments[unassigned.pop()] = marker_color
    for cell in unassigned:
        assignments[cell] = rng.choice(active_markers)

    return assignments


def _find_legend_slot(
    shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    occupied_output: np.ndarray,
) -> tuple[int, int] | None:
    h, w = shape
    row_bands = (range(0, h // 2 - 4), range(h // 2, h - 4))
    col_bands = (range(0, w // 2 - 4), range(w // 2, w - 4))
    candidates: list[tuple[int, int]] = []
    for rows in row_bands:
        for cols in col_bands:
            for r0 in rows:
                for c0 in cols:
                    label = (r0 + 4, c0 + 4)
                    footprint = {(rr, cc) for rr in range(r0, r0 + 4) for cc in range(c0, c0 + 4)}
                    footprint.add(label)
                    if any(cell in occupied_input for cell in footprint):
                        continue
                    if any(occupied_output[rr, cc] != 0 for rr, cc in footprint):
                        continue
                    candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_legend(grid: np.ndarray, marker_color: int, r0: int, c0: int) -> None:
    rotated = _rotated_mask(grid.shape, r0, c0, GLYPHS[marker_color])
    glyph_color = _glyph_color_for_position(grid.shape, r0, c0)
    patch = grid[r0 : r0 + 4, c0 : c0 + 4]
    patch[rotated] = glyph_color
    grid[r0 + 4, c0 + 4] = marker_color


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        row_count = rng.choice(AXIS_COUNTS)
        col_count = rng.choice(AXIS_COUNTS)
        rows = _sample_axis_positions(rng, GRID_SIZE, row_count)
        cols = _sample_axis_positions(rng, GRID_SIZE, col_count)
        if rows is None or cols is None:
            continue

        assignments = _assign_markers(rng, (GRID_SIZE, GRID_SIZE), rows, cols)
        if assignments is None:
            continue

        markers = sorted((r, c, marker_color) for (r, c), marker_color in assignments.items())
        expected = _render_markers(
            (GRID_SIZE, GRID_SIZE),
            markers,
            use_rotation=True,
            use_quadrant_colors=True,
        )
        base = _render_markers(
            (GRID_SIZE, GRID_SIZE),
            markers,
            use_rotation=False,
            use_quadrant_colors=False,
        )
        rotation_only = _render_markers(
            (GRID_SIZE, GRID_SIZE),
            markers,
            use_rotation=True,
            use_quadrant_colors=False,
        )
        color_only = _render_markers(
            (GRID_SIZE, GRID_SIZE),
            markers,
            use_rotation=False,
            use_quadrant_colors=True,
        )

        if np.array_equal(expected, base):
            continue
        if np.array_equal(expected, rotation_only):
            continue
        if np.array_equal(expected, color_only):
            continue

        focus_quadrants = {
            _quadrant_key((GRID_SIZE, GRID_SIZE), r, c)
            for (r, c), marker_color in assignments.items()
            if marker_color == FOCUS_MARKER
        }
        support_quadrants = {
            _quadrant_key((GRID_SIZE, GRID_SIZE), r, c)
            for (r, c), marker_color in assignments.items()
            if marker_color == SUPPORT_MARKER
        }
        if focus_quadrants != set(QUADRANT_GLYPH_COLORS):
            continue
        if support_quadrants != set(QUADRANT_GLYPH_COLORS):
            continue

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for (r, c), marker_color in assignments.items():
            grid[r, c] = marker_color

        if rng.random() < LEGEND_PROBABILITY:
            slot = _find_legend_slot(grid.shape, set(assignments), expected)
            if slot is not None:
                legend_choices = [FOCUS_MARKER, SUPPORT_MARKER]
                extras = [marker for marker in sorted(set(assignments.values())) if marker not in legend_choices]
                legend_choices.extend(extras)
                _draw_legend(grid, rng.choice(legend_choices), *slot)

        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a non-degenerate combo instance for dfadab01")
