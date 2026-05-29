# Combo variant of dfadab01 (index 33)
# combo_spec: D1/1+D5/0+D8/2
# Name: 象限旋转+旋转变体字形+象限色

from __future__ import annotations

import math
import random

import numpy as np


BASE_MASK = np.array(
    [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
    ],
    dtype=bool,
)

MARKER_ROTATIONS: dict[int, int] = {2: 0, 3: 1, 5: 2, 8: 3}
FIXED_MARKER_COLORS: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
QUADRANT_ROTATIONS: dict[tuple[int, int], int] = {
    (0, 0): 0,
    (0, 1): 3,
    (1, 0): 1,
    (1, 1): 2,
}
QUADRANT_COLORS: dict[tuple[int, int], int] = {
    (0, 0): 4,
    (0, 1): 1,
    (1, 0): 6,
    (1, 1): 7,
}

MARKER_COLORS = tuple(MARKER_ROTATIONS.keys())
OUTPUT_COLORS = tuple(sorted(set(QUADRANT_COLORS.values())))
ALLOWED_INPUT_COLORS = {0, *MARKER_COLORS, *OUTPUT_COLORS}
GRID_SIZE = 20
LEGEND_PROBABILITY = 0.55
MAX_GENERATION_ATTEMPTS = 400

# Design analysis for combo D1/1 + D5/0 + D8/2
# a) Concrete behavior and trace:
#    - The canvas still shows isolated marker pixels in colors 2/3/5/8 on black, optionally plus
#      one legend made of a pre-drawn 4x4 glyph and a same-rule label marker at (+4, +4).
#    - Marker color no longer selects one of four unrelated masks. Instead all four marker types
#      are quarter-turns of the same asymmetric 4x4 base glyph: 2/3/5/8 mean base rotations
#      0/90/180/270 degrees.
#    - Quadrant contributes two more effects at once. Its location adds an extra rotation
#      (TL 0, TR 90 clockwise, BL 270 clockwise, BR 180) and also chooses the output color
#      (TL 4, TR 1, BL 6, BR 7).
#    - Example trace on a 20x20 grid: marker 3 at (1, 1) uses base rot90, no extra quadrant
#      rotation, and paints color 4. Marker 3 at (1, 12) uses the same base rot90, then receives
#      the TR extra turn, so the visible glyph becomes rot0 and paints color 1. In the original
#      task those two cells would have produced the same fixed blue glyph everywhere.
# b) Complexity calibration:
#    - The fused task has exactly three latent factors with four states each: marker base rotation,
#      quadrant extra rotation, and quadrant color. All factors are small and visibly repeated.
#    - The shared base mask is intentionally non-symmetric, so every quarter-turn is distinct and
#      the D1 quadrant rotation cannot collapse away.
#    - The generator forces two complementary witnesses: one marker type repeated across all four
#      quadrants, and one quadrant containing all four marker types. This separates "marker decides
#      base orientation" from "quadrant decides extra rotation and color".
# c) Implicit-assumption audit of the original solver:
#    - The original `GLYPHS` table bundled final mask and final color together. That assumption is
#      broken twice here: mask depends on marker rotation plus quadrant rotation, and color depends
#      only on quadrant. The solver therefore computes both at runtime from `(marker_color, r, c)`.
#    - `_has_source_glyph(...)` in the source task matched legends against one canonical mask/color
#      pair. In the combo variant a legend patch must already reflect the composed transformation at
#      its own top-left anchor, so legend matching must call the same mask/color helpers as normal
#      rendering.
#    - `_paint_clipped(...)` remains valid because clipping is still purely local after the final
#      mask is resolved.
#    - The main marker loop now relies on an even grid split to define quadrants unambiguously.
#      The generator fixes the board to 20x20 so midpoint-based quadrants are stable.
# d) Placement legality:
#    - Real markers still live on a regular lattice with gaps 4-6, so 4x4 glyph footprints never
#      overlap. Mandatory evidence markers are chosen from distinct lattice cells.
#    - The generator requires at least two lattice rows and columns on each side of the midpoint,
#      guaranteeing that every quadrant exists and that one quadrant has at least four usable cells.
#    - Optional legends are placed only inside an empty 5x5 area that is zero in both the input
#      marker layer and the expected output, so they cannot cover real expansions or fake overlaps.
# e) Visual observability:
#    - Repeating one marker color across all quadrants makes the same "base rotation" appear in
#      four different final orientations and four different output colors.
#    - Placing all four marker colors inside one quadrant makes several different shapes share one
#      color, proving that marker hue controls geometry while quadrant controls paint color.
#    - Because these two witness patterns are present in every generated instance, a human can
#      infer the factorization from a few examples instead of memorizing 16 independent templates.
# f) Visual cleanliness:
#    - The board stays sparse: input uses single-pixel markers only, output glyphs do not overlap,
#      and legends are confined to empty regions within a single quadrant.
#    - Using one shared asymmetric base glyph avoids a noisy alphabet while still making rotations
#      easy to compare by eye.
# g) Color-rule inferability:
#    - Only four output colors are used, exactly one per quadrant, so the recoloring signal is tied
#      to a coarse spatial partition rather than to hidden numeric properties of the marker colors.
#    - The generator rejects cases that would accidentally match a no-quadrant-color baseline, so
#      samples always contain direct evidence that location, not marker identity, controls color.
# Additional review of new solver assumptions:
#    - `main()` assumes the quadrant is determined by the marker anchor cell, not by the visible
#      clipped remainder of the 4x4 glyph. The generator keeps the mandatory witness markers fully
#      in bounds and always uses the same anchor-based rule when drawing legends.
#    - `main()` assumes a legend glyph is the fully composed final glyph for its anchor position,
#      not merely the base marker rotation. `generate_input()` satisfies that by drawing legends via
#      `_render_assignments`'s same mask/color decomposition.


def _quadrant_key(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    return int(r >= h // 2), int(c >= w // 2)


def _quadrant_rotation_k(shape: tuple[int, int], r: int, c: int) -> int:
    return QUADRANT_ROTATIONS[_quadrant_key(shape, int(r), int(c))]


def _glyph_color_for_position(shape: tuple[int, int], r: int, c: int) -> int:
    return QUADRANT_COLORS[_quadrant_key(shape, int(r), int(c))]


def _mask_for_marker(
    shape: tuple[int, int],
    r: int,
    c: int,
    marker_color: int,
    *,
    quadrant_rotation: bool,
) -> np.ndarray:
    rotation_k = MARKER_ROTATIONS[marker_color]
    if quadrant_rotation:
        rotation_k = (rotation_k + _quadrant_rotation_k(shape, int(r), int(c))) % 4
    return np.rot90(BASE_MASK, k=rotation_k)


def _glyph_color(
    shape: tuple[int, int],
    r: int,
    c: int,
    marker_color: int,
    *,
    quadrant_color: bool,
) -> int:
    if quadrant_color:
        return _glyph_color_for_position(shape, int(r), int(c))
    return FIXED_MARKER_COLORS[marker_color]


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, marker_color: int) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    mask = _mask_for_marker(g.shape, r0, c0, marker_color, quadrant_rotation=True)
    color = _glyph_color(g.shape, r0, c0, marker_color, quadrant_color=True)
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


def _extract_real_markers(g: np.ndarray) -> tuple[list[tuple[int, int, int]], bool]:
    markers: list[tuple[int, int, int]] = []
    seen_marker = False
    for marker_color in MARKER_COLORS:
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, marker_color):
                continue
            markers.append((rr, cc, marker_color))
    return markers, seen_marker


def _render_assignments(
    shape: tuple[int, int],
    assignments: dict[tuple[int, int], int],
    *,
    quadrant_rotation: bool,
    quadrant_color: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for (r, c), marker_color in assignments.items():
        mask = _mask_for_marker(shape, r, c, marker_color, quadrant_rotation=quadrant_rotation)
        color = _glyph_color(shape, r, c, marker_color, quadrant_color=quadrant_color)
        _paint_clipped(out, r, c, color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each control pixel into a shared 4x4 glyph family whose final orientation and color are split across marker identity and quadrant.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one legend entry: a 4x4 glyph already drawn in its final output
      appearance plus a same-type label marker one row and one column beyond the glyph's
      bottom-right corner.

    Rule:
    - All four marker types come from one asymmetric 4x4 base glyph. Marker colors 2, 3, 5, and 8
      choose the base glyph rotations 0/90/180/270 degrees.
    - Before painting, rotate that marker-selected glyph once more according to the marker's
      quadrant: top-left adds 0 degrees, top-right adds 90 clockwise, bottom-left adds 270
      clockwise, and bottom-right adds 180.
    - Paint the resulting glyph with its top-left corner anchored at the marker cell, clipping at
      the grid boundary if necessary, and color it only by quadrant: TL->4, TR->1, BL->6, BR->7.
    - Remove legend glyphs and suppress their below-right label markers instead of expanding them.

    Key insight:
    - Marker hue and coarse position contribute different parts of the output. Marker color tells
      you which base rotation to start from, while quadrant simultaneously decides the extra
      rotation and the paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0
    assert set(np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    markers, seen_marker = _extract_real_markers(g)
    assignments = {(r, c): marker_color for r, c, marker_color in markers}
    out = _render_assignments(
        g.shape,
        assignments,
        quadrant_rotation=True,
        quadrant_color=True,
    )

    assert seen_marker
    return out


def _sample_axis_positions(rng: random.Random, size: int, count: int) -> list[int] | None:
    half = size // 2
    for _ in range(200):
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
        bottom_count = count - top_count
        if top_count < 2 or bottom_count < 2:
            continue
        return positions
    return None


def _build_assignments(
    rng: random.Random,
    row_positions: list[int],
    col_positions: list[int],
) -> tuple[dict[tuple[int, int], int], int, tuple[int, int]] | None:
    shape = (GRID_SIZE, GRID_SIZE)
    all_cells = [(r, c) for r in row_positions for c in col_positions]
    quadrant_cells = {quadrant: [] for quadrant in QUADRANT_ROTATIONS}
    for cell in all_cells:
        quadrant_cells[_quadrant_key(shape, cell[0], cell[1])].append(cell)

    if any(len(cells) < 1 for cells in quadrant_cells.values()):
        return None

    focus_marker = rng.choice(MARKER_COLORS)
    focus_quadrant = rng.choice(tuple(QUADRANT_ROTATIONS.keys()))
    if len(quadrant_cells[focus_quadrant]) < len(MARKER_COLORS):
        return None

    assignments: dict[tuple[int, int], int] = {}
    quartet_cells = rng.sample(quadrant_cells[focus_quadrant], k=len(MARKER_COLORS))
    for cell, marker_color in zip(quartet_cells, MARKER_COLORS):
        assignments[cell] = marker_color

    for quadrant, cells in quadrant_cells.items():
        if quadrant == focus_quadrant:
            continue
        available = [cell for cell in cells if cell not in assignments]
        if not available:
            return None
        assignments[rng.choice(available)] = focus_marker

    min_fill = max(len(assignments), math.ceil(0.7 * len(all_cells)))
    filled_count = rng.randint(min_fill, len(all_cells))
    remaining = [cell for cell in all_cells if cell not in assignments]
    extra_cells = rng.sample(remaining, filled_count - len(assignments))
    for cell in extra_cells:
        assignments[cell] = rng.choice(MARKER_COLORS)

    return assignments, focus_marker, focus_quadrant


def _find_legend_slot(
    rng: random.Random,
    inp: np.ndarray,
    expected: np.ndarray,
) -> tuple[int, int] | None:
    h, w = inp.shape
    row_bands = (range(0, h // 2 - 4), range(h // 2, h - 4))
    col_bands = (range(0, w // 2 - 4), range(w // 2, w - 4))
    candidates: list[tuple[int, int]] = []
    for rows in row_bands:
        for cols in col_bands:
            for r0 in rows:
                for c0 in cols:
                    if np.any(inp[r0 : r0 + 4, c0 : c0 + 4] != 0):
                        continue
                    if inp[r0 + 4, c0 + 4] != 0:
                        continue
                    if np.any(expected[r0 : r0 + 4, c0 : c0 + 4] != 0):
                        continue
                    if expected[r0 + 4, c0 + 4] != 0:
                        continue
                    candidates.append((r0, c0))
    if not candidates:
        return None
    return rng.choice(candidates)


def _draw_legend(inp: np.ndarray, marker_color: int, r0: int, c0: int) -> None:
    mask = _mask_for_marker(inp.shape, r0, c0, marker_color, quadrant_rotation=True)
    color = _glyph_color(inp.shape, r0, c0, marker_color, quadrant_color=True)
    patch = inp[r0 : r0 + 4, c0 : c0 + 4]
    patch[mask] = color
    inp[r0 + 4, c0 + 4] = marker_color


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        row_count = 4 if rng.random() < 0.75 else 5
        col_count = 4 if rng.random() < 0.75 else 5
        row_positions = _sample_axis_positions(rng, GRID_SIZE, row_count)
        col_positions = _sample_axis_positions(rng, GRID_SIZE, col_count)
        if row_positions is None or col_positions is None:
            continue

        built = _build_assignments(rng, row_positions, col_positions)
        if built is None:
            continue
        assignments, focus_marker, focus_quadrant = built

        expected = _render_assignments(
            (GRID_SIZE, GRID_SIZE),
            assignments,
            quadrant_rotation=True,
            quadrant_color=True,
        )
        no_extra_rotation = _render_assignments(
            (GRID_SIZE, GRID_SIZE),
            assignments,
            quadrant_rotation=False,
            quadrant_color=True,
        )
        no_quadrant_color = _render_assignments(
            (GRID_SIZE, GRID_SIZE),
            assignments,
            quadrant_rotation=True,
            quadrant_color=False,
        )
        marker_only = _render_assignments(
            (GRID_SIZE, GRID_SIZE),
            assignments,
            quadrant_rotation=False,
            quadrant_color=False,
        )
        if np.array_equal(expected, no_extra_rotation):
            continue
        if np.array_equal(expected, no_quadrant_color):
            continue
        if np.array_equal(expected, marker_only):
            continue

        focus_quadrants = {
            _quadrant_key((GRID_SIZE, GRID_SIZE), r, c)
            for (r, c), marker_color in assignments.items()
            if marker_color == focus_marker
        }
        if focus_quadrants != set(QUADRANT_ROTATIONS.keys()):
            continue

        focus_quadrant_markers = {
            marker_color
            for (r, c), marker_color in assignments.items()
            if _quadrant_key((GRID_SIZE, GRID_SIZE), r, c) == focus_quadrant
        }
        if focus_quadrant_markers != set(MARKER_COLORS):
            continue

        inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for (r, c), marker_color in assignments.items():
            inp[r, c] = marker_color

        if rng.random() < LEGEND_PROBABILITY:
            slot = _find_legend_slot(rng, inp, expected)
            if slot is not None:
                legend_marker = rng.choice(list(set(assignments.values())))
                _draw_legend(inp, legend_marker, *slot)

        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid dfadab01 combo instance")
