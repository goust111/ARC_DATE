# Variant of dfadab01
# Dimension: D1 — 字形替换变换方式 (transform)
# Change: 每个标记像素被替换为固定4x4字形图案，标记位置为字形左上角，字形使用与标记色不同的固定输出色... → 字形旋转后替换

from __future__ import annotations

import math
import random

import numpy as np


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}

MARKER_COLORS = tuple(GLYPHS.keys())
GLYPH_COLORS = tuple(sorted({glyph_color for glyph_color, _ in GLYPHS.values()}))
ALLOWED_INPUT_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}
GRID_SIZES = (10, 20)
FOCUS_MARKER = 5
MAX_GENERATION_ATTEMPTS = 400

# Design analysis for the D1 conditional-rotation variant
# a) Concrete behavior and trace:
#    - The canvas still contains isolated marker pixels in colors 2/3/5/8 on black, plus an
#      optional legend made of one already-drawn 4x4 glyph and its label marker at (+4, +4).
#    - The change happens exactly at glyph expansion time: before painting a marker's 4x4 glyph,
#      compute which quadrant the marker anchor belongs to. Top-left keeps the canonical glyph,
#      top-right uses a 90° clockwise rotation, bottom-left uses a 270° clockwise rotation, and
#      bottom-right uses a 180° rotation.
#    - Small trace: suppose marker 5 is at (2, 2) in a 20x20 grid. In the base task it always
#      paints the same magenta 4x4 mask. In this variant, marker 5 at (2, 2) paints the canonical
#      form, while another marker 5 at (2, 14) paints the same mask rotated clockwise, and marker
#      5 at (14, 14) paints the 180° form. The input still shows only single pixels; the output is
#      where the quadrant-conditioned rotation becomes visible.
# b) Complexity calibration:
#    - The glyph alphabet size, output colors, legend convention, placement lattice, and clipping
#      rule stay the same. Only the transform is conditioned on position.
#    - To make that conditioning visible, one glyph (marker 5 -> color 6) is changed from a
#      180°-symmetric diagonal-block pattern into an asymmetric 8-pixel shape. The other glyphs
#      remain unchanged, which keeps the task close to the source while ensuring at least one
#      marker type exhibits all four orientations.
#    - The generator always places the focus marker in all four quadrants, so the learner sees
#      repeated evidence that orientation follows quadrant, not marker identity.
# c) Implicit-assumption review of the original solver:
#    - The original GLYPHS dictionary encoded final paint masks directly; that relied on a hidden
#      assumption that every marker used a single fixed orientation. The variant breaks that
#      assumption, so masks must now be treated as canonical templates and rotated at runtime.
#    - _has_source_glyph() originally checked only the canonical legend sample. That would fail if
#      the legend itself sits in a quadrant requiring rotation, so the legend matcher now rotates
#      the canonical mask using the legend glyph's own top-left position before testing the patch.
#    - _paint_clipped() remains valid because clipping is independent of orientation once the
#      correct rotated mask has been chosen.
#    - The main marker loop previously ignored marker position except for placement. In this
#      variant, every marker expansion depends on `(r, c)` relative to the grid midpoint; the
#      generator therefore uses even-sized grids only, making the quadrant split deterministic.
# d) Placement legality in the generator:
#    - Markers are still placed on a regular 2D lattice with row/col gaps in the original 4..6
#      range, so 4x4 glyph footprints do not overlap.
#    - The generator chooses row positions spanning both the top and bottom halves, and column
#      positions spanning both the left and right halves. This guarantees that the lattice itself
#      contains all four quadrants.
#    - An optional legend is added only when its 4x4 patch and label cell are empty in the input
#      and correspond to zero cells in the expected output, so it cannot obscure real marker
#      expansions or create false legend matches.
# e) Visual observability:
#    - Every generated instance contains four focus markers of color 5, one in each quadrant.
#      Because marker 5's glyph is asymmetric under all quarter-turns, the same color-6 shape
#      appears in multiple orientations in the output, making the new rule directly inferable.
#    - The generator rejects any case where the quadrant-rotated output equals the legacy
#      unrotated output, so passing tests cannot happen via accidental degeneration.
# f) Visual cleanliness:
#    - Glyph footprints do not overlap and the legend is kept away from nonzero output regions,
#      preventing clutter between rotated glyphs and legend hints.
#    - Size 10 instances use a compact 2x2 lattice; size 20 instances may use denser lattices, but
#      all mandatory focus markers remain readable because their glyphs are fully in-bounds.
# g) Color-rule inferability:
#    - No new color remapping is introduced: marker colors still map to the same output colors.
#      The only new signal is geometric rotation tied to a visible spatial partition (the grid
#      quadrants), which is inferable from position alone.
# Additional review of new solver assumptions:
#    - main() assumes quadrant is determined by the marker's anchor cell, not by the glyph's
#      clipped visible footprint. generate_input() respects this by constructing markers on the
#      lattice first and never using odd-sized grids that would make the midpoint ambiguous.
#    - main() assumes any glyph-colored 4x4 legend sample already obeys the same rotation rule as
#      a real marker at that top-left position. generate_input() enforces exactly that when it
#      draws optional legends.


def _quadrant(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    top = int(r) < int(h // 2)
    left = int(c) < int(w // 2)
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _rotation_k(shape: tuple[int, int], r: int, c: int) -> int:
    quadrant = _quadrant(shape, r, c)
    if quadrant == "tl":
        return 0
    if quadrant == "tr":
        return 3
    if quadrant == "bl":
        return 1
    return 2


def _rotated_mask(shape: tuple[int, int], r: int, c: int, mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, k=_rotation_k(shape, r, c))


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    rotated = _rotated_mask(g.shape, r0, c0, mask)
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
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            markers.append((rr, cc, marker_color))
    return markers, seen_marker


def _render_markers(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    *,
    use_rotation: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        glyph_color, mask = GLYPHS[marker_color]
        paint_mask = _rotated_mask(shape, r, c, mask) if use_rotation else mask
        _paint_clipped(out, r, c, glyph_color, paint_mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each control pixel into its 4x4 glyph, rotating that glyph by the marker's quadrant.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one legend sample: a 4x4 glyph already drawn in its output color,
      plus a same-type marker one row and one column beyond the sample's bottom-right corner.

    Rule:
    - Each real marker still maps to a fixed glyph/color pair, but the canonical 4x4 glyph is
      rotated before placement according to the marker's quadrant: top-left 0°, top-right 90°
      clockwise, bottom-left 270° clockwise, bottom-right 180°.
    - The marker cell remains the top-left anchor of the 4x4 placement window, and any part of the
      rotated glyph falling outside the grid is clipped away.
    - Legend glyphs and their below-right label markers are removed rather than copied to output.

    Key insight:
    - Marker color identifies which glyph family to use, while marker position decides the glyph's
      orientation; the pre-drawn legend is only a hint and must itself be interpreted with the
      same quadrant-dependent rotation rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    markers, seen_marker = _extract_real_markers(g)
    out = _render_markers(g.shape, markers, use_rotation=True)

    assert seen_marker
    return out


def _sample_axis_positions(rng: random.Random, size: int, count: int) -> list[int] | None:
    for _ in range(160):
        gaps = [rng.randint(4, 6) for _ in range(count - 1)]
        span = sum(gaps)
        max_start = size - 4 - span
        if max_start < 0:
            continue
        start = rng.randint(0, max_start)
        positions = [start]
        for gap in gaps:
            positions.append(positions[-1] + gap)
        if not any(pos < size // 2 for pos in positions):
            continue
        if not any(pos >= size // 2 for pos in positions):
            continue
        return positions
    return None


def _choose_filled_cells(
    rng: random.Random,
    size: int,
    row_positions: list[int],
    col_positions: list[int],
) -> list[tuple[int, int]]:
    quadrant_cells: dict[str, list[tuple[int, int]]] = {key: [] for key in ("tl", "tr", "bl", "br")}
    all_cells = [(r, c) for r in row_positions for c in col_positions]
    for cell in all_cells:
        quadrant_cells[_quadrant((size, size), cell[0], cell[1])].append(cell)

    mandatory = {rng.choice(quadrant_cells[key]) for key in quadrant_cells}
    min_filled = max(len(mandatory), math.ceil(0.6 * len(all_cells)))
    filled_count = rng.randint(min_filled, len(all_cells))

    remaining = [cell for cell in all_cells if cell not in mandatory]
    extra = rng.sample(remaining, filled_count - len(mandatory))
    return sorted(mandatory | set(extra))


def _assign_marker_types(
    rng: random.Random,
    size: int,
    cells: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    seen_focus_quadrants: set[str] = set()
    extra_palette = [color for color in MARKER_COLORS if color != FOCUS_MARKER]
    type_count = rng.randint(1, min(4, len(cells)))
    active_types = [FOCUS_MARKER]
    if type_count > 1:
        active_types.extend(rng.sample(extra_palette, type_count - 1))

    for r, c in cells:
        quadrant = _quadrant((size, size), r, c)
        if quadrant not in seen_focus_quadrants:
            marker_color = FOCUS_MARKER
            seen_focus_quadrants.add(quadrant)
        else:
            marker_color = rng.choice(active_types)
        markers.append((r, c, marker_color))
    return markers


def _try_add_legend(
    rng: random.Random,
    inp: np.ndarray,
    expected: np.ndarray,
    used_marker_colors: set[int],
) -> bool:
    if not used_marker_colors or rng.random() < 0.45:
        return False

    preferred: list[int] = []
    if FOCUS_MARKER in used_marker_colors:
        preferred.append(FOCUS_MARKER)
    preferred.extend(color for color in sorted(used_marker_colors) if color != FOCUS_MARKER)

    h, w = inp.shape
    for marker_color in preferred:
        glyph_color, mask = GLYPHS[marker_color]
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - 4):
            for c0 in range(w - 4):
                label = (r0 + 4, c0 + 4)
                patch_cells = {(rr, cc) for rr in range(r0, r0 + 4) for cc in range(c0, c0 + 4)}
                patch_cells.add(label)
                if any(inp[rr, cc] != 0 for rr, cc in patch_cells):
                    continue
                if any(expected[rr, cc] != 0 for rr, cc in patch_cells):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            continue

        r0, c0 = rng.choice(candidates)
        rotated = _rotated_mask(inp.shape, r0, c0, mask)
        for dr in range(4):
            for dc in range(4):
                if bool(rotated[dr, dc]):
                    inp[r0 + dr, c0 + dc] = glyph_color
        inp[r0 + 4, c0 + 4] = marker_color
        return True

    return False


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        size = rng.choices(GRID_SIZES, weights=(1, 3), k=1)[0]
        max_axis_count = 2 if size == 10 else 5
        row_count = rng.randint(2, max_axis_count)
        col_count = rng.randint(2, max_axis_count)

        row_positions = _sample_axis_positions(rng, size, row_count)
        col_positions = _sample_axis_positions(rng, size, col_count)
        if row_positions is None or col_positions is None:
            continue

        filled_cells = _choose_filled_cells(rng, size, row_positions, col_positions)
        markers = _assign_marker_types(rng, size, filled_cells)

        expected = _render_markers((size, size), markers, use_rotation=True)
        legacy = _render_markers((size, size), markers, use_rotation=False)
        if not np.any(expected != legacy):
            continue

        focus_quadrants = {
            _quadrant((size, size), r, c)
            for r, c, marker_color in markers
            if marker_color == FOCUS_MARKER
        }
        if focus_quadrants != {"tl", "tr", "bl", "br"}:
            continue

        inp = np.zeros((size, size), dtype=int)
        for r, c, marker_color in markers:
            inp[r, c] = marker_color

        _try_add_legend(rng, inp, expected, {marker_color for _, _, marker_color in markers})

        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a non-degenerate D1 rotation instance")
