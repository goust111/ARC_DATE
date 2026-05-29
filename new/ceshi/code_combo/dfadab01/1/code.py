# Combo variant of dfadab01 (index 1)
# combo_spec: D1/1+D3/3
# Name: 象限旋转+中心定位

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
MAX_GENERATION_ATTEMPTS = 500

# Design analysis for the D1/1 + D3/3 combo variant
# a) Concrete behavior and trace:
#    - Base task: a marker at (r, c) directly became a fixed-orientation 4x4 glyph whose top-left
#      corner was also (r, c). A legend, when present, was a pre-drawn 4x4 glyph with a same-type
#      label marker at (+4, +4), and both legend pieces vanished in the output.
#    - Combo task: the marker is no longer the glyph's top-left anchor. Instead, it marks the
#      glyph's center reference, so the 4x4 placement window starts at (r-2, c-2). Before that
#      window is painted, the canonical mask is rotated by the marker's quadrant:
#      TL -> 0°, TR -> 90° clockwise, BR -> 180°, BL -> 270° clockwise.
#    - Small trace: in a 10x10 grid, a marker-5 at (2, 2) paints the canonical magenta glyph into
#      rows 0..3, cols 0..3. Another marker-5 at (2, 6) paints the same glyph into rows 0..3,
#      cols 4..7, but rotated 90° clockwise because the marker center lies in the top-right half.
#      Under the base task those two glyphs would have started at (2, 2) and (2, 6) with no center
#      shift and no positional reorientation, so both the footprint and the orientation differ.
# b) Complexity calibration:
#    - The combo changes exactly two low-arity factors: anchor interpretation (top-left vs center)
#      and rotation bucket (four quadrants). The color mapping and glyph alphabet stay fixed.
#    - The focus glyph for marker 5 is made asymmetric under quarter turns, so quadrant rotation is
#      visible. The other glyphs remain close to the source task, which keeps the overall puzzle
#      recognizable rather than turning every color into a new arbitrary sprite family.
#    - The generator forces one focus marker in each quadrant and rejects cases that collapse to
#      either single-dimension ablation: rotation-only top-left anchoring or center-only placement
#      without rotation.
# c) Implicit-assumption review of the original solver:
#    - The original paint call assumed marker coordinates were already the glyph top-left. That is
#      false now; all rendering must go through `(r-2, c-2)`.
#    - The original legend matcher assumed the sample glyph used the canonical orientation at its
#      own top-left. That also breaks here. A visible 4x4 sample at top-left `(r0, c0)` should be
#      matched against the rotation induced by its conceptual center `(r0+2, c0+2)`, because that
#      is where a real marker would sit for the same placed glyph.
#    - The original main loop treated marker position only as placement. In the combo, position now
#      affects both translation and rotation, so every expansion path must consult the marker cell.
#    - `_paint_clipped()` itself remains valid because clipping still happens after the final 4x4
#      mask has been chosen and placed.
# d) Placement legality in the generator:
#    - Real markers are placed on a regular lattice with row/col gaps in 4..6, so their 4x4
#      center-anchored footprints remain disjoint. The center shift subtracts the same 2-cell
#      offset from every lattice point, so non-overlap is preserved.
#    - The generator samples row centers spanning both top/bottom halves and column centers
#      spanning both left/right halves. Their Cartesian product therefore contains all quadrants.
#    - Optional legends are inserted only where the entire legend patch plus its bottom-right label
#      lie on zero input cells and on zero cells of the expected output, so the legend cannot hide
#      real evidence or create ambiguous collisions.
# e) Visual observability:
#    - Humans can compare marker centers in the input against the output glyph footprints and see
#      that every glyph is shifted two cells up-left from its marker, not attached by its corner.
#    - Repeating the same marker color 5 across all four quadrants shows that orientation depends
#      on location rather than color identity. Because the center shift is global and the rotation
#      is quadrant-specific, both cues recur in every example.
#    - The generator rejects degenerate instances where combo output equals either ablation, so the
#      training distribution keeps both rules visually active.
# f) Visual cleanliness:
#    - The lattice spacing avoids overlapping 4x4 glyphs, and legends are kept off all nonzero
#      output cells. This preserves clear isolated motifs instead of layered clutter.
#    - The generator prefers fully visible real glyphs for the mandatory focus markers. Clipping is
#      still supported by the solver, but readability takes priority because the fused rule itself,
#      not boundary artifacts, must be inferable from 2-4 examples.
# g) Change inferability:
#    - No extra color condition is added. Marker color still chooses the glyph family and output
#      color, while spatial cues alone determine anchor shift and rotation. This keeps the mapping
#      learnable from structure rather than hidden numeric conventions.
# Additional review of new solver assumptions:
#    - `main()` assumes even-sized grids so the quadrant split at `h//2` and `w//2` is unambiguous;
#      the generator uses only 10x10 and 20x20 grids.
#    - `main()` assumes every legend label sits exactly at `(glyph_top_left+4, glyph_top_left+4)`;
#      the generator enforces that fixed convention.
#    - `main()` assumes the optional legend sample itself is already rotated according to the
#      quadrant of its conceptual center `(r0+2, c0+2)`, even though the label is a special legend
#      tag rather than a real center marker. The generator draws legends that way.


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


def _marker_top_left(r: int, c: int) -> tuple[int, int]:
    return int(r) - 2, int(c) - 2


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    rotated = _rotated_mask(g.shape, r0 + 2, c0 + 2, mask)
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
    use_center_anchor: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        glyph_color, mask = GLYPHS[marker_color]
        paint_mask = _rotated_mask(shape, r, c, mask) if use_rotation else mask
        r0, c0 = _marker_top_left(r, c) if use_center_anchor else (int(r), int(c))
        _paint_clipped(out, r0, c0, glyph_color, paint_mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marker with a 4x4 glyph whose center sits on the marker and whose orientation depends on quadrant.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one legend sample: a 4x4 glyph already drawn in its output color,
      plus a same-type label marker exactly one cell beyond the sample's bottom-right corner.

    Rule:
    - Each real marker still selects one fixed glyph/color pair, but the marker gives the glyph's
      center reference rather than its top-left corner, so the 4x4 placement window starts two
      rows up and two columns left from the marker.
    - Before painting, rotate the canonical glyph by the marker's quadrant: top-left 0°,
      top-right 90° clockwise, bottom-right 180°, bottom-left 270° clockwise.
    - Clip any off-grid portion of the placed glyph, and delete legend glyphs together with their
      bottom-right label markers instead of copying them into the output.

    Key insight:
    - Marker color chooses the glyph family, while marker position contributes two independent
      spatial facts at once: where the glyph must be centered and which orientation it must take.
      The visible legend is only a lookup hint and follows the same rotated glyph vocabulary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    markers, seen_marker = _extract_real_markers(g)
    out = _render_markers(g.shape, markers, use_rotation=True, use_center_anchor=True)

    assert seen_marker
    return out


def _sample_center_positions(rng: random.Random, size: int, count: int) -> list[int] | None:
    for _ in range(200):
        gaps = [rng.randint(4, 6) for _ in range(count - 1)]
        span = sum(gaps)
        max_start = size - 3 - span
        if max_start < 2:
            continue
        start = rng.randint(2, max_start)
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


def _focus_orientation_count(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> int:
    _, base_mask = GLYPHS[FOCUS_MARKER]
    signatures = {
        _rotated_mask(shape, r, c, base_mask).astype(np.uint8).tobytes()
        for r, c, marker_color in markers
        if marker_color == FOCUS_MARKER
    }
    return len(signatures)


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
        rotated = _rotated_mask(inp.shape, r0 + 2, c0 + 2, mask)
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

        row_positions = _sample_center_positions(rng, size, row_count)
        col_positions = _sample_center_positions(rng, size, col_count)
        if row_positions is None or col_positions is None:
            continue

        filled_cells = _choose_filled_cells(rng, size, row_positions, col_positions)
        markers = _assign_marker_types(rng, size, filled_cells)

        expected = _render_markers(
            (size, size),
            markers,
            use_rotation=True,
            use_center_anchor=True,
        )
        rotation_only = _render_markers(
            (size, size),
            markers,
            use_rotation=True,
            use_center_anchor=False,
        )
        center_only = _render_markers(
            (size, size),
            markers,
            use_rotation=False,
            use_center_anchor=True,
        )
        if not np.any(expected != rotation_only):
            continue
        if not np.any(expected != center_only):
            continue

        focus_quadrants = {
            _quadrant((size, size), r, c)
            for r, c, marker_color in markers
            if marker_color == FOCUS_MARKER
        }
        if focus_quadrants != {"tl", "tr", "bl", "br"}:
            continue
        if _focus_orientation_count((size, size), markers) < 4:
            continue

        inp = np.zeros((size, size), dtype=int)
        for r, c, marker_color in markers:
            inp[r, c] = marker_color

        _try_add_legend(rng, inp, expected, {marker_color for _, _, marker_color in markers})

        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a non-degenerate combo D1/1+D3/3 instance")
