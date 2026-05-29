# Combo variant of dfadab01 (index 14)
# combo_spec: D1/1+D5/0
# Name: 象限旋转+旋转变体字形

from __future__ import annotations

import math
import random

import numpy as np


# design_analysis:
#   a_concretization_and_trace:
#     - The base task paints a fixed 4x4 glyph for each marker color and ignores marker position.
#     - In this combo, the glyph family itself is rotational: colors 2/3/5/8 select the
#       0/90/180/270-degree turns of one asymmetric base mask.
#     - Then placement adds a second rotation based on quadrant: TL +0, TR +90 clockwise,
#       BR +180, BL +270 clockwise. The final orientation is the sum of the two rotations.
#     - Concrete trace: marker 3 uses the base mask rotated once. If that marker sits in the
#       top-left quadrant, it stays at rot90. If the same marker sits in the top-right quadrant,
#       it receives one more clockwise quarter-turn and becomes rot180. The original task had
#       only "lookup by color"; the combo becomes "lookup by color, then add quadrant rotation".
#   b_complexity_calibration:
#     - The base mask is intentionally asymmetric and has eight filled cells, so all four quarter
#       turns stay visually distinct and the rotational family is observable.
#     - The generator always exposes both factors separately: all four marker colors appear inside
#       one top-left 2x2 block (same quadrant, different intrinsic rotations), and one focus color
#       appears once in every quadrant (same intrinsic rotation, different added quadrant turns).
#     - Grid size is fixed to 20 here because a 10x10 board cannot reliably host both comparison
#       structures while preserving the source task's 4-6 spacing and clean non-overlap.
#   c_implicit_assumption_audit:
#     - The original solver assumed each marker color owned one final mask. That no longer holds;
#       masks must now be derived at runtime from base-rotation + quadrant-rotation.
#     - Legend detection can no longer compare against a color's intrinsic mask alone. A legend
#       sample at (r0, c0) must match the same composite rotation that a real marker of that color
#       would produce from that top-left anchor.
#     - Quadrant membership is computed from the marker anchor cell, not from the clipped visible
#       fragment. The generator therefore uses even-sized boards and places all informative markers
#       fully in bounds so the split remains unambiguous and human-readable.
#     - Painting and clipping logic remain valid because they only need the already-resolved mask.
#   d_placement_legality:
#     - Real markers stay on a regular lattice with row/column gaps in 4..6, so 4x4 footprints do
#       not overlap.
#     - Legend placement searches only regions that are empty in both the input and expected
#       output, and requires the full 4x4 patch plus the (+4, +4) label cell to fit.
#   e_visual_observability:
#     - The top-left 2x2 block shows the four intrinsic glyph rotations under the same quadrant
#       rule, so the learner can see that the alphabet is rotational rather than four unrelated
#       templates.
#     - The focus marker color appears in all four quadrants, so the learner can also see a single
#       color acquire different extra turns from position alone.
#   f_visual_cleanliness:
#     - Mandatory evidence cells are separated by the original spacing constraints, and legends are
#       only inserted into blank zones where the expected output is zero.
#     - No extra guide marks or background noise are introduced; the scene stays close to the
#       source task's sparse black canvas style.
#   g_inferability_of_the_change:
#     - Color mapping stays fixed and low-cardinality: 2->4, 3->1, 5->6, 8->7.
#     - The only added freedom is rotational, and both controlling variables are visible in the
#       examples: color reveals intrinsic orientation and quadrant reveals the extra quarter-turn.
#   post_change_main_assumptions:
#     - main() assumes the shared base mask has four distinct quarter-turns; BASE_MASK enforces it.
#     - main() assumes any glyph-colored legend sample obeys the same composite rule as a real
#       marker at that top-left location; generate_input() constructs legends exactly that way.
#     - main() assumes isolated marker pixels are the only occurrences of marker colors outside the
#       optional legend labels; generate_input() preserves that input grammar.

BASE_MASK = np.array(
    [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
    ],
    dtype=bool,
)

MARKER_TO_COLOR: dict[int, int] = {2: 4, 3: 1, 5: 6, 8: 7}
MARKER_BASE_ROTATIONS: dict[int, int] = {2: 0, 3: 1, 5: 2, 8: 3}
MARKER_COLORS = tuple(MARKER_TO_COLOR.keys())
GLYPH_COLORS = tuple(sorted(MARKER_TO_COLOR.values()))
ALLOWED_COLORS = {0, *MARKER_COLORS, *GLYPH_COLORS}
GRID_SIZES = (20,)
FOCUS_MARKER = 2
MAX_GENERATION_ATTEMPTS = 400


def _quadrant(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    top = r < h // 2
    left = c < w // 2
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _quadrant_rotation_k(shape: tuple[int, int], r: int, c: int) -> int:
    quadrant = _quadrant(shape, r, c)
    if quadrant == "tl":
        return 0
    if quadrant == "tr":
        return 3
    if quadrant == "bl":
        return 1
    return 2


def _effective_rotation_k(
    shape: tuple[int, int],
    r: int,
    c: int,
    marker_color: int,
    *,
    use_intrinsic_rotation: bool,
    use_quadrant_rotation: bool,
) -> int:
    k = 0
    if use_intrinsic_rotation:
        k += MARKER_BASE_ROTATIONS[marker_color]
    if use_quadrant_rotation:
        k += _quadrant_rotation_k(shape, r, c)
    return k % 4


def _effective_mask(
    shape: tuple[int, int],
    r: int,
    c: int,
    marker_color: int,
    *,
    use_intrinsic_rotation: bool,
    use_quadrant_rotation: bool,
) -> np.ndarray:
    return np.rot90(
        BASE_MASK,
        k=_effective_rotation_k(
            shape,
            r,
            c,
            marker_color,
            use_intrinsic_rotation=use_intrinsic_rotation,
            use_quadrant_rotation=use_quadrant_rotation,
        ),
    )


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, marker_color: int) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    glyph_color = MARKER_TO_COLOR[marker_color]
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    mask = _effective_mask(
        g.shape,
        r0,
        c0,
        marker_color,
        use_intrinsic_rotation=True,
        use_quadrant_rotation=True,
    )
    return bool(np.all(patch[mask] == glyph_color) and np.all(patch[~mask] != glyph_color))


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


def _render_markers(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    *,
    use_intrinsic_rotation: bool,
    use_quadrant_rotation: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        glyph_color = MARKER_TO_COLOR[marker_color]
        mask = _effective_mask(
            shape,
            r,
            c,
            marker_color,
            use_intrinsic_rotation=use_intrinsic_rotation,
            use_quadrant_rotation=use_quadrant_rotation,
        )
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker into a quarter-turned member of a shared glyph family, then add a quadrant turn.

    Input structure:
    - A black grid contains isolated marker pixels in colors 2, 3, 5, and/or 8.
    - Some inputs also include 0-2 legend entries: a full 4x4 glyph drawn in its output color and
      a same-type marker placed one cell below-right of that glyph's bounding box.

    Rule:
    - Marker color chooses one intrinsic quarter-turn of a single asymmetric 4x4 base glyph:
      2/3/5/8 correspond to the 0/90/180/270-degree variants.
    - Before painting, add an extra rotation determined by the marker's quadrant: top-left 0,
      top-right 90 clockwise, bottom-right 180, bottom-left 270 clockwise.
    - Paint the resulting glyph in the marker's fixed output color with the marker as the 4x4
      top-left anchor, clip anything outside the grid, and erase legend glyphs and legend labels.

    Key insight:
    - The final orientation is a composition of two visible factors: the marker color tells which
      rotation of the shared prototype to start from, and the marker position tells how much more
      to rotate it before drawing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_COLORS)

    markers, seen_marker = _extract_real_markers(g)
    out = _render_markers(
        g.shape,
        markers,
        use_intrinsic_rotation=True,
        use_quadrant_rotation=True,
    )

    assert seen_marker
    return out


def _sample_axis_positions(rng: random.Random, size: int, count: int) -> list[int] | None:
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
        top_count = sum(pos < size // 2 for pos in positions)
        bottom_count = len(positions) - top_count
        if top_count >= 2 and bottom_count >= 2:
            return positions
    return None


def _cells_by_quadrant(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = {key: [] for key in ("tl", "tr", "bl", "br")}
    for cell in cells:
        grouped[_quadrant(shape, cell[0], cell[1])].append(cell)
    return grouped


def _build_mandatory_markers(
    rng: random.Random,
    size: int,
    row_positions: list[int],
    col_positions: list[int],
) -> dict[tuple[int, int], int] | None:
    top_rows = [r for r in row_positions if r < size // 2]
    left_cols = [c for c in col_positions if c < size // 2]
    if len(top_rows) < 2 or len(left_cols) < 2:
        return None

    tl_cells = [(r, c) for r in top_rows[:2] for c in left_cols[:2]]
    if len(tl_cells) < 4:
        return None

    all_cells = [(r, c) for r in row_positions for c in col_positions]
    quadrant_cells = _cells_by_quadrant((size, size), all_cells)
    mandatory: dict[tuple[int, int], int] = {}

    shuffled_colors = list(MARKER_COLORS)
    rng.shuffle(shuffled_colors)
    for cell, marker_color in zip(tl_cells, shuffled_colors):
        mandatory[cell] = marker_color

    for quadrant in ("tr", "bl", "br"):
        candidates = [cell for cell in quadrant_cells[quadrant] if cell not in mandatory]
        if not candidates:
            return None
        mandatory[rng.choice(candidates)] = FOCUS_MARKER

    return mandatory


def _try_add_legend(
    rng: random.Random,
    inp: np.ndarray,
    expected: np.ndarray,
    marker_color: int,
) -> bool:
    h, w = inp.shape
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
        return False

    r0, c0 = rng.choice(candidates)
    glyph_color = MARKER_TO_COLOR[marker_color]
    mask = _effective_mask(
        inp.shape,
        r0,
        c0,
        marker_color,
        use_intrinsic_rotation=True,
        use_quadrant_rotation=True,
    )
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                inp[r0 + dr, c0 + dc] = glyph_color
    inp[r0 + 4, c0 + 4] = marker_color
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        size = rng.choice(GRID_SIZES)
        row_count = rng.randint(4, 5)
        col_count = rng.randint(4, 5)

        row_positions = _sample_axis_positions(rng, size, row_count)
        col_positions = _sample_axis_positions(rng, size, col_count)
        if row_positions is None or col_positions is None:
            continue

        mandatory = _build_mandatory_markers(rng, size, row_positions, col_positions)
        if mandatory is None:
            continue

        all_cells = [(r, c) for r in row_positions for c in col_positions]
        remaining_cells = [cell for cell in all_cells if cell not in mandatory]
        min_fill = max(len(mandatory), math.ceil(0.6 * len(all_cells)))
        filled_count = rng.randint(min_fill, len(all_cells))
        extra_count = filled_count - len(mandatory)
        extra_cells = rng.sample(remaining_cells, extra_count)

        markers: list[tuple[int, int, int]] = [
            (r, c, marker_color) for (r, c), marker_color in sorted(mandatory.items())
        ]
        for r, c in extra_cells:
            markers.append((r, c, rng.choice(MARKER_COLORS)))

        expected = _render_markers(
            (size, size),
            markers,
            use_intrinsic_rotation=True,
            use_quadrant_rotation=True,
        )
        no_quadrant = _render_markers(
            (size, size),
            markers,
            use_intrinsic_rotation=True,
            use_quadrant_rotation=False,
        )
        no_intrinsic = _render_markers(
            (size, size),
            markers,
            use_intrinsic_rotation=False,
            use_quadrant_rotation=True,
        )
        if np.array_equal(expected, no_quadrant):
            continue
        if np.array_equal(expected, no_intrinsic):
            continue

        focus_cells = [(r, c) for r, c, marker_color in markers if marker_color == FOCUS_MARKER]
        focus_quadrants = {_quadrant((size, size), r, c) for r, c in focus_cells}
        if focus_quadrants != {"tl", "tr", "bl", "br"}:
            continue

        inp = np.zeros((size, size), dtype=int)
        for r, c, marker_color in markers:
            inp[r, c] = marker_color

        legend_count = 0
        if rng.random() < 0.75:
            target = 2 if rng.random() < 0.35 else 1
            legend_types = rng.sample(list(MARKER_COLORS), k=target)
            for marker_color in legend_types:
                if _try_add_legend(rng, inp, expected, marker_color):
                    legend_count += 1

        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue

        if legend_count == 0 and rng.random() < 0.2:
            continue

        return inp

    raise RuntimeError("failed to generate a valid combo rotation instance")
