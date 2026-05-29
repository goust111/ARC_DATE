# Combo variant of dfadab01 (index 34)
# combo_spec: D7/2+D9/0
# Name: 间距编码信息+按色批处理

from __future__ import annotations

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
                [1, 1, 0, 0],
                [0, 0, 1, 1],
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

PROCESS_ORDER: tuple[int, ...] = (2, 3, 5, 8)
MARKER_COLORS = tuple(PROCESS_ORDER)
ALLOWED_INPUT_COLORS = {0, *MARKER_COLORS, 1, 4, 6, 7}
DENSE_SPACING = 2
ROTATE_SPACING = 3
PLAIN_SPACING = 4
SPACING_TO_ROTATION_K = {
    ROTATE_SPACING: 3,  # 90 degrees clockwise
    PLAIN_SPACING: 0,
}

# Design analysis for the combo D7/2 + D9/0 variant
# a) Concrete behavior and trace:
#    - The input still consists of single-cell markers in colors 2/3/5/8 on black, plus an
#      optional legend that shows one canonical 4x4 glyph and its label marker at (+4, +4).
#    - The lattice now has two roles at once. One axis stays dense with spacing 2 so expanded 4x4
#      glyphs overlap and batch order becomes visible. The other axis is uniformly spaced by either
#      3 or 4, and that spacing is semantic: 3 means every real glyph must be rotated 90 degrees
#      clockwise before painting; 4 means use the canonical orientation.
#    - Small trace: suppose a horizontal layout uses row spacing 3 and col spacing 2. A marker 5 at
#      (1, 1) and a marker 8 at (1, 3) overlap after expansion. In the base task, spacing>=4 meant
#      no interaction at all. In the D9-only setting, color 5 would still use its canonical diagonal
#      blocks and then be partly overwritten by later color 8. In this combo, the same color-5 mask
#      is rotated first because the row gap is 3, changing exactly which overlap cells survive after
#      the 2->3->5->8 paint batches. Both the spacing code and the color order affect the result.
# b) Complexity calibration:
#    - Only two spacing codes are allowed: 3 = rotate, 4 = do not rotate. The dense overlap axis is
#      fixed at 2. This keeps the rule space small enough to infer from a few examples.
#    - Color 5 is always present and repeated, because among the original glyphs it is the one whose
#      90-degree rotation is visibly different. Other glyphs keep the task anchored to the source.
#    - Layouts are stripe-like rather than fully chaotic: one axis is dense, the orthogonal axis is
#      still regular and moderately spaced, so humans can measure the informative gap by eye.
# c) Implicit-assumption review of the original main():
#    - The old solver assumed marker spacing had no semantic meaning. That breaks here: main() must
#      first recover the real marker lattice, ignore any legend label, and infer whether spacing 3
#      or 4 is present on the non-dense axis.
#    - The old fixed-mask expansion is no longer sufficient. The chosen spacing code now selects
#      either the canonical mask or its 90-degree clockwise rotation for every real marker.
#    - Iterating GLYPHS.items() was previously harmless because the source generator avoided overlap.
#      Here overlap is intentional, so rendering must use an explicit PROCESS_ORDER constant.
#    - _has_source_glyph() still checks the canonical legend sample. This is a deliberate contract:
#      the legend explains the base alphabet only; spacing metadata decides whether real placements
#      are rotated at render time.
# d) Placement legality:
#    - Real markers never overlap in the input because they are still single pixels on distinct
#      lattice sites. Overlap happens only after 4x4 expansion.
#    - The generator always creates at least a 2x2 lattice and uses one dense axis plus one coded
#      axis, so spacing inference is well-defined.
#    - Optional legends are placed only in empty 5x5 regions, so they cannot erase markers or
#      create ambiguous fake lattice coordinates.
# e) Visual observability:
#    - Every generated instance has overlap_pixels>0 and order_diff_pixels>0, proving that the
#      color batches matter.
#    - Every generated instance also has rotation_diff_pixels>0 between the rotated and unrotated
#      renderings, proving that the spacing code changes the geometry rather than being decorative.
#    - Because the coded gap is repeated across several rows or columns, the learner can measure it
#      visually instead of relying on hidden numeric conventions.
# f) Visual cleanliness:
#    - Only one axis is dense. This produces readable horizontal or vertical collision bands instead
#      of an all-over tangle.
#    - Row/column counts are capped so that even when the coded spacing is 3, the output remains a
#      structured lattice of partially overlapping 4x4 stamps rather than a filled solid block.
# g) Color-rule inferability:
#    - No invisible priority trick is used. The order is the visible marker-color order 2->3->5->8,
#      and the spacing cue is a repeated geometric distance that a person can count directly.
# Additional review of new solver assumptions:
#    - main() assumes there is exactly one non-dense repeated spacing in {3,4} among real markers.
#      generate_input() enforces that by building either a horizontal or vertical stripe lattice.
#    - main() assumes the legend, if present, is canonical rather than already rotated. The generator
#      draws legends that way on purpose so the spacing code remains the only transform signal.
#    - main() assumes enough color-5 markers survive ordering for the rotation choice to matter.
#      The generator checks this with rotation_diff_pixels and rejects degenerate samples.


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
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


def _rotated_mask(mask: np.ndarray, spacing_code: int) -> np.ndarray:
    return np.rot90(mask, k=SPACING_TO_ROTATION_K[spacing_code])


def _extract_real_markers(g: np.ndarray) -> tuple[list[tuple[int, int, int]], bool]:
    markers: list[tuple[int, int, int]] = []
    seen_marker = False
    for marker_color in PROCESS_ORDER:
        glyph_color, mask = GLYPHS[marker_color]
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if _has_source_glyph(g, rr - 4, cc - 4, glyph_color, mask):
                continue
            markers.append((marker_color, rr, cc))
    return markers, seen_marker


def _infer_spacing_code(markers: list[tuple[int, int, int]]) -> int:
    rows = sorted({r for _color, r, _c in markers})
    cols = sorted({c for _color, _r, c in markers})

    row_gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    col_gaps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    code_gaps = {gap for gap in row_gaps + col_gaps if gap in SPACING_TO_ROTATION_K}

    assert len(code_gaps) == 1
    spacing_code = next(iter(code_gaps))
    assert spacing_code in (ROTATE_SPACING, PLAIN_SPACING)
    return spacing_code


def _render_markers(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    *,
    order: tuple[int, ...],
    spacing_code: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    paint_masks = {
        marker_color: _rotated_mask(mask, spacing_code)
        for marker_color, (_glyph_color, mask) in GLYPHS.items()
    }
    for marker_color in order:
        glyph_color, _mask = GLYPHS[marker_color]
        paint_mask = paint_masks[marker_color]
        for current_color, r, c in markers:
            if current_color != marker_color:
                continue
            _paint_clipped(out, r, c, glyph_color, paint_mask)
    return out


def _coverage_map(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    *,
    spacing_code: int,
) -> np.ndarray:
    coverage = np.zeros(shape, dtype=int)
    h, w = shape
    for marker_color, r0, c0 in markers:
        _glyph_color, mask = GLYPHS[marker_color]
        paint_mask = _rotated_mask(mask, spacing_code)
        for dr in range(4):
            rr = r0 + dr
            if rr < 0 or rr >= h:
                continue
            for dc in range(4):
                cc = c0 + dc
                if 0 <= cc < w and bool(paint_mask[dr, dc]):
                    coverage[rr, cc] += 1
    return coverage


def _repeat_to_length(seq: list[int], length: int) -> list[int]:
    out: list[int] = []
    while len(out) < length:
        out.extend(seq)
    return out[:length]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each real marker into its 4x4 glyph, with spacing-coded rotation and color-batch overwriting.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - Real markers sit on a regular lattice whose dense axis has spacing 2, while the orthogonal repeated spacing is either 3 or 4.
    - Some inputs also include one legend entry: a canonical 4x4 sample glyph in its output color plus a matching label marker one row and one column beyond the sample's bottom-right corner.

    Rule:
    - Ignore legend labels whose matching canonical sample glyph is found at (-4, -4).
    - Measure the repeated non-dense marker spacing: spacing 3 means rotate every real glyph 90 degrees clockwise before painting, while spacing 4 means keep the canonical orientation.
    - Paint glyphs in global marker-color batches 2, then 3, then 5, then 8; later batches overwrite earlier ones wherever the 4x4 footprints overlap. Clip any part that falls outside the grid.

    Key insight:
    - Marker color still chooses the glyph family, but the lattice now carries extra meaning: spacing tells you which orientation to use, and the dense overlaps reveal that colors must be processed in a fixed batch order rather than all at once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    markers, seen_marker = _extract_real_markers(g)
    assert seen_marker and markers

    spacing_code = _infer_spacing_code(markers)
    return _render_markers(g.shape, markers, order=PROCESS_ORDER, spacing_code=spacing_code)


def _place_legend(g: np.ndarray, used_colors: list[int], rng: random.Random) -> bool:
    h, w = g.shape
    candidates = [
        (0, 0),
        (0, w - 5),
        (h - 5, 0),
        (h - 5, w - 5),
        (0, max(0, (w - 5) // 2)),
        (max(0, (h - 5) // 2), 0),
    ]
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        if r0 < 0 or c0 < 0 or r0 + 4 >= h or c0 + 4 >= w:
            continue
        if np.any(g[r0 : r0 + 5, c0 : c0 + 5] != 0):
            continue
        marker_color = rng.choice(used_colors)
        glyph_color, mask = GLYPHS[marker_color]
        for dr in range(4):
            for dc in range(4):
                if bool(mask[dr, dc]):
                    g[r0 + dr, c0 + dc] = glyph_color
        g[r0 + 4, c0 + 4] = marker_color
        return True
    return False


def _build_marker_grid(rng: random.Random) -> np.ndarray:
    orientation = rng.choice(["horizontal", "vertical"])
    size = rng.choice([10, 20])
    spacing_code = rng.choices([ROTATE_SPACING, PLAIN_SPACING], weights=[2, 1], k=1)[0]
    g = np.zeros((size, size), dtype=int)

    used_count = rng.choices([3, 4], weights=[3, 4], k=1)[0]
    used_colors = [5, *rng.sample([c for c in PROCESS_ORDER if c != 5], used_count - 1)]
    palette = rng.sample(used_colors, len(used_colors))
    reverse_palette = list(reversed(palette))

    if orientation == "horizontal":
        row_spacing = spacing_code
        col_spacing = DENSE_SPACING
        rows_max = 1 + (size - 4) // row_spacing
        cols_max = 1 + (size - 4) // col_spacing
        rows = min(rows_max, rng.choice([2, 3, 4]))
        cols = min(cols_max, rng.choice([4, 5, 6]))
        rows = max(rows, 2)
        cols = max(cols, 4)
        height = 4 + (rows - 1) * row_spacing
        width = 4 + (cols - 1) * col_spacing
        r_start = rng.randint(0, size - height)
        c_start = rng.randint(0, size - width)
        row_positions = [r_start + i * row_spacing for i in range(rows)]
        col_positions = [c_start + i * col_spacing for i in range(cols)]

        for ridx, r in enumerate(row_positions):
            if ridx == 0:
                base = palette
            elif ridx == 1:
                base = reverse_palette
            else:
                shift = ridx % len(palette)
                base = palette[shift:] + palette[:shift]
            seq = _repeat_to_length(base, cols)
            for cidx, c in enumerate(col_positions):
                g[r, c] = seq[cidx]
    else:
        row_spacing = DENSE_SPACING
        col_spacing = spacing_code
        rows_max = 1 + (size - 4) // row_spacing
        cols_max = 1 + (size - 4) // col_spacing
        rows = min(rows_max, rng.choice([4, 5, 6]))
        cols = min(cols_max, rng.choice([2, 3, 4]))
        rows = max(rows, 4)
        cols = max(cols, 2)
        height = 4 + (rows - 1) * row_spacing
        width = 4 + (cols - 1) * col_spacing
        r_start = rng.randint(0, size - height)
        c_start = rng.randint(0, size - width)
        row_positions = [r_start + i * row_spacing for i in range(rows)]
        col_positions = [c_start + i * col_spacing for i in range(cols)]

        for cidx, c in enumerate(col_positions):
            if cidx == 0:
                base = palette
            elif cidx == 1:
                base = reverse_palette
            else:
                shift = cidx % len(palette)
                base = palette[shift:] + palette[:shift]
            seq = _repeat_to_length(base, rows)
            for ridx, r in enumerate(row_positions):
                g[r, c] = seq[ridx]

    if rng.random() < 0.5:
        _place_legend(g, used_colors, rng)

    return g


def _variant_metrics(g: np.ndarray) -> dict[str, int]:
    markers, _seen = _extract_real_markers(g)
    spacing_code = _infer_spacing_code(markers)
    alternate_code = PLAIN_SPACING if spacing_code == ROTATE_SPACING else ROTATE_SPACING

    fused = _render_markers(g.shape, markers, order=PROCESS_ORDER, spacing_code=spacing_code)
    reversed_order = _render_markers(
        g.shape,
        markers,
        order=tuple(reversed(PROCESS_ORDER)),
        spacing_code=spacing_code,
    )
    toggled_rotation = _render_markers(
        g.shape,
        markers,
        order=PROCESS_ORDER,
        spacing_code=alternate_code,
    )
    coverage = _coverage_map(g.shape, markers, spacing_code=spacing_code)

    return {
        "spacing_code": spacing_code,
        "real_markers": len(markers),
        "focus_markers": sum(1 for color, _r, _c in markers if color == 5),
        "overlap_pixels": int(np.count_nonzero(coverage >= 2)),
        "order_diff_pixels": int(np.count_nonzero(fused != reversed_order)),
        "rotation_diff_pixels": int(np.count_nonzero(fused != toggled_rotation)),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(300):
        g = _build_marker_grid(rng)
        out = main(g)
        metrics = _variant_metrics(g)
        if metrics["real_markers"] < 6:
            continue
        if metrics["focus_markers"] < 2:
            continue
        if metrics["overlap_pixels"] < 6:
            continue
        if metrics["order_diff_pixels"] < 4:
            continue
        if metrics["rotation_diff_pixels"] < 4:
            continue
        assert out.shape == g.shape
        return g

    raise RuntimeError("failed to generate a visible D7/2 + D9/0 combo instance")
