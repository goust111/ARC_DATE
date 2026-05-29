# Combo variant of dfadab01 (index 18)
# combo_spec: D3/4+D12/0
# Name: 十字标记编码字形+减少标记色

from __future__ import annotations

import itertools
import random

import numpy as np

"""
Design analysis for the combo variant.

a) Concrete behavior and trace:
   - Original task: each real control is one pixel whose color directly names one of
     four fixed 4x4 glyphs. A legend, when present, shows the glyph itself plus a
     below-right label pixel of the same marker color.
   - Combo behavior: each real control is now a two-color cross. The center cell is
     either red (2) or green (3), and every arm cell uses the opposite color. The
     center color picks one of two glyph families; the arm count picks one of two
     glyphs inside that family.
   - Concrete trace: a red center at (6, 9) with two green arms no longer means
     "paint whatever red used to mean". Instead the solver reads the center color
     (red family) and counts two opposite-color arms, then paints the yellow frame
     glyph with top-left anchor at (6, 9). A green center with the same two-arm count
     maps to a different glyph, so color alone and count alone are both insufficient.
   - Legends still consist of one sample output glyph and a matching label control at
     (+4, +4) from the sample glyph's top-left corner. The label is a full combo-style
     cross and is removed in the output.

b) Complexity calibration:
   - The fused rule keeps four target glyphs, but compresses the marker palette to two
     colors total. This preserves enough variety to stay nontrivial while making the
     input visibly cleaner than the source task.
   - To avoid degenerating into a count-only rule, both center colors reuse the same
     two arm-count values: 2 and 4. The four semantic types are therefore the Cartesian
     product {red-center, green-center} x {2 arms, 4 arms}.
   - Counts 2 and 4 are repeated across samples and are visually distinct, while
     two-arm direction sets vary so the stable feature is the count rather than the
     orientation.

c) Implicit-assumption audit for main():
   - The original loop over marker colors 2/3/5/8 breaks immediately because input
     colors no longer identify glyphs one-to-one.
   - The original logic assumed every marker cell was itself the anchor. Here only the
     center cell is the anchor, so the solver must reject arm pixels. Using only arm
     counts {2, 4} guarantees every arm pixel has exactly one opposite-color neighbor
     and never masquerades as a valid center.
   - The legend-skip check still depends on the control anchor being exactly four cells
     below-right of the sample glyph. That remains valid because the cross center is the
     logical anchor.
   - Painting still assumes output glyphs do not overlap. The generator keeps anchor
     spacing at least four, so clipped 4x4 outputs remain disjoint.

d) Placement legality:
   - Real marker centers are restricted to rows/cols 1..H-2 so every possible arm stays
     in bounds.
   - Marker anchors live on a sparse regular grid with gaps 4-6, which prevents cross
     overlap in the input and glyph overlap in the output.
   - Legend placement is accepted only if the sample glyph and its label cross are fully
     in bounds and disjoint from every real marker cell.

e) Visual observability:
   - The input now visibly differs from the seed task because controls are small crosses
     instead of isolated pixels, and because only red/green are used for all controls.
   - Across examples, identical arm counts appear under both center colors, and both arm
     counts appear under the same center color. This makes the dual encoding observable
     from a few training pairs.

f) Visual cleanliness:
   - Crosses are isolated by black padding, so no marker touches a glyph-colored legend
     sample or another control.
   - Using only two marker colors keeps the scene readable even when many controls are
     present.

g) Change inferability:
   - No hidden numeric ordering is used. The only inputs to decoding are visible center
     color and visible orthogonal arm count.
   - The generator always places all four semantic marker types among the real controls,
     so the combined color+count code is exercised instead of collapsing to a simpler
     subrule.
"""


RED = 2
GREEN = 3
MARKER_COLORS: tuple[int, int] = (RED, GREEN)
CARDINAL_DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARM_COUNTS: tuple[int, int] = (2, 4)

GLYPHS: dict[tuple[int, int], tuple[int, np.ndarray]] = {
    (RED, 2): (
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
    (RED, 4): (
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
    (GREEN, 2): (
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
    (GREEN, 4): (
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

ARM_SUBSETS: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    count: tuple(itertools.combinations(CARDINAL_DIRS, count)) for count in ARM_COUNTS
}


def _other_marker_color(color: int) -> int:
    return GREEN if color == RED else RED


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


def _count_opposite_arms(g: np.ndarray, r: int, c: int) -> int:
    h, w = g.shape
    center_color = int(g[r, c])
    arm_color = _other_marker_color(center_color)
    count = 0
    for dr, dc in CARDINAL_DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == arm_color:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each two-color cross by combining its center color and arm count, then paint the matched 4x4 glyph.

    Input structure:
    - Black background with control markers drawn only in red and green.
    - Each control is a small cross whose center is one of those colors and whose arms use the opposite color.
    - The center color selects a glyph family, while the number of orthogonal arms selects one of two glyphs inside that family; an optional legend still shows one output glyph plus a matching label cross at (+4, +4).

    Rule:
    - Scan red and green cells as possible cross centers, count how many opposite-color arm cells touch orthogonally, and keep only the valid two-arm or four-arm centers.
    - Map `(center color, arm count)` to the learned glyph/color pair, then paint the 4x4 glyph with its top-left corner at the center cell, clipping at the grid boundary.
    - If that center is the label cross for a legend glyph sitting four cells up-left, skip it; legend samples themselves are removed from the output.

    Key insight:
    - The marker code is genuinely two-dimensional: the same arm count means different glyphs under different center colors, and the same center color means different glyphs under different arm counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, RED, GREEN, 1, 4, 6, 7})

    out = np.zeros_like(g)
    candidates = np.argwhere(np.isin(g, MARKER_COLORS))
    assert candidates.size

    seen_marker = False
    for r, c in candidates:
        center_color = int(g[r, c])
        arm_count = _count_opposite_arms(g, int(r), int(c))
        glyph = GLYPHS.get((center_color, arm_count))
        if glyph is None:
            continue
        seen_marker = True
        glyph_color, mask = glyph
        if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
            continue
        _paint_clipped(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


def _sample_axis_positions(length: int, count: int, rng: random.Random) -> list[int]:
    max_center = length - 2
    for _ in range(200):
        gaps = [rng.randint(4, 6) for _ in range(count - 1)]
        span = sum(gaps)
        max_start = max_center - span
        if max_start < 1:
            continue
        start = rng.randint(1, max_start)
        pos = [start]
        for gap in gaps:
            pos.append(pos[-1] + gap)
        return pos
    raise RuntimeError("failed to sample axis positions")


def _marker_cells(r: int, c: int, arm_dirs: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    cells = {(r, c)}
    for dr, dc in arm_dirs:
        cells.add((r + dr, c + dc))
    return cells


def _draw_marker(
    grid: np.ndarray, r: int, c: int, center_color: int, arm_dirs: tuple[tuple[int, int], ...]
) -> None:
    grid[r, c] = center_color
    arm_color = _other_marker_color(center_color)
    for dr, dc in arm_dirs:
        grid[r + dr, c + dc] = arm_color


def _choose_arm_dirs(arm_count: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    return tuple(rng.choice(ARM_SUBSETS[arm_count]))


def _build_expected_output(
    shape: tuple[int, int], markers: list[tuple[int, int, int, int]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, center_color, arm_count in markers:
        glyph_color, mask = GLYPHS[(center_color, arm_count)]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _find_legend(
    shape: tuple[int, int],
    used_types: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, int, tuple[tuple[int, int], ...]] | None:
    h, w = shape
    order = [(r, c) for r in range(h - 5) for c in range(w - 5)]
    rng.shuffle(order)
    type_order = used_types[:]
    rng.shuffle(type_order)

    for center_color, arm_count in type_order:
        glyph_color, mask = GLYPHS[(center_color, arm_count)]
        arm_variants = list(ARM_SUBSETS[arm_count])
        rng.shuffle(arm_variants)
        for r0, c0 in order:
            glyph_cells = {
                (r0 + dr, c0 + dc) for dr in range(4) for dc in range(4) if bool(mask[dr, dc])
            }
            for arm_dirs in arm_variants:
                center_r, center_c = r0 + 4, c0 + 4
                label_cells = _marker_cells(center_r, center_c, tuple(arm_dirs))
                all_cells = glyph_cells | label_cells
                if not all(0 <= rr < h and 0 <= cc < w for rr, cc in all_cells):
                    continue
                if all_cells & occupied:
                    continue
                return r0, c0, center_color, tuple(arm_dirs)
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    base_types = list(GLYPHS.keys())

    for _ in range(200):
        size = rng.choice([10, 20])
        h = w = size
        if size == 10:
            row_count = col_count = 2
        else:
            row_count = rng.randint(2, 5)
            col_count = rng.randint(2, 5)

        row_positions = _sample_axis_positions(h, row_count, rng)
        col_positions = _sample_axis_positions(w, col_count, rng)
        slots = [(r, c) for r in row_positions for c in col_positions]
        min_fill = max(4, int(np.ceil(0.6 * len(slots))))
        occupied_count = rng.randint(min_fill, len(slots))
        chosen_slots = rng.sample(slots, occupied_count)

        marker_types = base_types[:] + [rng.choice(base_types) for _ in range(occupied_count - 4)]
        rng.shuffle(marker_types)

        grid = np.zeros((h, w), dtype=int)
        occupied_cells: set[tuple[int, int]] = set()
        markers: list[tuple[int, int, int, int]] = []
        valid = True

        for (r, c), (center_color, arm_count) in zip(chosen_slots, marker_types, strict=True):
            arm_dirs = _choose_arm_dirs(arm_count, rng)
            cells = _marker_cells(r, c, arm_dirs)
            if any(cell in occupied_cells for cell in cells):
                valid = False
                break
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                valid = False
                break
            _draw_marker(grid, r, c, center_color, arm_dirs)
            occupied_cells |= cells
            markers.append((r, c, center_color, arm_count))

        if not valid:
            continue

        if rng.random() < 0.75:
            legend = _find_legend((h, w), base_types, occupied_cells, rng)
            if legend is not None:
                r0, c0, center_color, arm_dirs = legend
                arm_count = len(arm_dirs)
                glyph_color, mask = GLYPHS[(center_color, arm_count)]
                for dr in range(4):
                    for dc in range(4):
                        if bool(mask[dr, dc]):
                            grid[r0 + dr, c0 + dc] = glyph_color
                _draw_marker(grid, r0 + 4, c0 + 4, center_color, arm_dirs)

        expected = _build_expected_output((h, w), markers)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
