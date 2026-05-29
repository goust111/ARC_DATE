# Variant of dfadab01
# Dimension: D10 — 字形之间的重叠处理 (meta)
# Change: 标记间距足够大，字形之间通常不重叠；若发生边缘裁剪则简单截断... → 重叠区域标记为特殊颜色

from __future__ import annotations

import random

import numpy as np


OVERLAP_COLOR = 9

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

# Design analysis
# design:
#   a_concrete_behavior_and_trace:
#     input_still_uses_single_pixel_markers plus an optional legend glyph with a same-type marker at (+4,+4).
#     output_still_expands each real marker into a fixed 4x4 glyph anchored at the marker cell.
#     new_difference: when two or more expanded glyphs cover the same output cell, that cell becomes 9 instead
#     of whichever glyph happened to be painted last.
#     example:
#       original: markers 2 at (1,1) and 3 at (3,1) create two 4x4 boxes with 2-row overlap; the later loop
#       silently overwrites shared cells.
#       variant: both glyphs are still expanded, but every shared output cell in rows 3-4 / cols 1-4 is marked 9.
#   b_complexity_calibration:
#     keep the same 4-symbol alphabet and optional single legend so the only new degree of freedom is overlap.
#     generator varies spacing mainly along one axis, sometimes both, so overlap is visible without making the
#     canvas unreadably dense.
#   c_hidden_assumptions_in_original_main:
#     _paint_clipped previously assumed one glyph owns each output cell; unconditional assignment was harmless only
#     because spacing>=4 made collisions impossible. That assumption no longer holds, so rendering must count
#     coverage before choosing colors.
#     legend detection via _has_source_glyph still works because the legend remains a fully drawn 4x4 sample in
#     colors 1/4/6/7, which are disjoint from marker colors 2/3/5/8.
#   d_placement_legality:
#     generator never overlaps input markers with each other; overlap happens only after expansion in output space.
#     optional legends are placed only in empty 4x4 areas with an empty label cell, so they cannot erase markers
#     or create false skip cases.
#   e_visual_observability:
#     every generated sample requires overlap_count>0 and also keeps some non-overlapped glyph pixels, so the red
#     9-cells are visibly a new phenomenon rather than total repainting.
#   f_visual_cleanliness:
#     dense spacing is usually applied to one axis while the other axis stays near the original 4-5 gap, producing
#     readable vertical or horizontal collision bands. Extremely saturated overlap cases are rejected.
#   g_color_inference:
#     the only new color is 9, and its role is structurally indicated by exact coincidence regions of multiple
#     glyphs. No invisible ordering or color-id arithmetic is used.
#   main_generator_contract:
#     main assumes legends, if present, are complete 4x4 exemplars with the label at (+4,+4); generate_input
#     enforces that. main also assumes at least one real marker exists; generator always places several.


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _accumulate_clipped(
    cover_count: np.ndarray,
    cover_color: np.ndarray,
    r0: int,
    c0: int,
    color: int,
    mask: np.ndarray,
) -> None:
    h, w = cover_count.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                if cover_count[rr, cc] == 0:
                    cover_color[rr, cc] = color
                cover_count[rr, cc] += 1


def _render_markers(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> tuple[np.ndarray, int]:
    cover_count = np.zeros(shape, dtype=int)
    cover_color = np.zeros(shape, dtype=int)
    for r, c, marker_color in markers:
        glyph_color, mask = GLYPHS[marker_color]
        _accumulate_clipped(cover_count, cover_color, r, c, glyph_color, mask)

    out = np.zeros(shape, dtype=int)
    single = cover_count == 1
    out[single] = cover_color[single]
    out[cover_count >= 2] = OVERLAP_COLOR
    return out, int(np.count_nonzero(cover_count >= 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each true control pixel into its 4x4 glyph and expose overlaps in red.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one legend entry: a full 4x4 sample glyph in its output color plus a matching marker at one cell below-right of the sample.

    Rule:
    - Ignore legend labels whose matching sample glyph sits at (-4, -4) from the marker.
    - Replace every remaining marker by its fixed 4x4 glyph, anchored at the marker position and clipped by the grid border.
    - Cells covered by exactly one expanded glyph keep that glyph's normal color; cells covered by two or more glyphs become the special overlap color 9.

    Key insight:
    - The marker-to-glyph dictionary is unchanged from the original task; the only semantic change is that rendering is now collision-aware, so shared coverage must be detected instead of resolved by paint order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

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

    assert seen_marker and markers
    out, _ = _render_markers(g.shape, markers)
    return out


def _choose_layout(rng: random.Random, size: int) -> tuple[list[int], list[int]]:
    layout = rng.choices(
        ["row_dense", "col_dense", "both_dense"],
        weights=[4, 4, 1],
        k=1,
    )[0]
    row_dense = layout in {"row_dense", "both_dense"}
    col_dense = layout in {"col_dense", "both_dense"}

    if size == 10:
        if layout == "both_dense":
            rows = rng.randint(2, 3)
            cols = rng.randint(2, 3)
        else:
            rows = rng.randint(3, 4) if row_dense else rng.randint(2, 3)
            cols = rng.randint(3, 4) if col_dense else rng.randint(2, 3)
    else:
        if layout == "both_dense":
            rows = rng.randint(3, 4)
            cols = rng.randint(3, 4)
        else:
            rows = rng.randint(3, 5) if row_dense else rng.randint(2, 4)
            cols = rng.randint(3, 5) if col_dense else rng.randint(2, 4)

    row_steps = _axis_steps(rng, size, rows, dense=row_dense)
    col_steps = _axis_steps(rng, size, cols, dense=col_dense)
    return _axis_positions(rng, size, row_steps), _axis_positions(rng, size, col_steps)


def _axis_steps(rng: random.Random, size: int, count: int, dense: bool) -> list[int]:
    palette = (2, 3) if dense else (4, 5)
    for _ in range(60):
        steps = [rng.choice(palette) for _ in range(max(0, count - 1))]
        if sum(steps) <= size - 1:
            return steps
    raise RuntimeError("could not pick feasible axis steps")


def _axis_positions(rng: random.Random, size: int, steps: list[int]) -> list[int]:
    max_start = size - 1 - sum(steps)
    assert max_start >= 0
    anchor_edge = rng.random() < 0.3
    start = max_start if anchor_edge else rng.randint(0, max_start)
    positions = [start]
    for step in steps:
        positions.append(positions[-1] + step)
    return positions


def _choose_markers(
    rng: random.Random,
    row_positions: list[int],
    col_positions: list[int],
) -> list[tuple[int, int, int]]:
    cells = [(r, c) for r in row_positions for c in col_positions]
    total = len(cells)
    min_count = max(4, int(np.ceil(total * 0.7)))
    active_count = rng.randint(min_count, total)
    active_cells = rng.sample(cells, active_count)

    max_types = 4 if active_count >= 6 and rng.random() < 0.2 else 3
    type_count = rng.randint(2, min(max_types, active_count))
    used_types = rng.sample(sorted(GLYPHS), type_count)
    palette = used_types[:]
    while len(palette) < active_count:
        palette.append(rng.choice(used_types))
    rng.shuffle(palette)
    return [(r, c, color) for (r, c), color in zip(active_cells, palette, strict=True)]


def _find_legend_spots(size: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    spots: list[tuple[int, int]] = []
    for r0 in range(size - 4):
        for c0 in range(size - 4):
            cells = {(r, c) for r in range(r0, r0 + 4) for c in range(c0, c0 + 4)}
            label = (r0 + 4, c0 + 4)
            if cells & occupied:
                continue
            if label in occupied:
                continue
            spots.append((r0, c0))
    return spots


def _add_legend(
    rng: random.Random,
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
) -> None:
    if rng.random() >= 0.6:
        return
    occupied = {(r, c) for r, c, _ in markers}
    spots = _find_legend_spots(grid.shape[0], occupied)
    if not spots:
        return

    marker_color = rng.choice([color for _, _, color in markers])
    glyph_color, mask = GLYPHS[marker_color]
    r0, c0 = rng.choice(spots)
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                grid[r0 + dr, c0 + dc] = glyph_color
    grid[r0 + 4, c0 + 4] = marker_color


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        size = rng.choice([10, 20])
        row_positions, col_positions = _choose_layout(rng, size)
        markers = _choose_markers(rng, row_positions, col_positions)
        expected, overlap_pixels = _render_markers((size, size), markers)

        painted_pixels = int(np.count_nonzero(expected))
        if overlap_pixels == 0:
            continue
        if overlap_pixels < 3:
            continue
        if overlap_pixels * 2 > painted_pixels:
            continue

        grid = np.zeros((size, size), dtype=int)
        for r, c, color in markers:
            grid[r, c] = color
        _add_legend(rng, grid, markers)

        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a non-degenerate overlap sample")
