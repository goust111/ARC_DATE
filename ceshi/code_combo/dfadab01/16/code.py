# Combo variant of dfadab01 (index 16)
# combo_spec: D1/3+D12/0
# Name: 字形嵌套+减少标记色

from __future__ import annotations

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
}

PARTNER: dict[int, int] = {2: 3, 3: 2}
MICRO_GLYPHS: dict[int, np.ndarray] = {
    2: np.ones((2, 2), dtype=bool),
    3: np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=bool,
    ),
}


def _make_nested_pattern(marker_color: int) -> np.ndarray:
    outer_color, outer_mask = GLYPHS[marker_color]
    inner_marker = PARTNER[marker_color]
    inner_color, _inner_mask = GLYPHS[inner_marker]
    inner_micro = MICRO_GLYPHS[inner_marker]

    pattern = np.zeros((4, 4), dtype=int)
    pattern[outer_mask] = outer_color
    pattern[1:3, 1:3][inner_micro] = inner_color
    return pattern


PATTERNS: dict[int, np.ndarray] = {
    marker_color: _make_nested_pattern(marker_color) for marker_color in GLYPHS
}


# Design analysis for D1/3 + D12/0
# a) Concrete behavior:
#    Original: a red marker 2 at (r,c) became a hollow yellow 4x4 frame; a green marker 3 became a hollow blue ring.
#    Variant: only markers 2 and 3 remain. Marker 2 now becomes the yellow 4x4 frame with the center 2x2 filled by
#    marker-3's micro-glyph; marker 3 becomes the blue 4x4 ring with the center 2x2 filled by marker-2's micro-glyph.
#    Example patch for marker 2 is [[4,4,4,4],[4,0,1,4],[4,1,0,4],[4,4,4,4]].
# b) Complexity calibration:
#    Reducing to two marker types keeps the mapping learnable, but the mutual nesting adds visible structure. Both colors
#    are forced to repeat, so the solver cannot get away with a single one-off coincidence.
# c) Hidden assumptions audit:
#    The original code assumed every glyph could be represented by a single output color mask and that legends showed the
#    same flat glyph as the output. That is no longer true. The new solver therefore paints full 4x4 integer patterns and
#    detects legend labels by exact nested-pattern matching rather than mask/color matching.
# d) Placement legality:
#    Real markers still live on a regular grid with spacing 4-6, so two 4x4 outputs never overlap. Legends are only
#    placed when their 4x4 sample and (+4,+4) label fit fully and stay away from real marker output footprints.
# e) Visual observability:
#    The two allowed outer glyphs both have a hollow 2x2 cavity, so humans can directly see that the cavity is no longer
#    empty. The partner relation is symmetric: frame contains mini-ring, ring contains mini-frame.
# f) Visual cleanliness:
#    Only two marker colors are used, and only one nesting layer is added. No object stacks on top of another object's
#    output area, so the grid remains readable.
# g) Color-change inferability:
#    Colors stay tied to the original glyph identities (2->4 outer frame, 3->1 outer ring). The inner color is always the
#    partner glyph's output color, which is supported repeatedly within the same example.
# h) Micro-glyph choice:
#    A 2x2 "closed frame" cannot preserve a hollow center, so the partner mini-frame is rendered as the only legible
#    shorthand: a solid 2x2 square. The partner mini-ring is rendered as a 2-pixel diagonal signature. This keeps the two
#    inner miniatures visually distinct while still reading as reduced versions of the original outer glyphs.
# i) New solver assumptions and generator guarantees:
#    `_is_legend_label` assumes legends are complete unclipped nested 4x4 samples with their label exactly one cell below
#    and right. `generate_input()` guarantees that shape, guarantees both marker colors are present, and keeps colors 1/4
#    exclusive to legends so real control pixels remain unambiguous.


def _paint_clipped(out: np.ndarray, r0: int, c0: int, pattern: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and int(pattern[dr, dc]) != 0:
                out[rr, cc] = int(pattern[dr, dc])


def _is_legend_label(g: np.ndarray, r: int, c: int, marker_color: int) -> bool:
    r0 = r - 4
    c0 = c - 4
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r >= h or c >= w:
        return False
    if r0 + 4 > h or c0 + 4 > w:
        return False
    return bool(np.array_equal(g[r0 : r0 + 4, c0 : c0 + 4], PATTERNS[marker_color]))


def _real_marker_coords(g: np.ndarray) -> list[tuple[int, int, int]]:
    coords: list[tuple[int, int, int]] = []
    for marker_color in GLYPHS:
        for r, c in np.argwhere(g == marker_color):
            rr = int(r)
            cc = int(c)
            if not _is_legend_label(g, rr, cc, marker_color):
                coords.append((marker_color, rr, cc))
    return coords


def _fully_visible_marker_colors(g: np.ndarray) -> list[int]:
    h, w = g.shape
    colors: list[int] = []
    for marker_color, r, c in _real_marker_coords(g):
        if r + 3 < h and c + 3 < w:
            colors.append(marker_color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each control pixel by a two-level glyph whose hollow center contains the other glyph's miniature.

    Input structure:
    - Black background with single-pixel markers in only two colors: 2 and 3.
    - Some inputs also include one legend entry: a fully drawn nested 4x4 output sample plus a matching marker at its lower-right corner.

    Rule:
    - Marker 2 expands to the yellow 4x4 frame, but its inner 2x2 cavity is filled by a blue 2x2 micro-version of glyph 3.
    - Marker 3 expands to the blue 4x4 ring, but its inner 2x2 cavity is filled by a yellow 2x2 micro-version of glyph 2.
    - Each nested glyph is anchored by its top-left corner at the marker position, clipped by the grid border, and legend samples with their labels disappear.

    Key insight:
    - The task no longer asks for four independent flat glyph substitutions; it asks for a symmetric two-type partnership where each outer glyph contains the other type's miniature.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, 1, 2, 3, 4})

    out = np.zeros_like(g)
    real_markers = _real_marker_coords(g)
    assert real_markers
    for marker_color, r, c in real_markers:
        _paint_clipped(out, r, c, PATTERNS[marker_color])
    return out


def _pick_axis_positions(size: int, count: int, spacing: int) -> list[int] | None:
    max_start = size - 1 - (count - 1) * spacing
    if max_start < 0:
        return None
    start = int(np.random.randint(0, max_start + 1))
    return [start + i * spacing for i in range(count)]


def _can_place_legend(grid: np.ndarray, marker_coords: list[tuple[int, int]], r0: int, c0: int) -> bool:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + 5 > h or c0 + 5 > w:
        return False
    if np.any(grid[r0 : r0 + 5, c0 : c0 + 5] != 0):
        return False
    for mr, mc in marker_coords:
        if abs(mr - r0) < 4 and abs(mc - c0) < 4:
            return False
    return True


def _add_legend_if_possible(
    grid: np.ndarray,
    marker_coords: list[tuple[int, int]],
    used_colors: set[int],
) -> None:
    h, w = grid.shape
    if h < 12 or w < 12 or np.random.rand() < 0.35:
        return

    legend_color = int(np.random.choice(sorted(used_colors)))
    for _ in range(100):
        r0 = int(np.random.randint(0, h - 4))
        c0 = int(np.random.randint(0, w - 4))
        if not _can_place_legend(grid, marker_coords, r0, c0):
            continue
        grid[r0 : r0 + 4, c0 : c0 + 4] = PATTERNS[legend_color]
        grid[r0 + 4, c0 + 4] = legend_color
        return


def _gen_attempt() -> np.ndarray | None:
    size = int(np.random.choice([10, 20]))
    row_spacing = int(np.random.randint(4, 7))
    col_spacing = int(np.random.randint(4, 7))

    max_rows = min(5, (size - 1) // row_spacing + 1)
    max_cols = min(5, (size - 1) // col_spacing + 1)
    if max_rows < 2 or max_cols < 2:
        return None

    rows = int(np.random.randint(2, max_rows + 1))
    cols = int(np.random.randint(2, max_cols + 1))
    row_positions = _pick_axis_positions(size, rows, row_spacing)
    col_positions = _pick_axis_positions(size, cols, col_spacing)
    if row_positions is None or col_positions is None:
        return None

    total_cells = rows * cols
    min_filled = max(4, int(np.ceil(0.6 * total_cells)))
    fill_count = int(np.random.randint(min_filled, total_cells + 1))
    flat_ids = np.arange(total_cells)
    np.random.shuffle(flat_ids)
    filled_ids = flat_ids[:fill_count]

    grid = np.zeros((size, size), dtype=int)
    marker_coords: list[tuple[int, int]] = []
    marker_colors = np.full(fill_count, 2, dtype=int)
    seeds = np.array([2, 3, 2, 3][:fill_count], dtype=int)
    marker_colors[: seeds.size] = seeds
    if fill_count > seeds.size:
        marker_colors[seeds.size :] = np.random.choice([2, 3], size=fill_count - seeds.size)
    np.random.shuffle(marker_colors)

    for idx, cell_id in enumerate(filled_ids):
        r = row_positions[int(cell_id) // cols]
        c = col_positions[int(cell_id) % cols]
        color = int(marker_colors[idx])
        grid[r, c] = color
        marker_coords.append((r, c))

    used_colors = {int(x) for x in marker_colors}
    if used_colors != {2, 3}:
        return None

    _add_legend_if_possible(grid, marker_coords, used_colors)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is None:
            continue
        real_markers = _real_marker_coords(grid)
        if len(real_markers) < 4:
            continue
        if {marker_color for marker_color, _r, _c in real_markers} != {2, 3}:
            continue
        visible_colors = _fully_visible_marker_colors(grid)
        if len(visible_colors) < 2 or set(visible_colors) != {2, 3}:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("generate_input failed after 400 attempts")
