# Variant of 58490d8a
# Dimension: D3 — 目标位置：图例框的结构 (dst)
# Change: 黑色矩形框，内部第2列（index=1）每隔一行放置一个关键颜色像素作为图例条目，输出画布即为该框的裁剪区域... → 图例改为行列式网格

from __future__ import annotations

import random
import sys
from functools import lru_cache
from itertools import combinations

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis for the D3 variant
# a) Concrete structure and behavior:
#    The old task used one black rectangle whose interior was also black, with key pixels stacked in one column.
#    This variant turns the destination into a visible 2x2 legend grid. The legend keeps a black outer border and
#    one black horizontal + one black vertical divider. Each cell has a bg-colored interior; on the center row of
#    each cell, the leftmost interior pixel is the key color in the input. In the output, that pixel grows rightward
#    into a same-color tally segment whose length equals the number of outside connected components of that color.
#    Example: old logic for color red wrote dots on one global row at columns 1,3,5,... inside a single column-based
#    legend. New logic still counts outside red components the same way, but now red owns one specific grid cell,
#    and its dots stay within that cell on a local left-to-right track. The solver therefore has to decode cell
#    boundaries before it can assign counts to locations.
# b) Complexity calibration:
#    A true grid needs at least two row groups and two column groups; otherwise it visually collapses back into a list.
#    To make the D3 change obvious in every sample, the generator fixes the number of key colors to 4 and fills a 2x2
#    layout completely. Counts still vary from 1 to 5, so the original counting rule remains intact while the target
#    structure becomes genuinely two-dimensional.
# c) Old implicit assumptions that break here:
#    - The original solver searched for the leftmost non-black column of the legend bbox. That fails now because cell
#      interiors use the background color, so many columns are non-black before the key pixels.
#    - It assumed keys live on a single global column and that output rows can be processed independently. The variant
#      distributes keys across separate cells, so the solver must infer row spans and column spans from full-black
#      separator lines.
#    - It initialized the output as all-black. That would erase the visible grid. The new solver reconstructs the
#      legend frame as black separator lines plus bg-colored cell interiors, then draws tallies per cell.
# d) Placement legality:
#    The legend patch is built output-first and pasted onto a bg canvas exactly as before. Outside sprites still use
#    `random_free_location` with one-cell 8-connected padding, so they cannot touch each other or the legend grid.
#    The legend bbox itself remains a single connected black component because its border and dividers all connect.
# e) Visual observability:
#    A human can now see four legend entries arranged in quadrants rather than one column. The only thing that changes
#    between input and output inside each cell is the horizontal growth of same-color dots, which is easy to compare
#    across 2-4 examples. No color-id ordering is used; cell order is purely row-major in visible space.
# f) Visual cleanliness:
#    Black lines only mark the frame and dividers; cell interiors are plain bg. Key pixels and tally dots live on the
#    center row of each cell, so there is no overlap between separators and marks. All four cells share one consistent
#    geometry, avoiding clutter.
# g) Color inference:
#    Colors are not remapped. Each cell's key pixel explicitly identifies which outside sprite color should be counted,
#    so the extra spatial complexity comes from legend decoding rather than hidden color transforms.
# h) Variant-specific generator guardrail:
#    Because every sample has one internal horizontal divider and one internal vertical divider, the generated legend is
#    never visually mistaken for the original one-column legend. The post-generation checks also compare against the
#    legacy solver externally during validation.
# i) New solver assumptions and generator guarantees:
#    The solver assumes the legend bbox contains full-black separator rows/cols and that each populated cell has at
#    least one non-black, non-bg key pixel. The generator guarantees this by constructing a complete 2x2 frame and
#    placing exactly one key pixel at the left edge of every cell.


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
DISTRACTOR_PROB = 0.5

# Grid legend layout: always a filled 2x2 table.
CELL_GRID_ROWS = 2
CELL_GRID_COLS = 2
KEY_COUNT = CELL_GRID_ROWS * CELL_GRID_COLS
CELL_INNER_HEIGHT = 3
CELL_CENTER_ROW = CELL_INNER_HEIGHT // 2
CELL_MIN_WIDTH = 4
CELL_EXTRA_WIDTH_RANGE = (0, 2)

# Counting and source sprites.
TARGET_COUNT_RANGE = (1, 5)
SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

# Canvas size and legend placement.
GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

# Retry budget.
GENERATION_ATTEMPTS = 300


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _legend_shape(cell_inner_width: int) -> tuple[int, int]:
    bh = CELL_GRID_ROWS * CELL_INNER_HEIGHT + (CELL_GRID_ROWS + 1)
    bw = CELL_GRID_COLS * cell_inner_width + (CELL_GRID_COLS + 1)
    return bh, bw


def _cell_bounds_from_width(cell_inner_width: int) -> list[tuple[int, int, int, int]]:
    bounds = []
    for rr in range(CELL_GRID_ROWS):
        r0 = 1 + rr * (CELL_INNER_HEIGHT + 1)
        r1 = r0 + CELL_INNER_HEIGHT - 1
        for cc in range(CELL_GRID_COLS):
            c0 = 1 + cc * (cell_inner_width + 1)
            c1 = c0 + cell_inner_width - 1
            bounds.append((r0, c0, r1, c1))
    return bounds


def _frame_patch(shape: tuple[int, int], bg: int) -> np.ndarray:
    patch = np.full(shape, int(bg), dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    for rr in range(1, CELL_GRID_ROWS):
        line_r = rr * (CELL_INNER_HEIGHT + 1)
        patch[line_r, :] = BLACK
    return patch


def _decorate_vertical_dividers(patch: np.ndarray, cell_inner_width: int) -> None:
    for cc in range(1, CELL_GRID_COLS):
        line_c = cc * (cell_inner_width + 1)
        patch[:, line_c] = BLACK


def _find_legend_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    best = None
    for coords in components_mask(g == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != BLACK).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return tuple(map(int, best[1]))


def _cell_spans_from_box(box: np.ndarray) -> list[tuple[int, int, int, int]]:
    black_rows = [rr for rr in range(box.shape[0]) if np.all(box[rr, :] == BLACK)]
    black_cols = [cc for cc in range(box.shape[1]) if np.all(box[:, cc] == BLACK)]

    row_spans = [(a + 1, b - 1) for a, b in zip(black_rows, black_rows[1:]) if b - a > 1]
    col_spans = [(a + 1, b - 1) for a, b in zip(black_cols, black_cols[1:]) if b - a > 1]

    cells = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cells.append((int(r0), int(c0), int(r1), int(c1)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and write the tallies into a gridded legend.

    Input structure:
    - A black connected legend frame is divided into multiple rectangular cells by black separator lines.
    - Each cell has a background-colored interior with one key-colored pixel near its left edge, while same-colored objects lie outside the frame.

    Rule:
    - For each populated cell, count the 8-connected components of that cell's key color outside the legend bbox.
    - Output the cropped legend grid: preserve the black separators and fill the cell's center row from left to right with one key-colored dot per counted object.

    Key insight: The counting rule is unchanged, but the destination is no longer a single indexed column; the solver must first parse the legend as a 2D table of cells and then tally each cell locally.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    r0, c0, r1, c1 = _find_legend_bbox(g)
    box = crop_bbox(g, (r0, c0, r1, c1))

    outside = np.ones((H, W), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    out = np.where(box == BLACK, BLACK, bg)

    for cell_r0, cell_c0, cell_r1, cell_c1 in _cell_spans_from_box(box):
        cell = box[cell_r0 : cell_r1 + 1, cell_c0 : cell_c1 + 1]
        marks = np.argwhere((cell != BLACK) & (cell != bg))
        if marks.size == 0:
            continue

        key_idx = min(range(len(marks)), key=lambda i: (int(marks[i][1]), int(marks[i][0])))
        key_r, key_c = map(int, marks[key_idx])
        color = int(cell[key_r, key_c])

        mask = (g == color) & outside
        _labels, count = label_mask(mask, connectivity=8)

        draw_r = cell_r0 + key_r
        for j in range(int(count)):
            draw_c = cell_c0 + key_c + j
            if draw_c <= cell_c1:
                out[draw_r, draw_c] = color

    return out


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    masks = []
    for cells in combinations(range(h * w), area):
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell // w, cell % w] = True
        spans_edges = mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
        if not spans_edges:
            continue
        _lab, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if n == 1:
            masks.append(mask)
    return tuple(masks)


def _make_sprite(color: int, bg: int) -> np.ndarray:
    area = _randint(SPRITE_AREA_RANGE)
    dims = []
    for h, w in SPRITE_BBOXES:
        if h * w >= area and _valid_masks(h, w, area):
            dims.append((h, w))
    h, w = random.choice(dims)
    mask = random.choice(_valid_masks(h, w, area))
    sprite = np.full((h, w), int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _sample_palette() -> tuple[int, list[int], int | None]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, KEY_COUNT)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts() -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(KEY_COUNT)]


def _cell_inner_width(max_count: int) -> int:
    return max(CELL_MIN_WIDTH, int(max_count)) + _randint(CELL_EXTRA_WIDTH_RANGE)


def _build_expected(key_colors: list[int], counts: list[int], bg: int, cell_inner_width: int) -> np.ndarray:
    expected = _frame_patch(_legend_shape(cell_inner_width), bg)
    _decorate_vertical_dividers(expected, cell_inner_width)

    for (r0, c0, _r1, c1), color, count in zip(_cell_bounds_from_width(cell_inner_width), key_colors, counts):
        draw_r = r0 + CELL_CENTER_ROW
        for offset in range(int(count)):
            draw_c = c0 + offset
            if draw_c <= c1:
                expected[draw_r, draw_c] = int(color)
    return expected


def _build_legend(key_colors: list[int], bg: int, cell_inner_width: int) -> np.ndarray:
    legend = _frame_patch(_legend_shape(cell_inner_width), bg)
    _decorate_vertical_dividers(legend, cell_inner_width)

    for (r0, c0, _r1, _c1), color in zip(_cell_bounds_from_width(cell_inner_width), key_colors):
        legend[r0 + CELL_CENTER_ROW, c0] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> np.ndarray | None:
    bh, bw = map(int, legend.shape)
    H = _randint(GRID_HEIGHT_RANGE)
    W = _randint(GRID_WIDTH_RANGE)
    min_h = bh + 2 * LEGEND_EDGE_MARGIN
    min_w = bw + 2 * LEGEND_EDGE_MARGIN
    if H < min_h or W < min_w:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    r0 = random.randint(LEGEND_EDGE_MARGIN, H - bh - LEGEND_EDGE_MARGIN)
    c0 = random.randint(LEGEND_EDGE_MARGIN, W - bw - LEGEND_EDGE_MARGIN)
    paste(grid, legend, r0, c0)
    return grid


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, count in zip(key_colors, counts):
        sprite = _make_sprite(color, bg)
        for _ in range(count):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        sprites.append(_make_sprite(distractor_color, bg))
    random.shuffle(sprites)
    return sprites


def _place_sprites(grid: np.ndarray, sprites: list[np.ndarray], bg: int) -> None:
    for sprite in sprites:
        sr, sc = random_free_location(
            grid,
            sprite,
            transparent=bg,
            padding=SPRITE_PADDING,
            padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
        )
        paste(grid, sprite, sr, sc, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts()
    cell_inner_width = _cell_inner_width(max(counts))

    expected = _build_expected(key_colors, counts, bg, cell_inner_width)
    legend = _build_legend(key_colors, bg, cell_inner_width)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
    try:
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a D3 input")
