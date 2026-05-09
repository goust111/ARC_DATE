# Combo variant of 58490d8a (index 29)
# combo_spec: D3/0+D14/0
# Name: 网格图例+排序重排

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


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D3/0 + D14/0
#
# a) Concrete behavior and trace
#    - The original task used one black rectangle with keys stacked on one
#      column; each key stayed on its own row after counting.
#    - This combo turns the destination into a 2x2 gridded legend. Each cell
#      has a bg-colored interior, one key pixel on its center row at the left
#      edge, and room to grow a horizontal tally inside that cell.
#    - The input cell order is only the measurement order. After counting
#      outside sprites, colors are sorted by count descending and reassigned to
#      the four grid cells in row-major order.
#    - Example trace:
#        input cells (row-major) = [blue, red, yellow, green]
#        outside counts          = [2, 5, 3, 1]
#        no-sort grid output     = [blue x2, red x5, yellow x3, green x1]
#        combo grid output       = [red x5, yellow x3, blue x2, green x1]
#      The D3 change localizes tallies per cell, and the D14 change permutes
#      which color occupies each visible cell slot.
#
# b) Complexity calibration
#    - A genuine grid needs both multiple rows and multiple columns, so the
#      generator always uses a full 2x2 layout with four populated cells.
#    - To keep the sorting effect visible, the four counts are distinct and the
#      unsorted row-major input order is rejected if it already matches the
#      descending order. This prevents collapse back to either the original task
#      or the pure D3 grid-only variant.
#
# c) Implicit-assumption review
#    - The old "leftmost non-black column = key column" assumption breaks
#      because the legend interior is mostly background color, not black.
#    - The old "one global row per key" assumption also breaks: keys now live in
#      separate cells and the output destinations are rank positions, not the
#      original key positions.
#    - The old all-black output initialization would destroy the visible table.
#      The new solver must reconstruct black borders/dividers and keep cell
#      interiors background-colored.
#    - Sorting introduces a new assumption that every destination cell uses the
#      same local geometry. The generator enforces one shared cell size and one
#      shared key position in each cell.
#
# d) Placement legality
#    - The legend is built output-first, then converted back into an input patch
#      with only the unsorted key pixels kept. It is pasted onto the canvas
#      before any sprites are placed.
#    - Outside sprites still use `random_free_location` with one-cell 8-neighbor
#      padding, so they cannot overlap or touch the legend or each other.
#    - The black frame plus dividers form one connected black component, which
#      keeps legend detection stable.
#
# e) Visual observability
#    - Humans can see both control variables directly: a 2x2 table instead of a
#      single list, and a reordering driven by visibly different tally lengths.
#    - The sort key is only the counted multiplicity of same-color sprites, not
#      a hidden numeric property such as the color id.
#    - Tie-breaking uses original row-major cell order for determinism, but the
#      generator mostly avoids ties by sampling distinct counts.
#
# f) Visual cleanliness
#    - Black only marks the outer border and grid dividers. Cell interiors stay
#      plain background, and all tally dots remain on the center row of their
#      cells, so there is no separator overlap or spatial ambiguity.
#
# g) New solver assumptions and generator guarantees
#    - The solver assumes full-black separator rows and columns, one key color
#      per populated cell, and a shared local key offset within each cell. The
#      generator guarantees all three conditions explicitly.
#    - The solver sorts populated cells by counted multiplicity, then redraws
#      into the first N row-major cells. The generator keeps all four cells
#      populated, so the permutation is always over the complete 2x2 grid.
# ---------------------------------------------------------------------------


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
DISTRACTOR_PROB = 0.5

# Grid legend layout: always a full 2x2 table so the D3 change is unambiguous.
CELL_GRID_ROWS = 2
CELL_GRID_COLS = 2
KEY_COUNT = CELL_GRID_ROWS * CELL_GRID_COLS
CELL_INNER_HEIGHT = 3
CELL_CENTER_ROW = CELL_INNER_HEIGHT // 2
CELL_MIN_WIDTH = 5
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
            bounds.append((int(r0), int(c0), int(r1), int(c1)))
    return bounds


def _frame_patch(shape: tuple[int, int], bg: int) -> np.ndarray:
    patch = np.full(shape, int(bg), dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    for rr in range(1, CELL_GRID_ROWS):
        patch[rr * (CELL_INNER_HEIGHT + 1), :] = BLACK
    return patch


def _decorate_vertical_dividers(patch: np.ndarray, cell_inner_width: int) -> None:
    for cc in range(1, CELL_GRID_COLS):
        patch[:, cc * (cell_inner_width + 1)] = BLACK


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


def _sorted_key_records(key_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored sprites, then reorder a gridded legend by those counts.

    Input structure:
    - A connected black legend frame is split by full black separator lines into a 2D grid of rectangular cells.
    - Each populated cell contains one key-colored pixel at its left side on the center row, and matching monochrome sprites appear outside the frame on the background.

    Rule:
    - For every populated cell, count the 8-connected components of that key color outside the legend bbox.
    - Sort the cell records from largest count to smallest, break ties by the original row-major cell order, then output the cropped legend grid with each ranked color drawn as a left-to-right tally inside its new cell slot.

    Key insight:
    - The legend must be decoded twice: first as a spatial grid to know which colors to measure, and then as an ordered set of destination slots whose visible positions are reassigned after sorting.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    r0, c0, r1, c1 = _find_legend_bbox(g)
    box = crop_bbox(g, (r0, c0, r1, c1))
    cells = _cell_spans_from_box(box)

    outside = np.ones((H, W), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    out = np.where(box == BLACK, BLACK, bg)
    if not cells:
        return out

    records: list[tuple[int, int, int, int, int]] = []
    for cell_idx, (cell_r0, cell_c0, cell_r1, cell_c1) in enumerate(cells):
        cell = box[cell_r0 : cell_r1 + 1, cell_c0 : cell_c1 + 1]
        marks = np.argwhere((cell != BLACK) & (cell != bg))
        if marks.size == 0:
            continue

        key_idx = min(range(len(marks)), key=lambda i: (int(marks[i][1]), int(marks[i][0])))
        key_r, key_c = map(int, marks[key_idx])
        color = int(cell[key_r, key_c])

        mask = (g == color) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(cell_idx), int(color), int(count), int(key_r), int(key_c)))

    records.sort(key=lambda t: (-int(t[2]), int(t[0])))

    for rank_idx, (_orig_idx, color, count, key_r, key_c) in enumerate(records):
        if rank_idx >= len(cells):
            break
        cell_r0, cell_c0, _cell_r1, cell_c1 = cells[rank_idx]
        draw_r = cell_r0 + key_r
        for offset in range(int(count)):
            draw_c = cell_c0 + key_c + offset
            if draw_c <= cell_c1:
                out[draw_r, draw_c] = int(color)

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
        _labels, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
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
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, KEY_COUNT)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


def _cell_inner_width(max_count: int) -> int:
    return max(CELL_MIN_WIDTH, int(max_count)) + _randint(CELL_EXTRA_WIDTH_RANGE)


def _build_unsorted_expected(key_colors: list[int], counts: list[int], bg: int, cell_inner_width: int) -> np.ndarray:
    expected = _frame_patch(_legend_shape(cell_inner_width), bg)
    _decorate_vertical_dividers(expected, cell_inner_width)
    for (r0, c0, _r1, c1), color, count in zip(_cell_bounds_from_width(cell_inner_width), key_colors, counts):
        draw_r = r0 + CELL_CENTER_ROW
        for offset in range(int(count)):
            draw_c = c0 + offset
            if draw_c <= c1:
                expected[draw_r, draw_c] = int(color)
    return expected


def _build_expected(key_colors: list[int], counts: list[int], bg: int, cell_inner_width: int) -> np.ndarray:
    expected = _frame_patch(_legend_shape(cell_inner_width), bg)
    _decorate_vertical_dividers(expected, cell_inner_width)
    cell_bounds = _cell_bounds_from_width(cell_inner_width)

    for out_idx, (_orig_idx, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        r0, c0, _r1, c1 = cell_bounds[out_idx]
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
    unsorted_expected = _build_unsorted_expected(key_colors, counts, bg, cell_inner_width)
    if np.array_equal(expected, unsorted_expected):
        return None

    legend = _build_legend(key_colors, bg, cell_inner_width)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
    try:
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    solved = main(grid)
    if np.array_equal(solved, unsorted_expected):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
