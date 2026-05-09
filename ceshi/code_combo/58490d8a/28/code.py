# Combo variant of 58490d8a (index 28)
# combo_spec: D3/0+D5/2+D15/2
# Name: 网格图例+递增间距+交替圆点色

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

# Design analysis for the combo variant
# a) Concreteized behavior and trace:
#    The fused task keeps the original "count outside components by key color" core, but the destination now has three
#    linked changes at once. The legend is no longer one black rectangle with a single indexed key column; instead it is
#    a visible 2x2 grid whose black border and black internal dividers form one connected frame. Each cell has a
#    background-colored interior and exactly one key pixel at the left edge of its center row in the input. In the
#    output, that key grows into a local dot sequence inside its own cell. The dot positions are not equally spaced:
#    with local key column 0, counts 1/2/3/4 land at offsets [0], [0,2], [0,2,5], [0,2,5,9]. The dot colors alternate
#    key/black/key/black. Example: a cell keyed by red with count 4 becomes red at offset 0, black at 2, red at 5, and
#    black at 9. Relative to the original task, the solver must now decode a 2D legend layout first, then apply a
#    cumulative spacing rule, then apply a color alternation rule while still matching counts to outside objects.
# b) Complexity calibration:
#    A true table should be visually unmistakable, so the generator fixes the legend to a filled 2x2 layout with four
#    populated cells. This avoids degeneration back into a single column. The growing-gap rule only becomes visibly
#    different from the old fixed-gap rule at count >= 3, and the alternating color only becomes visible beyond the
#    first dot, so generation enforces at least one count >= 3 and at least two rows with count >= 2. To keep the
#    increasing-gap sequence physically compatible with a 2x2 legend on ARC-scale canvases, counts are capped at 4
#    instead of 5; otherwise the fifth local offset 14 would force a legend width that is too wide for a compact grid.
# c) Hidden assumptions from the original solver that break here:
#    - "leftmost non-black column = key column for the whole legend" is no longer enough, because most interior columns
#      are background-colored and keys are distributed across separate cells.
#    - "the output canvas can start as all black" is false, because the grid interior must stay background-colored for
#      the cell structure and black alternating dots to remain visible.
#    - "a constant +2 stride places dots" is false; the new placement uses cumulative offsets 0,2,5,9.
#    - "legend occupancy alone blocks sprite placement" is false once cell interiors are background-colored; the
#      generator must explicitly reserve the full legend bbox, not just the visible black frame, before placing sprites.
# d) Placement legality:
#    The generator remains output-first. It first builds the expected 2x2 legend crop, derives the corresponding input
#    legend with only one key pixel per cell, pastes that legend into a larger background canvas, and then places
#    sprites outside the entire legend bbox. A separate blocking canvas marks the whole bbox as occupied before calling
#    `random_free_location`, so no sprite can slip into a background-colored cell interior. The same 8-connected
#    padding of one cell is retained between sprites and between sprites and the legend.
# e) Visual observability:
#    Across a few examples, a human sees four colored keys arranged in quadrants, not a vertical list. Rows with
#    count 3 or 4 visibly separate the new spacing pattern from the old 0,2,4,6 pattern because the third dot shifts
#    farther right. The alternating black dots are easy to see against the background-colored cell interiors, so no
#    hidden numeric color ordering is needed to infer the second color.
# f) Visual cleanliness:
#    Black serves two roles only: frame/dividers and alternating dots. They never overlap because dots live on each
#    cell's center row away from separator lines. Cell interiors are plain background color, so the table stays legible
#    and the counted prefix is easy to compare across cells.
# g) Change inferability:
#    Color behavior is structurally signaled. Each cell's leftmost colored key identifies which outside sprite color to
#    count, and the only color change inside the output is the deterministic alternation between that key color and
#    black. Because the second color is position-driven rather than identity-driven, the solver does not need any
#    invisible palette ordering assumptions.
# h) New solver assumptions and generator guarantees:
#    The solver assumes the legend bbox contains full-black separator rows and columns, each cell has exactly one
#    non-black non-background key pixel in the input, and the cell is wide enough for the sampled count's cumulative
#    offsets. The generator enforces all three by constructing the frame directly, using one row-major key per cell,
#    and sizing every cell from the sampled maximum count.

BLACK = 0
COLOR_POOL = tuple(range(1, 10))
DISTRACTOR_PROB = 0.5

# Fixed 2x2 legend grid to keep the D3 change visually undeniable.
CELL_GRID_ROWS = 2
CELL_GRID_COLS = 2
KEY_COUNT = CELL_GRID_ROWS * CELL_GRID_COLS
CELL_INNER_HEIGHT = 3
CELL_CENTER_ROW = CELL_INNER_HEIGHT // 2
CELL_EXTRA_WIDTH_RANGE = (0, 1)

# Count range is tightened to 1..4 so the D5 growing-gap sequence fits inside a 2x2 grid legend.
TARGET_COUNT_RANGE = (1, 4)
MIN_VISIBLE_GROWTH_COUNT = 3
MIN_ROWS_WITH_ALTERNATION = 2

# Larger canvas than the seed task to leave room around a wider grid legend.
GRID_HEIGHT_RANGE = (22, 28)
GRID_WIDTH_RANGE = (27, 33)
LEGEND_EDGE_MARGIN = 1

SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 400


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _tally_offset(index: int) -> int:
    """Local column offset for the zero-based tally dot under increasing gaps."""
    idx = int(index)
    return idx * (idx + 3) // 2


def _tally_col(key_col: int, index: int) -> int:
    return int(key_col) + _tally_offset(index)


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
    Count colored objects and rewrite their totals inside a gridded legend.

    Input structure:
    - A connected black legend frame is divided by black separator lines into multiple rectangular cells.
    - Each cell has a background-colored interior and exactly one key-colored pixel near the left side of its center row.
    - Outside the legend bbox, disconnected monochrome sprites of those key colors are scattered on the canvas.

    Rule:
    - For each populated cell, count the 8-connected components of that cell's key color outside the legend bbox.
    - Output the cropped legend grid itself: keep the black frame and background interiors, then place one local tally
      per counted object on the cell's key row using increasing gaps, with colors alternating key, black, key, black.

    Key insight:
    - The legend is now a 2D table rather than one indexed column, so the solver must first recover cell boundaries
      from the black separators. Once a key color is tied to a cell, the count is still taken outside the legend, but
      the answer is encoded by a local cumulative-spacing sequence with alternating colors.
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
            draw_c = cell_c0 + _tally_col(key_c, j)
            if draw_c <= cell_c1:
                out[draw_r, draw_c] = int(color) if j % 2 == 0 else BLACK

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
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(KEY_COUNT)]
        if max(counts) < MIN_VISIBLE_GROWTH_COUNT:
            continue
        if sum(count >= 2 for count in counts) < MIN_ROWS_WITH_ALTERNATION:
            continue
        return counts


def _cell_inner_width(max_count: int) -> int:
    last_dot = _tally_col(0, max_count - 1)
    return int(last_dot + 1 + _randint(CELL_EXTRA_WIDTH_RANGE))


def _build_expected(key_colors: list[int], counts: list[int], bg: int, cell_inner_width: int) -> np.ndarray:
    expected = _frame_patch(_legend_shape(cell_inner_width), bg)
    _decorate_vertical_dividers(expected, cell_inner_width)

    for (r0, c0, _r1, c1), color, count in zip(_cell_bounds_from_width(cell_inner_width), key_colors, counts):
        draw_r = r0 + CELL_CENTER_ROW
        for j in range(int(count)):
            draw_c = c0 + _tally_offset(j)
            if draw_c <= c1:
                expected[draw_r, draw_c] = int(color) if j % 2 == 0 else BLACK
    return expected


def _build_legend(key_colors: list[int], bg: int, cell_inner_width: int) -> np.ndarray:
    legend = _frame_patch(_legend_shape(cell_inner_width), bg)
    _decorate_vertical_dividers(legend, cell_inner_width)

    for (r0, c0, _r1, _c1), color in zip(_cell_bounds_from_width(cell_inner_width), key_colors):
        legend[r0 + CELL_CENTER_ROW, c0] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
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
    return grid, (int(r0), int(c0), int(r0 + bh - 1), int(c0 + bw - 1))


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


def _place_sprites(
    grid: np.ndarray,
    sprites: list[np.ndarray],
    bg: int,
    legend_bbox: tuple[int, int, int, int],
) -> None:
    blocked = np.array(grid, copy=True)
    r0, c0, r1, c1 = legend_bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    for sprite in sprites:
        sr, sc = random_free_location(
            blocked,
            sprite,
            transparent=bg,
            padding=SPRITE_PADDING,
            padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
        )
        paste(grid, sprite, sr, sc, transparent=bg)
        paste(blocked, sprite, sr, sc, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts()
    cell_inner_width = _cell_inner_width(max(counts))

    expected = _build_expected(key_colors, counts, bg, cell_inner_width)
    legend = _build_legend(key_colors, bg, cell_inner_width)
    placed = _new_grid_with_legend(bg, legend)
    if placed is None:
        return None
    grid, legend_bbox = placed

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
    try:
        _place_sprites(grid, sprites, bg, legend_bbox)
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
