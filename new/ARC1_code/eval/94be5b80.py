from __future__ import annotations

import numpy as np

from arc_utils import bbox, find_components, get_bg, is_contiguous as _gen_is_contiguous, paste as _gen_paste, random_free_location as _gen_random_free_location
import random as _gen_random


def _find_color_bar_legend(g: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False):
        r0, c0, r1, c1 = comp["bbox"]
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if comp["area"] != h * w or h < 2 or w < 2:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if len(set(map(int, patch.ravel()))) < 2:
            continue

        colors = []
        for c in range(w):
            col = patch[:, c]
            if int(col[0]) == bg or not np.all(col == col[0]):
                break
            colors.append(int(col[0]))
        else:
            if len(set(colors)) == len(colors):
                candidates.append(((r0, c0, r1, c1), colors))

    assert len(candidates) == 1, "expected one solid multi-color vertical-bar legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the color-bar legend and complete the vertical stack by copying the shared sprite mask into every ordered color slot.

    Input structure
    The grid has a black background, one solid rectangular legend made of
    adjacent vertical color bars, and one or more placed instances of the same
    sprite mask in colors from that legend.

    Rule
    Read the legend colors from left to right.  These colors fill consecutive
    vertical sprite slots from top to bottom.  The existing sprite instances
    provide the common mask, left edge, and vertical anchor of the stack; missing
    legend colors are inserted into their corresponding slots.

    Key insight
    Existing sprites are already in the same rank positions named by the legend,
    so any one of them fixes the whole ordered stack geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    (lr0, lc0, lr1, lc1), order = _find_color_bar_legend(g, bg)
    assert set(map(int, g[g != bg])).issubset(set(order)), "legend must name all foreground colors"
    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    existing = []
    for rank, color in enumerate(order):
        mask = (g == color) & ~legend_mask
        if not mask.any():
            continue
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        existing.append((rank, b, mask[r0 : r1 + 1, c0 : c1 + 1]))

    assert existing, "expected at least one existing sprite instance"
    template = existing[0][2]
    sprite_h, sprite_w = template.shape
    stack_top = existing[0][1][0] - existing[0][0] * sprite_h
    stack_left = existing[0][1][1]

    for rank, (r0, c0, r1, c1), crop in existing:
        assert crop.shape == template.shape, "existing sprites must have equal bboxes"
        assert np.array_equal(crop, template), "existing sprites must share one mask"
        assert c0 == stack_left, "existing sprites must share one left edge"
        assert r0 - rank * sprite_h == stack_top, "existing sprites must align to legend rank"
        assert r1 == r0 + sprite_h - 1 and c1 == c0 + sprite_w - 1

    h, w = g.shape
    assert stack_top >= 0 and stack_left >= 0
    assert stack_top + len(order) * sprite_h <= h
    assert stack_left + sprite_w <= w

    out = g.copy()
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg
    out[stack_top : stack_top + len(order) * sprite_h, stack_left : stack_left + sprite_w] = bg
    for rank, color in enumerate(order):
        r0 = stack_top + rank * sprite_h
        slot = out[r0 : r0 + sprite_h, stack_left : stack_left + sprite_w]
        slot[template] = color

    return out

# --------------- generator (o5x) ---------------
# Palette and canvas bounds.
COLOR_VALUES = range(1, 10)
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 3, 5
GRID_HEIGHTS, GRID_WIDTHS = (17, 18), (13, 14, 15)
CANVAS_MARGIN, BACKGROUND = 1, 0
# Sprite mask shape and density.
SPRITE_HEIGHT_MIN, SPRITE_HEIGHT_MAX = 3, 5
SPRITE_WIDTH_MIN, SPRITE_WIDTH_MAX = 6, 8
SPRITE_DENSITY_MIN, SPRITE_DENSITY_MAX = 0.50, 0.60
SPRITE_CONNECTIVITY = 8
# Legend placement and retained stack slots.
LEGEND_HEIGHT = 3
KEEP_COUNT_MIN, KEEP_COUNT_MAX = 1, 2
LEGEND_BORDER = LEGEND_PADDING = 1
LEGEND_PADDING_CONNECTIVITY = 4
# Retry policy.
GENERATION_ATTEMPTS = 300

def _gen_symmetric_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_cells = int(np.ceil(SPRITE_DENSITY_MIN * area))
    max_cells = int(np.floor(SPRITE_DENSITY_MAX * area))
    max_pair = width // 2 if width % 2 else width // 2 - 1
    def unit_size(pair: int) -> int:
        return 1 if width % 2 and pair == max_pair else 2
    def add_path(units: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
        row, pair = start
        units.add((row, pair))
        while (row, pair) != end:
            erow, epair = end
            moves = []
            if row != erow:
                moves.append((row + (1 if erow > row else -1), pair))
            if pair != epair:
                moves.append((row, pair + (1 if epair > pair else -1)))
            if row != erow and pair != epair:
                moves.append((row + (1 if erow > row else -1), pair + (1 if epair > pair else -1)))
            row, pair = _gen_random.choice(moves)
            units.add((row, pair))
    def adjacent_to(units: set[tuple[int, int]], unit: tuple[int, int]) -> bool:
        row, pair = unit
        return any(abs(row - r2) <= 1 and abs(pair - p2) <= 1 for r2, p2 in units)
    top = (0, _gen_random.randint(0, max_pair))
    bottom = (height - 1, _gen_random.randint(0, max_pair))
    side = (_gen_random.randrange(height), 0)
    units: set[tuple[int, int]] = set()
    add_path(units, top, side)
    add_path(units, side, bottom)
    base_cells = sum(unit_size(pair) for _, pair in units)
    targets = [cells for cells in range(min_cells, max_cells + 1) if cells >= base_cells]
    if not targets:
        return None
    target = _gen_random.choice(targets)
    cell_count = base_cells
    while cell_count < target:
        remaining = target - cell_count
        candidates = []
        for row in range(height):
            for pair in range(max_pair + 1):
                unit = (row, pair)
                size = unit_size(pair)
                if unit not in units and size <= remaining and adjacent_to(units, unit):
                    candidates.append(unit)
        if not candidates:
            return None
        unit = _gen_random.choice(candidates)
        units.add(unit)
        cell_count += unit_size(unit[1])
    mask = np.zeros((height, width), dtype=bool)
    for row, pair in units:
        if width % 2 and pair == max_pair:
            mask[row, width // 2] = True
        else:
            mask[row, pair] = True
            mask[row, width - 1 - pair] = True
    if not _gen_is_contiguous(mask.astype(int), background=BACKGROUND, connectivity=SPRITE_CONNECTIVITY):
        return None
    return mask

def _gen_choose_dimensions(n_colors: int) -> tuple[int, int, int, int] | None:
    """Choose sprite and grid dimensions that leave the required stack margin."""
    sprite_h_options = [
        h for h in range(SPRITE_HEIGHT_MIN, SPRITE_HEIGHT_MAX + 1)
        if any(grid_h >= n_colors * h + 2 * CANVAS_MARGIN for grid_h in GRID_HEIGHTS)
    ]
    if not sprite_h_options:
        return None
    sprite_h = _gen_random.choice(sprite_h_options)
    sprite_w = _gen_random.randint(SPRITE_WIDTH_MIN, SPRITE_WIDTH_MAX)
    grid_h_options = [grid_h for grid_h in GRID_HEIGHTS if grid_h >= n_colors * sprite_h + 2 * CANVAS_MARGIN]
    grid_w_options = [grid_w for grid_w in GRID_WIDTHS if grid_w >= max(sprite_w, n_colors) + 2 * CANVAS_MARGIN]
    if not grid_h_options or not grid_w_options:
        return None
    grid_h = _gen_random.choice(grid_h_options)
    grid_w = _gen_random.choice(grid_w_options)
    return sprite_h, sprite_w, grid_h, grid_w

def _gen_render_stack(colors: list[int], sprite_mask: np.ndarray, stack_top: int, stack_left: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Render the solved grid with every legend-ordered sprite slot filled."""
    sprite_h, sprite_w = sprite_mask.shape
    out = np.zeros((grid_h, grid_w), dtype=int)
    for rank, color in enumerate(colors):
        patch = np.zeros((sprite_h, sprite_w), dtype=int)
        patch[sprite_mask] = color
        _gen_paste(out, patch, stack_top + rank * sprite_h, stack_left, transparent=BACKGROUND)
    return out

def _gen_select_kept_ranks(n_colors: int) -> set[int]:
    keep_count = _gen_random.randint(KEEP_COUNT_MIN, KEEP_COUNT_MAX)
    first_keep = _gen_random.randint(0, n_colors - keep_count)
    return set(range(first_keep, first_keep + keep_count))

def _gen_place_legend(occupied: np.ndarray, colors: list[int]) -> tuple[np.ndarray, int, int] | None:
    """Place the solid color-bar legend without touching kept sprite slots."""
    legend = np.tile(np.array(colors, dtype=int), (LEGEND_HEIGHT, 1))
    try:
        legend_r, legend_c = _gen_random_free_location(
            occupied, legend, transparent=BACKGROUND, border=LEGEND_BORDER,
            padding=LEGEND_PADDING, padding_connectivity=LEGEND_PADDING_CONNECTIVITY,
        )
    except ValueError:
        return None
    return legend, legend_r, legend_c

def _gen_attempt() -> np.ndarray | None:
    n_colors = _gen_random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = _gen_random.sample(COLOR_VALUES, n_colors)
    dimensions = _gen_choose_dimensions(n_colors)
    if dimensions is None:
        return None
    sprite_h, sprite_w, grid_h, grid_w = dimensions
    sprite_mask = _gen_symmetric_mask(sprite_h, sprite_w)
    if sprite_mask is None or sprite_mask.mean() < SPRITE_DENSITY_MIN or sprite_mask.mean() > SPRITE_DENSITY_MAX:
        return None
    stack_h = n_colors * sprite_h
    stack_top = _gen_random.randint(CANVAS_MARGIN, grid_h - stack_h - CANVAS_MARGIN)
    stack_left = _gen_random.randint(CANVAS_MARGIN, grid_w - sprite_w - CANVAS_MARGIN)
    out = _gen_render_stack(colors, sprite_mask, stack_top, stack_left, grid_h, grid_w)
    kept_ranks = _gen_select_kept_ranks(n_colors)
    occupied = np.zeros_like(out)
    for rank in kept_ranks:
        r0 = stack_top + rank * sprite_h
        occupied[r0:r0 + sprite_h, stack_left:stack_left + sprite_w] = 1
    legend_placement = _gen_place_legend(occupied, colors)
    if legend_placement is None:
        return None
    legend, legend_r, legend_c = legend_placement
    inp = out.copy()
    for rank in range(n_colors):
        if rank not in kept_ranks:
            r0 = stack_top + rank * sprite_h
            inp[r0:r0 + sprite_h, stack_left:stack_left + sprite_w] = BACKGROUND
    _gen_paste(inp, legend, legend_r, legend_c)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 94be5b80 input")
