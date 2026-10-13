from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, paste
import random
from itertools import combinations


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    return tuple(map(tuple, (coords - [r0, c0]).tolist()))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each repeated source sprite from the same-position pixel in the compact colored legend.

    Input structure
    - A background grid contains one repeated 8-connected sprite color and a compact legend made from other foreground colors.
    - The legend pixels have the same row/column occupancy pattern as the layout of the repeated sprites.

    Rule
    - Rank sprite bounding boxes by their top row and left column, and rank legend pixels by their row and column in the legend.
    - Recolor every pixel of each sprite with the legend color at the matching rank position; leave the legend and background unchanged.

    Key insight
    - The legend is a miniature color map for the arrangement of the larger repeated sprites, not a sprite to copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    components = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for component in components:
        by_color[int(component["color"])].append(component)

    source_colors = []
    for color, color_components in by_color.items():
        shapes = {_normalized_shape(component["coords"]) for component in color_components}
        if (
            len(color_components) >= 2
            and len(shapes) == 1
            and all(int(component["area"]) > 1 for component in color_components)
        ):
            source_colors.append(color)
    assert len(source_colors) == 1
    source_color = source_colors[0]
    sprites = by_color[source_color]

    legend_coords = np.argwhere((g != bg) & (g != source_color))
    assert len(legend_coords) == len(sprites)

    sprite_rows = sorted({int(sprite["bbox"][0]) for sprite in sprites})
    sprite_cols = sorted({int(sprite["bbox"][1]) for sprite in sprites})
    legend_rows = sorted({int(r) for r, _c in legend_coords})
    legend_cols = sorted({int(c) for _r, c in legend_coords})
    assert len(sprite_rows) == len(legend_rows)
    assert len(sprite_cols) == len(legend_cols)

    legend_by_layout: dict[tuple[int, int], int] = {}
    for r, c in legend_coords:
        key = (legend_rows.index(int(r)), legend_cols.index(int(c)))
        assert key not in legend_by_layout
        legend_by_layout[key] = int(g[r, c])

    out = g.copy()
    sprite_keys = set()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        key = (sprite_rows.index(int(r0)), sprite_cols.index(int(c0)))
        assert key in legend_by_layout
        assert key not in sprite_keys
        sprite_keys.add(key)

        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = legend_by_layout[key]

    assert sprite_keys == set(legend_by_layout)
    return out

# --------------- generator (o5x) ---------------

# Canvas/colors.
BG_COLOR, SOURCE_COLOR = 0, 1
LEGEND_COLORS = range(2, 10)
GRID_HEIGHTS, GRID_WIDTHS = range(10, 19), range(12, 19)

# Sprite/legend/layout.
SPRITE_SIDES = (2, 3)
SHAPE_AREA_BY_SIDE = {2: 3, 3: 6}
MIN_LEGEND_COLORS, MAX_LEGEND_COLORS = 2, 4
SPRITE_GAP = 1

# Validation.
CONNECTIVITY, GEN_ATTEMPTS = 8, 300

def _gen_valid_shapes(n):
    area = SHAPE_AREA_BY_SIDE[n]
    cells = [(r, c) for r in range(n) for c in range(n)]
    shapes = []
    for picked in combinations(cells, area):
        mask = np.zeros((n, n), dtype=int)
        for r, c in picked:
            mask[r, c] = 1
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if is_contiguous(mask, background=BG_COLOR, connectivity=CONNECTIVITY):
            shapes.append(tuple(sorted(picked)))
    return shapes


VALID_SHAPES_BY_SIDE = {n: _gen_valid_shapes(n) for n in SPRITE_SIDES}


def _gen_connected_subset(cells, k):
    cells = list(cells)
    if k >= len(cells):
        return set(cells)

    chosen = {random.choice(cells)}
    while len(chosen) < k:
        options = [
            (r, c) for r, c in cells
            if (r, c) not in chosen and any(max(abs(r - rr), abs(c - cc)) == 1 for rr, cc in chosen)
        ]
        chosen.add(random.choice(options))
    return chosen


def _gen_legend_assignment(shape_cells):
    shape_cells = list(shape_cells)
    distinct_count = random.randint(MIN_LEGEND_COLORS, min(MAX_LEGEND_COLORS, len(shape_cells)))
    palette = random.sample(LEGEND_COLORS, distinct_count)

    repeat_count = len(shape_cells) - distinct_count + 1
    repeat_cells = _gen_connected_subset(shape_cells, repeat_count)
    assignment = {cell: palette[0] for cell in repeat_cells}

    singles = [cell for cell in shape_cells if cell not in repeat_cells]
    random.shuffle(singles)
    for cell, color in zip(singles, palette[1:]):
        assignment[cell] = color
    return assignment


def _gen_split_positions(total, size_a, size_b):
    free = total - size_a - size_b
    leading = random.randint(0, free)
    gap = random.randint(0, free - leading)
    return leading, leading + size_a + gap


def _gen_place_rects(h, w, cluster_size, legend_size):
    modes = []
    if h >= cluster_size + legend_size:
        modes.append("vertical")
    if w >= cluster_size + legend_size:
        modes.append("horizontal")

    mode = random.choice(modes)
    cluster_first = random.choice([False, True])
    first_size = cluster_size if cluster_first else legend_size
    second_size = legend_size if cluster_first else cluster_size

    if mode == "vertical":
        first_major, second_major = _gen_split_positions(h, first_size, second_size)
        first_minor = random.randint(0, w - first_size)
        second_minor = random.randint(0, w - second_size)
        first_pos = (first_major, first_minor)
        second_pos = (second_major, second_minor)
    else:
        first_major, second_major = _gen_split_positions(w, first_size, second_size)
        first_minor = random.randint(0, h - first_size)
        second_minor = random.randint(0, h - second_size)
        first_pos = (first_minor, first_major)
        second_pos = (second_minor, second_major)

    cluster_pos, legend_pos = (first_pos, second_pos) if cluster_first else (second_pos, first_pos)
    return *cluster_pos, *legend_pos


def _gen_legal_dims(cluster_size, legend_size):
    separated_extent = cluster_size + legend_size
    return [(h, w) for h in GRID_HEIGHTS for w in GRID_WIDTHS
            if h >= cluster_size and w >= cluster_size
            and (h >= separated_extent or w >= separated_extent)]


def _gen_sprite_and_legend(n, shape_cells, colors_by_cell):
    sprite_mask = np.zeros((n, n), dtype=int)
    legend = np.zeros((n, n), dtype=int)
    rows, cols = zip(*shape_cells)
    sprite_mask[rows, cols] = 1
    legend[rows, cols] = [colors_by_cell[cell] for cell in shape_cells]
    return sprite_mask, legend


def _gen_rendered_pair(h, w, n, shape_cells, colors_by_cell, cluster_pos, legend_pos):
    sprite_mask, legend = _gen_sprite_and_legend(n, shape_cells, colors_by_cell)
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    source_sprite = np.where(sprite_mask, SOURCE_COLOR, BG_COLOR)
    step = n + SPRITE_GAP

    for layout_r, layout_c in shape_cells:
        sprite_r = cluster_pos[0] + layout_r * step
        sprite_c = cluster_pos[1] + layout_c * step
        color = colors_by_cell[(layout_r, layout_c)]
        paste(inp, source_sprite, sprite_r, sprite_c, transparent=BG_COLOR)
        paste(out, np.where(sprite_mask, color, BG_COLOR), sprite_r, sprite_c, transparent=BG_COLOR)

    paste(inp, legend, *legend_pos, transparent=BG_COLOR)
    paste(out, legend, *legend_pos, transparent=BG_COLOR)
    return inp, out


def _gen_attempt():
    n = random.choice(SPRITE_SIDES)
    shape_cells = random.choice(VALID_SHAPES_BY_SIDE[n])
    colors_by_cell = _gen_legend_assignment(shape_cells)

    cluster_size = n * n + (n - 1) * SPRITE_GAP
    h, w = random.choice(_gen_legal_dims(cluster_size, n))
    cluster_r, cluster_c, legend_r, legend_c = _gen_place_rects(h, w, cluster_size, n)
    inp, out = _gen_rendered_pair(
        h, w, n, shape_cells, colors_by_cell, (cluster_r, cluster_c), (legend_r, legend_c)
    )

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fafd9572 input")
