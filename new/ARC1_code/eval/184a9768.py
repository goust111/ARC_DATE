import numpy as np
from scipy.ndimage import binary_fill_holes

from arc_utils import Color, bbox_coords, components_mask, find_components, get_bg


def _component_holes(g: np.ndarray, comp: dict) -> list[dict]:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    local_object = g[r0 : r1 + 1, c0 : c1 + 1] == color
    local_holes = binary_fill_holes(local_object) & ~local_object

    holes = []
    for local_coords in components_mask(local_holes, connectivity=4):
        coords = local_coords + np.array([r0, c0])
        holes.append(
            {
                "coords": coords,
                "cells": frozenset(map(tuple, coords.tolist())),
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return holes


def _placements_for(sprite: dict, hole: dict) -> list[frozenset[tuple[int, int]]]:
    rel = sprite["rel"]
    max_r = max(r for r, _c in rel)
    max_c = max(c for _r, c in rel)
    r0, c0, r1, c1 = hole["bbox"]
    hole_cells = hole["cells"]

    placements = []
    for anchor_r in range(r0 - max_r, r1 + 1):
        for anchor_c in range(c0 - max_c, c1 + 1):
            cells = frozenset((anchor_r + dr, anchor_c + dc) for dr, dc in rel)
            if cells <= hole_cells:
                placements.append(cells)
    return placements


def _unique_tiling(sprites: list[dict], holes: list[dict]) -> list[tuple[int, frozenset[tuple[int, int]]]]:
    all_hole_cells = frozenset(cell for hole in holes for cell in hole["cells"])
    assert sum(hole["area"] for hole in holes) == len(all_hole_cells)
    assert sum(sprite["area"] for sprite in sprites) == len(all_hole_cells)

    candidates = []
    for sprite_index, sprite in enumerate(sprites):
        for hole in holes:
            for cells in _placements_for(sprite, hole):
                candidates.append((sprite_index, cells))

    cell_to_candidates = {cell: [] for cell in all_hole_cells}
    for candidate_index, (_sprite_index, cells) in enumerate(candidates):
        for cell in cells:
            cell_to_candidates[cell].append(candidate_index)

    solution_keys: set[tuple[tuple[tuple[int, int], int], ...]] = set()
    solutions: list[list[tuple[int, frozenset[tuple[int, int]]]]] = []

    def record(assignments: list[tuple[int, frozenset[tuple[int, int]]]]) -> None:
        colored_cells = []
        for sprite_index, cells in assignments:
            color = sprites[sprite_index]["color"]
            colored_cells.extend((cell, color) for cell in cells)
        key = tuple(sorted(colored_cells))
        if key not in solution_keys:
            solution_keys.add(key)
            solutions.append(list(assignments))

    def search(
        unused_sprites: frozenset[int],
        remaining_cells: frozenset[tuple[int, int]],
        assignments: list[tuple[int, frozenset[tuple[int, int]]]],
    ) -> None:
        if len(solutions) > 1:
            return
        if not remaining_cells:
            if not unused_sprites:
                record(assignments)
            return

        best_options = None
        for cell in remaining_cells:
            options = [
                candidate_index
                for candidate_index in cell_to_candidates[cell]
                if candidates[candidate_index][0] in unused_sprites
                and candidates[candidate_index][1] <= remaining_cells
            ]
            if best_options is None or len(options) < len(best_options):
                best_options = options
            if best_options == []:
                return

        assert best_options is not None
        for candidate_index in best_options:
            sprite_index, cells = candidates[candidate_index]
            search(
                unused_sprites - {sprite_index},
                remaining_cells - cells,
                [*assignments, (sprite_index, cells)],
            )

    search(frozenset(range(len(sprites))), all_hole_cells, [])
    assert len(solutions) == 1
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every small colored sprite into the enclosed holes of the large objects and erase the grey markers.

    Input structure
    - The background is the dominant color.
    - Grey pixels are isolated marker noise.
    - Some non-grey foreground components are large monochromatic objects with enclosed background holes.
    - The remaining non-grey foreground components are source sprites outside those holes.

    Rule
    - Detect the enclosed background holes inside the large objects.
    - Translate each source sprite without rotation or recoloring.
    - Use every source sprite exactly once so the translated sprite masks exactly partition all hole cells.
    - Clear the original source sprite cells and all grey marker pixels; preserve the large objects.

    Key insight
    - The destination is not chosen by color or position: the set of sprite shapes has a unique exact tiling of the enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    grey_components = [comp for comp in components if int(comp["color"]) == Color.GREY]
    assert grey_components
    assert all(comp["area"] == 1 for comp in grey_components)

    containers = set()
    holes = []
    for index, comp in enumerate(components):
        if int(comp["color"]) == Color.GREY:
            continue
        comp_holes = _component_holes(g, comp)
        if comp_holes:
            containers.add(index)
            holes.extend(comp_holes)

    assert containers
    assert holes

    sprites = []
    for index, comp in enumerate(components):
        if index in containers or int(comp["color"]) == Color.GREY:
            continue
        coords = comp["coords"]
        r0, c0, _r1, _c1 = comp["bbox"]
        rel = coords - np.array([r0, c0])
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "rel": [tuple(map(int, rc)) for rc in rel.tolist()],
                "area": int(comp["area"]),
            }
        )

    assert sprites
    tiling = _unique_tiling(sprites, holes)

    out = g.copy()
    for comp in grey_components:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for sprite_index, cells in tiling:
        color = sprites[sprite_index]["color"]
        for r, c in cells:
            out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

from arc_utils import paste, random_free_location

# Canvas and colors.
BACKGROUND_COLOR, MARKER_COLOR, BLOCKED_COLOR = Color.BLACK, Color.GREY, Color.MAROON
GRID_ROW_RANGE, GRID_COL_RANGE = (20, 26), (20, 27)
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]

# Container and hole structure.
ONE_CONTAINER_PROB, ONE_CONTAINER_HOLE_COUNTS = 0.15, [4]
TWO_CONTAINER_HOLE_COUNT_CHOICES = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
CONTAINER_HEIGHT_RANGE, CONTAINER_WIDTH_RANGE = (6, 17), (8, 14)
TWO_COLUMN_MIN_HOLES = 3
TWO_COLUMN_LAYOUT_PROB = 0.7

# Hole dimension sampling.
ANTICHAIN_RETRIES, ANTICHAIN_HEIGHT_CHOICES, ANTICHAIN_START_WIDTH = 80, np.arange(1, 8), 12
HOLE_MAX_WIDTH, HOLE_MAX_AREA = 11, 18
SMALL_HOLE_DIMS, TALL_HOLE_DIMS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1)], [(11, 1), (12, 1), (13, 1)]
SMALL_HOLE_PROB, TALL_HOLE_MAX_COUNT, TALL_HOLE_PROB = 0.55, 5, 0.28

# Sprite partitioning and coloring.
MIN_SPRITES, MAX_SPRITES = 5, 10
MAX_EXTRA_PIECES_PER_HOLE, MAX_PIECES_PER_HOLE = 2, 3
MIN_SPRITE_COLORS, MAX_SPRITE_COLORS = 3, 5
HORIZONTAL_SPLIT_PROB = 0.65

# Placement and retry limits.
MARKER_COUNT_RANGE = (6, 12)
CONTAINER_PLACEMENT = {"border": 1, "padding": 1, "padding_connectivity": 8}
SPRITE_PLACEMENT = {"border": 0, "padding": 1, "padding_connectivity": 4}
MARKER_PLACEMENT = {"border": 0, "padding": 1, "padding_connectivity": 8}
GENERATION_RETRIES = 300


def _distribute_extra_gaps(n_gaps: int, extra_units: int) -> list[int]:
    gaps = [1] * n_gaps
    for _ in range(extra_units):
        gaps[int(np.random.randint(0, len(gaps)))] += 1
    return gaps

def _gen_antichain_dims(n: int) -> list[tuple[int, int]]:
    for _ in range(ANTICHAIN_RETRIES):
        heights = sorted(np.random.choice(ANTICHAIN_HEIGHT_CHOICES, size=n, replace=False).tolist())
        dims = []
        prev_w = ANTICHAIN_START_WIDTH
        ok = True
        for i, h in enumerate(heights):
            remaining = n - i - 1
            hi = min(HOLE_MAX_WIDTH, HOLE_MAX_AREA // h, prev_w - 1)
            lo = remaining + 1
            if lo > hi:
                ok = False
                break
            w = int(np.random.randint(lo, hi + 1))
            dims.append((int(h), w))
            prev_w = w
        if ok:
            if np.random.rand() < SMALL_HOLE_PROB:
                dims[int(np.random.randint(0, n))] = SMALL_HOLE_DIMS[int(np.random.randint(0, len(SMALL_HOLE_DIMS)))]
            if n <= TALL_HOLE_MAX_COUNT and np.random.rand() < TALL_HOLE_PROB:
                dims[int(np.random.randint(0, n))] = TALL_HOLE_DIMS[int(np.random.randint(0, len(TALL_HOLE_DIMS)))]
            if len(set(dims)) != len(dims):
                continue
            np.random.shuffle(dims)
            return dims
    return []

def _gen_piece_rects(h: int, w: int, n: int) -> list[tuple[int, int, int, int]] | None:
    if n == 1:
        return [(0, 0, h, w)]
    if w >= n and (w >= h or h < n or np.random.rand() < HORIZONTAL_SPLIT_PROB):
        cuts = sorted(np.random.choice(np.arange(1, w), size=n - 1, replace=False).tolist())
        edges = [0, *cuts, w]
        return [(0, edges[i], h, edges[i + 1] - edges[i]) for i in range(n)]
    if h >= n:
        cuts = sorted(np.random.choice(np.arange(1, h), size=n - 1, replace=False).tolist())
        edges = [0, *cuts, h]
        return [(edges[i], 0, edges[i + 1] - edges[i], w) for i in range(n)]
    return None

def _gen_piece_counts(n_holes: int, target_sprites: int, dims: list[tuple[int, int]]) -> list[int] | None:
    counts = [1] * n_holes
    extras = target_sprites - n_holes
    order = np.random.permutation(n_holes).tolist()
    while extras > 0:
        changed = False
        for i in order:
            h, w = dims[i]
            if counts[i] < MAX_PIECES_PER_HOLE and max(h, w) >= counts[i] + 1:
                counts[i] += 1
                extras -= 1
                changed = True
                if extras == 0:
                    return counts
        if not changed:
            return None
    return counts

def _gen_two_column_layout(hole_dims):
    indices = np.random.permutation(len(hole_dims)).tolist()
    for split in range(1, len(indices)):
        left = indices[:split]
        right = indices[split:]
        left_w = max(hole_dims[i][1] for i in left)
        right_w = max(hole_dims[i][1] for i in right)
        min_w = max(CONTAINER_WIDTH_RANGE[0], left_w + right_w + 3)
        if min_w > CONTAINER_WIDTH_RANGE[1] - 1:
            continue
        left_h = sum(hole_dims[i][0] for i in left) + len(left) + 1
        right_h = sum(hole_dims[i][0] for i in right) + len(right) + 1
        base_h = max(left_h, right_h)
        if base_h > CONTAINER_HEIGHT_RANGE[1] - 1:
            continue
        box_h = int(np.random.randint(max(CONTAINER_HEIGHT_RANGE[0], base_h), CONTAINER_HEIGHT_RANGE[1]))
        box_w = int(np.random.randint(min_w, CONTAINER_WIDTH_RANGE[1]))
        extra_x = box_w - (left_w + right_w + 3)
        x_gaps = _distribute_extra_gaps(3, extra_x)
        col_starts = [x_gaps[0], x_gaps[0] + left_w + x_gaps[1]]
        col_widths = [left_w, right_w]
        rects = [None] * len(hole_dims)
        for col, group in enumerate((left, right)):
            sum_h = sum(hole_dims[i][0] for i in group)
            extra_gap = box_h - sum_h - (len(group) + 1)
            gaps = _distribute_extra_gaps(len(group) + 1, extra_gap)
            r = gaps[0]
            for j, i in enumerate(group):
                h, w = hole_dims[i]
                c = col_starts[col] + int(np.random.randint(0, col_widths[col] - w + 1))
                rects[i] = (r, c, h, w)
                r += h + gaps[j + 1]
        return box_h, box_w, rects
    return None

def _gen_single_column_layout(hole_dims):
    sum_h = sum(h for h, _w in hole_dims)
    base_h = sum_h + len(hole_dims) + 1
    max_w = max(w for _h, w in hole_dims)
    min_w = max(CONTAINER_WIDTH_RANGE[0], max_w + 2)
    if base_h > CONTAINER_HEIGHT_RANGE[1] - 1 or min_w > CONTAINER_WIDTH_RANGE[1] - 1:
        return None
    box_h = int(np.random.randint(max(CONTAINER_HEIGHT_RANGE[0], base_h), CONTAINER_HEIGHT_RANGE[1]))
    box_w = int(np.random.randint(min_w, CONTAINER_WIDTH_RANGE[1]))
    extra_gap = box_h - sum_h - (len(hole_dims) + 1)
    gaps = _distribute_extra_gaps(len(hole_dims) + 1, extra_gap)

    rects = []
    r = gaps[0]
    for i, (h, w) in enumerate(hole_dims):
        c = int(np.random.randint(1, box_w - w))
        rects.append((r, c, h, w))
        r += h + gaps[i + 1]
    return box_h, box_w, rects

def _gen_container_layout(hole_dims: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    if len(hole_dims) >= TWO_COLUMN_MIN_HOLES and np.random.rand() < TWO_COLUMN_LAYOUT_PROB:
        layout = _gen_two_column_layout(hole_dims)
        if layout is not None:
            return layout
    return _gen_single_column_layout(hole_dims)

def _gen_choose_hole_counts(n_containers: int) -> list[int]:
    if n_containers == 1:
        return ONE_CONTAINER_HOLE_COUNTS.copy()
    return list(TWO_CONTAINER_HOLE_COUNT_CHOICES[int(np.random.randint(0, len(TWO_CONTAINER_HOLE_COUNT_CHOICES)))])

def _gen_scene_plan():
    rows = int(np.random.randint(*GRID_ROW_RANGE))
    cols = int(np.random.randint(*GRID_COL_RANGE))
    n_containers = 1 if np.random.rand() < ONE_CONTAINER_PROB else 2
    holes_per_container = _gen_choose_hole_counts(n_containers)
    n_holes = sum(holes_per_container)
    dims = _gen_antichain_dims(n_holes)
    if not dims:
        return None

    max_sprites = min(MAX_SPRITES, n_holes + sum(min(MAX_EXTRA_PIECES_PER_HOLE, max(h, w) - 1) for h, w in dims))
    min_sprites = max(MIN_SPRITES, n_holes)
    if min_sprites > max_sprites:
        return None
    target_sprites = int(np.random.randint(min_sprites, max_sprites + 1))
    piece_counts = _gen_piece_counts(n_holes, target_sprites, dims)
    if piece_counts is None:
        return None
    return rows, cols, n_containers, holes_per_container, dims, piece_counts

def _gen_colors(n_containers: int, n_holes: int) -> tuple[list[int], list[int]]:
    palette = list(COLOR_POOL)
    np.random.shuffle(palette)
    container_colors = palette[:n_containers]
    sprite_palette = palette[n_containers:] + container_colors
    n_sprite_colors = int(np.random.randint(MIN_SPRITE_COLORS, min(MAX_SPRITE_COLORS, n_holes, len(sprite_palette)) + 1))
    hole_colors = sprite_palette[:n_sprite_colors]
    while len(hole_colors) < n_holes:
        hole_colors.append(sprite_palette[int(np.random.randint(0, n_sprite_colors))])
    np.random.shuffle(hole_colors)
    return container_colors, hole_colors

def _gen_container_groups(holes_per_container, dims):
    groups = []
    start = 0
    for count in holes_per_container:
        group_dims = dims[start : start + count]
        layout = _gen_container_layout(group_dims)
        if layout is None:
            return None
        groups.append((group_dims, layout))
        start += count
    return groups

def _random_location(blocked, patch, placement):
    try:
        return random_free_location(blocked, patch, transparent=BACKGROUND_COLOR, **placement)
    except ValueError:
        return None

def _render_scene(rows, cols, groups, container_colors, hole_colors, piece_counts):
    inp = np.full((rows, cols), BACKGROUND_COLOR, dtype=int)
    out = np.full((rows, cols), BACKGROUND_COLOR, dtype=int)
    blocked = np.full((rows, cols), BACKGROUND_COLOR, dtype=int)
    sprite_specs = []
    hole_index = 0

    for container_index, (_dims, layout) in enumerate(groups):
        box_h, box_w, local_rects = layout
        solid_patch = np.full((box_h, box_w), container_colors[container_index], dtype=int)
        loc = _random_location(blocked, solid_patch, CONTAINER_PLACEMENT)
        if loc is None:
            return None
        cr, cc = loc
        paste(inp, solid_patch, cr, cc)
        paste(out, solid_patch, cr, cc)
        paste(blocked, np.full((box_h, box_w), BLOCKED_COLOR, dtype=int), cr, cc)

        for local_r, local_c, h, w in local_rects:
            color = hole_colors[hole_index]
            pieces = _gen_piece_rects(h, w, piece_counts[hole_index])
            if pieces is None:
                return None
            inp[cr + local_r : cr + local_r + h, cc + local_c : cc + local_c + w] = BACKGROUND_COLOR
            for pr, pc, ph, pw in pieces:
                out[
                    cr + local_r + pr : cr + local_r + pr + ph,
                    cc + local_c + pc : cc + local_c + pc + pw,
                ] = color
                sprite_specs.append((ph, pw, color))
            hole_index += 1
    np.random.shuffle(sprite_specs)
    for ph, pw, color in sprite_specs:
        patch = np.full((ph, pw), color, dtype=int)
        loc = _random_location(blocked, patch, SPRITE_PLACEMENT)
        if loc is None:
            return None
        sr, sc = loc
        paste(inp, patch, sr, sc)
        paste(blocked, patch, sr, sc)

    marker_patch = np.array([[MARKER_COLOR]], dtype=int)
    for _ in range(int(np.random.randint(*MARKER_COUNT_RANGE))):
        loc = _random_location(blocked, marker_patch, MARKER_PLACEMENT)
        if loc is None:
            return None
        gr, gc = loc
        inp[gr, gc] = MARKER_COLOR
        blocked[gr, gc] = MARKER_COLOR
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp

def _gen_attempt() -> np.ndarray | None:
    plan = _gen_scene_plan()
    if plan is None:
        return None
    rows, cols, n_containers, holes_per_container, dims, piece_counts = plan
    container_colors, hole_colors = _gen_colors(n_containers, sum(holes_per_container))
    groups = _gen_container_groups(holes_per_container, dims)
    if groups is None:
        return None
    return _render_scene(rows, cols, groups, container_colors, hole_colors, piece_counts)

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 184a9768 input")
