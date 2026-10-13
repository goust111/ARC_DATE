import numpy as np

from arc_utils import bbox_w, colors_present, crop_bbox, find_components, get_bg, is_contiguous, paste
import itertools
import random


def _cells(component):
    return set(map(tuple, component["coords"]))


def _is_top_bracket(component) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    width = c1 - c0 + 1
    if r0 != 0 or r1 - r0 + 1 != 2 or width < 3:
        return False
    expected = {(r0, c) for c in range(c0, c1 + 1)}
    expected |= {(r1, c0), (r1, c1)}
    return _cells(component) == expected


def _is_bottom_bracket(component, height: int) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    width = c1 - c0 + 1
    if r1 != height - 1 or r1 - r0 + 1 != 2 or width < 3:
        return False
    expected = {(r1, c) for c in range(c0, c1 + 1)}
    expected |= {(r0, c0), (r0, c1)}
    return _cells(component) == expected


def main(grid: np.ndarray) -> np.ndarray:
    """Move the colored object from the lower bracket lane to the upper bracket lane and drop it to the bottom edge.

    Input structure
    - The background is the dominant color.
    - One frame color forms exactly two 4-connected bracket objects: a top-edge bracket open downward and a bottom-edge bracket open upward.
    - One other 4-connected foreground object sits in the lower bracket's interior column span.

    Rule
    - Remove the movable object from its original cells.
    - Keep its shape and color unchanged.
    - Paste it with its bbox columns aligned to the top bracket's interior columns and with its bbox bottom aligned to the grid bottom.

    Key insight
    - The lower bracket identifies the object's source lane; the upper bracket provides the target lane, and the final vertical placement is bottom-aligned.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = get_bg(g)

    components_by_color = {}
    bracket_candidates = []
    for color in colors_present(g, exclude={bg}):
        components = find_components(
            g,
            bg=bg,
            colors=[color],
            connectivity=4,
            monochromatic=True,
        )
        components_by_color[color] = components
        top = [component for component in components if _is_top_bracket(component)]
        bottom = [
            component for component in components
            if _is_bottom_bracket(component, height)
        ]
        if len(components) == 2 and len(top) == 1 and len(bottom) == 1:
            bracket_candidates.append((color, top[0], bottom[0]))

    assert len(bracket_candidates) == 1
    frame_color, top_bracket, bottom_bracket = bracket_candidates[0]

    movers = [
        component
        for color, components in components_by_color.items()
        if color != frame_color
        for component in components
    ]
    assert len(movers) == 1
    mover = movers[0]

    _, mc0, _, mc1 = mover["bbox"]
    _br0, bc0, _br1, bc1 = bottom_bracket["bbox"]
    _tr0, tc0, _tr1, tc1 = top_bracket["bbox"]
    assert (mc0, mc1) == (bc0 + 1, bc1 - 1)
    assert bbox_w(mover["bbox"]) == tc1 - tc0 - 1

    patch = crop_bbox(g, mover["bbox"])
    dest_r = height - patch.shape[0]
    dest_c = tc0 + 1
    assert dest_r >= 0
    assert dest_c >= 0 and dest_c + patch.shape[1] <= width

    out = g.copy()
    mover_coords = mover["coords"]
    out[mover_coords[:, 0], mover_coords[:, 1]] = bg

    colored_patch = patch != bg
    target = out[dest_r:dest_r + patch.shape[0], dest_c:dest_c + patch.shape[1]]
    assert np.all(target[colored_patch] == bg)

    paste(out, patch, dest_r, dest_c, transparent=bg)
    return out

# --------------- generator (o5x) ---------------

# Fixed task colors.
BG_COLOR, BRACKET_COLOR = 7, 4
MOVER_COLORS = [0, 1, 2, 3, 5, 6, 8, 9]
# Canvas, bracket, and mover-size bounds.
GRID_SIZE_RANGE = (8, 9)
BRACKET_WIDTH_RANGE = (3, 4)
MOVER_HEIGHT_RANGE = (1, 3)
MOVER_MAX_CELLS = 4
MOVER_MIN_DENSITY_NUM = 2
MOVER_MIN_DENSITY_DEN = 3
# Layout choices and retry budget.
TOP_LEFT_OPTIONS = [True, False]
MAX_GENERATION_ATTEMPTS = 300

def _gen_half_starts(grid_width, bracket_width, left_side):
    if left_side:
        return list(range(0, grid_width // 2 - bracket_width + 1))
    return list(range((grid_width + 1) // 2, grid_width - bracket_width + 1))

def _gen_bracket_columns(grid_width, bracket_width):
    """Choose non-overlapping top/bottom bracket starts on opposite halves."""
    top_left = random.choice(TOP_LEFT_OPTIONS)
    top_starts = _gen_half_starts(grid_width, bracket_width, top_left)
    bottom_starts = _gen_half_starts(grid_width, bracket_width, not top_left)
    if not top_starts or not bottom_starts:
        return None
    return random.choice(top_starts), random.choice(bottom_starts)

def _gen_add_bracket(grid, c0, width, top_edge):
    edge_row, arm_row = (0, 1) if top_edge else (-1, -2)
    grid[edge_row, c0:c0 + width] = BRACKET_COLOR
    grid[arm_row, c0] = BRACKET_COLOR
    grid[arm_row, c0 + width - 1] = BRACKET_COLOR

def _gen_sprite_masks(height, width):
    area = height * width
    min_cells = (MOVER_MIN_DENSITY_NUM * area + MOVER_MIN_DENSITY_DEN - 1) // MOVER_MIN_DENSITY_DEN
    max_cells = min(MOVER_MAX_CELLS, area)
    masks = []
    for cell_count in range(min_cells, max_cells + 1):
        for cells in itertools.combinations(range(area), cell_count):
            mask = np.zeros((height, width), dtype=bool)
            mask[np.unravel_index(cells, (height, width))] = True
            if not mask.any(axis=0).all() or not mask.any(axis=1).all():
                continue
            if is_contiguous(mask.astype(int), background=0, connectivity=4):
                masks.append(mask)
    return masks

def _gen_mover_sprite(height, width, color, bg):
    masks = _gen_sprite_masks(height, width)
    if not masks:
        return None
    mask = random.choice(masks)
    sprite = np.full((height, width), bg, dtype=int)
    sprite[mask] = color
    return sprite

def _gen_paste_if_clear(grid, sprite, row, col, sprite_mask):
    """Paste sprite only when every colored sprite cell lands on background."""
    target = grid[row:row + sprite.shape[0], col:col + sprite.shape[1]]
    if not np.all(target[sprite_mask] == BG_COLOR):
        return False
    paste(grid, sprite, row, col, transparent=BG_COLOR)
    return True

def _gen_attempt():
    height = random.randint(*GRID_SIZE_RANGE)
    grid_width = random.randint(*GRID_SIZE_RANGE)
    bracket_width = random.randint(*BRACKET_WIDTH_RANGE)
    interior_width = bracket_width - 2

    bracket_columns = _gen_bracket_columns(grid_width, bracket_width)
    if bracket_columns is None:
        return None
    top_c0, bottom_c0 = bracket_columns

    mover_height = random.randint(*MOVER_HEIGHT_RANGE)
    mover_color = random.choice(MOVER_COLORS)
    sprite = _gen_mover_sprite(mover_height, interior_width, mover_color, BG_COLOR)
    if sprite is None:
        return None

    out = np.full((height, grid_width), BG_COLOR, dtype=int)
    _gen_add_bracket(out, top_c0, bracket_width, top_edge=True)
    _gen_add_bracket(out, bottom_c0, bracket_width, top_edge=False)

    dest_r = height - mover_height
    dest_c = top_c0 + 1
    sprite_mask = sprite != BG_COLOR
    if not _gen_paste_if_clear(out, sprite, dest_r, dest_c, sprite_mask):
        return None

    inp = out.copy()
    target = inp[dest_r:dest_r + mover_height, dest_c:dest_c + interior_width]
    target[sprite_mask] = BG_COLOR

    source_c = bottom_c0 + 1
    max_source_r = height - 2 - mover_height
    if max_source_r < 1:
        return None
    source_r = random.randint(1, max_source_r)
    if not _gen_paste_if_clear(inp, sprite, source_r, source_c, sprite_mask):
        return None

    if np.array_equal(main(inp), out):
        return inp
    return None

def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate df9fd884 input")
