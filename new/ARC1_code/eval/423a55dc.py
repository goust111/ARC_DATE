from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, components_mask, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shear the single foreground object left by one cell for each row above its bottom edge.

    Input structure:
    - A dominant background surrounds exactly one 4-connected foreground object.
    - The foreground object uses one non-background color.

    Rule:
    - Clear the canvas to the background color.
    - Let bottom be the object's bottommost row.
    - For each foreground pixel at (r, c), paint it at (r, c - (bottom - r)) if
      that column remains in bounds.

    Key insight:
    - The bottom row is the anchor; each row above it slides left by its vertical
      distance from that anchor, producing a left-leaning shear with left-edge
      clipping.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected one foreground color"
    color = fg_colors[0]
    assert int(np.sum(g == bg)) > int(np.sum(g == color)), "background must be the unique dominant color"
    mask = g == color

    components = components_mask(mask, connectivity=4)
    assert len(components) == 1, "expected one 4-connected foreground object"

    obj_bbox = bbox(mask)
    assert obj_bbox is not None, "expected a non-empty foreground object"
    _, _, bottom, _ = obj_bbox

    out = np.full_like(g, bg)
    width = g.shape[1]
    for r, c in components[0]:
        r = int(r)
        c = int(c)
        assert 0 <= r < g.shape[0] and 0 <= c < width
        new_c = c - (int(bottom) - r)
        if 0 <= new_c < width:
            out[r, new_c] = color

    return out

# --------------- generator (o5x) ---------------

from arc_utils import paste as _gen_paste


# Canvas, object, and color bounds.
GRID_HEIGHT_BOUNDS, GRID_WIDTH_BOUNDS = (7, 11), (7, 13)
OBJECT_HEIGHT_BOUNDS, OBJECT_WIDTH_BOUNDS = (2, 7), (3, 6)
EDGE_MARGIN, BACKGROUND_COLOR = 1, 0
COLOR_BOUNDS = (1, 9)

# Foreground fill constraints.
OBJECT_PIXEL_BOUNDS, MIN_BBOX_DENSITY = (6, 22), 0.27
MAX_OBJECT_CANVAS_FRACTION = 0.5

# Clipping preference and retry limits.
MAX_LOST_PIXELS, CLIPPED_LEFT_PROBABILITY = 3, 0.35
SPRITE_RETRY_LIMIT, GENERATOR_RETRY_LIMIT = 20, 300

# Initial connected-line templates.
MODE_FREE_CROSS, MODE_EDGE_L, MODE_TOP_BOTTOM, MODE_LEFT_RIGHT, MODE_FRAME = range(5)
BASE_LINE_MODES = [MODE_FREE_CROSS, MODE_EDGE_L]


def _gen_line_mask(h: int, w: int, mode: int) -> np.ndarray:
    """Return the seed mask for one connected-line template."""
    mask = np.zeros((h, w), dtype=bool)
    if mode == MODE_FREE_CROSS:
        rr = np.random.randint(0, h)
        cc = np.random.randint(0, w)
        mask[rr, :] = True
        mask[:, cc] = True
    elif mode == MODE_EDGE_L:
        rr = np.random.choice([0, h - 1])
        cc = np.random.choice([0, w - 1])
        mask[rr, :] = True
        mask[:, cc] = True
    elif mode == MODE_TOP_BOTTOM:
        cc = np.random.randint(0, w)
        mask[0, :] = True
        mask[h - 1, :] = True
        mask[:, cc] = True
    elif mode == MODE_LEFT_RIGHT:
        rr = np.random.randint(0, h)
        mask[:, 0] = True
        mask[:, w - 1] = True
        mask[rr, :] = True
    else:
        mask[0, :] = True
        mask[h - 1, :] = True
        mask[:, 0] = True
        mask[:, w - 1] = True
    return mask


def _gen_frontier(mask: np.ndarray) -> np.ndarray:
    """Return empty cells 4-adjacent to the current connected mask."""
    frontier = np.zeros_like(mask, dtype=bool)
    frontier[1:, :] |= mask[:-1, :]
    frontier[:-1, :] |= mask[1:, :]
    frontier[:, 1:] |= mask[:, :-1]
    frontier[:, :-1] |= mask[:, 1:]
    frontier &= ~mask
    return frontier


def _gen_grow_connected(mask: np.ndarray, target_count: int) -> np.ndarray | None:
    """Grow mask by random frontier cells until it reaches target_count."""
    while int(mask.sum()) < target_count:
        frontier = np.argwhere(_gen_frontier(mask))
        if len(frontier) == 0:
            return None
        r, c = frontier[np.random.randint(0, len(frontier))]
        mask[r, c] = True
    return mask


def _gen_make_sprite(h: int, w: int, color: int) -> np.ndarray | None:
    """Build one connected colored sprite within bbox fill constraints."""
    area = h * w
    min_count = max(OBJECT_PIXEL_BOUNDS[0], int(np.ceil(MIN_BBOX_DENSITY * area)), h + w - 1)
    max_count = min(OBJECT_PIXEL_BOUNDS[1], area)
    if min_count > max_count:
        return None

    for _ in range(SPRITE_RETRY_LIMIT):
        target_count = np.random.randint(min_count, max_count + 1)
        possible_modes = list(BASE_LINE_MODES)
        if 2 * w + h - 2 <= target_count:
            possible_modes.append(MODE_TOP_BOTTOM)
        if 2 * h + w - 2 <= target_count:
            possible_modes.append(MODE_LEFT_RIGHT)
        if 2 * h + 2 * w - 4 <= target_count:
            possible_modes.append(MODE_FRAME)
        mode = possible_modes[np.random.randint(0, len(possible_modes))]
        mask = _gen_line_mask(h, w, mode)
        if int(mask.sum()) > target_count:
            continue
        mask = _gen_grow_connected(mask, target_count)
        if mask is not None:
            sprite = np.zeros((h, w), dtype=int)
            sprite[mask] = color
            return sprite
    return None


def _gen_shear(grid: np.ndarray, color: int) -> tuple[np.ndarray, int]:
    """Apply the task shear and return the output plus clipped pixel count."""
    coords = np.argwhere(grid == color)
    bottom = int(coords[:, 0].max())
    out = np.zeros_like(grid)
    shifted_cols = coords[:, 1] - (bottom - coords[:, 0])
    valid = (0 <= shifted_cols) & (shifted_cols < grid.shape[1])
    out[coords[valid, 0], shifted_cols[valid]] = color
    lost = int(np.count_nonzero(~valid))
    return out, lost


def _gen_sample_dimensions() -> tuple[int, int, int, int]:
    """Sample canvas and sprite dimensions that leave the required margin."""
    H = np.random.randint(GRID_HEIGHT_BOUNDS[0], GRID_HEIGHT_BOUNDS[1] + 1)
    W = np.random.randint(GRID_WIDTH_BOUNDS[0], GRID_WIDTH_BOUNDS[1] + 1)
    h = np.random.randint(OBJECT_HEIGHT_BOUNDS[0], min(OBJECT_HEIGHT_BOUNDS[1], H - 2 * EDGE_MARGIN) + 1)
    w = np.random.randint(OBJECT_WIDTH_BOUNDS[0], min(OBJECT_WIDTH_BOUNDS[1], W - 2 * EDGE_MARGIN) + 1)
    return H, W, h, w


def _gen_left_offsets(H: int, W: int, r0: int, sprite: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    """Partition legal left offsets by whether the shear clips pixels."""
    w = sprite.shape[1]
    valid_lefts = []
    clipped_lefts = []
    for c0 in range(EDGE_MARGIN, W - w - EDGE_MARGIN + 1):
        grid = np.zeros((H, W), dtype=int)
        _gen_paste(grid, sprite, r0, c0, transparent=BACKGROUND_COLOR)
        _, lost = _gen_shear(grid, color)
        if lost <= MAX_LOST_PIXELS:
            valid_lefts.append(c0)
            if lost > 0:
                clipped_lefts.append(c0)
    return valid_lefts, clipped_lefts


def _gen_choose_left(valid_lefts: list[int], clipped_lefts: list[int]) -> int:
    """Choose a valid left offset with the configured clipping bias."""
    if clipped_lefts and np.random.random() < CLIPPED_LEFT_PROBABILITY:
        return clipped_lefts[np.random.randint(0, len(clipped_lefts))]
    return valid_lefts[np.random.randint(0, len(valid_lefts))]


def _gen_attempt() -> np.ndarray | None:
    H, W, h, w = _gen_sample_dimensions()
    color = np.random.randint(COLOR_BOUNDS[0], COLOR_BOUNDS[1] + 1)

    sprite = _gen_make_sprite(h, w, color)
    if sprite is None:
        return None
    if int(np.count_nonzero(sprite)) >= H * W * MAX_OBJECT_CANVAS_FRACTION:
        return None

    r0 = np.random.randint(EDGE_MARGIN, H - h - EDGE_MARGIN + 1)
    valid_lefts, clipped_lefts = _gen_left_offsets(H, W, r0, sprite, color)
    if not valid_lefts:
        return None

    c0 = _gen_choose_left(valid_lefts, clipped_lefts)
    inp = np.zeros((H, W), dtype=int)
    _gen_paste(inp, sprite, r0, c0, transparent=BACKGROUND_COLOR)
    out, lost = _gen_shear(inp, color)
    if not (0 <= lost <= MAX_LOST_PIXELS):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
