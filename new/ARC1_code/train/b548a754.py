import numpy as np

from arc_utils import bbox, colors_present, components_mask, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Stretch the colored rectangular frame outward until its far edge reaches the lone marker pixel.

    Input structure
    - The background is the most common color.
    - A single marker pixel is the only foreground color that appears once.
    - All other foreground pixels form one solid rectangular frame with one border color and one interior color.

    Rule
    - If the marker is above or below the frame and its column lies within the frame, extend the frame vertically to the marker row.
    - If the marker is left or right of the frame and its row lies within the frame, extend the frame horizontally to the marker column.
    - Redraw the expanded rectangle with the same border and interior colors; everything outside remains background.

    Key insight
    The marker selects which side of the rectangle moves and where the new outer border lies; it is consumed by that border.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    colors = colors_present(g, exclude={bg})
    singleton_colors = [color for color in colors if int((g == color).sum()) == 1]
    assert len(singleton_colors) == 1
    marker_color = singleton_colors[0]
    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape == (1, 2)
    marker_r, marker_c = map(int, marker_coords[0])

    frame_mask = (g != bg) & (g != marker_color)
    frame_components = components_mask(frame_mask, connectivity=4)
    assert len(frame_components) == 1
    frame_bbox = bbox(frame_mask)
    assert frame_bbox is not None
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 + 1 >= 3
    assert c1 - c0 + 1 >= 3

    frame = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = frame.shape
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True

    border_colors = set(map(int, frame[edge_mask]))
    interior_colors = set(map(int, frame[~edge_mask]))
    assert len(border_colors) == 1
    assert len(interior_colors) == 1
    border_color = border_colors.pop()
    interior_color = interior_colors.pop()
    assert border_color != bg
    assert interior_color != bg
    assert border_color != interior_color

    expected_frame = np.full_like(frame, interior_color)
    expected_frame[edge_mask] = border_color
    assert np.array_equal(frame, expected_frame)

    expanded_boxes = []
    if marker_r < r0 and c0 <= marker_c <= c1:
        expanded_boxes.append((marker_r, c0, r1, c1))
    if marker_r > r1 and c0 <= marker_c <= c1:
        expanded_boxes.append((r0, c0, marker_r, c1))
    if marker_c < c0 and r0 <= marker_r <= r1:
        expanded_boxes.append((r0, marker_c, r1, c1))
    if marker_c > c1 and r0 <= marker_r <= r1:
        expanded_boxes.append((r0, c0, r1, marker_c))
    assert len(expanded_boxes) == 1
    nr0, nc0, nr1, nc1 = expanded_boxes[0]
    assert 0 <= nr0 <= nr1 < g.shape[0]
    assert 0 <= nc0 <= nc1 < g.shape[1]
    assert nr1 - nr0 + 1 >= 3
    assert nc1 - nc0 + 1 >= 3

    out = np.full_like(g, bg)
    out[nr0 : nr1 + 1, nc0 : nc1 + 1] = interior_color
    out[nr0, nc0 : nc1 + 1] = border_color
    out[nr1, nc0 : nc1 + 1] = border_color
    out[nr0 : nr1 + 1, nc0] = border_color
    out[nr0 : nr1 + 1, nc1] = border_color
    return out

# --------------- generator (o5x) ---------------

GRID_SIZE_RANGE = (11, 13)
BACKGROUND_COLOR = 0

# Palette choices.
MARKER_COLOR = 8
FRAME_COLOR_POOL = [color for color in range(1, 10) if color != MARKER_COLOR]

# Frame and expansion geometry.
DIRECTIONS = ["up", "down", "left", "right"]
VERTICAL_DIRECTIONS = {"up", "down"}
ORIGINAL_SIDE_RANGE = range(3, 6)
EXPANSION_DISTANCE_RANGE = range(4, 9)
EXPANDED_SIDE_MIN = 5
EXPANDED_SIDE_MAX = 12
FIXED_CROSS_AXIS_SIZE = 5

# Retry budget.
GENERATION_RETRIES = 300


def _gen_draw_frame(grid, r0, c0, r1, c1, border_color, interior_color):
    grid[r0 : r1 + 1, c0 : c1 + 1] = interior_color
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color


def _gen_valid_specs(vertical, axis_limit):
    valid_specs = []
    fixed = FIXED_CROSS_AXIS_SIZE
    for variable_side in ORIGINAL_SIDE_RANGE:
        for d in EXPANSION_DISTANCE_RANGE:
            expanded_side = variable_side + d
            if EXPANDED_SIDE_MIN <= expanded_side <= min(axis_limit, EXPANDED_SIDE_MAX):
                if vertical:
                    valid_specs.append((variable_side, fixed, d, expanded_side, fixed))
                else:
                    valid_specs.append((fixed, variable_side, d, fixed, expanded_side))
    return valid_specs


def _gen_shrink_toward_marker(direction, expanded_box, d):
    nr0, nc0, nr1, nc1 = expanded_box
    if direction == "up":
        frame_box = (nr0 + d, nc0, nr1, nc1)
        marker = (nr0, random.randint(nc0, nc1))
    elif direction == "down":
        frame_box = (nr0, nc0, nr1 - d, nc1)
        marker = (nr1, random.randint(nc0, nc1))
    elif direction == "left":
        frame_box = (nr0, nc0 + d, nr1, nc1)
        marker = (random.randint(nr0, nr1), nc0)
    else:
        frame_box = (nr0, nc0, nr1, nc1 - d)
        marker = (random.randint(nr0, nr1), nc1)
    return frame_box, marker


def _gen_render_pair(h, w, frame_box, expanded_box, marker, border_color, interior_color):
    inp = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    out = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    _gen_draw_frame(inp, *frame_box, border_color, interior_color)
    inp[marker] = MARKER_COLOR
    _gen_draw_frame(out, *expanded_box, border_color, interior_color)
    return inp, out


def _gen_attempt():
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    border_color, interior_color = random.sample(FRAME_COLOR_POOL, 2)

    direction = random.choice(DIRECTIONS)
    vertical = direction in VERTICAL_DIRECTIONS
    axis_limit = h if vertical else w
    valid_specs = _gen_valid_specs(vertical, axis_limit)
    if not valid_specs:
        return None

    _, _, d, exp_h, exp_w = random.choice(valid_specs)
    nr0 = random.randint(0, h - exp_h)
    nc0 = random.randint(0, w - exp_w)
    expanded_box = (nr0, nc0, nr0 + exp_h - 1, nc0 + exp_w - 1)
    frame_box, marker = _gen_shrink_toward_marker(direction, expanded_box, d)
    inp, out = _gen_render_pair(h, w, frame_box, expanded_box, marker, border_color, interior_color)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b548a754 input")
