import numpy as np

from arc_utils import colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a foreground-color rectangular frame just inside the four marker pixels.

    Input structure:
    - The grid has a dominant background color, one foreground drawing color, and one marker color.
    - The marker color appears exactly four times, with unique top, bottom, left, and right extrema.
    - The foreground drawing color is the only non-background, non-marker color.

    Rule:
    - Use the min/max marker rows and columns as outside guides.
    - Paint the rectangle border from one row/column inside those marker extrema in the foreground color.
    - Preserve all existing pixels, including the markers and the original foreground object.

    Key insight:
    - The four markers are control pixels for the frame sides; the existing foreground object supplies the color to draw with.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    counts = {color: int(np.sum(g == color)) for color in colors}
    marker_candidates = [color for color, count in counts.items() if count == 4]
    assert len(marker_candidates) == 1, "expected exactly one four-pixel marker color"
    marker = marker_candidates[0]

    draw_colors = [color for color in colors if color != marker]
    assert len(draw_colors) == 1, "expected exactly one foreground drawing color"
    draw_color = draw_colors[0]

    marker_coords = np.argwhere(g == marker)
    assert marker_coords.shape == (4, 2), "expected four marker pixels"
    rows = marker_coords[:, 0]
    cols = marker_coords[:, 1]
    assert np.sum(rows == rows.min()) == 1, "expected a unique top marker"
    assert np.sum(rows == rows.max()) == 1, "expected a unique bottom marker"
    assert np.sum(cols == cols.min()) == 1, "expected a unique left marker"
    assert np.sum(cols == cols.max()) == 1, "expected a unique right marker"

    r0 = int(rows.min()) + 1
    r1 = int(rows.max()) - 1
    c0 = int(cols.min()) + 1
    c1 = int(cols.max()) - 1
    height, width = map(int, g.shape)
    assert 0 <= r0 < r1 < height, "marker rows must enclose a nonempty frame"
    assert 0 <= c0 < c1 < width, "marker columns must enclose a nonempty frame"

    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[r0, c0 : c1 + 1] = True
    frame_mask[r1, c0 : c1 + 1] = True
    frame_mask[r0 : r1 + 1, c0] = True
    frame_mask[r0 : r1 + 1, c1] = True
    assert not np.any(g[frame_mask] == marker), "frame must not overwrite markers"

    out = g.copy()
    out[frame_mask] = draw_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and shape bounds (upper bounds are exclusive where passed to randint).
GRID_H_RANGE, GRID_W_RANGE = (12, 16), (13, 16)
FRAME_H_RANGE, FRAME_W_RANGE = (9, 12), (8, 11)
OBJECT_SIZE_RANGE = (3, 5)
# Color, spacing, and retry controls.
BACKGROUND_COLOR, MARKER_COLOR = 0, 5
DRAW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
OUTER_MARGIN = 1
OBJECT_FRAME_GAP = 2
MAX_ATTEMPTS = 300


def _paint_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_layout(h: int, w: int):
    """Return frame bbox, four outside markers, and hollow object bbox."""
    frame_h_choices = [x for x in range(*FRAME_H_RANGE) if x <= h - 2 * OUTER_MARGIN]
    frame_h = int(np.random.choice(frame_h_choices))
    frame_w = int(np.random.randint(*FRAME_W_RANGE))
    r0 = int(np.random.randint(OUTER_MARGIN, h - frame_h))
    c0 = int(np.random.randint(OUTER_MARGIN, w - frame_w))
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    markers = [
        (r0 - OUTER_MARGIN, int(np.random.randint(c0 + OUTER_MARGIN, c1))),
        (r1 + OUTER_MARGIN, int(np.random.randint(c0 + OUTER_MARGIN, c1))),
        (int(np.random.randint(r0 + OUTER_MARGIN, r1)), c0 - OUTER_MARGIN),
        (int(np.random.randint(r0 + OUTER_MARGIN, r1)), c1 + OUTER_MARGIN),
    ]
    obj_h = int(np.random.randint(*OBJECT_SIZE_RANGE))
    obj_w = int(np.random.randint(*OBJECT_SIZE_RANGE))
    obj_r0 = int(np.random.randint(r0 + OBJECT_FRAME_GAP, r1 - obj_h))
    obj_c0 = int(np.random.randint(c0 + OBJECT_FRAME_GAP, c1 - obj_w))
    return (r0, c0, r1, c1), markers, (obj_r0, obj_c0, obj_r0 + obj_h - 1, obj_c0 + obj_w - 1)


def _render_pair(shape: tuple[int, int], draw_color: int, frame_bbox, markers, object_bbox):
    """Build matching input/output grids for the sampled layout."""
    out = np.zeros(shape, dtype=int)
    _paint_rect_border(out, frame_bbox, draw_color)
    _paint_rect_border(out, object_bbox, draw_color)
    for marker in markers:
        out[marker] = MARKER_COLOR
    inp = out.copy()
    _paint_rect_border(inp, frame_bbox, BACKGROUND_COLOR)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*GRID_H_RANGE))
    w = int(np.random.randint(*GRID_W_RANGE))
    draw_color = int(np.random.choice(DRAW_COLORS))
    frame_bbox, markers, object_bbox = _sample_layout(h, w)
    inp, out = _render_pair((h, w), draw_color, frame_bbox, markers, object_bbox)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 928ad970 input")
