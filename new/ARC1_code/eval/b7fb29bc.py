import numpy as np

from arc_utils import Color, bbox, bbox_h, bbox_w, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of a rectangular frame with alternating red and yellow square rings around its inner control pixel.

    Input structure
    - The grid has one background color and one foreground color.
    - The foreground forms a rectangular one-pixel-thick frame.
    - Exactly one foreground control pixel lies inside the frame.

    Rule
    - Preserve the frame and the control pixel.
    - For each background pixel inside the frame, compute its Chebyshev distance
      from the control pixel.
    - Odd distances become yellow; even distances become red.

    Key insight
    - The added colors are concentric square rings centered on the lone interior
      control pixel, with ring color determined by distance parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    frame_color = int(fg_colors[0])

    frame_bbox = bbox(g != bg)
    assert frame_bbox is not None, "expected a foreground frame"
    r0, c0, r1, c1 = frame_bbox
    assert bbox_h(frame_bbox) >= 3 and bbox_w(frame_bbox) >= 3, "frame must have an interior"
    assert np.all(g[r0, c0 : c1 + 1] == frame_color), "top frame edge must be solid"
    assert np.all(g[r1, c0 : c1 + 1] == frame_color), "bottom frame edge must be solid"
    assert np.all(g[r0 : r1 + 1, c0] == frame_color), "left frame edge must be solid"
    assert np.all(g[r0 : r1 + 1, c1] == frame_color), "right frame edge must be solid"

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert np.all((interior == bg) | (interior == frame_color)), "interior must contain only bg and control pixels"
    markers = np.argwhere(interior == frame_color)
    assert markers.shape[0] == 1, "expected exactly one interior control pixel"
    marker_r = r0 + 1 + int(markers[0, 0])
    marker_c = c0 + 1 + int(markers[0, 1])

    out = g.copy()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if g[r, c] == bg:
                distance = max(abs(r - marker_r), abs(c - marker_c))
                assert distance > 0, "control pixel should be the only distance-zero interior foreground"
                out[r, c] = Color.YELLOW if distance % 2 == 1 else Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas/frame bounds.
CANVAS_MIN_SIZE, CANVAS_MAX_SIZE = 10, 20
FRAME_MIN_SIZE, FRAME_MAX_SIZE, FRAME_SIZE_STEP, FRAME_MARGIN = 5, 13, 2, 1
# Palette and oracle fill colors.
BACKGROUND_COLOR = Color.BLACK
FRAME_COLORS = [Color.BLUE, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
ODD_RING_COLOR, EVEN_RING_COLOR = Color.YELLOW, Color.RED
# Rejection sampling budget.
GEN_RETRY_LIMIT = 300

def _gen_frame_choices(canvas_size):
    return list(range(FRAME_MIN_SIZE, min(FRAME_MAX_SIZE, canvas_size - 2) + 1, FRAME_SIZE_STEP))

def _gen_expected_output_b7fb29bc(inp, r0, c0, r1, c1, marker_r, marker_c):
    out = inp.copy()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if inp[r, c] == BACKGROUND_COLOR:
                dist = max(abs(r - marker_r), abs(c - marker_c))
                out[r, c] = ODD_RING_COLOR if dist % 2 else EVEN_RING_COLOR
    return out

def _gen_attempt():
    h = int(np.random.randint(CANVAS_MIN_SIZE, CANVAS_MAX_SIZE + 1))
    w = int(np.random.randint(CANVAS_MIN_SIZE, CANVAS_MAX_SIZE + 1))

    fh_choices = _gen_frame_choices(h)
    fw_choices = _gen_frame_choices(w)
    if not fh_choices or not fw_choices:
        return None

    fh = int(np.random.choice(fh_choices))
    fw = int(np.random.choice(fw_choices))
    r0 = int(np.random.randint(FRAME_MARGIN, h - fh))
    c0 = int(np.random.randint(FRAME_MARGIN, w - fw))
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1

    frame_color = int(np.random.choice(FRAME_COLORS))
    inp = np.zeros((h, w), dtype=int)
    inp[r0, c0 : c1 + 1] = frame_color
    inp[r1, c0 : c1 + 1] = frame_color
    inp[r0 : r1 + 1, c0] = frame_color
    inp[r0 : r1 + 1, c1] = frame_color

    marker_r = int(np.random.randint(r0 + 1, r1))
    marker_c = int(np.random.randint(c0 + 1, c1))
    inp[marker_r, marker_c] = frame_color

    expected = _gen_expected_output_b7fb29bc(inp, r0, c0, r1, c1, marker_r, marker_c)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None

def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid b7fb29bc input")
