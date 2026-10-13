import numpy as np

from arc_utils import bbox, paste


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize the framed input motif into a 5x5 icon and stamp two copies on a 15x15 canvas.

    Input structure
    The input is a rounded rectangular frame: the four corners are the background
    color, and every other outer-border pixel has one frame color. Non-background
    pixels inside that frame are the motif to preserve.

    Rule
    Make a 5x5 version of the rounded frame with a 3x3 interior. Crop the input
    interior motif to its bounding box, center that crop in the 3x3 interior, and
    paste the resulting 5x5 icon at upper-left positions (2, 2) and (8, 8) in a
    15x15 background canvas.

    Key insight
    The variable-size frame is only a carrier; the meaningful object is the
    compact interior motif, which is normalized to the fixed 3x3 icon interior.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3

    corners = (g[0, 0], g[0, w - 1], g[h - 1, 0], g[h - 1, w - 1])
    assert len(set(map(int, corners))) == 1
    bg = int(corners[0])

    border = np.concatenate((g[0, :], g[h - 1, :], g[:, 0], g[:, w - 1]))
    frame_colors = set(map(int, border)) - {bg}
    assert len(frame_colors) == 1
    (frame_color,) = tuple(frame_colors)

    assert np.all(g[0, 1 : w - 1] == frame_color)
    assert np.all(g[h - 1, 1 : w - 1] == frame_color)
    assert np.all(g[1 : h - 1, 0] == frame_color)
    assert np.all(g[1 : h - 1, w - 1] == frame_color)

    icon = np.full((5, 5), bg, dtype=int)
    icon[0, 1:4] = frame_color
    icon[4, 1:4] = frame_color
    icon[1:4, 0] = frame_color
    icon[1:4, 4] = frame_color

    interior = g[1 : h - 1, 1 : w - 1]
    motif_box = bbox(interior != bg)
    assert motif_box is not None
    r0, c0, r1, c1 = motif_box
    motif = interior[r0 : r1 + 1, c0 : c1 + 1]
    mh, mw = motif.shape
    assert mh <= 3 and mw <= 3

    icon_r = 1 + (3 - mh) // 2
    icon_c = 1 + (3 - mw) // 2
    icon[icon_r : icon_r + mh, icon_c : icon_c + mw] = motif

    out = np.full((15, 15), bg, dtype=int)
    out[2:7, 2:7] = icon
    out[8:13, 8:13] = icon
    return out

# --------------- generator (att2) ---------------

# Canvas and color constants.
_GEN_BG = 7
_GEN_FRAME_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
_GEN_PALETTE = [c for c in range(10) if c != _GEN_BG]

# Input/output geometry.
_GEN_INPUT_H_RANGE = (6, 10)
_GEN_INPUT_W_RANGE = (5, 10)
_GEN_ICON_SIZE = 5
_GEN_ICON_INTERIOR = 3
_GEN_OUTPUT_SIZE = 15
_GEN_OUTPUT_POSITIONS = ((2, 2), (8, 8))

# Motif sampling.
_GEN_MOTIF_DIM_RANGE = (1, 4)
_GEN_NON_BG_COUNT_RANGE = (1, 8)
_GEN_COLOR_COUNT_RANGE = (1, 3)
_GEN_FRAME_COLOR_INJECTION_RATE = 0.35

_GEN_MAX_ATTEMPTS = 300


def _gen_min_pixels_for_bbox(mh, mw):
    return 1 if mh == 1 and mw == 1 else 2


def _gen_choose_motif_shape(rng, non_bg_count):
    choices = []
    for mh in range(*_GEN_MOTIF_DIM_RANGE):
        for mw in range(*_GEN_MOTIF_DIM_RANGE):
            if _gen_min_pixels_for_bbox(mh, mw) <= non_bg_count <= mh * mw:
                choices.append((mh, mw))
    return choices[int(rng.integers(len(choices)))]


def _gen_full_bbox_coords(rng, mh, mw, non_bg_count):
    if mh == 1 and mw == 1:
        return [(0, 0)]

    if mh == 1:
        required = [(0, 0), (0, mw - 1)]
    elif mw == 1:
        required = [(0, 0), (mh - 1, 0)]
    elif int(rng.integers(2)) == 0:
        required = [(0, 0), (mh - 1, mw - 1)]
    else:
        required = [(0, mw - 1), (mh - 1, 0)]

    coords = list(required)
    pool = [(r, c) for r in range(mh) for c in range(mw) if (r, c) not in coords]
    need = non_bg_count - len(coords)
    if need:
        picked = rng.choice(len(pool), size=need, replace=False)
        coords.extend(pool[int(i)] for i in np.atleast_1d(picked))
    return coords


def _gen_make_motif(rng, bg, frame_color):
    non_bg_count = int(rng.integers(*_GEN_NON_BG_COUNT_RANGE))
    mh, mw = _gen_choose_motif_shape(rng, non_bg_count)
    coords = _gen_full_bbox_coords(rng, mh, mw, non_bg_count)

    color_count = 1 if non_bg_count == 1 else int(rng.integers(*_GEN_COLOR_COUNT_RANGE))
    colors = list(rng.choice(_GEN_PALETTE, size=color_count, replace=False))
    if color_count == 2 and frame_color not in colors and rng.random() < _GEN_FRAME_COLOR_INJECTION_RATE:
        colors[int(rng.integers(2))] = frame_color

    motif = np.full((mh, mw), bg, dtype=int)
    rng.shuffle(coords)
    if color_count == 1:
        for r, c in coords:
            motif[r, c] = int(colors[0])
    else:
        motif[coords[0]] = int(colors[0])
        motif[coords[1]] = int(colors[1])
        for r, c in coords[2:]:
            motif[r, c] = int(colors[int(rng.integers(2))])
    return motif


def _gen_draw_rounded_frame(grid, frame_color):
    """Paint the non-corner border cells in-place."""
    h, w = grid.shape
    grid[0, 1 : w - 1] = frame_color
    grid[h - 1, 1 : w - 1] = frame_color
    grid[1 : h - 1, 0] = frame_color
    grid[1 : h - 1, w - 1] = frame_color


def _gen_build_icon(bg, frame_color, motif):
    icon = np.full((_GEN_ICON_SIZE, _GEN_ICON_SIZE), bg, dtype=int)
    _gen_draw_rounded_frame(icon, frame_color)

    mh, mw = motif.shape
    icon_r = 1 + (_GEN_ICON_INTERIOR - mh) // 2
    icon_c = 1 + (_GEN_ICON_INTERIOR - mw) // 2
    paste(icon, motif, icon_r, icon_c)
    return icon


def _gen_build_input(rng, bg, frame_color, motif):
    h = int(rng.integers(*_GEN_INPUT_H_RANGE))
    w = int(rng.integers(*_GEN_INPUT_W_RANGE))
    inp = np.full((h, w), bg, dtype=int)
    _gen_draw_rounded_frame(inp, frame_color)

    mh, mw = motif.shape
    interior_h, interior_w = h - 2, w - 2
    r = int(rng.integers(0, interior_h - mh + 1))
    c = int(rng.integers(0, interior_w - mw + 1))
    paste(inp, motif, 1 + r, 1 + c)
    return inp


def _gen_attempt(rng):
    bg = _GEN_BG
    frame_color = int(rng.choice(_GEN_FRAME_COLORS))
    motif = _gen_make_motif(rng, bg, frame_color)

    icon = _gen_build_icon(bg, frame_color, motif)
    out = np.full((_GEN_OUTPUT_SIZE, _GEN_OUTPUT_SIZE), bg, dtype=int)
    for r, c in _GEN_OUTPUT_POSITIONS:
        paste(out, icon, r, c)

    inp = _gen_build_input(rng, bg, frame_color, motif)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid db118e2a input")
