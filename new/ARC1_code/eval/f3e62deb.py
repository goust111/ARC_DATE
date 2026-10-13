import numpy as np

from arc_utils import Color, bbox, colors_present, get_bg
import random


EDGE_BY_COLOR = {
    Color.GREEN: "left",
    Color.YELLOW: "down",
    Color.PINK: "up",
    Color.TEAL: "right",
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the single hollow square to the grid edge indicated by its color.

    Input structure:
    - The grid has a dominant background color and exactly one foreground color.
    - The foreground is one 3x3 hollow-square sprite of eight pixels.

    Rule:
    - Green moves left, yellow moves down, pink moves up, and teal moves right.
    - Erase the original sprite and translate the same sprite until its bbox is flush with that edge.

    Key insight:
    - The sprite color is a direction code; the shape is not changed, only slid to the corresponding boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H >= 3 and W >= 3, "expected room for a 3x3 sprite"

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    color = int(fg_colors[0])
    assert color in EDGE_BY_COLOR, "expected a known direction color"

    mask = g == color
    assert int(mask.sum()) == 8, "expected an eight-pixel hollow square"
    box = bbox(mask)
    assert box is not None, "expected foreground pixels"
    r0, c0, r1, c1 = box

    sprite = g[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.full((3, 3), color, dtype=int)
    expected[1, 1] = bg
    assert sprite.shape == (3, 3), "expected a 3x3 sprite bbox"
    assert np.array_equal(sprite, expected), "expected a hollow 3x3 square"

    edge = EDGE_BY_COLOR[color]
    if edge == "left":
        nr0, nc0 = r0, 0
    elif edge == "right":
        nr0, nc0 = r0, W - 3
    elif edge == "up":
        nr0, nc0 = 0, c0
    else:
        assert edge == "down", "unexpected edge"
        nr0, nc0 = H - 3, c0

    assert 0 <= nr0 <= H - 3 and 0 <= nc0 <= W - 3, "target sprite is out of bounds"
    out = np.full_like(g, bg)
    out[nr0 : nr0 + 3, nc0 : nc0 + 3] = sprite
    return out

# --------------- generator (o5x) ---------------
# Canvas and sprite geometry.
_GEN_H, _GEN_W = 10, 10
_GEN_BG = 0
_GEN_SIZE = 3

# Direction colors; tuple order preserves equal random.choice weights.
_GEN_EDGE_BY_COLOR = {
    int(Color.GREEN): "left",
    int(Color.YELLOW): "down",
    int(Color.PINK): "up",
    int(Color.TEAL): "right",
}
_GEN_COLORS = tuple(_GEN_EDGE_BY_COLOR)

# Source distance and oracle-check retry budget.
_GEN_DISTANCE_RANGE = (3, 5)
_GEN_MAX_ATTEMPTS = 300


def _gen_hollow_square(color):
    sprite = np.full((_GEN_SIZE, _GEN_SIZE), int(color), dtype=int)
    sprite[1, 1] = _GEN_BG
    return sprite


def _gen_render_square(color, r0, c0):
    grid = np.full((_GEN_H, _GEN_W), _GEN_BG, dtype=int)
    grid[r0 : r0 + _GEN_SIZE, c0 : c0 + _GEN_SIZE] = _gen_hollow_square(color)
    return grid


def _gen_destination(edge):
    max_r = _GEN_H - _GEN_SIZE
    max_c = _GEN_W - _GEN_SIZE
    if edge in {"left", "right"}:
        return random.randint(0, max_r), 0 if edge == "left" else max_c
    return 0 if edge == "up" else max_r, random.randint(0, max_c)


def _gen_source(edge, dest_r, dest_c, distance):
    max_r = _GEN_H - _GEN_SIZE
    max_c = _GEN_W - _GEN_SIZE
    if edge == "left":
        return dest_r, distance
    elif edge == "right":
        return dest_r, max_c - distance
    elif edge == "up":
        return distance, dest_c
    return max_r - distance, dest_c


def _gen_attempt():
    color = random.choice(_GEN_COLORS)
    edge = _GEN_EDGE_BY_COLOR[color]

    dest_r, dest_c = _gen_destination(edge)
    out = _gen_render_square(color, dest_r, dest_c)

    distance = random.randint(*_GEN_DISTANCE_RANGE)
    src_r, src_c = _gen_source(edge, dest_r, dest_c, distance)
    inp = _gen_render_square(color, src_r, src_c)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid f3e62deb input")
