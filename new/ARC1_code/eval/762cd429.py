from __future__ import annotations

import numpy as np

from arc_utils import bbox, get_bg, scale_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the left-edge 2x2 motif into rightward copies scaled by powers of two.

    Input structure
    - The background is the most common color.
    - The only foreground pixels form a filled 2x2 motif in columns 0 and 1.
    Rule
    - Start with a background-filled output of the same shape.
    - For scales 1, 2, 4, ... that fit horizontally, stamp the motif scaled by that factor.
    - Place each scaled stamp at column 2 * (scale - 1), with its top row band ending on the
      original motif's top row and its bottom row band starting on the original motif's bottom row.
    Key insight
    - The original 2x2 motif is the smallest copy in a dyadic sequence: each next copy doubles
      both dimensions, starts immediately to the right, and grows away from the seam between
      the motif's two rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    box = bbox(g != bg)
    assert box is not None, "expected foreground motif"
    r0, c0, r1, c1 = box
    assert c0 == 0 and c1 == 1, "motif must occupy the first two columns"
    assert r1 == r0 + 1, "motif must be exactly two rows tall"

    motif = g[r0 : r1 + 1, c0 : c1 + 1]
    assert motif.shape == (2, 2)
    assert np.all(motif != bg), "motif must be a filled 2x2 foreground block"

    height, width = g.shape
    out = np.full_like(g, bg)

    scale = 1
    while 2 * (scale - 1) + 2 * scale <= width:
        top = r0 - scale + 1
        left = 2 * (scale - 1)
        assert top >= 0 and top + 2 * scale <= height, "scaled motif must fit vertically"

        stamp = scale_sprite(motif, scale)
        out[top : top + 2 * scale, left : left + 2 * scale] = stamp
        scale *= 2

    assert scale > 1, "expected at least the original 2x2 motif to fit"
    return out

# --------------- generator (o5x) ---------------
# Canvas geometry and retry bound.
SCALE_COUNT_CHOICES = [2, 3, 4]
MAX_GRID_HEIGHT, MAX_SLACK_ROWS = 30, 2
GENERATION_RETRIES = 300
# Motif and color sampling.
BACKGROUND_COLOR, MOTIF_SIZE = 0, 2
MOTIF_CELLS = MOTIF_SIZE * MOTIF_SIZE
MOTIF_COLOR_COUNT_MIN, MOTIF_COLOR_COUNT_MAX_EXCL = 2, 5
NON_BACKGROUND_COLORS = np.arange(1, 10)

def _gen_expected_output(inp: np.ndarray, motif: np.ndarray, r0: int) -> np.ndarray:
    out = np.zeros_like(inp)
    scale = 1
    height, width = inp.shape
    while MOTIF_SIZE * (2 * scale - 1) <= width:
        top = r0 - scale + 1
        left = MOTIF_SIZE * (scale - 1)
        size = MOTIF_SIZE * scale
        if top < 0 or top + size > height:
            return np.full((0, 0), -1, dtype=int)
        stamp = scale_sprite(motif, scale)
        out[top : top + size, left : left + size] = stamp
        scale *= 2
    return out

def _gen_motif() -> np.ndarray:
    color_count = int(np.random.randint(MOTIF_COLOR_COUNT_MIN, MOTIF_COLOR_COUNT_MAX_EXCL))
    colors = np.random.choice(NON_BACKGROUND_COLORS, size=color_count, replace=False)
    vals = list(colors)
    vals.extend(np.random.choice(colors, size=MOTIF_CELLS - color_count, replace=True).tolist())
    np.random.shuffle(vals)
    return np.array(vals, dtype=int).reshape(MOTIF_SIZE, MOTIF_SIZE)

def _gen_attempt() -> np.ndarray | None:
    n_scales = int(np.random.choice(SCALE_COUNT_CHOICES))
    scale_max = 2 ** (n_scales - 1)
    max_extra = min(MAX_SLACK_ROWS, MAX_GRID_HEIGHT - MOTIF_SIZE * scale_max)
    height_extra = int(np.random.randint(0, max_extra + 1))
    height = MOTIF_SIZE * scale_max + height_extra
    width = 2 * MOTIF_SIZE * scale_max - MOTIF_SIZE
    r0 = int(np.random.randint(scale_max - 1, height - scale_max))

    motif = _gen_motif()
    inp = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, 0:MOTIF_SIZE] = motif

    expected = _gen_expected_output(inp, motif, r0)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 762cd429 input")
