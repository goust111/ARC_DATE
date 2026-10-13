import numpy as np

from arc_utils import Color, bbox, components_mask, crop_bbox, get_bg, paste
import random
from itertools import combinations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop to the tight foreground bounding box and replace retained background pixels with black.

    Input structure
    - A single dominant background color surrounds one foreground mask made from one or more non-background colors.
    - The output keeps only the rectangular bounding box of that foreground mask.

    Rule
    - Find all pixels whose color is not the input background color.
    - Crop the input to the tight bounding box of those foreground pixels.
    - Inside that crop, replace the input background color with black and preserve every foreground color.

    Key insight
    - The output is not a connected-object extraction; it is the foreground mask's rectangular crop, so background gaps inside the crop become the output blank color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    foreground = g != bg
    crop_box = bbox(foreground)
    assert crop_box is not None, "expected at least one foreground pixel"

    out = crop_bbox(g, crop_box)
    out[out == bg] = Color.BLACK
    return out

# --------------- generator (o5x) ---------------
# Grid, crop, and palette ranges.
GRID_ROWS, GRID_COLS = (5, 7), (6, 7)
CROP_ROWS, CROP_COLS = (2, 3), (2, 3)
COLOR_RANGE = (1, 9)

# Foreground mask constraints.
FOREGROUND_PIXELS = (3, 6)
MIN_FILL_FRACTION = (2, 3)
COMPONENT_COUNT, COMPONENT_CONNECTIVITY = (1, 2), 4

# Placement and retry controls.
INTERIOR_MARGIN, GEN_RETRIES = 1, 300

_GEN_MASK_CACHE = {}


def _gen_valid_masks(bh, bw, k):
    """Return cached masks that span the sampled crop and meet component bounds."""
    key = (bh, bw, k)
    if key in _GEN_MASK_CACHE:
        return _GEN_MASK_CACHE[key]
    cells = [(r, c) for r in range(bh) for c in range(bw)]
    masks = []
    for coords in combinations(cells, k):
        mask = np.zeros((bh, bw), dtype=bool)
        rr, cc = zip(*coords)
        mask[rr, cc] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        n_components = len(components_mask(mask, connectivity=COMPONENT_CONNECTIVITY))
        if not COMPONENT_COUNT[0] <= n_components <= COMPONENT_COUNT[1]:
            continue
        masks.append(mask)
    _GEN_MASK_CACHE[key] = masks
    return masks


def _build_output(mask, bg):
    """Paint the foreground mask with two non-background colors, both present."""
    colors = range(COLOR_RANGE[0], COLOR_RANGE[1] + 1)
    fg_colors = random.sample([c for c in colors if c != bg], 2)
    out = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    split = random.randint(1, len(coords) - 1)
    for color, cells in ((fg_colors[0], coords[:split]), (fg_colors[1], coords[split:])):
        for rr, cc in cells:
            out[rr, cc] = color
    return out


def _place_output(out, h, w, bg):
    """Embed the crop with output-black cells restored to the input background."""
    bh, bw = out.shape
    inp = np.full((h, w), bg, dtype=int)
    patch = out.copy()
    patch[patch == Color.BLACK] = bg
    r0 = random.randint(INTERIOR_MARGIN, h - bh - INTERIOR_MARGIN)
    c0 = random.randint(INTERIOR_MARGIN, w - bw - INTERIOR_MARGIN)
    paste(inp, patch, r0, c0)
    return inp


def _gen_attempt():
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    bg = random.randint(*COLOR_RANGE)
    bh = random.randint(*CROP_ROWS)
    bw = random.randint(*CROP_COLS)
    area = bh * bw
    lo = max(FOREGROUND_PIXELS[0], int(np.ceil(area * MIN_FILL_FRACTION[0] / MIN_FILL_FRACTION[1])))
    hi = min(FOREGROUND_PIXELS[1], area)
    k = random.randint(lo, hi)
    masks = _gen_valid_masks(bh, bw, k)
    if not masks:
        return None
    out = _build_output(random.choice(masks), bg)
    inp = _place_output(out, h, w, bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
