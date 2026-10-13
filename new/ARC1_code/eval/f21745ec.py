import numpy as np

from arc_utils import bbox, colors_present, get_bg, paste, random_free_location
import math
import random


def _border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the interior motif from the unique patterned frame into every same-sized frame, recolored to each frame, and remove all other frames.

    Input structure
    - The grid contains a uniform background and several one-cell-thick rectangular frames.
    - Each foreground color marks one frame, and exactly one frame also has same-colored interior motif pixels.
    Rule
    - Use the patterned frame's outer height, width, and interior motif as the template.
    - Redraw only frames whose outer dimensions match the template frame, using each frame's own color for both the border and copied motif.
    - Leave frames with other dimensions as background.
    Key insight
    - The unique non-hollow frame supplies the motif; matching outer dimensions distinguish true target frames from distractor frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frames = []
    for color in colors_present(g, exclude={bg}):
        color_mask = g == color
        box = bbox(color_mask)
        assert box is not None
        r0, c0, r1, c1 = box
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = window.shape
        assert h >= 3 and w >= 3
        assert np.all((window == color) | (window == bg))

        local_mask = window == color
        border = _border_mask(h, w)
        assert np.all(local_mask[border])
        interior = local_mask & ~border
        frames.append((color, box, (h, w), border, interior))

    patterned = [frame for frame in frames if np.any(frame[4])]
    assert len(patterned) == 1
    _, _, template_shape, template_border, template_interior = patterned[0]
    template = template_border | template_interior

    out = np.full_like(g, bg)
    for color, box, shape, _, _ in frames:
        if shape != template_shape:
            continue
        r0, c0, r1, c1 = box
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[template] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and color space.
CANVAS_ROWS, CANVAS_COLS = (19, 23), (19, 22)
BG = 0
COLOR_POOL = range(1, 10)

# Frame size bounds.
TEMPLATE_H, TEMPLATE_W = (5, 7), (5, 8)
DISTRACTOR_H, DISTRACTOR_W = range(4, 8), range(4, 11)

# Motif density and frame-count choices.
MOTIF_FILL = (0.35, 0.65)
COUNT_PAIRS = [(3, 1), (3, 2), (3, 3), (4, 1), (4, 2)]

# Placement and retry controls.
PAD, PAD_CONNECTIVITY = 1, 8
GENERATION_RETRIES = 300


def _gen_frame_patch(h: int, w: int, color: int, motif: np.ndarray | None = None) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if motif is not None:
        patch[1:-1, 1:-1][motif] = color
    return patch


def _gen_random_motif(h: int, w: int) -> np.ndarray:
    inner_h = h - 2
    inner_w = w - 2
    area = inner_h * inner_w
    lo = math.ceil(area * MOTIF_FILL[0])
    hi = math.floor(area * MOTIF_FILL[1])
    count = random.randint(lo, hi)
    motif = np.zeros((inner_h, inner_w), dtype=bool)
    motif.flat[random.sample(range(area), count)] = True
    return motif


def _gen_stamp_frame(targets, occupancy, h, w, color, motif=None):
    """Place one padded frame and paste it into each target grid."""
    full_box = np.ones((h, w), dtype=int)
    r, c = random_free_location(occupancy, full_box, transparent=BG, padding=PAD, padding_connectivity=PAD_CONNECTIVITY)
    paste(occupancy, full_box, r, c, transparent=0)
    patch = _gen_frame_patch(h, w, color, motif)
    for target in targets:
        paste(target, patch, r, c, transparent=BG)
    return r, c


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*CANVAS_ROWS)
    cols = random.randint(*CANVAS_COLS)
    template_h = random.randint(*TEMPLATE_H)
    template_w = random.randint(*TEMPLATE_W)
    motif = _gen_random_motif(template_h, template_w)
    n_match, n_distract = random.choice(COUNT_PAIRS)
    colors = random.sample(COLOR_POOL, n_match + n_distract)
    match_colors = colors[:n_match]
    distract_colors = colors[n_match:]

    out = np.full((rows, cols), BG, dtype=int)
    inp = np.zeros_like(out)
    occupancy = np.zeros_like(out)
    match_locs: list[tuple[int, int]] = []

    try:
        for color in match_colors:
            r, c = _gen_stamp_frame((out, inp), occupancy, template_h, template_w, color, motif)
            match_locs.append((r, c))

        source_i = random.randrange(n_match)
        for i, (r, c) in enumerate(match_locs):
            if i == source_i:
                continue
            inp[r + 1 : r + template_h - 1, c + 1 : c + template_w - 1] = 0

        shape_pool = [
            (h, w)
            for h in DISTRACTOR_H
            for w in DISTRACTOR_W
            if (h, w) != (template_h, template_w)
        ]
        distract_shapes = random.sample(shape_pool, n_distract)
        for color, (h, w) in zip(distract_colors, distract_shapes):
            _gen_stamp_frame((inp,), occupancy, h, w, color)
    except ValueError:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
