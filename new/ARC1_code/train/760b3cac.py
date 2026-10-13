from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, components_mask, get_bg
import itertools
import random


def _two_vertical_color_masks(
    g: np.ndarray, bg: int
) -> tuple[tuple[int, tuple[int, int, int, int]], tuple[int, tuple[int, int, int, int]]]:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two non-background colors"

    items: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        b = bbox(g == color)
        assert b is not None, "non-background color must have pixels"
        items.append((int(color), b))

    first, second = items
    if first[1][2] < second[1][0]:
        return first, second
    if second[1][2] < first[1][0]:
        return second, first
    raise AssertionError("expected one upper mask and one lower marker")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the upper sprite into the adjacent side selected by the lower marker, reflecting it horizontally.

    Input structure:
    - The background contains two non-background color masks in vertically separated, aligned bboxes.
    - The upper mask is the source sprite.
    - The lower mask is a marker with one pixel on either the left or right end of its top row.

    Rule:
    - If the marker's top-row pixel is on the left end, use the same-size block immediately left of the source bbox.
    - If it is on the right end, use the same-size block immediately right of the source bbox.
    - Horizontally reflect the source mask and paint it into that destination block with the source color.

    Key insight:
    The lower marker is a direction code for which side should receive the reflected copy of the upper sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    (source_color, source_bbox), (marker_color, marker_bbox) = _two_vertical_color_masks(g, bg)
    sr0, sc0, sr1, sc1 = source_bbox
    mr0, mc0, mr1, mc1 = marker_bbox

    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    mh, mw = mr1 - mr0 + 1, mc1 - mc0 + 1
    assert (sh, sw) == (mh, mw), "source and marker bboxes must have the same shape"
    assert (sc0, sc1) == (mc0, mc1), "source and marker bboxes must be column-aligned"

    marker_crop = g[mr0 : mr1 + 1, mc0 : mc1 + 1] == marker_color
    top_cols = np.flatnonzero(marker_crop[0])
    assert len(top_cols) == 1, "marker top row must contain one control pixel"
    control_col = int(top_cols[0])
    assert control_col in (0, mw - 1), "marker control pixel must be on a horizontal end"

    if control_col == 0:
        dst_c0 = sc0 - sw
    else:
        dst_c0 = sc1 + 1
    dst_c1 = dst_c0 + sw - 1
    assert 0 <= dst_c0 and dst_c1 < g.shape[1], "destination block must be in bounds"

    dest = g[sr0 : sr1 + 1, dst_c0 : dst_c1 + 1]
    assert np.all(dest == bg), "destination block must be empty background"

    source_mask = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == source_color
    out = g.copy()
    out_dest = out[sr0 : sr1 + 1, dst_c0 : dst_c1 + 1]
    out_dest[np.fliplr(source_mask)] = source_color
    return out

# --------------- generator (o5x) ---------------
# Canvas and panel geometry.
GRID_SHAPE, PANEL_SHAPE = (6, 9), (3, 3)
SOURCE_ROWS, SOURCE_COLS = slice(0, 3), slice(3, 6)
MARKER_ROWS, MARKER_COLS = slice(3, 6), SOURCE_COLS
DEST_COLS = {True: 0, False: 6}

# Colors and sampling controls.
BG_COLOR, SOURCE_COLOR, MARKER_COLOR = 0, 8, 4
MIN_SOURCE_PIXELS, MAX_SOURCE_PIXELS, MAX_SOURCE_COMPONENTS = 4, 7, 3
LEFT_PROBABILITY, GENERATION_ATTEMPTS = 0.5, 300

# Direction marker masks.
MARKER_MASKS = {
    True: np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    False: np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
}

def _gen_source_mask(pixel_count: int) -> np.ndarray:
    masks: list[np.ndarray] = []
    for cells in itertools.combinations(range(PANEL_SHAPE[0] * PANEL_SHAPE[1]), pixel_count):
        mask = np.zeros(PANEL_SHAPE, dtype=bool)
        rows, cols = np.divmod(cells, PANEL_SHAPE[1])
        mask[rows, cols] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if (
            np.array_equal(mask, np.fliplr(mask))
            or len(components_mask(mask, connectivity=4)) > MAX_SOURCE_COMPONENTS
        ):
            continue
        masks.append(mask)
    assert masks, "no valid source masks for requested pixel count"
    return random.choice(masks).copy()

def _render_input(source_mask: np.ndarray, go_left: bool) -> np.ndarray:
    """Paint the fixed source and direction-marker panels."""
    inp = np.zeros(GRID_SHAPE, dtype=int)
    inp[SOURCE_ROWS, SOURCE_COLS] = np.where(source_mask, SOURCE_COLOR, BG_COLOR)
    inp[MARKER_ROWS, MARKER_COLS] = np.where(MARKER_MASKS[go_left], MARKER_COLOR, BG_COLOR)
    return inp

def _gen_attempt() -> np.ndarray | None:
    source_mask = _gen_source_mask(random.randint(MIN_SOURCE_PIXELS, MAX_SOURCE_PIXELS))
    go_left = random.random() < LEFT_PROBABILITY
    inp = _render_input(source_mask, go_left)

    out = inp.copy()
    dst_c0 = DEST_COLS[go_left]
    dst_panel = out[SOURCE_ROWS, dst_c0 : dst_c0 + PANEL_SHAPE[1]]
    dst_panel[:] = np.where(np.fliplr(source_mask), SOURCE_COLOR, BG_COLOR)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 760b3cac")
