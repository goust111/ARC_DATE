import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, paste as _gen_paste, random_free_location as _gen_random_free_location
import random as _gen_random


def _frame_patch(h: int, w: int) -> np.ndarray:
    assert h >= 6 and w >= 6
    assert h % 2 == 0 and w % 2 == 0

    patch = np.full((h, w), 4, dtype=int)
    mid_r = h // 2
    mid_c = w // 2

    patch[:2, :mid_c] = 6
    patch[:2, mid_c:] = 1
    patch[h - 2 :, :mid_c] = 2
    patch[h - 2 :, mid_c:] = 3

    patch[2:mid_r, :2] = 6
    patch[2:mid_r, w - 2 :] = 1
    patch[mid_r : h - 2, :2] = 2
    patch[mid_r : h - 2, w - 2 :] = 3

    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each solid teal rectangle with a two-pixel frame colored by quadrant and a yellow interior.

    Input structure
    The grid contains black background and one or more solid teal rectangular objects,
    each with even height and width and at least six cells in both dimensions.

    Rule
    For every teal rectangle, make a same-size patch. The cells at least two cells
    away from every edge are yellow. The two-cell frame is colored by quadrant:
    upper-left pink, upper-right blue, lower-left red, and lower-right green.

    Key insight
    Each rectangle is independently recolored from relative row/column position and
    distance to its rectangle border; the black background is unchanged.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    assert colors <= {0, 8}

    out = g.copy()
    rects = find_components(g, bg=0, colors=[8], connectivity=4)
    assert rects
    assert sum(int(rect["area"]) for rect in rects) == int(np.count_nonzero(g == 8))

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        h = bbox_h(rect["bbox"])
        w = bbox_w(rect["bbox"])
        assert int(rect["area"]) == h * w
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == 8)
        out[r0 : r1 + 1, c0 : c1 + 1] = _frame_patch(h, w)

    return out

# --------------- generator (o5x) ---------------

# Generator controls: canvas/object bounds, colors, spacing, retries.
_GEN_GRID_SHAPE, _GEN_RECT_COUNTS = (23, 23), (2, 3)
_GEN_RECT_SIZES = (6, 8, 10, 12, 14)
_GEN_BACKGROUND, _GEN_RECT_COLOR = 0, 8
_GEN_EDGE_MARGIN, _GEN_RECT_GAP = 1, 1
_GEN_PADDING_CONNECTIVITY = 8
_GEN_RETRY_LIMIT = 300

def _gen_attempt() -> np.ndarray | None:
    grid = np.full(_GEN_GRID_SHAPE, _GEN_BACKGROUND, dtype=int)

    for _ in range(_gen_random.choice(_GEN_RECT_COUNTS)):
        h = _gen_random.choice(_GEN_RECT_SIZES)
        w = _gen_random.choice(_GEN_RECT_SIZES)
        patch = np.full((h, w), _GEN_RECT_COLOR, dtype=int)
        try:
            r, c = _gen_random_free_location(
                grid,
                patch,
                transparent=_GEN_BACKGROUND,
                border=_GEN_EDGE_MARGIN,
                padding=_GEN_RECT_GAP,
                padding_connectivity=_GEN_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        _gen_paste(grid, patch, r, c)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        candidate = _gen_attempt()
        if candidate is None:
            continue
        try:
            main(candidate)
        except AssertionError:
            continue
        return candidate
    raise RuntimeError("failed to generate a valid 639f5a19 input")
