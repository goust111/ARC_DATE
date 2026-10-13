import random

import numpy as np


ORIENTATIONS = ("horizontal", "vertical", "diagonal")
_BRANCH_BAG: list[str] = []

GRID_MARGIN_MIN = 1
GRID_MARGIN_MAX = 6
INNER_HEIGHT_MIN = 2
INNER_HEIGHT_MAX = 9
INNER_WIDTH_MIN = 2
INNER_WIDTH_MAX = 9
GEN_RETRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _stripe_block(height: int, width: int, orientation: str, color_a: int, color_b: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    if orientation == "horizontal":
        use_a = rr % 2 == 0
    elif orientation == "vertical":
        use_a = cc % 2 == 0
    else:
        use_a = (rr - cc) % 2 == 0
    return np.where(use_a, color_a, color_b).astype(int)


def _detect_canvas(grid: np.ndarray) -> tuple[int, int, tuple[int, int, int, int], str, int, int]:
    bg = _most_common_color(grid)
    fg_mask = grid != bg
    assert np.any(fg_mask), "expected a framed canvas"

    coords = np.argwhere(fg_mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    frame = _frame_mask(grid.shape, (r0, c0, r1, c1))
    assert np.array_equal(fg_mask, frame), "expected exactly one rectangular frame"
    assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg), "canvas interior must start as background"

    corners = {
        "tl": int(grid[r0, c0]),
        "tr": int(grid[r0, c1]),
        "bl": int(grid[r1, c0]),
        "br": int(grid[r1, c1]),
    }

    non_corner = frame.copy()
    non_corner[r0, c0] = False
    non_corner[r0, c1] = False
    non_corner[r1, c0] = False
    non_corner[r1, c1] = False
    border_colors = np.unique(grid[non_corner])
    assert len(border_colors) == 1, "frame sides must use one border color"
    border_color = int(border_colors[0])
    assert border_color != bg, "border color must differ from background"

    corner_palette = set(corners.values())
    assert len(corner_palette) == 2, "corners must encode exactly two marker colors"
    assert all(color not in {bg, border_color} for color in corner_palette), "corner markers must be distinct"

    tl = corners["tl"]
    tr = corners["tr"]
    bl = corners["bl"]
    br = corners["br"]

    if tl == tr and bl == br and tl != bl:
        orientation = "horizontal"
        color_a, color_b = tl, bl
    elif tl == bl and tr == br and tl != tr:
        orientation = "vertical"
        color_a, color_b = tl, tr
    elif tl == br and tr == bl and tl != tr:
        orientation = "diagonal"
        color_a, color_b = tl, tr
    else:
        raise AssertionError("corner layout does not match any supported pattern")

    return bg, border_color, (r0, c0, r1, c1), orientation, color_a, color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the framed canvas with stripes whose direction is encoded by the corner colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, _, (r0, c0, r1, c1), orientation, color_a, color_b = _detect_canvas(g)
    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = _stripe_block(
        r1 - r0 - 1,
        c1 - c0 - 1,
        orientation,
        color_a,
        color_b,
    )
    return out


def _next_orientation(rng: random.Random) -> str:
    global _BRANCH_BAG
    if not _BRANCH_BAG:
        _BRANCH_BAG = list(ORIENTATIONS)
        rng.shuffle(_BRANCH_BAG)
    return _BRANCH_BAG.pop()


def _corner_values(orientation: str, color_a: int, color_b: int) -> tuple[int, int, int, int]:
    if orientation == "horizontal":
        return color_a, color_a, color_b, color_b
    if orientation == "vertical":
        return color_a, color_b, color_a, color_b
    return color_a, color_b, color_b, color_a


def _gen_attempt(rng: random.Random) -> np.ndarray:
    inner_h = rng.randint(INNER_HEIGHT_MIN, INNER_HEIGHT_MAX)
    inner_w = rng.randint(INNER_WIDTH_MIN, INNER_WIDTH_MAX)
    top = rng.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    bottom = rng.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    left = rng.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    right = rng.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)

    bg, border_color, color_a, color_b = rng.sample(list(range(10)), 4)
    orientation = _next_orientation(rng)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid = np.full((top + outer_h + bottom, left + outer_w + right), bg, dtype=int)

    r0 = top
    c0 = left
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color

    tl, tr, bl, br = _corner_values(orientation, color_a, color_b)
    grid[r0, c0] = tl
    grid[r0, c1] = tr
    grid[r1, c0] = bl
    grid[r1, c1] = br

    detected = _detect_canvas(grid)
    assert detected[3] == orientation
    assert detected[4] == color_a
    assert detected[5] == color_b

    expected = grid.copy()
    expected[r0 + 1 : r1, c0 + 1 : c1] = _stripe_block(inner_h, inner_w, orientation, color_a, color_b)
    assert np.array_equal(main(grid), expected)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
