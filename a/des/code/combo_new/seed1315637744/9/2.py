import random

import numpy as np


BG = 0
OVERLAP_COLOR = 3
INPUT_COLORS = tuple(color for color in range(1, 10) if color != OVERLAP_COLOR)
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300
OVERLAP_MIN = 2
OVERLAP_MAX = 8
EXTEND_MIN = 1
EXTEND_MAX = 8
CANVAS_SLACK_MAX = 8
SIDE_OWNER_PATTERNS = tuple(
    (top_owner, bottom_owner, left_owner, right_owner)
    for top_owner in ("A", "B")
    for bottom_owner in ("A", "B")
    for left_owner in ("A", "B")
    for right_owner in ("A", "B")
    if len({top_owner, bottom_owner, left_owner, right_owner}) > 1
)


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _rect_corners(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    top, left, bottom, right = rect
    return {
        (int(top), int(left)),
        (int(top), int(right)),
        (int(bottom), int(left)),
        (int(bottom), int(right)),
    }


def _extract_rectangle(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (4, 2), f"color {color} must appear exactly four times"

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    assert len(rows) == 2 and len(cols) == 2, "four points must define a rectangle"

    rect = (int(rows[0]), int(cols[0]), int(rows[1]), int(cols[1]))
    assert {tuple(map(int, pt)) for pt in coords} == _rect_corners(rect)
    return rect


def _intersection(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    top = max(int(rect_a[0]), int(rect_b[0]))
    left = max(int(rect_a[1]), int(rect_b[1]))
    bottom = min(int(rect_a[2]), int(rect_b[2]))
    right = min(int(rect_a[3]), int(rect_b[3]))
    if top > bottom or left > right:
        return None
    return (top, left, bottom, right)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    top, left, bottom, right = rect
    return int(bottom - top + 1) * int(right - left + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the two corner-marked rectangles and paint their overlap with a fixed mixed color.

    Input structure:
    - The background is black.
    - Exactly two non-background colors appear, each as four isolated pixels.
    - The four pixels of each color are the corners of one axis-aligned rectangle.
    - The two rectangles overlap in a positive-area region.

    Rule:
    - For each foreground color, treat its four pixels as the rectangle's four corners.
    - Fill the full rectangle interior, including its border, with that color.
    - Recolor every cell covered by both rectangles to the fixed overlap color green.
    - Leave all remaining cells as background.

    Key insight:
    - The visible single pixels are only corner markers; the real objects are the hidden rectangles reconstructed from those four-point sets, after which the overlap must be handled separately.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 2, "expected exactly two foreground colors"

    rects = [_extract_rectangle(g, color) for color in colors]

    out = np.full(g.shape, bg, dtype=int)
    coverage = np.zeros(g.shape, dtype=np.int8)
    for color, (top, left, bottom, right) in zip(colors, rects):
        out[top : bottom + 1, left : right + 1] = int(color)
        coverage[top : bottom + 1, left : right + 1] += 1

    out[coverage >= 2] = OVERLAP_COLOR
    return out


# --------------- generator ---------------

def _build_rectangles(
    owners: tuple[str, str, str, str],
    overlap_rect: tuple[int, int, int, int],
    extents: dict[str, dict[str, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    overlap_top, overlap_left, overlap_bottom, overlap_right = overlap_rect
    top_owner, bottom_owner, left_owner, right_owner = owners

    if top_owner == "A":
        top_a = overlap_top
        top_b = overlap_top - int(extents["top"]["B"])
    else:
        top_a = overlap_top - int(extents["top"]["A"])
        top_b = overlap_top

    if bottom_owner == "A":
        bottom_a = overlap_bottom
        bottom_b = overlap_bottom + int(extents["bottom"]["B"])
    else:
        bottom_a = overlap_bottom + int(extents["bottom"]["A"])
        bottom_b = overlap_bottom

    if left_owner == "A":
        left_a = overlap_left
        left_b = overlap_left - int(extents["left"]["B"])
    else:
        left_a = overlap_left - int(extents["left"]["A"])
        left_b = overlap_left

    if right_owner == "A":
        right_a = overlap_right
        right_b = overlap_right + int(extents["right"]["B"])
    else:
        right_a = overlap_right + int(extents["right"]["A"])
        right_b = overlap_right

    return (top_a, left_a, bottom_a, right_a), (top_b, left_b, bottom_b, right_b)


def _sample_extents() -> dict[str, dict[str, int]]:
    return {
        side: {
            rect_name: random.randint(EXTEND_MIN, EXTEND_MAX)
            for rect_name in ("A", "B")
        }
        for side in ("top", "bottom", "left", "right")
    }


def _gen_attempt() -> np.ndarray | None:
    owners = random.choice(SIDE_OWNER_PATTERNS)
    extents = _sample_extents()
    overlap_h = random.randint(OVERLAP_MIN, OVERLAP_MAX)
    overlap_w = random.randint(OVERLAP_MIN, OVERLAP_MAX)

    top_owner, bottom_owner, left_owner, right_owner = owners
    outer_top = int(extents["top"]["B"] if top_owner == "A" else extents["top"]["A"])
    outer_bottom = int(extents["bottom"]["B"] if bottom_owner == "A" else extents["bottom"]["A"])
    outer_left = int(extents["left"]["B"] if left_owner == "A" else extents["left"]["A"])
    outer_right = int(extents["right"]["B"] if right_owner == "A" else extents["right"]["A"])

    used_h = overlap_h + outer_top + outer_bottom
    used_w = overlap_w + outer_left + outer_right
    if used_h + 2 > MAX_GRID_SIZE or used_w + 2 > MAX_GRID_SIZE:
        return None

    extra_h = random.randint(2, min(CANVAS_SLACK_MAX, MAX_GRID_SIZE - used_h))
    extra_w = random.randint(2, min(CANVAS_SLACK_MAX, MAX_GRID_SIZE - used_w))
    grid_h = used_h + extra_h
    grid_w = used_w + extra_w

    top_margin = random.randint(1, extra_h - 1)
    left_margin = random.randint(1, extra_w - 1)
    overlap_top = top_margin + outer_top
    overlap_left = left_margin + outer_left
    overlap_rect = (
        overlap_top,
        overlap_left,
        overlap_top + overlap_h - 1,
        overlap_left + overlap_w - 1,
    )

    rect_a, rect_b = _build_rectangles(owners, overlap_rect, extents)
    if _intersection(rect_a, rect_b) is None:
        return None
    if _rect_corners(rect_a) & _rect_corners(rect_b):
        return None

    all_corners = list(_rect_corners(rect_a) | _rect_corners(rect_b))
    if len(all_corners) != 8:
        return None
    if not all(0 <= r < grid_h and 0 <= c < grid_w for r, c in all_corners):
        return None

    overlap = _intersection(rect_a, rect_b)
    assert overlap is not None
    if _rect_area(overlap) >= _rect_area(rect_a) or _rect_area(overlap) >= _rect_area(rect_b):
        return None

    color_a, color_b = random.sample(INPUT_COLORS, 2)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for r, c in _rect_corners(rect_a):
        grid[r, c] = int(color_a)
    for r, c in _rect_corners(rect_b):
        grid[r, c] = int(color_b)

    try:
        out = main(grid)
    except AssertionError:
        return None

    out_colors = set(map(int, np.unique(out)))
    if not {int(color_a), int(color_b), OVERLAP_COLOR}.issubset(out_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
