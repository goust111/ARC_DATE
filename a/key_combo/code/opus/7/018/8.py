import random
from typing import Iterable

import numpy as np


GREY = 5
BLACK = 0
CORNER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_CHOICES = (8, 10, 12, 14, 16, 18)
MAX_GENERATION_ATTEMPTS = 400


def _quadrant_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    mh, mw = h // 2, w // 2
    tl = np.zeros(shape, dtype=bool)
    tr = np.zeros(shape, dtype=bool)
    bl = np.zeros(shape, dtype=bool)
    br = np.zeros(shape, dtype=bool)
    tl[:mh, :mw] = True
    tr[:mh, mw:] = True
    bl[mh:, :mw] = True
    br[mh:, mw:] = True
    return tl, tr, bl, br


def _colorize_diff(diff: np.ndarray, corners: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(diff.shape, dtype=int)
    tl_mask, tr_mask, bl_mask, br_mask = _quadrant_masks(diff.shape)
    out[diff & tl_mask] = corners[0]
    out[diff & tr_mask] = corners[1]
    out[diff & bl_mask] = corners[2]
    out[diff & br_mask] = corners[3]
    return out


def _shape_library() -> list[np.ndarray]:
    raw = [
        ["##"],
        ["#", "#"],
        ["###"],
        ["#", "#", "#"],
        ["##", "##"],
        ["##", "#."],
        ["##", ".#"],
        ["#.", "##"],
        [".#", "##"],
        ["###", ".#."],
        [".#.", "###"],
        ["#.", "##", "#."],
        [".#", "##", ".#"],
        ["###", "#.."],
        ["###", "..#"],
        ["#..", "###"],
        ["..#", "###"],
        ["##.", ".##"],
        [".##", "##."],
    ]
    shapes = []
    for rows in raw:
        arr = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
        shapes.append(arr)
    return shapes


SHAPES = _shape_library()


def _rot180_coords(coords: Iterable[tuple[int, int]], side: int) -> list[tuple[int, int]]:
    return [(side - 1 - r, side - 1 - c) for r, c in coords]


def _paint_shape(
    pattern: np.ndarray,
    shape: np.ndarray,
    top: int,
    left: int,
    *,
    color: int = GREY,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    h, w = shape.shape
    for r in range(h):
        for c in range(w):
            if shape[r, c]:
                rr, cc = top + r, left + c
                pattern[rr, cc] = color
                coords.append((rr, cc))
    return coords


def _can_place(pattern: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    side = pattern.shape[0]
    h, w = shape.shape
    if top < 0 or left < 0 or top + h > side or left + w > side:
        return False
    coords = []
    for r in range(h):
        for c in range(w):
            if shape[r, c]:
                rr, cc = top + r, left + c
                if (rr, cc) in ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)):
                    return False
                coords.append((rr, cc))
    if not coords:
        return False
    rot = _rot180_coords(coords, side)
    if len(set(coords)) != len(coords) or len(set(rot)) != len(rot):
        return False
    for rr, cc in coords:
        if pattern[rr, cc] != BLACK:
            return False
    return True


def _random_location_for_quadrant(side: int, shape: np.ndarray, quadrant: str) -> tuple[int, int] | None:
    mid = side // 2
    h, w = shape.shape
    if quadrant == "TL":
        r_lo, r_hi = 1, mid - h
        c_lo, c_hi = 1, mid - w
    elif quadrant == "TR":
        r_lo, r_hi = 1, mid - h
        c_lo, c_hi = mid, side - 2 - w + 1
    elif quadrant == "BL":
        r_lo, r_hi = mid, side - 2 - h + 1
        c_lo, c_hi = 1, mid - w
    else:
        r_lo, r_hi = mid, side - 2 - h + 1
        c_lo, c_hi = mid, side - 2 - w + 1
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _try_add_symmetric_feature(pattern: np.ndarray) -> bool:
    side = pattern.shape[0]
    for _ in range(40):
        shape = random.choice(SHAPES)
        quadrant = random.choice(("TL", "TR"))
        loc = _random_location_for_quadrant(side, shape, quadrant)
        if loc is None or not _can_place(pattern, shape, *loc):
            continue
        coords = []
        h, w = shape.shape
        for r in range(h):
            for c in range(w):
                if shape[r, c]:
                    rr, cc = loc[0] + r, loc[1] + c
                    coords.append((rr, cc))
        rot = _rot180_coords(coords, side)
        if any(pattern[rr, cc] != BLACK for rr, cc in rot):
            continue
        for rr, cc in coords + rot:
            pattern[rr, cc] = GREY
        return True
    return False


def _try_add_asymmetric_feature(pattern: np.ndarray, quadrant: str) -> bool:
    side = pattern.shape[0]
    for _ in range(80):
        shape = random.choice(SHAPES)
        loc = _random_location_for_quadrant(side, shape, quadrant)
        if loc is None or not _can_place(pattern, shape, *loc):
            continue
        coords = []
        h, w = shape.shape
        for r in range(h):
            for c in range(w):
                if shape[r, c]:
                    rr, cc = loc[0] + r, loc[1] + c
                    coords.append((rr, cc))
        rot = _rot180_coords(coords, side)
        # Keep the source feature visibly one-sided; its rotated partner is absent in the input.
        if any(pattern[rr, cc] != BLACK for rr, cc in rot):
            continue
        for rr, cc in coords:
            pattern[rr, cc] = GREY
        return True
    return False


def _input_corners(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    return int(grid[0, 0]), int(grid[0, w - 1]), int(grid[h - 1, 0]), int(grid[h - 1, w - 1])


def _pattern_without_corners(grid: np.ndarray) -> np.ndarray:
    pat = np.array(grid, dtype=int, copy=True)
    h, w = pat.shape
    pat[0, 0] = BLACK
    pat[0, w - 1] = BLACK
    pat[h - 1, 0] = BLACK
    pat[h - 1, w - 1] = BLACK
    return pat


def _expected_output(grid: np.ndarray) -> np.ndarray:
    pattern = _pattern_without_corners(grid)
    diff = pattern != np.rot90(pattern, 2)
    return _colorize_diff(diff, _input_corners(grid))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the non-corner pattern by 180 degrees, xor it with the original, and color the differences by corner-defined quadrants.

    Input structure:
    - The input is an even-sized square grid on a black background.
    - The four corner cells are distinct non-black legend pixels that define the colors for the top-left, top-right, bottom-left, and bottom-right quadrants.
    - Apart from the corners, the only foreground pattern cells are grey pixels placed on the black field.

    Rule:
    - Copy the grid and erase the four corner legend pixels to black.
    - Rotate that pattern-only grid by 180 degrees and compare it cell by cell with the unrotated pattern-only grid.
    - Produce a black output grid of the same size, and for every cell where the two pattern grids differ, fill that output cell with the legend color of its quadrant.

    Key insight:
    - The output is not the rotated pattern itself; it is the 180-degree asymmetry mask of the grey pattern, with each differing cell recolored solely by which quadrant it occupies.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    h, w = g.shape
    if h != w or h % 2 != 0:
        raise ValueError("expected an even-sized square grid")
    pattern = _pattern_without_corners(g)
    diff = pattern != np.rot90(pattern, 2)
    return _colorize_diff(diff, _input_corners(g))


def _quadrants_with_differences(diff: np.ndarray) -> set[str]:
    h, w = diff.shape
    mh, mw = h // 2, w // 2
    seen = set()
    if np.any(diff[:mh, :mw]):
        seen.add("TL")
    if np.any(diff[:mh, mw:]):
        seen.add("TR")
    if np.any(diff[mh:, :mw]):
        seen.add("BL")
    if np.any(diff[mh:, mw:]):
        seen.add("BR")
    return seen


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] % 2 != 0 or g.shape[0] > 30:
        return False
    side = g.shape[0]
    corners = _input_corners(g)
    if len(set(corners)) != 4:
        return False
    if any(c in (BLACK, GREY) for c in corners):
        return False
    pattern = _pattern_without_corners(g)
    allowed = {BLACK, GREY}
    if set(map(int, np.unique(pattern))) - allowed:
        return False

    diff = pattern != np.rot90(pattern, 2)
    if int(diff.sum()) < 4:
        return False
    if len(_quadrants_with_differences(diff)) < 2:
        return False

    out = main(g)
    if not np.array_equal(out, _colorize_diff(diff, corners)):
        return False
    if not set(map(int, np.unique(out))) <= {BLACK, *corners}:
        return False

    # Reject ambiguous examples where a simpler transform reproduces the same output.
    alt_masks = [
        pattern != BLACK,
        pattern != np.rot90(pattern, 1),
        pattern != np.rot90(pattern, -1),
        pattern != np.flipud(pattern),
        pattern != np.fliplr(pattern),
    ]
    if any(np.array_equal(out, _colorize_diff(mask, corners)) for mask in alt_masks):
        return False

    # Ensure quadrant coloring is necessary by requiring at least two used quadrant colors.
    used_colors = set(map(int, out[out != BLACK].ravel()))
    if len(used_colors) < 2:
        return False

    # Ensure xor is necessary by requiring some 180-symmetric grey cells that disappear.
    symmetric_grey = (pattern == GREY) & (np.rot90(pattern, 2) == GREY)
    if not np.any(symmetric_grey):
        return False

    # Ensure the mask is not trivial by requiring both filled->empty and empty->filled mismatches.
    if not np.any((pattern == GREY) & diff):
        return False
    if not np.any((np.rot90(pattern, 2) == GREY) & diff):
        return False

    # Corners themselves must stay black in the output because they are excluded before comparison.
    if any(out[r, c] != BLACK for r, c in ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        side = random.choice(SIDE_CHOICES)
        grid = np.zeros((side, side), dtype=int)
        corners = tuple(random.sample(CORNER_PALETTE, 4))
        grid[0, 0], grid[0, side - 1], grid[side - 1, 0], grid[side - 1, side - 1] = corners

        pattern = np.zeros((side, side), dtype=int)

        symmetric_count = random.randint(1, 3)
        if not all(_try_add_symmetric_feature(pattern) for _ in range(symmetric_count)):
            continue

        quadrants = ["TL", "TR", "BL", "BR"]
        random.shuffle(quadrants)
        # Use one or two diagonal families so some samples hit all four quadrants and some only two.
        primary_count = random.randint(2, 4)
        selected_quadrants = quadrants[:primary_count]
        if random.random() < 0.7 and not ({"TL", "BR"} <= set(selected_quadrants) or {"TR", "BL"} <= set(selected_quadrants)):
            selected_quadrants[:2] = ["TL", "TR"]
        if not all(_try_add_asymmetric_feature(pattern, q) for q in selected_quadrants):
            continue

        grid[pattern == GREY] = GREY
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
