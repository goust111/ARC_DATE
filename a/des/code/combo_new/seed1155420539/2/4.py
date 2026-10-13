from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]
BBox = tuple[int, int, int, int]

SIDES: tuple[str, ...] = ("top", "bottom", "left", "right")
SIDE_DIR: dict[str, Direction] = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

COLOR_COUNT = 10
GENERATION_ATTEMPTS = 500
GRID_SIZE_RANGE = (16, 30)
RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (3, 5)
PLACE_ATTEMPTS = 250
RESERVE_PAD = 1


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _ray_cells(shape: tuple[int, int], gap: Coord, direction: Direction, length: int) -> list[Coord]:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    r, c = int(gap[0] + dr), int(gap[1] + dc)
    cells: list[Coord] = []
    for _ in range(int(length)):
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _rectangle_info(grid: np.ndarray, bg: int, color: int) -> tuple[Coord, Direction, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size, "expected at least one pixel for each object color"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height >= 3 and width >= 3, "rectangles must be large enough for a non-corner gap"

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all((window == int(color)) | (window == int(bg)))), (
        "a rectangle bbox may contain only its color and the background gap"
    )

    missing = np.argwhere(window != int(color))
    assert missing.shape == (1, 2), "each colored rectangle must have exactly one missing pixel"
    gr = int(r0 + missing[0, 0])
    gc = int(c0 + missing[0, 1])
    assert int(grid[gr, gc]) == int(bg), "the missing rectangle pixel must be background"

    sides: list[str] = []
    if gr == r0:
        sides.append("top")
    if gr == r1:
        sides.append("bottom")
    if gc == c0:
        sides.append("left")
    if gc == c1:
        sides.append("right")
    assert len(sides) == 1, "the gap must lie on exactly one non-corner side"

    expected_area = int(height * width - 1)
    assert coords.shape[0] == expected_area, "rectangle must be solid except for the single gap"
    return (gr, gc), SIDE_DIR[sides[0]], int(coords.shape[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-color rays outward from the one-cell gap in each colored rectangle.

    Input structure:
    - The most common color is the background.
    - Several distinct non-background colors each form one solid rectangular
      block, except that exactly one non-corner boundary pixel of the rectangle
      is background rather than the block color.
    - Rectangles are separated by background, and their outward gap rays have
      clear background space until either the required length is drawn or the
      grid edge truncates the ray.

    Rule:
    - For every rectangle color, find the unique background gap inside that
      color's bounding box and determine which side of the rectangle contains it.
    - Count the colored pixels of that rectangle as N.
    - Starting just outside the gap, draw up to N consecutive background pixels
      in the outward normal direction of that side, stopping at the grid edge.
    - Preserve the original rectangle and keep the gap pixel itself as background.

    Key insight:
    - The gap is an exit that supplies a direction, while the two-dimensional
      colored area of the rectangle supplies the one-dimensional ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    assert colors, "expected at least one colored rectangle"

    out = g.copy()
    for color in colors:
        gap, direction, area = _rectangle_info(g, bg, color)
        for r, c in _ray_cells(g.shape, gap, direction, area):
            assert int(out[r, c]) == int(bg), "rays must grow through background only"
            out[r, c] = int(color)

    return out


# --------------- generator ---------------


def _bbox_cells(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = map(int, bbox)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _reserve(mask: np.ndarray, cells: list[Coord], pad: int = RESERVE_PAD) -> None:
    h, w = map(int, mask.shape)
    for r, c in cells:
        for rr in range(max(0, int(r) - pad), min(h, int(r) + pad + 1)):
            for cc in range(max(0, int(c) - pad), min(w, int(c) + pad + 1)):
                mask[rr, cc] = True


def _blocked(mask: np.ndarray, cells: list[Coord]) -> bool:
    return any(bool(mask[int(r), int(c)]) for r, c in cells)


def _gap_for_side(bbox: BBox, side: str) -> Coord:
    r0, c0, r1, c1 = map(int, bbox)
    if side == "top":
        return int(r0), random.randint(int(c0 + 1), int(c1 - 1))
    if side == "bottom":
        return int(r1), random.randint(int(c0 + 1), int(c1 - 1))
    if side == "left":
        return random.randint(int(r0 + 1), int(r1 - 1)), int(c0)
    assert side == "right"
    return random.randint(int(r0 + 1), int(r1 - 1)), int(c1)


def _outside_space(shape: tuple[int, int], bbox: BBox, side: str) -> int:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    if side == "top":
        return int(r0)
    if side == "bottom":
        return int(h - 1 - r1)
    if side == "left":
        return int(c0)
    assert side == "right"
    return int(w - 1 - c1)


def _paint_rectangle(grid: np.ndarray, bbox: BBox, gap: Coord, color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    grid[int(gap[0]), int(gap[1])] = int(_most_common_color(grid))


def _dominant_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = int(counts.get(int(bg), 0))
    return all(bg_count > int(count) for color, count in counts.items() if int(color) != int(bg))


def _try_place_one(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    side: str,
    want_truncated: bool,
) -> bool:
    h, w = map(int, grid.shape)
    for _ in range(PLACE_ATTEMPTS):
        rect_h = random.randint(*RECT_SIDE_RANGE)
        rect_w = random.randint(*RECT_SIDE_RANGE)
        area = int(rect_h * rect_w - 1)
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        bbox = (int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1))
        space = _outside_space(grid.shape, bbox, side)
        if want_truncated:
            if not (1 <= space < area):
                continue
        elif space < area:
            continue

        gap = _gap_for_side(bbox, side)
        ray = _ray_cells(grid.shape, gap, SIDE_DIR[side], area)
        if len(ray) != (space if want_truncated else area):
            continue

        footprint = _bbox_cells(bbox) + ray
        if _blocked(reserved, footprint):
            continue

        _paint_rectangle(grid, bbox, gap, color)
        _reserve(reserved, footprint)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    bg = random.randrange(COLOR_COUNT)
    grid = np.full((height, width), int(bg), dtype=int)

    colors = [c for c in range(COLOR_COUNT) if c != int(bg)]
    random.shuffle(colors)
    rect_count = random.randint(*RECT_COUNT_RANGE)
    colors = colors[:rect_count]

    sides = list(SIDES)
    random.shuffle(sides)
    while len(sides) < rect_count:
        sides.extend(random.sample(list(SIDES), k=len(SIDES)))
    sides = sides[:rect_count]

    truncated = [random.choice([False, True]) for _ in range(rect_count)]
    if rect_count >= 2 and len(set(truncated)) == 1:
        truncated[0] = False
        truncated[1] = True
        random.shuffle(truncated)

    reserved = np.zeros((height, width), dtype=bool)
    for color, side, want_truncated in zip(colors, sides, truncated):
        if not _try_place_one(grid, reserved, int(color), str(side), bool(want_truncated)):
            return None

    if not _dominant_bg(grid, bg):
        return None

    try:
        main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
