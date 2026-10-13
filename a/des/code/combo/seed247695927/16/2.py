import random
from typing import Iterable

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _diag_segment_slope_pos(grid: np.ndarray, r: int, c: int, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0 = r, c
    while r0 - 1 >= 0 and c0 - 1 >= 0 and int(grid[r0 - 1, c0 - 1]) != bg:
        r0 -= 1
        c0 -= 1
    cells: list[tuple[int, int]] = []
    rr, cc = r0, c0
    while rr < h and cc < w and int(grid[rr, cc]) != bg:
        cells.append((rr, cc))
        rr += 1
        cc += 1
    return cells


def _diag_segment_slope_neg(grid: np.ndarray, r: int, c: int, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0 = r, c
    while r0 - 1 >= 0 and c0 + 1 < w and int(grid[r0 - 1, c0 + 1]) != bg:
        r0 -= 1
        c0 += 1
    cells: list[tuple[int, int]] = []
    rr, cc = r0, c0
    while rr < h and cc >= 0 and int(grid[rr, cc]) != bg:
        cells.append((rr, cc))
        rr += 1
        cc -= 1
    return cells


def _rectangle_footprint(grid: np.ndarray, color: int, bg: int) -> set[tuple[int, int]] | None:
    coords = np.argwhere(grid == color)
    if len(coords) == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1 or c0 == c1:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(patch == bg):
        return None
    if not np.any(patch != color):
        return None
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _diagonal_footprint(grid: np.ndarray, color: int, bg: int) -> set[tuple[int, int]] | None:
    coords = np.argwhere(grid == color)
    if len(coords) < 2:
        return None

    diffs = coords[:, 0] - coords[:, 1]
    if np.all(diffs == diffs[0]):
        cells = _diag_segment_slope_pos(grid, int(coords[0, 0]), int(coords[0, 1]), bg)
    else:
        sums = coords[:, 0] + coords[:, 1]
        if not np.all(sums == sums[0]):
            return None
        cells = _diag_segment_slope_neg(grid, int(coords[0, 0]), int(coords[0, 1]), bg)

    if len(cells) < 3:
        return None
    if not all((int(r), int(c)) in {(int(x), int(y)) for x, y in coords} for r, c in cells if int(grid[r, c]) == color):
        return None
    if not any(int(grid[r, c]) != color for r, c in cells):
        return None
    return set(cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete partially occluded simple foreground shapes by repainting hidden footprint cells.

    Rule:
    - For a foreground color whose full rectangular bounding box is entirely foreground
      and contains other colors, repaint that whole rectangle with the color.
    - Otherwise, if all visible cells of the color lie on a 45-degree diagonal and the
      missing diagonal cells are foreground rather than background, repaint that whole
      diagonal segment with the color.
    - Leave all other cells unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in [int(c) for c in np.unique(g) if int(c) != bg]:
        footprint = _rectangle_footprint(g, color, bg)
        if footprint is None:
            footprint = _diagonal_footprint(g, color, bg)
        if footprint is None:
            continue
        for r, c in footprint:
            out[r, c] = color

    return out


_GEN_ATTEMPTS = 300
_GRID_SIDE_RANGE = (12, 19)
_RECT_H_RANGE = (3, 6)
_RECT_W_RANGE = (4, 7)
_DIAG_LEN_RANGE = (4, 8)
_DISTRACTOR_COUNT_RANGE = (0, 3)


def _rng_color_pool(bg: int) -> list[int]:
    pool = list(range(10))
    pool.remove(bg)
    random.shuffle(pool)
    return pool


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _find_free_bbox(
    h: int,
    w: int,
    bh: int,
    bw: int,
    used: list[tuple[int, int, int, int]],
    pad: int = 1,
    tries: int = 80,
) -> tuple[int, int, int, int] | None:
    for _ in range(tries):
        r0 = random.randint(1, h - bh - 1)
        c0 = random.randint(1, w - bw - 1)
        box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        if any(_rects_overlap(box, other, pad=pad) for other in used):
            continue
        return box
    return None


def _paint_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _paint_diag(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _diag_cells(box: tuple[int, int, int, int], slope: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    length = r1 - r0 + 1
    if slope == 1:
        return [(r0 + i, c0 + i) for i in range(length)]
    return [(r0 + i, c1 - i) for i in range(length)]


def _occlude_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], occ_color: int) -> int:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))
    occluded = 0

    if inner_rows and inner_cols:
        oh = random.randint(1, max(1, h - 2))
        ow = random.randint(1, max(1, w - 2))
        rr = random.randint(r0 + 1, r1 - oh)
        cc = random.randint(c0 + 1, c1 - ow)
        grid[rr : rr + oh, cc : cc + ow] = occ_color
        occluded += oh * ow

    edge_choices = []
    if h >= 3:
        edge_choices.extend([(r0, c) for c in range(c0 + 1, c1)])
        edge_choices.extend([(r1, c) for c in range(c0 + 1, c1)])
    if w >= 3:
        edge_choices.extend([(r, c0) for r in range(r0 + 1, r1)])
        edge_choices.extend([(r, c1) for r in range(r0 + 1, r1)])
    random.shuffle(edge_choices)
    for r, c in edge_choices[: random.randint(0, min(3, len(edge_choices)))]:
        grid[r, c] = occ_color
        occluded += 1
    return occluded


def _occlude_diagonal(grid: np.ndarray, cells: list[tuple[int, int]], occ_colors: list[int]) -> int:
    choices = list(range(1, len(cells) - 1))
    random.shuffle(choices)
    n_occ = random.randint(1, min(2, len(choices)))
    for i in range(n_occ):
        r, c = cells[choices[i]]
        grid[r, c] = occ_colors[i]
    return n_occ


def _place_distractor(grid: np.ndarray, used: list[tuple[int, int, int, int]], color: int) -> bool:
    h, w = grid.shape
    if random.random() < 0.5:
        bh = random.randint(2, 4)
        bw = random.randint(2, 4)
        if bh == bw == 2:
            bw = 3
        box = _find_free_bbox(h, w, bh, bw, used, pad=1, tries=30)
        if box is None:
            return False
        _paint_rect(grid, box, color)
        used.append(box)
        return True

    bh = bw = random.randint(3, 4)
    box = _find_free_bbox(h, w, bh, bw, used, pad=1, tries=30)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1), (r0 + 1, c0 + 1)]
    for r, c in cells:
        grid[r, c] = color
    used.append(box)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_SIDE_RANGE)
    w = random.randint(*_GRID_SIDE_RANGE)
    bg = random.randint(0, 9)
    grid = np.full((h, w), bg, dtype=int)
    color_pool = _rng_color_pool(bg)
    if len(color_pool) < 6:
        return None

    used: list[tuple[int, int, int, int]] = []

    rect_color = color_pool.pop()
    rect_occ_color = color_pool.pop()
    rect_h = random.randint(*_RECT_H_RANGE)
    rect_w = random.randint(*_RECT_W_RANGE)
    rect_box = _find_free_bbox(h, w, rect_h, rect_w, used, pad=1)
    if rect_box is None:
        return None
    _paint_rect(grid, rect_box, rect_color)
    rect_occluded = _occlude_rectangle(grid, rect_box, rect_occ_color)
    if rect_occluded == 0:
        return None
    used.append(rect_box)

    diag_color = color_pool.pop()
    diag_len = random.randint(*_DIAG_LEN_RANGE)
    diag_box = _find_free_bbox(h, w, diag_len, diag_len, used, pad=1)
    if diag_box is None:
        return None
    slope = random.choice((1, -1))
    diag_cells = _diag_cells(diag_box, slope)
    _paint_diag(grid, diag_cells, diag_color)
    diag_occ_colors = [color_pool.pop(), color_pool.pop()]
    diag_occluded = _occlude_diagonal(grid, diag_cells, diag_occ_colors)
    if diag_occluded == 0:
        return None
    used.append(diag_box)

    distractor_count = random.randint(*_DISTRACTOR_COUNT_RANGE)
    for _ in range(distractor_count):
        if not color_pool:
            break
        _place_distractor(grid, used, color_pool.pop())

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    rect_coords = np.argwhere(grid == rect_color)
    if len(rect_coords) == 0:
        return None
    rr0, rc0, rr1, rc1 = _bbox(rect_coords)
    if (rr0, rc0, rr1, rc1) != rect_box:
        return None

    diag_coords = np.argwhere(grid == diag_color)
    if len(diag_coords) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_ATTEMPTS} attempts")
