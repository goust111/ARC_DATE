from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

BASE_TEMPLATE = np.array(
    [
        [0, 1, 0],
        [1, 4, 0],
        [0, 1, 1],
    ],
    dtype=int,
)

GRID_SIZE_RANGE = (18, 24)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    yellow = np.argwhere(grid == YELLOW)
    assert yellow.shape == (1, 2), "expected exactly one yellow anchor in the input template"
    ar, ac = map(int, yellow[0])

    allowed = {BLUE, YELLOW}
    seen = {(ar, ac)}
    queue = deque([(ar, ac)])
    cells: list[tuple[int, int]] = []
    while queue:
        r, c = queue.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) not in allowed:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(grid[r, c])

    area = int(np.count_nonzero(patch))
    assert area in (4, 5, 6), "template must have 4-6 colored cells"
    assert patch.shape[0] <= 3 and patch.shape[1] <= 3, "template must fit inside 3x3"
    return patch, (ar - r0, ac - c0)


def _path_cells(red: tuple[int, int], green: tuple[int, int], *, horizontal_first: bool) -> list[tuple[int, int]]:
    rr, rc = red
    gr, gc = green
    assert rr != gr and rc != gc, "red and green must define an L path"

    out: list[tuple[int, int]] = []
    if horizontal_first:
        c_step = 1 if gc > rc else -1
        out.extend((rr, c) for c in range(rc, gc + c_step, c_step))
        r_step = 1 if gr > rr else -1
        out.extend((r, gc) for r in range(rr + r_step, gr, r_step))
    else:
        r_step = 1 if gr > rr else -1
        out.extend((r, rc) for r in range(rr, gr + r_step, r_step))
        c_step = 1 if gc > rc else -1
        out.extend((gr, c) for c in range(rc + c_step, gc, c_step))
    return out


def _stamp_patch(dst: np.ndarray, patch: np.ndarray, anchor_rc: tuple[int, int], path_rc: tuple[int, int]) -> None:
    pr, pc = anchor_rc
    base_r = path_rc[0] - pr
    base_c = path_rc[1] - pc
    ph, pw = patch.shape
    h, w = dst.shape
    assert 0 <= base_r and 0 <= base_c and base_r + ph <= h and base_c + pw <= w, "stamp leaves the grid"
    mask = patch != BG
    dst[base_r : base_r + ph, base_c : base_c + pw][mask] = patch[mask]


def _count_closed_black_regions(grid: np.ndarray) -> int:
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if int(grid[r0, c0]) != BG or vis[r0, c0]:
                continue
            queue = deque([(r0, c0)])
            vis[r0, c0] = True
            touches_border = False
            while queue:
                r, c = queue.popleft()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if vis[nr, nc] or int(grid[nr, nc]) != BG:
                        continue
                    vis[nr, nc] = True
                    queue.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _crop_square(grid: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    r, c = center
    r0 = max(0, r - radius)
    r1 = min(grid.shape[0] - 1, r + radius)
    c0 = max(0, c - radius)
    c1 = min(grid.shape[1] - 1, c + radius)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _simulate(
    grid: np.ndarray,
    *,
    horizontal_first: bool = True,
    stamp: bool = True,
) -> tuple[np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    reds = np.argwhere(g == RED)
    greens = np.argwhere(g == GREEN)
    assert reds.shape == (1, 2), "expected exactly one red marker"
    assert greens.shape == (1, 2), "expected exactly one green marker"
    red = tuple(map(int, reds[0]))
    green = tuple(map(int, greens[0]))

    patch, anchor_rc = _extract_template(g)
    out = g.copy()
    path = _path_cells(red, green, horizontal_first=horizontal_first)
    for r, c in path:
        out[r, c] = GREY
    if stamp:
        for idx, coord in enumerate(path):
            if idx % 3 == 0:
                _stamp_patch(out, patch, anchor_rc, coord)
    elbow = (red[0], green[1]) if horizontal_first else (green[0], red[1])
    return out, elbow, _count_closed_black_regions(out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the horizontal-first L path, stamp the corner sprite along it, then crop around the elbow by the resulting hole count.

    Input structure:
    - A mostly black grid contains exactly one red start pixel and one green end pixel in different rows and columns.
    - One corner also contains a small blue sprite template with a single yellow anchor pixel inside a 3x3-or-smaller bbox.
    - The input has no pre-drawn path; the template is the only multi-pixel foreground object.

    Rule:
    - From the red pixel, draw a grey L path horizontally to the green column and then vertically toward the green row, excluding the green pixel itself.
    - Starting from the red path cell, stamp a translated copy of the blue/yellow template onto every 3rd path cell (1st, 4th, 7th, ...), aligning the yellow anchor to that path cell.
    - Count the black regions fully enclosed by non-black pixels in the resulting full grid, and crop the final answer to the square centered at the L elbow with radius equal to that count.

    Key insight:
    - The red/green markers define the path geometry, the path geometry fixes the sprite stamp positions, and only the resulting stamped topology reveals the crop radius.
    """
    solved, elbow, holes = _simulate(np.asarray(grid, dtype=int), horizontal_first=True, stamp=True)
    return _crop_square(solved, elbow, holes)


def _flip_lr(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid)


def _flip_ud(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid)


def _build_top_right_grid(h: int, w: int, red: tuple[int, int], green: tuple[int, int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0:3, w - 3 : w] = BASE_TEMPLATE
    grid[red] = RED
    grid[green] = GREEN
    return grid


def _wrong_crop_matches(grid: np.ndarray, holes: int, solved: np.ndarray, elbow: tuple[int, int]) -> bool:
    actual = _crop_square(solved, elbow, holes)
    for alt in {max(0, holes - 1), holes + 1}:
        if alt == holes:
            continue
        if np.array_equal(actual, _crop_square(solved, elbow, alt)):
            return True
    return False


def _validate_generated_input(grid: np.ndarray, expected_holes: int) -> bool:
    try:
        solved, elbow, holes = _simulate(grid, horizontal_first=True, stamp=True)
        if holes != expected_holes or holes < 1:
            return False
        if expected_holes == 2 and holes == 1:
            return False
        if np.array_equal(main(grid), _crop_square(solved, elbow, holes)) is False:
            return False
        if _wrong_crop_matches(grid, holes, solved, elbow):
            return False

        vertical_solved, vertical_elbow, vertical_holes = _simulate(grid, horizontal_first=False, stamp=True)
        vertical_out = _crop_square(vertical_solved, vertical_elbow, vertical_holes)
        if np.array_equal(vertical_out, main(grid)):
            return False

        path_only, _, path_only_holes = _simulate(grid, horizontal_first=True, stamp=False)
        if path_only_holes >= holes:
            return False
        if np.array_equal(_crop_square(path_only, elbow, holes), main(grid)):
            return False
    except AssertionError:
        return False
    return True


def _generate_target_two() -> np.ndarray:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    red = (2, w - 4)
    green = (h - random.randint(2, 6), w - random.choice((9, 12)))
    grid = _build_top_right_grid(h, w, red, green)
    return grid


def _generate_target_one() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    red = (random.randint(4, 8), random.randint(w - 9, w - 5))
    green = (random.randint(h - 7, h - 2), random.randint(1, w - 10))
    if red[0] == green[0] or red[1] == green[1]:
        return None
    if abs(red[0] - green[0]) < 6 or abs(red[1] - green[1]) < 5:
        return None
    if abs(red[0] - green[0]) + abs(red[1] - green[1]) < 8:
        return None
    return _build_top_right_grid(h, w, red, green)


def _maybe_reflect(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.choice((False, True)):
        out = _flip_lr(out)
    if random.choice((False, True)):
        out = _flip_ud(out)
    return out.copy()


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        target_holes = random.choice((1, 2))
        if target_holes == 2:
            grid = _generate_target_two()
        else:
            grid = _generate_target_one()
            if grid is None:
                continue
        grid = _maybe_reflect(grid)
        if _validate_generated_input(grid, target_holes):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
