from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, in_bounds


WALL_COLOR = Color.GREY
WALL_MARK = Color.YELLOW
PATH_COLOR = Color.GREEN
MIN_SIDE = 4
NOISE_COLORS = (Color.BLUE, Color.RED, Color.ORANGE, Color.MAROON)
GEN_ATTEMPTS = 400
NOISE_REPAIR_PASSES = 80


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return int(r) in (0, h - 1) or int(c) in (0, w - 1)


def _right_runs(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    runs = np.ones((h, w), dtype=int)
    for r in range(h):
        for c in range(w - 2, -1, -1):
            if int(grid[r, c]) == int(grid[r, c + 1]):
                runs[r, c] = runs[r, c + 1] + 1
    return runs


def _maximal_uniform_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    runs = _right_runs(grid)
    rects: set[tuple[int, int, int, int, int]] = set()

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            max_w = w - c0
            for r1 in range(r0, h):
                if int(grid[r1, c0]) != color:
                    break
                max_w = min(max_w, int(runs[r1, c0]))
                if r1 - r0 + 1 >= MIN_SIDE and max_w >= MIN_SIDE:
                    rects.add((r0, c0, r1, c0 + max_w - 1, color))

    maximal: list[tuple[int, int, int, int, int]] = []
    for rect in rects:
        r0, c0, r1, c1, color = rect
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        contained = False
        for other in rects:
            if other == rect:
                continue
            or0, oc0, or1, oc1, other_color = other
            if other_color != color:
                continue
            other_area = (or1 - or0 + 1) * (oc1 - oc0 + 1)
            if other_area > area and or0 <= r0 and oc0 <= c0 and r1 <= or1 and c1 <= oc1:
                contained = True
                break
        if not contained:
            maximal.append(rect)

    return sorted(maximal)


def _wall_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    walls: list[tuple[int, int, int, int]] = []
    for r0, c0, r1, c1, color in _maximal_uniform_rectangles(grid):
        if color != WALL_COLOR:
            continue
        if r1 - r0 + 1 < MIN_SIDE or c1 - c0 + 1 < MIN_SIDE:
            continue
        walls.append((r0, c0, r1, c1))
    return walls


def _start_bar(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == PATH_COLOR)]
    assert len(pts) == 2, "expected exactly two green start pixels"
    (r0, c0), (r1, c1) = pts
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"
    border_pts = [p for p in pts if _is_border(grid.shape, p[0], p[1])]
    assert len(border_pts) == 1, "exactly one start pixel must touch the border"
    border = border_pts[0]
    inner = pts[0] if pts[1] == border else pts[1]
    dr = int(inner[0] - border[0])
    dc = int(inner[1] - border[1])
    return border, inner, (dr, dc)


def _trace_path(
    wall_mask: np.ndarray,
    start_info: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    border, inner, (dr, dc) = start_info
    path = [border, inner]
    turns = 0
    r, c = inner
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path should not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(wall_mask.shape, nr, nc):
            break

        if not bool(wall_mask[nr, nc]):
            r, c = int(nr), int(nc)
            path.append((r, c))
            if _is_border(wall_mask.shape, r, c) and (r, c) != border:
                break
            continue

        choices: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if in_bounds(wall_mask.shape, tr, tc) and not bool(wall_mask[tr, tc]):
                choices.append((int(tdr), int(tdc)))
        assert len(choices) == 1, "each wall collision must force a unique turn"
        dr, dc = choices[0]
        turns += 1

    return path, turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the hidden grey wall rectangles, recolor them yellow, and trace the green corridor path between them.

    Input structure:
    - A noisy 2D grid contains 2-4 large solid grey rectangles and no pre-existing yellow pixels.
    - Exactly one green 2-pixel start bar touches the border with one pixel on the edge and one pixel just inside.
    - The rectangles are separated from each other and from the border, while the rest of the grid is multicolor noise.

    Rule:
    - Find every maximal solid same-color rectangle; keep only grey rectangles whose height and width are both at least 4.
    - Recolor those detected wall rectangles to yellow.
    - From the border start pixel, move through the other green pixel into the grid, continue straight through non-wall cells, and paint every visited cell green.
    - Whenever the next cell is a yellow wall, turn 90 degrees to the unique perpendicular non-wall neighbor and continue until the path reaches the border.

    Key insight:
    - The corridor is not drawn explicitly in the input: the large hidden rectangles define the walls, so rectangle detection must happen before the deterministic collision-and-turn walk can be recovered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    walls = _wall_rectangles(g)
    assert walls, "expected at least one hidden wall rectangle"

    out = g.copy()
    wall_mask = np.zeros(g.shape, dtype=bool)
    for r0, c0, r1, c1 in walls:
        wall_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        out[r0 : r1 + 1, c0 : c1 + 1] = WALL_MARK

    start_info = _start_bar(g)
    path, _turns = _trace_path(wall_mask, start_info)
    for r, c in path:
        if not wall_mask[r, c]:
            out[r, c] = PATH_COLOR
    return out


def _rect_mask(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _rects_are_valid(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> bool:
    h, w = shape
    mask = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        if not (1 <= r0 <= r1 <= h - 2 and 1 <= c0 <= c1 <= w - 2):
            return False
        if r1 - r0 + 1 < MIN_SIDE or c1 - c0 + 1 < MIN_SIDE:
            return False
        mask[r0 : r1 + 1, c0 : c1 + 1] += 1
    if np.any(mask > 1):
        return False

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (a1 < b0 or b1 < a0)

    for i, (r0, c0, r1, c1) in enumerate(rects):
        for j, (or0, oc0, or1, oc1) in enumerate(rects):
            if i >= j:
                continue
            if overlaps(r0, r1, or0, or1) and overlaps(c0, c1, oc0, oc1):
                return False
            if overlaps(r0, r1, or0, or1) and (c1 + 1 == oc0 or oc1 + 1 == c0):
                return False
            if overlaps(c0, c1, oc0, oc1) and (r1 + 1 == or0 or or1 + 1 == r0):
                return False
    return True


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    out = np.rot90(out, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _repair_noise(grid: np.ndarray, wall_mask: np.ndarray, fixed_mask: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(NOISE_REPAIR_PASSES):
        changed = False
        for r in range(h - MIN_SIDE + 1):
            for c in range(w - MIN_SIDE + 1):
                block = grid[r : r + MIN_SIDE, c : c + MIN_SIDE]
                if not np.all(block == int(block[0, 0])):
                    continue
                if np.any(wall_mask[r : r + MIN_SIDE, c : c + MIN_SIDE]):
                    continue
                coords = [
                    (rr, cc)
                    for rr in range(r, r + MIN_SIDE)
                    for cc in range(c, c + MIN_SIDE)
                    if not fixed_mask[rr, cc]
                ]
                if not coords:
                    continue
                for rr, cc in coords:
                    grid[rr, cc] = random.choice(NOISE_COLORS)
                changed = True
        if not changed:
            return True
    return False


def _build_canonical_layout(count: int) -> np.ndarray | None:
    assert 2 <= count <= 4

    if count == 2:
        h = random.randint(18, 24)
        w = random.randint(18, 24)
    elif count == 3:
        h = random.randint(20, 26)
        w = random.randint(20, 26)
    else:
        h = random.randint(22, 28)
        w = random.randint(22, 28)

    h0 = random.randint(4, 6)
    w1 = random.randint(4, 6)
    r1_max = h - (6 if count == 2 else 10)
    if h0 + 1 > r1_max:
        return None
    r1 = random.randint(h0 + 1, r1_max)

    c1_min = w1 + 3
    c1_max = w - 7
    if c1_min > c1_max:
        return None
    c1 = random.randint(c1_min, c1_max)

    rects: list[tuple[int, int, int, int]] = []

    if count == 4:
        w3 = random.randint(4, 6)
        c3_min = c1 + 6
        c3_max = w - w3 - 3
        if c3_min > c3_max:
            return None
        c3 = random.randint(c3_min, c3_max)
        w0_max = min(7, c3 - c1 - 2)
    else:
        c3 = -1
        w3 = 0
        w0_max = min(10, w - c1 - 2)
    if w0_max < 4:
        return None
    w0 = random.randint(4, w0_max)
    rects.append((r1 - h0, c1, r1 - 1, c1 + w0 - 1))

    if count == 2:
        bottom1 = random.randint(r1 + 3, h - 2)
        rects.append((r1, c1 - w1, bottom1, c1 - 1))
    else:
        h2 = random.randint(4, 6)
        r2_min = r1 + 4
        r2_max = h - h2 - 2
        if r2_min > r2_max:
            return None
        r2 = random.randint(r2_min, r2_max)

        rects.append((r1, c1 - w1, r2, c1 - 1))

        if count == 3:
            w2_max = min(10, w - c1 - 2)
            if w2_max < 4:
                return None
            w2 = random.randint(4, w2_max)
            rects.append((r2 + 1, c1, r2 + h2, c1 + w2 - 1))
        else:
            rects.append((r2 + 1, c1, r2 + h2, c3, ))
            top3 = random.randint(1, r2 - 3)
            rects.append((top3, c3 + 1, r2, c3 + w3))

    if not _rects_are_valid((h, w), rects):
        return None

    grid = np.random.choice(NOISE_COLORS, size=(h, w)).astype(int)
    wall_mask = _rect_mask((h, w), rects)
    grid[wall_mask] = WALL_COLOR

    if count == 2:
        bar = ((r1, w - 1), (r1, w - 2))
    else:
        bar = ((r1, w - 1), (r1, w - 2))
    start_mask = np.zeros((h, w), dtype=bool)
    for r, c in bar:
        if wall_mask[r, c]:
            return None
        grid[r, c] = PATH_COLOR
        start_mask[r, c] = True

    fixed_mask = wall_mask | start_mask
    if not _repair_noise(grid, wall_mask, fixed_mask):
        return None
    return grid


def _validate_generated(grid: np.ndarray, expected_count: int) -> bool:
    try:
        walls = _wall_rectangles(grid)
        if len(walls) != expected_count:
            return False
        wall_mask = _rect_mask(grid.shape, walls)
        start_info = _start_bar(grid)
        path, turns = _trace_path(wall_mask, start_info)
        if turns != expected_count - 1:
            return False
        if len(path) < 10:
            return False
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == WALL_MARK) != int(np.count_nonzero(wall_mask)):
        return False
    if np.count_nonzero(out == PATH_COLOR) <= 2:
        return False
    return True


def _gen_for_count(count: int, attempts: int = 24) -> np.ndarray | None:
    for _ in range(attempts):
        grid = _build_canonical_layout(count)
        if grid is None:
            continue
        grid = _random_symmetry(grid)
        if _validate_generated(grid, count):
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    count = random.choices((2, 3, 4), weights=(3, 3, 4))[0]
    return _gen_for_count(count)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-rectangle corridor grid")
