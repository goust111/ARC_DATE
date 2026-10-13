import random

import numpy as np


BLACK = 0
ORANGE = 7
INPUT_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

GEN_HEIGHT_RANGE = (10, 31)
GEN_WIDTH_RANGE = (10, 31)
GEN_MIN_PAIRS = 2
GEN_MAX_PAIRS = 6
GEN_MARGIN = 1
GEN_RETRIES = 400
PAIR_PLACEMENT_TRIES = 200


def _colors_in_scan_order(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in grid.flat:
        color = int(value)
        if color != BLACK and color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _paint_path(
    out: np.ndarray,
    origin: tuple[int, int],
    steps: int,
    axis: int,
    direction: int,
    color: int,
) -> None:
    keep_count = steps // 2
    for idx in range(steps):
        r, c = origin
        if axis == 0:
            r += direction * idx
        else:
            c += direction * idx
        out[r, c] = color if idx < keep_count else ORANGE


def _final_coord(origin: tuple[int, int], steps: int, axis: int, direction: int) -> tuple[int, int]:
    r, c = origin
    if axis == 0:
        r += direction * steps
    else:
        c += direction * steps
    return (r, c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each aligned color pair toward the center and leave orange-tipped trails behind.

    Input structure:
    - The background is black.
    - Every non-black color appears in exactly two isolated single pixels.
    - Each same-color pair lies on one shared row or one shared column, with only black cells
      between the two pixels, and their straight corridors do not interfere with other pairs.

    Rule:
    - For each same-color pair, move the two pixels toward one another along their shared row
      or column until they become a central adjacent pair.
    - This requires the endpoint distance to be odd, so both pixels travel the same number of
      steps; along each travelled path, the half nearest the starting pixel stays the original
      color, while the half nearest the center becomes orange.
    - If an individual travelled path has odd length, its middle trail cell is orange.
      The two final central pixels remain in the original color.

    Key insight:
    - Pairing is determined purely by color, while alignment determines the movement axis; the
      output is a symmetric center-merge whose trails are split by distance from each origin.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full_like(g, BLACK)

    for color in _colors_in_scan_order(g):
        points = np.argwhere(g == color)
        assert points.shape == (2, 2), f"color {color} must appear exactly twice"

        p0 = tuple(map(int, points[0]))
        p1 = tuple(map(int, points[1]))

        same_row = p0[0] == p1[0]
        same_col = p0[1] == p1[1]
        assert same_row ^ same_col, f"color {color} must align in one axis"

        if same_row:
            axis = 1
            left, right = sorted((p0, p1), key=lambda p: p[1])
            distance = right[1] - left[1]
            assert bool((g[left[0], left[1] + 1 : right[1]] == BLACK).all())
        else:
            axis = 0
            left, right = sorted((p0, p1), key=lambda p: p[0])
            distance = right[0] - left[0]
            assert bool((g[left[0] + 1 : right[0], left[1]] == BLACK).all())

        assert distance >= 1, "paired pixels must be distinct"
        assert distance % 2 == 1, "endpoint distance must be odd for a central adjacent pair"

        steps = (distance - 1) // 2
        _paint_path(out, left, steps, axis=axis, direction=1, color=int(color))
        _paint_path(out, right, steps, axis=axis, direction=-1, color=int(color))

        fr0, fc0 = _final_coord(left, steps, axis=axis, direction=1)
        fr1, fc1 = _final_coord(right, steps, axis=axis, direction=-1)
        out[fr0, fc0] = int(color)
        out[fr1, fc1] = int(color)

    return out


def _corridor_box(
    start: tuple[int, int],
    end: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[slice, slice]:
    r0 = max(0, min(start[0], end[0]) - 1)
    r1 = min(shape[0], max(start[0], end[0]) + 2)
    c0 = max(0, min(start[1], end[1]) - 1)
    c1 = min(shape[1], max(start[1], end[1]) + 2)
    return slice(r0, r1), slice(c0, c1)


def _try_place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    orientation: str,
    steps_parity: int,
) -> bool:
    h, w = grid.shape
    span = w if orientation == "h" else h
    max_steps = (span - 2 * GEN_MARGIN - 2) // 2
    if max_steps < 1:
        return False

    choices = [s for s in range(1, max_steps + 1) if s % 2 == steps_parity]
    if not choices:
        choices = list(range(1, max_steps + 1))

    for _ in range(PAIR_PLACEMENT_TRIES):
        steps = random.choice(choices)
        distance = 2 * steps + 1
        length = distance + 1

        if orientation == "h":
            if w - length <= GEN_MARGIN:
                continue
            row = random.randrange(GEN_MARGIN, h - GEN_MARGIN)
            col0 = random.randrange(GEN_MARGIN, w - length)
            start = (row, col0)
            end = (row, col0 + distance)
        else:
            if h - length <= GEN_MARGIN:
                continue
            col = random.randrange(GEN_MARGIN, w - GEN_MARGIN)
            row0 = random.randrange(GEN_MARGIN, h - length)
            start = (row0, col)
            end = (row0 + distance, col)

        rs, cs = _corridor_box(start, end, grid.shape)
        if blocked[rs, cs].any():
            continue

        blocked[rs, cs] = True
        grid[start] = color
        grid[end] = color
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(*GEN_HEIGHT_RANGE)
    w = random.randrange(*GEN_WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_pairs_by_area = max(GEN_MIN_PAIRS, min(GEN_MAX_PAIRS, (h * w) // 55 + 1, len(INPUT_COLORS)))
    pair_count = random.randint(GEN_MIN_PAIRS, max_pairs_by_area)
    colors = random.sample(list(INPUT_COLORS), pair_count)

    orientations = [random.choice(("h", "v")) for _ in range(pair_count)]
    parities = [random.choice((0, 1)) for _ in range(pair_count)]
    if pair_count >= 2 and len(set(orientations)) == 1:
        orientations[-1] = "v" if orientations[0] == "h" else "h"
    if pair_count >= 2 and len(set(parities)) == 1:
        parities[-1] = 1 - parities[0]

    order = list(range(pair_count))
    random.shuffle(order)
    order.sort(key=lambda idx: (orientations[idx] == "h", parities[idx]), reverse=True)

    for idx in order:
        color = colors[idx]
        orientation = orientations[idx]
        parity = parities[idx]

        if _try_place_pair(grid, blocked, color, orientation, parity):
            continue
        if _try_place_pair(grid, blocked, color, "v" if orientation == "h" else "h", parity):
            continue
        if _try_place_pair(grid, blocked, color, orientation, 1 - parity):
            continue
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
