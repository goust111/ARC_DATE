from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

ENDPOINT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _straight_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "paired endpoints must align on a straight ARC line"
    steps = max(abs(dr), abs(dc))
    sr = _sgn(dr)
    sc = _sgn(dc)
    return [(r0 + i * sr, c0 + i * sc) for i in range(steps + 1)]


def _black_components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            coords: list[tuple[int, int]] = []
            touches_edge = False
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    rr = cr + dr
                    cc2 = cc + dc
                    if 0 <= rr < h and 0 <= cc2 < w and mask[rr, cc2] and not seen[rr, cc2]:
                        seen[rr, cc2] = True
                        q.append((rr, cc2))
            comps.append((coords, touches_edge))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete color-matched path fragments into grey loops and fill loop interiors by relative area.

    Input structure:
    - The background is black.
    - Grey pixels form several short 4-connected path fragments.
    - Each fragment has two non-black, non-grey endpoint pixels, and every endpoint color occurs exactly twice in the grid.

    Rule:
    - Pair the two endpoints of each non-black, non-grey color.
    - Draw a grey horizontal, vertical, or 45-degree straight segment between each same-color pair, recoloring the endpoints grey.
    - Treat the completed grey paths as closed barriers and find black regions that do not touch the grid edge using 4-connected flood fill.
    - Fill the largest enclosed region red, the smallest enclosed region blue, and any remaining enclosed regions green; if there is only one enclosed region, fill it red.

    Key insight:
    Matching endpoint colors reveal which fragments should be joined; after the grey loops are completed, only the relative enclosed areas determine the fill colors.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    endpoint_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY)):
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 2, "each endpoint color must appear exactly twice"
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        endpoint_pairs.append((a, b))

    for a, b in endpoint_pairs:
        for r, c in _straight_line(a, b):
            out[r, c] = GREY

    enclosed = [coords for coords, touches in _black_components(out == BLACK) if not touches]
    if not enclosed:
        return out

    if len(enclosed) == 1:
        for r, c in enclosed[0]:
            out[r, c] = RED
        return out

    areas = [len(coords) for coords in enclosed]
    largest = max(range(len(enclosed)), key=lambda i: areas[i])
    smallest = min(range(len(enclosed)), key=lambda i: areas[i])
    assert areas[largest] != areas[smallest], "generated puzzle instances avoid tied enclosed areas"

    for i, coords in enumerate(enclosed):
        fill = GREEN
        if i == largest:
            fill = RED
        elif i == smallest:
            fill = BLUE
        for r, c in coords:
            out[r, c] = fill
    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 30
MAX_RETRIES = 500


def _loop_area(loop: dict[str, int | str | tuple[int, int]]) -> int:
    return (int(loop["h"]) - 2) * (int(loop["w"]) - 2)


def _paint_input_loop(grid: np.ndarray, loop: dict[str, int | str | tuple[int, int]]) -> None:
    r0 = int(loop["r0"])
    c0 = int(loop["c0"])
    h = int(loop["h"])
    w = int(loop["w"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    color_a, color_b = loop["colors"]  # type: ignore[index]

    if loop["orient"] == "vertical_sides":
        grid[r0, c0] = int(color_a)
        grid[r0, c1] = int(color_a)
        grid[r1, c0] = int(color_b)
        grid[r1, c1] = int(color_b)
        grid[r0 + 1 : r1, c0] = GREY
        grid[r0 + 1 : r1, c1] = GREY
    else:
        grid[r0, c0] = int(color_a)
        grid[r1, c0] = int(color_a)
        grid[r0, c1] = int(color_b)
        grid[r1, c1] = int(color_b)
        grid[r0, c0 + 1 : c1] = GREY
        grid[r1, c0 + 1 : c1] = GREY


def _paint_expected_loop(out: np.ndarray, loop: dict[str, int | str | tuple[int, int]], fill: int) -> None:
    r0 = int(loop["r0"])
    c0 = int(loop["c0"])
    h = int(loop["h"])
    w = int(loop["w"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    out[r0 + 1 : r1, c0 + 1 : c1] = int(fill)


def _expected_grid(shape: tuple[int, int], loops: list[dict[str, int | str | tuple[int, int]]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    if len(loops) == 1:
        fills = [RED]
    else:
        areas = [_loop_area(loop) for loop in loops]
        largest = max(range(len(loops)), key=lambda i: areas[i])
        smallest = min(range(len(loops)), key=lambda i: areas[i])
        fills = [GREEN] * len(loops)
        fills[largest] = RED
        fills[smallest] = BLUE

    for loop, fill in zip(loops, fills):
        _paint_expected_loop(out, loop, fill)
    return out


def _dimension_choices(h: int, w: int, used_areas: set[int]) -> list[tuple[str, int, int]]:
    choices: list[tuple[str, int, int]] = []
    for lh in range(3, min(8, h - 2) + 1):
        for lw in range(3, min(10, w - 2) + 1):
            area = (lh - 2) * (lw - 2)
            if area not in used_areas:
                choices.append(("vertical_sides", lh, lw))
    for lh in range(3, min(10, h - 2) + 1):
        for lw in range(3, min(8, w - 2) + 1):
            area = (lh - 2) * (lw - 2)
            if area not in used_areas:
                choices.append(("horizontal_sides", lh, lw))
    return choices


def _position_choices(occupied: np.ndarray, lh: int, lw: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    choices: list[tuple[int, int]] = []
    for r0 in range(1, h - lh):
        for c0 in range(1, w - lw):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + lh + 1)
            cc1 = min(w, c0 + lw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                choices.append((r0, c0))
    return choices


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, lh: int, lw: int) -> None:
    h, w = occupied.shape
    occupied[max(0, r0 - 1) : min(h, r0 + lh + 1), max(0, c0 - 1) : min(w, c0 + lw + 1)] = True


def _gen_attempt() -> np.ndarray | None:
    loop_count = random.choice([1, 2, 3])
    low = GRID_MIN + 3 * (loop_count - 1)
    h = random.randint(low, GRID_MAX)
    w = random.randint(low, GRID_MAX)

    endpoint_colors = random.sample(ENDPOINT_COLORS, 2 * loop_count)
    occupied = np.zeros((h, w), dtype=bool)
    loops: list[dict[str, int | str | tuple[int, int]]] = []
    used_areas: set[int] = set()

    for i in range(loop_count):
        dims = _dimension_choices(h, w, used_areas)
        random.shuffle(dims)
        placed = False
        for orient, lh, lw in dims[:80]:
            positions = _position_choices(occupied, lh, lw)
            if not positions:
                continue
            r0, c0 = random.choice(positions)
            loop = {
                "r0": r0,
                "c0": c0,
                "h": lh,
                "w": lw,
                "orient": orient,
                "colors": (endpoint_colors[2 * i], endpoint_colors[2 * i + 1]),
            }
            loops.append(loop)
            used_areas.add(_loop_area(loop))
            _mark_occupied(occupied, r0, c0, lh, lw)
            placed = True
            break
        if not placed:
            return None

    inp = np.full((h, w), BLACK, dtype=int)
    for loop in loops:
        _paint_input_loop(inp, loop)

    counts_ok = True
    for color in ENDPOINT_COLORS:
        count = int(np.count_nonzero(inp == color))
        counts_ok = counts_ok and count in (0, 2)
    if not counts_ok:
        return None

    expected = _expected_grid((h, w), loops)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-loop puzzle instance")
