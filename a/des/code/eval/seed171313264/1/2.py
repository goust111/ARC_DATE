import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MAROON = 9
PATH_COLORS = (1, 2, 3, 6, 7, 8)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _is_path_color(color: int) -> bool:
    return color not in (BG, YELLOW, GREY, MAROON)


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_one(g: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(g == color)
    assert len(pts) == 1
    return int(pts[0, 0]), int(pts[0, 1])


def _record_cells(g: np.ndarray, end: tuple[int, int], count: int) -> list[tuple[int, int]]:
    h, w = g.shape
    r, c = end
    if r in (0, h - 1):
        candidates = []
        for dc in (1, -1):
            cells = []
            cc = c + dc
            while 0 <= cc < w and int(g[r, cc]) == BG and len(cells) < count:
                cells.append((r, cc))
                cc += dc
            candidates.append(cells)
        return max(candidates, key=len)[:count]
    candidates = []
    for dr in (1, -1):
        cells = []
        rr = r + dr
        while 0 <= rr < h and int(g[rr, c]) == BG and len(cells) < count:
            cells.append((rr, c))
            rr += dr
        candidates.append(cells)
    return max(candidates, key=len)[:count]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the colored orthogonal path from the yellow edge marker to the maroon
    edge marker. At grey switches, horizontal entry leaves vertically and
    vertical entry leaves horizontally. The visited path colors are written next
    to the maroon marker along the edge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    start = _find_one(g, YELLOW)
    end = _find_one(g, MAROON)

    start_neighbors = [(r, c) for r, c in _neighbors(*start, h, w) if _is_path_color(int(g[r, c]))]
    assert len(start_neighbors) == 1
    prev = start
    cur = start_neighbors[0]
    sequence: list[int] = []
    last_color = None

    for _ in range(h * w * 4):
        color = int(g[cur])
        if _is_path_color(color):
            if color != last_color:
                sequence.append(color)
                last_color = color
            if any((rr, cc) == end for rr, cc in _neighbors(*cur, h, w)):
                break

            next_cells = [
                (rr, cc)
                for rr, cc in _neighbors(*cur, h, w)
                if (rr, cc) != prev and (int(g[rr, cc]) == color or int(g[rr, cc]) == GREY)
            ]
            assert next_cells
            grey_next = [p for p in next_cells if int(g[p]) == GREY]
            nxt = grey_next[0] if grey_next else next_cells[0]
            prev, cur = cur, nxt
            continue

        assert color == GREY
        enter_dr = cur[0] - prev[0]
        enter_dc = cur[1] - prev[1]
        if enter_dc != 0:
            ordered = [(-1, 0), (1, 0)]
        else:
            ordered = [(0, 1), (0, -1)]
        options = []
        for dr, dc in ordered:
            rr, cc = cur[0] + dr, cur[1] + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) != prev and _is_path_color(int(g[rr, cc])):
                options.append((rr, cc))
        assert options
        prev, cur = cur, options[0]
    else:
        raise AssertionError("path trace did not terminate")

    out = g.copy()
    for (r, c), color in zip(_record_cells(g, end, len(sequence)), sequence):
        out[r, c] = color
    return out


def _draw_segment(g: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        g[r0, lo : hi + 1] = color
    else:
        lo, hi = sorted((r0, r1))
        g[lo : hi + 1, c0] = color


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(14, 20)
        w = random.randint(16, 22)
        r0 = random.randint(2, max(2, h // 3))
        c1 = random.randint(4, w - 9)
        r2 = random.randint(r0 + 3, h - 5)
        c2 = random.randint(c1 + 3, w - 6)
        end_r = h - 2
        if r2 >= end_r - 1:
            continue

        colors = random.sample(PATH_COLORS, 4)
        g = np.full((h, w), BG, dtype=int)
        g[r0, 0] = YELLOW
        g[h - 1, c2] = MAROON

        n1 = (r0, c1)
        n2 = (r2, c1)
        n3 = (r2, c2)
        end_path = (end_r, c2)
        _draw_segment(g, (r0, 1), (r0, c1 - 1), colors[0])
        _draw_segment(g, (r0 + 1, c1), (r2 - 1, c1), colors[1])
        _draw_segment(g, (r2, c1 + 1), (r2, c2 - 1), colors[2])
        _draw_segment(g, (r2 + 1, c2), end_path, colors[3])
        for node in (n1, n2, n3):
            g[node] = GREY
        if c2 + 4 >= w:
            continue
        return g
    raise RuntimeError("failed to generate seed171313264/1/2 input")
