from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _source_colors(grid: np.ndarray) -> set[int]:
    return set(map(int, np.unique(grid))) - {BG, GREY}


def _network_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == GREY) | np.isin(grid, list(_source_colors(grid)))


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    return sum(1 for rr, cc in _neighbors4(r, c, h, w) if mask[rr, cc])


def _sources(grid: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    out = []
    for color in sorted(_source_colors(grid)):
        pts = np.argwhere(grid == color)
        assert len(pts) == 1, "each source color must occur once"
        r, c = map(int, pts[0])
        assert _degree(mask, r, c) == 1, "source must sit on a network endpoint"
        out.append((r, c, color))
    return out


def _source_limit(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    nbrs = [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if mask[rr, cc]]
    assert len(nbrs) == 1
    prev = (r, c)
    cur = nbrs[0]
    dist = 1
    while _degree(mask, cur[0], cur[1]) == 2:
        nxts = [(rr, cc) for rr, cc in _neighbors4(cur[0], cur[1], h, w) if mask[rr, cc] and (rr, cc) != prev]
        if len(nxts) != 1:
            break
        prev, cur = cur, nxts[0]
        dist += 1
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each endpoint source through the grey line network for its first-segment length.
    Larger original propagation limits win conflicts; equal limits produce yellow.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = _network_mask(g)
    arrivals: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for sr, sc, color in _sources(g, mask):
        limit = _source_limit(mask, sr, sc)
        q = deque([(sr, sc, 0)])
        seen = {(sr, sc)}
        while q:
            r, c, d = q.popleft()
            arrivals.setdefault((r, c), []).append((limit, color))
            if d == limit:
                continue
            for nr, nc in _neighbors4(r, c, h, w):
                if mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc, d + 1))

    out = g.copy()
    for r, c in zip(*np.where(mask), strict=False):
        vals = arrivals.get((int(r), int(c)), [])
        if not vals:
            out[r, c] = GREY
            continue
        best = max(limit for limit, _color in vals)
        colors = {color for limit, color in vals if limit == best}
        out[r, c] = next(iter(colors)) if len(colors) == 1 else YELLOW
    return out


def _draw_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = GREY
    else:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = GREY


def _endpoints(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = []
    for r, c in zip(*np.where(mask), strict=False):
        if _degree(mask, int(r), int(c)) == 1:
            pts.append((int(r), int(c)))
    return pts


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 25)
    w = random.randint(15, 27)
    grid = np.zeros((h, w), dtype=int)
    cr = random.randint(5, h - 6)
    cc = random.randint(5, w - 6)

    up = random.randint(3, min(7, cr - 1))
    down = random.randint(3, min(7, h - cr - 2))
    left = random.randint(3, min(7, cc - 1))
    right = random.randint(3, min(7, w - cc - 2))
    _draw_line(grid, (cr - up, cc), (cr + down, cc))
    _draw_line(grid, (cr, cc - left), (cr, cc + right))

    if random.random() < 0.75:
        br = random.randint(cr - up + 1, cr - 1)
        span = random.randint(2, min(5, w - cc - 2))
        _draw_line(grid, (br, cc), (br, cc + span))
    if random.random() < 0.75:
        bc = random.randint(cc + 1, cc + right - 1)
        span = random.randint(2, min(5, h - cr - 2))
        _draw_line(grid, (cr, bc), (cr + span, bc))

    mask = grid == GREY
    ends = _endpoints(mask)
    if len(ends) < 2:
        return None
    random.shuffle(ends)
    colors = random.sample([1, 2, 3, 6, 7, 8, 9], random.randint(2, min(4, len(ends))))
    for (r, c), color in zip(ends, colors, strict=False):
        grid[r, c] = color

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate propagation-network input")
