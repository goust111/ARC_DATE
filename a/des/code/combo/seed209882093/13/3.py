from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

L_COLORS = (4, 5, 6, 7, 8, 9)
RELATIONS = ("equal", "horizontal_longer", "vertical_longer")
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))  # (vdir, hdir)

GEN_ATTEMPTS = 300
PLACE_ATTEMPTS = 140
MARKER_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append((color, pts))
    return comps


def _parse_l_component(pts: list[tuple[int, int]]) -> dict[str, int] | None:
    pset = {(int(r), int(c)) for r, c in pts}
    if len(pset) < 3:
        return None

    candidates: list[dict[str, int]] = []
    for r0, c0 in pset:
        if not all(r == r0 or c == c0 for r, c in pset):
            continue

        row_cols = sorted(c for r, c in pset if r == r0)
        col_rows = sorted(r for r, c in pset if c == c0)
        if len(row_cols) < 2 or len(col_rows) < 2:
            continue
        if row_cols != list(range(row_cols[0], row_cols[-1] + 1)):
            continue
        if col_rows != list(range(col_rows[0], col_rows[-1] + 1)):
            continue
        if len(pset) != len(row_cols) + len(col_rows) - 1:
            continue

        cmin, cmax = row_cols[0], row_cols[-1]
        rmin, rmax = col_rows[0], col_rows[-1]

        if c0 == cmin and cmax > c0:
            hdir = 1
        elif c0 == cmax and cmin < c0:
            hdir = -1
        else:
            continue

        if r0 == rmin and rmax > r0:
            vdir = 1
        elif r0 == rmax and rmin < r0:
            vdir = -1
        else:
            continue

        candidates.append(
            {
                "corner_r": int(r0),
                "corner_c": int(c0),
                "hdir": int(hdir),
                "vdir": int(vdir),
                "hlen": int(len(row_cols)),
                "vlen": int(len(col_rows)),
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _fill_color(hlen: int, vlen: int) -> int:
    if hlen == vlen:
        return GREEN
    if hlen > vlen:
        return RED
    return BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find L-shaped monochrome objects and fill a square inside each opening.

    The square side length is the shorter arm length. Its color depends on the
    arm-length comparison: equal -> green, horizontal longer -> red, vertical
    longer -> blue. Existing non-black pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for _color, pts in _components_by_color(g):
        info = _parse_l_component(pts)
        if info is None:
            continue

        s = min(int(info["hlen"]), int(info["vlen"]))
        fill = _fill_color(int(info["hlen"]), int(info["vlen"]))
        r0 = int(info["corner_r"])
        c0 = int(info["corner_c"])
        vdir = int(info["vdir"])
        hdir = int(info["hdir"])

        for dr in range(s):
            rr = int(r0 + vdir * dr)
            for dc in range(s):
                cc = int(c0 + hdir * dc)
                if int(g[rr, cc]) == BLACK:
                    out[rr, cc] = fill

    return out


def _l_cells(corner: tuple[int, int], hlen: int, vlen: int, hdir: int, vdir: int) -> set[tuple[int, int]]:
    r0, c0 = corner
    cells = {(int(r0), int(c0 + hdir * i)) for i in range(hlen)}
    cells.update((int(r0 + vdir * i), int(c0)) for i in range(vlen))
    return cells


def _square_cells(corner: tuple[int, int], size: int, hdir: int, vdir: int) -> set[tuple[int, int]]:
    r0, c0 = corner
    return {
        (int(r0 + vdir * dr), int(c0 + hdir * dc))
        for dr in range(size)
        for dc in range(size)
    }


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            rr = int(r + dr)
            if not (0 <= rr < h):
                continue
            for dc in (-1, 0, 1):
                cc = int(c + dc)
                if 0 <= cc < w:
                    blocked[rr, cc] = True


def _diverse_choices(options: tuple, n: int, rng: random.Random) -> list:
    base = list(options)
    rng.shuffle(base)
    result = base[: min(n, len(base))]
    while len(result) < n:
        result.append(rng.choice(options))
    rng.shuffle(result)
    return result


def _try_place_l(
    grid: np.ndarray,
    blocked: np.ndarray,
    relation: str,
    direction: tuple[int, int],
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    vdir, hdir = direction
    s_cap = min(6, h - 1, w - 1)
    if s_cap < 2:
        return False

    for _ in range(PLACE_ATTEMPTS):
        s = rng.randint(2, s_cap)
        if relation == "equal":
            hlen = s
            vlen = s
        elif relation == "horizontal_longer":
            hlen = s + rng.randint(1, min(4, max(1, w - s)))
            vlen = s
        else:
            hlen = s
            vlen = s + rng.randint(1, min(4, max(1, h - s)))

        if hlen > w or vlen > h:
            continue

        row_lo = vlen - 1 if vdir < 0 else 0
        row_hi = h - 1 if vdir < 0 else h - vlen
        col_lo = hlen - 1 if hdir < 0 else 0
        col_hi = w - 1 if hdir < 0 else w - hlen
        if row_lo > row_hi or col_lo > col_hi:
            continue

        corner = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
        l_cells = _l_cells(corner, hlen, vlen, hdir, vdir)
        square = _square_cells(corner, min(hlen, vlen), hdir, vdir)
        reserve = l_cells | square

        if any(blocked[r, c] for r, c in reserve):
            continue

        for r, c in l_cells:
            grid[r, c] = color
        _mark_blocked(blocked, reserve)
        return True

    return False


def _try_place_marker(grid: np.ndarray, blocked: np.ndarray, color: int, rng: random.Random) -> bool:
    h, w = grid.shape
    for _ in range(MARKER_ATTEMPTS):
        r = rng.randrange(h)
        c = rng.randrange(w)
        if blocked[r, c] or int(grid[r, c]) != BLACK:
            continue
        grid[r, c] = color
        _mark_blocked(blocked, {(int(r), int(c))})
        return True
    return False


def _valid_generated_grid(grid: np.ndarray, expected_l_count: int) -> bool:
    if not np.any(grid == RED) or not np.any(grid == BLUE):
        return False

    l_count = 0
    for color, pts in _components_by_color(grid):
        if len(pts) == 1 and color in (RED, BLUE):
            continue
        info = _parse_l_component(pts)
        if info is None:
            return False
        l_count += 1

    if l_count != expected_l_count:
        return False
    return bool(np.any(main(grid) != grid))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_l = 4 if h * w < 350 else 5
    num_l = rng.randint(2, max_l)
    relations = _diverse_choices(RELATIONS, num_l, rng)
    directions = _diverse_choices(DIRECTIONS, num_l, rng)

    for i in range(num_l):
        color = rng.choice(L_COLORS)
        ok = _try_place_l(grid, blocked, relations[i], directions[i], color, rng)
        if not ok:
            return None

    marker_colors = [RED] * rng.randint(1, 3) + [BLUE] * rng.randint(1, 3)
    rng.shuffle(marker_colors)
    for color in marker_colors:
        if not _try_place_marker(grid, blocked, color, rng):
            return None

    if not _valid_generated_grid(grid, num_l):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
