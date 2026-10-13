from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
ALL_L_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
GEN_ATTEMPTS = 800
PAIR_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_of_color(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(g == int(color)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []

        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c, h, w):
                if not seen[nr, nc] and int(g[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    q.append((nr, nc))

        comps.append(comp)

    return comps


def _l_piece_info(coords: list[tuple[int, int]]) -> dict[str, object]:
    assert len(coords) == 3, "each fragment must have exactly three cells"
    arr = np.asarray(coords, dtype=int)
    r0, c0 = map(int, arr.min(axis=0))
    r1, c1 = map(int, arr.max(axis=0))
    assert r1 - r0 == 1 and c1 - c0 == 1, "each fragment must fit a 2x2 box"

    local = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = [cell for cell in ALL_L_CELLS if cell not in local]
    assert len(missing) == 1 and len(local) == 3, "fragment must be a 2x2 L triomino"

    return {
        "bbox": (r0, c0, r1, c1),
        "center2": (r0 + r1, c0 + c1),
        "missing": missing[0],
    }


def _pair_square(piece_a: dict[str, object], piece_b: dict[str, object]) -> tuple[int, int, int, int]:
    ar2, ac2 = piece_a["center2"]  # twice the bbox-center coordinates
    br2, bc2 = piece_b["center2"]
    ar2, ac2, br2, bc2 = int(ar2), int(ac2), int(br2), int(bc2)

    d2 = abs(ar2 - br2)
    assert d2 > 0 and d2 % 2 == 0, "vertical center distance must be a positive integer"
    d = d2 // 2

    mid_r = (ar2 + br2) // 4
    mid_c = (ac2 + bc2) // 4
    r0 = int(mid_r - d // 2)
    c0 = int(mid_c - d // 2)
    return r0, c0, r0 + d, c0 + d


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace paired same-color L fragments with distance-sized solid squares.

    Input structure:
    - The background is black.
    - Every non-black color appears as exactly two 4-connected L fragments, each
      a 2x2 bounding box with one corner missing.
    - The two fragments of a color are matching mirror pieces that indicate one
      pair, while different colors form independent pairs.

    Rule:
    - Pair fragments by color and erase all original fragment cells.
    - For each pair, measure d as the absolute row-distance between the two
      fragment bounding-box centers.
    - Take the integer midpoint of the two centers and draw a solid d by d
      square of that pair color centered there, clipping cells outside the grid.

    Key insight: The mirrored L shapes identify which two pieces belong
    together; their vertical separation encodes square size rather than a move
    direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = np.zeros((h, w), dtype=int)

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each color must have exactly two fragments"
        pieces = [_l_piece_info(comp) for comp in comps]
        r0, c0, r1, c1 = _pair_square(pieces[0], pieces[1])

        rr0, cc0 = max(0, r0), max(0, c0)
        rr1, cc1 = min(h, r1), min(w, c1)
        assert rr0 < rr1 and cc0 < cc1, "clipped square must leave visible cells"

        target = out[rr0:rr1, cc0:cc1]
        assert np.all((target == BG) | (target == color)), "output squares must not overlap ambiguously"
        out[rr0:rr1, cc0:cc1] = color

    return out


def _mirror_missing(missing: tuple[int, int], axis: str) -> tuple[int, int]:
    r, c = missing
    if axis == "vertical":
        return r, 1 - c
    return 1 - r, c


def _l_cells(r0: int, c0: int, missing: tuple[int, int]) -> list[tuple[int, int]]:
    return [
        (int(r0 + dr), int(c0 + dc))
        for dr, dc in ALL_L_CELLS
        if (dr, dc) != missing
    ]


def _can_place(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc]:
                    return False
    return True


def _mark(cells: list[tuple[int, int]], occupied: np.ndarray) -> None:
    for r, c in cells:
        occupied[int(r), int(c)] = True


def _candidate_columns(w: int, d: int, want_clip: bool) -> tuple[int, int] | None:
    if want_clip:
        if random.random() < 0.5:
            hi = min(w - 2, max(0, d // 2 - 1))
            if hi < 0:
                return None
            return random.randint(0, hi), random.randint(0, hi)

        lo = max(0, w - d + d // 2 + 1)
        if lo > w - 2:
            return None
        return random.randint(lo, w - 2), random.randint(lo, w - 2)

    return random.randint(0, w - 2), random.randint(0, w - 2)


def _rect_from_topleft(r_a: int, c_a: int, r_b: int, c_b: int) -> tuple[int, int, int, int]:
    piece_a = {"center2": (2 * r_a + 1, 2 * c_a + 1)}
    piece_b = {"center2": (2 * r_b + 1, 2 * c_b + 1)}
    return _pair_square(piece_a, piece_b)


def _reserve_rect(rect: tuple[int, int, int, int], reserved: np.ndarray, want_clip: bool) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = rect
    clipped = r0 < 0 or c0 < 0 or r1 > h or c1 > w
    if clipped != want_clip:
        return False

    rr0, cc0 = max(0, r0), max(0, c0)
    rr1, cc1 = min(h, r1), min(w, c1)
    if rr0 >= rr1 or cc0 >= cc1:
        return False

    mr0, mc0 = max(0, rr0 - 1), max(0, cc0 - 1)
    mr1, mc1 = min(h, rr1 + 1), min(w, cc1 + 1)
    if bool(reserved[mr0:mr1, mc0:mc1].any()):
        return False

    reserved[mr0:mr1, mc0:mc1] = True
    return True


def _place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    color: int,
    want_clip: bool,
) -> bool:
    h, w = grid.shape
    max_d = min(12, h - 3)
    if not want_clip:
        max_d = min(max_d, w)
    min_d = 5 if want_clip else 3
    if min_d > max_d:
        return False

    for _ in range(PAIR_ATTEMPTS):
        d = random.randint(min_d, max_d)
        r_top = random.randint(0, h - 2 - d)
        r_bot = r_top + d

        cols = _candidate_columns(w, d, want_clip)
        if cols is None:
            continue
        c_a, c_b = cols

        rect = _rect_from_topleft(r_top, c_a, r_bot, c_b)
        test_reserved = reserved.copy()
        if not _reserve_rect(rect, test_reserved, want_clip):
            continue

        missing_a = random.choice(ALL_L_CELLS)
        missing_b = _mirror_missing(missing_a, random.choice(("horizontal", "vertical")))
        if random.random() < 0.5:
            r1, c1, m1 = r_top, c_a, missing_a
            r2, c2, m2 = r_bot, c_b, missing_b
        else:
            r1, c1, m1 = r_bot, c_b, missing_a
            r2, c2, m2 = r_top, c_a, missing_b

        cells1 = _l_cells(r1, c1, m1)
        cells2 = _l_cells(r2, c2, m2)
        test_occupied = occupied.copy()
        if not _can_place(cells1, test_occupied):
            continue
        _mark(cells1, test_occupied)
        if not _can_place(cells2, test_occupied):
            continue
        _mark(cells2, test_occupied)

        for r, c in cells1 + cells2:
            grid[int(r), int(c)] = int(color)
        occupied[:, :] = test_occupied
        reserved[:, :] = test_reserved
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 30)
    w = random.randint(13, 30)
    area = h * w
    max_pairs = 3 if area < 300 else 4 if area < 650 else 5
    pair_count = random.randint(2, max_pairs)
    colors = random.sample(PALETTE, pair_count)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)

    clipped_index = random.randrange(pair_count) if random.random() < 0.5 else -1
    for idx, color in enumerate(colors):
        if not _place_pair(grid, occupied, reserved, int(color), idx == clipped_index):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-L square input")
