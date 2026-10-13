from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
VISIBLE_COLORS = (1, 2, 3, 4, 8, 9)

GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (12, 18)
TILE_PERIODS = (3, 4)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120


def _tiled_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tr, tc = tile.shape
    h, w = shape
    rows = np.arange(h) % tr
    cols = np.arange(w) % tc
    return tile[rows[:, None], cols[None, :]]


def _row_period(tile: np.ndarray) -> int:
    tr = tile.shape[0]
    for period in range(1, tr + 1):
        ok = True
        for r in range(tr):
            if not np.array_equal(tile[r], tile[r % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("row period must exist")


def _col_period(tile: np.ndarray) -> int:
    tc = tile.shape[1]
    for period in range(1, tc + 1):
        ok = True
        for c in range(tc):
            if not np.array_equal(tile[:, c], tile[:, c % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("col period must exist")


def _sample_exact_tile(tr: int, tc: int) -> np.ndarray | None:
    palette = random.sample(VISIBLE_COLORS, random.randint(2, 3))
    for _ in range(120):
        tile = np.array(
            [[random.choice(palette) for _ in range(tc)] for _ in range(tr)],
            dtype=int,
        )
        if len(set(map(int, tile.flat))) < 2:
            continue
        if _row_period(tile) != tr or _col_period(tile) != tc:
            continue
        return tile
    return None


def _candidate_tile(
    grid: np.ndarray,
    unknown: np.ndarray,
    tr: int,
    tc: int,
) -> np.ndarray | None:
    tile = np.full((tr, tc), -1, dtype=int)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if unknown[r, c]:
                continue
            rr = r % tr
            cc = c % tc
            color = int(grid[r, c])
            if tile[rr, cc] < 0:
                tile[rr, cc] = color
            elif int(tile[rr, cc]) != color:
                return None
    if np.any(tile < 0):
        return None
    return tile


def _infer_min_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    unknown = (g == BLACK) | (g == GREY)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for tr in range(1, h + 1):
        for tc in range(1, w + 1):
            tile = _candidate_tile(g, unknown, tr, tc)
            if tile is not None:
                candidates.append((tr, tc, tile))
    assert candidates, "visible pixels must determine at least one tile"
    min_area = min(tr * tc for tr, tc, _tile in candidates)
    best = [(tr, tc, tile) for tr, tc, tile in candidates if tr * tc == min_area]
    assert len(best) == 1, "minimum repeating tile must be unique"
    return best[0]


def _outside_mask(grey_mask: np.ndarray) -> np.ndarray:
    h, w = grey_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not grey_mask[r, c] and not outside[r, c]:
            outside[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            push(nr, nc)
    return outside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden repeating tile, then repair only the black occlusions trapped inside grey fences.

    Input structure:
    - The grid is a cropped view of a 2D periodic color tiling whose visible colors exclude grey.
    - Some cells are black occlusions, and one or more 4-connected grey fence loops partition the canvas.
    - Visible non-black, non-grey cells are enough to uniquely determine the minimum repeating tile.

    Rule:
    - Treat both black and grey cells as unknown when inferring the minimum rectangular tile consistent with all visible cells.
    - Flood-fill 4-connectedly from the four edges without crossing grey to mark the outside region.
    - Replace each black cell in an unfilled internal region with the inferred tile color at the same global residue; keep outside black cells black.
    - Preserve every grey fence pixel and every originally visible non-black, non-grey pixel.

    Key insight:
    - Black means two different things at once: it hides tile content everywhere, but only the black cells topologically sealed off by grey fences are allowed to be reconstructed from the inferred periodic pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    tr, tc, tile = _infer_min_tile(g)
    outside = _outside_mask(g == GREY)

    out = g.copy()
    repair = (g == BLACK) & ~outside
    rows, cols = np.nonzero(repair)
    for r, c in zip(rows.tolist(), cols.tolist()):
        out[r, c] = int(tile[r % tr, c % tc])
    return out


def _fence_masks(
    h: int,
    w: int,
    fences: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    fence_mask = np.zeros((h, w), dtype=bool)
    inside_mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in fences:
        fence_mask[r0, c0 : c1 + 1] = True
        fence_mask[r1, c0 : c1 + 1] = True
        fence_mask[r0 : r1 + 1, c0] = True
        fence_mask[r0 : r1 + 1, c1] = True
        inside_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return fence_mask, inside_mask


def _sample_fences(
    h: int,
    w: int,
    reserved: np.ndarray,
) -> list[tuple[int, int, int, int]] | None:
    count = 1 if random.random() < 0.55 else 2
    padded = reserved.copy()
    fences: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            fh = random.randint(4, min(7, h - 2))
            fw = random.randint(4, min(7, w - 2))
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r1 + 2)
            cc1 = min(w, c1 + 2)
            if padded[rr0:rr1, cc0:cc1].any():
                continue
            fences.append((r0, c0, r1, c1))
            padded[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return fences


def _sample_rect_in_box(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    used: np.ndarray,
) -> tuple[int, int, int, int] | None:
    max_h = min(3, r1 - r0 + 1)
    max_w = min(3, c1 - c0 + 1)
    for _ in range(PLACEMENT_RETRIES):
        rh = random.randint(1, max_h)
        rw = random.randint(1, max_w)
        sr = random.randint(r0, r1 - rh + 1)
        sc = random.randint(c0, c1 - rw + 1)
        if used[sr : sr + rh, sc : sc + rw].any():
            continue
        return sr, sc, rh, rw
    return None


def _sample_outside_rect(
    outside: np.ndarray,
    used: np.ndarray,
    reserve: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = outside.shape
    for _ in range(PLACEMENT_RETRIES):
        rh = random.randint(1, 3)
        rw = random.randint(1, 3)
        if rh > h or rw > w:
            return None
        sr = random.randint(0, h - rh)
        sc = random.randint(0, w - rw)
        block = outside[sr : sr + rh, sc : sc + rw]
        if not np.all(block):
            continue
        if used[sr : sr + rh, sc : sc + rw].any():
            continue
        if reserve[sr : sr + rh, sc : sc + rw].any():
            continue
        return sr, sc, rh, rw
    return None


def _build_expected(inp: np.ndarray, base: np.ndarray) -> np.ndarray:
    outside = _outside_mask(inp == GREY)
    out = inp.copy()
    repair = (inp == BLACK) & ~outside
    out[repair] = base[repair]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    tr = random.choice(TILE_PERIODS)
    tc = random.choice(TILE_PERIODS)
    tile = _sample_exact_tile(tr, tc)
    if tile is None:
        return None

    base = _tiled_grid(tile, (h, w))

    witness_r = random.randint(0, h - tr)
    witness_c = random.randint(0, w - tc)
    reserve = np.zeros((h, w), dtype=bool)
    reserve[witness_r : witness_r + tr, witness_c : witness_c + tc] = True

    fences = _sample_fences(h, w, reserve)
    if fences is None:
        return None
    fence_mask, inside_mask = _fence_masks(h, w, fences)

    inp = base.copy()
    inp[fence_mask] = GREY

    used = fence_mask.copy()
    inside_rects = []
    chosen_fences = random.sample(fences, random.randint(1, len(fences)))
    for r0, c0, r1, c1 in chosen_fences:
        rect = _sample_rect_in_box(r0 + 1, c0 + 1, r1 - 1, c1 - 1, used)
        if rect is None:
            return None
        inside_rects.append(rect)
        sr, sc, rh, rw = rect
        inp[sr : sr + rh, sc : sc + rw] = BLACK
        used[sr : sr + rh, sc : sc + rw] = True

    outside = _outside_mask(fence_mask)
    outside_rects = []
    for _ in range(random.randint(1, 3)):
        rect = _sample_outside_rect(outside, used, reserve)
        if rect is None:
            if not outside_rects:
                return None
            break
        outside_rects.append(rect)
        sr, sc, rh, rw = rect
        inp[sr : sr + rh, sc : sc + rw] = BLACK
        used[sr : sr + rh, sc : sc + rw] = True

    expected = _build_expected(inp, base)
    repaired = (inp == BLACK) & (expected != BLACK)
    preserved = (inp == BLACK) & (expected == BLACK)
    if not np.any(repaired) or not np.any(preserved):
        return None
    if np.array_equal(inp, expected):
        return None

    fully_repaired = inp.copy()
    fully_repaired[inp == BLACK] = base[inp == BLACK]
    if np.array_equal(expected, fully_repaired):
        return None

    tr2, tc2, inferred = _infer_min_tile(inp)
    if (tr2, tc2) != (tr, tc):
        return None
    if not np.array_equal(inferred, tile):
        return None

    visible_witness = inp[witness_r : witness_r + tr, witness_c : witness_c + tc]
    if np.any((visible_witness == BLACK) | (visible_witness == GREY)):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective tile-repair input")
