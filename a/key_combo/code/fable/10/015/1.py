import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
NONBLACK = {BLUE, YELLOW}

MAX_GEN_ATTEMPTS = 400
GRID_LIMIT = 26
SOURCE_COUNT_CHOICES = (2, 3)
SOURCE_HEIGHT_RANGE = (2, 5)
SOURCE_WIDTH_RANGE = (2, 5)
MIN_GAP = 2
MIN_MARGIN = 1


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    ps = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    ps[1:, 1:] = mask.astype(int)
    ps = ps.cumsum(axis=0).cumsum(axis=1)
    return ps


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _distribute(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _place_intervals(lengths: list[int]) -> tuple[int, list[tuple[int, int]]] | None:
    n = len(lengths)
    order = list(range(n))
    random.shuffle(order)
    base = sum(lengths) + (n - 1) * MIN_GAP + 2 * MIN_MARGIN
    if base > GRID_LIMIT:
        return None
    extra_total = random.randint(0, GRID_LIMIT - base)
    extras = _distribute(extra_total, n + 1)
    gaps = [MIN_MARGIN + extras[0]]
    for i in range(1, n):
        gaps.append(MIN_GAP + extras[i])
    gaps.append(MIN_MARGIN + extras[-1])

    intervals = [None] * n
    pos = gaps[0]
    for rank, idx in enumerate(order):
        span = lengths[idx]
        intervals[idx] = (pos, pos + span - 1)
        pos += span
        pos += gaps[rank + 1]
    return pos, intervals  # type: ignore[return-value]


def _source_masks(
    shape: tuple[int, int],
    source: tuple[int, int, int, int],
    blocked_side: str | None,
    blue_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    r0, c0, r1, c1 = source
    hmask = np.zeros(shape, dtype=bool)
    vmask = np.zeros(shape, dtype=bool)

    if blocked_side != "left":
        for r in range(r0, r1 + 1):
            for c in range(c0 - 1, -1, -1):
                if blue_mask[r, c]:
                    break
                hmask[r, c] = True
    if blocked_side != "right":
        for r in range(r0, r1 + 1):
            for c in range(c1 + 1, w):
                if blue_mask[r, c]:
                    break
                hmask[r, c] = True
    if blocked_side != "up":
        for c in range(c0, c1 + 1):
            for r in range(r0 - 1, -1, -1):
                if blue_mask[r, c]:
                    break
                vmask[r, c] = True
    if blocked_side != "down":
        for c in range(c0, c1 + 1):
            for r in range(r1 + 1, h):
                if blue_mask[r, c]:
                    break
                vmask[r, c] = True
    return hmask, vmask


def _render_input(
    shape: tuple[int, int],
    sources: list[tuple[int, int, int, int]],
    blocked_sides: list[str | None],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    blue_mask = np.zeros((h, w), dtype=bool)

    for (r0, c0, r1, c1), side in zip(sources, blocked_sides):
        if side == "left":
            blue_mask[r0 : r1 + 1, c0 - 1] = True
        elif side == "right":
            blue_mask[r0 : r1 + 1, c1 + 1] = True
        elif side == "up":
            blue_mask[r0 - 1, c0 : c1 + 1] = True
        elif side == "down":
            blue_mask[r1 + 1, c0 : c1 + 1] = True

    grid[blue_mask] = BLUE
    hmasks: list[np.ndarray] = []
    vmasks: list[np.ndarray] = []
    yellow_mask = np.zeros((h, w), dtype=bool)
    for source, side in zip(sources, blocked_sides):
        hmask, vmask = _source_masks(shape, source, side, blue_mask)
        hmasks.append(hmask)
        vmasks.append(vmask)
        yellow_mask |= hmask | vmask
    grid[yellow_mask] = YELLOW
    return grid, hmasks, vmasks


def _rows_with_any(mask: np.ndarray) -> list[int]:
    return list(np.flatnonzero(mask.any(axis=1)))


def _cols_with_any(mask: np.ndarray) -> list[int]:
    return list(np.flatnonzero(mask.any(axis=0)))


def _ray_side_ok(grid: np.ndarray, rect: tuple[int, int, int, int], side: str) -> bool:
    r0, c0, r1, c1 = rect
    if side == "left":
        for r in range(r0, r1 + 1):
            for c in range(c0 - 1, -1, -1):
                if grid[r, c] == BLUE:
                    break
                if grid[r, c] != YELLOW:
                    return False
        return True
    if side == "right":
        for r in range(r0, r1 + 1):
            for c in range(c1 + 1, grid.shape[1]):
                if grid[r, c] == BLUE:
                    break
                if grid[r, c] != YELLOW:
                    return False
        return True
    if side == "up":
        for c in range(c0, c1 + 1):
            for r in range(r0 - 1, -1, -1):
                if grid[r, c] == BLUE:
                    break
                if grid[r, c] != YELLOW:
                    return False
        return True
    if side == "down":
        for c in range(c0, c1 + 1):
            for r in range(r1 + 1, grid.shape[0]):
                if grid[r, c] == BLUE:
                    break
                if grid[r, c] != YELLOW:
                    return False
        return True
    raise ValueError(side)


def _find_cavities(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    runs_by_span: dict[tuple[int, int], list[int]] = {}
    black_ps = _prefix_sum(g == BLACK)

    for r in range(1, h - 1):
        c = 1
        while c < w - 1:
            if g[r, c] != BLACK:
                c += 1
                continue
            c0 = c
            while c < w - 1 and g[r, c] == BLACK:
                c += 1
            c1 = c - 1
            if g[r, c0 - 1] in NONBLACK and g[r, c1 + 1] in NONBLACK:
                runs_by_span.setdefault((c0, c1), []).append(r)

    found: list[tuple[int, int, int, int]] = []
    for (c0, c1), rows in runs_by_span.items():
        rows = sorted(rows)
        start = rows[0]
        prev = rows[0]
        for r in rows[1:] + [rows[-1] + 2]:
            if r == prev + 1:
                prev = r
                continue
            r0, r1 = start, prev
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if _rect_sum(black_ps, r0, c0, r1, c1) == area:
                left = g[r0 : r1 + 1, c0 - 1]
                right = g[r0 : r1 + 1, c1 + 1]
                top = g[r0 - 1, c0 : c1 + 1]
                bottom = g[r1 + 1, c0 : c1 + 1]
                sides = (left, right, top, bottom)
                if all(np.all(side == side[0]) and int(side[0]) in NONBLACK for side in sides):
                    if (left[0] == YELLOW or right[0] == YELLOW) and (top[0] == YELLOW or bottom[0] == YELLOW):
                        rect = (r0, c0, r1, c1)
                        ok = True
                        if left[0] == YELLOW:
                            ok &= _ray_side_ok(g, rect, "left")
                        if right[0] == YELLOW:
                            ok &= _ray_side_ok(g, rect, "right")
                        if top[0] == YELLOW:
                            ok &= _ray_side_ok(g, rect, "up")
                        if bottom[0] == YELLOW:
                            ok &= _ray_side_ok(g, rect, "down")
                        if ok:
                            found.append(rect)
            start = r
            prev = r
    return sorted(found)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each removed light-source rectangle from its yellow ray residue and fill the black cavity green.

    Input structure:
    - The background is black; blue pixels are fixed blocker walls.
    - Yellow pixels are horizontal and vertical ray bands left behind by one or more hidden solid source rectangles.
    - The source rectangles themselves are absent from the input, so their cells remain black; ray crossings stay yellow.

    Rule:
    - Detect every maximal black rectangle whose four sides are immediately bordered by yellow rays or blue flush walls.
    - A side bordered by yellow means that source side emitted rays in that direction; a side bordered by blue means that direction was fully blocked right next to the source.
    - Recolor each such black cavity rectangle green and leave all yellow rays, blue walls, and remaining black background unchanged.

    Key insight:
    - The true source is not a yellow intersection but the black cavity whose height is bracketed by horizontal ray rows and whose width is bracketed by vertical ray columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert GREEN not in set(map(int, np.unique(g)))

    out = g.copy()
    cavities = _find_cavities(g)
    assert cavities, "expected at least one enclosed black source cavity"
    for r0, c0, r1, c1 in cavities:
        out[r0 : r1 + 1, c0 : c1 + 1] = GREEN
    return out


def _expected_output(inp: np.ndarray, sources: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for r0, c0, r1, c1 in sources:
        out[r0 : r1 + 1, c0 : c1 + 1] = GREEN
    return out


def _same_rects(a: list[tuple[int, int, int, int]], b: list[tuple[int, int, int, int]]) -> bool:
    return sorted(a) == sorted(b)


def _crossing_count(hmasks: list[np.ndarray], vmasks: list[np.ndarray]) -> int:
    total = np.zeros(hmasks[0].shape, dtype=int)
    n = len(hmasks)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total += (hmasks[i] & vmasks[j]).astype(int)
    return int(np.count_nonzero(total))


def _check_hidden_semantics(
    inp: np.ndarray,
    sources: list[tuple[int, int, int, int]],
    blocked_sides: list[str | None],
    hmasks: list[np.ndarray],
    vmasks: list[np.ndarray],
) -> bool:
    colors = set(map(int, np.unique(inp)))
    if not colors.issubset({BLACK, BLUE, YELLOW}):
        return False
    if BLUE not in colors or YELLOW not in colors:
        return False
    if _crossing_count(hmasks, vmasks) == 0:
        return False

    for (r0, c0, r1, c1), blocked, hmask, vmask in zip(sources, blocked_sides, hmasks, vmasks):
        if np.any(inp[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
            return False
        rows = list(range(r0, r1 + 1))
        cols = list(range(c0, c1 + 1))
        if _rows_with_any(hmask) != rows:
            return False
        if _cols_with_any(vmask) != cols:
            return False

        side_colors = {
            "left": inp[r0 : r1 + 1, c0 - 1],
            "right": inp[r0 : r1 + 1, c1 + 1],
            "up": inp[r0 - 1, c0 : c1 + 1],
            "down": inp[r1 + 1, c0 : c1 + 1],
        }
        for side, arr in side_colors.items():
            if not np.all(arr == arr[0]):
                return False
            expected = BLUE if side == blocked else YELLOW
            if int(arr[0]) != expected:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(SOURCE_COUNT_CHOICES)
    heights = [random.randint(*SOURCE_HEIGHT_RANGE) for _ in range(n)]
    widths = [random.randint(*SOURCE_WIDTH_RANGE) for _ in range(n)]
    row_pack = _place_intervals(heights)
    col_pack = _place_intervals(widths)
    if row_pack is None or col_pack is None:
        return None
    h, row_intervals = row_pack
    w, col_intervals = col_pack
    if max(h, w) > 30:
        return None

    sources = []
    for i in range(n):
        r0, r1 = row_intervals[i]
        c0, c1 = col_intervals[i]
        sources.append((r0, c0, r1, c1))

    blocked_sides: list[str | None] = [None] * n
    blocked_indices = random.sample(range(n), k=random.randint(1, min(2, n)))
    for idx in blocked_indices:
        blocked_sides[idx] = random.choice(["left", "right", "up", "down"])

    inp, hmasks, vmasks = _render_input((h, w), sources, blocked_sides)
    if not _check_hidden_semantics(inp, sources, blocked_sides, hmasks, vmasks):
        return None

    found = _find_cavities(inp)
    if not _same_rects(found, sources):
        return None

    expected = _expected_output(inp, sources)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid source-cavity ray-inversion input")
