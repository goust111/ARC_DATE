from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
PINK = 6
ORANGE = 7

STRIPE_PALETTE = (BLUE, RED, GREEN, PINK, ORANGE)
MAX_ATTEMPTS = 600
_SEEN_OUTPUT_SIGS: set[bytes] = set()
_SIDE_QUEUE: list[int] = []


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask as coordinate lists."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_gray_square(grid: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for comp in _components4(grid == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w:
            continue
        if len(comp) != side_h * side_w:
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GRAY):
            candidates.append((r0, c0, side_h))
    assert len(candidates) == 1, "expected exactly one solid gray square"
    return candidates[0]


def _find_stripes(grid: np.ndarray) -> list[dict[str, int]]:
    stripes: list[dict[str, int]] = []
    mask = (grid != BLACK) & (grid != GRAY)
    for comp in _components4(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 1:
            continue
        if len(comp) != height:
            continue
        stripes.append({"r0": r0, "c": c0, "h": height, "color": color})
    stripes.sort(key=lambda item: item["c"])
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered vertical stripes by column and paint their colors into the gray square as staggered congruence-diagonal entries.

    Input structure:
    - A black background contains one solid gray square and several colored vertical stripe segments.
    - Each stripe is a 1-cell-wide monochrome 4-connected vertical segment, and stripe columns are all distinct.
    - The gray square side length equals the number of stripes.

    Rule:
    - Detect the solid gray square and use its side length S to define an SxS black output canvas.
    - Detect every non-black, non-gray vertical stripe segment and sort the stripes by column from left to right.
    - For the stripe with sorted index k and color c, fill output column k only at rows where (row + k) mod S == 0 with color c; leave all other cells black.

    Key insight:
    - The stripe order is used twice: once to choose the output column and again to choose the congruence-class phase offset, creating an interlaced diagonal pattern rather than simple solid columns.
    """
    g = np.asarray(grid, dtype=int)
    _sq_r, _sq_c, side = _find_gray_square(g)
    stripes = _find_stripes(g)
    assert len(stripes) == side, "stripe count must match gray-square side length"

    out = np.zeros((side, side), dtype=int)
    for k, stripe in enumerate(stripes):
        color = int(stripe["color"])
        for row in range(side):
            if (row + k) % side == 0:
                out[row, k] = color
    return out


def _sample_columns(square_col: int, count: int) -> list[int] | None:
    candidates = list(range(1, square_col - 1))
    if len(candidates) < count:
        return None
    for _ in range(100):
        cols = sorted(random.sample(candidates, count))
        if min(cols[i + 1] - cols[i] for i in range(count - 1)) >= 2:
            return cols
    return None


def _nontrivial_permutation(size: int) -> list[int]:
    base = list(range(size))
    while True:
        perm = base[:]
        random.shuffle(perm)
        if perm != base and perm != base[::-1]:
            return perm


def _construct_expected(stripes: list[dict[str, int]], side: int) -> np.ndarray:
    expected = np.zeros((side, side), dtype=int)
    for k, stripe in enumerate(stripes):
        expected[(-k) % side, k] = int(stripe["color"])
    return expected


def _next_side() -> int:
    global _SIDE_QUEUE
    if not _SIDE_QUEUE:
        _SIDE_QUEUE = [3, 4, 5]
        random.shuffle(_SIDE_QUEUE)
    return _SIDE_QUEUE.pop()


def _validate_example(grid: np.ndarray) -> np.ndarray:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= 30

    sq_r, sq_c, side = _find_gray_square(grid)
    assert 3 <= side <= 5
    stripes = _find_stripes(grid)
    assert len(stripes) == side

    colors = [int(s["color"]) for s in stripes]
    cols = [int(s["c"]) for s in stripes]
    tops = [int(s["r0"]) for s in stripes]
    heights = [int(s["h"]) for s in stripes]

    assert len(set(colors)) == side
    assert len(set(cols)) == side
    assert len(set(tops)) == side
    assert len(set(heights)) >= 2
    assert cols == sorted(cols)
    row_order = sorted(range(side), key=lambda idx: tops[idx])
    assert row_order != list(range(side))
    assert row_order != list(range(side - 1, -1, -1))

    for stripe in stripes:
        r0 = int(stripe["r0"])
        c = int(stripe["c"])
        h = int(stripe["h"])
        assert 3 <= h <= 5
        assert c < sq_c - 1
        assert np.all(grid[r0 : r0 + h, c] == int(stripe["color"]))
        assert not (sq_r <= r0 + h - 1 and r0 <= sq_r + side - 1 and sq_c <= c <= sq_c + side - 1)

    out = main(grid)
    assert out.shape == (side, side)
    assert max(out.shape) <= 30
    expected = _construct_expected(stripes, side)
    assert np.array_equal(out, expected)
    assert int((out != BLACK).sum()) == side
    assert all(int((out[:, c] != BLACK).sum()) == 1 for c in range(side))
    assert all(int((out[r] != BLACK).sum()) == 1 for r in range(side))
    column_fill_shortcut = np.tile(np.array(colors, dtype=int), (side, 1))
    assert not np.array_equal(out, column_fill_shortcut)

    return out


def _gen_attempt(side: int) -> np.ndarray | None:
    height = random.randint(max(12, side + 8), 18)
    width = random.randint(max(14, 3 * side + 2), min(22, 3 * side + 7))

    square_row = random.randint(max(1, height - side - 4), height - side - 1)
    square_col = random.randint(max(2 * side + 1, width - side - 4), width - side - 1)

    cols = _sample_columns(square_col, side)
    if cols is None:
        return None

    top_choices = sorted(random.sample(range(0, height - 4), side))
    perm = _nontrivial_permutation(side)
    tops = [top_choices[p] for p in perm]
    heights = [random.randint(3, 5) for _ in range(side)]
    if len(set(heights)) < 2:
        return None

    colors = random.sample(list(STRIPE_PALETTE), side)
    grid = np.zeros((height, width), dtype=int)
    grid[square_row : square_row + side, square_col : square_col + side] = GRAY

    for col, top, stripe_h, color in zip(cols, tops, heights, colors):
        grid[top : top + stripe_h, col] = int(color)

    try:
        out = _validate_example(grid)
    except AssertionError:
        return None

    sig = out.tobytes() + str(out.shape).encode()
    if sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(sig)
    return grid


def generate_input() -> np.ndarray:
    side = _next_side()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(side)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
