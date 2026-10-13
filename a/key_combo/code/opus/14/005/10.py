from __future__ import annotations

import random

import numpy as np


def _classify_row(row: np.ndarray) -> tuple[int, bool, np.ndarray]:
    vals = sorted({int(v) for v in row.tolist()})
    nonzero = [v for v in vals if v != 0]
    assert len(nonzero) == 1, "each stripe row must contain exactly one non-black color"
    color = int(nonzero[0])
    if len(vals) == 1:
        return color, False, np.ones(row.shape[0], dtype=bool)

    assert vals == [0, color], "dashed rows may only use one color plus black"
    assert row.shape[0] >= 2, "dashed rows need at least two columns"
    assert np.all(row[1:] != row[:-1]), "dashed rows must alternate color and black"
    return color, True, row == color


def _parse_stripes(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 1 and width >= 2, "expected a non-empty wide stripe grid"

    stripes: list[dict[str, object]] = []
    r = 0
    while r < height:
        color, dashed, pattern = _classify_row(g[r])
        start = r
        r += 1
        while r < height and np.array_equal(g[r], g[start]):
            r += 1
        stripes.append(
            {
                "color": int(color),
                "height": int(r - start),
                "dashed": bool(dashed),
                "pattern": pattern.copy(),
            }
        )

    colors = [int(s["color"]) for s in stripes]
    assert len(colors) == len(set(colors)), "stripe colors must be distinct"
    assert any(bool(s["dashed"]) for s in stripes), "expected at least one dashed stripe"
    assert any(not bool(s["dashed"]) for s in stripes), "expected at least one solid stripe"
    return stripes


def _fill_center(color: int, side: int, dashed: bool, pattern: np.ndarray) -> np.ndarray:
    if not dashed:
        return np.full((side, side), color, dtype=int)

    width = int(pattern.shape[0])
    out = np.zeros((side, side), dtype=int)
    for idx in range(side * side):
        if bool(pattern[idx % width]):
            out[idx // side, idx % side] = int(color)
    return out


def _frame_groups(outer_side: int, thickness: int) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []

    for c in range(outer_side):
        groups.append([(r, c) for r in range(thickness)])
    for r in range(thickness, outer_side - thickness):
        groups.append([(r, c) for c in range(outer_side - thickness, outer_side)])
    for c in range(outer_side - 1, -1, -1):
        groups.append([(r, c) for r in range(outer_side - thickness, outer_side)])
    for r in range(outer_side - thickness - 1, thickness - 1, -1):
        groups.append([(r, c) for c in range(thickness)])

    return groups


def _wrap_frame(cur: np.ndarray, color: int, thickness: int, dashed: bool, pattern: np.ndarray) -> np.ndarray:
    inner_side = int(cur.shape[0])
    outer_side = inner_side + 2 * int(thickness)
    out = np.zeros((outer_side, outer_side), dtype=int)
    out[thickness : thickness + inner_side, thickness : thickness + inner_side] = cur

    if not dashed:
        out[:thickness, :] = int(color)
        out[outer_side - thickness :, :] = int(color)
        out[thickness : outer_side - thickness, :thickness] = int(color)
        out[thickness : outer_side - thickness, outer_side - thickness :] = int(color)
        return out

    width = int(pattern.shape[0])
    for idx, cells in enumerate(_frame_groups(outer_side, int(thickness))):
        if not bool(pattern[idx % width]):
            continue
        for r, c in cells:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build concentric square layers from stacked horizontal solid and dashed stripes.

    Input structure:
    - The input is a wide grid made of full-width horizontal stripes stacked with no gaps.
    - Each stripe uses a unique non-black color and has height 1-3 cells.
    - A stripe is either solid (every cell is that color) or dashed (that color alternates with black across the row, with one-cell gaps and a fixed phase across all rows of the stripe).

    Rule:
    - Read the stripes from bottom to top.
    - The bottom stripe becomes the innermost square whose side length equals the stripe height; solid stripes make a solid square, while dashed stripes apply their first-row color/blank pattern cyclically over the square in row-major order.
    - Each higher stripe wraps the current shape with a square frame whose thickness equals that stripe height.
    - Solid stripes create a fully filled frame; dashed stripes create a frame whose color/blank pattern is copied cyclically from the stripe's first row onto ordered perimeter positions, with every position replicated across the full frame thickness.

    Key insight:
    - Stripe height controls geometry, while solid-versus-dashed structure controls whether the corresponding square layer is fully filled or periodically erased along its 1D traversal order.
    """
    stripes = _parse_stripes(grid)
    runs = list(reversed(stripes))

    first = runs[0]
    cur = _fill_center(
        int(first["color"]),
        int(first["height"]),
        bool(first["dashed"]),
        np.asarray(first["pattern"], dtype=bool),
    )

    for stripe in runs[1:]:
        cur = _wrap_frame(
            cur,
            int(stripe["color"]),
            int(stripe["height"]),
            bool(stripe["dashed"]),
            np.asarray(stripe["pattern"], dtype=bool),
        )

    return cur


_COLORS = tuple(range(1, 10))
_RETRY_LIMIT = 400


def _choose_flags(count: int) -> list[bool]:
    inner_dashed = bool(random.getrandbits(1))
    flags = [False] * count
    flags[-1] = inner_dashed
    for i in range(count - 1):
        flags[i] = bool(random.getrandbits(1))

    if not any(flags):
        flags[random.randrange(count)] = True
    if all(flags):
        pivot = random.randrange(count)
        flags[pivot] = False

    if inner_dashed and not any(not v for v in flags[:-1]):
        flags[random.randrange(count - 1)] = False
    if (not inner_dashed) and not any(v for v in flags[:-1]):
        flags[random.randrange(count - 1)] = True
    return flags


def _gen_heights(count: int) -> list[int] | None:
    for _ in range(80):
        heights = [random.randint(1, 3) for _ in range(count)]
        total = sum(heights)
        outer_side = 2 * total - heights[-1]
        if 7 <= total <= 12 and outer_side <= 30:
            return heights
    return None


def _make_row(width: int, color: int, dashed: bool) -> np.ndarray:
    if not dashed:
        return np.full(width, color, dtype=int)

    phase = random.randint(0, 1)
    row = np.zeros(width, dtype=int)
    row[(np.arange(width) + phase) % 2 == 0] = int(color)
    return row


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(4, 6)
    heights = _gen_heights(count)
    if heights is None:
        return None

    total_h = sum(heights)
    min_w = max(11, total_h + 3)
    max_w = min(18, total_h + 10)
    if min_w > max_w:
        return None
    width = random.randint(min_w, max_w)

    colors = random.sample(_COLORS, count)
    dashed_flags = _choose_flags(count)

    rows: list[np.ndarray] = []
    for color, height, dashed in zip(colors, heights, dashed_flags):
        row = _make_row(width, color, dashed)
        rows.extend([row.copy() for _ in range(height)])

    grid = np.vstack(rows)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if grid.ndim != 2 or out.ndim != 2:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if out.shape[0] != out.shape[1]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
