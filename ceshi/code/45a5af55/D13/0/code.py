# Variant of 45a5af55
# Dimension: D13 — 输入色条间的分隔与干扰 (src)
# Change: 色条之间直接相邻，无分隔行，无干扰元素... → 色条之间有背景色分隔行

from __future__ import annotations

import numpy as np

# Design analysis for the D13 variant
# a) Concrete behavior:
#    The input is still a stack of full-width solid-color stripes, but adjacent stripes are no
#    longer touching. Between every neighboring pair, insert 1 or 2 full-width background rows
#    of color 0. Example from bottom->top stripe heights [2, 1, 3] becomes, in top->bottom
#    layout: AAA / 00 / B / 0 / CC, where A/B/C are non-zero stripe colors and 0 is separator.
#    The output stays identical to the original task: read only the non-zero stripes bottom->top,
#    use the bottom stripe as the center solid square, and wrap outer square frames according to
#    the higher stripes' heights. The changed step is therefore input parsing only.
# b) Complexity calibration:
#    Every gap contains a separator, so the variant is visible in every sample. Gap height is
#    restricted to 1 or 2 rows, which is enough to show "formatting rows exist" without turning
#    the grid into noise or creating too many branches. Stripe heights remain 1..4 as before.
# c) Hidden assumptions in the original solver:
#    - `col = g[r, 0]` assumed every row belongs to a stripe. This now fails on separator rows.
#    - The inner `while ... == col` assumed stripe rows are contiguous with no foreign rows in
#      between. D13 breaks that adjacency, so separator rows must be skipped before measuring a
#      stripe's height.
#    - `assert runs` is still valid, but only after ignoring background rows.
# d) Placement legality:
#    The generator uses full-width rows only, so nothing can overlap horizontally. It inserts
#    separators only between stripes, never inside them, and keeps both input height and output
#    size within 30 to stay ARC-like and solver-safe.
# e) Visual observability:
#    With 2-6 stripes and a separator after every interior boundary, humans can immediately see
#    that blank rows are regular delimiters rather than meaningful stripes. Across samples, the
#    output ignoring those rows is stable and therefore inferable.
# f) Visual cleanliness:
#    Only one formatting device is added: uniform 0-colored rows. No extra markers or side noise
#    are introduced, so the input remains a tidy vertical stack.
# g) Color inference:
#    No color remapping is introduced. Color 0 is reserved for separators and never used as a
#    stripe color, so the added signal is easy to separate from the semantic stripes.
# h) New solver assumptions and generator guarantees:
#    - The solver now assumes separator rows are color 0 and span the full width.
#    - It also assumes every non-zero stripe row is full-width and monochrome.
#    - The generator enforces both conditions and verifies `main(input)` against the intended
#      output assembled from the stripe specification.


def _build_output_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    assert runs
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, h in runs[1:]:
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt
    return cur


def _parse_runs_with_separators(grid: np.ndarray, bg: int = 0) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    runs: list[tuple[int, int]] = []
    r = H - 1
    while r >= 0:
        col = int(g[r, 0])
        if col == bg:
            r -= 1
            continue
        assert bool((g[r] == col).all())
        h = 1
        rr = r - 1
        while rr >= 0 and int(g[rr, 0]) == col:
            assert bool((g[rr] == col).all())
            h += 1
            rr -= 1
        runs.append((col, h))
        r = rr
    assert runs
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    D13 variant:
    - Non-zero stripes are still full-width and read bottom->top.
    - Rows of color 0 may appear between adjacent stripes as separators and must be ignored.
    """
    runs = _parse_runs_with_separators(np.array(grid, dtype=int), bg=0)
    return _build_output_from_runs(runs)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    W = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    gaps = [int(np.random.randint(1, 3)) for _ in range(n_runs - 1)]

    output_size = heights[0] + 2 * sum(heights[1:])
    input_height = sum(heights) + sum(gaps)
    if output_size > 30 or input_height > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while c == bg or (prev is not None and c == prev):
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    pieces: list[np.ndarray] = []
    for idx in range(n_runs - 1, -1, -1):
        pieces.append(np.full((heights[idx], W), cols[idx], dtype=int))
        if idx > 0:
            pieces.append(np.full((gaps[idx - 1], W), bg, dtype=int))
    grid = np.vstack(pieces)

    runs = list(zip(cols, heights))
    expected = _build_output_from_runs(runs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
