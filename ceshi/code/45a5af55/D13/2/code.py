# Variant of 45a5af55
# Dimension: D13 — 输入色条间的分隔与干扰 (src)
# Change: 色条之间直接相邻，无分隔行，无干扰元素... → 色条中包含噪声像素（少量不同颜色的像素散布）

from __future__ import annotations

import numpy as np


# Design analysis for this D13 variant
# ===================================
# a) Concrete behavior
# - The input is still a vertical stack of full-width horizontal stripes, and output is
#   still the same nested-square construction.
# - The only structural change is inside each stripe: inject 1-2 off-color pixels into
#   the stripe rectangle. The stripe's "real" color is therefore no longer recoverable
#   from a single pixel or from an "all cells equal" test; it must be inferred by row
#   majority and then merged across consecutive rows with the same majority color.
# - Small trace:
#   Original stripe rows:
#       [4 4 4 4 4 4]
#       [4 4 4 4 4 4]
#   D13 variant:
#       [4 4 7 4 4 4]
#       [4 4 4 4 4 4]
#   Original solver logic uses g[r, 0] and `(row == color).all()`, so the noisy row
#   breaks the 2-row stripe into separate runs. The variant logic instead takes each
#   row's majority color (=4 for both rows), so the stripe height remains 2.
#
# b) Complexity calibration
# - Keep widths at 6-14 and noise at only 1-2 pixels per stripe. This keeps the visual
#   task readable while making the "majority color" step necessary.
# - Adjacent stripes still have different true colors, so there is no ambiguity after
#   majority reduction: consecutive rows from one stripe agree, rows from neighboring
#   stripes disagree.
#
# c) Original implicit assumptions that break
# - `col = g[r, 0]` assumed a single sampled pixel reveals stripe color. Not true once
#   the first-column pixel can be noisy.
# - `(g[rr] == col).all()` assumed every row in a stripe is perfectly uniform. Not true
#   once a single scattered noise pixel exists anywhere in the row.
# - The frame-building phase does not depend on uniform input rows; once runs are parsed
#   correctly, the remainder of the original logic stays valid.
#
# d) Placement legality
# - Each stripe remains a single contiguous rectangle occupying the full width, so there
#   are no overlap or boundary issues beyond choosing noise coordinates inside the stripe.
# - Noise count is capped at 1-2 pixels per stripe, always fewer than half the row width,
#   so row majority is stable.
#
# e) Visual observability
# - Humans can still see broad colored bands, but now some bands contain tiny anomalous
#   specks. Across 2-4 examples this makes it observable that those specks are irrelevant
#   and the dominant band color controls the output.
# - To avoid degeneration back to the original solver behavior on 1-row stripes, the
#   generator forces the noisy pixel into column 0 when a stripe has height 1; this makes
#   single-pixel sampling unreliable as well.
#
# f) Visual cleanliness
# - Noise is sparse and does not stack multiple objects in the same region; the dominant
#   stripe structure remains obvious.
#
# g) Color inference
# - No new color mapping is introduced. Noise colors are arbitrary non-stripe colors but
#   are not semantically meaningful; only the majority stripe color matters.
#
# Review of new solver assumptions
# - Every stripe row still has a unique strict majority color.
# - Consecutive rows from the same stripe share that majority color.
# - Generator guarantees both conditions by using width >= 6 and at most 2 noise pixels
#   per stripe, while keeping adjacent stripe base colors distinct.


def _majority_color(vec: np.ndarray) -> int:
    counts = np.bincount(np.asarray(vec, dtype=int), minlength=10)
    return int(np.argmax(counts))


def _build_output(runs: list[tuple[int, int]]) -> np.ndarray:
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, h in runs[1:]:
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of noisy horizontal color stripes.

    Each stripe may contain 1-2 off-color noise pixels. Recover each stripe's true color
    by majority color per row, then merge consecutive rows with the same majority color.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    row_colors = [_majority_color(g[r]) for r in range(H)]

    runs: list[tuple[int, int]] = []  # (color, height), bottom -> top
    r = int(H - 1)
    while r >= 0:
        col = int(row_colors[r])
        h = 1
        rr = int(r - 1)
        while rr >= 0 and int(row_colors[rr]) == int(col):
            h += 1
            rr -= 1
        runs.append((int(col), int(h)))
        r = int(rr)

    assert runs
    return _build_output(runs)


def _noise_color(true_color: int) -> int:
    c = int(np.random.randint(1, 10))
    while c == int(true_color):
        c = int(np.random.randint(1, 10))
    return int(c)


def _inject_noise_into_stripe(grid: np.ndarray, top: int, height: int, true_color: int) -> int:
    W = int(grid.shape[1])
    stripe_area = int(height * W)
    n_noise = int(np.random.randint(1, min(2, stripe_area) + 1))

    positions: set[tuple[int, int]] = set()

    # If the stripe is only one row tall, corrupt column 0 so single-pixel sampling fails.
    if height == 1:
        positions.add((int(top), 0))

    while len(positions) < n_noise:
        rr = int(top + np.random.randint(0, height))
        cc = int(np.random.randint(0, W))
        positions.add((rr, cc))

    for rr, cc in positions:
        grid[rr, cc] = int(_noise_color(true_color))

    return len(positions)


def _gen_attempt() -> np.ndarray | None:
    W = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if heights[0] + 2 * sum(heights[1:]) > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    runs_bottom_to_top = list(zip(cols, heights))
    expected = _build_output([(int(col), int(h)) for col, h in runs_bottom_to_top])

    stripes_top_to_bottom = list(reversed(runs_bottom_to_top))
    grid = np.vstack([np.full((h, W), col, dtype=int) for col, h in stripes_top_to_bottom])

    top = 0
    total_noise = 0
    for col, h in stripes_top_to_bottom:
        total_noise += int(_inject_noise_into_stripe(grid, int(top), int(h), int(col)))
        top += int(h)

    # Guard against accidental degeneration: every generated input must actually contain
    # noisy, non-uniform evidence for the D13 change and still solve to the intended output.
    if total_noise < n_runs:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
