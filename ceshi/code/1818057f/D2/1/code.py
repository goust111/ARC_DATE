# Variant of 1818057f
# Dimension: D2 — 变换的目标颜色 (transform)
# Change: 所有'+'字图案统一着色为青色(8/teal)，单一固定目标颜色... → 目标颜色由'+'数量的序号编码

from __future__ import annotations

import numpy as np
from arc_utils import get_bg

PLUS_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
PALETTE = (3, 8, 6)  # green, teal, pink; cycle in row-major plus order


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior:
#    - Input stays the same kind of scene: red background, one yellow foreground color,
#      and several embedded thin '+' motifs hidden inside larger yellow shapes.
#    - Output no longer paints every '+' with one fixed color. Instead, all plus centers
#      are sorted by reading order: smaller row first, then smaller column.
#      The 1st plus uses palette[0], the 2nd palette[1], the 3rd palette[2], the 4th
#      wraps back to palette[0], and so on.
#    - Example: if centers are at (2,7), (5,3), (5,11), (9,4), then their cross colors
#      are 3, 8, 6, 3 in that order. Under the original rule all four would have become 8.
#      The changed step is therefore the output-color assignment after detection.
#
# b) Complexity calibration:
#    - A 3-color palette is small enough to infer from a few examples, but not so small
#      that the task collapses into a single recolor.
#    - The generator biases toward 4+ pluses whenever the grid can hold them, so one
#      sample often already exhibits the wraparound from the 3rd to the 4th plus.
#    - We keep one foreground input color and the original plus geometry so the new
#      difficulty is sequence reasoning, not object recognition.
#
# c) Hidden assumptions in the original main():
#    - `bg = get_bg(...)` still works because background remains the majority color.
#    - `fg = first non-bg color` still works because inputs still use exactly one
#      foreground color before transformation.
#    - The neighbor test for a plus center is unchanged.
#    - The original unconditional `out[...] = 8` is no longer valid; target color now
#      depends on how many pluses were encountered before this one.
#    - To avoid color-assignment bugs, the solver first collects centers from the
#      unchanged input grid and only then recolors them in row-major order.
#
# d) Placement legality:
#    - Plus centers are kept away from the border and separated by L1 >= 4, preserving
#      non-overlapping 5-cell cross footprints.
#    - Extensions only grow into background cells, so they do not overwrite another plus.
#    - Distractor cells are added only when they do not create an unplanned plus center.
#    - Final validation checks that the actual detected centers match the planned set.
#
# e) Visual observability:
#    - The ordering cue is visible because row-major scan order is spatial: top rows are
#      colored earlier than lower rows, and ties break left-to-right.
#    - The palette cycle is visible because many samples contain at least 4 pluses.
#      That makes the repeated first color reappear later in the scan.
#
# f) Visual cleanliness:
#    - Crosses remain thin and separated, and distractors stay yellow. Only the 5 cells
#      of each detected plus change color, so the transformed structures remain readable.
#
# g) Color-rule inferability:
#    - Only three output colors are introduced, and they recur across samples.
#    - Their assignment is tied to a simple geometric order visible in the grid, not to
#      invisible ids or arbitrary per-instance mappings.


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in PLUS_OFFSETS[1:]):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' shapes in a single foreground color and recolor them by scan index.

    Input structure:
    - bg = most common color.
    - One foreground color forms mixed yellow shapes on the bg, and some local 5-cell '+' motifs are embedded inside those shapes.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Sort all such centers from top to bottom, then left to right; recolor each center and its 4 neighbors using a repeating palette (3, 8, 6) by that order.
    - All other pixels remain unchanged.

    Key insight: Detection is still purely local, but the output color is no longer fixed; it depends on the global row-major rank of each detected '+' among all pluses in the grid.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    centers = _find_plus_centers(g, fg)
    for idx, (r, c) in enumerate(centers):
        color = PALETTE[idx % len(PALETTE)]
        for dr, dc in PLUS_OFFSETS:
            out[r + dr, c + dc] = color
    return out


def _would_create_unplanned_plus(
    grid: np.ndarray, r: int, c: int, fg: int, allowed_centers: set[tuple[int, int]]
) -> bool:
    """Return True iff painting (r,c) as fg would create a plus center outside allowed_centers."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_OFFSETS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg and (cr, cc) not in allowed_centers:
            return True
    return False


def _build_output(grid: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for idx, (r, c) in enumerate(sorted(centers)):
        color = PALETTE[idx % len(PALETTE)]
        for dr, dc in PLUS_OFFSETS:
            out[r + dr, c + dc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    placed: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in placed):
            placed.append((cr, cc))

    max_k = min(8, len(placed))
    if max_k < 3:
        return None
    min_k = 4 if max_k >= 4 else 3
    K = int(rng.integers(min_k, max_k + 1))
    centers = placed[:K]
    center_set = set(centers)

    for cr, cc in centers:
        for dr, dc in PLUS_OFFSETS:
            grid[cr + dr, cc + dc] = fg

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(centers)
    for cr, cc in centers:
        n_ext = int(rng.integers(1, 3))
        chosen = rng.choice(4, size=n_ext, replace=False)
        attached = False
        for di in chosen:
            dr, dc = dirs[int(di)]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = int(rng.integers(1, 4))
            grew = False
            for step in range(1, ext_len + 1):
                nr, nc = tip_r + dr * step, tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if int(grid[nr, nc]) != bg:
                    break
                grid[nr, nc] = fg
                grew = True
            attached = attached or grew
        if not attached:
            return None

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_unplanned_plus(grid, int(r), int(c), fg, center_set):
            grid[int(r), int(c)] = fg
            current_fg += 1

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    actual_centers = _find_plus_centers(grid, fg)
    if sorted(actual_centers) != sorted(centers):
        return None

    inp = grid.copy()
    out = _build_output(inp, centers)
    if np.array_equal(out, inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
