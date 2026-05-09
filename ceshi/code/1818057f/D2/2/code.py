# Variant of 1818057f
# Dimension: D2 — 变换的目标颜色 (transform)
# Change: 所有'+'字图案统一着色为青色(8/teal)，单一固定目标颜色
#   -> 目标颜色由'+'的邻域密度决定

from __future__ import annotations

import numpy as np
from arc_utils import get_bg

# Design analysis
# a) Concrete behavior:
#    - The object to detect is unchanged: any foreground cell whose 4 orthogonal
#      neighbors are also foreground is a '+' center.
#    - The new branch is purely local. Around a valid center, inspect the four
#      diagonal cells of its 3x3 neighborhood. These are the only cells left
#      after removing the 5 cells of the '+' itself.
#    - If more than half of those four diagonals are foreground, the 5-cell plus
#      becomes teal (8); otherwise it becomes magenta (6).
#    - Example: a plus with 3 occupied diagonals used to become all-8; now that
#      same plus becomes 8, while a plus with only 1 occupied diagonal becomes 6.
# b) Complexity calibration:
#    - There are only two output colors, tied to an immediately visible local
#      cue: sparse versus crowded diagonals. This is richer than the seed task
#      but still learnable from a few examples.
#    - The generator forces both branches to appear repeatedly so the new split
#      is observable instead of collapsing back to a single-color task.
# c) Old implicit assumptions that no longer hold:
#    - The seed solver could unconditionally write 8 for every hit because all
#      pluses shared one target color. That is no longer valid.
#    - The new solver must read density from the original input grid, not from
#      the partially recolored output, otherwise earlier writes could corrupt
#      later density decisions.
# d) Placement legality:
#    - Planned plus centers are spaced by Chebyshev distance >= 5, so their
#      5x5 protected neighborhoods do not overlap.
#    - Each plus gets exactly one extra arm cell, keeping it embedded while
#      staying inside its protected neighborhood.
#    - Diagonal patterns are chosen to avoid creating accidental secondary plus
#      centers on the extended arm.
# e) Visual observability:
#    - Human-solvable signal: count the four corner cells around each cross.
#      Crowded corners -> teal, sparse corners -> magenta.
#    - Every generated grid contains at least one teal plus and one magenta plus.
# f) Visual cleanliness:
#    - Noise is kept away from each plus's 5x5 neighborhood so the decisive local
#      density cue is not muddied by unrelated clutter.
# g) Color inference:
#    - Only two result colors are introduced, both backed by repeated examples of
#      the same shape under two neighborhood-density conditions.

ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PLUS = [(0, 0), *ORTHO]
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
EXT_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' motifs in a single foreground color and recolor them by local diagonal density.

    Input structure:
    - The background is the most common color.
    - One non-background color forms mixed shapes on top of it, including embedded '+' crosses.
    - Around each '+' center, the four diagonal corner cells may or may not also be foreground.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are foreground too.
    - Count how many of the 4 diagonal cells around that center are foreground in the input.
    - If that diagonal count is 3 or 4, recolor the 5 cells of the '+' to teal (8); otherwise recolor them to magenta (6).
    - Leave every other cell unchanged.

    Key insight:
    - The shape test still depends only on the 5-cell cross, but the output color is controlled by the surrounding 4 diagonal corners, so the solver must combine pattern detection with a tiny local density check.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                diag_fg = sum(int(g[r + dr, c + dc]) == fg for dr, dc in DIAGS)
                target = 8 if diag_fg > 2 else 6
                for dr, dc in PLUS:
                    out[r + dr, c + dc] = target
    return out


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                centers.append((r, c))
    return centers


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether painting (r,c) foreground would create a plus centered nearby."""
    H, W = grid.shape
    for cr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        ok = True
        for dr, dc in PLUS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                ok = False
                break
        if ok:
            return True
    return False


def _diag_subsets_for_extension(dir_idx: int) -> dict[int, list[tuple[int, ...]]]:
    blocked_pairs = {
        0: {0, 1},  # north extension blocks NW+NE together
        1: {2, 3},  # south extension blocks SW+SE together
        2: {0, 2},  # west extension blocks NW+SW together
        3: {1, 3},  # east extension blocks NE+SE together
    }
    blocked = blocked_pairs[dir_idx]
    by_count: dict[int, list[tuple[int, ...]]] = {0: [tuple()], 1: [], 2: [], 3: []}
    for mask in range(1, 1 << 4):
        subset = tuple(i for i in range(4) if mask & (1 << i))
        if len(blocked & set(subset)) == 2:
            continue
        if len(subset) <= 3:
            by_count[len(subset)].append(subset)
    return by_count


def _paint_plus_region(
    grid: np.ndarray,
    center: tuple[int, int],
    dir_idx: int,
    diag_subset: tuple[int, ...],
    fg: int,
) -> None:
    r, c = center
    for dr, dc in PLUS:
        grid[r + dr, c + dc] = fg

    er, ec = EXT_DIRS[dir_idx]
    grid[r + 2 * er, c + 2 * ec] = fg

    for idx in diag_subset:
        dr, dc = DIAGS[idx]
        grid[r + dr, c + dc] = fg


def _add_protected_zone(
    protected: set[tuple[int, int]], shape: tuple[int, int], center: tuple[int, int]
) -> None:
    H, W = shape
    r, c = center
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                protected.add((nr, nc))


def _try_add_line(
    grid: np.ndarray,
    fg: int,
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
) -> int:
    H, W = grid.shape
    r = int(rng.integers(0, H))
    c = int(rng.integers(0, W))
    dr, dc = EXT_DIRS[int(rng.integers(0, 4))]
    length = int(rng.integers(1, 5))

    added = 0
    for step in range(length):
        nr, nc = r + dr * step, c + dc * step
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if (nr, nc) in protected or int(grid[nr, nc]) == fg:
            continue
        if _would_create_plus(grid, nr, nc, fg):
            continue
        grid[nr, nc] = fg
        added += 1
    return added


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4
    total = H * W

    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    spaced_centers: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 5 for rr, cc in spaced_centers):
            spaced_centers.append((r, c))

    max_k = min(9, len(spaced_centers), max(2, total // 36))
    if max_k < 2:
        return None
    K = int(rng.integers(2, max_k + 1))
    centers = spaced_centers[:K]

    plus_colors = [8, 6]
    if K > 2:
        plus_colors.extend(rng.choice([6, 8], size=K - 2).tolist())
    rng.shuffle(plus_colors)

    grid = np.full((H, W), bg, dtype=int)
    protected: set[tuple[int, int]] = set()
    planned_colors: dict[tuple[int, int], int] = {}

    for center, plus_color in zip(centers, plus_colors):
        dir_idx = int(rng.integers(0, 4))
        subsets = _diag_subsets_for_extension(dir_idx)
        if plus_color == 8:
            diag_subset = subsets[3][int(rng.integers(0, len(subsets[3])))]
        else:
            sparse_count = int(rng.integers(0, 3))
            diag_subset = subsets[sparse_count][int(rng.integers(0, len(subsets[sparse_count])))]

        _paint_plus_region(grid, center, dir_idx, diag_subset, fg)
        _add_protected_zone(protected, grid.shape, center)
        planned_colors[center] = plus_color

    actual_centers = set(_find_plus_centers(grid, fg))
    if actual_centers != set(centers):
        return None

    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    attempts = 0
    max_attempts = total * 25
    while current_fg < target_fg and attempts < max_attempts:
        current_fg += _try_add_line(grid, fg, protected, rng)
        attempts += 1

    current_fg = int(np.sum(grid != bg))
    if current_fg >= total // 2:
        return None

    actual_centers = set(_find_plus_centers(grid, fg))
    if actual_centers != set(centers):
        return None

    expected = grid.copy()
    for r, c in centers:
        color = planned_colors[(r, c)]
        for dr, dc in PLUS:
            expected[r + dr, c + dc] = color

    if not (np.any(expected == 6) and np.any(expected == 8)):
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
