# Variant of 1818057f
# Dimension: D10 — '+'检测的邻域类型 (meta)
# Change: 使用4-正交邻域(上下左右)检测'+'字图案... → 对角邻域检测'X'形

from __future__ import annotations

import numpy as np
from arc_utils import get_bg

# Design analysis for the D10 variant:
# a) Concretization + behavior trace:
#    - The grid still uses one dominant background color and one foreground color.
#    - The target motif is no longer a 5-cell orthogonal cross:
#      {(r,c), (r-1,c), (r+1,c), (r,c-1), (r,c+1)}.
#      It becomes a 5-cell diagonal X:
#      {(r,c), (r-1,c-1), (r-1,c+1), (r+1,c-1), (r+1,c+1)}.
#    - Small trace:
#      Old rule on a center with filled diagonals but empty orthogonals does nothing,
#      because the 4-neighbor test fails immediately.
#      New rule recolors the center plus the 4 diagonal neighbors to teal.
#      This makes the solver change happen at the neighborhood test and at the
#      footprint recoloring step, while background/foreground inference stays unchanged.
# b) Complexity calibration:
#    - The variant keeps the same color regime, density band, and local 5-cell recolor size.
#    - To keep the X rule visually inferable, the generator embeds each X into larger diagonal
#      structures and reserves the four orthogonal neighbors around every intended center.
#      That produces stable "X-only" motifs instead of dense blobs where diagonal/orthogonal
#      evidence would be ambiguous to a human viewer.
# c) Hidden assumptions audit:
#    - `get_bg` and "single non-bg color in input" still hold, so foreground extraction can stay.
#    - The interior scan `range(1, H-1)` still works because diagonal neighbors need a 1-cell margin.
#    - The old solver's orthogonal direction lists were the only rule-specific assumption; both the
#      center test and recolor footprint must switch to diagonal offsets.
#    - The old generator's helper only blocked new '+' centers. In this variant that is insufficient:
#      diagonal ray extensions can accidentally synthesize second-order X centers, so legality checks
#      must reject any newly created X center that was not intentionally planted.
# d) Placement legality:
#    - Intended centers are placed at least 2 cells from the border and with Chebyshev distance >= 4,
#      preventing footprint overlap and keeping diagonal extensions from colliding easily.
#    - Extensions and distractor fills are only accepted when they do not create any unintended X
#      center, and orthogonal neighbors around motif centers are reserved so the main solver sees
#      clean diagonal motifs.
# e) Visual observability:
#    - Each generated sample contains at least one center that satisfies the diagonal X test while
#      failing the old orthogonal plus test. This guarantees the D10 change is visible from input/output
#      examples and does not silently collapse back to the original behavior.
# f) Visual cleanliness:
#    - Large connected foreground still exists via diagonal arm extensions, but motif centers are given
#      a small orthogonal moat. That reduces local clutter and makes the recolored teal Xs readable.
# g) Color-change inferability:
#    - Only one recolor is used: foreground -> teal on the detected X footprint.
#      No color-id-based branching or hidden ordering is introduced.

_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_FOOTPRINT = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect compact diagonal 'X' motifs in a single foreground color and recolor them teal.

    Input structure:
    - The background is the most frequent color in the grid.
    - One foreground color forms a mixture of scattered cells, diagonal strands, and embedded 'X' motifs.

    Rule:
    - An 'X' center is any foreground pixel whose 4 diagonal neighbors are also foreground.
    - Recolor that center and those 4 diagonal neighbors to teal (8).
    - Leave every other pixel unchanged.

    Key insight: The match is purely local and diagonal, so the solver highlights every 5-cell X footprint
    even when it sits inside a larger foreground component rather than appearing as an isolated symbol.
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
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in _DIAGONALS):
                for dr, dc in _X_FOOTPRINT:
                    out[r + dr, c + dc] = 8
    return out


def _would_create_unintended_x(
    grid: np.ndarray, r: int, c: int, bg: int, allowed_centers: set[tuple[int, int]]
) -> bool:
    """Check whether occupying (r,c) would create any new X center outside `allowed_centers`."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in _X_FOOTPRINT:
            nr, nc = cr + dr, cc + dc
            occupied = (nr, nc) == (r, c) or int(grid[nr, nc]) != bg
            if not occupied:
                all_fg = False
                break
        if all_fg and (cr, cc) not in allowed_centers:
            return True
    return False


def _count_x_only_centers(grid: np.ndarray, fg: int) -> int:
    """Count centers that satisfy the new diagonal rule and fail the old orthogonal '+' rule."""
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            has_x = all(int(grid[r + dr, c + dc]) == fg for dr, dc in _DIAGONALS)
            has_plus = all(int(grid[r + dr, c + dc]) == fg for dr, dc in _ORTHOGONALS)
            if has_x and not has_plus:
                count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg, mark = 2, 4, 8

    output = np.full((H, W), bg, dtype=int)

    target_k = int(rng.integers(1, 10))
    centers: list[tuple[int, int]] = []
    for _ in range(target_k * 120):
        if len(centers) >= target_k:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        if all(max(abs(cr - er), abs(cc - ec)) >= 4 for er, ec in centers):
            centers.append((cr, cc))

    if not centers:
        return None

    allowed_centers = set(centers)
    reserved = set()

    for cr, cc in centers:
        for dr, dc in _X_FOOTPRINT:
            output[cr + dr, cc + dc] = mark
        for dr, dc in _ORTHOGONALS:
            reserved.add((cr + dr, cc + dc))

    for cr, cc in centers:
        n_ext = int(rng.integers(1, 4))
        chosen = rng.choice(len(_DIAGONALS), size=n_ext, replace=False)
        for idx in chosen:
            dr, dc = _DIAGONALS[int(idx)]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = int(rng.integers(1, 4))
            for step in range(1, ext_len + 1):
                nr = tip_r + dr * step
                nc = tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if int(output[nr, nc]) != bg:
                    break
                if _would_create_unintended_x(output, nr, nc, bg, allowed_centers):
                    break
                output[nr, nc] = fg

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(output != bg))

    bg_positions = list(zip(*np.where(output == bg)))
    rng.shuffle(bg_positions)

    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if (int(r), int(c)) in reserved:
            continue
        if _would_create_unintended_x(output, int(r), int(c), bg, allowed_centers):
            continue
        output[int(r), int(c)] = fg
        current_fg += 1

    input_grid = output.copy()
    input_grid[input_grid == mark] = fg

    if int(np.sum(input_grid != bg)) >= total // 2:
        return None
    if _count_x_only_centers(input_grid, fg) != len(centers):
        return None
    if not np.array_equal(main(input_grid), output):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
