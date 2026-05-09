# Variant of 1818057f
# Dimension: D1 — 核心变换操作：重着色规则 (transform)
# Change: 识别'+'字图案后，将构成'+'的5个像素从前景色(黄色4)统一重着色为青色(teal 8)，其余像素不变
#   → 根据臂延伸长度条件性重着色：只有至少一条臂从中心向外连续延伸超过 2 个前景像素时才重着色。

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import get_bg


# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior and trace
# - A candidate '+' still means one foreground center whose 4 orthogonal neighbors are foreground.
# - The new decision point is not the local cross itself, but whether one arm continues as a straight
#   foreground run for at least 3 cells away from the center in one direction.
# - Example: if a center has arm lengths [up=1, down=1, left=2, right=1], the old task recolors it
#   because the local 5-cell cross exists; the variant leaves it yellow because no arm length exceeds 2.
#   If the lengths are [1, 1, 3, 1], the same local cross is now recolored because the left arm is long.
# - Therefore the changed logic lives inside solver-side motif qualification, and generator-side arm
#   construction must explicitly create both short-arm and long-arm '+' motifs.
#
# b) Complexity calibration
# - Using a single threshold (arm length >= 3) keeps the rule visible and learnable.
# - Each generated grid includes both eligible and ineligible '+' centers, so the transformation cannot
#   collapse back to the original "recolor every plus" behavior.
#
# c) Hidden assumptions in the original main()
# - Original code assumed the immediate 4-neighbor check was sufficient. That is no longer enough.
# - Original unconditional recoloring was harmless because every detected center was supposed to change.
#   In the variant it would overpaint short-arm crosses, so recoloring must be gated by a second check.
# - The solver must read arm lengths from the original input grid, not from an already edited output grid,
#   otherwise earlier teal writes could interfere with later foreground-run measurements.
#
# d) Placement legality
# - Motifs are placed with a 4-neighbor moat, so different components never merge and intended centers
#   never overlap or accidentally lengthen each other's arms.
# - Distractors are added only outside the motif moat, and each added cell is screened so it cannot create
#   any new plus center. This also prevents distractors from turning a short-arm motif into a long-arm one.
#
# e/f/g) Visibility and cleanliness
# - The only changed color is still yellow -> teal, preserving the original visual language.
# - Samples contain repeated examples of both branches in the same palette, making the threshold rule
#   inferable from 2-3 examples without relying on hidden numeric color ids.
# - The moat keeps the scene readable: long-arm and short-arm crosses remain visually separable.


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    if not (1 <= r < H - 1 and 1 <= c < W - 1):
        return False
    if int(grid[r, c]) != fg:
        return False
    return all(int(grid[r + dr, c + dc]) == fg for dr, dc in DIRS)


def _arm_length(grid: np.ndarray, r: int, c: int, dr: int, dc: int, fg: int) -> int:
    H, W = grid.shape
    length = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == fg:
        length += 1
        nr += dr
        nc += dc
    return length


def _qualifies_long_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if not _is_plus_center(grid, r, c, fg):
        return False
    return max(_arm_length(grid, r, c, dr, dc, fg) for dr, dc in DIRS) >= 3


def _find_plus_centers(grid: np.ndarray, fg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    centers: list[tuple[int, int]] = []
    qualified: list[tuple[int, int]] = []
    H, W = grid.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
                if _qualifies_long_plus(grid, r, c, fg):
                    qualified.append((r, c))
    return centers, qualified


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect foreground '+' motifs and recolor only the ones with a long straight arm.

    Input structure:
    - The background is the most frequent color, typically red.
    - A single non-background color, typically yellow, forms thin crosses, arm extensions, and distractor fragments.

    Rule:
    - A candidate '+' is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Measure the consecutive foreground run length from that center along each of the 4 arm directions.
    - Recolor the center and its 4 immediate arm pixels to teal (8) only if at least one arm has length 3 or more.
    - Leave short-arm '+' motifs and every non-'+' foreground pixel unchanged.

    Key insight:
    - The local 5-cell cross is necessary but no longer sufficient; the solver must inspect how far the surrounding
      component keeps going in a straight line, because only '+' motifs embedded in a larger arm extension change color.
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
            if _qualifies_long_plus(g, r, c, fg):
                for dr, dc in [(0, 0), *DIRS]:
                    out[r + dr, c + dc] = 8
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether setting one cell to fg would create a plus center at that cell or a 4-neighbor."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in [(0, 0), *DIRS]:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _sample_arm_lengths(rng: np.random.Generator, want_long: bool) -> list[int]:
    lengths = [1, 1, 1, 1]
    if want_long:
        long_dir = int(rng.integers(0, 4))
        lengths[long_dir] = int(rng.integers(3, 5))  # total arm length from center: 3 or 4
        extra_dirs = rng.choice([d for d in range(4) if d != long_dir], size=int(rng.integers(0, 3)), replace=False)
        for di in np.atleast_1d(extra_dirs):
            lengths[int(di)] = int(rng.integers(1, 3))
    else:
        n_extended = int(rng.integers(1, 3))
        chosen = rng.choice(4, size=n_extended, replace=False)
        for di in np.atleast_1d(chosen):
            lengths[int(di)] = 2
    return lengths


def _motif_cells(cr: int, cc: int, arm_lengths: list[int]) -> set[tuple[int, int]]:
    cells = {(cr, cc)}
    for (dr, dc), length in zip(DIRS, arm_lengths):
        for step in range(1, length + 1):
            cells.add((cr + dr * step, cc + dc * step))
    return cells


def _neighbors4_of(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    moat = set(cells)
    for r, c in cells:
        for dr, dc in DIRS:
            moat.add((r + dr, c + dc))
    return moat


def _place_motifs(
    grid: np.ndarray, rng: np.random.Generator, kinds: list[bool], fg: int
) -> tuple[list[tuple[int, int]], list[bool]] | None:
    H, W = grid.shape
    occupied_moat: set[tuple[int, int]] = set()
    placed_centers: list[tuple[int, int]] = []
    placed_kinds: list[bool] = []

    for want_long in kinds:
        placed = False
        for _ in range(250):
            arm_lengths = _sample_arm_lengths(rng, want_long)
            cr = int(rng.integers(1, H - 1))
            cc = int(rng.integers(1, W - 1))
            cells = _motif_cells(cr, cc, arm_lengths)
            if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
                continue
            if any((r, c) in occupied_moat for r, c in cells):
                continue
            moat = _neighbors4_of(cells)
            if any(not (0 <= r < H and 0 <= c < W) for r, c in moat if (r, c) not in cells):
                continue
            for r, c in cells:
                grid[r, c] = fg
            occupied_moat |= moat
            placed_centers.append((cr, cc))
            placed_kinds.append(want_long)
            placed = True
            break
        if not placed:
            return None

    return placed_centers, placed_kinds


def _try_add_component(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    changed: list[tuple[int, int]] = []
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            for rr, cc in changed:
                grid[rr, cc] = bg
            return False
        if (r, c) in forbidden or int(grid[r, c]) != bg:
            for rr, cc in changed:
                grid[rr, cc] = bg
            return False
        if _would_create_plus(grid, r, c, fg):
            for rr, cc in changed:
                grid[rr, cc] = bg
            return False
        grid[r, c] = fg
        changed.append((r, c))
    return True


def _sprinkle_distractors(
    grid: np.ndarray,
    rng: np.random.Generator,
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
    target_fg: int,
) -> None:
    H, W = grid.shape
    attempts = 0
    while int(np.sum(grid != bg)) < target_fg and attempts < 3000:
        attempts += 1
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W))
        shape_type = int(rng.integers(0, 3))
        if shape_type == 0:
            cells = [(r, c)]
        else:
            dr, dc = DIRS[int(rng.integers(0, 4))]
            length = 2 if shape_type == 1 else 3
            cells = [(r + dr * step, c + dc * step) for step in range(length)]
        _try_add_component(grid, cells, fg, bg, forbidden)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    area = H * W
    max_k = min(7, max(2, area // 45))
    K = int(rng.integers(2, max_k + 1))

    n_long = int(rng.integers(1, K))
    kinds = [True] * n_long + [False] * (K - n_long)
    rng.shuffle(kinds)

    placed = _place_motifs(grid, rng, kinds, fg)
    if placed is None:
        return None
    intended_centers, intended_kinds = placed

    motif_cells = set(map(tuple, np.argwhere(grid == fg)))
    forbidden = _neighbors4_of(motif_cells)

    total = H * W
    target_density = float(rng.uniform(0.24, 0.38))
    target_fg = int(total * target_density)
    _sprinkle_distractors(grid, rng, fg, bg, forbidden, target_fg)

    fg_count = int(np.sum(grid != bg))
    if fg_count >= total // 2 or fg_count < int(total * 0.18):
        return None

    centers, qualified = _find_plus_centers(grid, fg)
    qualified_set = set(qualified)
    center_set = set(centers)
    intended_center_set = set(intended_centers)
    intended_qualified_set = {center for center, want_long in zip(intended_centers, intended_kinds) if want_long}

    if center_set != intended_center_set:
        return None
    if qualified_set != intended_qualified_set:
        return None
    if not qualified or len(qualified) == len(centers):
        return None

    out = main(grid)
    if not np.any(out == 8):
        return None
    if np.array_equal(out, _old_main(grid)):
        return None

    return grid


def _old_main(grid: np.ndarray) -> np.ndarray:
    """Reference for anti-degeneracy checks: recolor every local plus regardless of arm length."""
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
            if _is_plus_center(g, r, c, fg):
                for dr, dc in [(0, 0), *DIRS]:
                    out[r + dr, c + dc] = 8
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
