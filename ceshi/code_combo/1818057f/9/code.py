# Combo variant of 1818057f (index 9)
# combo_spec: D1/3+D8/0+D14/0
# Name: 唯一目标条件判断+高迷惑干扰

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The original task recolored every local '+' center: any foreground cell with all 4 orthogonal
#   neighbors present immediately changed together with those 4 arm pixels.
# - The fused variant keeps that local '+' definition, but constrains the scene to exactly one real
#   plus center. Around it, the grid contains many T-shaped distractors that have only 3 orthogonal
#   neighbors and therefore almost match the target without actually qualifying.
# - The unique plus is then split into 2 cases. Example 1: arm lengths [1, 1, 2, 1] still form a
#   true plus, but no arm extends past length 2, so output equals input. Example 2: [1, 1, 3, 1]
#   has the same local 5-cell cross plus one longer straight continuation, so only that center and
#   its 4 immediate arm pixels recolor to teal.
# - Relative to the original task, the solver must now do two checks in sequence: distinguish the one
#   genuine '+' from many near-miss T distractors, then test whether that plus has a long enough arm.
#
# b) Complexity calibration
# - The long-arm threshold stays binary and simple: arm length >= 3 in at least one direction.
# - Visual difficulty comes from quantity and similarity of distractors instead of from adding many
#   unrelated object types. T distractors may themselves have long stems, so "long line" alone is not
#   enough; the missing fourth arm still matters.
# - To keep the rule learnable across a small ARC dataset, generation alternates between long-arm and
#   short-arm true pluses, ensuring both the recolor and no-op branches appear repeatedly.
#
# c) Hidden assumptions in the original main()
# - Original code assumed every detected plus should be recolored. That assumption fails here, so the
#   recolor step must be gated by an arm-length test.
# - Original code also tolerated any number of plus centers. Here the transformation is defined around a
#   unique target, so the new solver explicitly requires exactly one plus center before applying change.
# - Arm-length measurement must read from the untouched input grid. Writing teal first and then continuing
#   to scan would corrupt later length checks if multiple centers existed, so the solver keeps reads on g
#   and writes on out.
#
# d) Placement legality
# - The generator places the single true plus and every T distractor as separate thin arm objects with a
#   4-neighbor moat. This prevents overlap, border spill, and accidental completion of a T into a plus.
# - Validation re-runs structural checks after placement: there must be exactly one plus center, that
#   center must match the intended long/short branch, and the number of T centers must stay high.
#
# e) Visual observability
# - A human can inspect input/output pairs and notice that only one object is ever even eligible: the
#   only shape with all 4 arms. Comparing changed and unchanged examples then reveals the second rule:
#   only true pluses with a visibly longer arm recolor.
#
# f) Visual cleanliness
# - Objects do not touch orthogonally, so each T silhouette remains legible. Using only background, one
#   foreground color, and teal output preserves the original palette while avoiding clutter from extra
#   labels or markers.
#
# g) Change inferability
# - The only color change is still fg -> teal on a 5-cell cross, but that change now correlates with two
#   visible structural cues: being the sole full plus and having a straight arm of length at least 3.
#   No hidden color ordering or non-visual metadata is needed to infer the rule.


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_COUNTER = 0


def _orth_fg_count(grid: np.ndarray, r: int, c: int, fg: int) -> int:
    return sum(int(grid[r + dr, c + dc]) == fg for dr, dc in DIRS)


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    if not (1 <= r < H - 1 and 1 <= c < W - 1):
        return False
    if int(grid[r, c]) != fg:
        return False
    return _orth_fg_count(grid, r, c, fg) == 4


def _arm_length(grid: np.ndarray, r: int, c: int, dr: int, dc: int, fg: int) -> int:
    H, W = grid.shape
    length = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == fg:
        length += 1
        nr += dr
        nc += dc
    return length


def _plus_has_long_arm(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if not _is_plus_center(grid, r, c, fg):
        return False
    return max(_arm_length(grid, r, c, dr, dc, fg) for dr, dc in DIRS) >= 3


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def _count_t_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    total = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) == fg and _orth_fg_count(grid, r, c, fg) == 3:
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the single genuine '+' hidden among many T-like lookalikes and recolor it only when one arm is long enough.

    Input structure:
    - The background is the most frequent color, typically red.
    - A single foreground color, typically yellow, forms one true thin plus and many thin T-shaped near-misses.
    - The T distractors can have long stems, but each is missing one of the 4 orthogonal arms.

    Rule:
    - First locate foreground cells whose 4 orthogonal neighbors are also foreground; the intended inputs contain exactly one such '+' center.
    - Measure the straight foreground run length from that center in each orthogonal direction.
    - Recolor the center and its 4 immediate arm pixels to teal (8) only if at least one arm has length 3 or more.
    - If the unique plus has no arm longer than 2, or if no unique plus exists, leave the grid unchanged.

    Key insight:
    - The challenge is layered: most cross-like objects are fake because they lack one arm, and even the real plus changes only when
      it sits inside a longer straight extension. So the solver must combine exact local shape recognition with a second size test.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    centers = _find_plus_centers(g, fg)
    if len(centers) != 1:
        return out

    r, c = centers[0]
    if _plus_has_long_arm(g, r, c, fg):
        for dr, dc in [(0, 0), *DIRS]:
            out[r + dr, c + dc] = 8
    return out


def _shape_cells(cr: int, cc: int, arm_lengths: list[int]) -> set[tuple[int, int]]:
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


def _sample_plus_arm_lengths(rng: np.random.Generator, want_long: bool) -> list[int]:
    lengths = [1, 1, 1, 1]
    if want_long:
        long_dir = int(rng.integers(0, 4))
        lengths[long_dir] = int(rng.integers(3, 5))
        extra_count = int(rng.integers(0, 3))
        if extra_count:
            choices = rng.choice([d for d in range(4) if d != long_dir], size=extra_count, replace=False)
            for di in np.atleast_1d(choices):
                lengths[int(di)] = int(rng.integers(1, 3))
    else:
        extra_count = int(rng.integers(1, 4))
        choices = rng.choice(4, size=extra_count, replace=False)
        for di in np.atleast_1d(choices):
            lengths[int(di)] = 2
    return lengths


def _sample_t_arm_lengths(rng: np.random.Generator) -> list[int]:
    missing = int(rng.integers(0, 4))
    lengths = [1, 1, 1, 1]
    lengths[missing] = 0
    present = [d for d in range(4) if d != missing]

    stem_dir = int(rng.choice(present))
    lengths[stem_dir] = int(rng.integers(2, 5))

    extra_count = int(rng.integers(0, 3))
    other_dirs = [d for d in present if d != stem_dir]
    if extra_count:
        choices = rng.choice(other_dirs, size=min(extra_count, len(other_dirs)), replace=False)
        for di in np.atleast_1d(choices):
            lengths[int(di)] = int(rng.integers(1, 3))
    return lengths


def _try_place_shape(
    grid: np.ndarray,
    rng: np.random.Generator,
    arm_lengths: list[int],
    fg: int,
    occupied_moat: set[tuple[int, int]],
) -> tuple[int, int] | None:
    H, W = grid.shape
    for _ in range(400):
        cr = int(rng.integers(1, H - 1))
        cc = int(rng.integers(1, W - 1))
        cells = _shape_cells(cr, cc, arm_lengths)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
            continue
        moat = _neighbors4_of(cells)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in moat):
            continue
        if any((r, c) in occupied_moat for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = fg
        occupied_moat |= moat
        return cr, cc
    return None


def _old_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    for r, c in _find_plus_centers(g, fg):
        for dr, dc in [(0, 0), *DIRS]:
            out[r + dr, c + dc] = 8
    return out


def _gen_attempt(want_long: bool) -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(14, 23))
    W = int(rng.integers(14, 23))
    bg, fg = 2, 4
    grid = np.full((H, W), bg, dtype=int)

    occupied_moat: set[tuple[int, int]] = set()

    plus_lengths = _sample_plus_arm_lengths(rng, want_long)
    plus_center = _try_place_shape(grid, rng, plus_lengths, fg, occupied_moat)
    if plus_center is None:
        return None

    area = H * W
    # Keep the field visually busy so the single true plus is consistently hidden among many T near-misses.
    min_t = 7 if area < 260 else 8
    max_t = min(18, max(min_t, area // 22))
    target_t = int(rng.integers(min_t, max_t + 1))

    for _ in range(target_t):
        arm_lengths = _sample_t_arm_lengths(rng)
        if _try_place_shape(grid, rng, arm_lengths, fg, occupied_moat) is None:
            return None

    centers = _find_plus_centers(grid, fg)
    if centers != [plus_center]:
        return None

    long_centers = [center for center in centers if _plus_has_long_arm(grid, center[0], center[1], fg)]
    expected_long = [plus_center] if want_long else []
    if long_centers != expected_long:
        return None

    t_count = _count_t_centers(grid, fg)
    if t_count < target_t:
        return None

    fg_count = int(np.sum(grid != bg))
    if fg_count >= (H * W) // 2:
        return None

    out = main(grid)
    if want_long:
        if not np.any(out == 8):
            return None
    else:
        if not np.array_equal(out, grid):
            return None
        if np.array_equal(out, _old_main(grid)):
            return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    want_long = (_GEN_COUNTER % 2) == 0
    _GEN_COUNTER += 1

    for _ in range(500):
        result = _gen_attempt(want_long)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after 500 attempts (want_long={want_long})")
