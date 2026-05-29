# Combo variant of 1818057f (index 1)
# combo_spec: D1/0+D3/0
# Name: 擦除T型图案

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# combo_variant_spec:
#   concretization_and_trace:
#     - Source task behavior:
#         . 4 .        original solver sees a full '+' only when the center has
#         4 4 4        all 4 orthogonal neighbors; then it recolors 5 cells.
#         . 4 .
#     - New fused behavior on a T:
#         . 4 .        a center with exactly 3 orthogonal fg neighbors is now the
#         4 4 4        target, and those 4 cells are erased back to background.
#         . . .
#       So this local patch becomes all background instead of staying yellow.
#     - Contrast case:
#         . 4 .        a true '+' now does nothing, because degree 4 is not an
#         4 4 4        exact T. This prevents the fusion from collapsing back to
#         . 4 .        the original source rule.
#   complexity_calibration:
#     - "Exactly 3 neighbors" is essential. Using "at least 3" would merge T and
#       '+' and make the D3 change visually ambiguous.
#     - Erase is kept structural rather than color-heavy: the board still has one
#       dominant background and one foreground color, so humans infer the rule from
#       missing 4-cell junctions rather than from arbitrary recolor tables.
#     - Each planted T gets at least one extension beyond one of its occupied arms.
#       That keeps the subtraction observable as a notch or gap inside a larger
#       foreground component instead of a trivial disappearing 4-cell sprite.
#   implicit_assumptions_review:
#     - `get_bg(g)` remains valid because the generator keeps background as the
#       most common color.
#     - `main()` still assumes a single non-background color; erase introduces no
#       third color, so selecting the lone non-bg color as foreground is sound.
#     - Detection must read from the immutable input grid `g`, not from `out`,
#       otherwise erasing one T could suppress a later nearby T. The solver keeps
#       that separation.
#     - Because the solver writes the center plus its 3 detected arm cells without
#       conflict resolution, the generator must forbid overlapping T footprints.
#       It does so by enforcing exact degree-3 centers only at planned locations
#       and degree <= 2 everywhere else.
#   placement_legality:
#     - Planned centers stay at least two cells from the border, so all four
#       orthogonal neighbors exist and at least one arm can extend safely.
#     - Candidate cells are added only through `_try_add_cells`, which rejects any
#       out-of-bounds write, any overlap with existing foreground, any accidental
#       extra T center, and any accidental '+' center.
#     - Center spacing stays at L1 >= 4, which keeps T footprints visually
#       separated and avoids shared arm cells.
#   visual_observability:
#     - The key visible regularity is "three-armed junctions disappear; four-armed
#       crosses do not appear in the generated inputs at all."
#     - Every generated board includes all four missing-side orientations at least
#       once, so 2-4 examples are enough for a human to infer rotation-invariant
#       T detection rather than a position-specific special case.
#   cleanliness_and_inferability:
#     - Distractors are admitted only when they preserve the degree invariant,
#       which prevents dense blobs with many confusing branch points.
#     - No hidden color ordering is used in the rule itself; the only cue is local
#       orthogonal structure, which is directly visible in 2-4 examples.
#   new_solver_assumptions_to_match_generator:
#     - Every actual degree-3 foreground junction should be intentional and should
#       be erasable independently. `generate_input()` validates that the detected
#       T centers exactly match the planned center set and that there are no '+'
#       centers, so the solver's exact-neighbor test matches the generated data.

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _t_arm_offsets(grid: np.ndarray, r: int, c: int, fg: int) -> list[tuple[int, int]] | None:
    """Return the occupied orthogonal arm offsets for an exact T center, else None."""
    if int(grid[r, c]) != int(fg):
        return None
    arms = [(dr, dc) for dr, dc in ORTHO_DIRS if int(grid[r + dr, c + dc]) == int(fg)]
    return arms if len(arms) == 3 else None


def _find_t_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _t_arm_offsets(grid, r, c, fg) is not None:
                centers.append((r, c))
    return centers


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTHO_DIRS):
                centers.append((r, c))
    return centers


def _missing_arm_direction(grid: np.ndarray, r: int, c: int, fg: int) -> tuple[int, int] | None:
    arms = _t_arm_offsets(grid, r, c, fg)
    if arms is None:
        return None
    arm_set = set(arms)
    for direction in ORTHO_DIRS:
        if direction not in arm_set:
            return direction
    return None


def _fg_degree(grid: np.ndarray, r: int, c: int, fg: int) -> int:
    H, W = grid.shape
    deg = 0
    for dr, dc in ORTHO_DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == int(fg):
            deg += 1
    return deg


def _respects_planned_t_centers(grid: np.ndarray, fg: int, centers: set[tuple[int, int]]) -> bool:
    """Planned centers have degree 3; every other fg cell has degree at most 2."""
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != int(fg):
                continue
            deg = _fg_degree(grid, r, c, fg)
            if (r, c) in centers:
                if deg != 3:
                    return False
            elif deg >= 3:
                return False
    return True


def _try_add_cells(
    grid: np.ndarray,
    fg: int,
    centers: set[tuple[int, int]],
    cells: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) == int(fg):
            return False
    trial = grid.copy()
    for r, c in cells:
        trial[r, c] = fg
    if not _respects_planned_t_centers(trial, fg, centers):
        return False
    grid[:, :] = trial
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin T junctions in a single foreground color and erase them to background.

    Input structure:
    - The most frequent color is the background.
    - One other color forms sparse-to-moderate foreground paths and branched shapes.
    - Some local 4-cell motifs are T shapes: a foreground center with exactly 3 orthogonal foreground neighbors.

    Rule:
    - Scan interior cells of the input grid.
    - Whenever a foreground pixel has exactly 3 orthogonal foreground neighbors, treat that center plus those 3 neighbors as one T motif.
    - Replace those 4 pixels with the background color.
    - Leave all other cells unchanged; a full '+' is ignored because it has 4 arms, not 3.

    Key insight:
    - The task is controlled by local orthogonal degree. The solver must spot every degree-3 junction in any rotation and carve it away, so the output is the input foreground with T-shaped holes removed.
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
            arms = _t_arm_offsets(g, r, c, fg)
            if arms is None:
                continue
            out[r, c] = bg
            for dr, dc in arms:
                out[r + dr, c + dc] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(12, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    max_k = min(9, max(4, (H * W) // 28))
    K = int(rng.integers(4, max_k + 1))

    centers: list[tuple[int, int]] = []
    center_set: set[tuple[int, int]] = set()
    dir_order = list(map(int, rng.permutation(4)))
    desired_missing = [ORTHO_DIRS[dir_order[i % 4]] for i in range(K)]

    for i in range(K):
        missing = desired_missing[i]
        placed = False
        for _ in range(160):
            cr = int(rng.integers(2, H - 2))
            cc = int(rng.integers(2, W - 2))
            if any(abs(cr - er) + abs(cc - ec) < 4 for er, ec in centers):
                continue

            present_arms = [d for d in ORTHO_DIRS if d != missing]
            cells = {(cr, cc)}
            for dr, dc in present_arms:
                cells.add((cr + dr, cc + dc))

            target_dirs = int(rng.integers(1, min(3, len(present_arms)) + 1))
            chosen_ids = rng.choice(len(present_arms), size=target_dirs, replace=False)
            for idx in chosen_ids:
                dr, dc = present_arms[int(idx)]
                ext_len = int(rng.integers(1, 4))
                for step in range(2, ext_len + 2):
                    cells.add((cr + dr * step, cc + dc * step))

            trial_centers = center_set | {(cr, cc)}
            if _try_add_cells(grid, fg, trial_centers, cells):
                centers.append((cr, cc))
                center_set = trial_centers
                placed = True
                break
        if not placed:
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
        if _try_add_cells(grid, fg, center_set, {(int(r), int(c))}):
            current_fg += 1

    current_fg = int(np.sum(grid != bg))
    if current_fg >= total // 2:
        return None

    inp = grid.copy()
    t_centers = _find_t_centers(inp, fg)
    plus_centers = _find_plus_centers(inp, fg)
    if sorted(t_centers) != sorted(centers):
        return None
    if plus_centers:
        return None
    observed_missing = {
        _missing_arm_direction(inp, r, c, fg)
        for r, c in t_centers
    }
    if observed_missing != set(ORTHO_DIRS):
        return None

    out = main(inp)
    changed = inp != out
    if not np.any(changed):
        return None
    if int(np.sum(changed)) != 4 * len(center_set):
        return None
    if not np.all(inp[changed] == fg):
        return None
    if not np.all(out[changed] == bg):
        return None

    # Reject accidental degeneration back to the source behavior: generated inputs
    # should expose at least one T and no '+', so the original plus-based solver
    # would not perform the same transformation on the same board.
    if len(t_centers) == 0 or len(plus_centers) != 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
