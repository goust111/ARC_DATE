# Combo variant of 1818057f (index 15)
# combo_spec: D3/0+D4/1
# Name: 双色层差异化变换T型

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import get_bg
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import get_bg


# Design analysis
# ---
# a_concretization_and_trace:
#   - The source task looked for 5-cell same-color '+' motifs and always recolored them.
#   - This combo changes two dimensions at once:
#       1. geometry: the target is now an exact T, meaning a same-color center with
#          exactly 3 orthogonal same-color neighbors and 1 missing side;
#       2. semantics: yellow T motifs recolor to teal, while green T motifs erase to bg.
#   - Local trace on a concrete patch:
#         2 4 2          2 3 2
#         4 4 4    and   3 3 3
#         2 2 2          2 2 2
#     Under the source rule, neither patch changes because neither center has 4 arms.
#     Under this combo, the yellow 4-cell T becomes teal, while the green 4-cell T
#     disappears into the red background. The two branches share the same detector and
#     differ only in the write value.
# b_complexity_calibration:
#   - The T test must use "exactly 3" same-color neighbors. Using "at least 3" would
#     collapse T and '+' together and blur the D3 change.
#   - Only two active foreground colors are used: yellow (recolor) and green (erase).
#     That is enough to encode the mapping while still being easy to infer visually.
#   - Every generated board contains at least one yellow T and one green T, so both
#     action branches are visible within a single example instead of being split across
#     examples.
# c_implicit_assumptions_review:
#   - The source `fg = colors[0]` assumption breaks because there are now two meaningful
#     foreground layers. The solver must scan yellow and green separately.
#   - The source `all(four neighbors == fg)` predicate breaks because T centers have
#     degree 3, not 4. It is replaced with an exact-neighbor-count test.
#   - The source `out[...] = 8` assumption breaks because only yellow T motifs recolor;
#     green T motifs write the background color instead.
#   - Reading from `g` rather than the evolving `out` remains important. If the erase
#     branch read from `out`, an earlier write could hide a later T center.
# d_placement_legality:
#   - T centers are placed at least 2 cells away from borders so all 4 orthogonal slots
#     exist and optional extensions can stay in-bounds.
#   - Planned centers across both colors keep L1 distance >= 4, preventing core-footprint
#     overlap and keeping transformed regions visually separate.
#   - The generator enforces a same-color degree invariant: planned centers have degree 3,
#     every other same-color pixel has degree at most 2. That forbids accidental extra T
#     centers and forbids '+' motifs entirely.
# e_visual_observability:
#   - A human sees two repeated structures with the same T geometry but different colors.
#     In output, all yellow T cells become teal and all green T cells become red bg.
#   - Because both colors appear in the same scenes and the geometry is shared, the color
#     to action mapping is inferable from 2-3 train pairs without hidden metadata.
# f_cleanliness:
#   - The boards keep the source task's red background and medium foreground density, but
#     footprints do not overlap and distractors never create competing degree-3 junctions.
#   - Extensions and scattered distractors preserve the "embedded in larger thin shapes"
#     flavor without producing dense blobs that obscure T motifs.
# g_change_inference:
#   - The changed output is tied to visible structure plus visible color only. There is no
#     reliance on color ordering, numeric ids, or non-local sorting. The generator always
#     instantiates both mappings so the rule cannot degenerate into a single-color task.


BG = 2
YELLOW = 4
GREEN = 3
TEAL = 8

ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTIVE_COLORS = (YELLOW, GREEN)
ACTION_BY_COLOR = {
    YELLOW: TEAL,
    GREEN: BG,
}


def _t_arm_offsets(grid: np.ndarray, r: int, c: int, fg: int) -> list[tuple[int, int]] | None:
    """Return the 3 occupied orthogonal offsets for an exact same-color T center."""
    if int(grid[r, c]) != fg:
        return None
    arms = [(dr, dc) for dr, dc in ORTHO if int(grid[r + dr, c + dc]) == fg]
    return arms if len(arms) == 3 else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect exact T motifs in two foreground colors and apply a color-specific transform.

    Input structure:
    - bg is the most common color.
    - Two foreground colors are mixed on that bg: yellow and green.
    - Same-color thin components can contain embedded T junctions whose center has exactly
      3 orthogonal neighbors of the same color.

    Rule:
    - A T center is any non-border foreground pixel with exactly 3 same-color orthogonal neighbors.
    - For yellow T motifs, recolor the center and its 3 arm cells to teal (8).
    - For green T motifs, replace that same 4-cell T footprint with the background color.
    - Full '+' junctions do not qualify because they have 4 arms. All other pixels stay unchanged.

    Key insight:
    - The task fuses shape recognition with color-conditioned output: both colors use the same
      local T detector, but yellow means "highlight this T" while green means "erase this T".
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    for fg in ACTIVE_COLORS:
        replacement = TEAL if fg == YELLOW else bg
        for r in range(1, g.shape[0] - 1):
            for c in range(1, g.shape[1] - 1):
                arms = _t_arm_offsets(g, r, c, fg)
                if arms is None:
                    continue
                out[r, c] = replacement
                for dr, dc in arms:
                    out[r + dr, c + dc] = replacement
    return out


def _same_color_degree(grid: np.ndarray, r: int, c: int, fg: int) -> int:
    degree = 0
    H, W = grid.shape
    for dr, dc in ORTHO:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == fg:
            degree += 1
    return degree


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
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                centers.append((r, c))
    return centers


def _respects_planned_t_centers(
    grid: np.ndarray,
    centers_by_color: dict[int, set[tuple[int, int]]],
) -> bool:
    """Only planned centers may have same-color degree 3; no pixel may reach degree 4."""
    H, W = grid.shape
    for fg in ACTIVE_COLORS:
        planned = centers_by_color[fg]
        for r in range(H):
            for c in range(W):
                if int(grid[r, c]) != fg:
                    continue
                degree = _same_color_degree(grid, r, c, fg)
                if (r, c) in planned:
                    if degree != 3:
                        return False
                elif degree >= 3:
                    return False
    return True


def _try_add_cells(
    grid: np.ndarray,
    fg: int,
    centers_by_color: dict[int, set[tuple[int, int]]],
    cells: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != BG:
            return False
    trial = grid.copy()
    for r, c in cells:
        trial[r, c] = fg
    if not _respects_planned_t_centers(trial, centers_by_color):
        return False
    grid[:, :] = trial
    return True


def _all_t_to_teal_output(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int)
    for fg in ACTIVE_COLORS:
        for r, c in _find_t_centers(grid, fg):
            out[r, c] = TEAL
            for dr, dc in _t_arm_offsets(grid, r, c, fg) or []:
                out[r + dr, c + dc] = TEAL
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    grid = np.full((H, W), BG, dtype=int)

    max_total = min(8, max(3, (H * W) // 26))
    total_k = int(rng.integers(3, max_total + 1))
    k_yellow = int(rng.integers(1, total_k))
    k_green = total_k - k_yellow
    if k_green == 0:
        return None

    motif_colors = [YELLOW] * k_yellow + [GREEN] * k_green
    rng.shuffle(motif_colors)

    dir_cycle = list(rng.permutation(len(ORTHO)))
    missing_dirs = [ORTHO[dir_cycle[i % len(ORTHO)]] for i in range(total_k)]

    centers_by_color: dict[int, set[tuple[int, int]]] = {
        YELLOW: set(),
        GREEN: set(),
    }
    placed_centers: list[tuple[int, int]] = []

    for fg, missing in zip(motif_colors, missing_dirs):
        placed = False
        for _ in range(180):
            cr = int(rng.integers(2, H - 2))
            cc = int(rng.integers(2, W - 2))
            if any(abs(cr - er) + abs(cc - ec) < 4 for er, ec in placed_centers):
                continue

            present_arms = [d for d in ORTHO if d != missing]
            cells = {(cr, cc)}
            for dr, dc in present_arms:
                cells.add((cr + dr, cc + dc))

            n_ext = int(rng.integers(1, 3))
            chosen_ids = rng.choice(len(present_arms), size=n_ext, replace=False)
            for idx in chosen_ids:
                dr, dc = present_arms[int(idx)]
                ext_len = int(rng.integers(1, 4))
                for step in range(2, ext_len + 2):
                    cells.add((cr + dr * step, cc + dc * step))

            trial_centers = {
                YELLOW: set(centers_by_color[YELLOW]),
                GREEN: set(centers_by_color[GREEN]),
            }
            trial_centers[fg].add((cr, cc))
            if _try_add_cells(grid, fg, trial_centers, cells):
                centers_by_color = trial_centers
                placed_centers.append((cr, cc))
                placed = True
                break
        if not placed:
            return None

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != BG))

    bg_positions = list(zip(*np.where(grid == BG)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        for fg in rng.permutation(ACTIVE_COLORS):
            if _try_add_cells(grid, int(fg), centers_by_color, {(int(r), int(c))}):
                current_fg += 1
                break

    if int(np.sum(grid != BG)) >= total // 2:
        return None

    yellow_t = _find_t_centers(grid, YELLOW)
    green_t = _find_t_centers(grid, GREEN)
    if sorted(yellow_t) != sorted(centers_by_color[YELLOW]):
        return None
    if sorted(green_t) != sorted(centers_by_color[GREEN]):
        return None
    if not yellow_t or not green_t:
        return None
    if _find_plus_centers(grid, YELLOW) or _find_plus_centers(grid, GREEN):
        return None

    out = main(grid)
    if not np.any(out == TEAL):
        return None
    if not np.any((grid == GREEN) & (out == BG)):
        return None
    if np.array_equal(out, _all_t_to_teal_output(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")


# Assumption review for the fused main():
# - Only yellow and green are action-carrying foreground colors. The generator always places
#   both, and distractors use only those same two colors, so `main()` does not need a generic
#   "every non-bg color" branch.
# - T detection is purely same-color and local. The generator guarantees that only designated
#   centers reach same-color degree 3 and that no same-color '+' exists, so every detection
#   corresponds to an intended motif rather than an accidental dense blob.
# - Output writes are safe on one shared `out` grid because detection reads from the untouched
#   input copy `g`, and motif footprints are spaced so different T cores do not overlap.
