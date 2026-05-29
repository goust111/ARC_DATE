# Variant of 1818057f
# Dimension: D3 — 识别目标：'+'字图案的定义 (src)
# Change: 黄色'+'字精灵：任何前景像素(r,c)若其4个正交邻居均为前景色，则(r,c)及其4邻居构成一个'+'... → T型图案识别

from __future__ import annotations

import numpy as np

from arc_utils import get_bg

# Design analysis
# ---
# concretization:
#   - The old target was a 5-cell '+' whose center has orthogonal degree 4.
#   - The new target is a 4-cell T motif: a foreground center with exactly 3
#     orthogonal foreground neighbors and exactly 1 missing orthogonal side.
#   - All 4 rotations are valid. Example:
#       2 4 2        old solver: no recolor (center degree 3)
#       4 4 4   ->   new solver: recolor the 4 yellow cells to teal
#       2 2 2
#     while
#       2 4 2        old solver: recolor 5 cells
#       4 4 4   ->   new solver: no recolor (center degree 4, not a T)
#       2 4 2
# complexity_calibration:
#   - "At least 3 neighbors" would collapse T and '+' together, so the variant
#     must use "exactly 3 neighbors". This makes the missing side visually
#     meaningful and prevents the task from degenerating back to the source.
#   - The generator mixes the 4 missing-side orientations instead of fixing one
#     direction, so rotation invariance is observable from a few examples.
# implicit_assumptions_review:
#   - `get_bg()` as most-common color still holds; generator keeps red (2) the
#     majority color.
#   - `main()` still assumes a single non-background foreground color and reads
#     only from the original input grid, so recoloring order cannot affect
#     later detections.
#   - The old `all(... four neighbors ...)` predicate is no longer safe because
#     it accepts '+' only. It is replaced by an exact-neighbor-count test.
# placement_legality:
#   - T centers are placed at least 2 cells from the border so every arm exists
#     and extensions can attach without going out of bounds.
#   - The generator maintains a strong invariant on the orthogonal adjacency
#     graph: designated T centers have degree exactly 3; every other foreground
#     pixel has degree at most 2. This forbids accidental extra T motifs and
#     forbids '+' motifs entirely.
#   - Candidate motif cells must not overlap existing foreground cells, and each
#     placement is validated against the invariant before being committed.
# visual_observability:
#   - Only one foreground color is used in the input and teal marks the output,
#     so the changed rule is purely shape-based and visible.
#   - The missing arm is the only difference between target and non-target
#     branch points; because '+' motifs are excluded, the human can infer that
#     "three arms counts, four arms does not".
# cleanliness_and_color_change:
#   - Colors stay close to the source task: bg=2, fg=4, output color=8.
#   - Distractors are added only when they preserve the invariant, so the grid
#     remains readable rather than turning into dense overlapping blobs.

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _t_arm_offsets(grid: np.ndarray, r: int, c: int, fg: int) -> list[tuple[int, int]] | None:
    """Return the 3 occupied arm offsets for an exact T center, else None."""
    if int(grid[r, c]) != fg:
        return None
    arms = [(dr, dc) for dr, dc in ORTHO_DIRS if int(grid[r + dr, c + dc]) == fg]
    return arms if len(arms) == 3 else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect degree-3 T junctions in a single foreground color and recolor them to teal.

    Input structure:
    - bg = most common color.
    - One foreground color (e.g. yellow) forms mixed paths and branched shapes on the bg.
    - Some local 4-cell motifs are T shapes: a center pixel with exactly 3 orthogonal fg neighbors.

    Rule:
    - A T center is any foreground pixel whose orthogonal neighborhood contains exactly 3 fg neighbors.
    - Recolor that center and those 3 neighboring fg arm cells to teal (8).
    - A full '+' does not qualify because it has 4 arms, not 3. All other pixels stay unchanged.

    Key insight:
    - The task is about local orthogonal degree, not component size: any fg pixel with degree 3 is highlighted,
      regardless of whether the T is isolated or embedded in a larger path-like component.
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
            out[r, c] = 8
            for dr, dc in arms:
                out[r + dr, c + dc] = 8
    return out


def _fg_degree(grid: np.ndarray, r: int, c: int, fg: int) -> int:
    deg = 0
    H, W = grid.shape
    for dr, dc in ORTHO_DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == fg:
            deg += 1
    return deg


def _respects_planned_t_centers(grid: np.ndarray, fg: int, centers: set[tuple[int, int]]) -> bool:
    """Only the planned centers may have degree 3, and none may become degree 4."""
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != fg:
                continue
            deg = _fg_degree(grid, r, c, fg)
            if (r, c) in centers:
                if deg != 3:
                    return False
            elif deg >= 3:
                return False
    return True


def _find_t_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _t_arm_offsets(grid, r, c, fg) is not None:
                out.append((r, c))
    return out


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DIRS):
                out.append((r, c))
    return out


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
        if int(grid[r, c]) == fg:
            return False
    trial = grid.copy()
    for r, c in cells:
        trial[r, c] = fg
    if not _respects_planned_t_centers(trial, fg, centers):
        return False
    grid[:, :] = trial
    return True


def _gen_attempt():
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    # Keep the original 1..9 flavor, but cap by area so the denser T-with-extension
    # motifs can still be placed legally on the smallest boards.
    max_k = min(9, max(1, (H * W) // 28))
    K = int(rng.integers(1, max_k + 1))

    centers: list[tuple[int, int]] = []
    center_set: set[tuple[int, int]] = set()
    dir_order = list(map(int, rng.permutation(4)))
    desired_missing = [ORTHO_DIRS[dir_order[i % 4]] for i in range(K)]

    for i in range(K):
        missing = desired_missing[i]
        placed = False
        for _ in range(120):
            cr = int(rng.integers(2, H - 2))
            cc = int(rng.integers(2, W - 2))
            if (cr, cc) in center_set:
                continue

            present_arms = [d for d in ORTHO_DIRS if d != missing]
            core = {(cr, cc)}
            for dr, dc in present_arms:
                core.add((cr + dr, cc + dc))

            n_ext = int(rng.integers(1, 3))
            chosen_ids = rng.choice(len(present_arms), size=n_ext, replace=False)
            cells = set(core)
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

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    inp = grid.copy()
    t_centers = _find_t_centers(inp, fg)
    plus_centers = _find_plus_centers(inp, fg)
    if sorted(t_centers) != sorted(centers):
        return None
    if plus_centers:
        return None

    out = main(inp)
    if not np.any(out == 8):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
