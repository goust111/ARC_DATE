# Variant of 1818057f
# Dimension: D1 — 核心变换操作：重着色规则 (transform)
# Change: 识别'+'字图案后，将构成'+'的5个像素从前景色(黄色4)统一重着色为青色(teal 8)，其余像素不变... → 旋转'+'字图案为'X'形

from __future__ import annotations

from collections import Counter
import numpy as np


# Design analysis for the D1 rotation variant
# a) Concrete behavior and trace
#    - Original behavior: detect a foreground '+' and recolor its existing 5 cells.
#    - Variant behavior: detect the same foreground '+' in the input, but rewrite its
#      footprint geometrically in the output.
#      Input local pattern:
#          . 4 .
#          4 4 4
#          . 4 .
#      Output local pattern:
#          8 . 8
#          . 4 .
#          8 . 8
#      Here '.' is the background color. The center stays foreground, the 4 orthogonal
#      arm cells are erased back to background, and 4 diagonal cells are painted teal.
#    - Small trace at center (r,c):
#      Original solver writes teal to {(r,c), (r±1,c), (r,c±1)}.
#      Variant solver instead keeps (r,c)=fg, writes bg to {(r±1,c), (r,c±1)},
#      and writes teal to {(r±1,c±1)}.
#      The detection step is unchanged; only the write step changes.
#
# b) Complexity calibration
#    - If the diagonals around a '+' were already occupied, the 45-degree rotation
#      would be visually ambiguous. The generator therefore keeps each motif's 3x3
#      neighborhood controlled: only the 5 '+' cells may occupy that box in the input.
#    - The task still has distractor foreground noise and extended arms, so the solver
#      must truly locate plus centers inside mixed components rather than operate on
#      isolated 5-cell tokens.
#
# c) Old assumptions that no longer hold
#    - `out[r+dr, c+dc] = 8` on the 5 cross cells was harmless before because the output
#      footprint equaled the input footprint. That is false now: orthogonal cells must
#      become background, and diagonals must be written instead.
#    - In-place mutation during detection would now be risky because a newly written teal
#      diagonal or cleared arm could affect subsequent neighbor tests. The variant first
#      collects centers from the original grid, then applies all rewrites to `out`.
#
# d) Placement legality
#    - Plus centers are kept at least two cells from the border, so both the input '+'
#      and the output 'X' fit entirely inside the grid.
#    - Centers are spaced so their 3x3 neighborhoods do not overlap. This prevents
#      diagonal landing collisions and keeps each rotated motif visually distinct.
#    - Extra arm extensions and distractors are forbidden from entering any center's 3x3
#      neighborhood, so the diagonal landing sites stay background in the input.
#
# e) Visual observability
#    - Across 2-4 examples, a human can see that the same local '+' trigger is present
#      in the input, but the output does not merely recolor it: the arms move diagonally.
#    - The center staying yellow while the new diagonal pixels become teal makes the
#      rotation visible even when only one motif is present.
#
# f) Visual cleanliness
#    - The generator avoids clutter around each motif, so the local before/after relation
#      is easy to read.
#    - Distractors are pushed outside protected 3x3 boxes and are also prevented from
#      forming accidental new pluses.
#
# g) Color-change inference
#    - Only one new color is introduced: teal marks the rotated arm endpoints.
#      The unchanged yellow center acts as the anchor that reveals the 45-degree turn.
#
# h) Quantitative anti-degeneration target
#    - For every intended plus center, the variant should remove exactly 4 foreground arm
#      cells and add exactly 4 teal diagonal cells. This is later checked in testing.
#
# i) New solver assumptions to honor in generation
#    - The solver assumes each intended center has 4 foreground orthogonal neighbors in
#      the input.
#    - The generator guarantees that any other added foreground cell never completes a new
#      '+' motif, so the detected centers match the intended ones.


BG = 2
FG = 4
ROT = 8
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != int(fg):
        return False
    return all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTHO)


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' shapes in a single foreground color and rotate each one into an 'X'.

    Input structure:
    - bg = most common color.
    - One foreground color (e.g. yellow) forms mixed shapes on the bg, and some local
      5-cell regions are exact '+' motifs.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Keep the center unchanged, clear its 4 orthogonal arm cells back to bg, and paint the
      4 diagonal cells teal (8) to form an 'X' around the same center.
    - All other pixels remain unchanged.

    Key insight: The pattern is still recognized from orthogonal adjacency in the input, but
    the output is produced by moving the arms diagonally, so detection must read from the
    original grid before any rewrites are applied.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    for r, c in _find_plus_centers(g, fg):
        out[r, c] = fg
        for dr, dc in ORTHO:
            out[r + dr, c + dc] = bg
        for dr, dc in DIAG:
            out[r + dr, c + dc] = ROT
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether turning one bg cell into fg would complete a plus anywhere nearby."""
    H, W = grid.shape
    for cr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in [(0, 0), *ORTHO]:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _reserve_neighborhoods(centers: list[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    reserved.add((nr, nc))
    return reserved


def _place_shape(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
    fg: int,
    reserved: set[tuple[int, int]],
) -> int:
    H, W = grid.shape
    temp = grid.copy()
    for r, c in cells:
        if r < 0 or r >= H or c < 0 or c >= W:
            return 0
        if (r, c) in reserved or int(temp[r, c]) != int(bg):
            return 0
        if _would_create_plus(temp, r, c, fg):
            return 0
        temp[r, c] = fg
    for r, c in cells:
        grid[r, c] = fg
    return len(cells)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    grid = np.full((H, W), BG, dtype=int)

    max_k = min(9, max(1, (H * W) // 36))
    target_k = int(rng.integers(1, max_k + 1))
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 3 for rr, cc in centers):
            centers.append((r, c))
            if len(centers) >= target_k:
                break

    if not centers:
        return None

    reserved = _reserve_neighborhoods(centers, H, W)

    for r, c in centers:
        grid[r, c] = FG
        for dr, dc in ORTHO:
            grid[r + dr, c + dc] = FG

    order = list(range(len(centers)))
    rng.shuffle(order)
    for idx in order:
        r, c = centers[idx]
        feasible_dirs: list[tuple[int, int]] = []
        for dr, dc in ORTHO:
            nr, nc = r + 2 * dr, c + 2 * dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in reserved and int(grid[nr, nc]) == BG:
                feasible_dirs.append((dr, dc))
        if not feasible_dirs:
            return None

        rng.shuffle(feasible_dirs)
        n_dirs = min(len(feasible_dirs), int(rng.integers(1, 3)))
        placed_any = False
        for dr, dc in feasible_dirs[:n_dirs]:
            trail: list[tuple[int, int]] = []
            for dist in range(2, 5):
                nr, nc = r + dr * dist, c + dc * dist
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if (nr, nc) in reserved or int(grid[nr, nc]) != BG:
                    break
                trail.append((nr, nc))
            if not trail:
                continue
            take = int(rng.integers(1, len(trail) + 1))
            for nr, nc in trail[:take]:
                grid[nr, nc] = FG
            placed_any = True
        if not placed_any:
            return None

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != BG))

    attempts = 0
    while current_fg < target_fg and attempts < total * 25:
        attempts += 1
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W))
        mode = int(rng.integers(0, 3))
        if mode == 0:
            shape = [(r, c)]
        else:
            dr, dc = ORTHO[int(rng.integers(0, 4))]
            length = int(rng.integers(2, 4))
            shape = [(r + dr * step, c + dc * step) for step in range(length)]
        current_fg += _place_shape(grid, shape, BG, FG, reserved)

    if current_fg < int(total * 0.26):
        bg_positions = list(zip(*np.where(grid == BG)))
        rng.shuffle(bg_positions)
        for r, c in bg_positions:
            if current_fg >= int(total * 0.26):
                break
            if (int(r), int(c)) in reserved:
                continue
            if _would_create_plus(grid, int(r), int(c), FG):
                continue
            grid[int(r), int(c)] = FG
            current_fg += 1

    fg_count = int(np.sum(grid != BG))
    if fg_count < int(total * 0.26) or fg_count > int(total * 0.41):
        return None
    if fg_count >= total // 2:
        return None

    found = _find_plus_centers(grid, FG)
    if sorted(found) != sorted(centers):
        return None

    out = main(grid)
    if not np.any(out == ROT):
        return None

    for r, c in centers:
        if any(int(grid[r + dr, c + dc]) != BG for dr, dc in DIAG):
            return None
        if any(int(out[r + dr, c + dc]) != ROT for dr, dc in DIAG):
            return None
        if any(int(out[r + dr, c + dc]) != BG for dr, dc in ORTHO):
            return None
        if int(out[r, c]) != FG:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
