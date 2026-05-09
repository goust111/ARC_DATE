# Combo variant of 1818057f (index 17)
# combo_spec: D1/1+D10/1
# Name: X形检测后旋转为十字

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Original task: detect a foreground '+' using orthogonal neighbors and recolor those
#      same 5 cells teal.
#    - Fused task: detect a foreground 'X' using diagonal neighbors, then rotate that local
#      shape by 45 degrees into a '+'. The center stays foreground, the 4 diagonal arm cells
#      are cleared back to bg, and the 4 orthogonal arm cells become teal.
#      Input local pattern:
#          4 . 4
#          . 4 .
#          4 . 4
#      Output local pattern:
#          . 8 .
#          8 4 8
#          . 8 .
#      Here '.' denotes the background. Relative to the original rule, both the trigger
#      offsets and the write footprint change.
# b) Complexity calibration
#    - If orthogonal neighbors around an intended center were already foreground, a human
#      could confuse the trigger with the old '+' rule. The generator therefore reserves
#      every target center's full 3x3 box except for the 5 intended X cells.
#    - Each intended X is extended along one or more diagonal rays so it is embedded in a
#      larger foreground component instead of appearing as an isolated token.
# c) Hidden assumptions audit
#    - The original solver could safely recolor in place because detection and output used
#      the same footprint. That is false here: the output writes to cells that were bg and
#      clears cells that were fg. The solver must first collect all centers from the input.
#    - `range(1, H-1)` still suffices because both diagonal detection and orthogonal output
#      only need a 1-cell margin, but the generator keeps centers >= 2 from the border so
#      extensions have room and the local 3x3 windows remain clean.
#    - The input still has one bg color and one fg color, so `get_bg` plus "first non-bg"
#      foreground inference remains valid.
# d) Placement legality
#    - Centers are spaced with Chebyshev distance >= 4, which keeps their 3x3 windows and
#      output '+' footprints disjoint.
#    - Any candidate distractor cell is rejected if it would create an unintended X center
#      or any orthogonal '+' center. This prevents overlap, border-induced partial motifs,
#      and accidental reintroduction of the old rule.
# e) Visual observability
#    - Across examples, the input visibly contains diagonal 5-cell X motifs with empty
#      orthogonal neighbors, while the output replaces them with orthogonal teal crosses.
#      Because the center remains yellow, the 45-degree rotation is directly readable.
# f) Visual cleanliness
#    - Protected 3x3 neighborhoods stop clutter from sitting on top of the transformed
#      region. Diagonal extensions start outside that box, so the before/after relation is
#      legible even inside larger components.
# g) Change inferability
#    - Only one new color is introduced: teal marks the rotated arm endpoints. The solver
#      never branches on numeric color identity beyond bg-vs-fg, so the structural rotation
#      is the only new concept the examples need to teach.
# h) Quantitative anti-degeneration target
#    - For each intended center, the input/output difference must contain exactly 8 changed
#      cells: 4 diagonal fg->bg removals and 4 orthogonal bg->teal additions. The generator
#      validates this, and later testing reports the distribution of this metric.
# i) New solver assumptions honored by generation
#    - Every detected center is an X-only center: all 4 diagonals are fg, all 4 orthogonals
#      are bg, and there are no orthogonal '+' centers anywhere else in the grid.
#    - Centers are found from the untouched input, so generation guarantees the intended set
#      of X centers exactly matches the solver's detected set.


BG = 2
FG = 4
ROT = 8
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
X_FOOTPRINT = [(0, 0), *DIAG]
PLUS_FOOTPRINT = [(0, 0), *ORTHO]


def _is_x_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != int(fg):
        return False
    return all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in DIAG)


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != int(fg):
        return False
    return all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTHO)


def _find_x_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_x_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


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
    Detect diagonal 'X' motifs in one foreground color and rotate each one into a '+'.

    Input structure:
    - The background is the most common color in the grid.
    - One foreground color forms diagonal strands, scattered distractors, and embedded 5-cell 'X' motifs.

    Rule:
    - An 'X' center is any foreground pixel whose 4 diagonal neighbors are also foreground.
    - Keep that center foreground, clear the 4 diagonal arm cells back to the background, and paint the
      4 orthogonal neighbors teal (8) so the local shape becomes a '+'.
    - Leave every other pixel unchanged.

    Key insight: The input evidence is diagonal but the output evidence is orthogonal, so the solver must
    read all X centers from the original grid first and only then rewrite each local 3x3 neighborhood.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    centers = _find_x_centers(g, fg)
    for r, c in centers:
        out[r, c] = fg
        for dr, dc in DIAG:
            out[r + dr, c + dc] = bg
        for dr, dc in ORTHO:
            out[r + dr, c + dc] = ROT
    return out


def _would_create_unintended_x(
    grid: np.ndarray,
    r: int,
    c: int,
    bg: int,
    allowed_centers: set[tuple[int, int]],
) -> bool:
    """Check whether occupying (r,c) would create an X center outside `allowed_centers`."""
    H, W = grid.shape
    for cr, cc in [(r, c), (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        full = True
        for dr, dc in X_FOOTPRINT:
            nr, nc = cr + dr, cc + dc
            occupied = (nr, nc) == (r, c) or int(grid[nr, nc]) != int(bg)
            if not occupied:
                full = False
                break
        if full and (cr, cc) not in allowed_centers:
            return True
    return False


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether occupying (r,c) with fg would create any orthogonal '+' center nearby."""
    H, W = grid.shape
    for cr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        full = True
        for dr, dc in PLUS_FOOTPRINT:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != int(fg):
                full = False
                break
        if full:
            return True
    return False


def _reserve_boxes(centers: list[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    reserved.add((nr, nc))
    return reserved


def _try_place_shape(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
    fg: int,
    reserved: set[tuple[int, int]],
    allowed_centers: set[tuple[int, int]],
) -> int:
    H, W = grid.shape
    temp = grid.copy()
    for r, c in cells:
        if r < 0 or r >= H or c < 0 or c >= W:
            return 0
        if (r, c) in reserved or int(temp[r, c]) != int(bg):
            return 0
        if _would_create_unintended_x(temp, r, c, bg, allowed_centers):
            return 0
        if _would_create_plus(temp, r, c, fg):
            return 0
        temp[r, c] = fg
    for r, c in cells:
        grid[r, c] = fg
    return len(cells)


def _changed_cell_count(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.sum(inp != out))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(12, 23))
    grid = np.full((H, W), BG, dtype=int)

    max_k = min(9, max(1, (H * W) // 36))
    target_k = int(rng.integers(1, max_k + 1))
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 4 for rr, cc in centers):
            centers.append((r, c))
            if len(centers) >= target_k:
                break

    if not centers:
        return None

    allowed_centers = set(centers)
    reserved = _reserve_boxes(centers, H, W)

    for r, c in centers:
        grid[r, c] = FG
        for dr, dc in DIAG:
            grid[r + dr, c + dc] = FG

    order = list(range(len(centers)))
    rng.shuffle(order)
    for idx in order:
        r, c = centers[idx]
        feasible_dirs: list[tuple[int, int]] = []
        for dr, dc in DIAG:
            nr, nc = r + 2 * dr, c + 2 * dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in reserved and int(grid[nr, nc]) == BG:
                feasible_dirs.append((dr, dc))
        if not feasible_dirs:
            return None

        rng.shuffle(feasible_dirs)
        n_dirs = min(len(feasible_dirs), int(rng.integers(1, 4)))
        placed_any = False
        for dr, dc in feasible_dirs[:n_dirs]:
            trail: list[tuple[int, int]] = []
            for dist in range(2, 5):
                nr, nc = r + dr * dist, c + dc * dist
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if (nr, nc) in reserved or int(grid[nr, nc]) != BG:
                    break
                trial = grid.copy()
                ok = True
                for tr, tc in trail + [(nr, nc)]:
                    if _would_create_unintended_x(trial, tr, tc, BG, allowed_centers):
                        ok = False
                        break
                    if _would_create_plus(trial, tr, tc, FG):
                        ok = False
                        break
                    trial[tr, tc] = FG
                if not ok:
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
    min_fg = int(np.ceil(total * 0.26))
    max_fg = int(np.floor(total * 0.41))
    target_fg = int(rng.integers(min_fg, max_fg + 1))
    current_fg = int(np.sum(grid != BG))

    attempts = 0
    while current_fg < target_fg and attempts < total * 30:
        attempts += 1
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W))
        mode = int(rng.integers(0, 4))
        if mode == 0:
            shape = [(r, c)]
        else:
            dr, dc = DIAG[int(rng.integers(0, len(DIAG)))]
            length = int(rng.integers(2, 4))
            shape = [(r + dr * step, c + dc * step) for step in range(length)]
        current_fg += _try_place_shape(grid, shape, BG, FG, reserved, allowed_centers)

    if current_fg < min_fg:
        bg_positions = list(zip(*np.where(grid == BG)))
        rng.shuffle(bg_positions)
        for r, c in bg_positions:
            if current_fg >= min_fg:
                break
            current_fg += _try_place_shape(
                grid,
                [(int(r), int(c))],
                BG,
                FG,
                reserved,
                allowed_centers,
            )

    fg_count = int(np.sum(grid != BG))
    if fg_count < min_fg or fg_count > max_fg:
        return None
    if fg_count >= total // 2:
        return None

    x_centers = _find_x_centers(grid, FG)
    if sorted(x_centers) != sorted(centers):
        return None
    if _find_plus_centers(grid, FG):
        return None

    for r, c in centers:
        if any(int(grid[r + dr, c + dc]) != BG for dr, dc in ORTHO):
            return None

    out = main(grid)
    if not np.any(out == ROT):
        return None
    if _changed_cell_count(grid, out) != 8 * len(centers):
        return None

    for r, c in centers:
        if int(out[r, c]) != FG:
            return None
        if any(int(out[r + dr, c + dc]) != BG for dr, dc in DIAG):
            return None
        if any(int(out[r + dr, c + dc]) != ROT for dr, dc in ORTHO):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
