# Variant of 1818057f
# Dimension: D3 — 识别目标：'+'字图案的定义 (src)
# Change: 黄色'+'字精灵：任何前景像素(r,c)若其4个正交邻居均为前景色，则(r,c)及其4邻居构成一个'+'... → 3x3实心方块识别

# Design analysis for the D3 variant
# - source_task: None, so the original 1818057f solver/spec is the only baseline.
# - a) Concretization and behavior trace:
#   The visible change is not "recolor cross-like centers" anymore, but "recolor any fully
#   filled 3x3 yellow window". In the original task, a pattern such as
#       .#.
#       ###
#       .#.
#   is enough to trigger recoloring of 5 cells. In this variant, the diagonals must also be
#   filled, so only
#       ###
#       ###
#       ###
#   qualifies, and all 9 cells become teal. Therefore the changed step is the motif test itself:
#   old solver checked 4-neighbors; new solver checks the entire 3x3 neighborhood.
# - b) Complexity calibration:
#   If the generator produced only obvious 3x3 squares, the task would be too easy and would not
#   visibly differ from the old one. To keep the D3 change observable, each sample includes both:
#   (1) true solid 3x3 blocks, and (2) plus-shaped distractors that satisfy the old rule but fail
#   the new one because the four diagonals are absent.
# - c) Old implicit assumptions audited:
#   `bg = get_bg(g)` still works because bg remains the most common color.
#   `colors[0]` as the fg color still works because the generator keeps exactly one non-bg color.
#   Iterating only interior cells remains valid because any 3x3 center needs a 1-cell margin.
#   Writing to `out` while reading from `g` is still required: overlapping 3x3 windows must be
#   detected from the original input, not from already recolored pixels.
#   The old unconditional recolor of 5 cells would be wrong here; it must recolor the whole 3x3.
# - d) Placement legality:
#   Core motifs are placed with a reserved 1-cell halo so separate designed motifs do not overlap,
#   merge unintentionally, or create ambiguous composite targets. Later random sprinkling may touch
#   existing shapes, but only when doing so cannot create any new solid 3x3 window.
# - e) Visual observability:
#   Human solvers can compare examples and notice that thin '+' shapes remain yellow while compact
#   filled 3x3 patches turn teal. This is the key contrast that exposes the new rule in 2-4 pairs.
# - f) Visual cleanliness:
#   Larger target blocks are kept separated during deliberate motif placement, and extensions are
#   thin 1-cell arms. This keeps the grid from collapsing into large undifferentiated rectangles.
# - g) Color inference:
#   Colors stay identical to the original task: red background, yellow foreground, teal output mark.
#   The only new inference burden is structural, not color-based.
# - h) Generator strategy adjustment:
#   The original generator was output-first around 5-cell crosses. Here an input-first generator is
#   simpler and safer because solid 3x3 windows can overlap inside larger filled regions. Correctness
#   is enforced by validating every generated input with the new `main()`.
# - i) New `main()` assumptions and how the generator satisfies them:
#   `main()` assumes the target motif is defined purely by a local 3x3 all-fg test, independent of
#   scan order. The generator guarantees at least one such window, at least one plus-only distractor,
#   one foreground color, and no accidental extra 3x3 windows from the random fill step.

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


def _is_solid_3x3(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    return bool(np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == fg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect fully filled 3x3 foreground blocks in a single foreground color and recolor them to teal.

    Input structure:
    - bg = most common color.
    - One foreground color (e.g. yellow) forms mixed blobs, lines, and cross-like shapes on the bg.
    - Some local regions are completely filled 3x3 foreground squares.

    Rule:
    - A target center is any foreground pixel whose entire 3x3 neighborhood is also foreground.
    - Recolor all 9 cells of each such 3x3 neighborhood to teal (8).
    - All other pixels remain unchanged.

    Key insight:
    - This is stricter than detecting a thin '+': the four diagonal cells must also be foreground.
      Thin crosses and partially filled squares stay yellow, while every solid 3x3 window inside a
      larger filled patch is highlighted.
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
            if _is_solid_3x3(g, r, c, fg):
                out[r - 1 : r + 2, c - 1 : c + 2] = 8
    return out


def _count_solid_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) == fg and _is_solid_3x3(grid, r, c, fg):
                count += 1
    return count


def _count_plus_only_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]):
                if not _is_solid_3x3(grid, r, c, fg):
                    count += 1
    return count


def _would_create_solid_3x3(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    for cr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
                continue
            ok = True
            for rr in range(cr - 1, cr + 2):
                for nc in range(cc - 1, cc + 2):
                    val = fg if (rr, nc) == (r, c) else int(grid[rr, nc])
                    if val != fg:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _can_place(cells: set[tuple[int, int]], reserved: np.ndarray, pad: int = 1) -> bool:
    H, W = reserved.shape
    for r, c in cells:
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        r0 = max(0, r - pad)
        r1 = min(H, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(W, c + pad + 1)
        if np.any(reserved[r0:r1, c0:c1]):
            return False
    return True


def _mark_reserved(cells: set[tuple[int, int]], reserved: np.ndarray, pad: int = 1) -> None:
    H, W = reserved.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(H, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(W, c + pad + 1)
        reserved[r0:r1, c0:c1] = True


def _build_block_shape(
    cr: int, cc: int, dirs: list[tuple[int, int]], lengths: list[int]
) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)}
    for (dr, dc), length in zip(dirs, lengths):
        for step in range(2, length + 2):
            cells.add((cr + dr * step, cc + dc * step))
    return cells


def _build_plus_shape(
    cr: int, cc: int, dirs: list[tuple[int, int]], lengths: list[int]
) -> set[tuple[int, int]]:
    cells = {(cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)}
    for (dr, dc), length in zip(dirs, lengths):
        for step in range(2, length + 2):
            cells.add((cr + dr * step, cc + dc * step))
    return cells


def _try_place_motifs(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    count: int,
    kind: str,
    fg: int,
) -> int:
    H, W = grid.shape
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    placed = 0

    for _ in range(count * 80):
        if placed >= count:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        n_ext = int(rng.integers(1, 3))
        chosen_idx = rng.choice(len(dirs), size=n_ext, replace=False)
        chosen_dirs = [dirs[int(i)] for i in chosen_idx]
        lengths = [int(rng.integers(1, 4)) for _ in chosen_dirs]

        if kind == "block":
            cells = _build_block_shape(cr, cc, chosen_dirs, lengths)
        else:
            cells = _build_plus_shape(cr, cc, chosen_dirs, lengths)

        if not _can_place(cells, reserved, pad=1):
            continue

        for r, c in cells:
            grid[r, c] = fg
        _mark_reserved(cells, reserved, pad=1)
        placed += 1

    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    max_blocks = max(1, min(6, (H * W) // 30))
    max_pluses = max(1, min(5, (H * W) // 45))
    target_blocks = int(rng.integers(1, max_blocks + 1))
    target_pluses = int(rng.integers(1, max_pluses + 1))

    placed_blocks = _try_place_motifs(rng, grid, reserved, target_blocks, "block", fg)
    placed_pluses = _try_place_motifs(rng, grid, reserved, target_pluses, "plus", fg)
    if placed_blocks == 0 or placed_pluses == 0:
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
        if not _would_create_solid_3x3(grid, int(r), int(c), fg):
            grid[int(r), int(c)] = fg
            current_fg += 1

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    if _count_solid_centers(grid, fg) == 0:
        return None
    if _count_plus_only_centers(grid, fg) == 0:
        return None

    out = main(grid)
    if not np.any(out == 8):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
