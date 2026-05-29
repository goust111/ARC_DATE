# Variant of 1818057f
# Dimension: D1 — 核心变换操作：重着色规则 (transform)
# Change: 识别'+'字图案后，将构成'+'的5个像素从前景色(黄色4)统一重着色为青色(teal 8)，其余像素不变... → 分解'+'：中心与臂分别处理

from __future__ import annotations

import numpy as np

from arc_utils import get_bg

# Design analysis for the D1 variant
# a) Concrete behavior and pipeline impact:
#    The input stays visually the same kind of scene as the original task:
#    red background, one yellow foreground color, and some local '+' motifs embedded
#    inside larger yellow components. The changed behavior happens only in the output.
#    For each detected '+', the center cell becomes teal (8) while the four arm cells
#    become magenta (6). Example on one motif:
#      input local patch            original output            variant output
#          . 4 .                        . 8 .                    . 6 .
#          4 4 4          ->            8 8 8        vs          6 8 6
#          . 4 .                        . 8 .                    . 6 .
#    So the detection stage is unchanged, but the recoloring stage now depends on the
#    role of each cell inside the same '+'.
# b) Complexity calibration:
#    The variant adds exactly one new distinction: center versus arm. It does not add
#    rotation cases, ordering by hidden ids, or per-object color choices. This keeps
#    the inference load moderate while making the output visibly different from the
#    original all-teal result.
# c) Implicit-assumption review for main():
#    - get_bg(...) still works because the input still has one dominant background.
#    - selecting the single non-bg color as fg still works because the input remains
#      two-colored; the extra output colors do not appear in the input.
#    - scanning only interior cells still works because plus centers still need a
#      1-cell margin.
#    - the old unconditional "paint all five cells teal" is no longer valid; if kept,
#      it would erase the center/arm distinction required by this variant.
# d) Placement legality:
#    Centers are placed at least 2 cells from the border and at L1 distance >= 4 from
#    each other, so the 5-cell plus footprints do not overlap. Extensions stop at the
#    grid edge or existing non-bg cells. Distractors are only added when they do not
#    create any new plus center.
# e) Visual observability:
#    In 2-4 examples, the human sees repeated cases where every qualifying '+' changes
#    into the same two-color pattern: one teal center and four magenta arms. Because
#    every transformed object repeats the same positional color split, the rule is
#    inferable from few samples.
# f) Visual cleanliness:
#    Only two output colors are introduced, both confined to detected '+' cells. The
#    surrounding distractor geometry stays yellow, so transformed motifs remain easy
#    to spot against the unchanged context.
# g) Color-change inferability:
#    The mapping is structural, not numeric: center -> 8 and arm -> 6. No hidden sort
#    key or color-id arithmetic is used.
# h) Variant-specific generator guarantee:
#    The generator preserves the original input regime, but it now constructs the
#    expected output with split colors and validates main(input) == output so the new
#    behavior is explicitly exercised rather than assumed.
# i) New solver assumptions and generator support:
#    main() assumes every input non-bg pixel belongs to a single foreground color and
#    that pluses are detected from the original input, not from partially recolored
#    output. The generator guarantees the single-fg input and uses a separate output
#    grid, so recoloring order cannot corrupt detection.

CENTER_COLOR = 8
ARM_COLOR = 6
PLUS_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ARM_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    return int(grid[r, c]) == fg and all(int(grid[r + dr, c + dc]) == fg for dr, dc in ARM_OFFSETS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' shapes in a single foreground color and split their recoloring by part.

    Input structure:
    - bg = most common color.
    - One foreground color (e.g. yellow) forms mixed shapes on the bg, and some local motifs are compact '+' crosses.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Recolor the center cell of every such '+' to teal (8), and recolor its 4 orthogonal arm cells to magenta (6).
    - All other pixels remain unchanged.

    Key insight:
    - Detection is still purely local and uses the same 5-cell '+' footprint as the original task; the only new reasoning step is to distinguish the center from the arms when assigning output colors.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not _is_plus_center(g, r, c, fg):
                continue
            out[r, c] = CENTER_COLOR
            for dr, dc in ARM_OFFSETS:
                out[r + dr, c + dc] = ARM_COLOR
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether setting (r, c) to fg would create any new plus center nearby."""
    h, w = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= h - 1 or cc < 1 or cc >= w - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_OFFSETS:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                all_fg = False
                break
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(10, 23))
    w = int(rng.integers(10, 23))
    bg, fg = 2, 4

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    k = int(rng.integers(1, 10))
    centers: list[tuple[int, int]] = []
    for _ in range(k * 30):
        if len(centers) >= k:
            break
        cr = int(rng.integers(2, h - 2))
        cc = int(rng.integers(2, w - 2))
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))

    if not centers:
        return None

    for cr, cc in centers:
        inp[cr, cc] = fg
        out[cr, cc] = CENTER_COLOR
        for dr, dc in ARM_OFFSETS:
            inp[cr + dr, cc + dc] = fg
            out[cr + dr, cc + dc] = ARM_COLOR

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for cr, cc in centers:
        n_ext = int(rng.integers(1, 3))
        chosen = rng.choice(4, size=n_ext, replace=False)
        placed_any = False
        for di in chosen:
            dr, dc = dirs[int(di)]
            ext_len = int(rng.integers(1, 4))
            for step in range(1, ext_len + 1):
                nr = cr + dr * (step + 1)
                nc = cc + dc * (step + 1)
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    break
                if int(inp[nr, nc]) != bg:
                    break
                inp[nr, nc] = fg
                out[nr, nc] = fg
                placed_any = True
        if not placed_any:
            return None

    total = h * w
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(inp != bg))

    bg_positions = list(zip(*np.where(inp == bg)))
    rng.shuffle(bg_positions)

    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(inp, int(r), int(c), fg):
            inp[r, c] = fg
            out[r, c] = fg
            current_fg += 1

    if int(np.sum(inp != bg)) >= total // 2:
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if not (np.any(solved == CENTER_COLOR) and np.any(solved == ARM_COLOR)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
