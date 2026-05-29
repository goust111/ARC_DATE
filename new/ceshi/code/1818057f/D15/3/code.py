# Variant of 1818057f
# Dimension: D15 — 图例/符号指令 (symbol_instructions)
# Change: 无图例或符号指令，变换规则完全隐含在输入输出的差异中
#   → 网格分区图例：上半部分给出一个示例带，下半部分是待处理区域。

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis
# a) Concrete layout and behavior trace
#    The full grid is split by a horizontal background separator row. The upper band is a legend:
#    its left panel shows an unsolved instance, and its right panel shows that same panel after
#    recoloring every thin '+' motif to teal. The lower band contains two unsolved task panels.
#    Input keeps the upper legend unchanged and leaves both lower panels unsolved. Output keeps the
#    legend unchanged and solves only the lower band. This differs from the original task at a
#    specific execution point: the original solver would recolor '+' motifs everywhere, including
#    the upper-left demo-input panel; the variant must treat that panel as an instruction artifact.
# b) Complexity calibration
#    Each panel still uses the original local visual language: one foreground color on red
#    background, embedded pluses, arm extensions, and distractor cells that do not create extra
#    pluses. The top legend includes both before and after, so the recoloring rule is repeated
#    visibly without needing extra symbolic marks or a fragile color lookup.
# c) Original-code assumptions that no longer hold
#    - "Search the whole grid" is no longer valid; only the lower band is actionable.
#    - "There is exactly one non-bg color" is no longer valid in the combined input because the
#      legend output panel introduces teal before solving.
#    - "Any plus made of the foreground color should change" is no longer valid because pluses in
#      the demo-input panel are intentionally frozen as part of the example.
# d) Placement legality
#    Panels are generated independently with a 1-cell functional margin for plus centers and a
#    stronger 2-cell sampling margin where possible, so crosses never spill across panel borders or
#    separators. The vertical separator column is background-only, preventing accidental cross-panel
#    pluses in the lower task band.
# e) Visual observability
#    The solved example is adjacent to its unsolved counterpart in the top half, so a human can
#    compare local differences directly and then transfer that rule to the bottom half. This avoids
#    a degenerate "top already solved, bottom unsolved" layout that would look different but still
#    be solvable by the original whole-grid rule without understanding the partition.
# f) Visual cleanliness
#    Only three colors appear in the combined grid: red background, yellow active foreground, teal
#    transformed plus pixels. Blank separator row/column keep the four panels visually distinct.
# g) New main() assumptions and generator guarantees
#    main() assumes odd height/width with a central separator row/column, teal reserved for the
#    legend/outputs, and at least one non-teal foreground color. generate_input() always constructs
#    exactly that layout and validates it against main().


TEAL = 8


def _solve_rows(grid: np.ndarray, row_start: int, row_end: int, fg: int) -> np.ndarray:
    """Apply the original plus recoloring rule, but only inside rows [row_start, row_end)."""
    out = grid.copy()
    h, w = grid.shape
    for r in range(max(1, row_start + 1), min(h - 1, row_end - 1)):
        for c in range(1, w - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    out[r + dr, c + dc] = TEAL
    return out


def _solve_panel(panel: np.ndarray, fg: int) -> np.ndarray:
    """Solve a standalone panel using the original local rule."""
    return _solve_rows(panel, 0, panel.shape[0], fg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper legend band to identify the local '+' recoloring rule, then apply it only below.

    Input structure:
    - The full grid is split into an upper example band and a lower task band by the middle row.
    - The upper band contains an unsolved panel beside its solved counterpart, so teal may already
      appear there before the task region is processed.
    - The lower band contains unsolved panels built from one active foreground color on the bg.

    Rule:
    - Keep the upper band unchanged; it is an instruction panel, not a target for editing.
    - In the lower band only, any active-foreground pixel whose 4 orthogonal neighbors are also
      active foreground is the center of a thin '+' motif.
    - Recolor that center and its 4 orthogonal neighbors to teal (8). Leave all other cells as-is.

    Key insight:
    - The example band's left panel can still contain untransformed '+' motifs, so scanning the
      entire grid would be wrong. The horizontal partition determines where the demonstrated rule
      should be executed, while teal itself marks the "after" state to ignore during fg detection.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg, TEAL})
    if not colors:
        return out
    fg = int(colors[0])

    mid_r = g.shape[0] // 2
    return _solve_rows(out, mid_r + 1, g.shape[0], fg)


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether painting (r,c) as fg would create a new plus at or near that cell."""
    h, w = grid.shape
    for cr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if cr < 1 or cr >= h - 1 or cc < 1 or cc >= w - 1:
            continue
        all_fg = True
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _count_plus_centers(grid: np.ndarray, fg: int) -> int:
    """Count plus centers made of fg in a standalone panel."""
    h, w = grid.shape
    total = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                total += 1
    return total


def _gen_panel_attempt(rng: np.random.Generator, h: int, w: int, bg: int, fg: int) -> np.ndarray | None:
    panel = np.full((h, w), bg, dtype=int)

    area = h * w
    max_centers = max(1, min(4, area // 18))
    want_centers = int(rng.integers(1, max_centers + 1))
    centers: list[tuple[int, int]] = []

    row_choices = list(range(2, h - 2))
    col_choices = list(range(2, w - 2))
    if not row_choices or not col_choices:
        return None

    for _ in range(want_centers * 50):
        if len(centers) >= want_centers:
            break
        cr = int(rng.choice(row_choices))
        cc = int(rng.choice(col_choices))
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))

    if len(centers) != want_centers:
        return None

    for cr, cc in centers:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            panel[cr + dr, cc + dc] = fg

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for cr, cc in centers:
        n_ext = int(rng.integers(1, 3))
        chosen = rng.choice(len(directions), size=n_ext, replace=False)
        for idx in chosen:
            dr, dc = directions[int(idx)]
            ext_len = int(rng.integers(1, 4))
            tip_r = cr + dr
            tip_c = cc + dc
            for step in range(1, ext_len + 1):
                nr = tip_r + dr * step
                nc = tip_c + dc * step
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    break
                panel[nr, nc] = fg

    target_density = float(rng.uniform(0.24, 0.37))
    target_fg = int(area * target_density)
    current_fg = int(np.sum(panel != bg))

    bg_positions = list(zip(*np.where(panel == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(panel, int(r), int(c), fg):
            panel[int(r), int(c)] = fg
            current_fg += 1

    if int(np.sum(panel != bg)) >= area // 2:
        return None
    if _count_plus_centers(panel, fg) != len(centers):
        return None

    solved = _solve_panel(panel, fg)
    if int(np.sum(solved == TEAL)) < 5:
        return None
    return panel


def _generate_panel(rng: np.random.Generator, h: int, w: int, bg: int, fg: int) -> np.ndarray:
    for _ in range(300):
        panel = _gen_panel_attempt(rng, h, w, bg, fg)
        if panel is not None:
            return panel
    raise RuntimeError("panel generation failed after 300 attempts")


def _assemble_grid(
    demo_in: np.ndarray,
    demo_out: np.ndarray,
    task_left: np.ndarray,
    task_right: np.ndarray,
    bg: int,
) -> np.ndarray:
    ph, pw = demo_in.shape
    full = np.full((2 * ph + 1, 2 * pw + 1), bg, dtype=int)
    full[:ph, :pw] = demo_in
    full[:ph, pw + 1 :] = demo_out
    full[ph + 1 :, :pw] = task_left
    full[ph + 1 :, pw + 1 :] = task_right
    return full


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg, fg = 2, 4

    for _ in range(300):
        panel_h = int(rng.integers(6, 11))   # 6..10
        panel_w = int(rng.integers(8, 11))   # 8..10

        demo_in = _generate_panel(rng, panel_h, panel_w, bg, fg)
        demo_out = _solve_panel(demo_in, fg)
        task_left = _generate_panel(rng, panel_h, panel_w, bg, fg)
        task_right = _generate_panel(rng, panel_h, panel_w, bg, fg)

        inp = _assemble_grid(demo_in, demo_out, task_left, task_right, bg)
        expected = _assemble_grid(demo_in, demo_out, _solve_panel(task_left, fg), _solve_panel(task_right, fg), bg)

        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(actual[:panel_h, :panel_w], _solve_panel(demo_in, fg)):
            continue
        if not np.any(inp[:panel_h, panel_w + 1 :] == TEAL):
            continue
        if np.any(inp[panel_h + 1 :, :] == TEAL):
            continue
        if not np.any(expected[panel_h + 1 :, :] == TEAL):
            continue
        return inp

    raise RuntimeError("generate_input failed after 300 attempts")
