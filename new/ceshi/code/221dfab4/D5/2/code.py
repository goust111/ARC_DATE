# Variant of 221dfab4
# Dimension: D5 — 间隙处理规则 (trajectory)
# Change: 条纹之间的1格间隙在线段跨度内强制清除为背景色... → 间隙宽度周期性变化
#
# Design analysis for the D5 variant
# ----------------------------------
# 1. Concrete behavior:
#    The original task emits 1-cell-wide stripes at equal spacing: stripe, 1-cell gap, stripe, 1-cell gap, ...
#    This variant keeps the stripe width at 1 and keeps the color cycle Y,Y,G, but the gap widths now alternate
#    1,2,1,2,... . Along a horizontal trajectory that starts on row 0, stripe rows therefore become:
#    0, 2, 5, 7, 10, 12, ... instead of 0, 2, 4, 6, 8, ... .
#    The cleared gap rows between them become:
#    [1], [3,4], [6], [8,9], ... .
#    The visual effect is that yellow/green stripes stay parallel to the border segment but are no longer evenly spaced.
#
# 2. Small execution trace:
#    Suppose the yellow border segment is on the top edge, spans columns 4..7, and bg=0.
#    Original logic:
#      step 0 -> row 0 = Y on cols 4..7, row 1 cleared on cols 4..7
#      step 1 -> row 2 = Y on cols 4..7, row 3 cleared on cols 4..7
#      step 2 -> row 4 = G on cols 4..7 and all non-bg pixels on row 4 recolored to G
#    Variant logic:
#      step 0 -> row 0 = Y, row 1 cleared
#      step 1 -> row 2 = Y, rows 3 and 4 cleared
#      step 2 -> row 5 = G, all non-bg pixels on row 5 recolored to G, row 6 cleared
#    The only changed stage is trajectory advancement / gap clearing; object recoloring semantics remain the same.
#
# 3. Complexity calibration:
#    A single widened gap would be too subtle. Alternating 1-2-1-2 produces several unequal intervals in 25-30 cell grids,
#    so the output no longer matches the original even-spacing pattern and the difference is visible from a few examples.
#
# 4. Hidden assumptions that had to change:
#    - `pos = start + dir * (2 * step)` assumed every gap was exactly 1. This is false now, so positions must be computed
#      cumulatively from the alternating gap widths.
#    - `gap = start + dir * (2 * step + 1)` assumed exactly one gap cell after every stripe. This must become a loop over
#      either 1 or 2 cells.
#    - The green-stripe recolor (`out[pos, :]` or `out[:, pos]`) still depends only on stripe positions, so it remains valid
#      once those positions are recomputed correctly.
#
# 5. Placement legality / generator guarantees:
#    - The yellow border segment remains a straight non-corner edge segment, so the solver can still infer orientation.
#    - Grid sizes stay large enough that the 1-2-1-2 trajectory yields many stripes; we also reject cases with too few
#      trajectory steps or no blob touching a green stripe line, so both the spacing rule and the original green flood are visible.
#    - Blobs are still placed with `random_free_location`, so they do not overlap or leave the grid.
#
# 6. Visual observability:
#    A human looking at a few outputs should see that the stripes are no longer evenly spaced: some adjacent stripes are
#    separated by 1 blank row/col, some by 2 blank rows/cols, and that pattern repeats. Because the border seed and the
#    Y,Y,G color cycle are unchanged, the new spacing is the salient difference.
#
# 7. New solver assumptions checked against the generator:
#    - The solver assumes exactly one straight yellow border component is the main seed; the generator creates one and keeps it
#      away from corners.
#    - The solver assumes background is still the most common color; the generator checks this before accepting a sample.
#    - The solver assumes the alternating-gap trajectory has enough room to display the pattern; the generator enforces that.

from __future__ import annotations

import random

import numpy as np

from arc2_solver_select.arc_utils import get_bg, label_mask, random_free_location, random_sprite


Y = 4
G = 3
GAP_CYCLE = (1, 2)
COLOR_CYCLE = (Y, Y, G)


def _iter_trajectory(start_pos: int, dir1: int, limit: int):
    """Yield `(step, stripe_pos, gap_cells_after)` for the 1-2-1-2 gap schedule."""
    pos = int(start_pos)
    step = 0
    while 0 <= pos < int(limit):
        gap_w = int(GAP_CYCLE[step % len(GAP_CYCLE)])
        gap_cells = []
        for off in range(1, gap_w + 1):
            gap = pos + int(dir1) * off
            if 0 <= gap < int(limit):
                gap_cells.append(int(gap))
        yield int(step), int(pos), gap_cells
        pos += int(dir1) * (gap_w + 1)
        step += 1


def _trajectory_positions(start_pos: int, dir1: int, limit: int) -> list[int]:
    return [pos for _step, pos, _gaps in _iter_trajectory(start_pos, dir1, limit)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate stripes from a yellow border segment toward the opposite edge.

    Variant D5 change:
    - Stripe colors still cycle Y, Y, G.
    - Yellow stripes still paint only inside the seed segment span.
    - Green stripes still recolor all non-bg pixels on the whole row/column.
    - Only the spacing changes: cleared gaps alternate widths 1,2,1,2,...
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(Y)) & border
    if not bool(smask.any()):
        return out

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return out

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        orient = "h"
        start_pos = int(r0)
        cols = sorted(int(c) for c in cs)
        span = list(range(int(cols[0]), int(cols[-1]) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = H
    elif c0 == c1:
        orient = "v"
        start_pos = int(c0)
        rows = sorted(int(r) for r in rs)
        span = list(range(int(rows[0]), int(rows[-1]) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = W
    else:
        return out

    for step, pos, gap_cells in _iter_trajectory(start_pos, dir1, limit):
        col = int(COLOR_CYCLE[step % len(COLOR_CYCLE)])

        if orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in span:
                out[pos, int(c)] = col
            for gap in gap_cells:
                for c in span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            for r in span:
                out[int(r), pos] = col
            for gap in gap_cells:
                for r in span:
                    out[int(r), int(gap)] = int(bg)

    return out


def _green_lines(start_pos: int, dir1: int, limit: int) -> list[int]:
    return [pos for step, pos, _gaps in _iter_trajectory(start_pos, dir1, limit) if step % 3 == 2]


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(4, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
        start_pos = int(edge_row)
        dir1 = 1 if edge == "top" else -1
        limit = H
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1
        start_pos = int(edge_col)
        dir1 = 1 if edge == "left" else -1
        limit = W

    traj_positions = _trajectory_positions(start_pos, dir1, limit)
    if len(traj_positions) < 6:
        return None

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        for c in span:
            grid[edge_row, c] = Y
    else:
        for r in span:
            grid[r, edge_col] = Y

    n_blobs = rng.randint(5, 9)
    canvas = grid.copy()

    for _ in range(n_blobs):
        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
        density = rng.uniform(0.4, 0.85)

        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[fg],
                connectivity=4,
                background=bg,
            )
        except (AssertionError, RecursionError):
            continue

        mask = sprite != bg
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0 : r1 + 1, c0 : c1 + 1]

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = (canvas != bg) & (canvas != Y)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if n_placed < 5:
        return None

    green_lines = _green_lines(start_pos, dir1, limit)
    if orient == "h":
        if not any(np.any(fg_mask[r, :]) for r in green_lines):
            return None
    else:
        if not any(np.any(fg_mask[:, c]) for c in green_lines):
            return None

    inp = canvas.copy()
    out = main(inp)

    if get_bg(inp) != bg:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
