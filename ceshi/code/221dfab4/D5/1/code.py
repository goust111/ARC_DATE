# Variant of 221dfab4
# Dimension: D5 — 间隙处理规则 (trajectory)
# Change: 条纹之间的1格间隙在线段跨度内强制清除为背景色... → 间隙也执行重着色（用第三种颜色）

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parents[5] / "arc2_solver_select"
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


def _find_segment(g: np.ndarray, yellow: int) -> tuple[str, int, list[int], int] | None:
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(yellow)) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))
    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        start_pos = int(r0)
        cols = sorted(int(c) for c in cs)
        return "h", start_pos, list(range(int(cols[0]), int(cols[-1]) + 1)), H
    if c0 == c1:
        start_pos = int(c0)
        rows = sorted(int(r) for r in rs)
        return "v", start_pos, list(range(int(rows[0]), int(rows[-1]) + 1)), W
    return None


def _trajectory_positions(start_pos: int, limit: int, direction: int) -> tuple[list[int], list[int]]:
    stripes = []
    gaps = []
    step = 0
    while True:
        pos = int(start_pos) + int(direction) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        stripes.append(int(pos))
        gap = int(start_pos) + int(direction) * (2 * int(step) + 1)
        if 0 <= gap < int(limit):
            gaps.append(int(gap))
        step += 1
    return stripes, gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Analysis for the D5 variant:
    - source_task is None, so there is no extra reference solver to inspect.
    - Concrete behavior: keep the original stripe rows/cols at offsets 0,2,4,... from the
      yellow border segment and keep their color cycle Y,Y,G. Only the 1-cell separators at
      offsets 1,3,5,... change: they are no longer erased to bg inside the segment span; they
      become red (2). Example with a top segment spanning cols 5..7:
      original rows 0..4 over that span = [444], [bgbgbg], [444], [bgbgbg], [333];
      variant rows 0..4 over that span  = [444], [222],   [444], [222],   [333].
      Green rows/cols still additionally recolor every non-bg cell on the whole row/column.
    - Complexity calibration: the variant must stay visually distinct from the original, so the
      generator forbids bg=2 and fg=2. Otherwise a red gap could collapse back into "looks like
      background" or become indistinguishable from the pre-existing blobs.
    - Old implicit assumption audit:
      1. The original gap-writing code assumed "overwrite with bg is always harmless". That is no
         longer true because the separator is now meaningful output, so the gap pass must paint 2.
      2. The segment parser assumes the yellow marker is one straight border component. The
         generator still guarantees one contiguous border segment not touching corners.
      3. The green flood uses `out != bg` on the current canvas. This still holds because bg stays
         fixed and the new red separators are intended to count as non-bg output once painted.
    - Placement legality and new solver assumptions:
      1. Objects must remain interior, non-overlapping, and away from the border start line so the
         yellow marker is the unique border signal. The generator keeps border=1, padding=1, and
         avoids the segment's own border row/col.
      2. The solver now assumes red is reserved for the separator rule. The generator enforces that
         by excluding 2 from both bg and blob colors.
    - Visual observability: even if no blob intersects a separator, the output still contains
      repeated red bars inside the segment span. Because every valid sample has at least one in-grid
      separator and bg != 2, the D5 change is always directly visible to a human solver.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    Y = 4
    R = 2
    G = 3

    segment = _find_segment(g, Y)
    if segment is None:
        return out

    orient, start_pos, span, limit = segment
    direction = 1 if start_pos == 0 else -1
    cycle = [int(Y), int(Y), int(G)]

    step = 0
    while True:
        pos = int(start_pos) + int(direction) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        stripe_color = int(cycle[int(step) % 3])
        if orient == "h":
            if stripe_color == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in span:
                out[pos, int(c)] = stripe_color

            gap = int(start_pos) + int(direction) * (2 * int(step) + 1)
            if 0 <= gap < H:
                for c in span:
                    out[int(gap), int(c)] = int(R)
        else:
            if stripe_color == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            for r in span:
                out[int(r), pos] = stripe_color

            gap = int(start_pos) + int(direction) * (2 * int(step) + 1)
            if 0 <= gap < W:
                for r in span:
                    out[int(r), int(gap)] = int(R)

        step += 1

    return out


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    # Red=2 is reserved for the new separator rule, so keep it absent from the input palette.
    available = [c for c in range(10) if c not in (2, 3, 4)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        for c in span:
            grid[edge_row, c] = 4
        forbidden_rows = {int(edge_row)}
        forbidden_cols = set()
    else:
        for r in span:
            grid[r, edge_col] = 4
        forbidden_rows = set()
        forbidden_cols = {int(edge_col)}

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

        sprite_mask = sprite != bg
        blob_rows = {int(pr + dr) for dr in range(sh) if sprite_mask[dr].any()}
        blob_cols = {int(pc + dc) for dc in range(sw) if sprite_mask[:, dc].any()}

        if orient == "h" and blob_rows & forbidden_rows:
            continue
        if orient == "v" and blob_cols & forbidden_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = (canvas != bg) & (canvas != 4)
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if n_placed < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if get_bg(inp) != bg:
        return None

    # D5-specific guard: every generated sample must visibly contain at least one red separator.
    segment = _find_segment(inp, 4)
    if segment is None:
        return None
    orient2, start_pos, span2, limit = segment
    direction = 1 if start_pos == 0 else -1
    _, gaps = _trajectory_positions(start_pos, limit, direction)
    if not gaps:
        return None
    red_gap_cells = 0
    if orient2 == "h":
        for gap in gaps:
            red_gap_cells += int(np.sum(out[int(gap), span2] == 2))
    else:
        for gap in gaps:
            red_gap_cells += int(np.sum(out[span2, int(gap)] == 2))
    if red_gap_cells != len(gaps) * len(span2):
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
