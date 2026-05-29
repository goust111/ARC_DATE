# Variant of 221dfab4
# Dimension: D10 — 线段跨度与条纹宽度的关系 (meta)
# Change: 黄色条纹的宽度严格等于黄色线段的跨度（1:1映射）... → 条纹宽度递增扩展

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design notes for the D10 variant
# a) Concretization and trace:
#    Original behavior keeps every stripe confined to the exact border-segment span.
#    Here the stripe at step s uses width = base_len + s. The interval grows outward
#    from the original span, with the odd extra cell rounded to the larger index side.
#    Example: base span [6..9] gives widths/intervals
#    s=0 -> [6..9], s=1 -> [6..10], s=2 -> [5..10], s=3 -> [5..11].
#    The stripe positions still occur every 2 cells and colors still cycle Y,Y,G.
#    The 1-cell gap is also cleared on the next wider interval, so the visible envelope
#    becomes a fan instead of a stack of fixed-width bars.
# b) Complexity calibration:
#    If expansion were allowed to clip immediately, many samples would look close to the
#    original puzzle. The generator therefore enforces side margins so several early
#    stripes widen before touching grid borders, making the D10 change visually obvious.
# c) Hidden-assumption audit of the original solver:
#    - `span = list(range(...))` assumed one immutable span for all steps. That is false now.
#    - Gap clearing only on the original span was harmless before, but would now leave
#      stray foreground inside the fan; this variant clears the next expanded interval.
#    - Green rows/cols still recolor all existing non-bg pixels globally, but the explicit
#      stripe paint must use the expanded interval rather than the base span.
# d) Placement legality:
#    Objects are still non-overlapping interior blobs. The generator keeps the border
#    segment away from corners and reserves blank space near the first few stripe/gap
#    positions so the widening fan is legible and solver assumptions remain valid.
# e) Observability:
#    Across examples, humans can compare successive yellow rows/cols and see widths
#    increase by exactly one cell per stripe, unlike the constant-width source task.
#
# New solver assumptions introduced by this variant, and how generation guarantees them:
# - The yellow seed is a single straight border component: generator places exactly one.
# - The border segment has enough side margin that at least a few stripes widen unclipped.
# - The grid is large enough for multiple stripe steps: generator uses sizes 24+.

Y = 4
G = 3


def _find_seed_segment(g: np.ndarray):
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == Y) & border
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
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span_lo": int(c0),
            "span_hi": int(c1),
            "dir1": 1 if int(r0) == 0 else -1,
            "travel_limit": H,
            "span_limit": W,
        }
    if c0 == c1:
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span_lo": int(r0),
            "span_hi": int(r1),
            "dir1": 1 if int(c0) == 0 else -1,
            "travel_limit": W,
            "span_limit": H,
        }
    return None


def _expanded_interval(span_lo: int, span_hi: int, step: int, limit: int) -> tuple[int, int]:
    # Width grows by 1 each stripe. Odd growth rounds the extra half-cell to the larger index side.
    grow_left = int(step) // 2
    grow_right = int(step) - grow_left
    lo = max(0, int(span_lo) - grow_left)
    hi = min(int(limit) - 1, int(span_hi) + grow_right)
    return int(lo), int(hi)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate stripes from the yellow border segment toward the opposite edge.

    Variant D10:
    - Stripe color cycle stays Y, Y, G on every other row/column.
    - Stripe width is no longer fixed. At step s, the painted span grows to base_len + s.
    - Green stripes still recolor all non-bg pixels on the full row/column to green.
    - The 1-cell gap between stripes is cleared on the next widened interval, preserving
      a clean fan-shaped envelope.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    seed = _find_seed_segment(g)
    if seed is None:
        return out

    cycle = [Y, Y, G]
    step = 0

    while True:
        pos = int(seed["start_pos"]) + int(seed["dir1"]) * (2 * int(step))
        if not (0 <= pos < int(seed["travel_limit"])):
            break

        col = int(cycle[int(step) % 3])
        lo, hi = _expanded_interval(
            int(seed["span_lo"]), int(seed["span_hi"]), int(step), int(seed["span_limit"])
        )

        if seed["orient"] == "h":
            if col == G:
                out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])
            out[pos, lo : hi + 1] = col

            gap = int(seed["start_pos"]) + int(seed["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < H:
                gap_lo, gap_hi = _expanded_interval(
                    int(seed["span_lo"]), int(seed["span_hi"]), int(step) + 1, int(seed["span_limit"])
                )
                out[gap, gap_lo : gap_hi + 1] = bg
        else:
            if col == G:
                out[:, pos] = np.where(out[:, pos] != bg, G, out[:, pos])
            out[lo : hi + 1, pos] = col

            gap = int(seed["start_pos"]) + int(seed["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < W:
                gap_lo, gap_hi = _expanded_interval(
                    int(seed["span_lo"]), int(seed["span_hi"]), int(step) + 1, int(seed["span_limit"])
                )
                out[gap_lo : gap_hi + 1, gap] = bg

        step += 1

    return out


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    orient = "h" if edge in ("top", "bottom") else "v"

    preview_steps = rng.randint(4, 6)
    side_margin = preview_steps // 2 + 2
    seg_len = rng.randint(3, 6)

    if orient == "h":
        max_start = W - seg_len - side_margin
        min_start = side_margin
        if min_start > max_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_row = 0 if edge == "top" else H - 1
        row_dir = 1 if edge == "top" else -1
        trajectory = {edge_row + row_dir * d for d in range(0, 2 * preview_steps + 1)}
        trajectory = {r for r in trajectory if 0 <= r < H}
    else:
        max_start = H - seg_len - side_margin
        min_start = side_margin
        if min_start > max_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_col = 0 if edge == "left" else W - 1
        col_dir = 1 if edge == "left" else -1
        trajectory = {edge_col + col_dir * d for d in range(0, 2 * preview_steps + 1)}
        trajectory = {c for c in trajectory if 0 <= c < W}

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        grid[edge_row, span] = Y
    else:
        grid[span, edge_col] = Y

    canvas = grid.copy()

    target_blobs = rng.randint(5, 8)
    placed = 0

    for _ in range(40):
        if placed >= target_blobs:
            break

        bh = rng.randint(2, 9)
        bw = rng.randint(2, 9)
        density = rng.uniform(0.4, 0.82)

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

        if orient == "h" and blob_rows & trajectory:
            continue
        if orient == "v" and blob_cols & trajectory:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]
        placed += 1

    fg_mask = (canvas != bg) & (canvas != Y)
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != bg:
        return None

    seed = _find_seed_segment(inp)
    if seed is None:
        return None

    base_width = int(seed["span_hi"]) - int(seed["span_lo"]) + 1
    unclipped_growth = 0
    for step in range(preview_steps):
        lo, hi = _expanded_interval(int(seed["span_lo"]), int(seed["span_hi"]), step, int(seed["span_limit"]))
        if hi - lo + 1 > base_width:
            unclipped_growth += 1
    if unclipped_growth < 3:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
