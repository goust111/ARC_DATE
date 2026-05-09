# Variant of 221dfab4
# Dimension: D10 — 线段跨度与条纹宽度的关系 (meta)
# Change: 黄色条纹的宽度严格等于黄色线段的跨度（1:1映射）... → 条纹宽度递减收缩

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite

# Design analysis for the D10 variant
# a) Concrete behavior:
#    A border yellow segment of length L still launches stripes every 2 cells with the
#    color cycle Y,Y,G. The change is local to stripe width: stripe k uses width L-k,
#    so the painted cross-section shrinks 1 cell per stripe until width 1, then stops.
#    Example with a top segment spanning cols 10..15 (L=6):
#    original widths by row = 6,6,6,6,6,...
#    variant widths by row = 6,5,4,3,2,1 on rows 0,2,4,6,8,10.
#    The gap row after each stripe is still forced to background, but only inside that
#    stripe's current narrowed span. Green stripes still recolor the entire row/column's
#    existing non-background pixels to green before repainting the narrowed stripe cells.
# b) Complexity calibration:
#    If L were too small, shrinking would be hard to notice and could visually collapse
#    back into the original task. The generator therefore uses L in [5,8], which yields
#    several distinct widths and at least one green flood stripe in normal-sized grids.
# c) Implicit-assumption audit against the original solver:
#    - The old code assumes one fixed `span` works for every stripe and every gap write.
#      That assumption is no longer valid; both stripe paint and gap clearing must use a
#      per-step shrunk span.
#    - The old unbounded while-loop stops only at the opposite edge. Here width can hit 0
#      earlier, so iteration must also stop after the width-1 stripe.
#    - Green flood order remains safe: recolor the full row/column first, then overwrite
#      the current stripe cells with green. This still matches the original semantics.
# d) Placement legality:
#    The generator still places one straight border segment plus several interior blobs
#    with border=1 and padding=1, so blobs do not overlap each other or the frame. Grid
#    sizes are chosen large enough that all shrinking stripes fit in-bounds for L<=8.
# e) Visual observability:
#    Samples are rejected unless the produced output shows multiple distinct stripe widths
#    and at least one green flood changes a foreground pixel outside the narrowed stripe.
#    This prevents most degenerate cases where the puzzle would look too close to the
#    original equal-width behavior.
# New solver assumptions checked against the generator:
#    `main()` assumes the yellow border mark is a straight contiguous component on one
#    edge. The generator explicitly constructs exactly that. It also assumes at least a
#    few stripe positions fit in-bounds; H,W >= 22 and L <= 8 guarantee that.

Y = 4
G = 3


def _find_segment(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(Y)) & border
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
            "limit": H,
        }
    if c0 == c1:
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span_lo": int(r0),
            "span_hi": int(r1),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
        }
    return None


def _shrunk_interval(span_lo: int, span_hi: int, step: int):
    base_len = int(span_hi) - int(span_lo) + 1
    width = int(base_len) - int(step)
    if width <= 0:
        return None

    # When parity forces a choice, keep the lower-index side one cell longer.
    left_trim = int(step) // 2
    right_trim = int(step) - int(left_trim)
    lo = int(span_lo) + int(left_trim)
    hi = int(span_hi) - int(right_trim)
    if lo > hi:
        return None
    return int(lo), int(hi)


def _stripe_steps(grid: np.ndarray, info=None):
    info = _find_segment(grid) if info is None else info
    if info is None:
        return []

    cycle = [int(Y), int(Y), int(G)]
    steps = []
    base_len = int(info["span_hi"]) - int(info["span_lo"]) + 1
    for step in range(base_len):
        pos = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step))
        if not (0 <= pos < int(info["limit"])):
            break
        interval = _shrunk_interval(int(info["span_lo"]), int(info["span_hi"]), int(step))
        if interval is None:
            break
        lo, hi = interval
        steps.append(
            {
                "orient": info["orient"],
                "pos": int(pos),
                "color": int(cycle[int(step) % 3]),
                "span_lo": int(lo),
                "span_hi": int(hi),
                "width": int(hi - lo + 1),
                "step": int(step),
            }
        )
    return steps


def _variant_metrics(inp: np.ndarray, out: np.ndarray):
    g = np.array(inp, dtype=int)
    o = np.array(out, dtype=int)
    bg = int(get_bg(g))
    steps = _stripe_steps(g)
    if not steps:
        return None

    widths = [int(s["width"]) for s in steps]
    yellow_widths = [int(s["width"]) for s in steps if int(s["color"]) == int(Y)]

    green_flood = 0
    for s in steps:
        if int(s["color"]) != int(G):
            continue
        pos = int(s["pos"])
        lo = int(s["span_lo"])
        hi = int(s["span_hi"])
        if s["orient"] == "h":
            if lo > 0:
                green_flood += int(np.sum((g[pos, :lo] != int(bg)) & (o[pos, :lo] == int(G))))
            if hi + 1 < g.shape[1]:
                green_flood += int(np.sum((g[pos, hi + 1 :] != int(bg)) & (o[pos, hi + 1 :] == int(G))))
        else:
            if lo > 0:
                green_flood += int(np.sum((g[:lo, pos] != int(bg)) & (o[:lo, pos] == int(G))))
            if hi + 1 < g.shape[0]:
                green_flood += int(np.sum((g[hi + 1 :, pos] != int(bg)) & (o[hi + 1 :, pos] == int(G))))

    return {
        "steps": len(steps),
        "distinct_widths": len(set(widths)),
        "yellow_distinct_widths": len(set(yellow_widths)),
        "width_drop": int(widths[0] - widths[-1]),
        "green_flood": int(green_flood),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch alternating Y/Y/G stripes from the yellow edge segment, but shrink the span
    by 1 cell on every stripe until the width reaches 1.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    info = _find_segment(g)
    if info is None:
        return out

    for stripe in _stripe_steps(g, info=info):
        pos = int(stripe["pos"])
        lo = int(stripe["span_lo"])
        hi = int(stripe["span_hi"])
        col = int(stripe["color"])

        if stripe["orient"] == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            out[pos, lo : hi + 1] = int(col)
            gap = pos + int(info["dir1"])
            if 0 <= gap < H:
                out[gap, lo : hi + 1] = int(bg)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            out[lo : hi + 1, pos] = int(col)
            gap = pos + int(info["dir1"])
            if 0 <= gap < W:
                out[lo : hi + 1, gap] = int(bg)

    return out


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (int(G), int(Y))]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(5, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1 or H < 2 * seg_len - 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
    else:
        max_start = H - 1 - seg_len
        if max_start < 1 or W < 2 * seg_len - 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        for c in span:
            grid[edge_row, c] = int(Y)
    else:
        for r in span:
            grid[r, edge_col] = int(Y)

    if orient == "h":
        forbidden_rows = {int(edge_row)}
        forbidden_cols = set()
    else:
        forbidden_rows = set()
        forbidden_cols = {int(edge_col)}

    target_blobs = rng.randint(5, 9)
    canvas = grid.copy()
    placed = 0
    attempts = 0

    while placed < target_blobs and attempts < target_blobs * 8:
        attempts += 1
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

        if blob_rows & forbidden_rows:
            continue
        if blob_cols & forbidden_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]
        placed += 1

    fg_mask = (canvas != bg) & (canvas != int(Y))
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
        return None

    metrics = _variant_metrics(inp, out)
    if metrics is None:
        return None
    if int(metrics["steps"]) < 5:
        return None
    if int(metrics["distinct_widths"]) < 4:
        return None
    if int(metrics["yellow_distinct_widths"]) < 3:
        return None
    if int(metrics["width_drop"]) < 4:
        return None
    if int(metrics["green_flood"]) <= 0:
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
