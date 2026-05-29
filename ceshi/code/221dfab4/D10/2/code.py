# Variant of 221dfab4
# Dimension: D10 — 线段跨度与条纹宽度的关系 (meta)
# Change: 黄色条纹的宽度严格等于黄色线段的跨度（1:1映射）... → 绿色条纹宽度是黄色的两倍

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


Y = 4
G = 3


# Design analysis for the D10 variant:
# - Concrete behavior:
#   If a top-edge yellow segment spans columns 8..12 (length 5), yellow stripes still
#   paint only columns 8..12 on the Y steps. The first green stripe, which used to
#   paint only 8..12, now paints 6..15: two extra cells on the left and three on the
#   right, for a total width of 10 = 2 * 5. The green row still recolors every
#   pre-existing non-background pixel on that whole row to green.
# - Flow impact:
#   The change only affects the "which transverse span gets painted" part of main().
#   Segment detection, stripe cadence (Y / gap / Y / gap / G / gap), and the
#   full-row/full-column recolor on green steps stay intact. The generator must also
#   move the seed segment away from side borders so the wider green band is visible
#   instead of being clipped away.
# - Hidden-assumption audit:
#   The original code used one `span` for both yellow and green painting. That relied
#   on the old invariant "all stripes share the same width". Under D10 this is false,
#   so yellow and green spans must be tracked separately. The gap clearing still uses
#   only the yellow span; otherwise the wider green shoulders would be erased on the
#   next gap and the intended width contrast would disappear.
# - Placement legality:
#   The generator chooses segment starts that satisfy the original "do not touch the
#   corners" rule and the new "green expansion fits in-bounds" rule. Blob placement
#   still uses `random_free_location`, so blobs remain non-overlapping and interior.
# - Visual observability:
#   Sparse blobs alone do not guarantee that the wider green shoulders are obvious.
#   To prevent near-degenerate cases, the generator compares the new output to an
#   old-width baseline and rejects samples whose widened-green effect changes too few
#   pixels. That keeps the D10 difference visible from a few examples.


def _find_segment(g: np.ndarray):
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
        seg_len = int(c1 - c0 + 1)
        extra_lo = seg_len // 2
        extra_hi = seg_len - extra_lo
        return {
            "orient": "h",
            "start_pos": int(r0),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": H,
            "yellow_lo": int(c0),
            "yellow_hi": int(c1),
            "green_lo": max(0, int(c0) - extra_lo),
            "green_hi": min(W - 1, int(c1) + extra_hi),
            "seg_len": seg_len,
        }

    if c0 == c1:
        seg_len = int(r1 - r0 + 1)
        extra_lo = seg_len // 2
        extra_hi = seg_len - extra_lo
        return {
            "orient": "v",
            "start_pos": int(c0),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
            "yellow_lo": int(r0),
            "yellow_hi": int(r1),
            "green_lo": max(0, int(r0) - extra_lo),
            "green_hi": min(H - 1, int(r1) + extra_hi),
            "seg_len": seg_len,
        }

    return None


def _solve(grid: np.ndarray, *, wide_green: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    info = _find_segment(g)
    if info is None:
        return out

    cycle = [Y, Y, G]
    step = 0

    while True:
        pos = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step))
        if not (0 <= pos < int(info["limit"])):
            break

        col = int(cycle[int(step) % 3])
        lo = int(info["green_lo"]) if wide_green and col == G else int(info["yellow_lo"])
        hi = int(info["green_hi"]) if wide_green and col == G else int(info["yellow_hi"])

        if info["orient"] == "h":
            if col == G:
                out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])
            out[pos, lo : hi + 1] = col

            gap = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < g.shape[0]:
                out[gap, int(info["yellow_lo"]) : int(info["yellow_hi"]) + 1] = bg
        else:
            if col == G:
                out[:, pos] = np.where(out[:, pos] != bg, G, out[:, pos])
            out[lo : hi + 1, pos] = col

            gap = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < g.shape[1]:
                out[int(info["yellow_lo"]) : int(info["yellow_hi"]) + 1, gap] = bg

        step += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate stripes from the yellow border segment toward the opposite side.

    Variant D10:
    - Yellow stripes keep the original segment width.
    - Green stripes use double that width by expanding beyond the yellow span on both
      sides by half a segment (floor on the low-index side, ceil on the high-index side).
    - Green stripes still recolor every non-background pixel on the full row/column.
    """
    return _solve(grid, wide_green=True)


def _choose_segment(rng: random.Random, H: int, W: int):
    edge = rng.choice(["top", "bottom", "left", "right"])

    if edge in ("top", "bottom"):
        max_len = min(8, W // 2)
        if max_len < 3:
            return None
        seg_len = rng.randint(3, max_len)
        extra_lo = seg_len // 2
        extra_hi = seg_len - extra_lo
        start_lo = max(1, extra_lo)
        start_hi = min(W - 1 - seg_len, W - seg_len - extra_hi)
        if start_hi < start_lo:
            return None
        seg_start = rng.randint(start_lo, start_hi)
        return {
            "edge": edge,
            "orient": "h",
            "seg_len": seg_len,
            "yellow_span": list(range(seg_start, seg_start + seg_len)),
            "green_span": list(range(seg_start - extra_lo, seg_start + seg_len + extra_hi)),
            "edge_row": 0 if edge == "top" else H - 1,
        }

    max_len = min(8, H // 2)
    if max_len < 3:
        return None
    seg_len = rng.randint(3, max_len)
    extra_lo = seg_len // 2
    extra_hi = seg_len - extra_lo
    start_lo = max(1, extra_lo)
    start_hi = min(H - 1 - seg_len, H - seg_len - extra_hi)
    if start_hi < start_lo:
        return None
    seg_start = rng.randint(start_lo, start_hi)
    return {
        "edge": edge,
        "orient": "v",
        "seg_len": seg_len,
        "yellow_span": list(range(seg_start, seg_start + seg_len)),
        "green_span": list(range(seg_start - extra_lo, seg_start + seg_len + extra_hi)),
        "edge_col": 0 if edge == "left" else W - 1,
    }


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    seg = _choose_segment(rng, H, W)
    if seg is None:
        return None

    grid = np.full((H, W), bg, dtype=int)

    if seg["orient"] == "h":
        for c in seg["yellow_span"]:
            grid[int(seg["edge_row"]), int(c)] = Y
        forbidden_rows = {int(seg["edge_row"])}
        forbidden_cols = set()
    else:
        for r in seg["yellow_span"]:
            grid[int(r), int(seg["edge_col"])] = Y
        forbidden_rows = set()
        forbidden_cols = {int(seg["edge_col"])}

    canvas = grid.copy()
    n_blobs_target = rng.randint(5, 9)

    for _ in range(n_blobs_target * 6):
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

        fg_mask = (canvas != bg) & (canvas != Y)
        _lab, n_placed = label_mask(fg_mask, connectivity=4)
        if int(n_placed) >= n_blobs_target:
            break

    fg_mask = (canvas != bg) & (canvas != Y)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)
    baseline = _solve(inp, wide_green=False)
    width_gain = int(np.sum(out != baseline))

    if width_gain < int(seg["seg_len"]):
        return None

    if get_bg(inp) != bg:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
