# Variant of 221dfab4
# Dimension: D2 — 步长与间距 (path)
# Change: 步长为2（stride 2），条纹之间有1格背景间隙... → 递增步长

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

HELPER_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(HELPER_DIR) not in sys.path:
    sys.path.append(str(HELPER_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the D2 variant:
# a) Concrete behavior and trace:
#    Keep the same yellow border segment, Y-Y-G color cycle, and "green floods the
#    whole row/column" behavior. Only the path spacing changes. If a top-edge
#    segment spans cols 5..8, the original solver paints stripe rows
#    0, 2, 4, 6, ... with single-cell gaps. The variant paints rows
#    0, 2, 5, 9, 14, ... because the gaps are 1, 2, 3, 4, ... cells long.
#    So the first local difference appears at the third stripe:
#    original row 4 becomes variant row 5. This affects the stripe-placement loop
#    and the gap-clearing code, but not segment detection or the color cycle.
# b) Complexity calibration:
#    To make the non-uniform path visible to a human, each generated grid must
#    admit at least three different gap lengths. With the chosen sizes, every
#    valid trajectory has 6+ visible stripes, so the increasing sequence is
#    visible directly from the yellow stripe spacing even when no blob is hit.
# c) Hidden-assumption audit:
#    - The original `pos = start + 2*step` assumes constant stride. This is the
#      main line that becomes invalid; we replace it with cumulative increments
#      2, 3, 4, ...
#    - The original single `gap = start + (2*step+1)` assumes exactly one gap
#      cell. In the variant every in-between cell must be reset to bg.
#    - The "largest yellow border component is the seed segment" assumption still
#      holds only if the generator never introduces extra yellow objects; it does
#      not, because fg colors exclude 3 and 4.
#    - The segment must remain straight and contiguous on one border; the
#      generator keeps that invariant.
# d) Placement legality:
#    Blobs are still placed with border/padding separation and never overlap the
#    seed segment. No special placement is needed for later stripes because they
#    do not exist in the input; the solver constructs them in the output.
# e) Visual observability:
#    The variable gap lengths are readable from the stripe geometry itself:
#    1-cell gap, then 2-cell gap, then 3-cell gap, etc. This remains observable
#    even if green flood rows/cols happen not to touch a blob in one sample.
# New solver assumptions after the change:
#    `main()` now assumes the border component is still the only straight yellow
#    seed and that the grid dimension along the motion axis is large enough to
#    expose at least three gaps. `generate_input()` guarantees both.

Y = 4
G = 3


def _extract_seed_segment(g: np.ndarray):
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
        cols = sorted(int(c) for c in cs)
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": list(range(int(cols[0]), int(cols[-1]) + 1)),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": int(H),
        }
    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": list(range(int(rows[0]), int(rows[-1]) + 1)),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": int(W),
        }

    return None


def _stripe_positions(start_pos: int, dir1: int, limit: int) -> list[int]:
    positions: list[int] = []
    pos = int(start_pos)
    step = 0
    while 0 <= pos < int(limit):
        positions.append(int(pos))
        pos += int(dir1) * (int(step) + 2)
        step += 1
    return positions


def _visible_gap_lengths(limit: int) -> tuple[int, ...]:
    pos = _stripe_positions(0, 1, limit)
    return tuple(int(pos[i + 1] - pos[i] - 1) for i in range(len(pos) - 1))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    seed = _extract_seed_segment(g)
    if seed is None:
        return out

    orient = str(seed["orient"])
    span = list(seed["span"])
    limit = int(seed["limit"])
    dir1 = int(seed["dir1"])

    cycle = [Y, Y, G]
    pos = int(seed["start_pos"])
    step = 0

    while 0 <= pos < limit:
        col = int(cycle[int(step) % 3])
        next_pos = int(pos) + int(dir1) * (int(step) + 2)

        if orient == "h":
            if col == G:
                out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])
            out[pos, span] = col
            if 0 <= next_pos < limit:
                lo, hi = sorted((int(pos), int(next_pos)))
                gap_rows = list(range(lo + 1, hi))
                if gap_rows:
                    out[np.ix_(gap_rows, span)] = bg
        else:
            if col == G:
                out[:, pos] = np.where(out[:, pos] != bg, G, out[:, pos])
            out[span, pos] = col
            if 0 <= next_pos < limit:
                lo, hi = sorted((int(pos), int(next_pos)))
                gap_cols = list(range(lo + 1, hi))
                if gap_cols:
                    out[np.ix_(span, gap_cols)] = bg

        pos = next_pos
        step += 1

    return out


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
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
        edge_pos = 0 if edge == "top" else H - 1
        motion_limit = H
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_pos = 0 if edge == "left" else W - 1
        motion_limit = W

    visible_gaps = _visible_gap_lengths(int(motion_limit))
    if len(visible_gaps) < 3 or len(set(visible_gaps[:3])) < 3:
        return None

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        grid[int(edge_pos), span] = Y
        forbidden_rows = {int(edge_pos)}
        forbidden_cols = set()
    else:
        grid[span, int(edge_pos)] = Y
        forbidden_rows = set()
        forbidden_cols = {int(edge_pos)}

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
    if n_placed < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
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
