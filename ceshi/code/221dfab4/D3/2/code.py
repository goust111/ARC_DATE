# Variant of 221dfab4
# Dimension: D3 — 颜色循环模式 (trajectory)
# Change: 三步周期循环 Y(4)-Y(4)-G(3)，即每三条条纹重复
#         -> 由线段长度编码周期，周期 = L 条黄色 + 1 条绿色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for this D3 variant:
# - source_task is None, so there is no extra reference solver to read beyond the original 221dfab4 code.
# - Concrete behavior: if a top-edge yellow segment has length L=5 and spans cols 7..11, then rows
#   0,2,4,6,8 paint yellow only inside cols 7..11, row 10 becomes the first green stripe, and only then
#   does the full-row recoloring happen. In the original task the first green stripe would already appear
#   at row 4 because the cycle was fixed Y,Y,G. The new distinguishing event is therefore:
#   original first green distance from border = 4 cells, variant first green distance from border = 2*L cells.
# - Complexity calibration: if L were allowed to be 2, the task would collapse back to the original Y,Y,G
#   timing. The generator therefore uses L in [3,6] and requires at least one visible green stripe.
# - Hidden-assumption audit for main():
#   * The old code could tolerate measuring span by bbox because the cycle length was constant.
#     Here the segment length itself drives the rule, so we explicitly require the detected border
#     component to be a straight contiguous run; otherwise measuring by min/max could overestimate L.
#   * Gap clearing remains safe because stripes are still spaced by 2, so no stripe overwrites a prior stripe.
#   * Green-row/column recoloring still happens before repainting the span, which is harmless because the
#     span should also be green on green steps.
# - Placement legality in the generator:
#   * The yellow border segment is still a single contiguous run that avoids corners.
#   * Random blobs are placed with border=1 and padding=1 so they do not overlap or merge.
#   * A small witness blob is forced onto the first green stripe outside the yellow span, guaranteeing that
#     the "full row/column turns green" effect is visibly present in the output.
# - Visual observability: every generated sample has L>=3 and at least one witness-bearing green stripe, so
#   more than two yellow stripes appear before the first green event and the sample cannot visually reduce to
#   the original fixed Y,Y,G timing.


def _find_segment_info(g: np.ndarray):
    H, W = g.shape
    Y = 4

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
    if pts.size == 0:
        return None

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        cols = sorted(int(c) for c in cs)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span0": int(cols[0]),
            "span1": int(cols[-1]),
            "seg_len": int(len(cols)),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": int(H),
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span0": int(rows[0]),
            "span1": int(rows[-1]),
            "seg_len": int(len(rows)),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": int(W),
        }

    return None


def _stripe_color(step: int, seg_len: int) -> int:
    return 3 if int(step) % (int(seg_len) + 1) == int(seg_len) else 4


def _green_positions(start_pos: int, dir1: int, limit: int, seg_len: int) -> list[int]:
    positions = []
    step = int(seg_len)
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        positions.append(int(pos))
        step += int(seg_len) + 1
    return positions


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    for dr in range(h):
        for dc in range(w):
            if int(patch[dr, dc]) != int(bg):
                canvas[int(top) + dr, int(left) + dc] = int(patch[dr, dc])


def _add_green_witness(
    canvas: np.ndarray,
    orient: str,
    span0: int,
    span1: int,
    green_pos: int,
    bg: int,
    fg: int,
    rng: random.Random,
) -> bool:
    patch = np.full((2, 2), int(fg), dtype=int)
    H, W = canvas.shape

    if orient == "h":
        top = int(green_pos) - 1
        if not (1 <= top and top + 1 < H - 1):
            return False
        choices = list(range(1, max(1, int(span0) - 1))) + list(range(int(span1) + 1, W - 2))
        rng.shuffle(choices)
        for left in choices:
            if left + 1 >= W - 1:
                continue
            if np.any(canvas[max(0, top - 1) : min(H, top + 3), max(0, left - 1) : min(W, left + 3)] != int(bg)):
                continue
            _paste_patch(canvas, patch, top, left, bg)
            return True
        return False

    left = int(green_pos) - 1
    if not (1 <= left and left + 1 < W - 1):
        return False
    choices = list(range(1, max(1, int(span0) - 1))) + list(range(int(span1) + 1, H - 2))
    rng.shuffle(choices)
    for top in choices:
        if top + 1 >= H - 1:
            continue
        if np.any(canvas[max(0, top - 1) : min(H, top + 3), max(0, left - 1) : min(W, left + 3)] != int(bg)):
            continue
        _paste_patch(canvas, patch, top, left, bg)
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    info = _find_segment_info(g)
    if info is None:
        return out

    orient = info["orient"]
    start_pos = int(info["start_pos"])
    span0 = int(info["span0"])
    span1 = int(info["span1"])
    seg_len = int(info["seg_len"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])

    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        col = int(_stripe_color(step, seg_len))

        if orient == "h":
            if col == 3:
                out[pos, :] = np.where(out[pos, :] != int(bg), 3, out[pos, :])
            out[pos, span0 : span1 + 1] = col
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < out.shape[0]:
                out[gap, span0 : span1 + 1] = int(bg)
        else:
            if col == 3:
                out[:, pos] = np.where(out[:, pos] != int(bg), 3, out[:, pos])
            out[span0 : span1 + 1, pos] = col
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < out.shape[1]:
                out[span0 : span1 + 1, gap] = int(bg)

        step += 1

    return out


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (3, 4)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 6)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        orient = "h"
        start_pos = 0 if edge == "top" else H - 1
        span0 = int(seg_start)
        span1 = int(seg_start + seg_len - 1)
        limit = int(H)
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        orient = "v"
        start_pos = 0 if edge == "left" else W - 1
        span0 = int(seg_start)
        span1 = int(seg_start + seg_len - 1)
        limit = int(W)

    dir1 = 1 if start_pos == 0 else -1
    green_positions = _green_positions(start_pos, dir1, limit, seg_len)
    if not green_positions:
        return None

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        grid[start_pos, span0 : span1 + 1] = 4
        forbidden_rows = {int(start_pos)}
        forbidden_cols = set()
    else:
        grid[span0 : span1 + 1, start_pos] = 4
        forbidden_rows = set()
        forbidden_cols = {int(start_pos)}

    canvas = grid.copy()
    if not _add_green_witness(canvas, orient, span0, span1, green_positions[0], bg, fg, rng):
        return None

    target_blobs = rng.randint(5, 8)
    attempts = 0
    while attempts < 80:
        attempts += 1
        bh = rng.randint(2, 8)
        bw = rng.randint(2, 8)
        density = rng.uniform(0.4, 0.8)

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
        if pix < 5 or pix > 45:
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

        _paste_patch(canvas, sprite, pr, pc, bg)

        fg_mask = (canvas != bg) & (canvas != 4)
        _lab, n_placed = label_mask(fg_mask, connectivity=4)
        if int(n_placed) >= int(target_blobs):
            break

    fg_mask = (canvas != bg) & (canvas != 4)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if orient == "h":
        witness_visible = any(
            np.any((canvas[pos, :] != bg) & ~((np.arange(W) >= span0) & (np.arange(W) <= span1)))
            for pos in green_positions
        )
    else:
        witness_visible = any(
            np.any((canvas[:, pos] != bg) & ~((np.arange(H) >= span0) & (np.arange(H) <= span1)))
            for pos in green_positions
        )
    if not witness_visible:
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
