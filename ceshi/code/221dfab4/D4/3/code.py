# Variant of 221dfab4
# Dimension: D4 — 绿色条纹的全行/列重着色范围 (trajectory)
# Change: 绿色条纹将全行/列的所有非背景像素重着色为绿色（全局 flood 效果）... → 距离约束的重着色范围
#
# Design analysis for this D4 variant
# -----------------------------------
# Reference-task note:
# - The allowed local workspace contains the official cb2d8a2c annotation and I/O
#   samples, but no standalone cb2d8a2c solver file. The distance-constraint
#   baseline below is therefore inferred from those local materials only.
#
# a) Concrete behavior:
# - Geometry is unchanged from 221dfab4: a yellow border segment emits stripes
#   every 2 cells with the repeating color cycle Y, Y, G, and the 1-cell gaps
#   are erased to background within the segment span.
# - Only the green-stripe side effect changes. Old behavior: if a stripe lands on
#   row r (or column c), every non-background pixel on that entire row/column
#   becomes green. New behavior: only non-background pixels whose 1D distance from
#   the stripe-span center is <= N are recolored, where N = segment length.
# - Small traced example for horizontal stripes:
#   span columns = 8..12, center = 10, N = 5. On a green stripe row, a pixel at
#   column 6 is recolored (distance 4), while a pixel at column 19 is not
#   recolored (distance 9). Under the original task both would become green.
#
# b) Complexity calibration:
# - If the generator placed only compact blobs near the segment center, the new
#   behavior would visually collapse back to the original full-row flood. To avoid
#   that, the generator injects probe blobs on green rows/columns both inside and
#   outside the allowed radius, so each sample shows a local-vs-far contrast.
#
# c) Old implicit assumptions that break under D4:
# - `out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])` assumed every
#   non-bg pixel on a green row/column should flip. That is no longer valid.
# - The original generator only needed "some foreground on green rows/cols".
#   Under D4 it must guarantee foreground exists on both sides of the radius
#   boundary, otherwise the sample degenerates to the old visual effect.
#
# d) Placement legality:
# - Probe blobs are placed as separated 2x2 blocks with a 1-cell padding check,
#   so they never overlap each other, the yellow edge segment, or later random
#   blobs. Random blobs still use `random_free_location`, preserving legality.
#
# e) Visual observability:
# - Each generated sample contains at least two green trajectory lines with
#   foreground blocks near the stripe center and other blocks farther away on the
#   same line. The output therefore shows "partial recolor on a green line" rather
#   than the original task's global full-line recolor.

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


Y = 4
G = 3


def _segment_info(grid: np.ndarray):
    g = np.array(grid, dtype=int)
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
        span = list(range(cols[0], cols[-1] + 1))
        return {
            "orient": "h",
            "start_pos": int(r0),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": H,
            "span": span,
            "center": 0.5 * (span[0] + span[-1]),
            "radius": len(span),
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        span = list(range(rows[0], rows[-1] + 1))
        return {
            "orient": "v",
            "start_pos": int(c0),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
            "span": span,
            "center": 0.5 * (span[0] + span[-1]),
            "radius": len(span),
        }

    return None


def _trajectory(start_pos: int, dir1: int, limit: int) -> list[tuple[int, int, int]]:
    cycle = [Y, Y, G]
    out: list[tuple[int, int, int]] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        out.append((int(step), int(pos), int(cycle[int(step) % 3])))
        step += 1
    return out


def _within_radius_indices(length: int, center: float, radius: int) -> np.ndarray:
    return np.abs(np.arange(int(length), dtype=float) - float(center)) <= float(radius)


def _variant_target_counts(grid: np.ndarray) -> tuple[int, int]:
    g = np.array(grid, dtype=int)
    info = _segment_info(g)
    if info is None:
        return 0, 0

    bg = int(get_bg(g))
    traj = _trajectory(info["start_pos"], info["dir1"], info["limit"])
    span = info["span"]
    center = float(info["center"])
    radius = int(info["radius"])
    near = 0
    far = 0

    if info["orient"] == "h":
        line_mask = _within_radius_indices(g.shape[1], center, radius)
        span_mask = np.zeros(g.shape[1], dtype=bool)
        span_mask[span[0] : span[-1] + 1] = True
        for _step, pos, col in traj:
            if col != G:
                continue
            fg = g[pos, :] != bg
            near += int(np.count_nonzero(fg & line_mask & ~span_mask))
            far += int(np.count_nonzero(fg & ~line_mask & ~span_mask))
    else:
        line_mask = _within_radius_indices(g.shape[0], center, radius)
        span_mask = np.zeros(g.shape[0], dtype=bool)
        span_mask[span[0] : span[-1] + 1] = True
        for _step, pos, col in traj:
            if col != G:
                continue
            fg = g[:, pos] != bg
            near += int(np.count_nonzero(fg & line_mask & ~span_mask))
            far += int(np.count_nonzero(fg & ~line_mask & ~span_mask))

    return near, far


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the original YYG striped pattern from the yellow border segment.

    D4 change:
    - Yellow stripes still paint only inside the segment span.
    - Green stripes still paint the span green, but their extra recolor effect is
      now local: only non-background pixels within radius N of the span center are
      recolored, where N is the segment length.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    info = _segment_info(g)
    if info is None:
        return out

    orient = info["orient"]
    start_pos = int(info["start_pos"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])
    span = list(info["span"])
    center = float(info["center"])
    radius = int(info["radius"])

    for step, pos, col in _trajectory(start_pos, dir1, limit):
        if orient == "h":
            if col == G:
                within = _within_radius_indices(out.shape[1], center, radius)
                mask = (out[pos, :] != bg) & within
                out[pos, mask] = G
            for c in span:
                out[pos, int(c)] = int(col)
            gap = start_pos + dir1 * (2 * int(step) + 1)
            if 0 <= gap < out.shape[0]:
                for c in span:
                    out[int(gap), int(c)] = bg
        else:
            if col == G:
                within = _within_radius_indices(out.shape[0], center, radius)
                mask = (out[:, pos] != bg) & within
                out[mask, pos] = G
            for r in span:
                out[int(r), pos] = int(col)
            gap = start_pos + dir1 * (2 * int(step) + 1)
            if 0 <= gap < out.shape[1]:
                for r in span:
                    out[int(r), int(gap)] = bg

    return out


def _rect_is_free(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, bg: int, pad: int = 1) -> bool:
    H, W = canvas.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(H, int(r0) + int(h) + int(pad))
    cc1 = min(W, int(c0) + int(w) + int(pad))
    return not bool((canvas[rr0:rr1, cc0:cc1] != bg).any())


def _paint_rect(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = int(color)


def _place_probe_pair(canvas: np.ndarray, orient: str, pos: int, near_start: int, far_start: int, bg: int, fg: int) -> bool:
    H, W = canvas.shape

    if orient == "h":
        near_r = int(pos) if int(pos) < H - 2 else int(pos) - 1
        far_r = near_r
        if not _rect_is_free(canvas, near_r, int(near_start), 2, 2, bg):
            return False
        if not _rect_is_free(canvas, far_r, int(far_start), 2, 2, bg):
            return False
        _paint_rect(canvas, near_r, int(near_start), 2, 2, fg)
        _paint_rect(canvas, far_r, int(far_start), 2, 2, fg)
        return True

    near_c = int(pos) if int(pos) < W - 2 else int(pos) - 1
    far_c = near_c
    if not _rect_is_free(canvas, int(near_start), near_c, 2, 2, bg):
        return False
    if not _rect_is_free(canvas, int(far_start), far_c, 2, 2, bg):
        return False
    _paint_rect(canvas, int(near_start), near_c, 2, 2, fg)
    _paint_rect(canvas, int(far_start), far_c, 2, 2, fg)
    return True


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 6)

    if edge in ("top", "bottom"):
        min_start = seg_len + 2
        max_start = W - (2 * seg_len) - 3
        if max_start < min_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
    else:
        min_start = seg_len + 2
        max_start = H - (2 * seg_len) - 3
        if max_start < min_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1

    canvas = np.full((H, W), bg, dtype=int)

    if orient == "h":
        for c in span:
            canvas[edge_row, c] = Y
        start_pos = edge_row
        limit = H
    else:
        for r in span:
            canvas[r, edge_col] = Y
        start_pos = edge_col
        limit = W

    center = 0.5 * (span[0] + span[-1])
    radius = seg_len
    traj = _trajectory(start_pos, 1 if start_pos == 0 else -1, limit)
    green_positions = [pos for _step, pos, col in traj if col == G]
    if len(green_positions) < 2:
        return None

    if orient == "h":
        near_start = span[0] - 2
        far_start = int(np.ceil(center + radius)) + 1
        if near_start < 1 or far_start + 1 >= W - 1:
            return None
        probe_lines = [p for p in green_positions if 1 <= p < H - 1]
    else:
        near_start = span[0] - 2
        far_start = int(np.ceil(center + radius)) + 1
        if near_start < 1 or far_start + 1 >= H - 1:
            return None
        probe_lines = [p for p in green_positions if 1 <= p < W - 1]

    if len(probe_lines) < 2:
        return None

    chosen_probe_lines = sorted(rng.sample(probe_lines, k=min(3, len(probe_lines))))
    placed_probe_lines = 0
    for pos in chosen_probe_lines:
        if _place_probe_pair(canvas, orient, int(pos), int(near_start), int(far_start), bg, fg):
            placed_probe_lines += 1
    if placed_probe_lines < 2:
        return None

    forbidden_rows = {edge_row} if orient == "h" else set()
    forbidden_cols = {edge_col} if orient == "v" else set()

    n_blobs = rng.randint(3, 5)
    for _ in range(n_blobs):
        bh = rng.randint(2, 8)
        bw = rng.randint(2, 8)
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
        if pix < 5 or pix > 50:
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

    fg_mask = (canvas != bg) & (canvas != Y)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    near_targets, far_targets = _variant_target_counts(canvas)
    if near_targets < 4 or far_targets < 4:
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
