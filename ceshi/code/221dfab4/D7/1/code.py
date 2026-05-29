# Variant of 221dfab4
# Dimension: D7 — 终止条件 (dst)
# Change: 推进至对侧边缘——条纹一直画到网格的另一边 → 距离由线段长度编码

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
UTILS_DIR = ROOT / "arc2_solver_select"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the D7 variant:
# a) Concrete behavior:
#    The yellow border segment still launches stripes orthogonally inward, with stripe colors
#    cycling Y, Y, G and 1-cell gaps reset to background. The only changed dimension is the stop rule:
#    the stripe train now occupies at most `segment_length * 3` cells of travel from the border segment,
#    instead of always reaching the opposite side.
#    Small trace: if a top-edge segment has length 4, the encoded distance is 12. Original logic paints
#    rows 0,2,4,... until the last in-bounds row; the variant paints rows 0,2,4,6,8,10,12 and then stops,
#    only clearing gap rows 1,3,5,7,9,11. Rows below 12 stay as they were.
# b) Complexity calibration:
#    The factor 3 produces 5-11 visible stripes for segment lengths 3-7, so the stop is large enough
#    to see and compare across examples, but still leaves untouched space in the grid. Generator samples
#    are forced to have spare room beyond the encoded frontier, so the task does not collapse back to the
#    original "paint to the far edge" behavior.
# c) Hidden assumptions in the original solver that matter here:
#    - The yellow segment must be a single straight 4-connected border component; otherwise min/max span
#      expansion would create fake coverage. Generator keeps exactly one contiguous segment.
#    - The old unconditional gap clearing was harmless because stripes always continued to the edge.
#      In this variant that would erase cells beyond the intended stop, so gap cells are now only cleared
#      while their distance is still within the encoded limit.
#    - The loop previously terminated only by falling off-grid. That no longer matches the rule, so the
#      solver now stops when `2 * step > encoded_distance`, even if more grid remains.
# d) Placement legality:
#    Blobs are still placed with border and padding, so they do not overlap each other or the segment.
#    The grid dimension along the propagation axis is chosen so `segment_length * 3 + 4 <= axis_length`,
#    guaranteeing at least one additional stripe would exist in the legacy task and making the variant
#    distinguishable. Samples are rejected unless some foreground lies on a reachable green stripe and
#    some foreground also lies beyond the stopping frontier.
# e) Visual observability:
#    Across examples, longer yellow segments yield deeper stripe trains. Because untouched space remains
#    after the frontier, a human can see that the stripe depth is not tied to the grid size. The generator
#    also rejects any sample whose output matches the legacy "paint to edge" solver.

YELLOW = 4
GREEN = 3
DIST_FACTOR = 3


def _detect_segment(g: np.ndarray) -> dict[str, object] | None:
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == YELLOW) & border
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
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": list(range(int(cols[0]), int(cols[-1]) + 1)),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": H,
            "seg_len": int(cols[-1] - cols[0] + 1),
        }
    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": list(range(int(rows[0]), int(rows[-1]) + 1)),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
            "seg_len": int(rows[-1] - rows[0] + 1),
        }
    return None


def _paint_pattern(g: np.ndarray, *, max_dist: int | None, info: dict[str, object] | None = None) -> np.ndarray:
    out = np.array(g, dtype=int).copy()
    bg = int(get_bg(out))
    info = _detect_segment(out) if info is None else info
    if info is None:
        return out

    orient = str(info["orient"])
    start_pos = int(info["start_pos"])
    span = list(info["span"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])

    cycle = [YELLOW, YELLOW, GREEN]
    step = 0

    while True:
        dist = 2 * step
        if max_dist is not None and int(dist) > int(max_dist):
            break

        pos = int(start_pos) + int(dir1) * int(dist)
        if not (0 <= pos < limit):
            break

        col = int(cycle[step % 3])

        if orient == "h":
            if col == GREEN:
                out[pos, :] = np.where(out[pos, :] != bg, GREEN, out[pos, :])
            for c in span:
                out[pos, int(c)] = col
            gap_dist = int(dist + 1)
            if max_dist is None or gap_dist <= int(max_dist):
                gap = int(start_pos) + int(dir1) * gap_dist
                if 0 <= gap < out.shape[0]:
                    for c in span:
                        out[int(gap), int(c)] = bg
        else:
            if col == GREEN:
                out[:, pos] = np.where(out[:, pos] != bg, GREEN, out[:, pos])
            for r in span:
                out[int(r), pos] = col
            gap_dist = int(dist + 1)
            if max_dist is None or gap_dist <= int(max_dist):
                gap = int(start_pos) + int(dir1) * gap_dist
                if 0 <= gap < out.shape[1]:
                    for r in span:
                        out[int(r), int(gap)] = bg

        step += 1

    return out


def _legacy_main_to_edge(grid: np.ndarray) -> np.ndarray:
    return _paint_pattern(np.array(grid, dtype=int), max_dist=None)


def _green_positions(info: dict[str, object], max_dist: int) -> list[int]:
    start_pos = int(info["start_pos"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])

    positions: list[int] = []
    step = 0
    while True:
        dist = 2 * step
        if int(dist) > int(max_dist):
            break
        pos = int(start_pos) + int(dir1) * int(dist)
        if not (0 <= pos < limit):
            break
        if step % 3 == 2:
            positions.append(int(pos))
        step += 1
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    info = _detect_segment(g)
    if info is None:
        return g.copy()
    max_dist = int(info["seg_len"]) * DIST_FACTOR
    return _paint_pattern(g, max_dist=max_dist, info=info)


def _in_untouched_region(info: dict[str, object], stop_dist: int, fg_mask: np.ndarray) -> bool:
    orient = str(info["orient"])
    start_pos = int(info["start_pos"])
    dir1 = int(info["dir1"])
    frontier = int(start_pos) + int(dir1) * int(stop_dist)

    if orient == "h":
        if dir1 == 1:
            return bool(np.any(fg_mask[int(frontier) + 1 :, :]))
        return bool(np.any(fg_mask[: int(frontier), :]))
    if dir1 == 1:
        return bool(np.any(fg_mask[:, int(frontier) + 1 :]))
    return bool(np.any(fg_mask[:, : int(frontier)]))


def _green_hit_count(info: dict[str, object], stop_dist: int, fg_mask: np.ndarray) -> int:
    positions = _green_positions(info, stop_dist)
    if str(info["orient"]) == "h":
        return sum(bool(np.any(fg_mask[int(r), :])) for r in positions)
    return sum(bool(np.any(fg_mask[:, int(c)])) for c in positions)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(26, 30)
    W = rng.randint(26, 30)

    available = [c for c in range(10) if c not in (GREEN, YELLOW)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    axis_len = (H - 1) if edge in ("top", "bottom") else (W - 1)
    max_seg_len = min(7, (axis_len - 4) // DIST_FACTOR)
    if max_seg_len < 3:
        return None
    seg_len = rng.randint(3, max_seg_len)
    stop_dist = seg_len * DIST_FACTOR

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
            grid[edge_row, c] = YELLOW
    else:
        for r in span:
            grid[r, edge_col] = YELLOW

    canvas = grid.copy()
    target_blobs = rng.randint(6, 10)
    placed_blobs = 0

    for _ in range(target_blobs * 5):
        if placed_blobs >= target_blobs:
            break

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

        if orient == "h" and edge_row in blob_rows:
            continue
        if orient == "v" and edge_col in blob_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]
        placed_blobs += 1

    fg_mask = (canvas != bg) & (canvas != YELLOW)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    info = _detect_segment(canvas)
    if info is None:
        return None
    if int(info["seg_len"]) != int(seg_len):
        return None
    if int(info["seg_len"]) * DIST_FACTOR != int(stop_dist):
        return None

    if _green_hit_count(info, stop_dist, fg_mask) < 1:
        return None
    if not _in_untouched_region(info, stop_dist, fg_mask):
        return None

    inp = canvas.copy()
    out = main(inp)
    legacy_out = _legacy_main_to_edge(inp)

    if get_bg(inp) != bg:
        return None
    if np.array_equal(out, legacy_out):
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
