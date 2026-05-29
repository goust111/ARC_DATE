# Variant of 221dfab4
# Dimension: D7 — 终止条件 (dst)
# Change: 推进至对侧边缘——条纹一直画到网格的另一边... → 碰到另一个彩色标记线停止

from __future__ import annotations

# Design analysis for the D7 variant:
# a) Concrete behavior:
#    The input still has one yellow border segment that defines the stripe span and travel direction.
#    A second straight segment of a different color is added farther along the same trajectory, with
#    the same orientation and the same span projection. Example: if the yellow segment is on the top
#    border at columns 5..9, the stop marker is another horizontal segment at some lower row, again
#    covering columns 5..9. Original logic would paint rows 0,2,4,6,8,... until the far edge. New
#    logic paints rows 0,2,4,... only while they are strictly before the stop row; if the next stripe
#    or gap would land on the stop row, propagation halts and the marker line is preserved.
# b) Complexity calibration:
#    The marker is internal and leaves visible unused space behind it, so the variant is observable.
#    The marker uses a dedicated color distinct from bg / yellow / green / blob color, avoiding
#    confusion with recolored foreground objects.
# c) Implicit-assumption audit:
#    - Original `while` loop assumes "every in-bounds stripe position is drawable"; this no longer
#      holds, so the loop must check a stop coordinate before painting.
#    - Original gap clearing assumes the next odd-offset cell can always be reset to bg; this becomes
#      wrong when that cell is the stop marker, so gaps are only cleared while they are strictly
#      before the stop.
#    - Original solver only needs the yellow source segment. The variant requires locating another
#      straight component that is aligned with the yellow span; blob components must therefore be
#      filtered out by straightness, exact span match, and forward-direction position.
# d) Placement legality:
#    The generator places the stop marker before blobs, keeps it disjoint from the yellow segment,
#    and forbids blobs from occupying the start row/column or stop row/column. This guarantees the
#    stop marker remains a clean standalone segment that the solver can detect unambiguously.
# e) Visual observability:
#    The stop marker is always internal and at least two stripe steps before the far edge. That leaves
#    a visible tail region where the original task would have continued drawing but the variant does not.
# New solver assumptions introduced by D7:
#    - There is exactly one dominant yellow border segment.
#    - There is at least one non-bg, non-yellow, non-green straight marker aligned with the source span
#      and lying ahead in the travel direction; the nearest such marker is the stop line.
#    - The generator enforces those properties directly.

import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path(__file__).resolve().parents[5] / "arc2_solver_select"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


def _extract_source_segment(g: np.ndarray, yellow: int) -> dict | None:
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
        cols = sorted(int(c) for c in cs)
        span = list(range(int(cols[0]), int(cols[-1]) + 1))
        if cols != span:
            return None
        start_pos = int(r0)
        return {
            "orient": "h",
            "start_pos": start_pos,
            "span": span,
            "dir1": 1 if start_pos == 0 else -1,
            "limit": H,
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        span = list(range(int(rows[0]), int(rows[-1]) + 1))
        if rows != span:
            return None
        start_pos = int(c0)
        return {
            "orient": "v",
            "start_pos": start_pos,
            "span": span,
            "dir1": 1 if start_pos == 0 else -1,
            "limit": W,
        }

    return None


def _find_stop_line(
    g: np.ndarray,
    bg: int,
    orient: str,
    span: list[int],
    start_pos: int,
    dir1: int,
    skip_colors: set[int],
) -> int | None:
    candidates: list[tuple[int, int]] = []

    for raw_color in np.unique(g):
        color = int(raw_color)
        if color in skip_colors:
            continue
        mask = (g == int(color))
        if not bool(mask.any()):
            continue
        lab, n = label_mask(mask, connectivity=4)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(pts.shape[0]) != len(span):
                continue
            rs = pts[:, 0]
            cs = pts[:, 1]

            if orient == "h":
                if int(rs.min()) != int(rs.max()):
                    continue
                cols = sorted(int(c) for c in cs)
                if cols != span:
                    continue
                pos = int(rs[0])
            else:
                if int(cs.min()) != int(cs.max()):
                    continue
                rows = sorted(int(r) for r in rs)
                if rows != span:
                    continue
                pos = int(cs[0])

            delta = int(dir1) * (int(pos) - int(start_pos))
            if delta <= 0:
                continue
            candidates.append((delta, int(pos)))

    if not candidates:
        return None

    candidates.sort()
    return int(candidates[0][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the Y,Y,G stripe cycle from a yellow border segment, but stop before an aligned marker line.

    The green stripe behavior is unchanged: on green steps, all non-bg pixels across the full row/column
    are recolored to green. The only D7 change is that painting halts when the marching front reaches the
    aligned stop marker instead of continuing to the opposite edge.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    Y = 4
    G = 3

    src = _extract_source_segment(g, Y)
    if src is None:
        return out

    orient = str(src["orient"])
    start_pos = int(src["start_pos"])
    span = list(src["span"])
    dir1 = int(src["dir1"])
    limit = int(src["limit"])

    stop_pos = _find_stop_line(
        g,
        bg,
        orient,
        span,
        start_pos,
        dir1,
        skip_colors={int(bg), int(Y), int(G)},
    )

    cycle = [int(Y), int(Y), int(G)]
    stop_dist = None if stop_pos is None else int(dir1) * (int(stop_pos) - int(start_pos))

    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        pos_dist = int(dir1) * (int(pos) - int(start_pos))
        if stop_dist is not None and pos_dist >= int(stop_dist):
            break

        col = int(cycle[int(step) % 3])

        if orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in span:
                out[pos, int(c)] = col

            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            gap_dist = int(dir1) * (int(gap) - int(start_pos))
            if 0 <= gap < H and (stop_dist is None or gap_dist < int(stop_dist)):
                for c in span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            for r in span:
                out[int(r), pos] = col

            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            gap_dist = int(dir1) * (int(gap) - int(start_pos))
            if 0 <= gap < W and (stop_dist is None or gap_dist < int(stop_dist)):
                for r in span:
                    out[int(r), int(gap)] = int(bg)

        step += 1

    return out


# --------------- generator ---------------
import random


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = (sprite != int(bg))
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not bool(rows_any.any()):
        return None
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _place_segment(grid: np.ndarray, orient: str, pos: int, span: list[int], color: int) -> None:
    if orient == "h":
        for c in span:
            grid[int(pos), int(c)] = int(color)
    else:
        for r in span:
            grid[int(r), int(pos)] = int(color)


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (3, 4)]
    bg, fg, stop_color = rng.sample(available, 3)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        start_pos = 0 if edge == "top" else H - 1
        dir1 = 1 if edge == "top" else -1
        limit = H
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        start_pos = 0 if edge == "left" else W - 1
        dir1 = 1 if edge == "left" else -1
        limit = W

    max_step = ((limit - 1 - start_pos) // 2) if dir1 == 1 else (start_pos // 2)
    if max_step < 5:
        return None

    # Internal stop only: this keeps the D7 change visible and prevents degeneration back to the original.
    stop_step = rng.randint(3, max_step - 2)
    stop_pos = int(start_pos) + int(dir1) * (2 * int(stop_step))

    grid = np.full((H, W), bg, dtype=int)
    _place_segment(grid, orient, start_pos, span, 4)
    _place_segment(grid, orient, stop_pos, span, stop_color)

    if orient == "h":
        forbidden_rows = {int(start_pos), int(stop_pos)}
        forbidden_cols: set[int] = set()
    else:
        forbidden_rows = set()
        forbidden_cols = {int(start_pos), int(stop_pos)}

    canvas = grid.copy()
    target_blobs = rng.randint(5, 9)
    placed = 0

    for _ in range(target_blobs * 8):
        if placed >= target_blobs:
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

        sprite = _trim_sprite(sprite, bg)
        if sprite is None:
            continue

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        sprite_mask = (sprite != int(bg))
        blob_rows = {int(pr + dr) for dr in range(sh) if bool(sprite_mask[dr].any())}
        blob_cols = {int(pc + dc) for dc in range(sw) if bool(sprite_mask[:, dc].any())}

        if blob_rows & forbidden_rows:
            continue
        if blob_cols & forbidden_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != int(bg):
                    canvas[int(pr + dr), int(pc + dc)] = int(sprite[dr, dc])
        placed += 1

    fg_mask = (canvas == int(fg))
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if get_bg(inp) != int(bg):
        return None

    # Basic structural checks for the D7 behavior.
    if orient == "h":
        if not np.all(inp[int(stop_pos), span] == int(stop_color)):
            return None
        if not np.all(out[int(stop_pos), span] == int(stop_color)):
            return None
    else:
        if not np.all(inp[span, int(stop_pos)] == int(stop_color)):
            return None
        if not np.all(out[span, int(stop_pos)] == int(stop_color)):
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
