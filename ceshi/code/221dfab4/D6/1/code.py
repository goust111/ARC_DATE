# Variant of 221dfab4
# Dimension: D6 — 源对象类型 (src)
# Change: 位于网格边缘的短黄色(4)线段，长度3-8格，距角至少1格... → 内部线段（非边缘）

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


# Design analysis for the D6 variant
# a) Concrete behavior / trace:
#    The original task has a unique yellow segment on the border, so the solver gets
#    the marching direction "for free" from the touched edge. Here the yellow segment
#    is fully internal: e.g. a horizontal `4444` may sit on row 8, cols 10..13.
#    The generator places all non-yellow foreground blobs only on one side of that row.
#    If the blobs are below the segment, the solver must infer "march downward":
#      original logic: row 8 would be impossible because `dir1 = +1 if row == 0 else -1`
#      variant logic: compare foreground mass above vs below row 8, choose the denser side
#      difference: source detection no longer uses a border mask, and direction is a
#      data-derived property instead of an edge-derived property.
# b) Complexity calibration:
#    Keeping the source unique and yellow preserves the original task identity, but
#    moving it away from the edge makes source recognition materially harder because
#    the solver must first isolate a straight internal yellow segment, then infer its
#    push direction from global structure. The generator enforces a strong one-sided
#    density cue so the new behavior is visible from a few examples instead of being
#    buried in noise.
# c) Original implicit assumptions that break under D6:
#    - `smask = (g == Y) & border`: assumes the source touches the border. Now false.
#    - `dir1 = 1 if start_pos == 0 else -1`: assumes orientation alone determines
#      direction because the segment is on one of the two opposite edges. Now false.
#    - generator's `edge`, `edge_row`, `edge_col`, `forbidden_rows/cols`: all assume
#      a border source and only protect that border line. With an internal source we
#      must instead reserve one half-plane for foreground and leave the other empty.
#    The stripe painting loop itself remains valid once `(orient, start_pos, span, dir1)`
#    are recovered correctly.
# d) Placement legality:
#    The generator keeps the segment at least two cells away from every border, keeps
#    all blobs inside the chosen target half-plane, and uses padded free placement to
#    avoid overlap. This guarantees the solver sees a unique straight internal source,
#    unambiguous one-sided mass, and valid non-overlapping objects.
# e) Visual observability:
#    A human can now see that the yellow segment is not attached to any border and that
#    only one side of it contains objects. In the output, stripes extend through that
#    occupied side only, so the D6 change is directly visible rather than hidden inside
#    a border-aligned example that accidentally behaves like the original task.

Y = 4
G = 3


def _find_source_segment(g: np.ndarray) -> dict | None:
    """Return the best straight yellow segment, preferring internal ones."""
    H, W = g.shape
    lab, n = label_mask(g == int(Y), connectivity=4)
    best: dict | None = None
    best_score: tuple[int, int] | None = None

    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue

        rs = pts[:, 0]
        cs = pts[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        area = int(len(pts))

        if r0 == r1 and area == (c1 - c0 + 1):
            orient = "h"
            start_pos = r0
            span = list(range(c0, c1 + 1))
        elif c0 == c1 and area == (r1 - r0 + 1):
            orient = "v"
            start_pos = c0
            span = list(range(r0, r1 + 1))
        else:
            continue

        internal = (r0 > 0) and (r1 < H - 1) and (c0 > 0) and (c1 < W - 1)
        score = (1 if internal else 0, area)
        if best is None or score > best_score:
            best = {
                "orient": orient,
                "start_pos": int(start_pos),
                "span": span,
                "bbox": (r0, c0, r1, c1),
                "internal": bool(internal),
            }
            best_score = score

    return best


def _infer_direction(g: np.ndarray, bg: int, source: dict) -> int:
    """Infer the marching direction from one-sided foreground density."""
    fg = (g != int(bg)) & (g != int(Y))
    H, W = g.shape
    orient = source["orient"]
    start_pos = int(source["start_pos"])
    span = source["span"]

    if orient == "h":
        before = int(fg[:start_pos, :].sum())
        after = int(fg[start_pos + 1 :, :].sum())
        if before == after:
            c0 = max(0, int(span[0]) - 2)
            c1 = min(W, int(span[-1]) + 3)
            before = int(fg[:start_pos, c0:c1].sum())
            after = int(fg[start_pos + 1 :, c0:c1].sum())
        if before == after:
            return 1 if (H - 1 - start_pos) >= start_pos else -1
        return 1 if after > before else -1

    before = int(fg[:, :start_pos].sum())
    after = int(fg[:, start_pos + 1 :].sum())
    if before == after:
        r0 = max(0, int(span[0]) - 2)
        r1 = min(H, int(span[-1]) + 3)
        before = int(fg[r0:r1, :start_pos].sum())
        after = int(fg[r0:r1, start_pos + 1 :].sum())
    if before == after:
        return 1 if (W - 1 - start_pos) >= start_pos else -1
    return 1 if after > before else -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    D6 variant of 221dfab4.

    Same stripe rule as the original task:
    - march every 2 cells with color cycle Y, Y, G
    - yellow paints only along the source span
    - green recolors the whole row/column's existing foreground
    - the 1-cell gap between stripes is reset to bg inside the span

    Changed source logic:
    - the yellow source segment is internal, not on the border
    - the marching direction is inferred from which side of the segment contains the
      foreground blobs, not from which border the segment touches
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    source = _find_source_segment(g)
    if source is None:
        return out

    orient = source["orient"]
    start_pos = int(source["start_pos"])
    span = source["span"]

    dir1 = _infer_direction(g, bg, source)
    limit = H if orient == "h" else W
    cycle = [int(Y), int(Y), int(G)]

    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        col = int(cycle[int(step) % 3])

        if orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            out[pos, span[0] : span[-1] + 1] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < H:
                out[gap, span[0] : span[-1] + 1] = int(bg)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            out[span[0] : span[-1] + 1, pos] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < W:
                out[span[0] : span[-1] + 1, gap] = int(bg)

        step += 1

    return out


def _stripe_positions(start_pos: int, dir1: int, limit: int) -> list[int]:
    pos_list: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        pos_list.append(int(pos))
        step += 1
    return pos_list


def _green_hit_count(grid: np.ndarray, orient: str, start_pos: int, dir1: int, bg: int) -> int:
    g = np.array(grid, dtype=int)
    positions = _stripe_positions(start_pos, dir1, g.shape[0] if orient == "h" else g.shape[1])
    green_positions = [positions[idx] for idx in range(len(positions)) if idx % 3 == 2]
    hits = 0
    for pos in green_positions:
        if orient == "h":
            if bool(np.any((g[pos, :] != int(bg)) & (g[pos, :] != int(Y)))):
                hits += 1
        else:
            if bool(np.any((g[:, pos] != int(bg)) & (g[:, pos] != int(Y)))):
                hits += 1
    return int(hits)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    source = _find_source_segment(g)
    if source is None:
        return {
            "internal_margin": -1,
            "active_mass": 0,
            "inactive_mass": 0,
            "green_hits": 0,
        }

    H, W = g.shape
    r0, c0, r1, c1 = source["bbox"]
    internal_margin = int(min(r0, c0, H - 1 - r1, W - 1 - c1))
    dir1 = _infer_direction(g, bg, source)
    fg = (g != int(bg)) & (g != int(Y))

    if source["orient"] == "h":
        before = int(fg[: source["start_pos"], :].sum())
        after = int(fg[source["start_pos"] + 1 :, :].sum())
    else:
        before = int(fg[:, : source["start_pos"]].sum())
        after = int(fg[:, source["start_pos"] + 1 :].sum())

    active_mass = after if dir1 == 1 else before
    inactive_mass = before if dir1 == 1 else after
    green_hits = _green_hit_count(g, source["orient"], source["start_pos"], dir1, bg)
    return {
        "internal_margin": int(internal_margin),
        "active_mass": int(active_mass),
        "inactive_mass": int(inactive_mass),
        "green_hits": int(green_hits),
    }


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return None
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[r0 : r1 + 1, c0 : c1 + 1]


def _paste(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    sh, sw = sprite.shape
    for dr in range(sh):
        for dc in range(sw):
            if int(sprite[dr, dc]) != int(bg):
                canvas[int(top) + dr, int(left) + dc] = int(sprite[dr, dc])


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (3, 4)]
    bg, fg = rng.sample(available, 2)

    orient = rng.choice(["h", "v"])
    dir1 = rng.choice([-1, 1])
    seg_len = rng.randint(3, 8)

    min_passive = 3
    min_active = 10
    side_margin = 2

    if orient == "h":
        if dir1 == 1:
            row_lo = min_passive
            row_hi = H - 1 - min_active
        else:
            row_lo = min_active
            row_hi = H - 1 - min_passive
        col_lo = side_margin
        col_hi = W - seg_len - side_margin
        if row_lo > row_hi or col_lo > col_hi:
            return None
        src_row = rng.randint(row_lo, row_hi)
        seg_start = rng.randint(col_lo, col_hi)
        span = list(range(seg_start, seg_start + seg_len))
        target_r0 = src_row + 1 if dir1 == 1 else 1
        target_r1 = H - 2 if dir1 == 1 else src_row - 1
        target_c0 = 1
        target_c1 = W - 2
    else:
        if dir1 == 1:
            col_lo = min_passive
            col_hi = W - 1 - min_active
        else:
            col_lo = min_active
            col_hi = W - 1 - min_passive
        row_lo = side_margin
        row_hi = H - seg_len - side_margin
        if row_lo > row_hi or col_lo > col_hi:
            return None
        src_col = rng.randint(col_lo, col_hi)
        seg_start = rng.randint(row_lo, row_hi)
        span = list(range(seg_start, seg_start + seg_len))
        target_r0 = 1
        target_r1 = H - 2
        target_c0 = src_col + 1 if dir1 == 1 else 1
        target_c1 = W - 2 if dir1 == 1 else src_col - 1

    if target_r0 > target_r1 or target_c0 > target_c1:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    if orient == "h":
        grid[src_row, span[0] : span[-1] + 1] = int(Y)
        start_pos = int(src_row)
    else:
        grid[span[0] : span[-1] + 1, src_col] = int(Y)
        start_pos = int(src_col)

    canvas = grid.copy()
    region = canvas[target_r0 : target_r1 + 1, target_c0 : target_c1 + 1]

    target_blobs = rng.randint(5, 9)
    placed = 0
    for _ in range(target_blobs * 8):
        if placed >= target_blobs:
            break

        bh = rng.randint(2, min(10, region.shape[0]))
        bw = rng.randint(2, min(10, region.shape[1]))
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
        except (AssertionError, RecursionError, ValueError):
            continue

        sprite = _trim_sprite(sprite, bg)
        if sprite is None:
            continue

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue
        if sh > region.shape[0] or sw > region.shape[1]:
            continue

        try:
            rr, cc = random_free_location(
                region,
                sprite,
                transparent=bg,
                border=0,
                padding=1,
            )
        except ValueError:
            continue

        _paste(canvas, sprite, target_r0 + rr, target_c0 + cc, bg)
        placed += 1

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if int(get_bg(canvas)) != int(bg):
        return None

    metrics = _variant_metrics(canvas)
    if metrics["internal_margin"] < 2:
        return None
    if metrics["active_mass"] <= 0 or metrics["inactive_mass"] != 0:
        return None
    if metrics["green_hits"] < 2:
        return None

    out = main(canvas)
    if np.array_equal(out, canvas):
        return None
    if not np.array_equal(main(canvas), out):
        return None

    return canvas


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
