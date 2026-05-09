# Combo variant of 221dfab4 (index 16)
# combo_spec: D6/1+D7/2
# Name: 内部线段+标记线终止

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "solvers"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for combo D6/1 + D7/2
# a) Concrete behavior / trace:
#    Example: a horizontal yellow segment `44444` sits internally at row 9, cols 8..12.
#    All amorphous foreground blobs lie only below that row, and an aligned magenta marker
#    line of the same span sits farther below at row 21. Original 221dfab4 would never
#    accept row 9 as a source because it only searches border yellows. The D6-only logic
#    fixes that by first finding the internal straight yellow segment, then inferring
#    "downward" from the fact that the colored mass is only below row 9. The D7 addition
#    then changes the painting loop: rows 9,11,13,15,17,19 are drawable stripe rows, but
#    row 21 is a stop marker, so propagation halts before painting row 21 and before
#    clearing any gap that would touch it. Difference from the original task: source
#    identification, direction recovery, and termination all become data-dependent.
# b) Complexity calibration:
#    The generator uses only one internal yellow line and one aligned stop line, with the
#    stop color distinct from blob color. That keeps the puzzle legible instead of turning
#    every straight segment into a candidate. At the same time, the source no longer touches
#    an edge and the stop line leaves a visible unused tail, so both mutations are obvious.
# c) Implicit-assumption audit against the original main():
#    - Border-only source search is invalid: the source is now internal.
#    - Edge-derived direction is invalid: direction comes from one-sided payload density.
#    - "Paint until out of bounds" is invalid: the nearest forward aligned marker halts the
#      loop, and gaps must not erase the marker line.
#    - Treating every non-bg aligned segment as paintable is invalid: the forward marker is
#      preserved, so the loop breaks strictly before `stop_pos`.
# d) Placement legality:
#    The generator keeps the yellow source at least two cells from every border, places the
#    stop line on the forward side with the same span/orientation, reserves the entire
#    passive half-plane empty, and places blobs only in the corridor between source and stop.
#    That guarantees no overlap, no out-of-bounds writes, an unambiguous direction cue, and
#    a unique aligned stop candidate.
# e) Visual observability:
#    Across a few samples, a human can see three stable signals: the yellow source is not on
#    an edge, colored blobs cluster on only one side of it, and an aligned non-yellow line
#    blocks the stripe train before the far edge. The output visibly stops at that marker,
#    leaving a tail region that the D6-only behavior would have painted through.
# New solver assumptions introduced here and enforced by the generator:
#    - There is a unique dominant internal straight yellow segment.
#    - Payload blobs occupy only the active half-plane, so half-plane density reveals the
#      marching direction.
#    - The stop marker is the nearest forward straight segment with the same span/orientation
#      and a color distinct from bg / yellow / green / blob color.
#    - There is enough room before the marker for at least two green stripes, so both the
#      stripe cycle and the stop behavior are visible.

Y = 4
G = 3


def _straight_segment_from_points(pts: np.ndarray) -> dict | None:
    if pts.size == 0:
        return None

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    area = int(pts.shape[0])

    if r0 == r1 and area == (c1 - c0 + 1):
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": list(range(int(c0), int(c1) + 1)),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "length": int(area),
        }

    if c0 == c1 and area == (r1 - r0 + 1):
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": list(range(int(r0), int(r1) + 1)),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "length": int(area),
        }

    return None


def _find_source_segment(g: np.ndarray) -> dict | None:
    H, W = g.shape
    lab, n = label_mask(g == int(Y), connectivity=4)
    best: dict | None = None
    best_score: tuple[int, int, int] | None = None

    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        info = _straight_segment_from_points(pts)
        if info is None:
            continue

        r0, c0, r1, c1 = info["bbox"]
        internal_margin = int(min(r0, c0, H - 1 - r1, W - 1 - c1))
        internal = int(internal_margin >= 1)
        score = (internal, int(info["length"]), int(internal_margin))
        if best is None or score > best_score:
            best = {
                **info,
                "internal": bool(internal),
                "internal_margin": int(internal_margin),
            }
            best_score = score

    return best


def _find_stop_line(
    g: np.ndarray,
    bg: int,
    orient: str,
    span: list[int],
    start_pos: int,
    dir1: int,
) -> tuple[int | None, int | None]:
    candidates: list[tuple[int, int, int]] = []

    for raw_color in np.unique(g):
        color = int(raw_color)
        if color in {int(bg), int(Y), int(G)}:
            continue

        lab, n = label_mask(g == int(color), connectivity=4)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            info = _straight_segment_from_points(pts)
            if info is None:
                continue
            if info["orient"] != orient or info["span"] != span:
                continue

            pos = int(info["start_pos"])
            delta = int(dir1) * (int(pos) - int(start_pos))
            if delta <= 0:
                continue
            candidates.append((int(delta), int(pos), int(color)))

    if not candidates:
        return None, None

    candidates.sort()
    _delta, pos, color = candidates[0]
    return int(pos), int(color)


def _infer_direction(g: np.ndarray, bg: int, source: dict) -> int:
    payload = (g != int(bg)) & (g != int(Y))
    H, W = g.shape
    orient = str(source["orient"])
    start_pos = int(source["start_pos"])
    span = list(source["span"])

    if orient == "h":
        before = int(payload[:start_pos, :].sum())
        after = int(payload[start_pos + 1 :, :].sum())
        if before == after:
            c0 = max(0, int(span[0]) - 2)
            c1 = min(W, int(span[-1]) + 3)
            before = int(payload[:start_pos, c0:c1].sum())
            after = int(payload[start_pos + 1 :, c0:c1].sum())
        if before == after:
            return 1 if (H - 1 - start_pos) >= start_pos else -1
        return 1 if after > before else -1

    before = int(payload[:, :start_pos].sum())
    after = int(payload[:, start_pos + 1 :].sum())
    if before == after:
        r0 = max(0, int(span[0]) - 2)
        r1 = min(H, int(span[-1]) + 3)
        before = int(payload[r0:r1, :start_pos].sum())
        after = int(payload[r0:r1, start_pos + 1 :].sum())
    if before == after:
        return 1 if (W - 1 - start_pos) >= start_pos else -1
    return 1 if after > before else -1


def _apply_pattern(
    grid: np.ndarray,
    bg: int,
    source: dict,
    dir1: int,
    stop_pos: int | None,
) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    orient = str(source["orient"])
    start_pos = int(source["start_pos"])
    span = list(source["span"])
    limit = out.shape[0] if orient == "h" else out.shape[1]
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
            out[pos, span[0] : span[-1] + 1] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            gap_dist = int(dir1) * (int(gap) - int(start_pos))
            if 0 <= gap < out.shape[0] and (stop_dist is None or gap_dist < int(stop_dist)):
                out[gap, span[0] : span[-1] + 1] = int(bg)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            out[span[0] : span[-1] + 1, pos] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            gap_dist = int(dir1) * (int(gap) - int(start_pos))
            if 0 <= gap < out.shape[1] and (stop_dist is None or gap_dist < int(stop_dist)):
                out[span[0] : span[-1] + 1, gap] = int(bg)
        step += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    source = _find_source_segment(g)
    if source is None or not bool(source["internal"]):
        return g.copy()

    dir1 = _infer_direction(g, bg, source)
    stop_pos, _stop_color = _find_stop_line(
        g,
        bg,
        str(source["orient"]),
        list(source["span"]),
        int(source["start_pos"]),
        int(dir1),
    )
    return _apply_pattern(g, bg, source, int(dir1), stop_pos)


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return None
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _paste(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    sh, sw = sprite.shape
    for dr in range(sh):
        for dc in range(sw):
            if int(sprite[dr, dc]) != int(bg):
                canvas[int(top) + dr, int(left) + dc] = int(sprite[dr, dc])


def _place_segment(grid: np.ndarray, orient: str, pos: int, span: list[int], color: int) -> None:
    if orient == "h":
        grid[int(pos), span[0] : span[-1] + 1] = int(color)
    else:
        grid[span[0] : span[-1] + 1, int(pos)] = int(color)


def _stop_mask(shape: tuple[int, int], orient: str, stop_pos: int | None, span: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if stop_pos is None:
        return mask
    if orient == "h":
        mask[int(stop_pos), span[0] : span[-1] + 1] = True
    else:
        mask[span[0] : span[-1] + 1, int(stop_pos)] = True
    return mask


def _stripe_positions(start_pos: int, dir1: int, limit: int, stop_pos: int | None) -> list[int]:
    out: list[int] = []
    stop_dist = None if stop_pos is None else int(dir1) * (int(stop_pos) - int(start_pos))
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        pos_dist = int(dir1) * (int(pos) - int(start_pos))
        if stop_dist is not None and pos_dist >= int(stop_dist):
            break
        out.append(int(pos))
        step += 1
    return out


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    source = _find_source_segment(g)
    if source is None:
        return {
            "internal_margin": -1,
            "stop_delta": -1,
            "tail_after_stop": -1,
            "payload_active_mass": 0,
            "payload_inactive_mass": 0,
            "green_hits": 0,
            "truncated_pixels": 0,
        }

    dir1 = _infer_direction(g, bg, source)
    stop_pos, _stop_color = _find_stop_line(
        g,
        bg,
        str(source["orient"]),
        list(source["span"]),
        int(source["start_pos"]),
        int(dir1),
    )
    stop_mask = _stop_mask(g.shape, str(source["orient"]), stop_pos, list(source["span"]))
    payload = (g != int(bg)) & (g != int(Y)) & (~stop_mask)

    start_pos = int(source["start_pos"])
    orient = str(source["orient"])
    span = list(source["span"])

    if orient == "h":
        before = int(payload[:start_pos, :].sum())
        after = int(payload[start_pos + 1 :, :].sum())
        limit = g.shape[0]
    else:
        before = int(payload[:, :start_pos].sum())
        after = int(payload[:, start_pos + 1 :].sum())
        limit = g.shape[1]

    payload_active_mass = int(after if dir1 == 1 else before)
    payload_inactive_mass = int(before if dir1 == 1 else after)

    green_hits = 0
    for idx, pos in enumerate(_stripe_positions(start_pos, dir1, limit, stop_pos)):
        if idx % 3 != 2:
            continue
        if orient == "h":
            if bool(np.any(payload[int(pos), :])):
                green_hits += 1
        else:
            if bool(np.any(payload[:, int(pos)])):
                green_hits += 1

    no_stop_out = _apply_pattern(g, bg, source, int(dir1), None)
    with_stop_out = _apply_pattern(g, bg, source, int(dir1), stop_pos)
    truncated_pixels = int(np.sum(no_stop_out != with_stop_out))

    stop_delta = -1
    tail_after_stop = -1
    if stop_pos is not None:
        stop_delta = int(dir1) * (int(stop_pos) - int(start_pos))
        tail_after_stop = int(limit - 1 - stop_pos) if dir1 == 1 else int(stop_pos)

    return {
        "internal_margin": int(source["internal_margin"]),
        "stop_delta": int(stop_delta),
        "tail_after_stop": int(tail_after_stop),
        "payload_active_mass": int(payload_active_mass),
        "payload_inactive_mass": int(payload_inactive_mass),
        "green_hits": int(green_hits),
        "truncated_pixels": int(truncated_pixels),
    }


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (int(Y), int(G))]
    bg, fg, stop_color = rng.sample(available, 3)

    orient = rng.choice(["h", "v"])
    dir1 = rng.choice([-1, 1])
    seg_len = rng.randint(3, 8)

    side_margin = 2
    min_src_margin = 3
    min_stop_step = 6
    min_tail_steps = 2

    if orient == "h":
        col_lo = side_margin
        col_hi = W - seg_len - side_margin
        if col_lo > col_hi:
            return None

        if dir1 == 1:
            row_lo = min_src_margin
            row_hi = H - 1 - 2 * (min_stop_step + min_tail_steps)
        else:
            row_lo = 2 * (min_stop_step + min_tail_steps)
            row_hi = H - 1 - min_src_margin
        if row_lo > row_hi:
            return None

        src_pos = rng.randint(row_lo, row_hi)
        seg_start = rng.randint(col_lo, col_hi)
        span = list(range(seg_start, seg_start + seg_len))
        limit = H
    else:
        row_lo = side_margin
        row_hi = H - seg_len - side_margin
        if row_lo > row_hi:
            return None

        if dir1 == 1:
            col_lo = min_src_margin
            col_hi = W - 1 - 2 * (min_stop_step + min_tail_steps)
        else:
            col_lo = 2 * (min_stop_step + min_tail_steps)
            col_hi = W - 1 - min_src_margin
        if col_lo > col_hi:
            return None

        src_pos = rng.randint(col_lo, col_hi)
        seg_start = rng.randint(row_lo, row_hi)
        span = list(range(seg_start, seg_start + seg_len))
        limit = W

    max_step = ((limit - 1 - src_pos) // 2) if dir1 == 1 else (src_pos // 2)
    if max_step < min_stop_step + min_tail_steps:
        return None

    stop_step = rng.randint(min_stop_step, max_step - min_tail_steps)
    stop_pos = int(src_pos) + int(dir1) * (2 * int(stop_step))

    grid = np.full((H, W), int(bg), dtype=int)
    _place_segment(grid, orient, int(src_pos), span, int(Y))
    _place_segment(grid, orient, int(stop_pos), span, int(stop_color))

    if orient == "h":
        target_r0 = int(src_pos) + 1 if dir1 == 1 else int(stop_pos) + 1
        target_r1 = int(stop_pos) - 1 if dir1 == 1 else int(src_pos) - 1
        target_c0 = 1
        target_c1 = W - 2
    else:
        target_r0 = 1
        target_r1 = H - 2
        target_c0 = int(src_pos) + 1 if dir1 == 1 else int(stop_pos) + 1
        target_c1 = int(stop_pos) - 1 if dir1 == 1 else int(src_pos) - 1

    if target_r0 > target_r1 or target_c0 > target_c1:
        return None

    canvas = grid.copy()
    region = canvas[target_r0 : target_r1 + 1, target_c0 : target_c1 + 1]
    if region.shape[0] < 2 or region.shape[1] < 2:
        return None

    target_blobs = rng.randint(5, 8)
    placed = 0
    for _ in range(target_blobs * 10):
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
                color_palette=[int(fg)],
                connectivity=4,
                background=int(bg),
            )
        except (AssertionError, RecursionError, ValueError):
            continue

        sprite = _trim_sprite(sprite, int(bg))
        if sprite is None:
            continue
        sh, sw = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue
        if sh > region.shape[0] or sw > region.shape[1]:
            continue

        try:
            rr, cc = random_free_location(region, sprite, transparent=int(bg), border=0, padding=1)
        except ValueError:
            continue

        _paste(canvas, sprite, int(target_r0 + rr), int(target_c0 + cc), int(bg))
        placed += 1

    fg_mask = canvas == int(fg)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if int(get_bg(canvas)) != int(bg):
        return None

    source = _find_source_segment(canvas)
    if source is None or not bool(source["internal"]):
        return None

    recovered_dir = _infer_direction(canvas, int(bg), source)
    recovered_stop_pos, recovered_stop_color = _find_stop_line(
        canvas,
        int(bg),
        str(source["orient"]),
        list(source["span"]),
        int(source["start_pos"]),
        int(recovered_dir),
    )
    if int(recovered_dir) != int(dir1):
        return None
    if recovered_stop_pos != int(stop_pos) or recovered_stop_color != int(stop_color):
        return None

    metrics = _variant_metrics(canvas)
    if metrics["internal_margin"] < 2:
        return None
    if metrics["stop_delta"] < 12:
        return None
    if metrics["tail_after_stop"] < 4:
        return None
    if metrics["payload_active_mass"] <= 0 or metrics["payload_inactive_mass"] != 0:
        return None
    if metrics["green_hits"] < 2:
        return None
    if metrics["truncated_pixels"] <= 0:
        return None

    out = main(canvas)
    if np.array_equal(out, canvas):
        return None

    if orient == "h":
        if not np.all(canvas[int(stop_pos), span] == int(stop_color)):
            return None
        if not np.all(out[int(stop_pos), span] == int(stop_color)):
            return None
    else:
        if not np.all(canvas[span, int(stop_pos)] == int(stop_color)):
            return None
        if not np.all(out[span, int(stop_pos)] == int(stop_color)):
            return None

    if not np.array_equal(main(canvas), out):
        return None

    return canvas


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
