# Combo variant of 221dfab4 (index 12)
# combo_spec: D4/3+D7/1+D10/1
# Name: 距离约束flood+距离编码终止+递减宽度
#
# Design analysis for this combo variant
# -------------------------------------
# a) Concrete behavior trace:
# - Start from the original 221dfab4 rule: a yellow border segment emits stripes
#   every 2 cells inward, colors cycle Y,Y,G, and the 1-cell gaps are cleared to
#   background inside the painted span.
# - D10/1 narrows the span by 1 on every stripe. With a top segment spanning
#   cols 8..13 (length L=6), the stripe widths become 6,5,4,3,... and the span
#   trims toward the center: rows 0,2,4,6 use cols 8..13, 8..12, 9..12, 9..11.
# - D4/3 changes only the green side effect. On a green stripe, old logic turned
#   every non-background pixel on that row/column green. New logic recolors only
#   non-background pixels whose 1D distance from the original segment center is
#   <= L. In the same example, center=10.5 and radius=6: a foreground pixel at
#   col 6 is recolored, while one at col 20 is not.
# - D7/1 makes the propagation depth depend on L as well. In this combo the
#   encoded frontier is `max_dist = L * 1`, so the solver stops after distance L
#   even if the grid continues and even if the cone could keep shrinking farther.
#   This yields a finite wide-to-narrow cone rather than painting to the far edge.
# - The fusion is therefore: width, flood radius, and total travel are all driven
#   by the same detected segment length. The output is a truncated cone whose only
#   green flood happens on reached green lines and only near the segment center.
#
# b) Complexity calibration:
# - A large distance factor such as x3 would be swallowed by the shrinking-width
#   stop rule: width would hit zero before the encoded frontier mattered. To keep
#   D7 visible in the fused task, the combo uses the smallest stable coefficient,
#   x1, so the frontier cuts the cone before it fully collapses.
# - The generator restricts segment lengths to 6..8. That guarantees:
#   * at least one reached green stripe (step 2),
#   * several distinct widths before the frontier,
#   * an unreached later green stripe (step 5) for a clear D7 witness.
# - Each sample includes three probe patterns:
#   * a near/far pair on the reached green line for D4,
#   * a foreground block on the first unreached green line for D7,
#   * a block on cells that constant-width painting would overwrite but the
#     shrinking interval leaves untouched for D10.
#
# c) Implicit-assumption audit against the original solver:
# - Original assumption: one fixed `span` applies to every stripe and every gap.
#   Broken by D10; the combo recomputes a per-step interval for both stripe paint
#   and gap clearing.
# - Original assumption: green flood may recolor the whole line. Broken by D4; the
#   combo builds an explicit radius mask around the original segment center.
# - Original assumption: the loop only stops at grid boundary. Broken by D7 and
#   D10; the combo stops at the earliest of: out of bounds, encoded distance, or
#   width exhaustion.
# - New combo assumption: the yellow marker is one straight 4-connected border
#   segment. The generator constructs exactly one such component.
# - New combo assumption: there is enough free room beside the segment center to
#   place pixels both inside and outside the green radius on the same reached
#   green line. The generator chooses segment start positions that guarantee this.
#
# d) Placement legality:
# - All probe objects are small rectangles placed with a 1-cell padding check, so
#   they do not overlap each other, the border segment, or later clutter blobs.
# - Extra clutter blobs are also rectangles placed by the same free-space check,
#   and they avoid the key witness rows/columns so they do not erase the intended
#   observability of the combo dimensions.
#
# e) Visual observability:
# - A human sees three independent visual cues in 2-4 examples:
#   * the stripe cone narrows as it advances,
#   * only the foreground near the original segment center flips green on the
#     reached green line while farther foreground on that same line stays unchanged,
#   * later rows/columns that would continue the pattern remain untouched.
# - The generator rejects degenerate cases by comparing the combo output against
#   three ablations: no-radius flood, no encoded stop, and no shrinking width.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask


Y = 4
G = 3
DIST_FACTOR = 1


def _detect_segment(grid: np.ndarray) -> dict[str, object] | None:
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
    if pts.size == 0:
        return None

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
            "seg_len": int(c1 - c0 + 1),
            "center": 0.5 * (int(c0) + int(c1)),
        }
    if c0 == c1:
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span_lo": int(r0),
            "span_hi": int(r1),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
            "seg_len": int(r1 - r0 + 1),
            "center": 0.5 * (int(r0) + int(r1)),
        }
    return None


def _shrunk_interval(span_lo: int, span_hi: int, step: int) -> tuple[int, int] | None:
    width = int(span_hi) - int(span_lo) + 1 - int(step)
    if width <= 0:
        return None

    left_trim = int(step) // 2
    right_trim = int(step) - int(left_trim)
    lo = int(span_lo) + int(left_trim)
    hi = int(span_hi) - int(right_trim)
    if lo > hi:
        return None
    return int(lo), int(hi)


def _within_radius_indices(length: int, center: float, radius: int) -> np.ndarray:
    return np.abs(np.arange(int(length), dtype=float) - float(center)) <= float(radius)


def _stripe_steps(
    info: dict[str, object],
    *,
    max_dist: int | None,
    shrink: bool,
) -> list[dict[str, int | str]]:
    cycle = [int(Y), int(Y), int(G)]
    steps: list[dict[str, int | str]] = []
    step = 0

    while True:
        dist = 2 * int(step)
        if max_dist is not None and int(dist) > int(max_dist):
            break

        pos = int(info["start_pos"]) + int(info["dir1"]) * int(dist)
        if not (0 <= pos < int(info["limit"])):
            break

        if shrink:
            interval = _shrunk_interval(int(info["span_lo"]), int(info["span_hi"]), int(step))
        else:
            interval = (int(info["span_lo"]), int(info["span_hi"]))
        if interval is None:
            break

        lo, hi = interval
        steps.append(
            {
                "orient": str(info["orient"]),
                "step": int(step),
                "dist": int(dist),
                "pos": int(pos),
                "color": int(cycle[int(step) % 3]),
                "span_lo": int(lo),
                "span_hi": int(hi),
                "width": int(hi - lo + 1),
            }
        )
        step += 1

    return steps


def _paint_variant(
    grid: np.ndarray,
    *,
    use_radius: bool,
    max_dist: int | None,
    shrink: bool,
    info: dict[str, object] | None = None,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    info = _detect_segment(g) if info is None else info
    if info is None:
        return out

    radius = int(info["seg_len"])
    center = float(info["center"])
    orient = str(info["orient"])

    if orient == "h":
        radius_mask = _within_radius_indices(out.shape[1], center, radius)
    else:
        radius_mask = _within_radius_indices(out.shape[0], center, radius)

    for stripe in _stripe_steps(info, max_dist=max_dist, shrink=shrink):
        pos = int(stripe["pos"])
        lo = int(stripe["span_lo"])
        hi = int(stripe["span_hi"])
        col = int(stripe["color"])

        if orient == "h":
            if col == int(G):
                if use_radius:
                    mask = (out[pos, :] != int(bg)) & radius_mask
                else:
                    mask = out[pos, :] != int(bg)
                out[pos, mask] = int(G)
            out[pos, lo : hi + 1] = int(col)
            gap_dist = int(stripe["dist"]) + 1
            if max_dist is None or gap_dist <= int(max_dist):
                gap = int(info["start_pos"]) + int(info["dir1"]) * int(gap_dist)
                if 0 <= gap < out.shape[0]:
                    out[int(gap), lo : hi + 1] = int(bg)
        else:
            if col == int(G):
                if use_radius:
                    mask = (out[:, pos] != int(bg)) & radius_mask
                else:
                    mask = out[:, pos] != int(bg)
                out[mask, pos] = int(G)
            out[lo : hi + 1, pos] = int(col)
            gap_dist = int(stripe["dist"]) + 1
            if max_dist is None or gap_dist <= int(max_dist):
                gap = int(info["start_pos"]) + int(info["dir1"]) * int(gap_dist)
                if 0 <= gap < out.shape[1]:
                    out[lo : hi + 1, int(gap)] = int(bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    info = _detect_segment(g)
    if info is None:
        return g.copy()
    max_dist = int(info["seg_len"]) * int(DIST_FACTOR)
    return _paint_variant(g, use_radius=True, max_dist=max_dist, shrink=True, info=info)


def _rect_is_free(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, bg: int, pad: int = 1) -> bool:
    H, W = canvas.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(H, int(r0) + int(h) + int(pad))
    cc1 = min(W, int(c0) + int(w) + int(pad))
    return not bool((canvas[rr0:rr1, cc0:cc1] != int(bg)).any())


def _paint_rect(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = int(color)


def _place_line_probe(
    canvas: np.ndarray,
    *,
    orient: str,
    pos: int,
    dir1: int,
    ortho_start: int,
    h: int,
    w: int,
    bg: int,
    fg: int,
) -> bool:
    if orient == "h":
        r0 = min(int(pos), int(pos) + int(dir1))
        c0 = int(ortho_start)
    else:
        r0 = int(ortho_start)
        c0 = min(int(pos), int(pos) + int(dir1))

    if not _rect_is_free(canvas, int(r0), int(c0), int(h), int(w), int(bg)):
        return False
    _paint_rect(canvas, int(r0), int(c0), int(h), int(w), int(fg))
    return True


def _combo_metrics(inp: np.ndarray, out: np.ndarray | None = None) -> dict[str, int] | None:
    g = np.array(inp, dtype=int)
    out = main(g) if out is None else np.array(out, dtype=int)
    info = _detect_segment(g)
    if info is None:
        return None

    max_dist = int(info["seg_len"]) * int(DIST_FACTOR)
    bg = int(get_bg(g))
    combo_steps = _stripe_steps(info, max_dist=max_dist, shrink=True)
    if not combo_steps:
        return None

    no_radius = _paint_variant(g, use_radius=False, max_dist=max_dist, shrink=True, info=info)
    no_stop = _paint_variant(g, use_radius=True, max_dist=None, shrink=True, info=info)
    no_shrink = _paint_variant(g, use_radius=True, max_dist=max_dist, shrink=False, info=info)

    widths = [int(s["width"]) for s in combo_steps]
    radius_saved = int(np.count_nonzero((g != int(bg)) & (out == g) & (no_radius != g)))
    frontier_saved = int(np.count_nonzero((g != int(bg)) & (out == g) & (no_stop != g)))
    shrink_saved = int(np.count_nonzero((g != int(bg)) & (out == g) & (no_shrink != g)))

    near_recolored = 0
    far_recolored = 0
    far_preserved = 0
    radius = int(info["seg_len"])
    center = float(info["center"])

    first_green = next((s for s in combo_steps if int(s["color"]) == int(G)), None)
    if first_green is not None:
        lo = int(first_green["span_lo"])
        hi = int(first_green["span_hi"])
        if str(info["orient"]) == "h":
            pos = int(first_green["pos"])
            within = _within_radius_indices(g.shape[1], center, radius)
            outside_span = np.ones(g.shape[1], dtype=bool)
            outside_span[lo : hi + 1] = False
            fg = g[pos, :] != int(bg)
            near_recolored = int(np.count_nonzero(fg & outside_span & within & (out[pos, :] == int(G))))
            far_recolored = int(np.count_nonzero(fg & outside_span & ~within & (out[pos, :] == int(G))))
            far_preserved = int(np.count_nonzero(fg & outside_span & ~within & (out[pos, :] == g[pos, :])))
        else:
            pos = int(first_green["pos"])
            within = _within_radius_indices(g.shape[0], center, radius)
            outside_span = np.ones(g.shape[0], dtype=bool)
            outside_span[lo : hi + 1] = False
            fg = g[:, pos] != int(bg)
            near_recolored = int(np.count_nonzero(fg & outside_span & within & (out[:, pos] == int(G))))
            far_recolored = int(np.count_nonzero(fg & outside_span & ~within & (out[:, pos] == int(G))))
            far_preserved = int(np.count_nonzero(fg & outside_span & ~within & (out[:, pos] == g[:, pos])))

    return {
        "step_count": int(len(combo_steps)),
        "distinct_widths": int(len(set(widths))),
        "width_drop": int(widths[0] - widths[-1]),
        "near_recolored": int(near_recolored),
        "far_recolored": int(far_recolored),
        "far_preserved": int(far_preserved),
        "radius_saved": int(radius_saved),
        "frontier_saved": int(frontier_saved),
        "shrink_saved": int(shrink_saved),
        "diff_no_radius": int(np.count_nonzero(out != no_radius)),
        "diff_no_stop": int(np.count_nonzero(out != no_stop)),
        "diff_no_shrink": int(np.count_nonzero(out != no_shrink)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(28, 30)
    W = rng.randint(28, 30)

    available = [c for c in range(10) if c not in (int(G), int(Y))]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(6, 8)

    if edge in ("top", "bottom"):
        max_start = W - int(np.ceil(1.5 * seg_len - 0.5)) - 4
        min_start = 4
        if max_start < min_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        edge_pos = 0 if edge == "top" else H - 1
        orient = "h"
    else:
        max_start = H - int(np.ceil(1.5 * seg_len - 0.5)) - 4
        min_start = 4
        if max_start < min_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        edge_pos = 0 if edge == "left" else W - 1
        orient = "v"

    canvas = np.full((H, W), int(bg), dtype=int)

    if orient == "h":
        canvas[int(edge_pos), int(seg_start) : int(seg_start) + int(seg_len)] = int(Y)
    else:
        canvas[int(seg_start) : int(seg_start) + int(seg_len), int(edge_pos)] = int(Y)

    info = _detect_segment(canvas)
    if info is None:
        return None

    max_dist = int(info["seg_len"]) * int(DIST_FACTOR)
    combo_steps = _stripe_steps(info, max_dist=max_dist, shrink=True)
    all_shrunk_steps = _stripe_steps(info, max_dist=None, shrink=True)
    if len(combo_steps) < 4:
        return None

    first_green = next((s for s in combo_steps if int(s["color"]) == int(G)), None)
    future_green = next((s for s in all_shrunk_steps if int(s["color"]) == int(G) and int(s["dist"]) > max_dist), None)
    step1 = next((s for s in combo_steps if int(s["step"]) == 1), None)
    if first_green is None or future_green is None or step1 is None:
        return None

    center = float(info["center"])
    near_start = int(info["span_lo"]) - 3
    far_start = int(np.ceil(center + int(info["seg_len"]))) + 1

    if near_start < 1 or far_start + 1 >= (W if orient == "h" else H) - 1:
        return None

    if not _place_line_probe(
        canvas,
        orient=orient,
        pos=int(first_green["pos"]),
        dir1=int(info["dir1"]),
        ortho_start=int(near_start),
        h=2 if orient == "h" else 2,
        w=2 if orient == "h" else 2,
        bg=int(bg),
        fg=int(fg),
    ):
        return None

    if not _place_line_probe(
        canvas,
        orient=orient,
        pos=int(first_green["pos"]),
        dir1=int(info["dir1"]),
        ortho_start=int(far_start),
        h=2 if orient == "h" else 2,
        w=2 if orient == "h" else 2,
        bg=int(bg),
        fg=int(fg),
    ):
        return None

    if not _place_line_probe(
        canvas,
        orient=orient,
        pos=int(future_green["pos"]),
        dir1=int(info["dir1"]),
        ortho_start=int(near_start),
        h=2 if orient == "h" else 2,
        w=2 if orient == "h" else 2,
        bg=int(bg),
        fg=int(fg),
    ):
        return None

    if orient == "h":
        w_h, w_w = 2, 1
        witness_start = int(info["span_hi"])
    else:
        w_h, w_w = 1, 2
        witness_start = int(info["span_hi"])

    if not _place_line_probe(
        canvas,
        orient=orient,
        pos=int(step1["pos"]),
        dir1=int(info["dir1"]),
        ortho_start=int(witness_start),
        h=int(w_h),
        w=int(w_w),
        bg=int(bg),
        fg=int(fg),
    ):
        return None

    if orient == "h":
        protected_rows = {
            int(info["start_pos"]),
            int(step1["pos"]),
            int(step1["pos"]) + int(info["dir1"]),
            int(first_green["pos"]),
            int(first_green["pos"]) + int(info["dir1"]),
            int(future_green["pos"]),
            int(future_green["pos"]) + int(info["dir1"]),
        }
        protected_cols: set[int] = set()
    else:
        protected_rows = set()
        protected_cols = {
            int(info["start_pos"]),
            int(step1["pos"]),
            int(step1["pos"]) + int(info["dir1"]),
            int(first_green["pos"]),
            int(first_green["pos"]) + int(info["dir1"]),
            int(future_green["pos"]),
            int(future_green["pos"]) + int(info["dir1"]),
        }

    extra_target = rng.randint(3, 5)
    placed_extra = 0
    attempts = 0
    while placed_extra < extra_target and attempts < 120:
        attempts += 1
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        r0 = rng.randint(1, H - h - 1)
        c0 = rng.randint(1, W - w - 1)
        row_set = set(range(int(r0), int(r0) + int(h)))
        col_set = set(range(int(c0), int(c0) + int(w)))
        if row_set & protected_rows:
            continue
        if col_set & protected_cols:
            continue
        if not _rect_is_free(canvas, int(r0), int(c0), int(h), int(w), int(bg)):
            continue
        _paint_rect(canvas, int(r0), int(c0), int(h), int(w), int(fg))
        placed_extra += 1

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _lab, n_comp = label_mask(fg_mask, connectivity=4)
    if int(n_comp) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)
    metrics = _combo_metrics(inp, out)
    if metrics is None:
        return None

    if int(get_bg(inp)) != int(bg):
        return None
    if int(metrics["distinct_widths"]) < 4:
        return None
    if int(metrics["width_drop"]) < 3:
        return None
    if int(metrics["near_recolored"]) < 2:
        return None
    if int(metrics["far_recolored"]) != 0:
        return None
    if int(metrics["far_preserved"]) < 2:
        return None
    if int(metrics["radius_saved"]) < 2:
        return None
    if int(metrics["frontier_saved"]) < 2:
        return None
    if int(metrics["shrink_saved"]) < 2:
        return None
    if int(metrics["diff_no_radius"]) <= 0:
        return None
    if int(metrics["diff_no_stop"]) <= 0:
        return None
    if int(metrics["diff_no_shrink"]) <= 0:
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
