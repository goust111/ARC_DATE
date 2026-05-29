# Combo variant of 221dfab4 (index 18)
# combo_spec: D7/0+D9/0+D10/1
# Name: 色块阻停+裁剪+递减收缩

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_sprite

# Design analysis for the combo variant:
# a) Concrete behavior and trace:
#    Start from the yellow border segment exactly as in the original task, but the local
#    painted interval shrinks by 1 cell on every stripe: widths L, L-1, ..., 1. The color
#    cycle is still Y, Y, G and the 1-cell gap is still cleared inside the current interval.
#    The D7 change is applied before every write:
#      - yellow stripe line: if the original input already has foreground inside that current
#        narrowed interval, stop before painting the stripe;
#      - green stripe line: if the original input has foreground anywhere on that row/column,
#        stop before painting it;
#      - gap line: if the original input has foreground inside the current narrowed interval,
#        stop after the stripe, without clearing the gap.
#    Example with a top seed of length 6 at cols 10..15:
#      original: widths stay 6 forever and rows 0,2,4,6,... are painted until the far edge;
#      combo: rows 0,2,4,6 would want widths 6,5,4,3. If a blocker first intersects row 6,
#      rows 0/1/2/3/4/5 are the only covered rows, and the final output is cropped to just
#      that wedge-shaped region instead of returning the whole canvas.
# b) Complexity calibration:
#    The generator uses segment lengths 6..8 and forces the stop at step >= 4 so the output
#    shows several distinct widths before stopping. It also keeps at least one unused stripe
#    position beyond the blocker, making early termination visibly different from the base
#    task and from a pure shrink-only variant.
# c) Implicit assumptions audited against the original solver:
#    - A single fixed `span` is no longer valid; every stripe and every cleared gap uses a
#      step-specific interval computed by `_shrunk_interval(...)`.
#    - Obstacle detection must read the original input, not the evolving output, otherwise
#      freshly drawn stripes would self-block later steps.
#    - Green stripes still recolor the full row/column, but tight crop must *not* use those
#      full-line writes as its bbox source, or the crop would expand back to almost full size.
#      The crop therefore follows only the local stripe/gap coverage mask.
#    - Because D7 turns foreground into blockers, successful green lines cannot contain any
#      original foreground. The generator therefore does not rely on visible full-line green
#      flooding; the observable signals are shrink, early stop, and crop.
# d) Placement legality:
#    The blocker blob is constrained to touch the chosen stop stripe line and to lie only on
#    that line or farther in the marching direction, so it cannot accidentally block earlier
#    checks. Extra blobs are placed strictly beyond the blocker, with 1-cell padding, border=1,
#    and no overlap, so the input remains legal and unambiguous.
# e) Visual observability:
#    Across examples, a human sees a long full canvas with scattered blobs in the input, then a
#    cropped wedge in the output whose stripe widths decrease monotonically and whose front ends
#    earlier than the opposite edge. That combination is the intended fused cue.
# New solver assumptions checked against the generator:
#    - The yellow seed is one straight contiguous edge component.
#    - The marching dimension is large enough that an unblocked run would realize all widths
#      down to 1; the generator enforces H/W >= 2*L-1 on the marching axis.
#    - The crop mask is non-empty because the seed stripe is always paintable before the forced
#      blocker line.

Y = 4
G = 3
CYCLE = [Y, Y, G]


def _extract_seed(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

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
            "bg": bg,
            "smask": smask,
            "orient": "h",
            "start_pos": int(r0),
            "span_lo": int(c0),
            "span_hi": int(c1),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": H,
        }
    if c0 == c1:
        return {
            "bg": bg,
            "smask": smask,
            "orient": "v",
            "start_pos": int(c0),
            "span_lo": int(r0),
            "span_hi": int(r1),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
        }
    return None


def _shrunk_interval(span_lo: int, span_hi: int, step: int):
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


def _stripe_descriptor(info: dict, step: int):
    pos = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step))
    if not (0 <= pos < int(info["limit"])):
        return None

    interval = _shrunk_interval(int(info["span_lo"]), int(info["span_hi"]), int(step))
    if interval is None:
        return None

    lo, hi = interval
    return {
        "step": int(step),
        "orient": str(info["orient"]),
        "pos": int(pos),
        "gap": int(pos + int(info["dir1"])),
        "color": int(CYCLE[int(step) % 3]),
        "span_lo": int(lo),
        "span_hi": int(hi),
        "width": int(hi - lo + 1),
    }


def _planned_stripes(info: dict) -> list[dict]:
    steps: list[dict] = []
    step = 0
    while True:
        stripe = _stripe_descriptor(info, step)
        if stripe is None:
            return steps
        steps.append(stripe)
        step += 1


def _line_has_obstacle(
    obstacle_mask: np.ndarray,
    orient: str,
    pos: int,
    span_lo: int,
    span_hi: int,
    *,
    full_line: bool,
) -> bool:
    if orient == "h":
        if full_line:
            return bool(obstacle_mask[int(pos), :].any())
        return bool(obstacle_mask[int(pos), int(span_lo) : int(span_hi) + 1].any())
    if full_line:
        return bool(obstacle_mask[:, int(pos)].any())
    return bool(obstacle_mask[int(span_lo) : int(span_hi) + 1, int(pos)].any())


def _mark_local(mask: np.ndarray, orient: str, pos: int, span_lo: int, span_hi: int) -> None:
    if orient == "h":
        mask[int(pos), int(span_lo) : int(span_hi) + 1] = True
    else:
        mask[int(span_lo) : int(span_hi) + 1, int(pos)] = True


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return np.array(arr, dtype=int).copy()
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.array(arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _simulate_combo(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    out = g.copy()
    coverage = np.zeros_like(g, dtype=bool)

    info = _extract_seed(g)
    if info is None:
        return out, coverage, {
            "stop_kind": "no_seed",
            "stop_step": None,
            "blocked_line": None,
            "painted_steps": 0,
            "total_steps": 0,
            "widths": [],
            "crop_shape": tuple(int(x) for x in g.shape),
        }

    steps = _planned_stripes(info)
    obstacle_mask = (g != int(info["bg"])) & (~info["smask"])
    stop_kind = "natural"
    stop_step = None
    blocked_line = None
    painted_steps = 0
    widths: list[int] = []

    for stripe in steps:
        pos = int(stripe["pos"])
        lo = int(stripe["span_lo"])
        hi = int(stripe["span_hi"])
        col = int(stripe["color"])
        orient = str(stripe["orient"])

        if _line_has_obstacle(
            obstacle_mask,
            orient,
            pos,
            lo,
            hi,
            full_line=(col == int(G)),
        ):
            stop_kind = "stripe_obstacle"
            stop_step = int(stripe["step"])
            blocked_line = int(pos)
            break

        if orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(info["bg"]), int(G), out[pos, :])
            out[pos, lo : hi + 1] = int(col)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(info["bg"]), int(G), out[:, pos])
            out[lo : hi + 1, pos] = int(col)

        _mark_local(coverage, orient, pos, lo, hi)
        painted_steps += 1
        widths.append(int(stripe["width"]))

        gap = int(stripe["gap"])
        if 0 <= gap < int(info["limit"]):
            if _line_has_obstacle(
                obstacle_mask,
                orient,
                gap,
                lo,
                hi,
                full_line=False,
            ):
                stop_kind = "gap_obstacle"
                stop_step = int(stripe["step"])
                blocked_line = int(gap)
                break
            if orient == "h":
                out[gap, lo : hi + 1] = int(info["bg"])
            else:
                out[lo : hi + 1, gap] = int(info["bg"])
            _mark_local(coverage, orient, gap, lo, hi)

    cropped = _crop_to_mask(out, coverage)
    return out, coverage, {
        "stop_kind": stop_kind,
        "stop_step": stop_step,
        "blocked_line": blocked_line,
        "painted_steps": int(painted_steps),
        "total_steps": int(len(steps)),
        "widths": widths,
        "crop_shape": tuple(int(x) for x in cropped.shape),
        "input_shape": tuple(int(x) for x in g.shape),
        "orient": None if info is None else str(info["orient"]),
    }


def _original_full(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    info = _extract_seed(g)
    if info is None:
        return out

    for step in range(int(info["limit"])):
        pos = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step))
        if not (0 <= pos < int(info["limit"])):
            break
        col = int(CYCLE[int(step) % 3])
        lo = int(info["span_lo"])
        hi = int(info["span_hi"])

        if str(info["orient"]) == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(info["bg"]), int(G), out[pos, :])
            out[pos, lo : hi + 1] = int(col)
            gap = pos + int(info["dir1"])
            if 0 <= gap < out.shape[0]:
                out[gap, lo : hi + 1] = int(info["bg"])
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(info["bg"]), int(G), out[:, pos])
            out[lo : hi + 1, pos] = int(col)
            gap = pos + int(info["dir1"])
            if 0 <= gap < out.shape[1]:
                out[lo : hi + 1, gap] = int(info["bg"])

    return out


def _combo_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    full_out, coverage, meta = _simulate_combo(g)
    cropped = _crop_to_mask(full_out, coverage)
    orig = _original_full(g)
    widths = list(meta["widths"])
    return {
        "stop_kind": meta["stop_kind"],
        "painted_steps": int(meta["painted_steps"]),
        "total_steps": int(meta["total_steps"]),
        "distinct_widths": int(len(set(widths))),
        "width_drop": int(widths[0] - widths[-1]) if widths else 0,
        "crop_h": int(cropped.shape[0]),
        "crop_w": int(cropped.shape[1]),
        "input_h": int(g.shape[0]),
        "input_w": int(g.shape[1]),
        "stopped_early": bool(meta["stop_kind"] in {"stripe_obstacle", "gap_obstacle"}),
        "differs_from_original": bool(not np.array_equal(full_out, orig)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    full_out, coverage, _ = _simulate_combo(np.array(grid, dtype=int))
    return _crop_to_mask(full_out, coverage)


def _random_blob_sprite(rng: random.Random, bg: int, fg: int) -> np.ndarray | None:
    for _ in range(60):
        bh = rng.randint(2, 7)
        bw = rng.randint(2, 7)
        density = rng.uniform(0.45, 0.85)
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
        if not bool(rows_any.any()):
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0 : r1 + 1, c0 : c1 + 1]

        pix = int(np.sum(sprite != bg))
        if 4 <= pix <= 30:
            return sprite
    return None


def _candidate_locations(H: int, W: int, sh: int, sw: int, border: int = 1) -> list[tuple[int, int]]:
    return [(r, c) for r in range(border, H - border - sh + 1) for c in range(border, W - border - sw + 1)]


def _is_past_barrier(index: int, barrier: int, dir1: int) -> bool:
    return int(dir1) * (int(index) - int(barrier)) > 0


def _all_on_or_beyond(
    pr: int,
    pc: int,
    sprite_mask: np.ndarray,
    orient: str,
    barrier: int,
    dir1: int,
    *,
    allow_barrier: bool,
) -> bool:
    if orient == "h":
        indices = [int(pr + dr) for dr in range(sprite_mask.shape[0]) if bool(sprite_mask[dr].any())]
    else:
        indices = [int(pc + dc) for dc in range(sprite_mask.shape[1]) if bool(sprite_mask[:, dc].any())]

    if not indices:
        return False
    if allow_barrier:
        return all(int(idx) == int(barrier) or _is_past_barrier(int(idx), int(barrier), int(dir1)) for idx in indices)
    return all(_is_past_barrier(int(idx), int(barrier), int(dir1)) for idx in indices)


def _hits_stop_interval(
    pr: int,
    pc: int,
    sprite_mask: np.ndarray,
    orient: str,
    stop_line: int,
    span_lo: int,
    span_hi: int,
) -> bool:
    if orient == "h":
        for dr in range(sprite_mask.shape[0]):
            if int(pr + dr) != int(stop_line):
                continue
            cols = np.where(sprite_mask[dr])[0]
            for dc in cols:
                abs_c = int(pc + dc)
                if int(span_lo) <= abs_c <= int(span_hi):
                    return True
        return False

    for dc in range(sprite_mask.shape[1]):
        if int(pc + dc) != int(stop_line):
            continue
        rows = np.where(sprite_mask[:, dc])[0]
        for dr in rows:
            abs_r = int(pr + dr)
            if int(span_lo) <= abs_r <= int(span_hi):
                return True
    return False


def _place_sprite(
    rng: random.Random,
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    allow_fn,
) -> bool:
    H, W = canvas.shape
    sh, sw = sprite.shape
    if sh > H - 2 or sw > W - 2:
        return False

    sprite_mask = sprite != bg
    occ = canvas != bg
    occ_pad = ndimage.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=1)

    candidates = _candidate_locations(H, W, sh, sw, border=1)
    rng.shuffle(candidates)
    for pr, pc in candidates:
        if bool((occ_pad[pr : pr + sh, pc : pc + sw] & sprite_mask).any()):
            continue
        if not bool(allow_fn(pr, pc, sprite_mask)):
            continue
        patch = canvas[pr : pr + sh, pc : pc + sw]
        patch[sprite_mask] = sprite[sprite_mask]
        return True
    return False


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(26, 32)
    W = rng.randint(24, 32)

    available = [c for c in range(10) if c not in (int(Y), int(G))]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(6, 8)

    if edge in ("top", "bottom"):
        if H < 2 * seg_len - 1:
            return None
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        edge_row = 0 if edge == "top" else H - 1
        grid = np.full((H, W), bg, dtype=int)
        grid[edge_row, seg_start : seg_start + seg_len] = int(Y)
    else:
        if W < 2 * seg_len - 1:
            return None
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        edge_col = 0 if edge == "left" else W - 1
        grid = np.full((H, W), bg, dtype=int)
        grid[seg_start : seg_start + seg_len, edge_col] = int(Y)

    info = _extract_seed(grid)
    if info is None:
        return None

    steps = _planned_stripes(info)
    if len(steps) < 6:
        return None

    stop_step = rng.randint(4, len(steps) - 2)
    stop = steps[int(stop_step)]
    stop_line = int(stop["pos"])
    stop_lo = int(stop["span_lo"])
    stop_hi = int(stop["span_hi"])
    orient = str(stop["orient"])
    dir1 = int(info["dir1"])

    canvas = grid.copy()

    blocker = _random_blob_sprite(rng, bg, fg)
    if blocker is None:
        return None

    def allow_blocker(pr: int, pc: int, sprite_mask: np.ndarray) -> bool:
        return _all_on_or_beyond(
            pr,
            pc,
            sprite_mask,
            orient,
            stop_line,
            dir1,
            allow_barrier=True,
        ) and _hits_stop_interval(pr, pc, sprite_mask, orient, stop_line, stop_lo, stop_hi)

    if not _place_sprite(rng, canvas, blocker, bg, allow_blocker):
        return None

    target_blobs = rng.randint(5, 8)
    placed = 1
    for _ in range(80):
        if placed >= target_blobs:
            break
        sprite = _random_blob_sprite(rng, bg, fg)
        if sprite is None:
            continue

        def allow_extra(pr: int, pc: int, sprite_mask: np.ndarray) -> bool:
            return _all_on_or_beyond(
                pr,
                pc,
                sprite_mask,
                orient,
                stop_line,
                dir1,
                allow_barrier=False,
            )

        if _place_sprite(rng, canvas, sprite, bg, allow_extra):
            placed += 1

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if int(get_bg(canvas)) != int(bg):
        return None

    out = main(canvas)
    metrics = _combo_metrics(canvas)
    if metrics["stop_kind"] != "stripe_obstacle":
        return None
    if int(metrics["painted_steps"]) != int(stop_step):
        return None
    if not bool(metrics["stopped_early"]):
        return None
    if int(metrics["total_steps"]) - int(metrics["painted_steps"]) < 1:
        return None
    if int(metrics["distinct_widths"]) < 4:
        return None
    if int(metrics["width_drop"]) < 3:
        return None
    if int(metrics["crop_h"]) >= int(metrics["input_h"]):
        return None
    if int(metrics["crop_w"]) >= int(metrics["input_w"]):
        return None
    if not bool(metrics["differs_from_original"]):
        return None
    if min(out.shape) < 4:
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
