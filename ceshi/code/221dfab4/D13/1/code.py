# Variant of 221dfab4
# Dimension: D13 — 前景色块的角色 (meta)
# Change: 前景色块是被动对象——仅被绿色条纹重着色... → 色块大小决定局部步长
#
# Design analysis
# a) Concrete behavior:
#    The yellow border segment still launches Y,Y,G stripes toward the opposite side.
#    The new visible change is that stripe spacing is no longer uniform. When a stripe first
#    touches an interior foreground blob, that blob changes the spacing used for later stripes:
#    small blob -> stride +1, large blob -> stride -1. A stride of 3 leaves two bg gap rows/cols;
#    a stride of 1 leaves no forced gap. To keep the interaction attributable, each blob contributes
#    only on first contact; later stripes may recolor it again but do not re-adjust the pace.
#    Example for a top seed: original positions are 0,2,4,6,8,... . In this variant, if row 2 holds
#    a small blob and row 11 holds a large blob, the path becomes 0,2,5,8,11,13,15,... instead of the
#    fixed even rows. That difference is the core visual signal.
# b) Complexity calibration:
#    If blob sizes were close together, the rule would be hard to infer and many samples would
#    accidentally mimic the original task. The generator therefore uses two clearly separated size
#    bands only: small areas <= 9 and large areas >= 18. It also forces both acceleration and
#    deceleration events so the output shows both wider and tighter stripe spacing.
# c) Old implicit assumptions that break:
#    The original code assumes a fixed `2 * step` offset and exactly one bg gap cell after every
#    stripe. Those assumptions are no longer valid once blobs can alter the pace. The solver now
#    maintains a mutable stride state, computes the next stripe position incrementally, and clears
#    all intermediate cells between consecutive stripes rather than a single hard-coded gap.
#    Another hidden assumption was that foreground objects are irrelevant to path planning; this is
#    replaced by explicit component extraction from the input before any recoloring happens.
# d) Placement legality:
#    The generator places three non-overlapping interior blobs with padding, entirely inside the
#    stripe span so yellow stripes can also trigger the new behavior. Each blob is anchored so its
#    first row/column touched by the advancing path is the intended trigger row/column. This avoids
#    accidental early activation while keeping shapes inside bounds.
# e) Visual observability:
#    Every generated sample contains at least three size-trigger events and includes both a small and
#    a large blob. The traced stride sequence must deviate from the original fixed stride on multiple
#    steps, otherwise the sample is rejected. This keeps the D13 change visible from a few examples.
# f) New solver assumptions and generator guarantees:
#    `main()` now assumes there is one dominant straight yellow border segment and that size-changing
#    blobs are separate connected components. The generator guarantees a single border seed, 4-connect
#    separation between blobs, areas in the two explicit size bands, and enough free space that the
#    planned trigger rows/cols exist before the path exits the grid.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    cand = parent / "arc2_solver_select"
    if cand.is_dir():
        sys.path.insert(0, str(cand))
        break

from arc_utils import get_bg, label_mask


Y = 4
G = 3
BASE_STRIDE = 2
MIN_STRIDE = 1
MAX_STRIDE = 4
SMALL_MAX = 9
LARGE_MIN = 18
TRIGGER_STEPS = (1, 4, 8)


def _find_seed_segment(g: np.ndarray) -> tuple[np.ndarray, str, int, np.ndarray, int, int]:
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == Y) & border
    if not bool(smask.any()):
        raise ValueError("missing yellow border segment")

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        raise ValueError("yellow border segment labeling failed")

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        orient = "h"
        start_pos = int(r0)
        span = np.arange(int(c0), int(c1) + 1, dtype=int)
        dir1 = 1 if start_pos == 0 else -1
        limit = H
    elif c0 == c1:
        orient = "v"
        start_pos = int(c0)
        span = np.arange(int(r0), int(r1) + 1, dtype=int)
        dir1 = 1 if start_pos == 0 else -1
        limit = W
    else:
        raise ValueError("yellow border segment is not straight")

    return smask, orient, start_pos, span, dir1, limit


def _collect_objects(g: np.ndarray, bg: int, smask: np.ndarray) -> list[dict]:
    mask = (g != int(bg)) & (~smask)
    lab, n = label_mask(mask, connectivity=4)
    objs: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        objs.append(
            {
                "id": int(cid),
                "area": int(pts.shape[0]),
                "rows": set(int(r) for r in pts[:, 0]),
                "cols": set(int(c) for c in pts[:, 1]),
            }
        )
    return objs


def _size_delta(area: int) -> int:
    area = int(area)
    if area <= SMALL_MAX:
        return 1
    if area >= LARGE_MIN:
        return -1
    return 0


def _touches_object(obj: dict, orient: str, pos: int, span_set: set[int], color: int) -> bool:
    if orient == "h":
        if int(pos) not in obj["rows"]:
            return False
        return bool(color == G or obj["cols"] & span_set)
    if int(pos) not in obj["cols"]:
        return False
    return bool(color == G or obj["rows"] & span_set)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    smask, orient, start_pos, span, dir1, limit = _find_seed_segment(g)
    span_list = [int(x) for x in span.tolist()]
    span_set = set(span_list)
    objects = _collect_objects(g, bg, smask)
    cycle = [Y, Y, G]

    activated: set[int] = set()
    stride = BASE_STRIDE
    step = 0
    pos = int(start_pos)
    trace: list[dict] = []

    while 0 <= int(pos) < int(limit):
        color = int(cycle[int(step) % 3])

        if orient == "h":
            if color == G:
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            out[pos, span] = int(color)
        else:
            if color == G:
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            out[span, pos] = int(color)

        touched_ids: list[int] = []
        delta = 0
        for obj in objects:
            oid = int(obj["id"])
            if oid in activated:
                continue
            if _touches_object(obj, orient, int(pos), span_set, int(color)):
                touched_ids.append(oid)
                delta += _size_delta(int(obj["area"]))

        activated.update(touched_ids)
        next_stride = max(MIN_STRIDE, min(MAX_STRIDE, int(stride) + int(delta)))
        next_pos = int(pos) + int(dir1) * int(next_stride)

        gap = int(pos) + int(dir1)
        while gap != int(next_pos):
            if 0 <= int(gap) < int(limit):
                if orient == "h":
                    out[gap, span] = int(bg)
                else:
                    out[span, gap] = int(bg)
            gap += int(dir1)

        trace.append(
            {
                "step": int(step),
                "pos": int(pos),
                "color": int(color),
                "stride_before": int(stride),
                "stride_after": int(next_stride),
                "delta": int(delta),
                "touched_ids": touched_ids,
            }
        )

        stride = int(next_stride)
        pos = int(next_pos)
        step += 1

    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch Y,Y,G stripes from the yellow edge segment, but let foreground blob size alter later stride.

    Small blob (area <= 9): accelerate, stride +1.
    Large blob (area >= 18): decelerate, stride -1.
    Each blob contributes only on first contact; later recolorings do not modify the pace again.
    """
    g = np.array(grid, dtype=int)
    try:
        return _simulate(g)[0]
    except ValueError:
        return g.copy()


def _make_blob(kind: str, fg: int, bg: int, rng: random.Random) -> np.ndarray:
    if kind == "small":
        choices = [(2, 3), (2, 4), (3, 3)]
    else:
        choices = [(3, 6), (3, 7), (4, 5)]
    h, w = rng.choice(choices)
    blob = np.full((h, w), int(fg), dtype=int)
    if kind == "small" and h >= 3 and w >= 3:
        blob[-1, -1] = int(bg)
    if kind == "large":
        blob[-1, -1] = int(bg)
    return blob


def _fits(canvas: np.ndarray, r0: int, c0: int, blob: np.ndarray, bg: int, padding: int = 1) -> bool:
    H, W = canvas.shape
    h, w = blob.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False

    occ = (canvas != int(bg))
    rr0 = max(0, int(r0) - int(padding))
    rr1 = min(H, int(r0) + h + int(padding))
    cc0 = max(0, int(c0) - int(padding))
    cc1 = min(W, int(c0) + w + int(padding))
    if bool(occ[rr0:rr1, cc0:cc1].any()):
        return False
    return True


def _paste(canvas: np.ndarray, r0: int, c0: int, blob: np.ndarray, bg: int) -> None:
    h, w = blob.shape
    mask = (blob != int(bg))
    canvas[r0 : r0 + h, c0 : c0 + w][mask] = blob[mask]


def _planned_trace(start_pos: int, dir1: int, limit: int, deltas_by_step: dict[int, int]) -> list[dict]:
    cycle = [Y, Y, G]
    pos = int(start_pos)
    stride = BASE_STRIDE
    step = 0
    trace: list[dict] = []
    while 0 <= int(pos) < int(limit) and step <= max(TRIGGER_STEPS) + 2:
        delta = int(deltas_by_step.get(int(step), 0))
        next_stride = max(MIN_STRIDE, min(MAX_STRIDE, int(stride) + int(delta)))
        trace.append(
            {
                "step": int(step),
                "pos": int(pos),
                "color": int(cycle[int(step) % 3]),
                "stride_before": int(stride),
                "stride_after": int(next_stride),
                "delta": int(delta),
            }
        )
        pos = int(pos) + int(dir1) * int(next_stride)
        stride = int(next_stride)
        step += 1
    return trace


def _place_trigger_blob(
    canvas: np.ndarray,
    blob: np.ndarray,
    orient: str,
    span: np.ndarray,
    trigger_pos: int,
    dir1: int,
    bg: int,
    rng: random.Random,
) -> bool:
    H, W = canvas.shape
    h, w = blob.shape

    if orient == "h":
        r0 = int(trigger_pos) if int(dir1) == 1 else int(trigger_pos) - h + 1
        c_candidates = list(range(int(span[0]), int(span[-1]) - w + 2))
        rng.shuffle(c_candidates)
        for c0 in c_candidates:
            if _fits(canvas, r0, int(c0), blob, int(bg), padding=1):
                _paste(canvas, r0, int(c0), blob, int(bg))
                return True
        return False

    c0 = int(trigger_pos) if int(dir1) == 1 else int(trigger_pos) - w + 1
    r_candidates = list(range(int(span[0]), int(span[-1]) - h + 2))
    rng.shuffle(r_candidates)
    for r0 in r_candidates:
        if _fits(canvas, int(r0), c0, blob, int(bg), padding=1):
            _paste(canvas, int(r0), c0, blob, int(bg))
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(30, 34)
    W = rng.randint(30, 34)
    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(8, 11)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = np.arange(seg_start, seg_start + seg_len, dtype=int)
        orient = "h"
        start_pos = 0 if edge == "top" else H - 1
        dir1 = 1 if edge == "top" else -1
        limit = H
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = np.arange(seg_start, seg_start + seg_len, dtype=int)
        orient = "v"
        start_pos = 0 if edge == "left" else W - 1
        dir1 = 1 if edge == "left" else -1
        limit = W

    grid = np.full((H, W), int(bg), dtype=int)
    if orient == "h":
        grid[int(start_pos), span] = int(Y)
    else:
        grid[span, int(start_pos)] = int(Y)

    delta_seq = rng.choice([[1, -1, 1], [1, 1, -1], [1, -1, -1]])
    deltas_by_step = {int(step): int(delta) for step, delta in zip(TRIGGER_STEPS, delta_seq)}
    plan = _planned_trace(int(start_pos), int(dir1), int(limit), deltas_by_step)
    plan_by_step = {int(item["step"]): item for item in plan}

    if not all(int(step) in plan_by_step for step in TRIGGER_STEPS):
        return None

    for step in TRIGGER_STEPS:
        item = plan_by_step[int(step)]
        kind = "small" if int(item["delta"]) > 0 else "large"
        blob = _make_blob(kind, int(fg), int(bg), rng)
        if not _place_trigger_blob(
            grid,
            blob,
            orient,
            span,
            int(item["pos"]),
            int(dir1),
            int(bg),
            rng,
        ):
            return None

    if int(get_bg(grid)) != int(bg):
        return None

    out, trace = _simulate(grid)
    trace_by_step = {int(item["step"]): item for item in trace}
    if not all(int(step) in trace_by_step for step in TRIGGER_STEPS):
        return None

    realized = [int(trace_by_step[int(step)]["delta"]) for step in TRIGGER_STEPS]
    if realized != [int(x) for x in delta_seq]:
        return None

    dynamic_steps = sum(1 for item in trace if int(item["stride_after"]) != BASE_STRIDE)
    if dynamic_steps < 3:
        return None

    if not any(int(item["delta"]) > 0 for item in trace):
        return None
    if not any(int(item["delta"]) < 0 for item in trace):
        return None

    if not np.array_equal(main(grid), out):
        return None

    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
