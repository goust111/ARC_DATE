# Combo variant of 221dfab4 (index 19)
# combo_spec: D2/1+D13/1
# Name: GCD间距+色块大小修改步长
#
# Design analysis
# a) Concrete behavior and trace:
#    Keep the original Y,Y,G stripe cycle and the original distinction between yellow
#    span-limited stripes and green full-row/full-column recoloring. The fused change
#    is on the path itself. Example with a top seed and foreground lead rows at
#    3, 6, 12, 18: their consecutive gaps are 3, 6, 6, so the initial stride is
#    gcd(3, 6, 6) = 3 rather than the old fixed 2. Starting from row 0, the first
#    few stripe rows are therefore 0, 3, 6, 9, ... before any blob is touched.
#    If the stripe at row 3 first hits a small blob, the later stride becomes 4,
#    so the next stripe goes to row 7; if that later stripe hits a large blob,
#    the stride drops back to 3. The visible result is that the first spacing is
#    numerically encoded by blob placement, then later spacings visibly widen or
#    tighten after specific contacts. This differs from the original task in two
#    places at once: the opening cadence is not always 2, and the cadence can keep
#    changing as the trace advances.
# b) Complexity calibration:
#    To keep the rule human-inferable, the generator restricts the initial GCD to
#    a small set {2, 3, 4} and uses only two clearly separated size bands:
#    small area <= 9 and large area >= 18. Each sample contains repeated encoded
#    multiples supporting the same GCD and includes both an acceleration event and
#    a deceleration event, so neither half of the fusion is a one-off accident.
# c) Implicit assumptions broken from the original solver:
#    The original `pos = start + dir * (2 * step)` hard-codes both a fixed stride
#    and a single-cell gap. Neither remains valid. The fused solver must compute
#    `pos` incrementally from a mutable stride, and it must clear every intermediate
#    row/column between consecutive stripes, not just one fixed gap cell.
#    Another original assumption was that interior blobs never influence control
#    flow. That also breaks: blobs now serve two roles simultaneously, first as
#    static markers whose lead coordinates encode the initial stride GCD, then as
#    dynamic modifiers that change later stride after first contact.
# d) Placement legality:
#    The generator places solid rectangular blobs fully inside the grid, keeps a
#    one-cell padding between blobs, keeps the yellow seed off the corners, and
#    constrains every trigger blob to overlap the seed span so a yellow or green
#    stripe can legally touch it. Two lateral lanes are used so blobs whose axial
#    intervals are close do not merge into one connected component.
# e) Visual observability:
#    Across a few examples, a human sees three stable signals: the opening stripe
#    spacing matches a repeated foreground spacing pattern, small rectangles are
#    consistently followed by wider later gaps, and large rectangles are followed
#    by tighter later gaps. Because the initial spacing is often 3 or 4 instead of
#    2, and because later gaps deviate again after contacts, the outputs do not
#    look like the original uniform-even-row propagation.
# f) New solver assumptions and generator guarantees:
#    `main()` assumes there is one dominant straight yellow border segment and that
#    foreground blobs are separated 4-connected components. It also assumes blob
#    lead coordinates along the propagation axis provide at least two positive gaps
#    so a GCD can be extracted; the generator guarantees four trigger blobs whose
#    lead coordinates have consecutive differences with gcd equal to the intended
#    initial stride. It further guarantees both small and large blobs are touched
#    exactly once for control-flow purposes, so dynamic stride updates are visible.
# g) Audit of new assumptions introduced by the fused solver:
#    `_derive_initial_stride()` assumes the relevant encoded coordinate of each blob
#    is its first cell encountered along the propagation direction; the generator
#    therefore anchors each blob by that lead edge on a stride-multiple row/column.
#    `_touches_object()` assumes yellow stripes only matter inside the seed span and
#    green stripes can trigger on the full row/column; the generator keeps trigger
#    blobs intersecting the span so they are visible under both stripe colors.
#    The activation set assumes each blob should modify the stride at most once;
#    the generator never relies on repeated activations from the same component.

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    solvers_dir = parent / "solvers"
    if solvers_dir.is_dir():
        sys.path.insert(0, str(solvers_dir))
        break

from arc_utils import get_bg, label_mask


Y = 4
G = 3
SMALL_MAX = 9
LARGE_MIN = 18
MIN_STRIDE = 1
TRIGGER_STEPS = (1, 2, 4, 6)
DELTA_OPTIONS = (
    (1, -1, 1, -1),
    (1, 1, -1, 1),
    (1, -1, -1, 1),
)


def _find_seed_segment(g: np.ndarray) -> tuple[np.ndarray, str, int, np.ndarray, int, int]:
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(Y)) & border
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
        dir1 = 1 if int(start_pos) == 0 else -1
        limit = H
    elif c0 == c1:
        orient = "v"
        start_pos = int(c0)
        span = np.arange(int(r0), int(r1) + 1, dtype=int)
        dir1 = 1 if int(start_pos) == 0 else -1
        limit = W
    else:
        raise ValueError("yellow border segment is not straight")

    return smask, orient, int(start_pos), span, int(dir1), int(limit)


def _lead_pos(obj: dict, orient: str, dir1: int) -> int:
    if orient == "h":
        return min(obj["rows"]) if int(dir1) == 1 else max(obj["rows"])
    return min(obj["cols"]) if int(dir1) == 1 else max(obj["cols"])


def _axis_distance(pos: int, start_pos: int, dir1: int) -> int:
    return int(dir1) * (int(pos) - int(start_pos))


def _collect_objects(g: np.ndarray, bg: int, smask: np.ndarray, orient: str, dir1: int) -> list[dict]:
    mask = (g != int(bg)) & (~smask)
    lab, n = label_mask(mask, connectivity=4)
    objs: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        rows = set(int(r) for r in pts[:, 0])
        cols = set(int(c) for c in pts[:, 1])
        obj = {
            "id": int(cid),
            "area": int(pts.shape[0]),
            "rows": rows,
            "cols": cols,
        }
        obj["lead"] = int(_lead_pos(obj, orient, dir1))
        objs.append(obj)
    return objs


def _derive_initial_stride(objects: list[dict], start_pos: int, dir1: int) -> int:
    lead_distances = sorted(
        {
            int(_axis_distance(int(obj["lead"]), int(start_pos), int(dir1)))
            for obj in objects
            if int(_axis_distance(int(obj["lead"]), int(start_pos), int(dir1))) > 0
        }
    )
    diffs = [int(b - a) for a, b in zip(lead_distances, lead_distances[1:]) if int(b) > int(a)]
    if not diffs:
        return 2
    stride = 0
    for diff in diffs:
        stride = math.gcd(int(stride), int(diff))
    return max(MIN_STRIDE, int(stride))


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
        return bool(int(color) == int(G) or obj["cols"] & span_set)
    if int(pos) not in obj["cols"]:
        return False
    return bool(int(color) == int(G) or obj["rows"] & span_set)


def _paint_stripe(out: np.ndarray, orient: str, pos: int, span: np.ndarray, color: int, bg: int) -> None:
    if orient == "h":
        if int(color) == int(G):
            out[int(pos), :] = np.where(out[int(pos), :] != int(bg), int(G), out[int(pos), :])
        out[int(pos), span] = int(color)
        return
    if int(color) == int(G):
        out[:, int(pos)] = np.where(out[:, int(pos)] != int(bg), int(G), out[:, int(pos)])
    out[span, int(pos)] = int(color)


def _clear_gap(out: np.ndarray, orient: str, span: np.ndarray, bg: int, cur_pos: int, next_pos: int, dir1: int) -> None:
    gap = int(cur_pos) + int(dir1)
    while int(gap) != int(next_pos):
        if orient == "h":
            out[int(gap), span] = int(bg)
        else:
            out[span, int(gap)] = int(bg)
        gap += int(dir1)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    smask, orient, start_pos, span, dir1, limit = _find_seed_segment(g)
    span_set = set(int(x) for x in span.tolist())
    objects = _collect_objects(g, int(bg), smask, orient, dir1)
    initial_stride = int(_derive_initial_stride(objects, int(start_pos), int(dir1)))

    cycle = [int(Y), int(Y), int(G)]
    activated: set[int] = set()
    pos = int(start_pos)
    stride = int(initial_stride)
    step = 0
    trace: list[dict] = []

    while 0 <= int(pos) < int(limit):
        color = int(cycle[int(step) % 3])
        _paint_stripe(out, orient, int(pos), span, int(color), int(bg))

        touched_ids: list[int] = []
        delta = 0
        for obj in objects:
            oid = int(obj["id"])
            if oid in activated:
                continue
            if _touches_object(obj, orient, int(pos), span_set, int(color)):
                touched_ids.append(oid)
                delta += int(_size_delta(int(obj["area"])))

        activated.update(touched_ids)
        next_stride = max(MIN_STRIDE, int(stride) + int(delta))
        next_pos = int(pos) + int(dir1) * int(next_stride)

        if 0 <= int(next_pos) < int(limit):
            _clear_gap(out, orient, span, int(bg), int(pos), int(next_pos), int(dir1))

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

        pos = int(next_pos)
        stride = int(next_stride)
        step += 1

    meta = {
        "bg": int(bg),
        "orient": orient,
        "start_pos": int(start_pos),
        "dir1": int(dir1),
        "limit": int(limit),
        "span": span.copy(),
        "initial_stride": int(initial_stride),
        "objects": objects,
    }
    return out, trace, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch Y,Y,G stripes from the yellow border seed.

    Initial stride = gcd of consecutive lead-coordinate gaps between foreground blobs
    along the propagation axis. After a stripe first touches a blob, the blob area
    changes the later stride: small (<= 9) -> +1, large (>= 18) -> -1.
    """
    g = np.array(grid, dtype=int)
    try:
        return _simulate(g)[0]
    except ValueError:
        return g.copy()


def _planned_trace(start_pos: int, dir1: int, limit: int, initial_stride: int, deltas_by_step: dict[int, int]) -> list[dict]:
    cycle = [int(Y), int(Y), int(G)]
    pos = int(start_pos)
    stride = int(initial_stride)
    step = 0
    trace: list[dict] = []
    max_step = int(max(TRIGGER_STEPS)) + 2
    while 0 <= int(pos) < int(limit) and int(step) <= int(max_step):
        delta = int(deltas_by_step.get(int(step), 0))
        next_stride = max(MIN_STRIDE, int(stride) + int(delta))
        trace.append(
            {
                "step": int(step),
                "pos": int(pos),
                "distance": int(_axis_distance(int(pos), int(start_pos), int(dir1))),
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


def _make_blob(kind: str, base_stride: int, fg: int, bg: int, orient: str) -> tuple[np.ndarray, int, int]:
    axis_len = int(base_stride) if kind == "small" else max(int(base_stride), 3)
    cross_len = 2 if kind == "small" else int(math.ceil(LARGE_MIN / axis_len))

    if orient == "h":
        h, w = int(axis_len), int(cross_len)
    else:
        h, w = int(cross_len), int(axis_len)

    blob = np.full((int(h), int(w)), int(fg), dtype=int)
    area = int(np.sum(blob != int(bg)))
    if kind == "small" and area > SMALL_MAX:
        raise ValueError("small blob too large")
    if kind == "large" and area < LARGE_MIN:
        raise ValueError("large blob too small")
    return blob, int(axis_len), int(cross_len)


def _fits(canvas: np.ndarray, r0: int, c0: int, blob: np.ndarray, bg: int, padding: int = 1) -> bool:
    H, W = canvas.shape
    h, w = blob.shape
    if r0 < 1 or c0 < 1 or int(r0) + h > H - 1 or int(c0) + w > W - 1:
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
    mask = (blob != int(bg))
    h, w = blob.shape
    canvas[int(r0) : int(r0) + h, int(c0) : int(c0) + w][mask] = blob[mask]


def _place_blob(
    canvas: np.ndarray,
    blob: np.ndarray,
    orient: str,
    dir1: int,
    lead_pos: int,
    lane_start: int,
    bg: int,
) -> bool:
    h, w = blob.shape
    if orient == "h":
        r0 = int(lead_pos) if int(dir1) == 1 else int(lead_pos) - h + 1
        c0 = int(lane_start)
    else:
        r0 = int(lane_start)
        c0 = int(lead_pos) if int(dir1) == 1 else int(lead_pos) - w + 1

    if not _fits(canvas, int(r0), int(c0), blob, int(bg), padding=1):
        return False
    _paste(canvas, int(r0), int(c0), blob, int(bg))
    return True


def _lead_distance_for_touch(touch_distance: int, base_stride: int) -> int:
    return max(int(base_stride), (int(touch_distance) // int(base_stride)) * int(base_stride))


def _edge_setup(rng: random.Random, H: int, W: int, seg_len: int) -> tuple[str, np.ndarray, int, int, int]:
    edge = rng.choice(["top", "bottom", "left", "right"])
    if edge in ("top", "bottom"):
        seg_start = rng.randint(1, W - 1 - seg_len)
        span = np.arange(int(seg_start), int(seg_start) + int(seg_len), dtype=int)
        start_pos = 0 if edge == "top" else H - 1
        dir1 = 1 if edge == "top" else -1
        limit = H
    else:
        seg_start = rng.randint(1, H - 1 - seg_len)
        span = np.arange(int(seg_start), int(seg_start) + int(seg_len), dtype=int)
        start_pos = 0 if edge == "left" else W - 1
        dir1 = 1 if edge == "left" else -1
        limit = W
    orient = "h" if edge in ("top", "bottom") else "v"
    return orient, span, int(start_pos), int(dir1), int(limit)


def _original_style_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    try:
        smask, orient, start_pos, span, dir1, limit = _find_seed_segment(g)
    except ValueError:
        return g.copy()

    bg = int(get_bg(g))
    out = g.copy()
    cycle = [int(Y), int(Y), int(G)]
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= int(pos) < int(limit)):
            break
        color = int(cycle[int(step) % 3])
        _paint_stripe(out, orient, int(pos), span, int(color), int(bg))
        gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
        if 0 <= int(gap) < int(limit):
            if orient == "h":
                out[int(gap), span] = int(bg)
            else:
                out[span, int(gap)] = int(bg)
        step += 1
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(32, 38)
    W = rng.randint(32, 38)
    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    base_stride = rng.choice([2, 3, 4])
    seg_len = rng.randint(14, 16)
    orient, span, start_pos, dir1, limit = _edge_setup(rng, H, W, seg_len)

    grid = np.full((H, W), int(bg), dtype=int)
    if orient == "h":
        grid[int(start_pos), span] = int(Y)
    else:
        grid[span, int(start_pos)] = int(Y)

    delta_seq = list(rng.choice(DELTA_OPTIONS))
    deltas_by_step = {int(step): int(delta) for step, delta in zip(TRIGGER_STEPS, delta_seq)}
    plan = _planned_trace(int(start_pos), int(dir1), int(limit), int(base_stride), deltas_by_step)
    plan_by_step = {int(item["step"]): item for item in plan}
    if not all(int(step) in plan_by_step for step in TRIGGER_STEPS):
        return None

    lane_sides = ["left", "right", "left", "right"]
    lead_distances: list[int] = []

    for lane_side, step, delta in zip(lane_sides, TRIGGER_STEPS, delta_seq):
        item = plan_by_step[int(step)]
        kind = "small" if int(delta) > 0 else "large"
        blob, axis_len, cross_len = _make_blob(kind, int(base_stride), int(fg), int(bg), orient)
        touch_distance = int(item["distance"])
        lead_distance = int(_lead_distance_for_touch(int(touch_distance), int(base_stride)))
        if int(lead_distance) == 0:
            return None
        lead_distances.append(int(lead_distance))

        if lane_side == "left":
            lane_start = int(span[0])
        else:
            lane_start = int(span[-1]) - int(cross_len) + 1

        if int(lane_start) < int(span[0]):
            return None

        lead_pos = int(start_pos) + int(dir1) * int(lead_distance)
        if not _place_blob(grid, blob, orient, int(dir1), int(lead_pos), int(lane_start), int(bg)):
            return None

        if int(lead_distance) - int(touch_distance) > 0:
            return None
        if int(touch_distance) - int(lead_distance) > int(axis_len) - 1:
            return None

    unique_leads = sorted(set(int(x) for x in lead_distances))
    diffs = [int(b - a) for a, b in zip(unique_leads, unique_leads[1:])]
    if len(unique_leads) < 4 or not diffs:
        return None
    gcd_val = 0
    for diff in diffs:
        gcd_val = math.gcd(int(gcd_val), int(diff))
    if int(gcd_val) != int(base_stride):
        return None

    if int(get_bg(grid)) != int(bg):
        return None

    out, trace, meta = _simulate(grid)
    trace_by_step = {int(item["step"]): item for item in trace}
    if int(meta["initial_stride"]) != int(base_stride):
        return None
    if not all(int(step) in trace_by_step for step in TRIGGER_STEPS):
        return None

    realized = [int(trace_by_step[int(step)]["delta"]) for step in TRIGGER_STEPS]
    if realized != [int(x) for x in delta_seq]:
        return None

    dynamic_events = sum(1 for item in trace if int(item["delta"]) != 0)
    if dynamic_events < 4:
        return None
    if not any(int(item["stride_before"]) != int(item["stride_after"]) for item in trace):
        return None
    if not any(int(item["delta"]) > 0 for item in trace):
        return None
    if not any(int(item["delta"]) < 0 for item in trace):
        return None

    if np.array_equal(out, _original_style_output(grid)):
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
