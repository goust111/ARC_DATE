# Combo variant of 271d71e2 (index 10)
# combo_spec: D6/0+D12/1
# Name: 多容器共享目标链式反应
#
# Design analysis
# ---------------
# a) Concrete behavior trace:
#    - We keep the original visual language: magenta bg, black bordered containers, grey/orange interior,
#      maroon head lines, and one maroon destination line.
#    - The fused scene is a single clean lane of same-direction containers sharing one destination line.
#      Example in the canonical right-facing layout:
#        A .... B .... C .... dest
#      where A is the backmost container and C is closest to the destination.
#    - Original 271d71e2 would move every container independently toward its own destination.
#      D6/0 alone would still let every container join the shared queue.
#      D12/1 alone would let A hit B, B hit C, etc., but would stop each mover at its first collision gap.
#    - In this fused variant:
#      1. A starts moving first.
#      2. If A reaches B's current rear face, B is activated.
#      3. Activated containers do not merely stop at their first collision; instead, the activated prefix
#         re-queues behind the nearest object ahead, preserving the shared-target stack.
#      4. If the active front container cannot reach the next inactive container, the chain stops there.
#      Thus the output shows both effects together: collision-triggered activation and a stacked convoy aimed
#      at one common target.
# b) Complexity calibration:
#    - Only one lane and one destination are generated, keeping the scene readable.
#    - All containers share the same cross-section span so the single destination line is a plausible shared target.
#    - We always generate at least 3 containers and a partial chain depth in [2, n-1], so:
#      1. at least two containers visibly participate in the chain,
#      2. at least one front container stays inactive, proving that chain order matters,
#      3. at least one rear container moves farther than a pure one-collision chain would, proving queue stacking.
# c) Hidden assumptions broken from the original solver:
#    - "nearest aligned destination per container" is no longer valid: only the front of the lane owns the
#      visible destination, yet several containers may still move because activation propagates.
#    - "container updates are independent" is no longer valid: activation depth depends on earlier movers.
#    - "first collision is the final resting place" is no longer valid: once a successor is activated, earlier
#      movers may need extra motion to close the shared queue behind that successor's final position.
# d) Placement legality:
#    - The generator builds one canonical right-facing lane, then rotates it.
#    - Input gaps between head lines and the next object are always at least 2, so there is at least one bg cell.
#    - Container widths are chosen after target step counts are fixed, guaranteeing enough grey cells to support
#      the intended activated prefix without overlaps or out-of-bounds placement.
# e) Visual observability:
#    - A human sees one destination line, several same-style containers on the same lane, and only a suffix of
#      them moving/charging. That rules out both the original 1:1 rule and a naive "everyone always moves" rule.
# f) Cleanliness:
#    - No extra colors or crossing trajectories are introduced.
#    - The lane is axis-aligned and isolated; rotation is the only orientation variation.
# g) Change inferability:
#    - Color roles remain exactly the same as the source task, so the new inference burden is purely structural:
#      shared target ownership and chain-trigger order.
# Post-change solver assumptions and generator guarantees:
#    - There is exactly one lane in generated samples.
#    - The destination line is a separate component (gap >= 2), so head-vs-destination matching stays unambiguous.
#    - Containers in one lane share direction, head shape, and cross-axis span.
#    - The generator guarantees `activated >= 2`, `queue_extra > 0`, and `chain_suppressed > 0`, so the produced
#      examples are not degenerate to either single-dimension variant.

from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
import random
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "solvers")
    ),
)
from arc_utils import bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _dominant_bg(g: np.ndarray) -> int:
    cnt = Counter(map(int, g.flatten()))
    return int(max((c for c in cnt if c != BORDER), key=lambda c: cnt[int(c)]))


def _fill_order_abs(
    box: tuple[int, int, int, int], direction: tuple[int, int], g: np.ndarray
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    grey_coords = [
        (int(r), int(c))
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(g[r, c]) == GREY
    ]
    if direction == (0, 1):
        return sorted(grey_coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_coords, key=lambda p: (p[0], p[1]))


def _shape_from_pts(pts: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _direction_from_head(
    box: tuple[int, int, int, int], head_pts: set[tuple[int, int]]
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    coords = np.array(list(head_pts), dtype=int)
    hr0, hc0, hr1, hc1 = bbox_coords(coords)
    if hc1 < c0:
        return (0, -1)
    if hc0 > c1:
        return (0, 1)
    if hr1 < r0:
        return (-1, 0)
    if hr0 > r1:
        return (1, 0)
    return (0, 0)


def _cross_key(box: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if direction[0] == 0:
        return (r0, r1)
    return (c0, c1)


def _sort_back_to_front(box: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    if direction == (0, 1):
        return c0
    if direction == (0, -1):
        return -c1
    if direction == (1, 0):
        return r0
    return -r1


def _distance_head_to_box(cont: dict, box: tuple[int, int, int, int]) -> int | None:
    direction = cont["dir"]
    hr, hc = cont["head_tl"]
    r0, c0, r1, c1 = box
    if direction == (0, 1):
        if cont["cross_key"] != (r0, r1):
            return None
        dist = c0 - hc
    elif direction == (0, -1):
        if cont["cross_key"] != (r0, r1):
            return None
        dist = hc - c1
    elif direction == (1, 0):
        if cont["cross_key"] != (c0, c1):
            return None
        dist = r0 - hr
    else:
        if cont["cross_key"] != (c0, c1):
            return None
        dist = hr - r1
    return int(dist) if dist > 0 else None


def _distance_head_to_dest(cont: dict, dest: dict) -> int | None:
    direction = cont["dir"]
    hr, hc = cont["head_tl"]
    dr, dc = dest["tl"]
    dr0, dc0, dr1, dc1 = dest["bbox"]
    if direction == (0, 1):
        if cont["cross_key"] != (dr0, dr1):
            return None
        dist = dc - hc
    elif direction == (0, -1):
        if cont["cross_key"] != (dr0, dr1):
            return None
        dist = hc - dc
    elif direction == (1, 0):
        if cont["cross_key"] != (dc0, dc1):
            return None
        dist = dr - hr
    else:
        if cont["cross_key"] != (dc0, dc1):
            return None
        dist = hr - dr
    return int(dist) if dist > 0 else None


def _shift_box(
    box: tuple[int, int, int, int], direction: tuple[int, int], steps: int
) -> tuple[int, int, int, int]:
    dr, dc = direction
    r0, c0, r1, c1 = box
    return (
        int(r0 + dr * steps),
        int(c0 + dc * steps),
        int(r1 + dr * steps),
        int(c1 + dc * steps),
    )


def _detect_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = _dominant_bg(g)

    containers: list[dict] = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append(
            {
                "id": len(containers),
                "box": (r0, c0, r1, c1),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )

    assert containers

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    next_comp_id = 0
    for col in colors_present(g, exclude={bg, BORDER, GREY, ORANGE}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = {(int(r), int(c)) for r, c in coords}
            if not pts:
                continue
            coords_arr = np.array(list(pts), dtype=int)
            r0, c0, r1, c1 = bbox_coords(coords_arr)
            adj_border = False
            for r, c in pts:
                for dr, dc in DIR4:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                        adj_border = True
                        break
                if adj_border:
                    break
            comp_by_color[int(col)].append(
                {
                    "id": next_comp_id,
                    "color": int(col),
                    "pts": pts,
                    "tl": (int(r0), int(c0)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": _shape_from_pts(pts),
                    "adj_border": adj_border,
                }
            )
            next_comp_id += 1

    used_head_components: set[int] = set()
    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (bg, BORDER, GREY, ORANGE):
                        head_seeds.append((rr, cc, val))
        assert head_seeds

        head_color = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == head_color}
        head_comp = None
        for comp in comp_by_color[head_color]:
            if comp["pts"] & seed_pts:
                head_comp = comp
                break
        assert head_comp is not None

        used_head_components.add(int(head_comp["id"]))
        direction = _direction_from_head(cont["box"], head_comp["pts"])
        assert direction != (0, 0)

        cont["color"] = head_color
        cont["dir"] = direction
        cont["head_pts"] = set(head_comp["pts"])
        cont["head_tl"] = tuple(head_comp["tl"])
        cont["head_shape"] = tuple(head_comp["shape"])
        cont["cross_key"] = _cross_key(cont["box"], direction)
        cont["grey_count"] = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == GREY))
        cont["fill_order"] = _fill_order_abs(cont["box"], direction, g)

    dests = []
    for comps in comp_by_color.values():
        for comp in comps:
            if int(comp["id"]) in used_head_components or comp["adj_border"]:
                continue
            dests.append(comp)

    return {"grid": g, "bg": bg, "containers": containers, "dests": dests}


def _pick_group_dest(chain: list[dict], dests: list[dict]) -> tuple[dict, int] | None:
    front = chain[-1]
    cand = []
    for dest in dests:
        if dest["color"] != front["color"]:
            continue
        if tuple(dest["shape"]) != tuple(front["head_shape"]):
            continue
        dist = _distance_head_to_dest(front, dest)
        if dist is None:
            continue
        cand.append((int(dist), dest))
    if not cand:
        return None
    dist, dest = min(cand, key=lambda item: item[0])
    return dest, int(dist)


def _simulate_group_chain(chain: list[dict], dest_dist: int) -> tuple[list[bool], list[int]]:
    n = len(chain)
    active = [False] * n
    steps = [0] * n
    active[0] = True
    for i in range(n):
        if not active[i]:
            break
        if i + 1 < n:
            limit = _distance_head_to_box(chain[i], chain[i + 1]["box"])
        else:
            limit = int(dest_dist)
        if limit is None:
            break
        steps[i] = min(int(chain[i]["grey_count"]), int(limit))
        if i + 1 < n and int(steps[i]) == int(limit):
            active[i + 1] = True
    return active, steps


def _simulate_group_queue(
    chain: list[dict], dest_dist: int, active: list[bool] | None
) -> list[int]:
    n = len(chain)
    steps = [0] * n
    final_boxes: dict[int, tuple[int, int, int, int]] = {}
    for i in range(n - 1, -1, -1):
        if active is not None and not active[i]:
            continue
        if i == n - 1:
            desired = int(dest_dist)
        else:
            anchor_box = final_boxes[i + 1] if i + 1 in final_boxes else chain[i + 1]["box"]
            desired = _distance_head_to_box(chain[i], anchor_box)
        if desired is None:
            continue
        steps[i] = min(int(chain[i]["grey_count"]), int(desired))
        final_boxes[i] = _shift_box(chain[i]["box"], chain[i]["dir"], int(steps[i]))
    return steps


def _simulate_scene(scene: dict) -> dict:
    containers = scene["containers"]
    dests = scene["dests"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cont in containers:
        key = (cont["dir"], cont["color"], cont["cross_key"], cont["head_shape"])
        groups[key].append(cont)

    steps_by_id = {int(cont["id"]): 0 for cont in containers}
    activated_by_id = {int(cont["id"]): False for cont in containers}
    chain_only_steps_by_id = {int(cont["id"]): 0 for cont in containers}
    no_chain_steps_by_id = {int(cont["id"]): 0 for cont in containers}

    for group in groups.values():
        group.sort(key=lambda cont: _sort_back_to_front(cont["box"], cont["dir"]))
        picked = _pick_group_dest(group, dests)
        if picked is None:
            continue
        _dest, dest_dist = picked

        active, chain_only_steps = _simulate_group_chain(group, dest_dist)
        queue_steps = _simulate_group_queue(group, dest_dist, active)
        no_chain_steps = _simulate_group_queue(group, dest_dist, None)

        for idx, cont in enumerate(group):
            cid = int(cont["id"])
            activated_by_id[cid] = bool(active[idx])
            steps_by_id[cid] = int(queue_steps[idx])
            chain_only_steps_by_id[cid] = int(chain_only_steps[idx])
            no_chain_steps_by_id[cid] = int(no_chain_steps[idx])

    return {
        "steps_by_id": steps_by_id,
        "activated_by_id": activated_by_id,
        "chain_only_steps_by_id": chain_only_steps_by_id,
        "no_chain_steps_by_id": no_chain_steps_by_id,
    }


def _render_scene(scene: dict, simulation: dict) -> np.ndarray:
    g = scene["grid"]
    out = np.full_like(g, int(scene["bg"]))

    for dest in scene["dests"]:
        for r, c in dest["pts"]:
            out[int(r), int(c)] = int(dest["color"])

    for cont in scene["containers"]:
        steps = int(simulation["steps_by_id"][int(cont["id"])])
        dr, dc = cont["dir"]
        tr, tc = int(dr * steps), int(dc * steps)
        r0, c0, _r1, _c1 = cont["box"]
        patch = cont["patch"].copy()
        for rr, cc in cont["fill_order"][:steps]:
            patch[int(rr - r0), int(cc - c0)] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

    for cont in scene["containers"]:
        steps = int(simulation["steps_by_id"][int(cont["id"])])
        dr, dc = cont["dir"]
        for r, c in cont["head_pts"]:
            rr = int(r + dr * steps)
            cc = int(c + dc * steps)
            out[rr, cc] = int(cont["color"])

    return out


def _measure_fusion_behavior(grid: np.ndarray) -> dict[str, int]:
    scene = _detect_scene(grid)
    sim = _simulate_scene(scene)
    steps = sim["steps_by_id"]
    chain_only = sim["chain_only_steps_by_id"]
    no_chain = sim["no_chain_steps_by_id"]
    activated = sim["activated_by_id"]

    moved = sum(1 for v in steps.values() if int(v) > 0)
    activated_count = sum(1 for v in activated.values() if bool(v))
    queue_extra = sum(
        max(0, int(steps[cid]) - int(chain_only[cid])) for cid in steps
    )
    chain_suppressed = sum(
        max(0, int(no_chain[cid]) - int(steps[cid])) for cid in steps
    )
    return {
        "moved": int(moved),
        "activated": int(activated_count),
        "queue_extra": int(queue_extra),
        "chain_suppressed": int(chain_suppressed),
    }


def main(grid: np.ndarray) -> np.ndarray:
    scene = _detect_scene(np.array(grid, dtype=int))
    simulation = _simulate_scene(scene)
    return _render_scene(scene, simulation)


def _interior_fill_order_right(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    cells.sort(key=lambda p: (-p[1], p[0]))
    return cells


def _draw_right_container(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    initial_orange: int,
) -> None:
    outer_h, outer_w = inner_h + 2, inner_w + 2
    grid[top, left : left + outer_w] = BORDER
    grid[top + outer_h - 1, left : left + outer_w] = BORDER
    grid[top : top + outer_h, left] = BORDER
    grid[top : top + outer_h, left + outer_w - 1] = BORDER
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for r, c in _interior_fill_order_right(inner_h, inner_w)[:initial_orange]:
        interior[r, c] = ORANGE
    grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = interior
    grid[top : top + outer_h, left + outer_w] = HEAD_COLOR


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4, 4])
    activated_count = 2 if n == 3 else random.choice([2, 3])

    outer_h = random.randint(4, 6 if n == 4 else 7)
    inner_h = outer_h - 2

    gap_hi = 3 if n == 4 else 4
    gaps = [random.randint(2, gap_hi) for _ in range(n - 1)]
    dest_gap = random.randint(2, 4)

    target_steps = [0] * n
    fail_idx = activated_count - 1
    target_steps[fail_idx] = random.randint(1, gaps[fail_idx] - 1)
    for i in range(fail_idx - 1, -1, -1):
        target_steps[i] = int(gaps[i] + target_steps[i + 1])

    specs: list[dict] = []
    for i in range(n):
        min_inner_w = 1
        if i < activated_count:
            min_inner_w = max(1, math.ceil(target_steps[i] / inner_h))
        max_inner_w = 4 if n == 4 else 5
        if min_inner_w > max_inner_w:
            return None
        inner_w = random.randint(min_inner_w, max_inner_w)
        area = int(inner_h * inner_w)

        if i < activated_count:
            if i == fail_idx:
                grey = int(target_steps[i])
            else:
                grey = random.randint(int(target_steps[i]), area)
        else:
            grey = random.randint(1, area)

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": inner_w + 2,
                "grey": int(grey),
                "initial_orange": int(area - grey),
            }
        )

    left_margin = random.randint(1, 2 if n == 4 else 3)
    right_margin = random.randint(1, 2 if n == 4 else 3)
    top_margin = random.randint(1, 2 if n == 4 else 3)
    bottom_margin = random.randint(1, 2 if n == 4 else 3)

    lefts = [left_margin]
    for i in range(1, n):
        prev = specs[i - 1]
        lefts.append(int(lefts[-1] + prev["outer_w"] + gaps[i - 1]))
    dest_x = int(lefts[-1] + specs[-1]["outer_w"] + dest_gap)

    width = int(dest_x + 1 + right_margin)
    height = int(top_margin + outer_h + bottom_margin)
    if width > 26 or height > 26:
        return None

    input_right = np.full((height, width), BG, dtype=int)
    for left, spec in zip(lefts, specs):
        _draw_right_container(
            input_right,
            top_margin,
            int(left),
            int(spec["inner_h"]),
            int(spec["inner_w"]),
            int(spec["initial_orange"]),
        )
    input_right[top_margin : top_margin + outer_h, dest_x] = HEAD_COLOR

    rot_k = random.randint(0, 3)
    grid = np.rot90(input_right, rot_k)
    if grid.shape[0] > 26 or grid.shape[1] > 26:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None

    metrics = _measure_fusion_behavior(grid)
    if metrics["activated"] < 2:
        return None
    if metrics["queue_extra"] <= 0:
        return None
    if metrics["chain_suppressed"] <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid shared-target chain sample after 500 attempts")
