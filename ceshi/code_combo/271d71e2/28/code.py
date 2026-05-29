# Combo variant of 271d71e2 (index 28)
# combo_spec: D6/0+D7/2+D12/1
# Name: 共享目标障碍物链式反应
#
# Design analysis
# ===============
# a) Concrete fused behavior
# - The grid contains one clean lane of 2-4 same-direction containers and one shared maroon destination line.
# - All containers keep the original visual language: black rectangular frame, grey/orange interior,
#   head line of 9 adjacent to one side, and orange charging from the head-facing side inward.
# - The fused rule is a forward push chain:
#   1. Containers are ordered from farthest to nearest relative to the shared target.
#   2. The rear container starts moving one cell at a time toward the target.
#   3. If its next step would overlap another container, that front container is recursively pushed
#      forward by one cell first. A pushed container spends one grey cell of its own.
#   4. If a pushed container hits an 8-colored obstacle, the obstacle is recursively pushed too.
#      Obstacles have no fuel, but they still need free space ahead.
#   5. If any object in that chain cannot advance, the whole attempted step fails and the rear
#      container stops there. Containers already moved by a chain do not later take a second free turn.
# - Small example in a rightward lane:
#   input: A ... B .. obstacle .. target
#   base task: A and B would each independently seek a private target; obstacle 8 would not exist.
#   combo task: A moves; once A reaches B, B is forced to move too; once B reaches the obstacle,
#   the obstacle is either pushed or blocks the whole chain. Orange gain equals actual movement of
#   each container, not the rear container's plan.
#
# b) Complexity calibration
# - Only one lane is generated per sample. This avoids visual clutter while still making the fusion
#   unmistakable: many containers, one target, and one obstacle-mediated chain.
# - All containers in the lane share the same head span and cross-section, so the human-visible lane
#   grouping is obvious. Widths may differ; that keeps the scene from degenerating into identical copies.
# - Obstacles use a single color (8) and a single compact shape (2x2). This is enough to show push
#   vs block behavior without introducing a second recognition problem.
#
# c) Hidden assumptions broken from the base solver
# - "Each container independently matches the nearest aligned target" is invalid because several
#   containers now share one target and must compress into a queue through collisions.
# - "Path cells are background" is invalid because 8-blocks occupy the corridor and participate in motion.
# - "Container processing order is irrelevant" is invalid because whether a later container moves at all
#   now depends on an earlier collision chain.
# - "Every non-background color component is a head/dest marker" is invalid because color 8 is purely obstacle.
#
# d) Placement legality guaranteed by the generator
# - The generator builds one canonical rightward lane, validates it, then rotates the whole scene.
# - Initial objects never overlap, remain in bounds, and have at least one background cell of separation
#   between consecutive non-background groups in the lane.
# - The obstacle is always fully inside the lane's sweep corridor and always ahead of the front container.
# - The destination line is always ahead of the obstacle so the obstacle is visibly "between convoy and target".
#
# e) Visual observability
# - A human sees one destination line but several containers aligned toward it, so D6/0 is directly visible.
# - A human also sees that an obstacle sits inside the same corridor and that output either moves the obstacle
#   or leaves the convoy packed up against it, so D7/2 is directly visible.
# - Because several containers move even though only the rear one had clear empty runway, D12/1 is visible
#   as collision propagation rather than independent motion.
#
# f) Visual cleanliness
# - The scene uses only one lane, one target line, and one obstacle block. There are no crossing paths,
#   no overlapping cues, and no need for hidden tie-breakers such as color order or component ids.
#
# g) New solver assumptions and generator support
# - Solver assumption: same-lane containers share direction, head shape, and cross-section.
#   Generator guarantee: every sample is constructed that way.
# - Solver assumption: only one shared destination is aligned ahead of that lane.
#   Generator guarantee: exactly one non-head 9-line is created.
# - Solver assumption: all obstacles relevant to the lane lie inside its cross-section.
#   Generator guarantee: every 8-block is placed fully inside the lane corridor.

from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "solvers"))
from arc_utils import bbox_coords, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
OBSTACLE = 8
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _dominant_bg(g: np.ndarray) -> int:
    cnt = Counter(map(int, g.flatten()))
    return int(max((c for c in cnt if int(c) != int(BORDER)), key=lambda c: cnt[int(c)]))


def _shape_from_pts(pts: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _bbox_from_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    return tuple(int(v) for v in bbox_coords(coords))


def _direction_from_head(box: tuple[int, int, int, int], head_pts: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    hr0, hc0, hr1, hc1 = _bbox_from_pts(head_pts)
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
        return (int(r0), int(r1))
    return (int(c0), int(c1))


def _front_key(box: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    if direction == (0, 1):
        return int(c1)
    if direction == (0, -1):
        return int(-c0)
    if direction == (1, 0):
        return int(r1)
    return int(-r0)


def _head_axis_coord(head_tl: tuple[int, int], direction: tuple[int, int]) -> int:
    return int(head_tl[0] if direction[0] != 0 else head_tl[1])


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


def _box_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _shift_box(box: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int, int, int]:
    dr, dc = direction
    return (int(box[0] + dr), int(box[1] + dc), int(box[2] + dr), int(box[3] + dc))


def _shift_pts(pts: set[tuple[int, int]], direction: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = direction
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _distance_head_to_dest(cont: dict, dest: dict) -> int | None:
    direction = cont["dir"]
    hr, hc = cont["head_tl"]
    dr, dc = dest["tl"]
    if direction == (0, 1):
        if cont["cross_key"] != dest["cross_key"]:
            return None
        dist = int(dc - hc)
    elif direction == (0, -1):
        if cont["cross_key"] != dest["cross_key"]:
            return None
        dist = int(hc - dc)
    elif direction == (1, 0):
        if cont["cross_key"] != dest["cross_key"]:
            return None
        dist = int(dr - hr)
    else:
        if cont["cross_key"] != dest["cross_key"]:
            return None
        dist = int(hr - dr)
    return dist if dist > 0 else None


def _pick_head_for_container(
    g: np.ndarray,
    box: tuple[int, int, int, int],
    comps9: list[dict],
    bg: int,
) -> dict | None:
    H, W = g.shape
    r0, c0, r1, c1 = box
    head_seeds = []
    for r, c in _rect_border_coords(r0, c0, r1, c1):
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(HEAD_COLOR):
                head_seeds.append((rr, cc))
    if not head_seeds:
        return None

    seed_pts = set(head_seeds)
    comp0 = None
    for comp in comps9:
        if comp["pts"] & seed_pts:
            comp0 = comp
            break
    if comp0 is None:
        return None

    head_pts = set(comp0["pts"])
    direction = _direction_from_head(box, head_pts)
    if direction == (0, 0):
        return None

    hr0, hc0, _hr1, _hc1 = _bbox_from_pts(head_pts)
    return {
        "color": int(HEAD_COLOR),
        "head_pts": head_pts,
        "head_tl": (int(hr0), int(hc0)),
        "head_shape": tuple(comp0["shape"]),
        "dir": direction,
        "comp_id": int(comp0["id"]),
    }


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
        if any(int(g[r, c]) != int(BORDER) for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append(
            {
                "id": len(containers),
                "kind": "container",
                "box": (r0, c0, r1, c1),
                "orig_box": (r0, c0, r1, c1),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )

    assert containers

    comps9 = []
    next_comp_id = 0
    for coords in components_mask(g == int(HEAD_COLOR), connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if not pts:
            continue
        bbox = _bbox_from_pts(pts)
        comps9.append(
            {
                "id": next_comp_id,
                "pts": pts,
                "bbox": bbox,
                "tl": (int(bbox[0]), int(bbox[1])),
                "shape": _shape_from_pts(pts),
            }
        )
        next_comp_id += 1

    used_head_comp_ids: set[int] = set()
    for cont in containers:
        head_info = _pick_head_for_container(g, cont["box"], comps9, bg)
        assert head_info is not None
        used_head_comp_ids.add(int(head_info["comp_id"]))

        r0, c0, r1, c1 = cont["box"]
        cont["color"] = int(HEAD_COLOR)
        cont["dir"] = head_info["dir"]
        cont["head_pts"] = set(head_info["head_pts"])
        cont["head_tl"] = tuple(head_info["head_tl"])
        cont["head_shape"] = tuple(head_info["head_shape"])
        cont["cross_key"] = _cross_key(cont["box"], cont["dir"])
        cont["fill_order"] = _fill_order_abs(cont["box"], cont["dir"], g)
        cont["grey_count"] = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == GREY))
        cont["remaining"] = int(cont["grey_count"])
        cont["moved"] = 0
        cont["triggered"] = False

    dests = []
    for comp in comps9:
        if int(comp["id"]) in used_head_comp_ids:
            continue
        bbox = tuple(comp["bbox"])
        cross_key = (
            (int(bbox[0]), int(bbox[2])) if bbox[0] != bbox[2] else (int(bbox[1]), int(bbox[3]))
        )
        dests.append(
            {
                "id": int(comp["id"]),
                "color": int(HEAD_COLOR),
                "pts": set(comp["pts"]),
                "bbox": bbox,
                "tl": tuple(comp["tl"]),
                "shape": tuple(comp["shape"]),
                "cross_key": cross_key,
            }
        )

    obstacles = []
    for oid, coords in enumerate(components_mask(g == int(OBSTACLE), connectivity=4)):
        pts = {(int(r), int(c)) for r, c in coords}
        if not pts:
            continue
        bbox = _bbox_from_pts(pts)
        obstacles.append(
            {
                "id": int(oid),
                "kind": "obstacle",
                "pts": set(pts),
                "bbox": bbox,
                "orig_pts": set(pts),
                "moved": 0,
            }
        )

    return {
        "grid": g,
        "bg": int(bg),
        "H": int(H),
        "W": int(W),
        "containers": containers,
        "dests": dests,
        "obstacles": obstacles,
    }


def _obstacle_in_lane(obs: dict, cross_key: tuple[int, int], direction: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = obs["bbox"]
    if direction[0] == 0:
        return int(cross_key[0]) <= int(r0) and int(r1) <= int(cross_key[1])
    return int(cross_key[0]) <= int(c0) and int(c1) <= int(cross_key[1])


def _build_groups(scene: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cont in scene["containers"]:
        key = (cont["dir"], cont["cross_key"], cont["head_shape"], cont["color"])
        groups[key].append(cont)

    group_infos = []
    for key, members in groups.items():
        members = list(members)
        members.sort(key=lambda c: _front_key(c["box"], c["dir"]), reverse=True)
        front = members[0]

        dest_candidates = []
        for dest in scene["dests"]:
            if dest["color"] != front["color"] or dest["shape"] != front["head_shape"]:
                continue
            dist = _distance_head_to_dest(front, dest)
            if dist is None:
                continue
            dest_candidates.append((dist, dest))
        if not dest_candidates:
            continue
        dest_candidates.sort(key=lambda x: x[0])
        dest = dest_candidates[0][1]

        obstacles = [
            obs
            for obs in scene["obstacles"]
            if _obstacle_in_lane(obs, key[1], key[0])
        ]
        group_infos.append(
            {
                "key": key,
                "containers_front_to_back": members,
                "containers_back_to_front": list(reversed(members)),
                "dest": dest,
                "dest_coord": _head_axis_coord(dest["tl"], key[0]),
                "obstacles": obstacles,
            }
        )
    return group_infos


def _simulate_group(scene: dict, group: dict) -> dict[str, int]:
    H, W = int(scene["H"]), int(scene["W"])
    direction = group["key"][0]
    dest_pts = set(group["dest"]["pts"])
    dest_coord = int(group["dest_coord"])

    container_by_id = {
        int(cont["id"]): cont for cont in group["containers_front_to_back"]
    }
    obstacle_by_id = {int(obs["id"]): obs for obs in group["obstacles"]}

    metrics = {
        "triggered_containers": 0,
        "pushed_obstacles": 0,
        "blocked_by_obstacle": 0,
    }
    triggered_once: set[int] = set()
    obstacle_moved_once: set[int] = set()

    def _all_objects():
        for cont in container_by_id.values():
            yield ("container", int(cont["id"]), _box_cells(cont["box"]) | set(cont["head_pts"]))
        for obs in obstacle_by_id.values():
            yield ("obstacle", int(obs["id"]), set(obs["pts"]))

    def _find_blockers(new_cells: set[tuple[int, int]], self_kind: str, self_id: int) -> list[tuple[str, int]]:
        blockers = []
        for kind, oid, cells in _all_objects():
            if kind == self_kind and int(oid) == int(self_id):
                continue
            if new_cells & cells:
                blockers.append((kind, int(oid)))
        return blockers

    def _obstacle_step(oid: int) -> bool:
        obs = obstacle_by_id[int(oid)]
        new_pts = _shift_pts(obs["pts"], direction)
        for r, c in new_pts:
            if not (0 <= r < H and 0 <= c < W):
                return False
        if new_pts & dest_pts:
            return False

        blockers = _find_blockers(new_pts, "obstacle", int(oid))
        for kind, bid in blockers:
            ok = _obstacle_step(int(bid)) if kind == "obstacle" else _container_step(int(bid), True)
            if not ok:
                return False

        if _find_blockers(new_pts, "obstacle", int(oid)):
            return False

        obs["pts"] = new_pts
        obs["bbox"] = _bbox_from_pts(new_pts)
        obs["moved"] += 1
        if int(oid) not in obstacle_moved_once:
            obstacle_moved_once.add(int(oid))
            metrics["pushed_obstacles"] += 1
        return True

    def _container_step(cid: int, recursive: bool) -> bool:
        cont = container_by_id[int(cid)]
        if recursive and not cont["triggered"]:
            cont["triggered"] = True
            if int(cid) not in triggered_once:
                triggered_once.add(int(cid))
                metrics["triggered_containers"] += 1

        if int(cont["remaining"]) <= 0:
            return False

        new_box = _shift_box(cont["box"], direction)
        new_head_pts = _shift_pts(cont["head_pts"], direction)
        new_head_tl = (
            int(cont["head_tl"][0] + direction[0]),
            int(cont["head_tl"][1] + direction[1]),
        )
        new_cells = _box_cells(new_box) | new_head_pts

        for r, c in new_cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
        if new_cells & dest_pts:
            return False

        axis = _head_axis_coord(new_head_tl, direction)
        if direction in ((0, 1), (1, 0)) and axis > dest_coord:
            return False
        if direction in ((0, -1), (-1, 0)) and axis < dest_coord:
            return False

        blockers = _find_blockers(new_cells, "container", int(cid))
        for kind, oid in blockers:
            ok = _obstacle_step(int(oid)) if kind == "obstacle" else _container_step(int(oid), True)
            if not ok:
                if kind == "obstacle":
                    metrics["blocked_by_obstacle"] += 1
                return False

        if _find_blockers(new_cells, "container", int(cid)):
            return False

        cont["box"] = new_box
        cont["head_pts"] = new_head_pts
        cont["head_tl"] = new_head_tl
        cont["remaining"] -= 1
        cont["moved"] += 1
        return True

    for cont in group["containers_back_to_front"]:
        if cont["triggered"]:
            continue
        while _container_step(int(cont["id"]), False):
            pass

    return metrics


def _render_scene(scene: dict) -> np.ndarray:
    out = np.full_like(scene["grid"], int(scene["bg"]))

    for dest in scene["dests"]:
        for r, c in dest["pts"]:
            out[int(r), int(c)] = int(dest["color"])

    for obs in scene["obstacles"]:
        for r, c in obs["pts"]:
            out[int(r), int(c)] = int(OBSTACLE)

    for cont in scene["containers"]:
        r0, c0, _r1, _c1 = cont["box"]
        patch = cont["patch"].copy()
        or0, oc0, _or1, _oc1 = cont["orig_box"]
        for rr, cc in cont["fill_order"][: int(cont["moved"])]:
            patch[int(rr - or0), int(cc - oc0)] = int(ORANGE)
        paste(out, patch, int(r0), int(c0), clip=True)

    for cont in scene["containers"]:
        for r, c in cont["head_pts"]:
            out[int(r), int(c)] = int(cont["color"])

    return out


def _simulate_scene(scene: dict) -> tuple[dict, list[dict]]:
    groups = _build_groups(scene)
    metrics = []
    for group in groups:
        metrics.append(_simulate_group(scene, group))
    return scene, metrics


def _measure_combo_features(grid: np.ndarray) -> dict[str, int]:
    scene = _detect_scene(np.array(grid, dtype=int))
    groups = _build_groups(scene)
    total = {
        "group_count": int(len(groups)),
        "dest_count": int(len(scene["dests"])),
        "lane_size": 0,
        "moved_containers": 0,
        "triggered_containers": 0,
        "pushed_obstacles": 0,
        "blocked_by_obstacle": 0,
    }
    if groups:
        total["lane_size"] = max(len(g["containers_front_to_back"]) for g in groups)
    for group in groups:
        metric = _simulate_group(scene, group)
        total["triggered_containers"] += int(metric["triggered_containers"])
        total["pushed_obstacles"] += int(metric["pushed_obstacles"])
        total["blocked_by_obstacle"] += int(metric["blocked_by_obstacle"])
    total["moved_containers"] = sum(1 for cont in scene["containers"] if int(cont["moved"]) > 0)
    return total


def main(grid: np.ndarray) -> np.ndarray:
    scene = _detect_scene(np.array(grid, dtype=int))
    _simulate_scene(scene)
    return _render_scene(scene)


def _draw_right_container(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    orange_count: int,
) -> None:
    outer_h, outer_w = inner_h + 2, inner_w + 2
    grid[top, left : left + outer_w] = BORDER
    grid[top + outer_h - 1, left : left + outer_w] = BORDER
    grid[top : top + outer_h, left] = BORDER
    grid[top : top + outer_h, left + outer_w - 1] = BORDER

    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    cells.sort(key=lambda p: (-p[1], p[0]))
    for r, c in cells[: int(orange_count)]:
        interior[int(r), int(c)] = ORANGE
    grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = interior
    grid[top : top + outer_h, left + outer_w] = HEAD_COLOR


def _build_canonical_lane() -> tuple[np.ndarray, str] | None:
    n = random.choice([2, 3, 3])
    mode = random.choice(["push", "block"])

    if n == 3:
        d_between = [2, 2]
        d_obs = 2
        obstacle_free = 1 if mode == "push" else 0
        pushed_steps = obstacle_free
    else:
        d_between = [random.randint(2, 4) for _ in range(n - 1)]
        d_obs = random.randint(2, 4)
        obstacle_free = random.randint(1, 3) if mode == "push" else 0
        pushed_steps = random.randint(1, obstacle_free) if mode == "push" else 0

    actual_back_steps = int(sum(d_between) + d_obs + pushed_steps)
    if mode == "block":
        back_grey_min = int(actual_back_steps + 1)
    else:
        back_grey_min = int(actual_back_steps)

    if n == 3:
        inner_h = 4
        if inner_h * 5 < back_grey_min:
            return None
    else:
        possible_inner_h = [h for h in (2, 3, 4) if h * 5 >= back_grey_min]
        if not possible_inner_h:
            return None
        inner_h = random.choice(possible_inner_h)
    outer_h = inner_h + 2

    required_moves = []
    prefix = 0
    for i in range(n):
        required_moves.append(max(0, actual_back_steps - prefix))
        if i < n - 1:
            prefix += int(d_between[i])

    inner_ws = []
    grey_counts = []
    for i, req in enumerate(required_moves):
        grey_min = int(req)
        if i == 0:
            grey_min = int(back_grey_min)
        inner_w_min = max(1, int(math.ceil(max(1, grey_min) / inner_h)))
        if inner_w_min > 5:
            return None
        if n == 3:
            inner_w = inner_w_min
        else:
            inner_w = random.randint(inner_w_min, 5)
        area = inner_h * inner_w
        grey = random.randint(grey_min, area)
        inner_ws.append(int(inner_w))
        grey_counts.append(int(grey))

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    outer_ws = [w + 2 for w in inner_ws]
    lefts = [int(left_margin)]
    for i in range(n - 1):
        head_col = int(lefts[i] + outer_ws[i])
        lefts.append(int(head_col + d_between[i]))

    front_head_col = int(lefts[-1] + outer_ws[-1])
    obstacle_left = int(front_head_col + d_obs)
    dest_col = int(obstacle_left + 2 + obstacle_free)

    width = int(dest_col + 1 + right_margin)
    height = int(top_margin + outer_h + bottom_margin)
    if width > 26 or height > 26:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for i, left in enumerate(lefts):
        area = inner_h * inner_ws[i]
        orange = int(area - grey_counts[i])
        _draw_right_container(grid, top_margin, int(left), inner_h, int(inner_ws[i]), orange)

    obstacle_top = top_margin + random.randint(0, outer_h - 2)
    grid[obstacle_top : obstacle_top + 2, obstacle_left : obstacle_left + 2] = OBSTACLE
    grid[top_margin : top_margin + outer_h, dest_col] = HEAD_COLOR

    return grid, mode


def _gen_attempt() -> np.ndarray | None:
    built = _build_canonical_lane()
    if built is None:
        return None
    grid, mode = built

    rot_k = random.randint(0, 3)
    grid = np.rot90(grid, rot_k)
    if grid.shape[0] > 26 or grid.shape[1] > 26:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    metrics = _measure_combo_features(grid)
    if metrics["group_count"] != 1:
        return None
    if metrics["dest_count"] != 1:
        return None
    if metrics["lane_size"] < 2:
        return None
    if metrics["moved_containers"] < 2:
        return None
    if metrics["triggered_containers"] < 1:
        return None
    if mode == "push" and metrics["pushed_obstacles"] < 1:
        return None
    if mode == "block" and metrics["blocked_by_obstacle"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid shared-target obstacle chain sample after 300 attempts")
