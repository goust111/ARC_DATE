# Variant of 271d71e2
# Dimension: D12 — 多容器处理顺序 (multi_step)
# Change: 各容器独立处理，互不影响（路径不交叉，无顺序依赖）... → 链式反应
#
# Design analysis
# ===============
# variant_spec:
#   preserved_visual_language:
#     - 背景仍为 6，容器仍是黑框矩形，内部仍由 5/7 组成。
#     - 头部仍是与容器边等长的 9 线，灰色仍按头侧向内依次变橙。
#   core_D12_change:
#     - 原题里每个容器各自匹配一个远处同色目标线并独立平移。
#     - 本变体改成“同向同轴链”。一条链上的容器都朝同一方向，只有链尾保留唯一目标线。
#     - 起点容器先动；若它恰好走到下一个容器的后侧边界，就触发后者也按同规则移动。
#     - 因此处理顺序不再可交换，必须按链顺序逐个模拟。
#   concrete_example:
#     - 右向链 A -> B -> C -> dest。
#     - 原逻辑：A/B/C 都会各自找目标线；这在新输入里不成立，因为 B/C 根本没有独立目标线。
#     - 新逻辑：A 先走 min(灰数, 到 B 的距离)；若正好碰到 B，则 B 再走 min(灰数, 到 C 的距离)；
#       若 B 也碰到 C，则 C 再走 min(灰数, 到 dest 的距离)。
#     - 差异点就在“后一个是否开始动”依赖前一个是否完成接触。
#   complexity_calibration:
#     - 仍只使用一种头色 9，避免颜色条件过多。
#     - 每个样本只生成 1 条整洁链，长度 2-4；这让顺序依赖强且肉眼可见。
#     - 传播深度会变化：有的样本只触发 1 个后继，有的能传到链尾，避免退化成固定模板。
#   hidden_assumptions_review:
#     - 原 solver 的 `for cont in containers` 默认“处理顺序无关”。在链式反应中这不成立，必须改成先分组再顺序模拟。
#     - 原 solver 默认每个容器都能找到同形状目标线；新变体里只有链尾容器有目标线，所以匹配逻辑必须改成
#       “前方最近障碍可能是容器，也可能是最终目标线”。
#     - 原 solver 在输出图上原地擦除/重贴时，默认别的对象不会依赖中间状态。新变体中触发依赖前一步结果，
#       所以先求每个容器的最终平移步数，再统一绘制最终图更安全。
#   legality_and_generator_guarantees:
#     - 链上容器共享同一横截面（水平链同高，垂直链同宽），这样“碰撞”就是整条边与整条边对齐接触。
#     - 输入中相邻容器之间始终留至少 1 个背景格，避免初始即接触。
#     - 目标线也与链尾头线同长、同轴，并保持至少 1 个背景格。
#     - 生成器只构造 1 条链，保证 solver 的分组假设成立；若旋转后尺寸越界则重试。
#   observability:
#     - 人眼能直接看到：只有链尾有远端目标线，但多个容器都发生位移，说明后续位移来自接触触发而非各自追目标。
#     - 由于不同样本的传播深度不同，2-3 组样例足以支撑“碰到才继续传递”的规律。
#   cleanliness:
#     - 一条链单独占据一个区域，没有交叉、重叠或多层标记，读图清楚。
#   color_inference:
#     - 颜色职责不变：5 是剩余燃料，7 是已充能，9 是头/终点标记。不存在额外颜色映射。
#   new_solver_assumptions_after_change:
#     - 同一条链内：方向一致、横截面对齐、容器在线性顺序上互不穿插。
#     - 只有链尾需要目标线；前面容器是否继续传播，仅由是否碰到下一个容器决定。
#     - 生成器通过构造保证上述条件，因此 solver 可以安全地按链顺序处理。

from __future__ import annotations

from collections import Counter, defaultdict
import os
import random
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "solvers"),
)
from arc_utils import bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
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
    return int(max((c for c in cnt if c != BORDER), key=lambda c: cnt[int(c)]))


def _fill_order_abs(box: tuple[int, int, int, int], direction: tuple[int, int], g: np.ndarray) -> list[tuple[int, int]]:
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


def _direction_from_head(box: tuple[int, int, int, int], head_pts: set[tuple[int, int]]) -> tuple[int, int]:
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


def _sort_key(box: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    if direction == (0, 1):
        return c0
    if direction == (0, -1):
        return -c1
    if direction == (1, 0):
        return r0
    return -r1


def _distance_box_to_box(src: dict, dst: dict) -> int | None:
    direction = src["dir"]
    sr0, sc0, sr1, sc1 = src["box"]
    dr0, dc0, dr1, dc1 = dst["box"]
    if direction == (0, 1):
        if (sr0, sr1) != (dr0, dr1):
            return None
        dist = dc0 - src["head_tl"][1]
    elif direction == (0, -1):
        if (sr0, sr1) != (dr0, dr1):
            return None
        dist = src["head_tl"][1] - dc1
    elif direction == (1, 0):
        if (sc0, sc1) != (dc0, dc1):
            return None
        dist = dr0 - src["head_tl"][0]
    else:
        if (sc0, sc1) != (dc0, dc1):
            return None
        dist = src["head_tl"][0] - dr1
    return int(dist) if dist > 0 else None


def _distance_box_to_dest(cont: dict, dest: dict) -> int | None:
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


def _detect_scene(g: np.ndarray) -> dict:
    g = np.array(g, dtype=int)
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
            r0, c0, r1, c1 = bbox_coords(np.array(list(pts), dtype=int))
            comp_by_color[int(col)].append(
                {
                    "id": next_comp_id,
                    "color": int(col),
                    "pts": pts,
                    "tl": (int(r0), int(c0)),
                    "shape": _shape_from_pts(pts),
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
    for col, comps in comp_by_color.items():
        for comp in comps:
            if int(comp["id"]) in used_head_components:
                continue
            shape = tuple(comp["shape"])
            pts = comp["pts"]
            coords = np.array(list(pts), dtype=int)
            r0, c0, r1, c1 = bbox_coords(coords)
            dests.append(
                {
                    "id": int(comp["id"]),
                    "color": int(col),
                    "pts": pts,
                    "tl": tuple(comp["tl"]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": shape,
                }
            )

    return {"grid": g, "bg": bg, "containers": containers, "dests": dests}


def _simulate_scene(scene: dict) -> dict:
    containers = scene["containers"]
    dests = scene["dests"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cont in containers:
        key = (cont["dir"], cont["color"], cont["cross_key"], cont["head_shape"])
        groups[key].append(cont)

    steps_by_id = {int(cont["id"]): 0 for cont in containers}
    activated_by_id = {int(cont["id"]): False for cont in containers}

    for key, chain in groups.items():
        chain.sort(key=lambda cont: _sort_key(cont["box"], cont["dir"]))
        head_shape = key[3]
        last = chain[-1]
        cand_dests = []
        for dest in dests:
            if dest["color"] != last["color"]:
                continue
            if dest["shape"] != head_shape:
                continue
            dist = _distance_box_to_dest(last, dest)
            if dist is None:
                continue
            cand_dests.append((dist, dest))
        if not cand_dests:
            continue
        cand_dests.sort(key=lambda item: item[0])
        tail_dist, tail_dest = cand_dests[0]

        active = True
        for idx, cont in enumerate(chain):
            if not active:
                break
            activated_by_id[int(cont["id"])] = True
            if idx + 1 < len(chain):
                limit = _distance_box_to_box(cont, chain[idx + 1])
                if limit is None:
                    break
                steps = min(int(cont["grey_count"]), int(limit))
                steps_by_id[int(cont["id"])] = int(steps)
                active = int(steps) == int(limit)
            else:
                steps = min(int(cont["grey_count"]), int(tail_dist))
                steps_by_id[int(cont["id"])] = int(steps)
                active = False

    return {"steps_by_id": steps_by_id, "activated_by_id": activated_by_id}


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


def _measure_chain_reaction(grid: np.ndarray) -> dict[str, int]:
    scene = _detect_scene(grid)
    sim = _simulate_scene(scene)
    moved = sum(1 for v in sim["steps_by_id"].values() if int(v) > 0)
    activated = sum(1 for v in sim["activated_by_id"].values() if bool(v))
    return {"moved": int(moved), "activated": int(activated)}


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


def _build_expected_right(specs: list[dict], outer_h: int, top: int, lefts: list[int], dest_x: int, width: int, height: int) -> np.ndarray:
    expected = np.full((height, width), BG, dtype=int)
    expected[top : top + outer_h, dest_x] = HEAD_COLOR
    active = True
    for i, spec in enumerate(specs):
        steps = 0
        if active:
            limit = int(spec["dist_to_next"] if i < len(specs) - 1 else spec["dest_dist"])
            steps = min(int(spec["grey"]), limit)
            active = i < len(specs) - 1 and steps == limit
        final_left = int(lefts[i] + steps)
        _draw_right_container(
            expected,
            top,
            final_left,
            int(spec["inner_h"]),
            int(spec["inner_w"]),
            int(spec["initial_orange"] + steps),
        )
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3, 3, 4, 4])
    outer_h = random.randint(4, 7)
    inner_h = outer_h - 2
    inner_ws = [random.randint(1, 5) for _ in range(n)]
    areas = [inner_h * inner_w for inner_w in inner_ws]

    trigger_depth = random.choices(range(2, n + 1), weights=list(range(1, n)), k=1)[0]

    specs: list[dict] = []
    dists_to_next = [random.randint(2, min(5, max(2, areas[i]))) for i in range(n - 1)]
    dest_dist = random.randint(2, 5)

    for i in range(n):
        area = int(areas[i])
        inner_w = int(inner_ws[i])
        if i < trigger_depth - 1:
            limit = int(dists_to_next[i])
            if area < limit:
                return None
            grey = random.randint(limit, area)
        elif i == trigger_depth - 1 and i < n - 1:
            limit = int(dists_to_next[i])
            if limit <= 1:
                return None
            grey = random.randint(1, min(area, limit - 1))
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
                "dist_to_next": int(dists_to_next[i]) if i < n - 1 else 0,
                "dest_dist": int(dest_dist) if i == n - 1 else 0,
            }
        )

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    lefts = []
    cur_left = left_margin
    for i, spec in enumerate(specs):
        lefts.append(int(cur_left))
        if i < n - 1:
            cur_left = cur_left + int(spec["outer_w"]) + int(spec["dist_to_next"])
    dest_x = int(lefts[-1] + specs[-1]["outer_w"] + dest_dist)

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

    expected_right = _build_expected_right(
        specs,
        outer_h,
        top_margin,
        lefts,
        dest_x,
        width,
        height,
    )

    rot_k = random.randint(0, 3)
    input_grid = np.rot90(input_right, rot_k)
    expected_grid = np.rot90(expected_right, rot_k)

    if input_grid.shape[0] > 26 or input_grid.shape[1] > 26:
        return None

    try:
        output = main(input_grid)
    except Exception:
        return None

    if not np.array_equal(output, expected_grid):
        return None

    metrics = _measure_chain_reaction(input_grid)
    if metrics["activated"] < 2:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid chain-reaction sample after 400 attempts")
