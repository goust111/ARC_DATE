# Combo variant of 581f7754 (index 28)
# combo_spec: D5/1+D6/0
# Name: 碰撞停止+复制而非移动

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis
# a) 具体化与行为追踪
#    - 画面仍由三类可见元素组成：边界孤立锚点、内部多色 sprite、背景。
#    - 原题流程是：识别 marker 颜色 -> 找同色锚点 -> 按锚点所在边决定水平/垂直平移 ->
#      清掉原 sprite -> 一次性把 sprite 粘到锚点列/行。
#    - 本融合变体同时改两处：
#         1. 不再清源位置，而是保留原 sprite，并额外生成一个沿同一轴滑动的副本。
#         2. 副本不是直接瞬移到目标列/行，而是逐格试走；下一步只要会撞到任何非背景像素就提前停下。
#    - 关键协同点在于：因为源 sprite 也保留，所以“原位副本”本身就是静态障碍的一部分。
#      例如顶边锚点在列 9，某 sprite 的 marker 在 (6,3)，按原题应向右移动 6 格。
#      若前方第 4 步会撞上另一个源 sprite，则融合版只在前 3 步的位置生成副本，且输入中的旧 sprite 仍留在原处。
#
# b) 复杂度校准
#    - 若沿运动轴保留原题那种致密实体块，副本第一步就常会撞到自己的源位置，样本会退化成“几乎都不动”。
#    - 因此生成器改用稀疏 checkerboard 风格的 8 连通 body：仍是多像素小精灵，但沿任一主轴平移 1 格不会与自身重叠。
#    - 同时每题强制至少 1 个 mover 被“保留的源 sprite”挡住，且至少 2 个 mover 真正生成新副本，
#      这样肉眼能同时看到 keep_source 与 collision_stop 两个分支。
#
# c) 隐式假设审查
#    - 锚点检测前提没变：边界单像素、4 邻域全是背景。
#    - marker 识别前提没变：组件内最少色像素就是 marker；单像素且颜色命中锚点色的内部小件也仍可移动。
#    - 原 solver 在 STEP 5 里先清 movers 再 paste，这隐含“路径上中间状态无关紧要”且“源位置不构成障碍”。
#      融合后这两条都失效，所以 solver 必须在完整原图上逐步模拟。
#    - 新 solver 还新增一个前提：若想看到真正的复制效果，source 与最终 copy 不能互相覆盖。
#      生成器通过 axis-safe 稀疏 sprite 和最小步长约束显式保证这一点。
#
# d) 放置合法性
#    - 生成器转为 input-first：先在输入里放 anchors 和 source sprites，再用 solver 语义推导输出。
#    - 放置新 source 时，不仅检查当前输入里不重叠、不越界、不粘连，还禁止落在先前 mover 的“未来行进路径”上，
#      否则会反向改变之前构造好的停止距离。
#    - 另外单独记录各 mover 的最终 copy 区域，避免不同 copy 在输出中彼此覆盖。
#
# e) 视觉可观察性
#    - 解题者能直接看到：同色 marker 仍对应同色边界锚点，锚点边仍决定水平/垂直轴。
#    - 输出里每个成功移动的 sprite 都会出现“原位保留 + 某处新增副本”；而至少有一个副本不会到达锚点线，
#      且阻挡它的是另一个仍留在输入位置的 sprite，而不是隐藏状态。
#
# f) 可视化整洁度
#    - 源 sprites 之间保持 8 邻域间隔；copy 之间也显式去重。
#    - 允许“停在障碍前一格”的贴近关系，因为那正是规则本身；但避免大面积重叠和难分辨的局部加粗。
#
# g) 变化的可推断性
#    - 颜色角色保持原题语义：锚点色 == marker 色；body 色只用于区分 sprite。
#    - 不引入任何基于颜色编号的排序、优先级或状态编码；所有推理都来自可见的颜色匹配与空间阻挡。
#
# new_solver_assumptions:
#    - 每个 mover 都能由最少色 marker 唯一映射到某个 anchor 色。
#    - 生成器保证至少一个 mover 的停止距离会因为“源 sprite 未被清除”而短于 clear-source 基线。
#    - 生成器保证最终 copy 两两不重叠，因此 solver 的顺序 paste 不会改变语义。

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 10)
_GEN_RETRY_LIMIT = 500
_PLACEMENT_TRY_LIMIT = 220
_SPRITE_BUILD_TRY_LIMIT = 80
_MIN_MOVE_STEPS = 2


def _norm(cells):
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _bbox(cells):
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _flip(cells):
    cells = set(cells)
    if random.random() < 0.5:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < 0.5:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _axis_size_choices(axis: str, area: int):
    choices = []
    if axis == "h":
        for h in _inclusive_range((2, 4)):
            for w in _inclusive_range((3, 7)):
                if (h * w + 1) // 2 >= area:
                    choices.append((h, w))
    else:
        for h in _inclusive_range((3, 5)):
            for w in _inclusive_range((2, 6)):
                if (h * w + 1) // 2 >= area:
                    choices.append((h, w))
    return choices


def _allowed_sparse_cells(height: int, width: int):
    return {
        (r, c)
        for r in range(int(height))
        for c in range(int(width))
        if (int(r) + int(c)) % 2 == 0
    }


def _sparse_frontier(cells, allowed):
    frontier = set()
    for r, c in cells:
        for dr, dc in _NEIGHBORS8:
            rr = int(r + dr)
            cc = int(c + dc)
            if (rr, cc) in allowed and (rr, cc) not in cells:
                frontier.add((rr, cc))
    return list(frontier)


def _grow_sparse_body(area: int, axis: str):
    choices = _axis_size_choices(axis, area)
    if not choices:
        raise ValueError("no sparse-body size choices")
    for _ in range(_SPRITE_BUILD_TRY_LIMIT):
        h, w = random.choice(choices)
        allowed = _allowed_sparse_cells(h, w)
        cells = {random.choice(list(allowed))}
        while len(cells) < int(area):
            frontier = _sparse_frontier(cells, allowed)
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == int(area):
            return _flip(cells)
    raise RuntimeError("failed to build sparse body")


def _make_axis_sprite(area: int, body_color: int, marker_color: int, bg: int, axis: str):
    for _ in range(_SPRITE_BUILD_TRY_LIMIT):
        body = _grow_sparse_body(int(area), axis)
        r0, c0, r1, c1 = _bbox(body)
        marker_candidates = set()
        for r, c in body:
            for dr, dc in _NEIGHBORS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if (rr, cc) in body:
                    continue
                if axis == "h" and rr not in (int(r0 - 1), int(r1 + 1)):
                    continue
                if axis == "v" and cc not in (int(c0 - 1), int(c1 + 1)):
                    continue
                marker_candidates.add((rr, cc))

        for marker in random.sample(list(marker_candidates), len(marker_candidates)):
            all_cells = set(body) | {marker}
            shifted = {(int(r), int(c + 1)) for r, c in all_cells} if axis == "h" else {(int(r + 1), int(c)) for r, c in all_cells}
            if all_cells & shifted:
                continue
            r0, c0, r1, c1 = _bbox(all_cells)
            patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
            cells = []
            for r, c in body:
                rr = int(r - r0)
                cc = int(c - c0)
                patch[rr, cc] = int(body_color)
                cells.append((int(rr), int(cc), int(body_color)))
            mr = int(marker[0] - r0)
            mc = int(marker[1] - c0)
            patch[mr, mc] = int(marker_color)
            cells.append((int(mr), int(mc), int(marker_color)))
            return {"patch": patch, "offset": (int(mr), int(mc)), "cells": cells}
    raise RuntimeError("failed to build axis-safe sprite")


def _sprite_abs_cells(sprite, marker_r: int, marker_c: int, bg: int):
    off_r, off_c = sprite["offset"]
    top = int(marker_r) - int(off_r)
    left = int(marker_c) - int(off_c)
    out = []
    for r, c, v in sprite["cells"]:
        if int(v) != int(bg):
            out.append((int(top + r), int(left + c), int(v)))
    return out


def _sprite_abs_coords(sprite, marker_r: int, marker_c: int, bg: int):
    return [(int(r), int(c)) for r, c, _v in _sprite_abs_cells(sprite, int(marker_r), int(marker_c), int(bg))]


def _cells_ok(canvas: np.ndarray, coords, bg: int, *, require_gap: bool, forbidden=frozenset()):
    H, W = canvas.shape
    forbidden = {(int(r), int(c)) for r, c in forbidden}
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in coord_set:
        if not (_INNER_BORDER_GAP <= int(r) < H - _INNER_BORDER_GAP):
            return False
        if not (_INNER_BORDER_GAP <= int(c) < W - _INNER_BORDER_GAP):
            return False
        if (int(r), int(c)) in forbidden:
            return False
        if int(canvas[int(r), int(c)]) != int(bg):
            return False
    if not require_gap:
        return True
    for r, c in coord_set:
        for dr, dc in _NEIGHBORS8:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (int(rr), int(cc)) in coord_set:
                continue
            if int(canvas[rr, cc]) != int(bg):
                return False
    return True


def _can_place_sprite(canvas: np.ndarray, sprite, marker_r: int, marker_c: int, bg: int, *, forbidden=frozenset()):
    coords = _sprite_abs_coords(sprite, int(marker_r), int(marker_c), int(bg))
    return _cells_ok(canvas, coords, int(bg), require_gap=True, forbidden=forbidden)


def _place_sprite(canvas: np.ndarray, sprite, marker_r: int, marker_c: int, bg: int):
    for r, c, v in _sprite_abs_cells(sprite, int(marker_r), int(marker_c), int(bg)):
        canvas[int(r), int(c)] = int(v)


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    S = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _detect_anchors_and_movers(g: np.ndarray, bg: int):
    H, W = g.shape
    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(int(r), int(c), int(H), int(W)):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert anchors_by_color
    anchor_cells = set(anchors_by_color.values())
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    movers = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        desired_dr = 0 if horizontal else int(ar - mp[0])
        desired_dc = int(ac - mp[1]) if horizontal else 0
        movers.append(
            {
                "coords": coords.astype(int),
                "marker_color": int(marker_col),
                "marker_pixel": (int(mp[0]), int(mp[1])),
                "anchor": (int(ar), int(ac)),
                "horizontal": bool(horizontal),
                "desired_dr": int(desired_dr),
                "desired_dc": int(desired_dc),
            }
        )
    return anchors_by_color, movers


def _collision_stop_shift(coords: np.ndarray, desired_dr: int, desired_dc: int, static: np.ndarray, bg: int):
    H, W = static.shape
    step_r = 0 if int(desired_dr) == 0 else (1 if int(desired_dr) > 0 else -1)
    step_c = 0 if int(desired_dc) == 0 else (1 if int(desired_dc) > 0 else -1)
    total_steps = abs(int(desired_dr)) + abs(int(desired_dc))
    current = [(int(r), int(c)) for r, c in coords]
    moved = 0
    for _ in range(total_steps):
        nxt = []
        impact_cells = []
        blocked_by_oob = False
        for r, c in current:
            rr = int(r + step_r)
            cc = int(c + step_c)
            if not (0 <= rr < H and 0 <= cc < W):
                blocked_by_oob = True
                continue
            if int(static[rr, cc]) != int(bg):
                impact_cells.append((int(rr), int(cc)))
                continue
            nxt.append((int(rr), int(cc)))
        if blocked_by_oob or impact_cells:
            return {
                "dr": int(step_r * moved),
                "dc": int(step_c * moved),
                "steps": int(moved),
                "target_steps": int(total_steps),
                "blocked": True,
                "step_r": int(step_r),
                "step_c": int(step_c),
                "impact_cells": [(int(r), int(c)) for r, c in impact_cells],
                "blocked_by_oob": bool(blocked_by_oob),
            }
        current = nxt
        moved += 1
    return {
        "dr": int(step_r * moved),
        "dc": int(step_c * moved),
        "steps": int(moved),
        "target_steps": int(total_steps),
        "blocked": False,
        "step_r": int(step_r),
        "step_c": int(step_c),
        "impact_cells": [],
        "blocked_by_oob": False,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy interior sprites toward matching border anchors, but stop the copied motion at the first collision.

    Input structure:
    - The most common color is background.
    - Interior foreground components are multi-color sprites whose least-frequent color is a marker matching some border anchor.
    - Border anchors are isolated single pixels, and any original foreground sprite that remains in place can act as a blocker.

    Rule:
    - Match each movable sprite to the anchor with the same marker color.
    - Top/bottom anchors make the copy slide horizontally; left/right anchors make it slide vertically.
    - Keep the source sprite unchanged, then simulate a copied version one grid step at a time on top of the full original input; if the next step would hit any non-background cell or leave the grid, stop there and stamp only the reachable copied position.

    Key insight: The anchor still picks the axis, but because the source is preserved, the original foreground layout itself becomes the obstacle map that determines how far the extra copy can travel.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _anchors, movers = _detect_anchors_and_movers(g, int(bg))

    out = g.copy()
    H, W = out.shape
    static = g.copy()
    for mover in movers:
        sim = _collision_stop_shift(
            mover["coords"],
            int(mover["desired_dr"]),
            int(mover["desired_dc"]),
            static,
            int(bg),
        )
        for r, c in mover["coords"]:
            rr = int(r) + int(sim["dr"])
            cc = int(c) + int(sim["dc"])
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])
    return out


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0 = max(0, int(r - radius))
    r1 = min(H, int(r + radius + 1))
    c0 = max(0, int(c - radius))
    c1 = min(W, int(c + radius + 1))
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (int(r) in (0, H - 1) or int(c) in (0, W - 1)):
        return False
    if int(r) in (0, H - 1) and int(c) in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, int(r), int(c), int(bg))


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    for r in rows:
        out.append(("v", r, 0))
        out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_marker(anchor, marker_r: int, marker_c: int):
    if anchor["axis"] == "h":
        return int(marker_r), int(anchor["c"])
    return int(anchor["r"]), int(marker_c)


def _desired_shift(anchor, marker_r: int, marker_c: int):
    target_r, target_c = _aligned_marker(anchor, int(marker_r), int(marker_c))
    return int(target_r - marker_r), int(target_c - marker_c)


def _path_cells(coords, step_r: int, step_c: int, count: int, shape):
    H, W = shape
    out = set()
    for t in range(1, int(count) + 1):
        for r, c in coords:
            rr = int(r + step_r * t)
            cc = int(c + step_c * t)
            if 0 <= rr < H and 0 <= cc < W:
                out.add((int(rr), int(cc)))
    return out


def _simulate_candidate(canvas: np.ndarray, sprite, anchor, marker_r: int, marker_c: int, bg: int):
    coords = np.array(_sprite_abs_coords(sprite, int(marker_r), int(marker_c), int(bg)), dtype=int)
    desired_dr, desired_dc = _desired_shift(anchor, int(marker_r), int(marker_c))
    sim = _collision_stop_shift(coords, int(desired_dr), int(desired_dc), canvas, int(bg))
    final_marker_r = int(marker_r + sim["dr"])
    final_marker_c = int(marker_c + sim["dc"])
    final_coords = _sprite_abs_coords(sprite, final_marker_r, final_marker_c, int(bg))
    sim["final_coords"] = [(int(r), int(c)) for r, c in final_coords]
    sim["source_coords"] = [(int(r), int(c)) for r, c in coords]
    return sim


def _marker_candidates(canvas: np.ndarray, anchor):
    H, W = canvas.shape
    out = []
    if anchor["axis"] == "h":
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
                if c == int(anchor["c"]):
                    continue
                if abs(int(anchor["c"]) - int(c)) < _MIN_MOVE_STEPS:
                    continue
                out.append((int(r), int(c)))
    else:
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
                if r == int(anchor["r"]):
                    continue
                if abs(int(anchor["r"]) - int(r)) < _MIN_MOVE_STEPS:
                    continue
                out.append((int(r), int(c)))
    random.shuffle(out)
    return out


def _find_placement(inp: np.ndarray, sprite, anchor, bg: int, reserved_path, reserved_final, *, mode: str, blocker_cells=frozenset()):
    blocker_cells = {(int(r), int(c)) for r, c in blocker_cells}
    tries = 0
    for mr, mc in _marker_candidates(inp, anchor):
        if tries >= _PLACEMENT_TRY_LIMIT:
            break
        tries += 1
        if not _can_place_sprite(inp, sprite, int(mr), int(mc), int(bg), forbidden=reserved_path):
            continue

        temp = inp.copy()
        _place_sprite(temp, sprite, int(mr), int(mc), int(bg))
        sim = _simulate_candidate(temp, sprite, anchor, int(mr), int(mc), int(bg))
        final_set = {(int(r), int(c)) for r, c in sim["final_coords"]}
        if final_set & {(int(r), int(c)) for r, c in reserved_final}:
            continue

        if mode == "clear":
            if sim["blocked"] or int(sim["steps"]) != int(sim["target_steps"]):
                continue
            if int(sim["steps"]) < _MIN_MOVE_STEPS:
                continue
            reserve_steps = int(sim["target_steps"])
        else:
            if not sim["blocked"] or int(sim["blocked_by_oob"]):
                continue
            if not blocker_cells.intersection(sim["impact_cells"]):
                continue
            if not (1 <= int(sim["steps"]) < int(sim["target_steps"])):
                continue
            reserve_steps = int(sim["steps"] + 1)

        return {
            "marker": (int(mr), int(mc)),
            "source_coords": [(int(r), int(c)) for r, c in sim["source_coords"]],
            "final_coords": [(int(r), int(c)) for r, c in sim["final_coords"]],
            "reserve_cells": _path_cells(sim["source_coords"], int(sim["step_r"]), int(sim["step_c"]), int(reserve_steps), inp.shape),
            "sim": sim,
        }
    return None


def _new_canvas():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, inp


def _place_anchors(inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != int(bg)], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(inp, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if int(np.count_nonzero(out != inp)) == 0:
        return False
    return bool(np.array_equal(main(inp), out))


def _analyze_combo_behavior(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _anchors, movers = _detect_anchors_and_movers(g, bg)
    static_keep = g.copy()
    static_clear = g.copy()
    for mover in movers:
        coords = mover["coords"]
        static_clear[coords[:, 0], coords[:, 1]] = int(bg)

    copied = 0
    blocked = 0
    source_blocked = 0
    for mover in movers:
        sim_keep = _collision_stop_shift(
            mover["coords"],
            int(mover["desired_dr"]),
            int(mover["desired_dc"]),
            static_keep,
            int(bg),
        )
        sim_clear = _collision_stop_shift(
            mover["coords"],
            int(mover["desired_dr"]),
            int(mover["desired_dc"]),
            static_clear,
            int(bg),
        )
        if int(sim_keep["steps"]) > 0:
            copied += 1
        if bool(sim_keep["blocked"]):
            blocked += 1
        if (int(sim_keep["steps"]), bool(sim_keep["blocked"])) != (int(sim_clear["steps"]), bool(sim_clear["blocked"])):
            source_blocked += 1
    return {"movers": int(len(movers)), "copied": int(copied), "blocked": int(blocked), "source_blocked": int(source_blocked)}


def _place_one_sprite(inp, anchor, marker, bg, used_body, reserved_path, reserved_final, *, mode: str, blocker_cells=frozenset()):
    for _ in range(_SPRITE_BUILD_TRY_LIMIT):
        body = _body_color(bg, marker, used_body)
        sprite = _make_axis_sprite(random.randint(*_BODY_AREA_RANGE), int(body), int(marker), int(bg), anchor["axis"])
        meta = _find_placement(
            inp,
            sprite,
            anchor,
            int(bg),
            reserved_path,
            reserved_final,
            mode=mode,
            blocker_cells=blocker_cells,
        )
        if meta is None:
            continue
        used_body.add(int(body))
        _place_sprite(inp, sprite, meta["marker"][0], meta["marker"][1], int(bg))
        return {
            "sprite": sprite,
            "body": int(body),
            "anchor": anchor,
            "marker_color": int(marker),
            "marker": meta["marker"],
            "source_coords": meta["source_coords"],
            "final_coords": meta["final_coords"],
            "reserve_cells": meta["reserve_cells"],
            "sim": meta["sim"],
        }
    return None


def _gen_attempt():
    bg, inp = _new_canvas()
    placed = _place_anchors(inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed
    by_color = {a["color"]: a for a in anchors}
    marker_colors = _marker_color_sequence(anchor_colors)
    if len(marker_colors) < 3:
        return None

    used_body = set()
    reserved_path = set()
    reserved_final = set()
    placed_movers = []

    blocker_marker = marker_colors[0]
    blocker_anchor = by_color[int(blocker_marker)]
    blocker = _place_one_sprite(
        inp,
        blocker_anchor,
        int(blocker_marker),
        int(bg),
        used_body,
        reserved_path,
        reserved_final,
        mode="clear",
    )
    if blocker is None:
        return None
    placed_movers.append(blocker)
    reserved_path |= blocker["reserve_cells"]
    reserved_final |= {(int(r), int(c)) for r, c in blocker["final_coords"]}
    blocker_cells = {(int(r), int(c)) for r, c in blocker["source_coords"]}

    blocked_marker = marker_colors[1]
    blocked_anchor = by_color[int(blocked_marker)]
    blocked = _place_one_sprite(
        inp,
        blocked_anchor,
        int(blocked_marker),
        int(bg),
        used_body,
        reserved_path,
        reserved_final,
        mode="blocked",
        blocker_cells=blocker_cells,
    )
    if blocked is None:
        return None
    placed_movers.append(blocked)
    reserved_path |= blocked["reserve_cells"]
    reserved_final |= {(int(r), int(c)) for r, c in blocked["final_coords"]}

    for marker in marker_colors[2:]:
        anchor = by_color[int(marker)]
        mover = _place_one_sprite(
            inp,
            anchor,
            int(marker),
            int(bg),
            used_body,
            reserved_path,
            reserved_final,
            mode="clear",
        )
        if mover is None:
            return None
        placed_movers.append(mover)
        reserved_path |= mover["reserve_cells"]
        reserved_final |= {(int(r), int(c)) for r, c in mover["final_coords"]}

    out = main(inp)
    stats = _analyze_combo_behavior(inp)
    if stats["source_blocked"] < 1:
        return None
    if stats["copied"] < 2:
        return None
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
