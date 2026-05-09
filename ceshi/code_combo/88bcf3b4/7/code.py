# Combo variant of 88bcf3b4 (index 7)
# combo_spec: D7/1+D14/0
# Name: 同色识别颜色转移

"""
Design analysis for the D7/1 + D14/0 combo

a) 具体化与行为追踪
- 原题里，bar、输入线段、障碍物三者颜色不同；solver 先靠颜色把 bar 和线段拆开，再让线段绕最近障碍物重定位，最后仍用原线段色重绘。
- 本融合版把 bar 和输入线段合并成同色 rope。输入中只能通过形状识别：从边界出发的笔直前缀是 bar，内端接着的弯曲后缀才是可移动 segment。
- 绕行几何保持原题语义：segment 从 terminal 脱离，靠近最近障碍物，贴着障碍物外皮转向，再向前对角离开。
- 新增的颜色转移发生在输出阶段：旧 segment 清空，新路径不刷 rope 原色，而是刷“它所绕过的那个障碍物”的颜色。
- 具体例子：若一条蓝色 rope 的 straight prefix 贴着上边界向下，terminal 右侧接同样蓝色的弯折后缀；最近障碍物是红色小块。原题输出会得到蓝色绕行线，本题输出则得到红色绕行线。差异既体现在“输入端不能靠颜色分角色”，也体现在“输出端颜色来自障碍物”。

b) 复杂度校准
- 若每个样本只有一个 triple，颜色转移仍成立，但人类较难确认“为什么输出变色”。因此 generator 在网格足够大时偏向 2-3 个 triple，让同一张图中出现多个障碍物颜色与多个对应输出。
- 若允许 rope 每组都用不同颜色，解题者可能把“颜色变化”误读成逐组独立替换。这里统一整张图的 rope 颜色，只让障碍物颜色按 triple 变化，能强化“输出颜色跟 obstacle 绑定”。
- 若障碍物颜色过多且无重复支撑，视觉会乱。这里上限仍是 3 个 triple，因此同图至多 3 个 obstacle/output 颜色，规模可读。

c) 隐式假设审查
- 原 solver 假设 `bar_color != seg_color`，因此能先找直线 bar，再在 terminal 旁找异色 segment。融合后这个前提失效，必须改成对每个非背景 8 连通分量做几何分类。
- 原 walkable 逻辑按 `seg_col` 放行；融合后整条 rope 同色，如果直接按颜色放行会误把别的 rope 也当成可走区域。因此新 solver 只把当前 segment 后缀的像素集 `seg_set` 视作可通行旧路径。
- 原 repaint 默认 `new_pts` 使用 `seg_col`。融合后必须把最近障碍物的颜色一路带到 `moves`，最终按 `obstacle_color` 重绘。
- 原 obstacle 过滤常依赖“排除 bar 色与 seg 色”。融合后 obstacle 直接定义为“未通过 rope 几何分类的非背景分量”，不再依赖颜色关系。

d) 放置合法性
- 继续采用 output-first：先构造合法的绕行输出几何与障碍物，再反推输入 rope 的弯曲后缀。
- 因为 bar 与输入 segment 同色，generator 必须保证 rope 整体仍是单像素粗细简单路径：恰有两个端点、仅一个端点在边界、没有 2x2 块、没有三叉。为此输入后缀强制先走一个对角步，并用与 solver 相同的分类器复验。
- triple 之间仍使用 padding 和最近障碍物校验，避免某个 terminal 更靠近别组 obstacle，或者同色 rope/同色 obstacle 在输入中误连。

e) 视觉可观察性
- 输入图能直接看出：一条同色 rope 从边界伸入，旁边有异色 obstacle；输出图里 rope 的弯曲后缀消失，障碍物边上长出一条同 obstacle 色的新路径。
- 当同一图中有 2-3 个 triple 时，可以横向对比出：虽然所有输入 rope 颜色相同，但每条输出路径分别继承自己附近障碍物的颜色。
- 全部规则只依赖可见结构与邻近关系，不依赖颜色编号大小、扫描顺序等不可见因素。

f) 可视化整洁度
- 输入端只有两类非背景角色：rope 色与 obstacle 色，层次比原题更简洁。
- 输出端 segment 与 obstacle 同色并可能连成一个 8 连通团块，但这不会遮蔽规则，因为 rope 的 straight bar 仍保留原色，且新路径在几何上是贴着 obstacle 外皮延伸出去的细线。
- 生成器保留 obstacle 大小、距离和三相绕行路径约束，避免输出变成大面积混色团。

g) 变化的可推断性
- 颜色变化只有一种：`output segment color = nearest obstacle color`。
- 结构提示足够明确：每条新路径都绕着某个具体 obstacle，且颜色恰与该 obstacle 相同；对应关系由空间邻近和包裹关系给出。
- 为避免退化回 D14-only，generator 保证每个 obstacle 颜色都不同于 rope 颜色，因此输出一定发生颜色转移。

h) 新隐式假设与生成保证
- 假设 1：每条 rope 都能被解析成“边界直线前缀 + 内侧弯曲后缀”。generator 用 `_classify_bar_segment_component` 逐组校验。
- 假设 2：每个 terminal 的最近 obstacle 就是它自己的 obstacle。generator 保留 own-vs-other 最近距离检查。
- 假设 3：绕行路径不会跨过别的物体。output-first 构造、placement padding 和 oracle `main(inp) == out` 共同保证。
- 假设 4：输入中 obstacle 不是另一条合法 rope。generator 让 obstacle 从 V apex 内侧生长，通常不接触边界，也不满足 rope 的 simple-path 几何约束。
"""

from __future__ import annotations

import random
import re
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 1400
PER_EDGE_RETRIES = 40
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _bfs_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _bfs_path_with_diag_filter(
    start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}

    def can_step(r: int, c: int, nr: int, nc: int) -> bool:
        dr = nr - r
        dc = nc - c
        if dr == 0 or dc == 0:
            return True
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if 0 <= mid1[0] < h and 0 <= mid1[1] < w and bool(walkable[mid1]):
            return False
        if 0 <= mid2[0] < h and 0 <= mid2[1] < w and bool(walkable[mid2]):
            return False
        return True

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            if not can_step(r, c, nr, nc):
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _greedy_manhattan_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape

    def md(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    cur = start
    path = [cur]
    seen = {cur}
    while cur != goal:
        cur_md = md(cur)
        best = None
        for nr, nc in neighbors8(cur[0], cur[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            nmd = md(nxt)
            if nmd >= cur_md:
                continue
            is_diag = 1 if (nr != cur[0] and nc != cur[1]) else 0
            key = (nmd, is_diag, nr, nc)
            if best is None or key < best:
                best = key
        if best is None:
            return None
        _, _, nr, nc = best
        cur = (nr, nc)
        path.append(cur)
        seen.add(cur)
    return path


def _stroke_neighbors(p: tuple[int, int], sset: set[tuple[int, int]]):
    r, c = p
    for q in neighbors4(r, c):
        if q in sset:
            yield q
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        q = (r + dr, c + dc)
        if q not in sset:
            continue
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if mid1 in sset or mid2 in sset:
            continue
        yield q


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in _stroke_neighbors(cur, sset):
            if nxt in prev:
                continue
            prev[nxt] = cur
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    far = max(dist.keys(), key=lambda p: (dist[p], p[0], p[1]))
    path = []
    cur = far
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _segment_degrees(seg_pts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    sset = set(seg_pts)
    return {p: sum(1 for _ in _stroke_neighbors(p, sset)) for p in sset}


def _allowed_border_fwds(p: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    r, c = p
    out = set()
    if r == 0:
        out.add((1, 0))
    if r == h - 1:
        out.add((-1, 0))
    if c == 0:
        out.add((0, 1))
    if c == w - 1:
        out.add((0, -1))
    return out


def _classify_bar_segment_component(
    comp: list[tuple[int, int]], h: int, w: int
) -> dict[str, object] | None:
    if len(comp) < BAR_LEN_MIN + SEG_LEN_MIN:
        return None

    deg = _segment_degrees(comp)
    if any(d < 1 or d > 2 for d in deg.values()):
        return None

    endpoints = sorted([p for p, d in deg.items() if d == 1])
    if len(endpoints) != 2:
        return None

    border_eps = [p for p in endpoints if _touches_border(p, h, w)]
    if len(border_eps) != 1:
        return None

    border_ep = border_eps[0]
    path = _segment_main_path(comp, border_ep)
    if len(path) != len(comp) or set(path) != set(comp):
        return None
    if len(path) < 2:
        return None

    fwd = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    if fwd not in _allowed_border_fwds(border_ep, h, w):
        return None
    if abs(fwd[0]) + abs(fwd[1]) != 1:
        return None

    terminal_idx = 1
    while terminal_idx < len(path):
        step = (path[terminal_idx][0] - path[terminal_idx - 1][0], path[terminal_idx][1] - path[terminal_idx - 1][1])
        if step != fwd:
            break
        terminal_idx += 1
    terminal_idx -= 1
    if terminal_idx + 1 >= len(path):
        return None

    bar_comp = path[: terminal_idx + 1]
    seg_comp = path[terminal_idx + 1 :]
    if len(bar_comp) < BAR_LEN_MIN or len(seg_comp) < SEG_LEN_MIN:
        return None

    attach_step = (seg_comp[0][0] - bar_comp[-1][0], seg_comp[0][1] - bar_comp[-1][1])
    if abs(attach_step[0]) + abs(attach_step[1]) != 1:
        return None
    if attach_step[0] * fwd[0] + attach_step[1] * fwd[1] != 0:
        return None

    return {
        "path": path,
        "bar_comp": bar_comp,
        "seg_comp": seg_comp,
        "seg_set": set(seg_comp),
        "terminal": bar_comp[-1],
        "seg_start": seg_comp[0],
        "fwd": fwd,
        "color": None,
    }


def _classify_scene_components(grid: np.ndarray) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

    ropes: list[dict[str, object]] = []
    obstacles: list[dict[str, object]] = []
    for col in colors:
        for comp in _components(g, col, conn8=True):
            item = {"color": col, "pts": comp, "set": set(comp)}
            parsed = _classify_bar_segment_component(comp, h, w)
            if parsed is None:
                obstacles.append(item)
            else:
                parsed["color"] = col
                parsed["full_comp"] = comp
                parsed["full_set"] = set(comp)
                ropes.append(parsed)
    return bg, ropes, obstacles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing the irregular suffix of each same-color rope around the nearest obstacle and recolor the relocated path.

    Input structure:
    - Background is the most frequent color.
    - Each movable object is a same-color rope: a straight border-anchored bar followed by an irregular 1-cell-thick suffix of the same color.
    - Nearby obstacle sprites use other colors and stay fixed.

    Rule:
    - Identify each rope by shape, not by color: the straight prefix touching the border is the bar, and the remaining suffix is the segment.
    - Remove the original suffix, reroute it toward the nearest obstacle, trace along the obstacle skin, and preserve the suffix pixel count.
    - Repaint the rerouted suffix with the obstacle's color instead of the rope's original color; bars and obstacles otherwise remain unchanged.

    Key insight:
    - The hard part is separating bar from segment when they share one color. Once the rope is split geometrically, the relocation geometry is the original whip-around-obstacle rule, but the output color transfers from the wrapped obstacle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, rope_infos, obstacle_infos = _classify_scene_components(g)

    out = g.copy()
    moves: list[tuple[list[tuple[int, int]], int, list[tuple[int, int]]]] = []
    for info in sorted(rope_infos, key=lambda x: (-len(x["bar_comp"]), x["color"], x["terminal"])):
        bar_comp = info["bar_comp"]
        seg_comp = info["seg_comp"]
        seg_set = info["seg_set"]
        terminal = info["terminal"]
        seg_start = info["seg_start"]
        fwd = info["fwd"]
        L = len(seg_comp)

        best = None
        obstacle_set = None
        obstacle_col = None
        for obstacle in obstacle_infos:
            comp = obstacle["pts"]
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            key = (d, mr, mc)
            if best is None or key < best:
                best = key
                obstacle_set = obstacle["set"]
                obstacle_col = int(obstacle["color"])
        assert obstacle_set is not None and obstacle_col is not None, "must find at least one obstacle component"

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]

        walkable = g == bg
        walkable[terminal] = True
        for p in seg_set:
            walkable[p] = True

        def current_seg_walkable(p: tuple[int, int]) -> bool:
            return int(g[p]) == bg or p in seg_set

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if not current_seg_walkable((er, ec)):
                    continue
                d = abs(pr - terminal[0]) + abs(pc - terminal[1])
                good.append((d, pr, pc))
            assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"
            _, pr, pc = min(good)
            return (pr, pc)

        p_near = pick_p_near()
        walkable[p_near] = True
        path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
        assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"

        shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted segment path must stay within grid"
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - terminal[0], cc - terminal[1])

        def skin_ok(p: tuple[int, int]) -> bool:
            return current_seg_walkable(p)

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)

        skin_walkable = np.zeros((h, w), dtype=bool)
        for p in skin:
            if skin_ok(p):
                skin_walkable[p] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]
        assert len(boundary_path) >= 2, "boundary path must have at least 2 cells to derive extension direction"

        dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
        dc_ext = boundary_path[-1][1] - boundary_path[-2][1]

        route = list(shifted)
        if boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if not current_seg_walkable(nxt):
                break
            route.append(nxt)
            cur = nxt

        new_pts = route[:L]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
        assert not any(p in set(bar_comp) for p in new_pts), "output segment overlaps the bar"

        step_classes = []
        for j in range(len(new_pts) - 1):
            dr = new_pts[j + 1][0] - new_pts[j][0]
            dc = new_pts[j + 1][1] - new_pts[j][1]
            fwd_comp = dr * fwd[0] + dc * fwd[1]
            perp_comp = dr * fwd[1] + dc * (-fwd[0])
            if abs(dr) + abs(dc) == 1:
                step_classes.append("F" if fwd_comp > 0 else "X")
            elif fwd_comp > 0 and perp_comp > 0:
                step_classes.append("FR")
            elif fwd_comp > 0 and perp_comp < 0:
                step_classes.append("FL")
            else:
                step_classes.append("X")
        s = "".join(step_classes)
        assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
            f"output segment not monotone wrap: {' '.join(step_classes)}"
        )

        first_step = step_classes[0] if step_classes else ""
        if first_step in ("FL", "FR"):
            p1_end = 0
            for j2, sc in enumerate(step_classes):
                if sc == first_step:
                    p1_end = j2 + 1
                else:
                    break
            apex = new_pts[p1_end]
            perp_dir = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
            inner = (-perp_dir[0], -perp_dir[1])
            apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
            assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"

        moves.append((seg_comp, obstacle_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, paint_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = paint_col
    return out


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int
) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _grow_obstacle(
    start: tuple[int, int], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    # The first forced diagonal keeps the same-color rope a single simple path after
    # it attaches perpendicularly to the bar terminal.
    coords = [(0, side), (-1, 2 * side)]
    remaining = seg_len - 2
    diag_step = (-1, side)
    orth_step = (-1, 0)
    last_was_diag = True
    turns_left = turns
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            run = remaining
        else:
            if last_was_diag:
                is_diag = random.random() < DIAG_CONTINUE_PROB
            else:
                is_diag = random.random() < DIAG_START_PROB
            step = diag_step if is_diag else orth_step
            if is_diag != last_was_diag:
                turns_left -= 1
            max_run = remaining - turns_left
            run = random.randint(1, max(1, min(max_run, remaining)))
        last_was_diag = step == diag_step
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _build_output_path(
    seg_len: int, side: int, p1: int, p2: int, p3: int
) -> list[tuple[int, int]]:
    coords = [(0, side)]
    phase_steps = [(1, side), (1, 0), (1, -side)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    return coords


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _make_triple(n: int) -> dict[str, object] | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(1, min(remaining - 1, PHASE_MAX))
    remaining -= p1
    p2 = random.randint(0, min(remaining, PHASE_MAX))
    p3 = remaining - p2

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set)

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _place_triple(
    triple: dict[str, object], edge: str, n: int, reserved: np.ndarray
) -> dict[str, object] | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    bar_len = triple["bar_len"]

    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        placed = {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
        if _classify_bar_segment_component(placed["bar"] + placed["input_seg"], n, n) is None:
            continue
        return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    # Bias accepted samples toward 2-3 triples so the same-color parsing and
    # obstacle-color transfer can be compared side-by-side within one board.
    triple_count = random.choices([2, 3], weights=[4, 1], k=1)[0]
    min_n = {2: 12, 3: 18}[triple_count]
    n = random.randint(min_n, GRID_SIZE_MAX)

    palette = random.sample(range(10), triple_count + 2)
    bg = palette[0]
    rope_color = palette[1]
    obstacle_colors = palette[2:]

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples: list[tuple[dict[str, object], int]] = []

    for edge, obstacle_color in zip(random.sample(list(EDGE_DATA), triple_count), obstacle_colors):
        placed = None
        for _ in range(PER_EDGE_RETRIES):
            triple = _make_triple(n)
            if triple is None:
                continue
            placed = _place_triple(triple, edge, n, reserved)
            if placed is not None:
                break
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], rope_color)
        paint_coords(inp, placed["input_seg"], rope_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)

        paint_coords(out, placed["bar"], rope_color)
        paint_coords(out, placed["output_seg"], obstacle_color)
        paint_coords(out, placed["obstacle"], obstacle_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append((placed, obstacle_color))

    if len(placed_triples) > 1:
        for i, (placed, _) in enumerate(placed_triples):
            tr, tc = placed["terminal"]
            own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
            other = min(
                abs(r - tr) + abs(c - tc)
                for j, (other_placed, _) in enumerate(placed_triples)
                if j != i
                for r, c in other_placed["obstacle"]
            )
            if own >= other:
                return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
