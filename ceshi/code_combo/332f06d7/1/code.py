# Combo variant of 332f06d7 (index 1)
# combo_spec: D1/1+D16/2
# Name: 对角反射+镜面障碍
#
# Design analysis
# a) Concrete behavior and trace
#    - 保留原题的蓝色河道、黑色 mover、红色 destination、绿色背景，并新增黄色镜面条。
#    - 原题把蓝色区域看成可 BFS 的宽走廊；只要相邻 n×n 放置窗口可行，黑块就能沿 4 邻域继续前进。
#    - 融合后，蓝色不再代表“任意可拐弯通道”，而是若干对角线 n×n 放置状态的并集。黑块从起点沿唯一的
#      45 度方向滑动；每次落到一个拐点状态后，只有当其前进方向同侧紧贴黄色镜面条时，才会翻转一个轴并继续。
#    - 小例子：若当前方向是 (1, 1)，下一段规划方向应改成 (-1, 1)，则在该拐点的“下边界”外放一条水平镜面。
#      黑块落到此点后检测到下侧镜面，于是 dr 取反，下一步变成 (-1, 1)。如果缺这条镜面，黑块不会主动拐弯，
#      因为沿旧方向 (1, 1) 的下一对角状态并不存在，于是它停在该拐点。
#    - 这与原题的差异是决定性地把“位置”改成了“位置 + 对角方向”，并把“通道自己允许拐弯”改成了“镜面显式触发反射”。
#
# b) Complexity calibration
#    - 每个样例规划 2-3 次反射，保证“多次反弹”在视觉上真实存在，而不是一次性的小修饰。
#    - 不可达样例不会引入额外红色诱饵分支，而是在同一条主轨迹中故意省略一个中后段必需镜面；这样样例同时展示
#      “镜面存在时确实会改变方向”与“镜面缺失时会提前停下”。
#    - 颜色总数固定为 5，镜面只有一种颜色，避免多颜色编码带来的归纳负担。
#
# c) Implicit assumptions review against the original main()
#    - 原题通过 `find_components` 找两个同尺寸方块并按颜色排序；这里不再依赖不可见的颜色排序，直接按可见角色
#      黑色= mover、红色= destination 读取。
#    - 原题的 `trav` + 4 邻域 BFS 默认任何相邻可行窗口都可达，这依赖“方块会自动沿走廊拐弯”的旧前提。
#      融合后该前提失效，因为河道只提供对角状态，且是否变向取决于镜面条。
#    - 原题“不可达时选 BFS 最远点”也失效；这里真正的停点是沿单条确定性轨迹模拟时第一次无法继续反射的位置。
#    - 新 solver 的隐式假设包括：起点恰有唯一对角出边；每个有效拐点至多放一类镜面条；镜面不覆盖河道。
#      生成器通过路径规划、精确状态集校验和镜面落位检查来保证这些前提成立。
#
# d) Placement legality
#    - 生成器先在放置坐标系中规划离散对角状态，再把每个状态渲染成 n×n 蓝块。
#    - 非相邻旧状态之间禁止切比雪夫距离 <= 1，避免渲染后产生计划外可行窗口。
#    - 镜面条总是贴着拐点状态的外侧边界放置，且必须完全落在背景上，不与蓝河、黑块、红块重叠。
#
# e) Visual observability
#    - 人眼可以直接看到：蓝色轨迹是对角折返链，黄色条总贴在拐点外侧；有镜面时黑块会折返，没有镜面时会在该处停下。
#    - 这不依赖颜色大小关系或隐藏编号，只依赖可见的几何关系与镜面存在性。
#
# f) Visual cleanliness
#    - 轨迹只使用一条主链，不叠多层分支；镜面条永远在蓝轨迹之外，避免元素互相遮挡。
#    - 角点和镜面是一一对应的，阅读负担稳定。
#
# g) Behavior non-degeneration
#    - 旧题的 4 邻域 BFS 在这些样例上通常一步都走不出去，因为可行状态只沿对角相连；即便忽略这一点，
#      是否反射仍由黄色镜面决定。因此样例不会退化回原始“宽走廊滑块”任务。

from __future__ import annotations

from collections import Counter
import random

import numpy as np


_BG, _RIVER, _MOVER, _DEST, _MIRROR = 3, 1, 0, 2, 4


def _find_solid_square(grid: np.ndarray, color: int) -> tuple[int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    n = int(r1 - r0 + 1)
    assert n == int(c1 - c0 + 1)
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert bool((patch == int(color)).all())
    assert int(coords.shape[0]) == int(patch.size)
    return int(r0), int(c0), int(n)


def _feasible_mask(grid: np.ndarray, n: int) -> np.ndarray:
    H, W = grid.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    trav = (grid == int(_RIVER)) | (grid == int(_MOVER)) | (grid == int(_DEST))
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _diag_dirs() -> list[tuple[int, int]]:
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _unique_start_dir(feasible: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    h2, w2 = feasible.shape
    dirs: list[tuple[int, int]] = []
    for dr, dc in _diag_dirs():
        rr, cc = sr + dr, sc + dc
        if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]):
            dirs.append((int(dr), int(dc)))
    assert len(dirs) == 1
    return dirs[0]


def _touch_row_mirror(grid: np.ndarray, r: int, c: int, n: int, dr: int) -> bool:
    if dr == 1:
        rr = r + n
        return rr < grid.shape[0] and bool((grid[rr, c : c + n] == int(_MIRROR)).all())
    rr = r - 1
    return rr >= 0 and bool((grid[rr, c : c + n] == int(_MIRROR)).all())


def _touch_col_mirror(grid: np.ndarray, r: int, c: int, n: int, dc: int) -> bool:
    if dc == 1:
        cc = c + n
        return cc < grid.shape[1] and bool((grid[r : r + n, cc] == int(_MIRROR)).all())
    cc = c - 1
    return cc >= 0 and bool((grid[r : r + n, cc] == int(_MIRROR)).all())


def _simulate(
    grid: np.ndarray,
    n: int,
    sr: int,
    sc: int,
    dr: int,
    dc: int,
    dest: tuple[int, int],
) -> dict:
    feasible = _feasible_mask(grid, n)
    h2, w2 = feasible.shape
    assert bool(feasible[sr, sc])
    cur_r, cur_c = int(sr), int(sc)
    reflections = 0
    seen = {(int(cur_r), int(cur_c), int(dr), int(dc))}

    while True:
        nr, nc = int(cur_r + dr), int(cur_c + dc)
        if not (0 <= nr < h2 and 0 <= nc < w2 and bool(feasible[nr, nc])):
            return {"pos": (int(cur_r), int(cur_c)), "reached": (int(cur_r), int(cur_c)) == dest, "reflections": int(reflections)}

        cur_r, cur_c = int(nr), int(nc)
        if (int(cur_r), int(cur_c)) == (int(dest[0]), int(dest[1])):
            return {"pos": (int(cur_r), int(cur_c)), "reached": True, "reflections": int(reflections)}

        row_hit = _touch_row_mirror(grid, cur_r, cur_c, n, dr)
        col_hit = _touch_col_mirror(grid, cur_r, cur_c, n, dc)
        if row_hit:
            dr = -int(dr)
        if col_hit:
            dc = -int(dc)
        if row_hit or col_hit:
            reflections += 1

        state = (int(cur_r), int(cur_c), int(dr), int(dc))
        if state in seen:
            return {"pos": (int(cur_r), int(cur_c)), "reached": (int(cur_r), int(cur_c)) == dest, "reflections": int(reflections)}
        seen.add(state)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the black square along a diagonal river and reflect only off explicit mirror strips.

    Rule:
    - The mover starts in the unique diagonal direction allowed by the blue placement chain.
    - It advances one diagonal placement at a time.
    - After landing, a yellow strip flush against the active row side flips `dr`; a strip flush against the
      active column side flips `dc`. With no needed mirror, the mover does not turn and will stop at the
      last reachable diagonal placement.
    - The starting square is restored to blue.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    assert H == W

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    assert bg == int(_BG)

    sr, sc, n = _find_solid_square(g, _MOVER)
    dr0, dc0, n2 = _find_solid_square(g, _DEST)
    assert n == n2

    feasible = _feasible_mask(g, n)
    assert bool(feasible[sr, sc])
    init_dr, init_dc = _unique_start_dir(feasible, sr, sc)
    trace = _simulate(g, n, sr, sc, init_dr, init_dc, (dr0, dc0))
    tr, tc = trace["pos"]

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(_RIVER)
    out[tr : tr + n, tc : tc + n] = int(_MOVER)
    return out


def _candidate_starts(h2: int, w2: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for c in range(1, w2 - 1):
        out.append((0, c, 1, random.choice([-1, 1])))
        out.append((h2 - 1, c, -1, random.choice([-1, 1])))
    for r in range(1, h2 - 1):
        out.append((r, 0, random.choice([-1, 1]), 1))
        out.append((r, w2 - 1, random.choice([-1, 1]), -1))
    random.shuffle(out)
    return out


def _too_close(p: tuple[int, int], states: list[tuple[int, int]]) -> bool:
    if not states:
        return False
    prev = states[-1]
    for q in states[:-1]:
        if max(abs(int(p[0]) - int(q[0])), abs(int(p[1]) - int(q[1]))) <= 1:
            return True
    return p == prev


def _walk_segment(
    states: list[tuple[int, int]],
    used: set[tuple[int, int]],
    cur: tuple[int, int],
    direction: tuple[int, int],
    seg_len: int,
    h2: int,
    w2: int,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    added: list[tuple[int, int]] = []
    rr, cc = cur
    for _ in range(seg_len):
        rr += int(dr)
        cc += int(dc)
        p = (int(rr), int(cc))
        if not (0 <= p[0] < h2 and 0 <= p[1] < w2):
            return None
        if p in used or _too_close(p, states + added):
            return None
        added.append(p)
    return added


def _bounce_options(cur_dir: tuple[int, int]) -> list[tuple[int, int, str]]:
    dr, dc = cur_dir
    return [(-int(dr), int(dc), "row"), (int(dr), -int(dc), "col")]


def _plan_main_states(h2: int, w2: int) -> tuple[list[tuple[int, int]], list[tuple[int, str]], tuple[int, int]] | None:
    n_bounces = random.choices([2, 3], weights=[3, 2])[0]
    min_seg = 2
    max_seg = 4 if min(h2, w2) < 10 else 5

    for sr, sc, dr0, dc0 in _candidate_starts(h2, w2)[:80]:
        states = [(int(sr), int(sc))]
        used = {(int(sr), int(sc))}
        cur = (int(sr), int(sc))
        cur_dir = (int(dr0), int(dc0))
        bounces: list[tuple[int, str]] = []
        ok = True

        for seg_idx in range(n_bounces + 1):
            remaining = n_bounces - seg_idx
            options: list[tuple[list[tuple[int, int]], tuple[int, int, str] | None]] = []
            for seg_len in range(max_seg, min_seg - 1, -1):
                added = _walk_segment(states, used, cur, cur_dir, seg_len, h2, w2)
                if added is None:
                    continue
                endpoint = added[-1]
                if remaining == 0:
                    options.append((added, None))
                    continue
                future_states = states + added
                for ndr, ndc, axis in _bounce_options(cur_dir):
                    nxt = (int(endpoint[0] + ndr), int(endpoint[1] + ndc))
                    if not (0 <= nxt[0] < h2 and 0 <= nxt[1] < w2):
                        continue
                    if nxt in used or nxt in set(added) or _too_close(nxt, future_states):
                        continue
                    options.append((added, (int(ndr), int(ndc), axis)))
            if not options:
                ok = False
                break

            added, bounce_choice = random.choice(options)
            for p in added:
                states.append(p)
                used.add(p)
            cur = states[-1]
            if bounce_choice is not None:
                ndr, ndc, axis = bounce_choice
                bounces.append((len(states) - 1, axis))
                cur_dir = (int(ndr), int(ndc))

        if ok and len(states) >= 9 and len(bounces) >= 2:
            return states, bounces, (int(dr0), int(dc0))
    return None


def _paint_states(grid: np.ndarray, states: list[tuple[int, int]], n: int, color: int) -> None:
    for r, c in states:
        grid[r : r + n, c : c + n] = int(color)


def _mirror_cells_at(
    state: tuple[int, int],
    n: int,
    incoming_dir: tuple[int, int],
    axis: str,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    r, c = state
    dr, dc = incoming_dir
    cells: list[tuple[int, int]] = []
    if axis == "row":
        rr = r + n if dr == 1 else r - 1
        if 0 <= rr < H:
            for cc in range(c, c + n):
                if 0 <= cc < W:
                    cells.append((int(rr), int(cc)))
    else:
        cc = c + n if dc == 1 else c - 1
        if 0 <= cc < W:
            for rr in range(r, r + n):
                if 0 <= rr < H:
                    cells.append((int(rr), int(cc)))
    return cells


def _incoming_dirs(states: list[tuple[int, int]]) -> list[tuple[int, int] | None]:
    inc: list[tuple[int, int] | None] = [None]
    for i in range(1, len(states)):
        pr, pc = states[i - 1]
        r, c = states[i]
        inc.append((int(r - pr), int(c - pc)))
    return inc


def _feasible_state_set(grid: np.ndarray, n: int) -> set[tuple[int, int]]:
    feasible = _feasible_mask(grid, n)
    return {(int(r), int(c)) for r, c in np.argwhere(feasible)}


def _expected_output(grid: np.ndarray, start: tuple[int, int], final_pos: tuple[int, int], n: int) -> np.ndarray:
    sr, sc = start
    tr, tc = final_pos
    out = np.array(grid, copy=True)
    out[sr : sr + n, sc : sc + n] = int(_RIVER)
    out[tr : tr + n, tc : tc + n] = int(_MOVER)
    return out


def _size_and_grid() -> tuple[int, int]:
    n = random.choice([1, 2, 3, 4])
    if n == 1:
        G = random.randint(10, 15)
    elif n == 2:
        G = random.randint(12, 17)
    elif n == 3:
        G = random.randint(14, 19)
    else:
        G = random.randint(17, 20)
    return n, G


def _gen_attempt() -> np.ndarray | None:
    reachable = random.random() < 0.55
    n, G = _size_and_grid()
    H = W = G
    h2 = H - n + 1
    w2 = W - n + 1
    if min(h2, w2) < 8:
        return None

    planned = _plan_main_states(h2, w2)
    if planned is None:
        return None
    main_states, bounces, init_dir = planned

    grid = np.full((H, W), int(_BG), dtype=int)
    _paint_states(grid, main_states, n, _RIVER)

    omit_turn_idx = None
    if not reachable:
        omit_candidates = [idx for idx, _axis in bounces[1:]] if len(bounces) > 1 else [bounces[0][0]]
        omit_turn_idx = random.choice(omit_candidates)

    inc = _incoming_dirs(main_states)
    mirrors_added = 0
    mirrors_before_omit = 0
    for idx, axis in bounces:
        incoming = inc[idx]
        assert incoming is not None
        cells = _mirror_cells_at(main_states[idx], n, incoming, axis, grid.shape)
        if not cells:
            return None
        if any(grid[r, c] != int(_BG) for r, c in cells):
            return None
        if idx == omit_turn_idx:
            continue
        for r, c in cells:
            grid[r, c] = int(_MIRROR)
        mirrors_added += 1
        if omit_turn_idx is None or idx < int(omit_turn_idx):
            mirrors_before_omit += 1

    if mirrors_added < 1:
        return None
    if reachable and mirrors_added < 2:
        return None

    sr, sc = main_states[0]
    dest_r, dest_c = main_states[-1]
    grid[sr : sr + n, sc : sc + n] = int(_MOVER)
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = int(_DEST)

    if Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0] != int(_BG):
        return None

    feasible_states = _feasible_state_set(grid, n)
    intended_states = set(main_states)
    if feasible_states != intended_states:
        return None

    try:
        got_dir = _unique_start_dir(_feasible_mask(grid, n), sr, sc)
    except AssertionError:
        return None
    if got_dir != init_dir:
        return None

    trace = _simulate(grid, n, sr, sc, got_dir[0], got_dir[1], (dest_r, dest_c))
    if reachable:
        if not trace["reached"] or trace["pos"] != (dest_r, dest_c):
            return None
        if trace["reflections"] != len(bounces):
            return None
        final_pos = (dest_r, dest_c)
    else:
        assert omit_turn_idx is not None
        final_pos = main_states[omit_turn_idx]
        if trace["reached"] or trace["pos"] != final_pos:
            return None
        if trace["reflections"] != mirrors_before_omit:
            return None

    out = _expected_output(grid, (sr, sc), final_pos, n)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
