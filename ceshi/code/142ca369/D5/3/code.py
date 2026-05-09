# Variant of 142ca369
# Dimension: D5 — 射线源对象形状 (src)
# Change: L形三角箭头——2x2外接矩形中缺少一个角的3像素L形，缺失角的方向决定射线发射的对角线方向
#   → 箭头仍保留该3像素L核心，但额外增加1或2个同色像素形成4/5像素变体；额外像素的位置编码该射线
#   的最大反射次数。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


"""
Design analysis for the D5 variant
==================================

a) Concrete behavior and execution trace
----------------------------------------
- 原题输入中的源对象是 3 像素 L，输出里只是从其直角顶点发出一条对角射线。
- 本变体把源对象扩大成 4 或 5 像素，但仍能一眼看出其核心是原来的 2x2-L：
    base core:
      S     = 直角顶点 / 发射起点
      V     = 竖臂像素 at (S-dr, S)
      H     = 横臂像素 at (S, S-dc)
      missing corner = (S-dr, S-dc), 仍然决定射线向 (dr,dc) 发射
    extra tags:
      VV at (S-2*dr, S)          -> 最大反射次数 = 1
      HH at (S, S-2*dc)          -> 最大反射次数 = 2
      VV + HH together           -> 最大反射次数 = 3
- 小例子：
    若箭头朝右下 (dr=1, dc=1)，核心是 {(r,c), (r-1,c), (r,c-1)}。
    额外加 (r-2,c) 表示这条射线最多反射 1 次；加 (r,c-2) 表示最多反射 2 次；
    两个都加表示最多反射 3 次。
    原题在这里会把“源对象能否反射几次”完全交给墙的布局；变体中则先从箭头形状读取一个参数，
    再按该参数追踪射线。

b) Complexity calibration
-------------------------
- 只保留 3 档参数：1 / 2 / 3 次反射。档位足够少，2-4 个样例里能被人类归纳；
  同时三种形状在视觉上明显不同：加长竖臂、加长横臂、双加长。
- 不再使用原始 3 像素箭头，避免新旧源对象混杂造成匹配歧义。
- 生成器保证每个样例至少同时出现三种编码中的三种或其中多数，并且反射次数与编码一致，
  让额外像素不是装饰，而是稳定控制输出折线复杂度。

c) Audit of old assumptions
---------------------------
- 原 solver 的 `is_arrow_pts()` 假设箭头长度恒为 3，且 bbox 恒为 2x2。
  这在变体里不成立，必须改成“查找 3 像素 L 核心 + 允许的臂端延长像素”。
- 原 `arrow_dir_and_start()` 通过“度数最大点”找直角顶点。5 像素双延长形中有 3 个度数为 2 的点，
  该写法会失效。现在改为直接枚举候选 `start + (dr,dc)`，从核心缺角关系反推出方向和起点。
- 原 solver 的状态只记 `(cur, dr, dc, cur_col)`。现在反射额度会影响后续行为，因此状态里必须再加上
  `remaining_reflections`，否则带预算的回路检测会误判。
- 原 generator 默认箭头只有 2x2 占地。现在 4/5 像素箭头需要最多 3x3 空间，起点选取范围也要随之放宽。

d) Placement validity
---------------------
- 生成器只放置 4/5 像素源对象；所有额外像素都与核心 4 邻接，形成单一同色连通块。
- 箭头颜色与墙颜色分离：墙只从未被箭头占用的颜色池里取，避免同色墙与箭头在 4 邻接上并块，
  影响 solver 的按颜色分量识别。
- 路径规划仍采用 output-first：先规划每条射线的若干段对角路径与每次 side-touch 反射的墙，
  再写入输入网格，最后用 `main(input)` 回验。

e) Visual observability
-----------------------
- 观察者可以直接比较不同箭头的“加长哪条臂”与输出中“折几次”的对应关系。
- 为了让规律在单个样例中也可见，生成器优先放置 3 个箭头，编码分别为 1/2/3 次反射。
- 输出仍保留原题的墙颜色继承机制，所以视觉上既有原题熟悉的彩色折线，也新增了能控制折线复杂度的源形状。

f) Visual cleanliness
---------------------
- 仍然保持：箭头、墙、射线轨迹不重叠。
- 源对象只在自身周围 3x3 小范围内扩展，不会像远距标记那样把整张图弄得杂乱。
- 反射墙依旧只是一像素或三像素直线段，画面结构与原题保持同一风格。

g) Color-change inferability
----------------------------
- 颜色规则不变：射线初始为箭头色，撞墙后继承墙色。
- 变体只把“最大反射次数”绑定到形状，不把额外含义再绑定到颜色，避免颜色自由度过高。

New solver assumptions and generator guarantees
-----------------------------------------------
- Solver 假设每个箭头连通块恰好匹配一种合法的 4/5 像素模板。Generator 通过模板化放置保证这一点。
- Solver 假设箭头颜色不会与相邻墙合并成更大同色连通块。Generator 通过箭头色/墙色分池保证这一点。
- Solver 允许预算耗尽后在下一次撞墙前继续直线飞行；若预算耗尽时再遇到墙，则在当前/撞前位置停止。
  Generator 主要规划“恰好用满预算后离开网格”，因此不会依赖更复杂的停机边界分支，但 solver 仍能处理。
"""


def _arrow_budget_from_extra(extra: set[tuple[int, int]], ext_v: tuple[int, int], ext_h: tuple[int, int]) -> int | None:
    if extra == {ext_v}:
        return 1
    if extra == {ext_h}:
        return 2
    if extra == {ext_v, ext_h}:
        return 3
    return None


def _parse_arrow_component(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    """Return ((dr, dc), start, max_reflections) for a legal 4/5-pixel arrow."""
    pts_set = {(int(r), int(c)) for r, c in pts}
    if len(pts_set) not in (4, 5):
        return None

    matches: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for sr, sc in pts_set:
        for dr in (-1, 1):
            for dc in (-1, 1):
                base = {(sr, sc), (sr - dr, sc), (sr, sc - dc)}
                miss = (sr - dr, sc - dc)
                ext_v = (sr - 2 * dr, sc)
                ext_h = (sr, sc - 2 * dc)
                if not base.issubset(pts_set):
                    continue
                if miss in pts_set:
                    continue
                extra = pts_set - base
                if not extra or not extra.issubset({ext_v, ext_h}):
                    continue
                budget = _arrow_budget_from_extra(extra, ext_v, ext_h)
                if budget is None:
                    continue
                matches.append(((int(dr), int(dc)), (int(sr), int(sc)), int(budget)))

    uniq: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    seen = set()
    for match in matches:
        if match not in seen:
            seen.add(match)
            uniq.append(match)
    if len(uniq) != 1:
        return None
    return uniq[0]


def _build_arrow_pts(sr: int, sc: int, dr: int, dc: int, budget: int) -> list[tuple[int, int]]:
    pts = {(sr, sc), (sr - dr, sc), (sr, sc - dc)}
    if budget in (1, 3):
        pts.add((sr - 2 * dr, sc))
    if budget in (2, 3):
        pts.add((sr, sc - 2 * dc))
    return [(int(r), int(c)) for r, c in pts]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from tagged L-shaped arrows.

    The 3-pixel L core still determines the launch direction. Extra pixels extending the
    vertical and/or horizontal arm encode a maximum reflection budget:
    - longer vertical arm only   -> 1 reflection
    - longer horizontal arm only -> 2 reflections
    - both arms longer           -> 3 reflections
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            info = _parse_arrow_component(pts)
            if info is None:
                continue
            (dr, dc), start, budget = info
            arrows.append((int(col), (int(dr), int(dc)), (int(start[0]), int(start[1])), int(budget)))
            arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, (dr, dc), start, remaining in arrows:
        cur = (int(start[0]), int(start[1]))
        cur_col = int(start_col)
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(remaining))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                if remaining <= 0:
                    break
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                remaining -= 1
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            terminated = False

            hit_r = (nr + dr, nc)
            if hit_r in wall_cells:
                if remaining <= 0:
                    terminated = True
                else:
                    wcol = int(g[hit_r[0], hit_r[1]])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dr = -int(dr)
                    remaining -= 1

            if terminated:
                break

            hit_c = (nr, nc + dc)
            if hit_c in wall_cells:
                if remaining <= 0:
                    break
                wcol = int(g[hit_c[0], hit_c[1]])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                remaining -= 1

    return out


def _distance_to_edge(H: int, W: int, r: int, c: int, dr: int, dc: int) -> int:
    return int(
        min(
            (H - 1 - r) if dr == 1 else r,
            (W - 1 - c) if dc == 1 else c,
        )
    )


def _next_segment_room_ok(H: int, W: int, r: int, c: int, dr: int, dc: int, future_bounces: int) -> bool:
    max_steps = _distance_to_edge(H, W, r, c, dr, dc)
    return max_steps >= (2 if future_bounces > 0 else 1)


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the D5 source-shape variant.

    Each arrow's shape fixes its reflection budget, and the planned ray uses exactly that
    many side-touch bounces before exiting the grid.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(3, 4)

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = palette[n_arrows:]
    if len(wall_palette) < 3:
        return None

    budgets = [1, 2, 3]
    while len(budgets) < n_arrows:
        budgets.append(_rng.choice([1, 2, 3]))
    _rng.shuffle(budgets)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        budget = int(budgets[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])

        placed = False
        for _ in range(140):
            if budget in (1, 3):
                r_lo = 2 if dr == 1 else 0
                r_hi = H - 1 if dr == 1 else H - 3
            else:
                r_lo = 1 if dr == 1 else 0
                r_hi = H - 1 if dr == 1 else H - 2
            if budget in (2, 3):
                c_lo = 2 if dc == 1 else 0
                c_hi = W - 1 if dc == 1 else W - 3
            else:
                c_lo = 1 if dc == 1 else 0
                c_hi = W - 1 if dc == 1 else W - 2
            if r_lo > r_hi or c_lo > c_hi:
                continue

            sr = _rng.randint(r_lo, r_hi)
            sc = _rng.randint(c_lo, c_hi)
            arrow_pts = _build_arrow_pts(sr, sc, dr, dc, budget)
            arrow_set = set(arrow_pts)
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            cur_col = color
            ray_cells: list[tuple[int, int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ok = True

            for seg_idx in range(budget + 1):
                bounces_left = budget - seg_idx
                max_steps = _distance_to_edge(H, W, cur_r, cur_c, cur_dr, cur_dc)
                if bounces_left > 0:
                    if max_steps < 3:
                        ok = False
                        break
                    seg_len = _rng.randint(2, max_steps - 1)
                else:
                    if max_steps < 2:
                        ok = False
                        break
                    seg_len = max_steps

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                if bounces_left <= 0:
                    continue

                options: list[tuple[list[tuple[int, int]], str, tuple[int, int]]] = []

                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    single = [(int(wr), int(wc))]
                    if _next_segment_room_ok(H, W, cur_r, cur_c, -cur_dr, cur_dc, bounces_left - 1):
                        options.append((single, "dr", (-cur_dr, cur_dc)))
                    wall3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _r, c in wall3):
                        if _next_segment_room_ok(H, W, cur_r, cur_c, -cur_dr, cur_dc, bounces_left - 1):
                            options.append((wall3, "dr", (-cur_dr, cur_dc)))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    single = [(int(wr2), int(wc2))]
                    if _next_segment_room_ok(H, W, cur_r, cur_c, cur_dr, -cur_dc, bounces_left - 1):
                        options.append((single, "dc", (cur_dr, -cur_dc)))
                    wall3 = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _c in wall3):
                        if _next_segment_room_ok(H, W, cur_r, cur_c, cur_dr, -cur_dc, bounces_left - 1):
                            options.append((wall3, "dc", (cur_dr, -cur_dc)))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.55
                options.sort(key=lambda item: len(item[0]), reverse=prefer_3)

                chosen = None
                for wpxs, flip, next_dir in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    chosen = (wpxs, flip, next_dir)
                    break
                if chosen is None:
                    ok = False
                    break

                wpxs, flip, next_dir = chosen
                wcol = int(_rng.choice(wall_palette))
                walls.append((wpxs, wcol))
                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = wcol
                cur_dr, cur_dc = int(next_dir[0]), int(next_dir[1])

            if not ok or len(ray_cells) < 4:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = color
            for r, c, col in ray_cells:
                out[r, c] = col
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = wcol
            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D5 source-shape variant of 142ca369."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
