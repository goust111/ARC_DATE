# Variant of 142ca369
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: 连续轨迹（continuous_trail）——射线经过的每个背景格子都被涂色，形成连续的彩色线条
#   → 渐变轨迹（段长递增）——每一段只给前 1,2,3,... 个经过格着色，后续真实经过格保持背景色
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍然只包含 3 像素 L 形箭头和单色墙；对象类型、反射方式、换色方式都保持原题。
#    - 输出不再把整条真实对角线路径全部画出来。每支箭的第 1 段只显式涂 1 格，第 2 段涂 2 格，
#      第 3 段涂 3 格，以此类推。这里的“段”指从发射点/上一次反射点到下一次反射点/出界点之间的
#      一整段真实运动。被隐藏的格子虽然仍被射线穿过，但在输出里保持背景色。
#    - 具体例子：若一条射线第一段真实前进 5 步再撞到侧墙，那么原题会把 5 格全涂成当前颜色；
#      本变体只把前 1 格涂成当前颜色，中间 3 格留空，反射点仍按原题规则改成墙的颜色。第二段若
#      真实前进 6 步，则只把该段起点后的前 2 格涂色，再在第二次反射点显示墙色。
#    - 差异集中在“输出阶段的轨迹显影长度”，几何路径、反射顺序、颜色继承仍按原题执行。
# b) 复杂度校准
#    - 仅靠单段射线看不出递增规律，所以生成器保证每个样本至少有一支“主箭”拥有 3 段真实路径，
#      让 1,2,3 的可见长度在同一张图里重复出现。
#    - 其他箭保留 2-4 段随机变化，避免样例过于单调，同时不引入额外对象类型或颜色编码。
# c) 隐式假设审查（相对原 main）
#    - 原 `out[nr, nc] = cur_col` 默认每次前进都可见；变体中这不成立，必须改成“仅当该格位于本段
#      的前 quota 步内时才上色”。
#    - 原 solver 在反射点无条件把当前格改成墙色，这在旧题里只是覆盖同色轨迹；变体里该写法仍可保留，
#      但它现在承担“暴露反射事件”的职责。若删掉，会让颜色切换来源难以观察。
#    - 原 `seen` 依赖颜色和方向有限且段内行为恒定；变体多了段编号，会扩大状态空间。这里改用
#      `H*W*12` 的循环保护，避免把“段号必须有限”当成新的输入假设。
# d) 放置合法性
#    - 生成器仍以真实路径为准做碰撞检查：即便某些经过格在输出里保持背景，它们也不能与箭头、墙、
#      其他射线路径重叠，否则 solver 的真实追踪会出错。
#    - 墙的放置仍保持原题的“距真实路径 1 格的侧碰墙”，且每次反射只放一个横墙或竖墙。
# e) 视觉可观察性
#    - 主箭固定为 3 段，且前两次反射前都保留至少 1 个空白间隔，这样人眼能直接看到“第一段只亮 1 格，
#      第二段亮 2 格，第三段亮 3 格”的递增长度。
#    - 终段长度被限制在接近 quota 的范围内，避免最后一段真实路径过长却只显示开头几格，导致看上去
#      像是无故中断。
# f) 可视化整洁度
#    - 由于隐藏段仍占用空间，生成器把它们也计入 `occupied`，从而避免“输出里看着分离，真实路径却穿过
#      别的对象”的杂乱情况。
#    - 箭、墙、可见轨迹三者不共格；颜色变化只出现在箭本体、墙和有限的轨迹前缀/反射点上，图面更清楚。
# g) 颜色变化可推断性
#    - 颜色继承仍然只由“撞到哪面墙”决定，不新增任何颜色映射。
#    - 生成器偏向选择与当前轨迹色不同的墙色，让“反射点换色 + 下一段沿用新色”在少量样例中足够明显。
# h) 变体核心指标
#    - 用 `hidden_steps = 真实经过格数 - 输出中可见轨迹格数` 衡量是否真的发生了“隐藏/渐变轨迹”。
#      若 `hidden_steps == 0`，该样本会退化成原题连续轨迹，因此生成器拒绝这种样本。
# i) main 的新假设审查
#    - solver 现在假设“段编号按真实反射次数递增”；generator 通过 output-first 规划明确控制每次反射。
#    - solver 假设反射点仍可见，generator 在构造输出时同样总把有墙的段终点涂成墙色，保证一致。

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg


def _segment_quota(segment_idx: int) -> int:
    """Visible prefix length for the given ray segment."""
    return int(segment_idx) + 1


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None
    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-arrows, but only reveal an increasing-length prefix
    of each straight segment: segment 1 paints 1 cell, segment 2 paints 2 cells, etc.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        segment_idx = 0
        segment_steps = 0

        for _ in range(H * W * 12):
            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                segment_idx += 1
                segment_steps = 0
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if segment_steps < _segment_quota(segment_idx) and int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)
            segment_steps += 1

            bounced = False

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                bounced = True
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                bounced = True

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                bounced = True
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                bounced = True

            if bounced:
                segment_idx += 1
                segment_steps = 0

    return out


def _choose_wall_color(palette: list[int], cur_col: int) -> int:
    options = [col for col in palette if int(col) != int(cur_col)]
    return int(_rng.choice(options if options else palette))


def _segment_visible_cells(
    segments: list[dict[str, object]],
) -> tuple[dict[tuple[int, int], int], int]:
    visible: dict[tuple[int, int], int] = {}
    traversed_count = 0
    for seg_idx, seg in enumerate(segments):
        cells = list(seg["cells"])
        traversed_count += len(cells)
        quota = min(_segment_quota(seg_idx), len(cells))
        for r, c in cells[:quota]:
            visible[(int(r), int(c))] = int(seg["start_color"])
        bounce_color = seg["bounce_color"]
        if bounce_color is not None and cells:
            r, c = cells[-1]
            visible[(int(r), int(c))] = int(bounce_color)
    hidden_steps = traversed_count - len(visible)
    return visible, int(hidden_steps)


def _gen_attempt() -> np.ndarray | None:
    """Output-first generator for the trajectory-gradient variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 4)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    total_hidden_steps = 0
    rich_arrow_hidden = 0

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        rich_arrow = ai == 0
        n_bounces = 2 if rich_arrow else _rng.choices([1, 2, 3], weights=[2, 4, 3])[0]
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])

        placed = False
        for _ in range(160):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(color)
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            walls: list[tuple[list[tuple[int, int]], int]] = []
            segments: list[dict[str, object]] = []
            ok = True

            for seg_idx in range(n_bounces + 1):
                quota = _segment_quota(seg_idx)
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if max_steps <= 0:
                    ok = False
                    break

                if seg_idx < n_bounces:
                    min_seg_len = quota + (3 if rich_arrow and seg_idx < 2 else 1)
                    if max_steps - 1 < min_seg_len:
                        ok = False
                        break
                    seg_len = _rng.randint(min_seg_len, max_steps - 1)
                else:
                    if rich_arrow:
                        if not (quota <= max_steps <= quota + 2):
                            ok = False
                            break
                    seg_len = max_steps

                seg_cells: list[tuple[int, int]] = []
                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    seg_cells.append((int(nr), int(nc)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                bounce_color: int | None = None
                if seg_idx < n_bounces:
                    options: list[tuple[list[tuple[int, int]], str]] = []

                    wr, wc = cur_r + cur_dr, cur_c
                    if 0 <= wr < H:
                        options.append(([(wr, wc)], "dr"))
                        w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                        if all(0 <= c < W for _, c in w3):
                            options.append((w3, "dr"))

                    wr2, wc2 = cur_r, cur_c + cur_dc
                    if 0 <= wc2 < W:
                        options.append(([(wr2, wc2)], "dc"))
                        w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                        if all(0 <= r < H for r, _ in w3v):
                            options.append((w3v, "dc"))

                    _rng.shuffle(options)
                    options.sort(key=lambda item: len(item[0]), reverse=_rng.random() < 0.6)

                    chosen_wall = None
                    for wpxs, flip in options:
                        if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                            continue
                        chosen_wall = (wpxs, flip)
                        break

                    if chosen_wall is None:
                        ok = False
                        break

                    wpxs, flip = chosen_wall
                    bounce_color = _choose_wall_color(palette, cur_col)
                    walls.append((wpxs, int(bounce_color)))
                    cur_col = int(bounce_color)
                    if flip == "dr":
                        cur_dr = -cur_dr
                    else:
                        cur_dc = -cur_dc

                segments.append(
                    {
                        "cells": seg_cells,
                        "start_color": 0,
                        "bounce_color": bounce_color,
                    }
                )

            if not ok or len(segments) != n_bounces + 1:
                continue

            # Fix segment start colors after the full bounce plan is known.
            running_color = int(color)
            for seg in segments:
                seg["start_color"] = running_color
                if seg["bounce_color"] is not None:
                    running_color = int(seg["bounce_color"])

            visible_cells, hidden_steps = _segment_visible_cells(segments)
            if rich_arrow and hidden_steps < 4:
                continue
            if hidden_steps <= 0:
                continue

            total_traversed = sum(len(list(seg["cells"])) for seg in segments)
            if total_traversed < 6:
                continue

            new_cells = set(arrow_pts) | ray_set
            wall_total = 0
            for wpxs, _wcol in walls:
                wall_total += len(wpxs)
                new_cells.update((int(r), int(c)) for r, c in wpxs)
            expected_count = len(arrow_pts) + len(ray_set) + wall_total
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(color)
            for (r, c), vis_col in visible_cells.items():
                out[int(r), int(c)] = int(vis_col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)

            occupied |= new_cells
            total_hidden_steps += hidden_steps
            if rich_arrow:
                rich_arrow_hidden = hidden_steps
            placed = True
            break

        if not placed:
            return None

    if rich_arrow_hidden <= 0 or total_hidden_steps <= 0:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the trajectory-gradient variant."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
