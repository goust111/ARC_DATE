# Combo variant of 142ca369 (index 27)
# combo_spec: D3/2+D6/1
# Name: 渐变轨迹+固定步数终止
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题对象：3 像素 L 形箭头、单色墙、对角线射线、撞墙换色。
#    - 新增 D6 编码：每个箭头后方的反向对角线上放同色 tail；tail 长度 = 该箭头真实前进总步数。
#    - 新增 D3 显影：真实路径仍完整行走并反射，但输出只显示每段前缀。第 1 段显示前 3 格，第 2 段
#      显示前 2 格，第 3 段显示前 1 格；反射点仍改成墙色，暴露换色位置。
#    - 小例子：若 tail=9，真实三段长度为 4,3,2，则 solver 实际走 9 步；输出只显影
#      段 1 的前 3 格、段 2 的前 2 格、段 3 的前 1 格。三段最后一格若是反射点，则显示墙色而不是
#      原轨迹色。若不用固定步数，这条路径还能继续；若不用渐变轨迹，则 9 步都会被全部画出。
# b) 复杂度校准
#    - 递减规则若允许很多段，会出现 3,2,1,0... 的歧义，因此把 generator 的反射次数限制为 0-2，
#      即每条射线最多 3 段。solver 仍对更长段序列稳健处理：第 4 段及以后按 1 格前缀显示。
#    - 每张图至少放 1 支“三段主箭”，让 3,2,1 同时出现；其他箭再混入 1-3 段与不同 tail 长度，
#      让“数 tail 决定总步数”也有重复支撑。
# c) 隐式假设审查
#    - 原 solver 假设所有非箭头非背景像素都是墙；融合后 tail 不是墙，必须先从同色箭头后方剥离。
#    - 原 solver 假设终止由出界保证；融合后终止由 tail 计数保证，所以状态里必须跟踪 `steps_used`。
#    - 原 solver 假设每次前进都可见；融合后只能在 `segment_steps < quota` 时显影，且反射点覆盖优先。
#    - 原 solver 的循环检测只依赖位置/方向/颜色；这里 corner-bounce 可能在不耗步时形成死循环，
#      所以把 `steps_used` 也纳入状态。
# d) 放置合法性
#    - generator 仍按真实路径占格做碰撞检查：即便输出里某些经过格保持背景，它们也不能被别的对象复用。
#    - tail、箭头、墙、真实路径、以及“本可继续前进的一格”都预留占位，防止固定步数与出界终止退化成同一件事。
# e) 视觉可观察性
#    - 主箭固定为三段，且真实长度至少为 4,3,2，因此人眼能直接看到 `3 -> 2 -> 1` 的显影递减。
#    - 终点前方至少再留 1 个可走对角格，所以输出会在网格内部停止，tail 的计数作用清晰可见。
# f) 可视化整洁度
#    - tail 放在箭头背后且不与箭头 4 连通；墙颜色与箭头颜色分离；不同箭对象之间用 `blocked` 留出缓冲。
#    - 输出只出现箭头、tail、墙、少量显影前缀和反射点，不会被整条隐藏路径搅乱。
# g) 变化的可推断性
#    - 颜色变化仍然只来自“撞到哪面墙就继承哪种颜色”，没有新的抽象颜色映射。
#    - tail 只编码数量，不编码颜色语义；同色 tail 与同色箭头一一配对，数数即可。
# h) 融合核心指标
#    - `hidden_steps = 真实总步数 - 输出中可见轨迹格数`：若为 0，就退化成连续轨迹。
#    - `extra_room >= 1`：固定步数终止后，按最终方向至少还能再前进一步；若为 0，就退化成边界终止。
# i) main 的新假设与生成保证
#    - 假设：tail 是箭头同色、连续、位于反向对角线。generator 始终这样构造，并强制箭头颜色唯一。
#    - 假设：墙不会复用箭头颜色。generator 从剩余颜色中取墙色，避免 tail 计数误吸收墙。
#    - 假设：每个反射只对应一类侧墙。generator 在终点与反射点周围做 clearance 预留，保证 solver 不会遇到
#      模糊的双侧同时反射样例。

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg, in_bounds


def _segment_quota(segment_idx: int) -> int:
    """Visible prefix length for the given segment: 3, 2, 1, 1, ..."""
    return max(1, 3 - int(segment_idx))


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_start_missing(
    pts: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
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
    return (int(dr), int(dc)), (int(start[0]), int(start[1])), (int(mr), int(mc))


def _arrow_points(start_r: int, start_c: int, dr: int, dc: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    miss = (int(start_r - dr), int(start_c - dc))
    pts = [
        (int(r), int(c))
        for r in (miss[0], start_r)
        for c in (miss[1], start_c)
        if (int(r), int(c)) != miss
    ]
    return pts, miss


def _tail_points(start_r: int, start_c: int, dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    return [(int(start_r - k * dr), int(start_c - k * dc)) for k in range(2, steps + 2)]


def _choose_wall_color(wall_palette: list[int], cur_col: int) -> int:
    options = [col for col in wall_palette if int(col) != int(cur_col)]
    return int(_rng.choice(options if options else wall_palette))


def _min_segment_lengths(n_bounces: int, rich_arrow: bool) -> list[int]:
    parts = int(n_bounces) + 1
    mins = [_segment_quota(i) for i in range(parts)]
    mins[-1] += 1  # force at least one hidden step on the last segment
    if rich_arrow and parts == 3:
        return [3, 2, 2]
    return [int(v) for v in mins]


def _sample_segment_lengths(total_steps: int, n_bounces: int, rich_arrow: bool) -> list[int] | None:
    parts = int(n_bounces) + 1
    mins = _min_segment_lengths(int(n_bounces), bool(rich_arrow))
    remaining = int(total_steps) - sum(mins)
    if remaining < 0:
        return None
    lengths = list(mins)
    for _ in range(remaining):
        lengths[_rng.randrange(parts)] += 1
    return [int(v) for v in lengths]


def _segment_visible_cells(
    segments: list[dict[str, object]],
) -> tuple[dict[tuple[int, int], int], int]:
    visible: dict[tuple[int, int], int] = {}
    traversed = 0
    for seg_idx, seg in enumerate(segments):
        cells = list(seg["cells"])
        traversed += len(cells)
        quota = min(_segment_quota(seg_idx), len(cells))
        for r, c in cells[:quota]:
            visible[(int(r), int(c))] = int(seg["start_color"])
        bounce_color = seg["bounce_color"]
        if bounce_color is not None and cells:
            r, c = cells[-1]
            visible[(int(r), int(c))] = int(bounce_color)
    hidden_steps = traversed - len(visible)
    return visible, int(hidden_steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-arrows, stop after a tail-encoded step budget,
    and only reveal a decreasing visible prefix per segment: 3, then 2, then 1.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows: list[dict[str, object]] = []
    arrow_cells: set[tuple[int, int]] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_start_missing(pts)
            if info is None:
                continue
            (dr, dc), start, missing = info
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": (int(dr), int(dc)),
                    "start": start,
                    "missing": missing,
                }
            )
            arrow_cells |= set(pts)

    marker_cells: set[tuple[int, int]] = set()
    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        sr, sc = arrow["start"]  # type: ignore[assignment]
        steps = 0
        k = 2
        while True:
            rr = int(sr - k * dr)
            cc = int(sc - k * dc)
            if not in_bounds((H, W), rr, cc):
                break
            if int(g[rr, cc]) != color:
                break
            marker_cells.add((int(rr), int(cc)))
            steps += 1
            k += 1
        arrow["steps"] = int(steps)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells and (int(r), int(c)) not in marker_cells
    }

    out = g.copy()

    for arrow in arrows:
        step_limit = int(arrow.get("steps", 0))
        if step_limit <= 0:
            continue

        dr, dc = arrow["dir"]  # type: ignore[assignment]
        cur = arrow["start"]  # type: ignore[assignment]
        cur_col = int(arrow["color"])
        steps_used = 0
        segment_idx = 0
        segment_steps = 0
        seen: set[tuple[tuple[int, int], int, int, int, int]] = set()

        while steps_used < step_limit:
            state = (cur, int(dr), int(dc), int(cur_col), int(steps_used))
            if state in seen:
                break
            seen.add(state)

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

            if (nr, nc) in arrow_cells or (nr, nc) in marker_cells:
                break

            cur = (int(nr), int(nc))
            steps_used += 1
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


def _trace_plan(
    H: int,
    W: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    start_col: int,
    segment_lengths: list[int],
    wall_palette: list[int],
    blocked: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    cur_r, cur_c = int(start_r), int(start_c)
    cur_dr, cur_dc = int(dr), int(dc)
    cur_col = int(start_col)

    ray_set: set[tuple[int, int]] = set()
    walls: list[tuple[list[tuple[int, int]], int]] = []
    wall_cells: set[tuple[int, int]] = set()
    local_blocked = set(reserved)
    segments: list[dict[str, object]] = []

    for seg_idx, seg_len in enumerate(segment_lengths):
        seg_cells: list[tuple[int, int]] = []
        seg_start_col = int(cur_col)
        max_steps = min(
            (H - 1 - cur_r) if cur_dr == 1 else cur_r,
            (W - 1 - cur_c) if cur_dc == 1 else cur_c,
        )
        need = int(seg_len) + (1 if seg_idx == len(segment_lengths) - 1 else 0)
        if max_steps < need:
            return None
        for _ in range(int(seg_len)):
            nr = int(cur_r + cur_dr)
            nc = int(cur_c + cur_dc)
            if not in_bounds((H, W), nr, nc):
                return None
            if (nr, nc) in blocked or (nr, nc) in local_blocked or (nr, nc) in ray_set:
                return None
            seg_cells.append((int(nr), int(nc)))
            ray_set.add((int(nr), int(nc)))
            cur_r, cur_c = int(nr), int(nc)

        bounce_color: int | None = None
        if seg_idx < len(segment_lengths) - 1:
            options: list[tuple[list[tuple[int, int]], str]] = []

            wr, wc = int(cur_r + cur_dr), int(cur_c)
            if 0 <= wr < H:
                options.append(([(int(wr), int(wc))], "dr"))
                w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                if all(0 <= cc < W for _, cc in w3):
                    options.append((w3, "dr"))

            wr2, wc2 = int(cur_r), int(cur_c + cur_dc)
            if 0 <= wc2 < W:
                options.append(([(int(wr2), int(wc2))], "dc"))
                w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                if all(0 <= rr < H for rr, _ in w3v):
                    options.append((w3v, "dc"))

            if not options:
                return None

            _rng.shuffle(options)
            options.sort(key=lambda item: len(item[0]), reverse=_rng.random() < 0.6)
            chosen = None
            future_candidates: list[tuple[int, list[tuple[int, int]], str]] = []
            next_len = int(segment_lengths[seg_idx + 1])
            next_need = next_len + (1 if seg_idx + 1 == len(segment_lengths) - 1 else 0)
            for wpxs, flip in options:
                if any((r, c) in blocked or (r, c) in local_blocked or (r, c) in ray_set for r, c in wpxs):
                    continue
                next_dr = -int(cur_dr) if flip == "dr" else int(cur_dr)
                next_dc = -int(cur_dc) if flip == "dc" else int(cur_dc)
                future_max = min(
                    (H - 1 - cur_r) if next_dr == 1 else cur_r,
                    (W - 1 - cur_c) if next_dc == 1 else cur_c,
                )
                if future_max < next_need:
                    continue
                future_candidates.append((int(future_max), wpxs, flip))
            if future_candidates:
                future_candidates.sort(key=lambda item: item[0], reverse=True)
                _best, best_wpxs, best_flip = future_candidates[0]
                chosen = (best_wpxs, best_flip)
            if chosen is None:
                return None

            wpxs, flip = chosen
            bounce_color = _choose_wall_color(wall_palette, cur_col)
            walls.append((wpxs, int(bounce_color)))
            wall_cells.update((int(r), int(c)) for r, c in wpxs)
            local_blocked.update((int(r), int(c)) for r, c in wpxs)
            cur_col = int(bounce_color)
            if flip == "dr":
                cur_dr = -int(cur_dr)
            else:
                cur_dc = -int(cur_dc)

        segments.append(
            {
                "cells": seg_cells,
                "start_color": int(seg_start_col),
                "bounce_color": bounce_color,
            }
        )

    next_diag = (int(cur_r + cur_dr), int(cur_c + cur_dc))
    if not in_bounds((H, W), next_diag[0], next_diag[1]):
        return None
    if next_diag in blocked or next_diag in local_blocked or next_diag in ray_set:
        return None

    raw_side_checks = [(int(cur_r + cur_dr), int(cur_c)), (int(cur_r), int(cur_c + cur_dc))]
    for pos in raw_side_checks:
        if pos in blocked or pos in local_blocked or pos in wall_cells or pos in ray_set:
            return None

    clearance = {next_diag}
    for pos in raw_side_checks:
        if in_bounds((H, W), pos[0], pos[1]):
            clearance.add((int(pos[0]), int(pos[1])))

    return {
        "segments": segments,
        "ray_set": ray_set,
        "walls": walls,
        "wall_cells": wall_cells,
        "clearance": clearance,
        "final_pos": (int(cur_r), int(cur_c)),
        "final_dir": (int(cur_dr), int(cur_dc)),
        "next_diag": next_diag,
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 3)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = [c for c in palette if c not in arrow_colors]
    if not wall_palette:
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    total_hidden_steps = 0

    for ai, color in enumerate(arrow_colors):
        rich_arrow = ai == 0
        placed = False

        for _ in range(180):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            n_bounces = 2 if rich_arrow else _rng.choices([0, 1, 2], weights=[1, 4, 4])[0]
            min_total = sum(_min_segment_lengths(int(n_bounces), rich_arrow))
            max_total = 8 if rich_arrow else 7
            if min_total > max_total:
                continue
            steps = _rng.randint(min_total, max_total)
            segment_lengths = _sample_segment_lengths(int(steps), int(n_bounces), rich_arrow)
            if segment_lengths is None:
                continue

            r_lo = steps + 1 if dr == 1 else 0
            r_hi = H - steps - 2 if dr == -1 else H - 1
            c_lo = steps + 1 if dc == 1 else 0
            c_hi = W - steps - 2 if dc == -1 else W - 1
            if r_lo > r_hi or c_lo > c_hi:
                continue

            start_r = _rng.randint(r_lo, r_hi)
            start_c = _rng.randint(c_lo, c_hi)
            arrow_pts, _missing = _arrow_points(start_r, start_c, dr, dc)
            tail_pts = _tail_points(start_r, start_c, dr, dc, int(steps))
            reserved = set(arrow_pts) | set(tail_pts)
            if any(not in_bounds((H, W), r, c) or (r, c) in blocked for r, c in reserved):
                continue

            plan = _trace_plan(
                H=H,
                W=W,
                start_r=start_r,
                start_c=start_c,
                dr=dr,
                dc=dc,
                start_col=int(color),
                segment_lengths=segment_lengths,
                wall_palette=wall_palette,
                blocked=blocked,
                reserved=reserved,
            )
            if plan is None:
                continue

            segments = plan["segments"]  # type: ignore[assignment]
            ray_set = plan["ray_set"]  # type: ignore[assignment]
            walls = plan["walls"]  # type: ignore[assignment]
            wall_cells = plan["wall_cells"]  # type: ignore[assignment]
            clearance = plan["clearance"]  # type: ignore[assignment]

            visible_cells, hidden_steps = _segment_visible_cells(segments)
            if hidden_steps <= 0:
                continue

            new_cells = set(reserved) | set(ray_set) | set(wall_cells)
            expected_count = len(reserved) + len(ray_set) + sum(len(wpxs) for wpxs, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = int(color)
                out[r, c] = int(color)
            for r, c in tail_pts:
                grid[r, c] = int(color)
                out[r, c] = int(color)
            for (r, c), vis_col in visible_cells.items():
                out[r, c] = int(vis_col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = int(wcol)
                    out[r, c] = int(wcol)

            occupied |= new_cells
            blocked |= new_cells | set(clearance)
            total_hidden_steps += int(hidden_steps)
            placed = True
            break

        if not placed:
            return None

    if total_hidden_steps <= 0:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for combo variant D3/2 + D6/1."""
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
