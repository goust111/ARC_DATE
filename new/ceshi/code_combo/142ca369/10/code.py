# Combo variant of 142ca369 (index 10)
# combo_spec: D11/0+D12/1
# Name: 单箭头+高反射次数

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg

# Design analysis
# fused_rule:
#   - 保留原题对象词汇：3 像素 L 形箭头、单色墙像素/短线段、沿对角线行进且会继承墙色的射线。
#   - D11/0 把“1-14 个独立箭头”收缩成“固定 1 个箭头”，所以推理重心不再是管理多源，而是深追踪一条路径。
#   - D12/1 把“0-4 次反射”抬高到“5-10 次反射”，并配套使用更大的 28-40 尺寸网格来容纳长距离锯齿线。
#   - 为了让高反射既稳定又整洁，生成器不靠随机碰撞，而是规划一条沿单调主轴前进的蛇形路径：
#     例如列始终递增、只在上下方向反复翻折；或者行始终递增、只在左右方向反复翻折。
# concrete_trace:
#   - 例：箭头右角源点在 (10,3)，朝右下发射。
#   - 原题常见情况：第一段走几步后遇到墙，只反射 0-4 次，总体是短折线；如果有多个箭头，人眼还要先分辨每条线是谁发出的。
#   - 本融合版：同一个箭头会连续命中 5-10 个侧墙。
#     设它先走到 (13,6)，其正下方 (14,6) 有墙，则当前格 (13,6) 改成墙色，dr 翻转，后续改为右上。
#     再走到 (9,10) 时其正上方 (8,10) 又有墙，于是再次改色并折回右下。
#     如此重复，形成一条单源、长距离、反复变色的锯齿轨迹，直到离开网格。
#   - 差异本质：
#     原题重点是“多条较短独立线”；本变体重点是“单条很长的深路径”。
# complexity_calibration:
#   - 只保留 1 个箭头，避免多箭头叠加后高反射路径变得不可读。
#   - 反射次数固定在 5-10 的可见高档位；每个样例都强制达到该范围，避免退化回原题的 0-4 次短路径。
#   - 墙色从少量重复调色板中抽取，保证“撞墙后继承颜色”的规律能从结构线索而非颜色 id 本身读出来。
# implicit_assumption_audit:
#   - 原 `for start_col, pts in arrows` 默认多箭头很常见；融合后生成器保证只有 1 个箭头，但 solver 仍兼容多箭头输入。
#   - 原 `state = (cur, dr, dc, cur_col)` 假设环境静态；本变体仍是静态墙场，因此这个去环前提继续成立。
#   - 原 side-bounce 代码默认墙就在当前行/列旁边 1 格；融合后仍成立，只是这种局部事件被重复 5-10 次。
#   - 原题里随机生成通常默认短路径不易回撞；融合后若继续随机，会更容易打到旧墙或旧轨迹。
#     因此生成器改成单调主轴规划，并在提交前用 solver 复算，拒绝任何额外碰撞/额外反射。
# placement_legality:
#   - 箭头、射线路径、墙像素三者必须严格不重叠，且全部在边界内。
#   - 每个墙都放在某个 bounce cell 的侧边 1 格，恰好对应一次侧碰，不允许提前挡住别的线段。
#   - 生成器在构造后再次跑箭头检测，确保没有墙组件伪装成第二个 L 形箭头。
# visual_observability:
#   - 输入里只有一个显眼的 L 形箭头和许多分散墙标记；输出里是一条长锯齿彩色路径，视觉上和原题的多短线明显不同。
#   - 关键控制变量“反射次数高”是可量化且可见的：样例中每条路径都需要至少 5 个折点。
# cleanliness:
#   - 路径沿一个主轴单调推进，避免自交和大面积回扫，能在大网格上保持整洁。
#   - 反射墙只服务于这条主路径，不引入额外对象层次，因此高复杂度来自深追踪而不是画面杂乱。
# new_solver_assumptions_and_generator_guards:
#   - solver 新增的隐式假设是：如果输入中存在本变体样例，长路径主要由 side-bounce 组成。
#     generator guard: 只摆放侧碰墙，并验证没有意外的 corner-bounce。
#   - solver 仍假设箭头可由单色 3 像素 L 形分量检测到。
#     generator guard: 墙不与箭头同色，且提交前检查检测结果恰好只有 1 个箭头。
#   - solver 对输入没有“必须单箭头”硬编码，但本组合的行为依赖单箭头。
#     generator guard: 每个生成样例严格只放 1 个箭头。


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
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


def _arrow_points_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _detect_arrows(g: np.ndarray, bg: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    arrows: list[dict[str, object]] = []
    arrow_cells: set[tuple[int, int]] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            direction, start = info
            arrows.append({"color": int(col), "pts": pts, "dir": direction, "start": start})
            arrow_cells |= set(pts)
    arrows.sort(key=lambda a: (a["start"][0], a["start"][1], a["color"]))  # type: ignore[index]
    return arrows, arrow_cells


def _trace_arrow(
    g: np.ndarray,
    out: np.ndarray,
    arrow: dict[str, object],
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    bg: int,
) -> dict[str, object]:
    dr, dc = arrow["dir"]  # type: ignore[assignment]
    cur = arrow["start"]  # type: ignore[assignment]
    cur_col = int(arrow["color"])
    seen: set[tuple[tuple[int, int], int, int, int]] = set()
    path_positions: list[tuple[int, int]] = []
    bounce_cells: list[tuple[int, int]] = []
    side_bounces = 0
    corner_bounces = 0

    while True:
        state = (cur, int(dr), int(dc), int(cur_col))
        if state in seen:
            break
        seen.add(state)

        nr = int(cur[0] + dr)
        nc = int(cur[1] + dc)
        if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
            break

        if (nr, nc) in wall_cells:
            wcol = int(g[nr, nc])
            if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                out[int(cur[0]), int(cur[1])] = int(wcol)
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            corner_bounces += 1
            bounce_cells.append((int(cur[0]), int(cur[1])))
            continue

        if (nr, nc) in arrow_cells:
            break

        cur = (int(nr), int(nc))
        path_positions.append(cur)
        if int(out[nr, nc]) == int(bg):
            out[nr, nc] = int(cur_col)

        bounced_here = False
        if int(dr) == 1 and (nr + 1, nc) in wall_cells:
            wcol = int(g[nr + 1, nc])
            out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)
            side_bounces += 1
            bounced_here = True
        elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
            wcol = int(g[nr - 1, nc])
            out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)
            side_bounces += 1
            bounced_here = True

        if int(dc) == 1 and (nr, nc + 1) in wall_cells:
            wcol = int(g[nr, nc + 1])
            out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)
            side_bounces += 1
            bounced_here = True
        elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
            wcol = int(g[nr, nc - 1])
            out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)
            side_bounces += 1
            bounced_here = True

        if bounced_here:
            bounce_cells.append((nr, nc))

    return {
        "start": arrow["start"],
        "dir": arrow["dir"],
        "color": int(arrow["color"]),
        "path_positions": path_positions,
        "path_len": len(path_positions),
        "bounce_cells": bounce_cells,
        "side_bounces": side_bounces,
        "corner_bounces": corner_bounces,
        "bounce_count": side_bounces + corner_bounces,
    }


def _simulate_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    arrows, arrow_cells = _detect_arrows(g, bg)
    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()
    stats: list[dict[str, object]] = []
    for arrow in arrows:
        stats.append(_trace_arrow(g, out, arrow, arrow_cells, wall_cells, bg))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows, reflecting off wall pixels and
    inheriting the wall color at each bounce.

    In this combo's generated data there is exactly one arrow, but the solver
    keeps the base-task semantics and remains tolerant to multi-arrow inputs.
    """
    out, _stats = _simulate_scene(grid)
    return out


def _build_wall_pixels(
    orientation: str,
    bounce_r: int,
    bounce_c: int,
    dr_before: int,
    dc_before: int,
    shape3_first: bool,
) -> list[list[tuple[int, int]]]:
    if orientation == "horizontal":
        wr, wc = bounce_r + dr_before, bounce_c
        options = [[(wr, wc)]]
        options3 = [[(wr, wc - 1), (wr, wc), (wr, wc + 1)]]
    else:
        wr, wc = bounce_r, bounce_c + dc_before
        options = [[(wr, wc)]]
        options3 = [[(wr - 1, wc), (wr, wc), (wr + 1, wc)]]
    return options3 + options if shape3_first else options + options3


def _path_offsets(lengths: list[int], first_sign: int) -> tuple[int, int]:
    acc = 0
    lo = 0
    hi = 0
    sign = int(first_sign)
    for seg_len in lengths:
        acc += sign * int(seg_len)
        lo = min(lo, acc)
        hi = max(hi, acc)
        sign = -sign
    return lo, hi


def _paint_planned_output(
    grid: np.ndarray,
    arrow_color: int,
    ray_positions: list[tuple[int, int]],
    segment_lengths: list[int],
    wall_colors: list[int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    cur_col = int(arrow_color)
    idx = 0
    for bounce_idx, seg_len in enumerate(segment_lengths[:-1]):
        for _ in range(seg_len):
            r, c = ray_positions[idx]
            out[r, c] = int(cur_col)
            idx += 1
        br, bc = ray_positions[idx - 1]
        cur_col = int(wall_colors[bounce_idx])
        out[br, bc] = int(cur_col)
    for _ in range(segment_lengths[-1]):
        r, c = ray_positions[idx]
        out[r, c] = int(cur_col)
        idx += 1
    return out


def _attempt_plan() -> tuple[np.ndarray, np.ndarray, dict[str, int | str]] | None:
    bounce_count = _rng.randint(5, 10)
    segment_lengths = [_rng.randint(2, 4) for _ in range(bounce_count)] + [_rng.randint(2, 5)]
    orientation = _rng.choice(["horizontal", "vertical"])
    dr = _rng.choice([-1, 1])
    dc = _rng.choice([-1, 1])
    arrow_color = _rng.randint(1, 9)
    wall_palette = [c for c in range(1, 10) if c != arrow_color]
    _rng.shuffle(wall_palette)
    wall_palette = wall_palette[: _rng.randint(3, 5)]
    total_main = sum(segment_lengths)

    if orientation == "horizontal":
        row_lo, row_hi = _path_offsets(segment_lengths[:-1], dr)
        H_min = (row_hi - row_lo) + 5
        H = max(26, H_min + _rng.randint(0, 6))
        if H > 36:
            return None
        src_r_lo = max(1, 1 - row_lo)
        src_r_hi = min(H - 2, H - 2 - row_hi)
        if src_r_lo > src_r_hi:
            return None
        sr = _rng.randint(src_r_lo, src_r_hi)

        W_lo = max(28, total_main + 2)
        W_hi = 40
        if W_lo > W_hi:
            return None
        W = _rng.randint(W_lo, W_hi)
        sc = W - 1 - total_main if dc == 1 else total_main
        if not (1 <= sc <= W - 2):
            return None
    else:
        col_lo, col_hi = _path_offsets(segment_lengths[:-1], dc)
        W_min = (col_hi - col_lo) + 5
        W = max(26, W_min + _rng.randint(0, 6))
        if W > 36:
            return None
        src_c_lo = max(1, 1 - col_lo)
        src_c_hi = min(W - 2, W - 2 - col_hi)
        if src_c_lo > src_c_hi:
            return None
        sc = _rng.randint(src_c_lo, src_c_hi)

        H_lo = max(28, total_main + 2)
        H_hi = 40
        if H_lo > H_hi:
            return None
        H = _rng.randint(H_lo, H_hi)
        sr = H - 1 - total_main if dr == 1 else total_main
        if not (1 <= sr <= H - 2):
            return None

    arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
    if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
        return None

    ray_positions: list[tuple[int, int]] = []
    bounce_specs: list[tuple[int, int, int, int]] = []
    cur_r, cur_c = sr, sc
    cur_dr, cur_dc = dr, dc

    for seg_len in segment_lengths[:-1]:
        for _ in range(seg_len):
            cur_r += cur_dr
            cur_c += cur_dc
            if not (0 <= cur_r < H and 0 <= cur_c < W):
                return None
            ray_positions.append((cur_r, cur_c))
        bounce_specs.append((cur_r, cur_c, cur_dr, cur_dc))
        if orientation == "horizontal":
            cur_dr = -cur_dr
        else:
            cur_dc = -cur_dc

    for _ in range(segment_lengths[-1]):
        cur_r += cur_dr
        cur_c += cur_dc
        if not (0 <= cur_r < H and 0 <= cur_c < W):
            return None
        ray_positions.append((cur_r, cur_c))

    if len(set(ray_positions)) != len(ray_positions):
        return None
    if set(ray_positions) & set(arrow_pts):
        return None
    if len(ray_positions) < 12:
        return None

    walls: list[tuple[list[tuple[int, int]], int]] = []
    wall_cells: set[tuple[int, int]] = set()
    reserved = set(ray_positions) | set(arrow_pts)
    wall_colors: list[int] = []

    for bounce_r, bounce_c, dr_before, dc_before in bounce_specs:
        options = _build_wall_pixels(
            orientation,
            bounce_r,
            bounce_c,
            dr_before,
            dc_before,
            shape3_first=(_rng.random() < 0.7),
        )
        chosen = None
        for cells in options:
            if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
                continue
            if any((r, c) in reserved or (r, c) in wall_cells for r, c in cells):
                continue
            chosen = cells
            break
        if chosen is None:
            return None
        wall_color = _rng.choice(wall_palette)
        walls.append((chosen, wall_color))
        wall_colors.append(wall_color)
        wall_cells.update(chosen)

    grid = np.zeros((H, W), dtype=int)
    for r, c in arrow_pts:
        grid[r, c] = int(arrow_color)
    for cells, wall_color in walls:
        for r, c in cells:
            grid[r, c] = int(wall_color)

    arrows, arrow_cells = _detect_arrows(grid, 0)
    if len(arrows) != 1:
        return None
    if set(arrow_pts) != arrow_cells:
        return None

    planned_out = _paint_planned_output(grid, arrow_color, ray_positions, segment_lengths, wall_colors)
    actual_out, stats = _simulate_scene(grid)
    if len(stats) != 1:
        return None
    stat = stats[0]
    if int(stat["bounce_count"]) != bounce_count:
        return None
    if int(stat["corner_bounces"]) != 0:
        return None
    if int(stat["side_bounces"]) != bounce_count:
        return None
    if int(stat["path_len"]) != len(ray_positions):
        return None
    if list(stat["path_positions"]) != ray_positions:
        return None
    if not np.array_equal(actual_out, planned_out):
        return None

    return grid, planned_out, {"bounces": bounce_count, "orientation": orientation}


def _gen_attempt() -> np.ndarray | None:
    result = _attempt_plan()
    if result is None:
        return None
    grid, _out, _meta = result
    return grid


def generate_input() -> np.ndarray:
    """Generate a valid single-arrow high-bounce input for the combo variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
