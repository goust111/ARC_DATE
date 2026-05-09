# Variant of 142ca369
# Dimension: D19 — 符号/图例编码 (symbol_instructions)
# Change: 无显式图例——规则完全从输入输出对中推断... → 墙壁形状编码反射类型

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

"""
variant_analysis:
  concretization:
    - 保留原题的 3 像素 L 形箭头、对角线射线、颜色继承、无重叠、18-20 尺寸范围。
    - 变体只改“反射类型如何被读出”：
      size-1 墙 = 翻转行方向 dr；size-3 墙 = 翻转列方向 dc。
    - 墙依然放在射线到达某个“拐点”后的前向侧边，但“放在上下侧还是左右侧”不再决定翻哪一轴。
      同一种尺寸会在不同侧边重复出现，迫使解题者从墙的尺寸而不是朝向推断规则。
  behavior_trace:
    - 原题：射线到达 (r,c) 后，若 (r+dr,c) 有墙则翻 dr；若 (r,c+dc) 有墙则翻 dc。
    - 变体：射线到达 (r,c) 后，检查同样的两个前向侧位；若触到的是 1 像素墙则翻 dr，若触到的是 3 像素墙则翻 dc。
    - 例子：射线当前方向为 (1,1)，墙放在右侧 (r,c+1)。
      原题会翻 dc 变成 (1,-1)。
      本变体若这堵墙只有 1 像素，则仍翻 dr，方向变成 (-1,1)。
      这就是 D19 的核心差异。
  complexity_calibration:
    - 只使用两个离散指令档位 {1,3}，自由度足够小，便于从 2-4 个样例归纳。
    - 生成器强制同时出现两种尺寸，并强制出现至少一次“尺寸指令”和“几何侧位”相矛盾的放置，防止退化回原题。
    - 3 像素墙仍保持轴对齐直线，视觉上像“墙”，但功能由长度编码。
  implicit_assumption_audit:
    - 原 solver 把所有非箭头像素扁平成 wall_cells 集合，这隐含“墙功能只取决于接触方位”。
      变体下必须恢复连通域级别信息，至少保留每堵墙的 size 和 color。
    - 原 solver 的 corner-reflection 分支把对角撞墙固定写成双轴翻转，这会把旧几何规则偷偷带回来。
      变体下改成：若对角正前方就是墙，也按墙 size 指令翻对应的一轴。
    - 原 solver 的四个 side-touch 分支分别硬编码“上下墙翻 dr、左右墙翻 dc”。
      变体下这些分支只能负责“找前向侧位的墙”，不能再决定翻哪一轴。
    - 原 solver 在 bounce 时用 g[cell] == bg 判断是否重涂接触点，这个前提仍成立，保留即可。
  placement_legality:
    - 箭头、射线路径、墙体都不能重叠。
    - 不同墙体不能 4 邻接，否则 solver 会把它们合并成一个更大的“指令组件”。
    - 每堵墙只允许与自己的那个拐点射线格 4 邻接，避免额外非计划反射。
    - 3 像素墙在“非原题侧位”放置时要做成非对称端点线段，避免射线下一步直接撞到同一堵墙。
  observability:
    - 每个样本至少包含两种墙尺寸，并至少包含一次“1 像素墙放在左右侧却翻 dr”或“3 像素墙放在上下侧却翻 dc”的反例。
    - 这样输入里能看到：相同侧位的墙，有时把射线折成上下镜像，有时折成左右镜像；真正稳定的线索是墙长度。
  visual_cleanliness:
    - 保持原题的稀疏布局与单色对象，避免引入额外图例或局部覆盖。
    - 3 像素墙仍是直线块，1 像素墙仍是单点，尺寸差异肉眼明显。
  color_reasoning:
    - 颜色规则不变：射线碰墙后继承墙色。
    - 因此解题者只需额外学会“尺寸决定翻哪一轴”，不会再叠加新的颜色编码负担。
"""


SIZE_TO_FLIP = {1: "dr", 3: "dc"}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _detect_arrows_and_walls(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, list[tuple[int, int]]]], set[tuple[int, int]], dict[tuple[int, int], int], list[dict]]:
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_mask = g != int(bg)
    for r, c in arrow_cells:
        wall_mask[r, c] = False

    wall_lookup: dict[tuple[int, int], int] = {}
    wall_components: list[dict] = []
    for comp_id, coords in enumerate(components_mask(wall_mask, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        wall_components.append(
            {
                "id": int(comp_id),
                "cells": set(pts),
                "size": int(len(pts)),
                "color": int(g[pts[0][0], pts[0][1]]),
            }
        )
        for p in pts:
            wall_lookup[p] = int(comp_id)

    return bg, arrows, arrow_cells, wall_lookup, wall_components


def _apply_symbolic_bounce(
    *,
    out: np.ndarray,
    g: np.ndarray,
    bg: int,
    contact: tuple[int, int],
    comp: dict,
    dr: int,
    dc: int,
    cur_col: int,
) -> tuple[int, int, int]:
    cr, cc = contact
    wcol = int(comp["color"])
    if int(g[cr, cc]) == int(bg):
        out[cr, cc] = int(wcol)
    cur_col = int(wcol)

    flip_axis = SIZE_TO_FLIP.get(int(comp["size"]))
    if flip_axis == "dr":
        dr = -int(dr)
    elif flip_axis == "dc":
        dc = -int(dc)
    return int(cur_col), int(dr), int(dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows.

    Variant rule:
    - Rays still move diagonally and inherit wall colors on contact.
    - Reflection axis is encoded by wall size, not by wall side/orientation:
      1-pixel wall -> flip dr
      3-pixel wall -> flip dc
    - Walls are read as connected components, so the solver must preserve size information.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_lookup, wall_components = _detect_arrows_and_walls(g)

    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_lookup:
                comp = wall_components[wall_lookup[(nr, nc)]]
                cur_col, dr, dc = _apply_symbolic_bounce(
                    out=out, g=g, bg=bg, contact=cur, comp=comp, dr=dr, dc=dc, cur_col=cur_col
                )
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            touched_ids: list[int] = []
            for wr, wc in ((nr + dr, nc), (nr, nc + dc)):
                if 0 <= wr < H and 0 <= wc < W and (wr, wc) in wall_lookup:
                    comp_id = int(wall_lookup[(wr, wc)])
                    if comp_id not in touched_ids:
                        touched_ids.append(comp_id)

            for comp_id in touched_ids:
                comp = wall_components[comp_id]
                cur_col, dr, dc = _apply_symbolic_bounce(
                    out=out, g=g, bg=bg, contact=cur, comp=comp, dr=dr, dc=dc, cur_col=cur_col
                )

    return out


def _plan_wall_cells(
    bounce: tuple[int, int],
    direction: tuple[int, int],
    size: int,
    anchor_kind: str,
    shape_mode: str,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    br, bc = bounce
    dr, dc = direction

    if anchor_kind == "row":
        ar, ac = br + dr, bc
    else:
        ar, ac = br, bc + dc
    if not (0 <= ar < H and 0 <= ac < W):
        return None

    if size == 1:
        cells = [(int(ar), int(ac))]
    elif size == 3:
        if anchor_kind == "row":
            if shape_mode == "centered":
                cells = [(ar, ac - 1), (ar, ac), (ar, ac + 1)]
            else:
                cells = [(ar, ac), (ar, ac + dc), (ar, ac + 2 * dc)]
        else:
            if shape_mode == "centered":
                cells = [(ar - 1, ac), (ar, ac), (ar + 1, ac)]
            else:
                cells = [(ar, ac), (ar + dr, ac), (ar + 2 * dr, ac)]
    else:
        return None

    if not all(0 <= r < H and 0 <= c < W for r, c in cells):
        return None
    return [(int(r), int(c)) for r, c in cells]


def _wall_ok(
    *,
    wpxs: list[tuple[int, int]],
    bounce: tuple[int, int],
    occupied: set[tuple[int, int]],
    ray_set: set[tuple[int, int]],
    arrow_set: set[tuple[int, int]],
) -> bool:
    wall_set = set(wpxs)
    if len(wall_set) != len(wpxs):
        return False
    if wall_set & occupied:
        return False
    if wall_set & ray_set:
        return False
    if wall_set & arrow_set:
        return False

    for r, c in wpxs:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in wall_set:
                continue
            if (nr, nc) in occupied:
                return False
            if (nr, nc) in arrow_set:
                return False
            if (nr, nc) in ray_set and (nr, nc) != bounce:
                return False

    touching_bounce = sum(1 for r, c in wpxs if abs(r - bounce[0]) + abs(c - bounce[1]) == 1)
    return touching_bounce == 1


def _wall_color_choices(arrow_colors: list[int]) -> list[int]:
    choices = [c for c in range(1, 10) if c not in arrow_colors]
    return choices if choices else list(range(1, 10))


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid D19 variant input for task 142ca369."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 5)

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = _wall_color_choices(arrow_colors)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    size_counts = {1: 0, 3: 0}
    incongruent_count = 0
    total_bounces = 0

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]

        placed = False
        for _ in range(100):
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
            cur_dr, cur_dc, cur_col = int(dr), int(dc), int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int, int, str]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set((int(r), int(c)) for r, c in arrow_pts)
            ok = True

            local_size_counts = {1: 0, 3: 0}
            local_incongruent = 0

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces:
                    if max_steps < 3:
                        ok = False
                        break
                    seg_len = _rng.randint(3, max(3, max_steps - 1))
                else:
                    seg_len = max_steps

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                if seg_idx >= n_bounces:
                    continue

                if size_counts[1] + local_size_counts[1] < size_counts[3] + local_size_counts[3]:
                    flip = "dr"
                elif size_counts[3] + local_size_counts[3] < size_counts[1] + local_size_counts[1]:
                    flip = "dc"
                else:
                    flip = _rng.choice(["dr", "dc"])

                size = 1 if flip == "dr" else 3
                congruent_anchor = "row" if flip == "dr" else "col"
                incongruent_anchor = "col" if flip == "dr" else "row"

                anchor_options: list[tuple[str, str]] = []
                if size == 1:
                    anchor_options.extend([(incongruent_anchor, "single"), (congruent_anchor, "single")])
                else:
                    anchor_options.extend([(incongruent_anchor, "extend"), (congruent_anchor, "centered")])
                    if congruent_anchor == "col":
                        anchor_options.append((congruent_anchor, "extend"))

                if size_counts[size] + local_size_counts[size] == 0:
                    anchor_options = [opt for opt in anchor_options if opt[0] == incongruent_anchor] + [
                        opt for opt in anchor_options if opt[0] != incongruent_anchor
                    ]
                else:
                    _rng.shuffle(anchor_options)

                chosen = None
                for anchor_kind, shape_mode in anchor_options:
                    wpxs = _plan_wall_cells(
                        (cur_r, cur_c),
                        (cur_dr, cur_dc),
                        size,
                        anchor_kind,
                        shape_mode,
                        H,
                        W,
                    )
                    if wpxs is None:
                        continue
                    if not _wall_ok(
                        wpxs=wpxs,
                        bounce=(cur_r, cur_c),
                        occupied=occupied,
                        ray_set=ray_set,
                        arrow_set=arrow_set,
                    ):
                        continue
                    chosen = (wpxs, anchor_kind)
                    break

                if chosen is None:
                    ok = False
                    break

                wpxs, anchor_kind = chosen
                wcol = int(_rng.choice(wall_palette))
                walls.append((wpxs, wcol, size, anchor_kind))

                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = int(wcol)
                if size == 1:
                    cur_dr = -int(cur_dr)
                else:
                    cur_dc = -int(cur_dc)

                local_size_counts[size] += 1
                if anchor_kind != congruent_anchor:
                    local_incongruent += 1

            if not ok or len(ray_cells) < 3:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _wcol, _size, _anchor in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _c, _s, _a in walls)
            if len(new_cells) != expected_count:
                continue
            if new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol, _size, _anchor in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)

            occupied |= new_cells
            for _wpxs, _wcol, size, anchor_kind in walls:
                size_counts[size] += 1
                total_bounces += 1
                if anchor_kind != ("row" if size == 1 else "col"):
                    incongruent_count += 1

            placed = True
            break

        if not placed:
            return None

    if total_bounces < 4:
        return None
    if size_counts[1] == 0 or size_counts[3] == 0:
        return None
    if incongruent_count == 0:
        return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D19 variant of task 142ca369."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
