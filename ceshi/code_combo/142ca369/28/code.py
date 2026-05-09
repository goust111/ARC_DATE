# Combo variant of 142ca369 (index 28)
# combo_spec: D7/0+D19/1
# Name: 墙壁形状编码指令+L/T形墙壁

from __future__ import annotations

import random as _rng
import sys

import numpy as np

try:
    from arc_utils import colors_present, components_mask, get_bg
except ImportError:  # pragma: no cover - fallback for direct file execution
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import colors_present, components_mask, get_bg

"""
design_analysis:
  a_concretization_and_trace:
    - 保留原题的 3 像素 2x2 L 形箭头、对角线射线、碰墙继承颜色、走到出界结束。
    - 墙不再是单点/直线，而是 4 像素 L 或 T 多格组件。求解时先把每个墙连通域识别成
      “L 墙”或“T 墙”，再由形状决定反射轴：
        L 墙 -> 翻转 dr（行方向）
        T 墙 -> 翻转 dc（列方向）
    - 例子：射线到达 (r,c)，当前方向 (dr,dc)=(+1,+1)。
      原题若右侧 (r,c+1) 有墙，就翻 dc，离开方向变成 (+1,-1)。
      融合后如果右侧贴着的是一个 L 形墙，虽然它放在“列侧”，仍然翻 dr，方向变成 (-1,+1)。
      如果同样放在右侧的是一个 T 形墙，则翻 dc，方向变成 (+1,-1)。
      所以“接触在哪一侧”不再足够，必须读出墙的整体形状。
    - 对角正前方直接撞进墙组件时，也不再沿用原题的“双轴翻转”隐规则，而是仍按该墙的
      形状指令翻一轴。这样 D19 的“符号编码”真正覆盖了所有碰墙入口。
  b_complexity_calibration:
    - 只保留两类墙形状 {L, T}，并且只保留两条指令 {flip dr, flip dc}，自由度足够低，
      但与原题相比视觉差异非常明显。
    - 生成器强制同一组样本里出现“同侧不同形”的墙：同样贴在 row-side 或 col-side，
      L 和 T 会导致不同转向。还强制出现至少两个“形状指令与原题侧位规则矛盾”的反例，
      防止退化回“哪边接触就翻哪轴”。
  c_implicit_assumption_audit:
    - 原 solver 把所有墙像素扁平化成 wall_cells，并把翻哪一轴写死在 side-touch 分支里。
      这在融合后失效，因为 axis 不再由 side 决定，而由整块墙的形状决定。
    - 原 solver 的 corner-hit 分支固定双轴翻转，这会偷偷把旧规则带回来；这里改成仍读取
      墙组件形状。
    - 原 solver 默认“知道某个单像素是否是墙就够了”。融合后必须保留组件级信息：
      每个组件的 cells、color、kind(L/T) 都要跟踪。
    - 新 solver 额外假设：一个计划中的反射点最多只被一个墙组件触发，避免多组件同时接触
      带来的颜色/顺序歧义。generator 会显式保证这一点。
  d_placement_legality:
    - 箭头、射线路径、墙组件都不能重叠，所有墙都在网格内。
    - 墙组件之间不能 4 邻接，否则会在 wall_mask 上合并成一个更大的连通域。
    - 墙组件只能与自己的 bounce cell 4 邻接；不能侵入任何已提交射线格的“前向触发槽位”，
      否则会给别的射线制造隐藏反射。
    - 墙颜色与箭头颜色分离，避免 per-color 箭头检测被同色墙粘连破坏。
  e_visual_observability:
    - 每个样本至少含一个同侧 L/T 对照组，以及至少两个 incongruent bounces：
      例如墙放在 col-side，但 L 仍翻 dr；或墙放在 row-side，但 T 仍翻 dc。
    - 这样人类在 2-4 个样例里能直接观察到：真正稳定的决定因素是墙轮廓，不是墙在射线的
      上下/左右哪一边。
  f_visual_cleanliness:
    - 只使用稀疏摆放的单色箭头和单色墙组件，不叠加额外图例。
    - 每个墙只贴一个 bounce cell，避免多个组件挤在同一拐点附近。
  g_inferability_of_change:
    - 颜色规则完全沿用原题：射线一旦碰墙，就改成墙色继续延伸。
    - 因此新增推理负担只有“读形状 -> 选翻转轴”，而不是再引入新的颜色语义。
"""


_BASE_L4 = {(0, 0), (1, 0), (2, 0), (2, 1)}
_BASE_T4 = {(0, 0), (0, 1), (0, 2), (1, 1)}
_SHAPE_TO_FLIP = {"L": "dr", "T": "dc"}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(
    cells: set[tuple[int, int]], rot: int, reflect: bool
) -> set[tuple[int, int]]:
    out = set(cells)
    if reflect:
        out = {(r, -c) for r, c in out}
    for _ in range(rot % 4):
        out = {(c, -r) for r, c in out}
    return set(_normalize_shape(out))


def _all_orientations(
    base: set[tuple[int, int]], *, allow_reflect: bool
) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for rot in range(4):
        variants.add(_normalize_shape(_transform_shape(base, rot, False)))
        if allow_reflect:
            variants.add(_normalize_shape(_transform_shape(base, rot, True)))
    return sorted(variants)


_WALL_VARIANTS = {
    "L": _all_orientations(_BASE_L4, allow_reflect=True),
    "T": _all_orientations(_BASE_T4, allow_reflect=False),
}


def _max_diag_steps(H: int, W: int, r: int, c: int, dr: int, dc: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(
    pts: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
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
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _component_kind(cells: set[tuple[int, int]]) -> str | None:
    if len(cells) != 4:
        return None
    degs = sorted(
        sum((nr, nc) in cells for nr, nc in _neighbors4(r, c)) for r, c in cells
    )
    if degs == [1, 1, 2, 2]:
        return "L"
    if degs == [1, 1, 1, 3]:
        return "T"
    return None


def _detect_objects(g: np.ndarray):
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
    wall_components: dict[int, dict] = {}
    wid = 0
    for coords in components_mask(wall_mask, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if not pts:
            continue
        kind = _component_kind(pts)
        if kind is None:
            continue
        wall_components[wid] = {
            "id": int(wid),
            "cells": pts,
            "kind": kind,
            "color": int(g[next(iter(pts))[0], next(iter(pts))[1]]),
        }
        for cell in pts:
            wall_lookup[cell] = int(wid)
        wid += 1

    return bg, arrows, arrow_cells, wall_lookup, wall_components


def _apply_component_bounce(
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

    flip_axis = _SHAPE_TO_FLIP.get(str(comp["kind"]))
    if flip_axis == "dr":
        dr = -int(dr)
    elif flip_axis == "dc":
        dc = -int(dc)
    return int(cur_col), int(dr), int(dc)


def _run_solver(
    grid: np.ndarray, *, mode: str, collect_events: bool = False
) -> tuple[np.ndarray, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_lookup, wall_components = _detect_objects(g)
    out = g.copy()
    events: list[dict] = []

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
                comp_id = int(wall_lookup[(nr, nc)])
                comp = wall_components[comp_id]
                if collect_events:
                    events.append(
                        {
                            "type": "diag",
                            "kind": str(comp["kind"]),
                            "anchor": "diag",
                            "row_touch": False,
                            "col_touch": False,
                        }
                    )
                if mode == "symbolic":
                    cur_col, dr, dc = _apply_component_bounce(
                        out=out,
                        g=g,
                        bg=bg,
                        contact=cur,
                        comp=comp,
                        dr=dr,
                        dc=dc,
                        cur_col=cur_col,
                    )
                else:
                    wcol = int(comp["color"])
                    if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                        out[int(cur[0]), int(cur[1])] = int(wcol)
                    cur_col = int(wcol)
                    dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            row_cell = (nr + int(dr), nc)
            col_cell = (nr, nc + int(dc))
            row_touch_id = (
                int(wall_lookup[row_cell])
                if 0 <= row_cell[0] < H and 0 <= row_cell[1] < W and row_cell in wall_lookup
                else None
            )
            col_touch_id = (
                int(wall_lookup[col_cell])
                if 0 <= col_cell[0] < H and 0 <= col_cell[1] < W and col_cell in wall_lookup
                else None
            )

            touched_ids: list[int] = []
            for comp_id in (row_touch_id, col_touch_id):
                if comp_id is not None and comp_id not in touched_ids:
                    touched_ids.append(int(comp_id))

            if not touched_ids:
                continue

            best_id = touched_ids[0]
            comp = wall_components[best_id]
            if collect_events:
                if row_touch_id is not None and col_touch_id is not None and row_touch_id == col_touch_id:
                    anchor = "both"
                elif row_touch_id is not None and col_touch_id is None:
                    anchor = "row"
                elif col_touch_id is not None and row_touch_id is None:
                    anchor = "col"
                else:
                    anchor = "multi"
                events.append(
                    {
                        "type": "side",
                        "kind": str(comp["kind"]),
                        "anchor": anchor,
                        "row_touch": row_touch_id is not None,
                        "col_touch": col_touch_id is not None,
                    }
                )

            if mode == "symbolic":
                cur_col = int(comp["color"])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(cur_col)
                if any(wall_components[comp_id]["kind"] == "L" for comp_id in touched_ids):
                    dr = -int(dr)
                if any(wall_components[comp_id]["kind"] == "T" for comp_id in touched_ids):
                    dc = -int(dc)
            else:
                wcol = int(comp["color"])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if row_touch_id is not None:
                    dr = -int(dr)
                if col_touch_id is not None:
                    dc = -int(dc)

    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from 3-cell L-shaped arrows.

    Combo rule:
    - Walls are 4-cell L or T polyominoes.
    - Reflection instruction is encoded by wall shape, not by wall side:
      L wall -> flip dr
      T wall -> flip dc
    - Side-touch and diagonal-hit both use the same symbolic shape instruction.
    """
    out, _events = _run_solver(grid, mode="symbolic", collect_events=False)
    return out


def _baseline_side_solver(grid: np.ndarray) -> np.ndarray:
    """Shape-blind comparison model: row-side flips dr, col-side flips dc."""
    out, _events = _run_solver(grid, mode="baseline", collect_events=False)
    return out


def _touch_positions_from_states(
    states: list[tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    touch = set()
    for r, c, dr, dc in states:
        touch.add((int(r + dr), int(c)))
        touch.add((int(r), int(c + dc)))
    return touch


def _relative_wall_candidates(
    kind: str, dr: int, dc: int, anchor: str
) -> list[tuple[tuple[int, int], ...]]:
    required = (dr, 0) if anchor == "row" else (0, dc)
    forbidden = (0, dc) if anchor == "row" else (dr, 0)
    opts: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for variant in _WALL_VARIANTS[kind]:
        pts = list(variant)
        for vr, vc in pts:
            shift_r = required[0] - vr
            shift_c = required[1] - vc
            shifted = tuple(sorted((r + shift_r, c + shift_c) for r, c in pts))
            shifted_set = set(shifted)
            if required not in shifted_set or forbidden in shifted_set or (0, 0) in shifted_set:
                continue
            if sum(abs(r) + abs(c) == 1 for r, c in shifted_set) != 1:
                continue
            if shifted not in seen:
                seen.add(shifted)
                opts.append(shifted)
    return opts


def _wall_ok(
    *,
    wall_cells: set[tuple[int, int]],
    bounce: tuple[int, int],
    occupied_all: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    ray_set: set[tuple[int, int]],
    arrow_set: set[tuple[int, int]],
    reserved_touch: set[tuple[int, int]],
) -> bool:
    if wall_cells & occupied_all:
        return False
    if wall_cells & ray_set:
        return False
    if wall_cells & arrow_set:
        return False
    if wall_cells & reserved_touch:
        return False
    if len(wall_cells) != 4:
        return False

    touching_bounce = 0
    for r, c in wall_cells:
        if abs(r - bounce[0]) + abs(c - bounce[1]) == 1:
            touching_bounce += 1
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in wall_cells:
                continue
            if (nr, nc) in occupied_input or (nr, nc) in arrow_set:
                return False
            if (nr, nc) in ray_set and (nr, nc) != bounce:
                return False
    return touching_bounce == 1


def _trace_segment(
    *,
    H: int,
    W: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    color: int,
    steps: int,
    occupied_all: set[tuple[int, int]],
    occupied_walls: set[tuple[int, int]],
    ray_set: set[tuple[int, int]],
    arrow_set: set[tuple[int, int]],
    local_walls: set[tuple[int, int]],
    ray_cells: list[tuple[int, int, int]],
    ray_states: list[tuple[int, int, int, int]],
) -> tuple[bool, int, int]:
    cur_r, cur_c = int(start_r), int(start_c)
    for _ in range(steps):
        nr, nc = cur_r + dr, cur_c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            return False, cur_r, cur_c
        if (nr, nc) in occupied_all or (nr, nc) in ray_set or (nr, nc) in arrow_set:
            return False, cur_r, cur_c
        if (nr + dr, nc) in occupied_walls or (nr + dr, nc) in local_walls:
            return False, cur_r, cur_c
        if (nr, nc + dc) in occupied_walls or (nr, nc + dc) in local_walls:
            return False, cur_r, cur_c
        ray_cells.append((int(nr), int(nc), int(color)))
        ray_states.append((int(nr), int(nc), int(dr), int(dc)))
        ray_set.add((int(nr), int(nc)))
        cur_r, cur_c = int(nr), int(nc)
    return True, int(cur_r), int(cur_c)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    symbolic_out, events = _run_solver(grid, mode="symbolic", collect_events=True)
    baseline_out = _baseline_side_solver(grid)

    side_bounces = 0
    incongruent = 0
    same_side_kinds = {"row": set(), "col": set()}
    L_count = 0
    T_count = 0

    for event in events:
        if event["type"] != "side":
            continue
        if event["kind"] == "L":
            L_count += 1
        elif event["kind"] == "T":
            T_count += 1
        if event["anchor"] not in {"row", "col"}:
            continue
        side_bounces += 1
        same_side_kinds[event["anchor"]].add(event["kind"])
        symbolic_axis = _SHAPE_TO_FLIP[event["kind"]]
        baseline_axis = "dr" if event["anchor"] == "row" else "dc"
        if symbolic_axis != baseline_axis:
            incongruent += 1

    same_side_pair = int(
        any(kinds == {"L", "T"} for kinds in same_side_kinds.values())
    )
    baseline_diff = int(np.count_nonzero(symbolic_out != baseline_out))
    return {
        "side_bounces": int(side_bounces),
        "incongruent_bounces": int(incongruent),
        "same_side_pair": int(same_side_pair),
        "L_count": int(L_count),
        "T_count": int(T_count),
        "baseline_diff": int(baseline_diff),
    }


def _forced_plans() -> list[tuple[str, str]]:
    paired_anchor = _rng.choice(["row", "col"])
    if paired_anchor == "col":
        return [("L", "col"), ("T", "col"), ("T", "row")]
    return [("L", "row"), ("T", "row"), ("L", "col")]


def _gen_attempt() -> np.ndarray | None:
    """Output-first generator for the D7/0 + D19/1 combo variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = 3

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = [col for col in range(1, 10) if col not in arrow_colors]
    if len(wall_palette) < 2:
        wall_palette = list(range(1, 10))

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied_all: set[tuple[int, int]] = set()
    occupied_input: set[tuple[int, int]] = set()
    occupied_walls: set[tuple[int, int]] = set()
    committed_ray_states: list[tuple[int, int, int, int]] = []

    plans = _forced_plans()

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        wall_kind, anchor = plans[ai]
        placed = False

        for _ in range(240):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [
                (r, c)
                for r in (mr, sr)
                for c in (mc, sc)
                if (r, c) != (mr, mc)
            ]
            arrow_set = {(int(r), int(c)) for r, c in arrow_pts}
            if any(
                not (0 <= r < H and 0 <= c < W) or (r, c) in occupied_all
                for r, c in arrow_set
            ):
                continue

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(color)

            ray_cells: list[tuple[int, int, int]] = []
            ray_states: list[tuple[int, int, int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            local_walls: set[tuple[int, int]] = set()
            walls: list[tuple[set[tuple[int, int]], int]] = []

            max_before = _max_diag_steps(H, W, cur_r, cur_c, cur_dr, cur_dc)
            if max_before < 4:
                continue
            seg_len = _rng.randint(3, max_before - 1)

            ok, cur_r, cur_c = _trace_segment(
                H=H,
                W=W,
                start_r=cur_r,
                start_c=cur_c,
                dr=cur_dr,
                dc=cur_dc,
                color=cur_col,
                steps=seg_len,
                occupied_all=occupied_all,
                occupied_walls=occupied_walls,
                ray_set=ray_set,
                arrow_set=arrow_set,
                local_walls=local_walls,
                ray_cells=ray_cells,
                ray_states=ray_states,
            )
            if not ok or not ray_cells:
                continue

            bounce = (int(cur_r), int(cur_c))
            reserved_touch = _touch_positions_from_states(
                committed_ray_states + ray_states[:-1]
            )
            candidates: list[tuple[set[tuple[int, int]], tuple[int, int]]] = []
            for rel_cells in _relative_wall_candidates(wall_kind, cur_dr, cur_dc, anchor):
                wall_cells = {(cur_r + rr, cur_c + cc) for rr, cc in rel_cells}
                if any(not (0 <= r < H and 0 <= c < W) for r, c in wall_cells):
                    continue
                if not _wall_ok(
                    wall_cells=wall_cells,
                    bounce=bounce,
                    occupied_all=occupied_all,
                    occupied_input=occupied_input,
                    ray_set=ray_set,
                    arrow_set=arrow_set,
                    reserved_touch=reserved_touch,
                ):
                    continue
                next_dr = -cur_dr if wall_kind == "L" else cur_dr
                next_dc = cur_dc if wall_kind == "L" else -cur_dc
                if (cur_r + next_dr, cur_c + next_dc) in wall_cells:
                    continue
                future_steps = _max_diag_steps(H, W, cur_r, cur_c, next_dr, next_dc)
                if future_steps < 2:
                    continue
                candidates.append((wall_cells, (int(next_dr), int(next_dc))))

            if not candidates:
                continue

            wall_cells, (next_dr, next_dc) = _rng.choice(candidates)
            wcol = int(_rng.choice(wall_palette))
            walls.append((wall_cells, wcol))
            local_walls |= wall_cells

            lr, lc, _ = ray_cells[-1]
            ray_cells[-1] = (int(lr), int(lc), int(wcol))
            cur_col = int(wcol)
            cur_dr, cur_dc = int(next_dr), int(next_dc)

            future_steps = _max_diag_steps(H, W, cur_r, cur_c, cur_dr, cur_dc)
            ok, cur_r, cur_c = _trace_segment(
                H=H,
                W=W,
                start_r=cur_r,
                start_c=cur_c,
                dr=cur_dr,
                dc=cur_dc,
                color=cur_col,
                steps=future_steps,
                occupied_all=occupied_all,
                occupied_walls=occupied_walls,
                ray_set=ray_set,
                arrow_set=arrow_set,
                local_walls=local_walls,
                ray_cells=ray_cells,
                ray_states=ray_states,
            )
            if not ok or len(ray_cells) < 5:
                continue

            new_cells = set(arrow_set) | set(ray_set) | set(local_walls)
            if new_cells & occupied_all:
                continue
            if len(new_cells) != len(arrow_set) + len(ray_set) + len(local_walls):
                continue

            for r, c in arrow_set:
                grid[r, c] = out[r, c] = int(color)
            for r, c, ray_col in ray_cells:
                out[r, c] = int(ray_col)
            for wall_pts, wall_col in walls:
                for r, c in wall_pts:
                    grid[r, c] = out[r, c] = int(wall_col)

            occupied_all |= new_cells
            occupied_input |= set(arrow_set) | set(local_walls)
            occupied_walls |= set(local_walls)
            committed_ray_states.extend(ray_states)
            placed = True
            break

        if not placed:
            return None

    solved = main(grid)
    if not np.array_equal(solved, out):
        return None

    metrics = _fusion_metrics(grid)
    if metrics["L_count"] == 0 or metrics["T_count"] == 0:
        return None
    if metrics["incongruent_bounces"] < 2:
        return None
    if metrics["same_side_pair"] == 0:
        return None
    if metrics["baseline_diff"] < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the combo variant of task 142ca369."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
