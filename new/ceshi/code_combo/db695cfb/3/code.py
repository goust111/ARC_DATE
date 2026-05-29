# Combo variant of db695cfb (index 3)
# combo_spec: D2/2+D3/2
# Name: 反射式延伸+遇线停止

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg

# Design analysis
# design_analysis:
#   a_concretization_and_trace:
#     - 保留原题的静态骨架：蓝色端点定义若干条 45 度主对角线，品红像素若恰好落在这些主线上，就充当触发点。
#     - 原题在触发点只会沿垂直对角线向两端发射品红线，而且无论途中遇到什么都一直扫到边界，只是跳过已有前景。
#     - 本融合版先同样补全蓝色主对角线；随后每个在线品红点都会产生四个品红分支：
#       1. 两个“透射”分支沿主对角线两端继续前进；
#       2. 两个“反射”分支沿垂直对角线两端前进。
#     - 与 D2 单变体不同，这四个分支不是一次性画到底，而是按回合同步生长。每回合每条活跃分支只尝试前进 1 格。
#     - 小例子：蓝端点在 (2,2) 和 (6,6)，触发点在 (4,4)；另有第二条蓝线占据 (1,7)-(5,3)。
#       原题：只会从 (4,4) 沿 "/" 打到边界，即使碰到别的线也继续越过去。
#       本融合版：还会出现 "\" 方向的透射分支；但 "/" 上朝右上的那一支若下一格链路最终撞到第二条蓝线，
#       就会在撞线前一格停下。若另一束品红分支同回合也想占同一空白格，则两束都停且该格保持背景。
#     - 差异因此有两层：一是新增“原方向透射”，二是分支长度由周围线条共同裁切，不再固定到边界。
#   b_complexity_calibration:
#     - 反射规则只保留一种半透镜语义：每个触发点固定分裂成 4 个对角半射线，不引入二次反射或颜色变化。
#     - 阻挡规则也只保留两类：撞到既有线条停止，或多束线同回合争抢同一背景格时全部停止。这样有交互，但档位有限。
#     - 生成器固定每条主线放 1 个真实触发点，并要求至少 2 条主线，以稳定出现“反射+截断”的可见组合效果。
#   c_implicit_assumption_audit:
#     - 原 `draw_segment()` 隐含“触发点只记录垂直方向”已失效；融合后必须同时记录主方向透射和垂直方向反射。
#     - 原发射循环隐含“各射线互不影响、顺序不重要”也失效；若继续逐条画到底，先画的射线会不公平地截断后画的射线。
#       因此这里改成全局同步 proposals：先收集所有分支下一步想去的格，再统一结算。
#     - 原代码默认“蓝线只是被填出的静态结果”；融合后蓝线还承担墙的角色。唯独透射分支允许穿过其所属主线，
#       否则会在第一步就被自己的载体蓝线挡住，无法体现“透射”。
#     - 原代码只要求不覆盖非背景；融合后新增假设是：若多个线色端点意外共用同一对角常数，solver 会把它们合并成更长主线。
#       生成器因此显式禁止不同段的蓝端点共享任一 45 度对角常数。
#   d_placement_legality:
#     - 主线段都放在网格内部，端点离边界至少 1 格，保证透射分支离开端点后确实有可见延伸空间。
#     - 不同主线的完整路径不重叠；蓝端点在两组对角键 (`r-c`, `r+c`) 上都保持互斥，避免 solver 误连额外蓝线。
#     - 干扰点只放在“无停止版本”的全体输出背景区，因此不会被任何主线或分支触及，也不会伪造新的触发关系。
#   e_visual_observability:
#     - 每个接受样本都强制包含：
#       1. 至少 1 个真正被涂出的透射像素；
#       2. 至少 1 个真正被涂出的反射像素；
#       3. 至少 1 次因撞线或同回合争抢而发生的提前停止。
#     - 这样解题者能在少量样例里同时看到“比原题多出的同向延伸”和“没有画到边界而是被别的线截断”。
#   f_visual_tidiness:
#     - 只用蓝色主线和品红分支两种前景职责，不额外引入新功能色。
#     - 通过限制为 2-3 条主线、每条 1 个触发点，并避开共享主对角线，输出保持为几条清晰的对角光路，而不是大面积网格噪声。
#   g_color_inference:
#     - 颜色规则不变：蓝色仍定义主线，品红仍负责触发并染色所有新增分支。
#     - 因而变化完全来自几何和同步阻挡，不依赖颜色编号、排序或隐藏元数据。
#   post_change_solver_assumptions:
#     - solver 新假设 1：透射分支只应穿过“自己的”那条主线，不应穿过别的蓝线；generator 通过逐段独立放置保证这一语义清晰。
#     - solver 新假设 2：若多束线同回合抢同一背景格，则该格留白且所有相关分支终止；generator 允许这种情况出现，但不会依赖
#       更细的同回合 tie-break。
#     - solver 新假设 3：在线触发点不会同时属于两条不同主线；generator 通过路径互斥保证这一点。


_LINE_COLOR = 1
_TRIGGER_COLOR = 6

_H_RANGE = (9, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (2, 3)
_SEG_LEN_RANGE = (4, 12)
_TRIGGERS_PER_SEG = (1, 1)

_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


class _Beam:
    __slots__ = ("pos", "direction", "color", "through", "kind")

    def __init__(
        self,
        pos: tuple[int, int],
        direction: tuple[int, int],
        color: int,
        through: frozenset[tuple[int, int]],
        kind: str,
    ) -> None:
        self.pos = pos
        self.direction = direction
        self.color = int(color)
        self.through = through
        self.kind = kind


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _find_line_color(
    by_color: dict[int, list[tuple[int, int]]],
) -> tuple[int, list[tuple[int, int]]]:
    best = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), int(col))
                if best is None or cand > best:
                    best = cand
    assert best is not None, "at least one diagonal-aligned same-color pair must exist"
    line_color = int(best[1])
    return line_color, by_color[line_color]


def _extract_segments(pts: list[tuple[int, int]]):
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segments = []
    for groups in (groups_a, groups_b):
        for group_pts in groups.values():
            if len(group_pts) < 2:
                continue
            group_pts = sorted(group_pts)
            p = group_pts[0]
            q = group_pts[-1]
            path = _diag_path(p, q)
            dr = 1 if q[0] > p[0] else -1
            dc = 1 if q[1] > p[1] else -1
            segments.append(
                {
                    "path": path,
                    "path_set": frozenset(path),
                    "main_dirs": ((dr, dc), (-dr, -dc)),
                    "reflect_dirs": ((dr, -dc), (-dr, dc)),
                }
            )
    return segments


def _prepare_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    non_bg = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if int(g[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(int(col), []).append((r, c))

    line_color, pts = _find_line_color(by_color)
    segments = _extract_segments(pts)
    assert segments, "expected at least one segment after grouping line-color pixels"
    return g, bg, line_color, segments


def _simulate(
    grid: np.ndarray,
    *,
    include_transmission: bool,
    stop_on_lines: bool,
    collect_stats: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    g, bg, line_color, segments = _prepare_scene(grid)
    h, w = g.shape
    out = g.copy()
    beams: list[_Beam] = []

    for seg in segments:
        path = seg["path"]
        path_set = seg["path_set"]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                if include_transmission:
                    for ddr, ddc in seg["main_dirs"]:
                        beams.append(_Beam((r, c), (ddr, ddc), v, path_set, "transmission"))
                for ddr, ddc in seg["reflect_dirs"]:
                    beams.append(_Beam((r, c), (ddr, ddc), v, frozenset(), "reflection"))

    stats = {
        "online_triggers": len(beams) // (4 if include_transmission else 2),
        "total_beams": len(beams),
        "transmission_painted": 0,
        "reflection_painted": 0,
        "blocked_by_fixed": 0,
        "blocked_by_magenta": 0,
        "contested_cells": 0,
        "contested_beams": 0,
        "boundary_stops": 0,
    }

    if not stop_on_lines:
        for beam in beams:
            r, c = beam.pos
            dr, dc = beam.direction
            while True:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    stats["boundary_stops"] += 1
                    break
                if (nr, nc) in beam.through:
                    r, c = nr, nc
                    continue
                if int(out[nr, nc]) == bg:
                    out[nr, nc] = beam.color
                    if beam.kind == "transmission":
                        stats["transmission_painted"] += 1
                    else:
                        stats["reflection_painted"] += 1
                r, c = nr, nc
        return out, stats if collect_stats else {}

    fixed_blockers = {(int(r), int(c)) for r, c in np.argwhere(out != bg)}
    painted_magenta: set[tuple[int, int]] = set()
    active = beams[:]

    for _ in range(h * w * 4):
        if not active:
            break

        proposals: dict[tuple[int, int], list[_Beam]] = {}
        next_active: list[_Beam] = []

        for beam in active:
            r, c = beam.pos
            dr, dc = beam.direction
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                stats["boundary_stops"] += 1
                continue

            nxt = (nr, nc)
            if nxt in beam.through:
                beam.pos = nxt
                next_active.append(beam)
                continue
            if nxt in fixed_blockers:
                stats["blocked_by_fixed"] += 1
                continue
            if nxt in painted_magenta:
                stats["blocked_by_magenta"] += 1
                continue
            proposals.setdefault(nxt, []).append(beam)

        for cell, contenders in proposals.items():
            if len(contenders) > 1:
                stats["contested_cells"] += 1
                stats["contested_beams"] += len(contenders)
                continue
            beam = contenders[0]
            out[cell] = beam.color
            painted_magenta.add(cell)
            beam.pos = cell
            next_active.append(beam)
            if beam.kind == "transmission":
                stats["transmission_painted"] += 1
            else:
                stats["reflection_painted"] += 1

        active = next_active

    return out, stats if collect_stats else {}


def _combo_output(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    return _simulate(grid, include_transmission=True, stop_on_lines=True, collect_stats=True)


def _reflection_only_output(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    return _simulate(grid, include_transmission=True, stop_on_lines=False, collect_stats=True)


def _baseline_output(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    return _simulate(grid, include_transmission=False, stop_on_lines=False, collect_stats=True)


# Overview:
#   Complete the blue diagonal backbones, then let every on-line trigger split into transmitted and reflected
#   magenta branches that advance synchronously and stop when they meet other lines.
# Input structure:
#   The dominant color is background. One special non-background color forms endpoint pairs on 45-degree
#   diagonals, and differently colored pixels may lie directly on those diagonals as triggers.
# Rule:
#   Fill each diagonal between the extreme same-color endpoints. Every trigger on such a diagonal creates
#   four diagonal magenta branches: two continue along the backbone direction and two reflect into the
#   perpendicular diagonal direction. These branches grow one step per round; a branch may pass through its
#   own backbone, but it stops before any other existing line, before any previously painted magenta branch,
#   and any cell claimed by multiple branches in the same round stays unchanged while all claimants stop.
# Key insight:
#   The output is determined by a synchronous light-path simulation, not by drawing independent rays in an
#   arbitrary order. The visible length of each branch depends on which other blue or magenta line reaches a
#   would-be next cell first.
def main(grid: np.ndarray) -> np.ndarray:
    return _combo_output(grid)[0]


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    used_a_keys: set[int],
    used_b_keys: set[int],
    occupied_paths: set[tuple[int, int]],
):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 2 - seg_len < 1 or W - 2 - seg_len < 1:
                return None
            r0 = random.randint(1, H - 2 - seg_len)
            c0 = random.randint(1, W - 2 - seg_len)
            dc = 1
        else:
            if H - 2 - seg_len < 1 or seg_len + 1 > W - 2:
                return None
            r0 = random.randint(1, H - 2 - seg_len)
            c0 = random.randint(seg_len + 1, W - 2)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(cell in occupied_paths for cell in path):
            continue

        ep0 = path[0]
        ep1 = path[-1]
        a0, a1 = ep0[0] - ep0[1], ep1[0] - ep1[1]
        b0, b1 = ep0[0] + ep0[1], ep1[0] + ep1[1]
        if a0 in used_a_keys or a1 in used_a_keys or b0 in used_b_keys or b1 in used_b_keys:
            continue
        return path, dc
    return None


def _build_input(
    H: int,
    W: int,
    bg: int,
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]],
) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _path in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR
    return inp


def _place_distractors(
    inp: np.ndarray,
    bg: int,
    unsafe: set[tuple[int, int]],
    used_a_keys: set[int],
    used_b_keys: set[int],
) -> None:
    H, W = inp.shape
    occupied = {(int(r), int(c)) for r, c in np.argwhere(inp != bg)}

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in unsafe:
                continue
            if (r - c) in used_a_keys or (r + c) in used_b_keys:
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            used_a_keys.add(r - c)
            used_b_keys.add(r + c)
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in unsafe:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(H - 3, W - 3, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    used_a_keys: set[int] = set()
    used_b_keys: set[int] = set()
    occupied_paths: set[tuple[int, int]] = set()
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]] = []

    target_segments = random.randint(*_SEGMENT_COUNT)
    for _ in range(target_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
        result = _try_place_segment(
            slope_type,
            seg_len,
            H,
            W,
            used_a_keys,
            used_b_keys,
            occupied_paths,
        )
        if result is None:
            continue

        path, _dc = result
        interior = path[1:-1]
        n_triggers = random.randint(*_TRIGGERS_PER_SEG)
        if len(interior) < n_triggers:
            continue
        trigger_positions = random.sample(interior, n_triggers)

        ep0 = path[0]
        ep1 = path[-1]
        segments.append((ep0, ep1, trigger_positions, path))
        occupied_paths.update(path)
        used_a_keys.add(ep0[0] - ep0[1])
        used_a_keys.add(ep1[0] - ep1[1])
        used_b_keys.add(ep0[0] + ep0[1])
        used_b_keys.add(ep1[0] + ep1[1])

    if len(segments) < _SEGMENT_COUNT[0]:
        return None

    inp = _build_input(H, W, bg, segments)

    try:
        combo_out, combo_stats = _combo_output(inp)
        reflect_out, _ = _reflection_only_output(inp)
        baseline_out, _ = _baseline_output(inp)
    except Exception:
        return None

    stop_events = (
        combo_stats["blocked_by_fixed"]
        + combo_stats["blocked_by_magenta"]
        + combo_stats["contested_cells"]
    )
    if combo_stats["transmission_painted"] == 0 or combo_stats["reflection_painted"] == 0:
        return None
    if stop_events == 0:
        return None
    if np.array_equal(combo_out, reflect_out):
        return None
    if np.array_equal(combo_out, baseline_out):
        return None

    unsafe = {(int(r), int(c)) for r, c in np.argwhere(reflect_out != bg)}
    _place_distractors(inp, bg, unsafe, used_a_keys.copy(), used_b_keys.copy())

    try:
        combo_out_2, combo_stats_2 = _combo_output(inp)
        reflect_out_2, _ = _reflection_only_output(inp)
        baseline_out_2, _ = _baseline_output(inp)
    except Exception:
        return None

    stop_events_2 = (
        combo_stats_2["blocked_by_fixed"]
        + combo_stats_2["blocked_by_magenta"]
        + combo_stats_2["contested_cells"]
    )
    if combo_stats_2["transmission_painted"] == 0 or combo_stats_2["reflection_painted"] == 0:
        return None
    if stop_events_2 == 0:
        return None
    if np.array_equal(combo_out_2, reflect_out_2):
        return None
    if np.array_equal(combo_out_2, baseline_out_2):
        return None
    if not np.array_equal(main(inp), combo_out_2):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
