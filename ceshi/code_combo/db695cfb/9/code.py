# Combo variant of db695cfb (index 9)
# combo_spec: D3/1+D16/2
# Name: 固定长度延伸+单方向

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


"""
design_analysis:
  a_concretization_and_trace:
    - 原题先找出同色蓝端点对，补全整条 45 度蓝对角线；遇到对角线上的品红触发点时，
      会沿垂直对角方向向两边一直打到边界。
    - 本融合版保留“补全蓝对角线”这一步，但把品红规则改成双重编码：每个触发点只打一条垂直对角线，
      且长度不再到边界，而是等于该触发点沿蓝段到最近蓝端点的步数。
    - 具体例子：蓝端点在 (1,1) 和 (7,7)，路径长度为 6；触发点在 (3,3) 与 (5,5)。
      原题会让两个触发点都沿 slash 方向的两侧打到边界。
      本题先按蓝段从上到下给触发点编号：第 1 个触发点 (3,3) 只向“上侧”发射，
      第 2 个触发点 (5,5) 只向“下侧”发射；两者长度都等于到最近蓝端点的距离 2。
      所以输出会出现两段短品红线，而且分别落在对角线的相反单侧。
  b_complexity_calibration:
    - 方向规则只保留一个低自由度分支：沿同一蓝段从上到下交替，奇数触发点走上侧，
      偶数触发点走下侧。这样样例里能重复出现同一规则，不会退化成难以归纳的逐点特判。
    - 长度规则也只依赖一个可见数字：触发点到最近蓝端点的对角距离。它和方向都由同一条蓝段上的位置给出，
      形成真正耦合，而不是两套互不相关的附加条件。
    - 为了防止视觉上退化回原题或任一单变体，generator 强制每个有效样例至少有两个同段触发点，
      且这些触发点的固定长度必须短于其单侧到边界的距离。
  c_implicit_assumption_audit:
    - 选 line color 的逻辑仍依赖“最长同色对角对来自真实蓝端点”。生成器继续让蓝端点段长
      严格大于任何品红对角配对，并禁止蓝色干扰点再次成对。
    - 按 diagonal key 聚组并连接极值点，隐含假设同一 key 上至多一个真实段。生成器继续对两种斜率
      分别去重 diagonal id，避免多段被错误合并。
    - 原始 draw_segment() 把每个触发点映射成两条方向；这个前提在融合后失效，必须先收集段内触发点顺序，
      再按顺序决定单方向，并同时记录对应固定长度。
    - 最后画射线的循环现在假设每个方向都附带一个长度；生成器保证所有被选触发点的该长度射线完整在界内，
      且长度之后至少还留一格背景，从而让“固定长度停止”肉眼可见。
  d_placement_legality:
    - 仍采用 output-first：先挑蓝段，再挑段内触发点，再预留蓝段填充区、真实射线路径和可见空隙。
    - 由于单方向规则需要让“未发射的一侧”也保持可见，生成器额外预留每个触发点另一侧的等长空白路径，
      防止后续段或干扰像素占进去，造成“看起来像被遮挡”。
    - 所有候选样本都经过新 main() 与 oracle 对比校验。
  e_visual_observability:
    - 解题者能在同一条蓝段上同时看到两个证据：奇偶触发点落在相反单侧，以及这些品红短线都与
      各自到最近蓝端点的距离等长。
    - 样本筛选拒绝两类退化：若融合输出与“单方向但打到边界”的输出一致，则 D3 不可见；
      若融合输出与“双方向但固定长度”的输出一致，则 D16 不可见。
  f_visual_cleanliness:
    - 段数降到 1-3，干扰点数也收紧，并拒绝真实射线、隐藏对侧空白带、蓝段填充之间的任何冲突。
    - 每条短射线后都保留一格背景间隙，终止位置不会贴着边界或别的对象。
  g_inferability:
    - 没有新增颜色语义；蓝仍负责定义载体对角线，品红仍负责触发垂直对角线。
    - 变化完全由可见几何关系决定：长度来自“离哪个端点更近”，方向来自“同段内排第几”。
  h_new_main_assumptions_checked:
    - main() 现在默认段内触发顺序按 path 从上到下可稳定恢复；生成器按同样顺序建 oracle。
    - main() 默认有效触发点不会被别的线段共享；生成器通过 reserved 集避免段交叉和射线穿越。
    - main() 对异常测试输入仍做保守处理：若同一触发点因意外被多个段命中，则按方向保留最大长度；
      生成器不会产生这种情况，但该兼容性避免 solver 崩溃。
"""


# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (6, 16)
_TRIGGERS_PER_SEG = (2, 4)
_MIN_VISIBLE_RAY_LEN = 2

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_TRIGGER_SUBSET_RETRIES = 80
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _perp_dirs_for_dc(dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    upward_dir = (-1, dc)
    downward_dir = (1, -dc)
    return upward_dir, downward_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color diagonal endpoint pairs, then let on-segment trigger pixels emit one finite perpendicular ray.

    Input structure:
    - The most common color is the background. One non-background color appears as endpoint pairs on 45-degree diagonals, and magenta trigger pixels may sit strictly between those endpoints on the same diagonal.

    Rule:
    - Fill each detected blue diagonal segment from one extreme aligned endpoint to the other.
    - Traverse that completed segment from top to bottom. Every non-blue pixel on the segment is a trigger. Its ray length equals its diagonal distance to the nearer blue endpoint, and its single perpendicular ray direction alternates by trigger order: 1st, 3rd, 5th... use the upward side; 2nd, 4th, 6th... use the downward side. Draw only that many cells and never overwrite non-background pixels.

    Key insight:
    - A trigger's position on the blue carrier diagonal now encodes two outputs at once: closeness to an endpoint gives the finite ray length, while odd/even order among triggers on the same carrier chooses which one of the two perpendicular sides is painted.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (dist, color)
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
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    triggers: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dc = 1 if q[1] > p[1] else -1
        upward_dir, downward_dir = _perp_dirs_for_dc(dc)
        steps = len(path) - 1
        segment_triggers: list[tuple[tuple[int, int, int], int]] = []

        for idx, (r, c) in enumerate(path):
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                ray_len = min(idx, steps - idx)
                if ray_len > 0:
                    segment_triggers.append(((r, c, v), ray_len))

        for order, (key, ray_len) in enumerate(segment_triggers):
            ray_dir = upward_dir if order % 2 == 0 else downward_dir
            dir_lengths = triggers.setdefault(key, {})
            dir_lengths[ray_dir] = max(dir_lengths.get(ray_dir, 0), ray_len)

    for group in groups_a.values():
        if len(group) >= 2:
            ordered = sorted(group)
            draw_segment(ordered[0], ordered[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            ordered = sorted(group)
            draw_segment(ordered[0], ordered[-1])

    for (r0, c0, tcol), dir_lengths in triggers.items():
        for (ddr, ddc), ray_len in dir_lengths.items():
            for step in range(1, ray_len + 1):
                r = r0 + step * ddr
                c = c0 + step * ddc
                if not (0 <= r < h and 0 <= c < w):
                    break
                if int(out[r, c]) == bg:
                    out[r, c] = tcol

    return out


def _trace_fixed_ray(
    start: tuple[int, int], direction: tuple[int, int], ray_len: int, H: int, W: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    r0, c0 = start
    ddr, ddc = direction
    cells = []
    for step in range(1, ray_len + 1):
        r = r0 + step * ddr
        c = c0 + step * ddc
        if not (0 <= r < H and 0 <= c < W):
            return None
        cells.append((r, c))

    gap = (r0 + (ray_len + 1) * ddr, c0 + (ray_len + 1) * ddc)
    if not (0 <= gap[0] < H and 0 <= gap[1] < W):
        return None
    return cells, gap


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for spec in seg["trigger_specs"]:
            tr, tc = spec["pos"]
            ddr, ddc = spec["direction"]
            for step in range(1, spec["ray_len"] + 1):
                r = tr + step * ddr
                c = tc + step * ddc
                if int(expected[r, c]) == bg:
                    expected[r, c] = _TRIGGER_COLOR
    return expected


def _build_boundary_single_baseline(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for spec in seg["trigger_specs"]:
            tr, tc = spec["pos"]
            ddr, ddc = spec["direction"]
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                if int(expected[r, c]) == bg:
                    expected[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc
    return expected


def _build_fixed_dual_baseline(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    expected = inp.copy()
    for seg in segments:
        upward_dir, downward_dir = _perp_dirs_for_dc(seg["dc"])
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for spec in seg["trigger_specs"]:
            tr, tc = spec["pos"]
            for ddr, ddc in (upward_dir, downward_dir):
                for step in range(1, spec["ray_len"] + 1):
                    r = tr + step * ddr
                    c = tc + step * ddc
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    used_backslash: set[int],
    used_slash: set[int],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
            diag_id = r0 - c0
            if diag_id in used_backslash:
                continue
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1
            diag_id = r0 + c0
            if diag_id in used_slash:
                continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in reserved for rc in path):
            continue
        return path, dc
    return None


def _candidate_triggers_for_segment(
    path: list[tuple[int, int]],
    dc: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
):
    upward_dir, downward_dir = _perp_dirs_for_dc(dc)
    blocked = occupied | reserved
    steps = len(path) - 1
    candidates = []
    for idx in range(1, steps):
        ray_len = min(idx, steps - idx)
        if ray_len < _MIN_VISIBLE_RAY_LEN:
            continue
        pos = path[idx]
        traces = {}
        valid = True
        for direction in (upward_dir, downward_dir):
            traced = _trace_fixed_ray(pos, direction, ray_len, H, W)
            if traced is None:
                valid = False
                break
            cells, gap = traced
            footprint = set(cells) | {gap}
            if footprint & blocked:
                valid = False
                break
            traces[direction] = {"cells": set(cells), "gap": gap}
        if valid:
            candidates.append({"idx": idx, "pos": pos, "ray_len": ray_len, "traces": traces})
    return candidates


def _choose_trigger_specs(candidates, dc: int):
    upward_dir, downward_dir = _perp_dirs_for_dc(dc)
    max_triggers = min(_TRIGGERS_PER_SEG[1], len(candidates))
    min_triggers = min(_TRIGGERS_PER_SEG[0], len(candidates))
    if max_triggers < _TRIGGERS_PER_SEG[0]:
        return None

    for _ in range(_TRIGGER_SUBSET_RETRIES):
        n_triggers = random.randint(min_triggers, max_triggers)
        chosen = sorted(random.sample(candidates, n_triggers), key=lambda item: item["idx"])

        assigned_cells: set[tuple[int, int]] = set()
        assigned_buffers: set[tuple[int, int]] = set()
        visibility_reserved: set[tuple[int, int]] = set()
        trigger_specs = []
        ok = True

        for order, cand in enumerate(chosen):
            direction = upward_dir if order % 2 == 0 else downward_dir
            other_direction = downward_dir if order % 2 == 0 else upward_dir
            actual = cand["traces"][direction]
            hidden = cand["traces"][other_direction]
            actual_footprint = actual["cells"] | {actual["gap"]}
            hidden_footprint = hidden["cells"] | {hidden["gap"]}
            if actual_footprint & (assigned_cells | assigned_buffers | visibility_reserved):
                ok = False
                break
            if hidden_footprint & (assigned_cells | assigned_buffers | visibility_reserved):
                ok = False
                break

            assigned_cells.update(actual["cells"])
            assigned_buffers.add(actual["gap"])
            visibility_reserved.update(hidden_footprint)
            trigger_specs.append(
                {
                    "pos": cand["pos"],
                    "ray_len": cand["ray_len"],
                    "direction": direction,
                }
            )

        if ok:
            return trigger_specs, assigned_cells, assigned_buffers, visibility_reserved

    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    segments = []
    used_backslash: set[int] = set()
    used_slash: set[int] = set()

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(
            slope_type, seg_len, H, W, occupied, reserved, used_backslash, used_slash
        )
        if result is None:
            continue
        path, dc = result

        candidates = _candidate_triggers_for_segment(path, dc, H, W, occupied, reserved)
        chosen = _choose_trigger_specs(candidates, dc)
        if chosen is None:
            continue

        trigger_specs, ray_cells, buffer_cells, visibility_reserved = chosen
        ep0, ep1 = path[0], path[-1]
        trigger_positions = [spec["pos"] for spec in trigger_specs]

        segments.append(
            {
                "ep0": ep0,
                "ep1": ep1,
                "path": path,
                "dc": dc,
                "trigger_specs": trigger_specs,
            }
        )
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        reserved.update(set(path[1:-1]) - set(trigger_positions))
        reserved.update(ray_cells)
        reserved.update(buffer_cells)
        reserved.update(visibility_reserved)
        if slope_type == "backslash":
            used_backslash.add(path[0][0] - path[0][1])
        else:
            used_slash.add(path[0][0] + path[0][1])

    total_triggers = sum(len(seg["trigger_specs"]) for seg in segments)
    if not segments or total_triggers < 2:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        inp[seg["ep0"]] = _LINE_COLOR
        inp[seg["ep1"]] = _LINE_COLOR
        for spec in seg["trigger_specs"]:
            inp[spec["pos"]] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    all_trigger_positions = [spec["pos"] for seg in segments for spec in seg["trigger_specs"]]
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if any(abs(r - tr) == abs(c - tc) and r != tr for tr, tc in all_trigger_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            all_trigger_positions.append((r, c))
            break

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(expected, _build_boundary_single_baseline(inp, bg, segments)):
        return None
    if np.array_equal(expected, _build_fixed_dual_baseline(inp, bg, segments)):
        return None

    try:
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
