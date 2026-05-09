# Combo variant of 195c6913 (index 20)
# combo_spec: D1/0+D3/1+D6/0
# Name: 对角线运动+碰撞变色+碰撞无角标记

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
# - 输入骨架仍保留原题三类对象：左上模板 2x2 色块、col 0 上的单像素 marker、右下背景区
#   的 2x2 终止块，以及中间那条单个 4 连通的粗对角 path 带。变化发生在“粒子如何在
#   path 内运动”和“什么时候换色”。
# - 原题是正交运动：marker 从 col 0 沿 path 直走，下一步撞到非 path 后按 CCW/CW 严格
#   交替转 90 度，且碰撞点会被 end_color 标出来；颜色则是每走一步就沿模板循环推进。
# - 融合后改成对角弹球：marker 从左边缘沿 45 度方向进入 path 带，每次尝试走到下一个
#   对角格；若对角方向被墙挡住，则依据被挡的是“竖向边界 / 横向边界 / 两者同时”做镜面
#   反射，继续从当前 path 格弹开。每个“碰撞到下一次碰撞”之间是一整段纯色斜线，成功反射
#   后才切到下一模板色；碰撞位置本身不再留下额外角标。
# - 具体例子：若模板为 [2,8,4]，marker=2。原题中第一段长度 5 时会画成 8,4,2,8,4；
#   本变体里第一条对角线整段都保持 2。碰到上边界后若发生单轴反射，则下一条斜线整段改成 8；
#   再次碰撞后改成 4。视觉上从“折线+角标+逐格跳色”变成“多段纯色斜条纹”。
#
# b) 复杂度校准
# - 若允许无碰撞样本，或者模板色重复，输出很容易退化成单条纯色斜线，人类无法看出“碰撞变色”。
#   因此生成器强制模板色互异，并筛掉总反射次数过少、段长过短、颜色段数不足的样本。
# - 若所有反射都只是双轴反转，视觉上会像简单折返，难以体现“镜面反射”而非“撞角回头”。
#   因此额外要求至少出现一次单轴反射，让样例里能看到真正的弹球式折返。
#
# c) 隐式假设审查
# - 原 solver 假设运动方向只来自 R/D/L/U，并且碰撞时总能通过 CCW/CW 转向逻辑选择新方向。
#   融合后这些前提全部失效，必须把状态改成四个对角方向之一，并把碰撞判定改成检查
#   `diag / vertical / horizontal` 三个相邻格的占用关系。
# - 原 `idx = (idx + 1) % len(pattern)` 写在“每前进一步”分支里；融合后颜色只应在成功反射后
#   切换，不能在段内逐格推进。
# - 原代码把背景碰撞格染成 end_color；D6/0 明确禁止碰撞角标，因此新 solver 不能再写任何
#   非 path 碰撞格，end_block 只保留为输入对象并在输出中擦除。
# - 新 solver 隐含了“marker 至少有一个可进入的对角 path 邻居”的假设；生成器必须显式验证。
#
# d) 放置合法性
# - 仍沿用原 band/fjord/island 构造，保持 path 单个 4 连通、模板区在左上背景区、end_block
#   在右下背景区且与 path 隔开、marker 覆盖 col 0 上原本的 path 单元。
# - 新增合法性检查：每个 marker 都要能以右上或右下对角进入 path；多 marker 时轨迹仍需
#   互不重叠；生成器继续用 solver 作为 oracle 过滤非法样本。
#
# e) 视觉可观察性
# - 解题者从少量样例里能直接看到三件事：轨迹是对角线而不是正交折线；颜色在碰撞之后整段切换，
#   而不是逐格循环；碰撞处没有单像素角标。
# - 颜色变化仍由输入顶部模板给出顺序线索，没有依赖颜色编号大小规律等人类不可见信息。
#
# f) 可视化整洁度
# - 输出中只保留 marker 本身和 path 内的纯色斜条纹；模板块与 end_block 被擦除，碰撞点也不
#   再单独着色，因此比原题更整洁，不会出现角点标记把局部挤乱的情况。
#
# g) 变化的可推断性
# - 模板色仍限制在 2-4 种，且通过“整段纯色 + 每次反射后换到下一色”形成强结构提示。
# - 生成器要求至少两种段色、至少一次单轴反射和足够长的段，使 2-3 组样例就能稳定暴露融合后的
#   核心规律，而不会退化成原题或某个过弱的单变体。


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: markers in col 0 replace path cells, not floating bg."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == int(path_color), (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == int(path_color), (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def _extract_scene(grid: np.ndarray) -> tuple[
    int,
    int,
    list[int],
    list[tuple[int, int]],
    int | None,
    tuple[int, int] | None,
    list[tuple[int, int, int]],
]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            [(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)]
        )[1]
    )
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                pattern.append(int(v))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))

    _assert_marker_embedded_in_path(g, markers, int(path_color))
    return template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers


def _initial_diag_direction(
    g: np.ndarray, path_color: int, marker: tuple[int, int, int]
) -> tuple[int, int] | None:
    """Choose the first reachable diagonal direction, preferring motion into the grid."""
    H, W = g.shape
    sr, sc, _ = marker
    for dr, dc in [(-1, 1), (1, 1), (-1, -1), (1, -1)]:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            return int(dr), int(dc)
    return None


def _next_segment_color(
    pattern: list[int],
    idx: int,
    cur_color: int,
    template_bg: int,
    path_color: int,
    wall_colors: list[int],
) -> tuple[int, int]:
    """Change color on bounce: inherit a colored obstacle if present, else advance cycle."""
    for wcol in wall_colors:
        if int(wcol) not in (int(template_bg), int(path_color)):
            if int(wcol) in pattern:
                return int(pattern.index(int(wcol))), int(wcol)
            return int(idx), int(wcol)
    if not pattern:
        return int(idx), int(cur_color)
    idx = int((idx + 1) % len(pattern))
    return int(idx), int(pattern[idx])


def _trace_marker(
    g: np.ndarray,
    path_color: int,
    pattern: list[int],
    template_bg: int,
    marker: tuple[int, int, int],
    out: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Trace one marker inside the path band using 45-degree motion and digital reflections.
    """
    H, W = g.shape
    sr, sc, sv = marker
    init = _initial_diag_direction(g, path_color, marker)

    if init is None or not pattern:
        return {
            "cells": set(),
            "path_steps": 0,
            "segment_colors": [int(sv)],
            "segment_lengths": [0],
            "side_bounces": 0,
            "corner_bounces": 0,
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    cur_color = int(sv) if int(sv) in pattern else int(pattern[idx])
    visited: set[tuple[int, int, int, int, int, int]] = set()
    my_cells: set[tuple[int, int]] = set()
    segment_colors = [int(cur_color)]
    segment_lengths = [0]
    path_steps = 0
    side_bounces = 0
    corner_bounces = 0

    for _ in range(H * W * 8):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(cur_color))
        if state in visited:
            break
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        diag_open = 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(path_color)

        vr = int(r + dr)
        vc = int(c)
        hr = int(r)
        hc = int(c + dc)
        vopen = 0 <= vr < H and 0 <= vc < W and int(g[vr, vc]) == int(path_color)
        hopen = 0 <= hr < H and 0 <= hc < W and int(g[hr, hc]) == int(path_color)

        if diag_open:
            if out is not None:
                out[nr, nc] = int(cur_color)
            my_cells.add((int(nr), int(nc)))
            segment_lengths[-1] += 1
            path_steps += 1
            r, c = int(nr), int(nc)
            continue

        flip_r = not vopen
        flip_c = not hopen
        if not flip_r and not flip_c:
            # Diagonal opening is blocked by a convex corner, so bounce on both axes.
            flip_r = True
            flip_c = True

        wall_colors: list[int] = []
        if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) != int(path_color):
            wall_colors.append(int(g[nr, nc]))
        if flip_r and 0 <= vr < H and 0 <= vc < W and int(g[vr, vc]) != int(path_color):
            wall_colors.append(int(g[vr, vc]))
        if flip_c and 0 <= hr < H and 0 <= hc < W and int(g[hr, hc]) != int(path_color):
            wall_colors.append(int(g[hr, hc]))

        ndr = -int(dr) if flip_r else int(dr)
        ndc = -int(dc) if flip_c else int(dc)
        nnr = int(r + ndr)
        nnc = int(c + ndc)
        if not (0 <= nnr < H and 0 <= nnc < W and int(g[nnr, nnc]) == int(path_color)):
            break

        idx, cur_color = _next_segment_color(
            pattern, idx, cur_color, template_bg, path_color, wall_colors
        )
        dr, dc = int(ndr), int(ndc)
        if flip_r ^ flip_c:
            side_bounces += 1
        else:
            corner_bounces += 1
        segment_colors.append(int(cur_color))
        segment_lengths.append(0)

    return {
        "cells": my_cells,
        "path_steps": int(path_steps),
        "segment_colors": segment_colors,
        "segment_lengths": segment_lengths,
        "side_bounces": int(side_bounces),
        "corner_bounces": int(corner_bounces),
    }


def _trajectory_stats(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    template_bg, path_color, pattern, _pat_cells, _end_color, _end_pos, markers = _extract_scene(g)
    traces = [
        _trace_marker(g, path_color, pattern, template_bg, marker, out=None)
        for marker in markers
    ]
    all_segment_lengths = [int(x) for t in traces for x in t["segment_lengths"]]  # type: ignore[index]
    all_segment_colors = [int(x) for t in traces for x in t["segment_colors"]]  # type: ignore[index]
    return {
        "marker_count": len(markers),
        "segment_lengths_per_marker": [list(map(int, t["segment_lengths"])) for t in traces],
        "segment_colors_per_marker": [list(map(int, t["segment_colors"])) for t in traces],
        "side_bounces_per_marker": [int(t["side_bounces"]) for t in traces],
        "corner_bounces_per_marker": [int(t["corner_bounces"]) for t in traces],
        "total_bounces": int(
            sum(int(t["side_bounces"]) + int(t["corner_bounces"]) for t in traces)
        ),
        "side_bounces": int(sum(int(t["side_bounces"]) for t in traces)),
        "corner_bounces": int(sum(int(t["corner_bounces"]) for t in traces)),
        "max_segment_len": int(max(all_segment_lengths, default=0)),
        "distinct_segment_colors": int(len(set(all_segment_colors))),
        "total_path_steps": int(sum(int(t["path_steps"]) for t in traces)),
        "pattern_len": int(len(pattern)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate diagonal particles inside the path band, switching color only after reflections.

    Input structure:
    - Top-left: a row of 2x2 monochrome template blocks gives the ordered segment colors.
    - Isolated template-colored pixels on col 0 are start markers embedded in the path band.
    - A separate 2x2 block in the opposite background region is still present as an auxiliary object and is erased in the output.

    Rule:
    - Each marker enters the path along a 45-degree diagonal and keeps moving diagonally through path cells.
    - When the next diagonal step would leave the path, the particle reflects off the boundary: one blocked axis flips that axis, two blocked axes flip both.
    - Every straight segment between two collisions is monochrome; after each successful reflection the color changes to the next template color, or to a colored obstacle if one is hit.
    - Collision sites themselves are never painted, and the template blocks plus the 2x2 terminal block are removed from the final grid.

    Key insight: The informative signal is no longer the old turn marker but the segmentation of the diagonal trace itself: bounces carve the path into pure-color slanted runs, so the solver must reason about reflective geometry and event-driven recoloring together.
    """
    g = np.array(grid, dtype=int)
    (
        template_bg,
        path_color,
        pattern,
        pat_cells,
        _end_color,
        end_pos,
        markers,
    ) = _extract_scene(g)

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for marker in markers:
        trace = _trace_marker(g, path_color, pattern, template_bg, marker, out=out)
        per_marker_cells.append(trace["cells"])  # type: ignore[arg-type]

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick meandering anti-diagonal path band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6)) for r in range(H)]
    )

    c_left = np.zeros(H, dtype=int)
    c_right = np.zeros(H, dtype=int)

    cc_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )
    w_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )

    for r in range(H):
        c_left_param = centers[r] - half_w[r]
        if c_left_param <= 0:
            c_left[r] = 0
            c_right[r] = int(round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2))
        else:
            width = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - width / 2))
            c_right[r] = c_left[r] + width - 1

    if lobe_rows:
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    for r in range(1, min(4, H)):
        c_left[r] = max(c_left[r], 12)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
    c_left[H - 1] = 0
    c_right[H - 1] = max(5, min(W - 1, c_right[H - 1]))

    for _ in range(2):
        for r in range(1, H - 1):
            if c_left[r] > c_left[r - 1] and c_left[r] > c_left[r + 1]:
                c_left[r] = max(c_left[r - 1], c_left[r + 1])

    for r in range(1, H):
        c_left[r] = min(c_left[r], c_left[r - 1] + 2)
        c_right[r] = min(c_right[r], c_right[r - 1] + 2)
        if c_left[r] > c_right[r - 1]:
            c_left[r] = c_right[r - 1]
        if c_right[r] < c_left[r - 1]:
            c_right[r] = c_left[r - 1]
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_left = np.clip(c_left, 0, W - 1)
    c_right = np.clip(c_right, 0, W - 1)
    c_right[0] = W - 1
    c_left[H - 1] = 0

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r] : c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a background separator from the left edge without disconnecting path."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    sr = random.randint(sr_min, sr_max)

    cur_top = sr
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        made_carve = True
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = random.sample(cycle_pool, n_blocks)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, r1, r2):
            return None
        if random.random() < 0.25:
            k_restore = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(k_restore, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
            return None
        for r in markers:
            if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
                return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= H or cc >= W or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r in markers:
        g[r, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    for r in markers:
        marker = (int(r), 0, int(cycle_colors[0]))
        if _initial_diag_direction(g, path_color, marker) is None:
            return None

    try:
        _ = main(g)
        stats = _trajectory_stats(g)
    except Exception:
        return None

    if int(stats["total_bounces"]) < 2:
        return None
    if int(stats["side_bounces"]) < 1:
        return None
    if int(stats["distinct_segment_colors"]) < 2:
        return None
    if int(stats["max_segment_len"]) < 3:
        return None
    if int(stats["total_path_steps"]) < 12:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
