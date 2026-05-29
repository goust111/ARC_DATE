# Variant of 195c6913
# Dimension: D3 — 轨迹涂色方式：颜色循环 (trajectory)
# Change: 粒子沿路径前进时用模板定义的颜色循环（2-4色）依次涂色，每步推进一个循环位；涂色连续不间断。... → 碰撞时变色（反射变色）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the D3 variant
# ==================================
# a) 具体化与行为追踪
# - 输入外形保持原题：左上角是一排 2x2 单色模板块，col 0 上有 1-2 个单像素 marker，
#   路径仍是单个 4 连通的粗对角带，右下背景区仍有一个 2x2 终止块。
# - 原题里，marker 一进入 path，每走一步就把颜色索引 idx 向前推进 1，所以同一段直线内部会
#   出现 A,B,C,A,B... 这种逐格循环。碰撞只影响方向，不影响段内配色。
# - 本变体里，模板仍定义“碰撞后的颜色顺序”，但段内不再逐格变色。marker 的起始段全部用
#   marker 自身颜色；当它撞到非 path 单元、且按照严格交替的 CCW/CW 规则成功转向时，
#   下一段才改色。若撞到的是普通背景，则切到模板循环中的下一色；若撞到的是有色障碍，
#   则优先继承障碍颜色。对本任务的几何布局来说，实际碰撞几乎总是撞到背景，因此可观察到的
#   规律是“每个直线段纯色，转弯后整体换成下一模板色”。
# - 具体小例子：模板为 [2,8,4]，marker=2。原题若第一段长 5，会画成 8,4,2,8,4；
#   本变体则五格全是 2。第一次碰撞并成功转向后，下一整段都变成 8；第二次再变成 4。
#   差异集中在“颜色变化与碰撞绑定，而不是与步数绑定”。
#
# b) 复杂度校准
# - 若允许模板色重复，或者大量样本根本没有碰撞，那么新旧输出可能视觉接近，D3 变化不明显。
#   因此生成器改为无放回抽样模板色，并拒绝 total_turns == 0 的样本。
# - 没有引入额外对象类型，也没有改动 turn rule / path 拓扑，因此复杂度仍与原题同级。
#
# c) 隐式假设审查
# - 原 solver 的 `idx = (idx + 1) % len(pattern)` 写在“每步进入 path”分支里，依赖旧题
#   的“逐步换色”前提；变体下这会错误地把一整段拆成棋盘式渐变，必须移到“碰撞且成功转向”
#   分支。
# - 原 `visited` 状态包含 `idx`，因为颜色取决于走了多少步。变体后颜色取决于“当前段颜色 +
#   已发生多少次碰撞”，所以状态必须加入 `cur_color`，否则遇到少见的有色障碍继承时会把
#   不同颜色状态错误合并。
# - 原代码在碰撞前无条件把背景碰撞点染成 `end_color`，这与 D3 无冲突，仍可保留。
# - 原 generator 用 `random.choice(cycle_pool)` 允许模板色重复；在旧题中重复只是降低循环信息，
#   仍“恰好无害”。在新题中会直接削弱“碰撞后换色”的可见性，因此改成 `random.sample(...)`。
#
# d) 放置合法性
# - 仍沿用原 band/fjord/island 几何构造与所有单连通约束，保证 marker、模板块、终止块互不重叠。
# - 额外合法性由 `_trajectory_stats` 检查：至少一次成功转向、至少一个长度 >= 2 的直线段，
#   并继续依赖 solver 的 disjoint trajectory 断言。
#
# e) 视觉可观察性
# - 人眼能直接看到：同一路径被切成若干“纯色长段”，颜色只在拐角后的下一段改变。
# - 模板仍保留在输入中，提供“下一段颜色来自模板顺序”的可见线索；没有使用颜色 id 排序这类
#   不可见属性。
#
# f) 可视化整洁度
# - 变体减少了原题那种逐格跳色的噪声，输出反而更整洁：每段直线是整块纯色，碰撞点仍用单像素
#   终止色标出，模板块和终止块照旧擦除，不会增加叠加层次。
#
# g) 颜色变化可推断性
# - 颜色种类仍限制在 2-4 个模板色 + 1 个终止色；颜色切换条件只有一个：成功碰撞并转向。
# - 因为 generator 强制出现碰撞，所以从 2-3 组样例中可以观察到“长段纯色 + 拐弯后整体换色”
#   的稳定模式。


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: markers in col 0 replace path cells, not free bg cells."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
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


def _trace_marker(
    g: np.ndarray,
    path_color: int,
    pattern: list[int],
    template_bg: int,
    end_color: int | None,
    marker: tuple[int, int, int],
    out: np.ndarray | None = None,
) -> dict[str, object]:
    H, W = g.shape
    sr, sc, sv = marker
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    init = None
    for dr, dc in dir_order:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            init = (int(dr), int(dc))
            break

    if init is None or not pattern:
        return {
            "cells": set(),
            "turns": 0,
            "path_steps": 0,
            "segment_colors": [int(sv)],
            "segment_lengths": [0],
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    cur_color = int(sv) if int(sv) in pattern else int(pattern[idx])
    hits = 0
    visited: set[tuple[int, int, int, int, int, int, int]] = set()
    my_cells: set[tuple[int, int]] = set()
    segment_colors = [int(cur_color)]
    segment_lengths = [0]
    path_steps = 0

    for _ in range(H * W * 4):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1), int(cur_color))
        if state in visited:
            break
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if int(g[nr, nc]) == int(path_color):
            if out is not None:
                out[nr, nc] = int(cur_color)
            my_cells.add((int(nr), int(nc)))
            segment_lengths[-1] += 1
            path_steps += 1
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == int(template_bg) and end_color is not None and out is not None:
            out[nr, nc] = int(end_color)

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (int(hits) % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            hit_color = int(g[nr, nc])
            if hit_color == int(template_bg) or hit_color == int(path_color):
                idx = int((idx + 1) % len(pattern))
                cur_color = int(pattern[idx])
            else:
                cur_color = int(hit_color)
                if hit_color in pattern:
                    idx = int(pattern.index(hit_color))
            dr, dc = int(tdr), int(tdc)
            hits += 1
            segment_colors.append(int(cur_color))
            segment_lengths.append(0)
            continue
        break

    return {
        "cells": my_cells,
        "turns": int(hits),
        "path_steps": int(path_steps),
        "segment_colors": segment_colors,
        "segment_lengths": segment_lengths,
    }


def _trajectory_stats(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    template_bg, path_color, pattern, _pat_cells, _end_color, _end_pos, markers = _extract_scene(g)
    traces = [
        _trace_marker(g, path_color, pattern, template_bg, _end_color, marker, out=None)
        for marker in markers
    ]
    all_segment_lengths = [int(x) for t in traces for x in t["segment_lengths"]]  # type: ignore[index]
    all_segment_colors = [int(x) for t in traces for x in t["segment_colors"]]  # type: ignore[index]
    return {
        "marker_count": len(markers),
        "turns_per_marker": [int(t["turns"]) for t in traces],
        "segment_lengths_per_marker": [list(map(int, t["segment_lengths"])) for t in traces],
        "segment_colors_per_marker": [list(map(int, t["segment_colors"])) for t in traces],
        "total_turns": int(sum(int(t["turns"]) for t in traces)),
        "max_segment_len": int(max(all_segment_lengths, default=0)),
        "distinct_segment_colors": int(len(set(all_segment_colors))),
        "pattern_len": int(len(pattern)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles along a path, but recolor only when a collision causes a turn.

    Input structure:
    - Top-left: a row of 2x2 monochrome template blocks still defines an ordered color sequence.
    - Isolated template-colored single pixels on the left edge are start markers embedded in the path.
    - A separate 2x2 block in the opposite bg region supplies the terminal collision color.

    Rule:
    - Each marker launches a particle along the path using its current segment color for every cell in that straight run.
    - When the next step is non-path, mark that impact cell with the end color, then try the scheduled 90° turn.
    - Turns still alternate strictly CCW, CW, CCW, CW; if the scheduled turn is not path, the particle stops.
    - After a successful bounce, the next straight segment changes color: usually to the next template color, or to the obstacle's own color if the obstacle is colored.

    Key insight: Geometry is unchanged from the original task; the only conceptual shift is that color is carried by whole straight segments and updates on bounce events instead of on every step.
    """
    g = np.array(grid, dtype=int)
    (
        template_bg,
        path_color,
        pattern,
        pat_cells,
        end_color,
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
        trace = _trace_marker(g, path_color, pattern, template_bg, end_color, marker, out=out)
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
            w = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - w / 2))
            c_right[r] = c_left[r] + w - 1

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

    try:
        _ = main(g)
        stats = _trajectory_stats(g)
    except Exception:
        return None

    if int(stats["total_turns"]) < 1:
        return None
    if int(stats["distinct_segment_colors"]) < 2:
        return None
    if int(stats["max_segment_len"]) < 2:
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
