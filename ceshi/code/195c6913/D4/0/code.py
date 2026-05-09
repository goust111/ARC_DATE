# Variant of 195c6913
# Dimension: D4 — 源对象：标记像素定义 (src)
# Change: 1-2个孤立的单像素标记，颜色=循环色[0]，位于col 0（左网格边缘），嵌入在path区段中；初始方向优先级为R,D... → 标记位于网格四条边缘任意一侧

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis
# ----------------
# concrete_behavior:
#   original_task:
#     - 标记总在左边界，solver 通过 R,D,L,U 的邻居优先级间接得到“向右出发”。
#     - generator 也专门在 col 0 为标记做 lobe 和分隔带。
#   variant:
#     - 标记仍然是 1-2 个孤立单像素，颜色仍等于循环色[0]，但它们可以出现在 top/right/bottom/left 任一边界。
#     - 起始方向不再依赖邻居扫描顺序，而是由边界法线唯一决定：
#       left→R, right→L, top→D, bottom→U。
#     - 为了让“从该边向内”在像素上可见，每个标记都放在非角点边界位置，并要求它覆盖的原 path 边界段在该边上有连续切向邻居，且 inward 邻居也是 path。
#   trace_example:
#     - 原逻辑里 top-edge 标记若右侧也是 path，R,D,L,U 会错误地先往右走。
#     - 新逻辑里同一个 top-edge 标记必须先往下，再按碰撞奇偶做 CCW/CW 交替转向。
#     - 因此 D4 的核心差异不是“标记能否被检测到”，而是“solver 必须先识别边，再决定第一步方向”。
# complexity_calibration:
#   - 不引入新的对象种类；模板、终止块、路径带、交替转向规则都保留。
#   - 两标记情况只允许从少量边型组合中挑选（同边或对边），避免分支过多到人类难以归纳。
#   - generator 仍生成 1 或 2 个标记，并要求轨迹长度/转向数达到原题量级，保证变化不是微弱噪声。
# implicit_assumptions_review:
#   - 原 _assert_marker_embedded_in_path 只检查 col 0 上下邻居；变体下必须改成“依据边类型检查切向邻居 + inward 邻居”。
#   - 原 main 里的 dir_order = R,D,L,U 默认左边界起步；变体下必须废弃这一推断，改成边界法线 helper。
#   - 原 generator 最终 `g[r,0] = cycle_colors[0]` 直接写死左边；变体下必须先枚举四边候选，再选择能产生合法轨迹的位置。
# placement_legality:
#   - 标记不能在角点，否则边类型不唯一；也避开左上模板区附近，防止模板和边界标记视觉重叠。
#   - 标记写入后仍要求 path_color 保持单个 4 连通分量。
#   - 两标记必须切向上不相邻，轨迹集合必须不重叠，并通过 solver oracle 再验证。
# visual_observability:
#   - 输入里最显著的变化就是“标记不再固定在左边”；输出里相应轨迹会从该边向内推进。
#   - 这在 2-4 组样例中可直接肉眼对比：同样的模板/路径机制，唯一起步边不同。
# neatness_and_color_inference:
#   - 颜色机制完全沿用原题：2x2 模板给循环色，终止色仍只来自单独 2x2 终止块。
#   - 不使用颜色 id 排序或隐藏分支；边缘位置是唯一决定起步方向的可见线索。
# generator_guarantees_for_new_main:
#   - 每个候选标记都先在含模板和终止块的最终输入背景上做一次轨迹模拟。
#   - 只有满足边界嵌入、步数足够、轨迹不重叠、写入后 path 仍单连通的候选才会被采纳。


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _edge_info(shape: tuple[int, int], r: int, c: int) -> tuple[str, tuple[int, int], list[tuple[int, int]]]:
    """Return (side, inward_dir, tangent_dirs) for a non-corner border cell."""
    H, W = map(int, shape)
    sides: list[tuple[str, tuple[int, int], list[tuple[int, int]]]] = []
    if r == 0:
        sides.append(("top", (1, 0), [(0, -1), (0, 1)]))
    if r == H - 1:
        sides.append(("bottom", (-1, 0), [(0, -1), (0, 1)]))
    if c == 0:
        sides.append(("left", (0, 1), [(-1, 0), (1, 0)]))
    if c == W - 1:
        sides.append(("right", (0, -1), [(-1, 0), (1, 0)]))
    assert len(sides) == 1, f"marker at ({r},{c}) must be on exactly one edge"
    return sides[0]


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker replaces a border path cell with a valid inward start."""
    H, W = g.shape
    for r, c, _ in markers:
        side, (dr, dc), tangents = _edge_info((H, W), int(r), int(c))
        rr = int(r + dr)
        cc = int(c + dc)
        assert 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color), (
            f"marker at ({r},{c}) on {side} must have inward path neighbor"
        )
        for tr, tc in tangents:
            nr = int(r + tr)
            nc = int(c + tc)
            assert 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(path_color), (
                f"marker at ({r},{c}) on {side} must have tangent path neighbor at ({nr},{nc})"
            )


def _detect_pattern(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Read the top-left row of 2x2 monochrome cycle blocks."""
    H, W = g.shape
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
    return pattern, pat_cells


def _detect_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
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
                return int(v), (int(r), int(c))
    return None, None


def _detect_markers(g: np.ndarray, pat_set: set[int], path_color: int) -> list[tuple[int, int, int]]:
    H, W = g.shape
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
    return markers


def _trace_particle(
    g: np.ndarray,
    out: np.ndarray | None,
    pattern: list[int],
    path_color: int,
    template_bg: int,
    end_color: int | None,
    start: tuple[int, int],
    init_dir: tuple[int, int],
    start_color: int,
) -> tuple[set[tuple[int, int]], int]:
    """Simulate one marker. Returns (visited path cells, successful turn count)."""
    H, W = g.shape
    if not pattern:
        return set(), 0

    r, c = map(int, start)
    dr, dc = map(int, init_dir)
    idx = int(pattern.index(int(start_color))) if int(start_color) in pattern else 0
    hits = 0
    visited_states: set[tuple[int, int, int, int, int, int]] = set()
    my_cells: set[tuple[int, int]] = set()

    for _ in range(H * W * 4):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
        if state in visited_states:
            break
        visited_states.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            if out is not None:
                out[nr, nc] = int(pattern[idx])
            my_cells.add((int(nr), int(nc)))
            r, c = nr, nc
            continue

        if out is not None and int(g[nr, nc]) == int(template_bg) and end_color is not None:
            out[nr, nc] = int(end_color)

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (int(hits) % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            dr, dc = int(tdr), int(tdc)
            hits += 1
            continue
        break

    return my_cells, int(hits)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate border-launched particles that recolor a path using an alternating turn rule.

    Input structure:
    - The upper-left area contains a row of 2x2 monochrome blocks that encode the color cycle.
    - One or two isolated single pixels of the first cycle color sit on one of the four grid edges.
    - A separate 2x2 monochrome block of another color marks the terminal collision color.

    Rule:
    - Each edge marker launches inward from its own edge normal, not from a global neighbor-priority scan.
    - While moving through path cells, the particle advances one step at a time and paints the next cycle color.
    - When the next cell is not path, that collision cell gets the terminal color if it is background, then the particle must turn 90 degrees.
    - Turn directions alternate strictly CCW, CW, CCW, CW...; if the scheduled turn does not point to path, that particle stops.

    Key insight:
    - The new deduction step is to identify which border the marker sits on, because that border alone fixes the first movement direction before the old alternating-turn mechanism takes over.
    """
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

    pattern, pat_cells = _detect_pattern(g, int(template_bg), int(path_color))
    pat_set = set(map(int, pattern))
    end_color, end_pos = _detect_end_block(g, int(template_bg), int(path_color), pat_set)
    markers = _detect_markers(g, pat_set, int(path_color))
    _assert_marker_embedded_in_path(g, markers, int(path_color))

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for sr, sc, sv in markers:
        _side, init, _tangents = _edge_info((H, W), int(sr), int(sc))
        cells, _turns = _trace_particle(
            g,
            out,
            pattern,
            int(path_color),
            int(template_bg),
            end_color,
            (int(sr), int(sc)),
            init,
            int(sv),
        )
        per_marker_cells.append(cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal path band that touches all four borders."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [
            0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
            for r in range(H)
        ]
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
    """Carve a thin background fjord from the left without disconnecting the path."""
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


def _candidate_sites(
    g: np.ndarray,
    path_color: int,
    pattern: list[int],
    template_bg: int,
    end_color: int,
) -> list[dict]:
    H, W = g.shape
    candidates: list[dict] = []
    border_cells: list[tuple[int, int]] = []
    for c in range(1, W - 1):
        border_cells.append((0, c))
        border_cells.append((H - 1, c))
    for r in range(1, H - 1):
        border_cells.append((r, 0))
        border_cells.append((r, W - 1))

    for r, c in border_cells:
        if int(g[r, c]) != int(path_color):
            continue
        side, inward, tangents = _edge_info((H, W), int(r), int(c))
        if side == "top" and c < 13:
            continue
        if side == "left" and r < 4:
            continue
        if side == "right" and r < 4:
            continue
        if any(int(g[r + tr, c + tc]) != int(path_color) for tr, tc in tangents):
            continue
        rr = int(r + inward[0])
        cc = int(c + inward[1])
        if int(g[rr, cc]) != int(path_color):
            continue
        cells, turns = _trace_particle(
            g,
            None,
            pattern,
            int(path_color),
            int(template_bg),
            int(end_color),
            (int(r), int(c)),
            inward,
            int(pattern[0]),
        )
        min_steps = 8 if side in {"top", "bottom"} else 15
        if not (min_steps <= len(cells) <= 80):
            continue
        if turns > 4:
            continue
        candidates.append(
            {
                "pos": (int(r), int(c)),
                "side": side,
                "dir": inward,
                "cells": cells,
                "turns": int(turns),
            }
        )
    return candidates


def _choose_markers(candidates: list[dict], n_markers: int) -> list[dict] | None:
    by_side: dict[str, list[dict]] = {side: [] for side in ["left", "right", "top", "bottom"]}
    for cand in candidates:
        by_side[cand["side"]].append(cand)
    for vals in by_side.values():
        random.shuffle(vals)

    if n_markers == 1:
        side_order = ["top", "right", "bottom", "left"]
        random.shuffle(side_order)
        for side in side_order:
            if by_side[side]:
                return [by_side[side][0]]
        return None

    patterns = [
        ("top", "top"),
        ("right", "right"),
        ("bottom", "bottom"),
        ("left", "left"),
        ("top", "bottom"),
        ("left", "right"),
    ]
    random.shuffle(patterns)
    for s1, s2 in patterns:
        left_list = by_side[s1]
        right_list = by_side[s2]
        if s1 == s2:
            for i in range(len(left_list)):
                for j in range(i + 1, len(left_list)):
                    a = left_list[i]
                    b = left_list[j]
                    if abs(a["pos"][0] - b["pos"][0]) + abs(a["pos"][1] - b["pos"][1]) < 4:
                        continue
                    if a["cells"] & b["cells"]:
                        continue
                    if a["turns"] + b["turns"] < 3:
                        continue
                    return [a, b]
        else:
            for a in left_list:
                for b in right_list:
                    if a["cells"] & b["cells"]:
                        continue
                    if a["turns"] + b["turns"] < 3:
                        continue
                    return [a, b]

    shuffled = candidates[:]
    random.shuffle(shuffled)
    for i in range(len(shuffled)):
        for j in range(i + 1, len(shuffled)):
            a = shuffled[i]
            b = shuffled[j]
            if a["cells"] & b["cells"]:
                continue
            if a["turns"] + b["turns"] < 3:
                continue
            return [a, b]
    return None


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    if n_markers == 1:
        support_rows = [random.randint(H // 2, H - 4)]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        support_rows = [r1, r2]

    mask = _build_band(H, W, lobe_rows=support_rows)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, support_rows[0], support_rows[1]):
            return None
        if random.random() < 0.25:
            K = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(K, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
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

    end_candidates: list[tuple[int, int]] = []
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

    candidates = _candidate_sites(g, int(path_color), cycle_colors, int(bg), int(end_color))
    chosen = _choose_markers(candidates, int(n_markers))
    if chosen is None:
        return None

    for cand in chosen:
        r, c = cand["pos"]
        g[int(r), int(c)] = cycle_colors[0]

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
        markers = _detect_markers(g, {int(c) for c in cycle_colors}, int(path_color))
        sides = [_edge_info(g.shape, r, c)[0] for r, c, _ in markers]
        if len(markers) != n_markers:
            return None
        if not all(side in {"left", "right", "top", "bottom"} for side in sides):
            return None
        _ = main(g)
    except Exception:
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
