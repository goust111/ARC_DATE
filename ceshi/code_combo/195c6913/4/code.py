# Combo variant of 195c6913 (index 4)
# combo_spec: D4/0+D11/0
# Name: 四边缘标记+颜色编码方向

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis
# ----------------
# a) Concrete behavior / trace:
#    - 原题里，标记总在左边界，solver 用 R,D,L,U 邻居优先级选第一步；于是起步几乎等价于“从左向右”。
#    - D4 让标记可落在四边，但仍可由边缘法线推出起步方向；D11 则把第一步改为“由标记颜色编码”。
#    - 融合后这两件事必须同时成立：标记可以在任意边缘，但边缘不再决定第一步。比如顶边红标记不是向下，而是向右；左边蓝标记不是向右，而是向下。
#    - 一个具体追踪：若某个顶边标记右邻和下邻都是 path，原题/D4 会把它当作向下或依赖局部几何；本变体里若它是红色，就必须先沿顶边向右，再在后续碰撞处按 CCW/CW 交替转向。
# b) Complexity calibration:
#    - 使用三种“方向色”：蓝(1)->Down，红(2)->Right，绿(3)->Left。
#    - 每个样本放两个标记：一个蓝标记放在 left/right，另一个从“红底边”或“绿顶边”里二选一。
#    - 这样单个样本仍然简洁，但跨样本能覆盖四条边，且每个实际标记都与“从边缘朝内走”的 D4 单变体明显不同。
# c) Hidden assumptions review:
#    - 原 main 的 `dir_order = R,D,L,U` 假设“邻居结构决定起步方向”；这里必须改成颜色查表。
#    - D4 的 `_edge_info()` 仍有价值，但只用于确认“它确实是单一边缘标记”，不能再拿它的 inward 方向当真起步方向。
#    - 原 `_assert_marker_embedded_in_path()` 只验证 col 0 垂直邻居；这里必须改成：标记颜色可映射到合法方向，且该方向上的相邻格一定是 path。
#    - 新冲突点是：颜色和边缘都携带潜在线索。为避免 solver 暗中重新依赖边缘，生成器强制挑选 `mapped_dir != inward_dir` 的候选。
# d) Placement legality:
#    - 标记必须位于非角点边界格，不能和模板/终止块重叠。
#    - 标记覆盖的原格必须本来是 path；其颜色方向的下一格也必须是 path，且还要有至少一个额外 path 邻居，避免把标记挂在单格毛刺上。
#    - 两条轨迹必须不重叠；写入两个标记后 path 仍保持单个 4 连通分量。
# e) Visual observability:
#    - 观察者能直接看到：蓝总是向下、红总是向右、绿总是向左，但它们都只是“边缘上的同类单像素”。
#    - 由于蓝只放在 left/right、红只放在 bottom、绿只放在 top，样本里不会出现“边缘法线恰好等于颜色方向”的退化案例。
# f) Visual neatness:
#    - 仍保留原题的模板、终止块、厚路径带，避免新对象种类膨胀。
#    - 顶边标记避开左上模板区，侧边标记避开前几行，减少模板与边缘标记贴在一起造成的阅读噪声。
# g) Inference of the change:
#    - 颜色映射不靠颜色编号排序等不可视因素，只靠跨样本重复出现的“蓝向下、红向右、绿向左”。
#    - 生成器额外验证：每个候选若按 D4 的边缘 inward 方向去走，得到的轨迹必须和颜色方向轨迹不同；否则该候选视为退化，不允许采纳。
# new_main_assumptions:
#    - main() 假设所有真实标记颜色都在 `DIR_BY_MARKER_COLOR` 里；generator 只发出蓝(1)、红(2)、绿(3)三种方向标记。
#    - main() 假设每个标记正前方存在 path；generator 在候选筛选阶段与最终 oracle 阶段都保证这一点。
#    - main() 继续假设模板块和终止块都可用 2x2 单色块检测；generator 保持原题模板/终止块结构不变。

DIR_BY_MARKER_COLOR: dict[int, tuple[int, int]] = {
    1: (1, 0),  # blue -> Down
    2: (0, 1),  # red -> Right
    3: (0, -1),  # green -> Left
}


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _edge_info(
    shape: tuple[int, int], r: int, c: int
) -> tuple[str, tuple[int, int], list[tuple[int, int]]]:
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
    """Find the single 2x2 end block."""
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


def _detect_markers(
    g: np.ndarray, pat_set: set[int], path_color: int
) -> list[tuple[int, int, int]]:
    """Detect isolated cycle-colored border pixels adjacent to the path."""
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


def _support_neighbor_count(g: np.ndarray, r: int, c: int, path_color: int) -> int:
    count = 0
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(path_color):
            count += 1
    return count


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each border marker has a valid color-coded launch into the path."""
    H, W = g.shape
    for r, c, v in markers:
        _side, _inward, _tangents = _edge_info((H, W), int(r), int(c))
        launch = DIR_BY_MARKER_COLOR.get(int(v))
        assert launch is not None, f"marker color {int(v)} must have a direction mapping"
        rr = int(r + launch[0])
        cc = int(c + launch[1])
        assert 0 <= rr < H and 0 <= cc < W, (
            f"marker at ({r},{c}) with dir {launch} must stay in bounds on first step"
        )
        assert int(g[rr, cc]) == int(path_color), (
            f"marker at ({r},{c}) with dir {launch} must face path_color {path_color}"
        )
        assert _support_neighbor_count(g, int(r), int(c), int(path_color)) >= 2, (
            f"marker at ({r},{c}) must replace a path segment, not a single-cell spur"
        )


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
    Simulate border markers whose launch direction is encoded by marker color.

    Input structure:
    - The upper-left region contains a row of 2x2 monochrome blocks that defines the paint cycle.
    - One or two isolated cycle-colored pixels sit on the outer border; their edge position is arbitrary.
    - A separate 2x2 monochrome block supplies the terminal collision color.

    Rule:
    - Each marker starts in the cycle phase named by its own color, then launches according to the fixed color-to-direction map rather than the edge it sits on.
    - While the particle advances through path cells, it paints successive colors from the cycle.
    - When the next step is not path, that collision cell is painted with the terminal color if it was background, then the particle must turn 90 degrees.
    - Turn directions alternate strictly CCW, CW, CCW, CW...; if the scheduled turn does not lead to path, that particle stops.

    Key insight:
    - The edge now tells you only where the marker is located, not how it starts moving. The essential deduction is to infer the marker-color direction code and then run the original alternating-turn path simulation from that heading.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
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
        init = DIR_BY_MARKER_COLOR.get(int(sv))
        if init is None or not pattern:
            per_marker_cells.append(set())
            continue
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
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal path band that touches all four borders."""
    bot_center = random.randint(0, W // 4)
    right_touch_depth = random.randint(4, max(5, H // 3))
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
    for r in range(1, min(H, right_touch_depth)):
        c_right[r] = W - 1
        if c_left[r] > c_right[r]:
            c_left[r] = c_right[r]
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


def _candidate_sites(
    g: np.ndarray,
    path_color: int,
    pattern: list[int],
    template_bg: int,
    end_color: int,
    marker_color: int,
) -> list[dict[str, object]]:
    """Enumerate border path cells that visibly realize the color-direction rule."""
    H, W = g.shape
    dr, dc = DIR_BY_MARKER_COLOR[int(marker_color)]
    candidates: list[dict[str, object]] = []
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
        side, inward, _tangents = _edge_info((H, W), int(r), int(c))
        rr = int(r + dr)
        cc = int(c + dc)
        if not (0 <= rr < H and 0 <= cc < W):
            continue
        if int(g[rr, cc]) != int(path_color):
            continue
        if side == "top" and c < 13:
            continue
        if side in {"left", "right"} and r < 4:
            continue
        if _support_neighbor_count(g, int(r), int(c), int(path_color)) < 2:
            continue

        cells, turns = _trace_particle(
            g,
            None,
            pattern,
            int(path_color),
            int(template_bg),
            int(end_color),
            (int(r), int(c)),
            (int(dr), int(dc)),
            int(marker_color),
        )
        if not (5 <= len(cells) <= 90):
            continue
        if not (0 <= turns <= 5):
            continue

        normal_cells, _ = _trace_particle(
            g,
            None,
            pattern,
            int(path_color),
            int(template_bg),
            int(end_color),
            (int(r), int(c)),
            inward,
            int(marker_color),
        )
        is_mismatch = (int(dr), int(dc)) != inward and cells != normal_cells

        candidates.append(
            {
                "pos": (int(r), int(c)),
                "side": side,
                "dir": (int(dr), int(dc)),
                "cells": cells,
                "turns": int(turns),
                "marker_color": int(marker_color),
                "is_mismatch": bool(is_mismatch),
            }
        )
    return candidates


def _pick_marker_pair(
    blue_candidates: list[dict[str, object]],
    red_candidates: list[dict[str, object]],
    green_candidates: list[dict[str, object]],
    target_family: tuple[str, str, int] | None = None,
) -> list[dict[str, object]] | None:
    """Choose two mismatch markers with disjoint paths and visible edge coverage."""
    by_side_blue: dict[str, list[dict[str, object]]] = {"left": [], "right": []}
    by_side_red: dict[str, list[dict[str, object]]] = {"bottom": []}
    by_side_green: dict[str, list[dict[str, object]]] = {"top": []}
    for cand in blue_candidates:
        if cand["side"] in by_side_blue:
            by_side_blue[str(cand["side"])].append(cand)
    for cand in red_candidates:
        if cand["side"] in by_side_red:
            by_side_red[str(cand["side"])].append(cand)
    for cand in green_candidates:
        if cand["side"] in by_side_green:
            by_side_green[str(cand["side"])].append(cand)

    for vals in list(by_side_blue.values()) + list(by_side_red.values()) + list(by_side_green.values()):
        random.shuffle(vals)

    feasible_by_family: dict[tuple[str, str, int], list[list[dict[str, object]]]] = {}
    all_vertical = by_side_red["bottom"] + by_side_green["top"]
    for blue_side in ["left", "right"]:
        for blue in by_side_blue[blue_side]:
            for vertical in all_vertical:
                if blue["cells"] & vertical["cells"]:
                    continue
                if not (bool(blue["is_mismatch"]) and bool(vertical["is_mismatch"])):
                    continue
                if int(blue["turns"]) + int(vertical["turns"]) < 1:
                    continue
                family = (
                    str(blue["side"]),
                    str(vertical["side"]),
                    int(vertical["marker_color"]),
                )
                feasible_by_family.setdefault(family, []).append([blue, vertical])

    if not feasible_by_family:
        return None

    families = list(feasible_by_family)
    if target_family is not None:
        if target_family not in feasible_by_family:
            return None
        chosen_family = target_family
    else:
        random.shuffle(families)
        chosen_family = random.choice(families)
    return random.choice(feasible_by_family[chosen_family])


def _gen_attempt(target_family: tuple[str, str, int] | None = None) -> np.ndarray | None:
    H = W = random.randint(20, 30)

    reserved_cycle = [1, 2, 3]
    free_colors = [c for c in range(10) if c not in reserved_cycle]
    random.shuffle(free_colors)
    bg, path_color, end_color = free_colors[:3]
    extra_pool = free_colors[3:]

    n_blocks = random.randint(3, 4)
    extra_count = n_blocks - len(reserved_cycle)
    cycle_colors = reserved_cycle + random.sample(extra_pool, extra_count)
    random.shuffle(cycle_colors)

    support_rows = [random.randint(H // 4, H // 2)]
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
                rr = template_row + dr
                cc = bc + dc
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

    blue_candidates = _candidate_sites(
        g, int(path_color), cycle_colors, int(bg), int(end_color), 1
    )
    red_candidates = _candidate_sites(
        g, int(path_color), cycle_colors, int(bg), int(end_color), 2
    )
    green_candidates = _candidate_sites(
        g, int(path_color), cycle_colors, int(bg), int(end_color), 3
    )
    chosen = _pick_marker_pair(
        blue_candidates,
        red_candidates,
        green_candidates,
        target_family=target_family,
    )
    if chosen is None:
        return None

    for cand in chosen:
        r, c = cand["pos"]
        g[int(r), int(c)] = int(cand["marker_color"])

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(count, color) for color, count in cnt.items() if color != bg], reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    try:
        markers = _detect_markers(g, {int(c) for c in cycle_colors}, int(path_color))
        if len(markers) != 2:
            return None
        side_set = {(_edge_info(g.shape, r, c)[0], int(v)) for r, c, v in markers}
        if not any(side in {"left", "right"} and color == 1 for side, color in side_set):
            return None
        if not any(
            (side == "bottom" and color == 2) or (side == "top" and color == 3)
            for side, color in side_set
        ):
            return None
        mismatch_count = 0
        for r, c, v in markers:
            _side, inward, _tangents = _edge_info(g.shape, int(r), int(c))
            if DIR_BY_MARKER_COLOR[int(v)] != inward:
                mismatch_count += 1
        if mismatch_count != len(markers):
            return None
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    target_families = [
        ("left", "top", 3),
        ("left", "bottom", 2),
        ("right", "top", 3),
        ("right", "bottom", 2),
    ]
    target_family = random.choice(target_families)
    for _ in range(400):
        try:
            g = _gen_attempt(target_family=target_family)
        except Exception:
            g = None
        if g is not None:
            return g
    for _ in range(100):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
