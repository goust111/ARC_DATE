# Combo variant of 195c6913 (index 2)
# combo_spec: D3/1+D12/2
# Name: 碰撞变色+碰撞重置循环

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for combo D3/1 + D12/2
# =======================================
# a) 具体化与行为追踪
# - 输入外形不变：左上角一排 2x2 模板色块给出有序循环，col 0 上有 1-2 个 marker，路径仍是单个
#   4 连通的粗对角带，右下背景区有一个 2x2 终止块。
# - 原题在“进入 path 的每一步”都推进一次 idx，所以模板若为 [A,B,C]，marker=A，首段 5 格会涂
#   B,C,A,B,C；碰撞只改变方向，不改变段内配色机制。
# - D3/1 单变体把“换色时机”从每一步改成“成功碰撞转弯之后”，因此同一段直行会整段同色；若继续
#   沿模板推进，则首段是 AAAAA，第一次成功转弯后下一段会变成 BBBB，之后再变成 CCCC。
# - 本融合再叠加 D12/2：每次成功碰撞后，虽然仍以“碰撞事件”为颜色切换触发点，但循环相位立刻
#   重置回 0。结果是“本来要切到下一模板色”这一动作被重置覆盖，下一段仍回到 A。
# - 具体例子：模板 [2,8,4]，marker=2。原题输出段内是 8,4,2,8,...；D3/1 是 2 段、8 段、4 段；
#   本融合则变成 2 段、2 段、2 段。几何转弯和终止色碰撞点仍保留，但整条轨迹只使用 cycle[0]。
#
# b) 复杂度校准
# - 若样本没有成功转弯，D12/2 的“重置”就不可见，只剩 D3/1 的单段纯色效果；因此 generator
#   必须强制至少一次成功转弯，而且转弯后的下一段长度不能太短。
# - 若模板只有 2 色但后续段太短，人眼依旧不容易区分 D3/1 与本融合；为此生成器收紧到 3-4 个
#   互异模板色，并要求融合输出与 D3/1 输出至少有多个像素不同。
# - 没有新增对象，也不改路径拓扑和 CCW/CW 交替规则，所以难度仍与原题同级，只是颜色规律更集中。
#
# c) 隐式假设审查
# - 原 solver 的 `idx = (idx + 1) % len(pattern)` 位于“每步进入 path”分支，隐含前提是颜色按步推进；
#   融合后这个前提失效，段内必须保持常色。
# - D3/1 中 `cur_color` 在成功碰撞后切换到下一模板色；加入 D12/2 后，这个“下一色”不能保留，
#   否则会退化回 D3/1。融合版必须在成功转弯后把当前段颜色重新绑定到 `pattern[0]`。
# - `visited` 状态若仍只记录 `(r,c,dr,dc,hits_parity)` 就足够描述融合行为，因为颜色已经塌缩到
#   单一首色；但为了统一对照模式实现，代码仍显式保留 `idx/cur_color`，避免与 D3-only 对照逻辑分叉。
# - 原代码把背景碰撞点涂成 `end_color`，这与融合规则不冲突，仍应保留；模板块与终止块擦除逻辑也不变。
# - 虽然当前 generator 继续令 marker=cycle[0]，但 `main()` 不能把这一点写死：首段仍应从 marker 对应
#   的循环位起步，只是在首次成功碰撞后被 D12/2 重置回 0。
#
# d) 放置合法性
# - band / fjord / interior-island 几何构造沿用原题，继续保证 path 单连通、marker 嵌入 col 0，
#   模板块在左上留白区，终止块在右下背景区且与 path 至少隔 1 格。
# - generator 仍依赖 solver 断言保证两条轨迹不重叠，并在构造后检查 `path_color` 仍是最常见非背景色。
# - 额外过滤：至少一次成功转弯、至少一个“转弯后的直行段”长度 >= 2、融合输出只使用一个轨迹色。
#
# e) 视觉可观察性
# - 人类从 2-4 组样例能看出的稳定线索是：模板明明给出多色循环，但输出轨迹所有 path 访问格都被
#   涂成 marker 同色，只在碰撞角上出现终止色小点。
# - 由于保留了多个模板块，解题者会自然对比“模板有多个颜色”与“轨迹却始终只有首色”，从而推断
#   出“碰撞后虽然以模板为依据，但每次都被重置回起点相位”。
# - 这里没有利用颜色编号大小、排序等人类不可见属性；全部线索都来自空间结构和可见颜色重复。
#
# f) 可视化整洁度
# - 输出比原题更干净：原题的棋盘式逐格换色被整段单色替代，同一路径不会叠加多套条纹。
# - 终止色仍只出现在碰撞点，模板块和终止块被擦除，轨迹、背景和碰撞角三类元素保持清晰分层。
#
# g) 变化的可推断性
# - 颜色变化被压缩成最少情形：轨迹始终使用 cycle[0]，碰撞点使用单一 `end_color`。
# - 为避免样本退化成“只是某次随机生成刚好看起来单色”，generator 会同时对照 legacy 和 D3-only：
#   1) 融合输出必须与 D3-only 不同，证明“碰撞后重置”确实生效；
#   2) 融合输出必须与原题不同，证明没有退回逐步循环。


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker replaces a path cell, not background."""
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


def _extract_scene(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    int,
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
    pat_set = set(pattern)
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
    return g, H, W, template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers


def _next_bounce_color(
    pattern: list[int],
    idx: int,
    hit_color: int,
    template_bg: int,
    path_color: int,
) -> tuple[int, int]:
    if hit_color == int(template_bg) or hit_color == int(path_color):
        idx = int((idx + 1) % len(pattern))
        return idx, int(pattern[idx])
    if hit_color in pattern:
        return int(pattern.index(hit_color)), int(hit_color)
    return int(idx), int(hit_color)


def _trace_marker(
    g: np.ndarray,
    path_color: int,
    pattern: list[int],
    template_bg: int,
    end_color: int | None,
    marker: tuple[int, int, int],
    *,
    mode: str,
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
            "segment_colors": [],
            "segment_lengths": [],
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    # D12/2 only resets after a successful bounce; the opening segment still
    # inherits the marker's cycle position.
    cur_color = int(pattern[idx])
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
            if mode == "legacy":
                idx = int((idx + 1) % len(pattern))
                paint_color = int(pattern[idx])
            else:
                paint_color = int(cur_color)
            if out is not None:
                out[nr, nc] = int(paint_color)
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
            dr, dc = int(tdr), int(tdc)
            hits += 1
            if mode == "bounce":
                idx, cur_color = _next_bounce_color(
                    pattern, idx, int(g[nr, nc]), template_bg, path_color
                )
            elif mode == "combo":
                # D3 would advance to the next segment color here, but D12/2
                # immediately resets the phase, collapsing every segment to cycle[0].
                idx = 0
                cur_color = int(pattern[0])
            segment_colors.append(int(cur_color if mode != "legacy" else pattern[idx]))
            segment_lengths.append(0)
            continue
        break

    return {
        "cells": my_cells,
        "turns": int(hits),
        "path_steps": int(path_steps),
        "segment_colors": [int(x) for x in segment_colors],
        "segment_lengths": [int(x) for x in segment_lengths],
    }


def _solve(
    grid: np.ndarray,
    *,
    mode: str,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    g, _H, _W, template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers = _extract_scene(
        grid
    )

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    traces: list[dict[str, object]] = []
    for marker in markers:
        trace = _trace_marker(
            g,
            path_color,
            pattern,
            template_bg,
            end_color,
            marker,
            mode=mode,
            out=out,
        )
        per_marker_cells.append(trace["cells"])  # type: ignore[arg-type]
        traces.append(trace)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    if not return_stats:
        return out

    positive_segment_colors: list[int] = []
    all_segment_lengths: list[int] = []
    post_turn_lengths: list[int] = []
    turned_segments = 0
    for trace in traces:
        colors = list(map(int, trace["segment_colors"]))  # type: ignore[arg-type]
        lengths = list(map(int, trace["segment_lengths"]))  # type: ignore[arg-type]
        for i, (color, length) in enumerate(zip(colors, lengths)):
            all_segment_lengths.append(int(length))
            if length > 0:
                positive_segment_colors.append(int(color))
                if i > 0:
                    turned_segments += 1
                    post_turn_lengths.append(int(length))

    return out, {
        "markers": int(len(markers)),
        "total_turns": int(sum(int(t["turns"]) for t in traces)),
        "markers_with_turn": int(sum(1 for t in traces if int(t["turns"]) > 0)),
        "turned_segments": int(turned_segments),
        "distinct_positive_segment_colors": int(len(set(positive_segment_colors))),
        "max_segment_len": int(max(all_segment_lengths, default=0)),
        "max_post_turn_segment_len": int(max(post_turn_lengths, default=0)),
        "segment_colors_per_marker": [
            list(map(int, t["segment_colors"])) for t in traces  # type: ignore[arg-type]
        ],
        "segment_lengths_per_marker": [
            list(map(int, t["segment_lengths"])) for t in traces  # type: ignore[arg-type]
        ],
    }


def _count_changed_cells(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.array(a, dtype=int) != np.array(b, dtype=int)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate the original alternating-turn path walk, but collapse every straight segment to the template's first color.

    Input structure:
    - A left-top row of 2x2 monochrome blocks still defines an ordered color cycle.
    - One or two isolated template-colored pixels on col 0 act as start markers embedded inside the path band.
    - A separate 2x2 block in the opposite background region provides the terminal collision color.

    Rule:
    - Each marker walks along the path in the first available direction and paints every visited path cell in its current segment with one constant color.
    - When the next step is non-path, that impact cell is painted with the end color, then the particle attempts the scheduled 90-degree turn.
    - Turn directions still alternate strictly CCW, CW, CCW, CW; if the scheduled turn is not path, the particle stops.
    - A successful bounce would normally advance to the next template color, but the fused reset immediately restores cycle index 0, so every post-bounce straight segment again paints with template color[0].

    Key insight:
    - The path geometry and alternating turns still determine where the trajectory bends; the fused color logic erases all post-bounce phase drift, so generated instances collapse to same-colored straight runs punctuated only by end-color collision dots.
    """
    return _solve(grid, mode="combo")


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
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


def _gen_attempt(n_markers: int | None = None) -> np.ndarray | None:
    H = W = random.randint(20, 30)
    if n_markers is None:
        n_markers = 2 if random.random() < 0.5 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(3, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

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
        combo_out, combo_stats = _solve(g, mode="combo", return_stats=True)
        bounce_out, bounce_stats = _solve(g, mode="bounce", return_stats=True)
        legacy_out = _solve(g, mode="legacy")
    except Exception:
        return None

    changed_vs_bounce = _count_changed_cells(combo_out, bounce_out)
    changed_vs_legacy = _count_changed_cells(combo_out, legacy_out)

    if int(combo_stats["total_turns"]) < 1:
        return None
    if int(combo_stats["turned_segments"]) < 1:
        return None
    if int(combo_stats["max_post_turn_segment_len"]) < 2:
        return None
    if int(combo_stats["distinct_positive_segment_colors"]) != 1:
        return None
    if int(bounce_stats["distinct_positive_segment_colors"]) < 2:
        return None
    if changed_vs_bounce < 2:
        return None
    if changed_vs_legacy < 6:
        return None

    return g


def generate_input() -> np.ndarray:
    target_markers = 2 if random.random() < 0.5 else 1
    for _ in range(350):
        try:
            g = _gen_attempt(target_markers)
        except Exception:
            g = None
        if g is not None:
            return g
    other_markers = 1 if target_markers == 2 else 2
    for _ in range(100):
        try:
            g = _gen_attempt(other_markers)
        except Exception:
            g = None
        if g is not None:
            return g
    for _ in range(50):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
