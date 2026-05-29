# Combo variant of 89565ca0 (index 26)
# combo_spec: D1/4+D3/2+D7/1
# Name: 嵌套方框+降序+反向映射

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 原题对象是“带分区的线框”，计数值来自封闭背景房间数；本融合版把源对象改成
#   同色嵌套方框，计数值直接等于可见方框层数。
# - 例：若某颜色可见 5 层同心方框，另一颜色 3 层，再另一颜色 2 层，则原 D3-only
#   版本会按层数升序输出 [2, 3, 5]，条长也分别是 [2, 3, 5]。
# - 本融合版先按层数降序排序，得到颜色顺序 [5, 3, 2]；再令 max_layers = 5，
#   条长改为 [1, 3, 4]，即 length = max_layers - layers + 1。
# - 差异不只一处：输入对象类型变了，输出排序方向反了，计数到长度的映射也反了。
#
# b) 复杂度校准
# - 每个对象仍只编码一个 1..6 的整数层数，规则可学；但嵌套层数、尺寸、间距、
#   少量重叠和噪声仍提供足够变化，视觉上明显不同于原题网格房间框。
# - 生成器保留 3-4 个对象并允许最多一组平局，避免全部相同造成排序/映射退化。
#
# c) 隐式假设审查
# - 原 main() 依赖“密集行列证据可重建墙”；对嵌套方框不成立，因此整条房间重建链
#   必须替换为“在方形 bbox 内检测多个方框边界”。
# - `rooms > 0` 识别真实对象的假设被替换成 `layers > 0`；生成器必须保证噪声色
#   不会偶然形成一个完整方框壳层。
# - `_cover_order()` 只依赖“同计数组内有重叠时，可从重叠像素看到谁在上面”。
#   该前提在嵌套方框下仍成立，所以平局逻辑可以保留。
# - D1/4 改了排序方向，因此不能再按升序遍历层数；D7/1 改了条长公式，因此也不能
#   继续 `out[r, :layers] = color`。
#
# d) 放置合法性
# - 每个真实对象都构造成正方形壳层掩码，放置时保证不越界。
# - 非重叠放置阶段允许“无重叠”或“受限小重叠”；强制重叠阶段只允许落在非保护像素，
#   并限制重叠像素比例，避免角点和关键边中点被吃掉。
#
# e) 视觉可观察性
# - 人类能直接看到：对象已不是房间网格，而是若干同色嵌套方框。
# - 输出中“层数最多的对象排最上面却最短”，“层数最少的对象排最下面却最长”是
#   连续样例里稳定复现的显著模式，不依赖颜色编号等不可视信息。
#
# f) 可视化整洁度
# - 噪声只撒在非对象可见像素上，不破坏壳层轮廓。
# - 重叠对数和重叠比例都较低，避免多层方框在同一局部堆得不可读。
#
# g) 变化的可推断性
# - 没有颜色重映射。颜色只承担对象身份，推理重点是“数方框层数 -> 降序排列 ->
#   用场景内最大层数做反向条长映射”。
#
# h) 防退化目标
# - 若样本退化回 D3-only，则会表现为升序且条长等于层数；若退化回仅 D1 或仅 D7，
#   则对象仍会不是嵌套方框或输出只改变一半。生成器通过真实嵌套方框输入、至少两种
#   不同层数、以及 oracle 检查共同约束，确保三项变化同时生效。
#
# i) 新 main() 假设与生成器保证
# - 每个真实颜色的可见 bbox 仍是正方形。
# - 每一层真实方框都保留四角，且四条边保留足够命中率以通过阈值检测。
# - 恰有一种非背景色不会检测出任何壳层，它就是噪声/填充色。
# - 若某组层数相同且需要平局打破，生成器会安排可见重叠使 `_cover_order()` 可判定。


_FRAME_COUNT_RANGE = (3, 4)
_LAYER_COUNT_RANGE = (1, 6)
_MAX_LAYERS_BY_FRAMES = {4: 5, 3: 6}
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    4: (26, 30, 26, 30, 24),
    3: (22, 30, 22, 30, 26),
}
_INNER_SIDE_RANGE = (3, 7)
_LAYER_GAP_RANGE = (1, 2)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_EXTRA_OVERLAP_RANGE = (0, 1)
_OVERLAP_RETRY = 160
_GEN_RETRIES = 300
_TARGETED_GEN_RETRIES = 220
_NEXT_FORCE_TIE = False

_SHELL_HIT_FRACTION = 0.5
_SMALL_SIDE_PROTECT_LIMIT = 4
_MAX_OVERLAP_SHARE = 0.2


def _paint_square_border(mask: np.ndarray, inset: int) -> None:
    n = int(mask.shape[0])
    r0 = int(inset)
    c0 = int(inset)
    r1 = int(n - 1 - inset)
    c1 = int(n - 1 - inset)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _mark_protected_points(protected: np.ndarray, inset: int) -> None:
    n = int(protected.shape[0])
    r0 = int(inset)
    c0 = int(inset)
    r1 = int(n - 1 - inset)
    c1 = int(n - 1 - inset)
    side = int(r1 - r0 + 1)
    if side <= _SMALL_SIDE_PROTECT_LIMIT:
        _paint_square_border(protected, inset)
        return

    mid = side // 2
    points = {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
        (r0, c0 + mid),
        (r1, c0 + mid),
        (r0 + mid, c0),
        (r0 + mid, c1),
    }
    for r, c in points:
        protected[int(r), int(c)] = True


def _detect_shell_insets(mask: np.ndarray) -> tuple[list[int], np.ndarray]:
    b = bbox(mask)
    if b is None:
        return [], np.zeros((0, 0), dtype=bool)

    cropped = crop_bbox(mask, b).astype(bool)
    h, w = cropped.shape
    if h != w or h < 3:
        return [], cropped

    n = int(h)
    insets: list[int] = []
    for inset in range(n // 2 + 1):
        side = int(n - 2 * inset)
        if side < 3:
            break
        r0 = int(inset)
        c0 = int(inset)
        r1 = int(n - 1 - inset)
        c1 = int(n - 1 - inset)
        top = int(cropped[r0, c0 : c1 + 1].sum())
        bottom = int(cropped[r1, c0 : c1 + 1].sum())
        left = int(cropped[r0 : r1 + 1, c0].sum())
        right = int(cropped[r0 : r1 + 1, c1].sum())
        threshold = max(3, math.ceil(_SHELL_HIT_FRACTION * side))
        corners_visible = (
            bool(cropped[r0, c0])
            and bool(cropped[r0, c1])
            and bool(cropped[r1, c0])
            and bool(cropped[r1, c1])
        )
        if corners_visible and min(top, bottom, left, right) >= threshold:
            insets.append(int(inset))

    return insets, cropped


def _layer_count_for_color(g: np.ndarray, color: int) -> int:
    insets, _cropped = _detect_shell_insets(g == int(color))
    return int(len(insets))


def _shells_for_color(g: np.ndarray, color: int) -> np.ndarray:
    mask = g == int(color)
    b = bbox(mask)
    out = np.zeros_like(g, dtype=bool)
    if b is None:
        return out

    insets, cropped = _detect_shell_insets(mask)
    if not insets:
        return out

    shell_mask = np.zeros_like(cropped, dtype=bool)
    for inset in insets:
        _paint_square_border(shell_mask, int(inset))

    r0, c0, r1, c1 = b
    out[r0 : r1 + 1, c0 : c1 + 1] = shell_mask
    return out


def _cover_order(
    g: np.ndarray, group: list[tuple[int, int, np.ndarray]]
) -> list[tuple[int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _layers, _shells in group}
    indegree: dict[int, int] = {color: 0 for color, _layers, _shells in group}

    for i, (ca, _la, sa) in enumerate(group):
        for cb, _lb, sb in group[i + 1 :]:
            overlap = sa & sb
            if not bool(overlap.any()):
                continue
            visible = set(int(v) for v in np.unique(g[overlap]))
            top_colors = visible & {int(ca), int(cb)}
            assert len(top_colors) == 1
            top = top_colors.pop()
            under = int(cb) if top == int(ca) else int(ca)
            if top not in above[under]:
                above[under].add(top)
                indegree[top] += 1

    ordered: list[tuple[int, int, np.ndarray]] = []
    remaining = set(by_color)
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1
        color = ready[0]
        remaining.remove(color)
        ordered.append(by_color[color])
        for nxt in above[color]:
            indegree[nxt] -= 1
    return ordered


def _inverse_bar_length(layers: int, max_layers: int) -> int:
    return int(max_layers - layers + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each nested-square object into a horizontal bar whose length decreases as the nesting depth increases.

    Input structure:
    - Black background with several same-color nested square drawings, plus one extra non-black noise color.
    - Each real drawing is a stack of 1px-thick concentric square borders sharing a color; the stack depth is its count value.
    - Drawings may overlap, but every true shell keeps enough visible perimeter to remain detectable inside a square bbox.

    Rule:
    - For each non-background color, examine its visible bbox and detect which insets still form square borders with all four corners present.
    - The number of detected shells is that color's layer count; the unique color with zero shells is the padding/noise color.
    - Order real colors by layer count in descending order, breaking ties by bottom-to-top cover order among overlapping reconstructed shell masks.
    - Let max_layers be the largest detected count. Output one row per real color with bar length `max_layers - layers + 1`, padded on the right by the noise color.

    Key insight:
    - The scene no longer hides its value inside room topology. Each object visibly advertises its count through nested square depth, but the output applies two inversions at once: deeper stacks move earlier in the ordering while also producing shorter bars.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        layers = _layer_count_for_color(g, int(color))
        shells = _shells_for_color(g, int(color))
        counts.append((int(color), int(layers), shells))

    drawings = [(color, layers, shells) for color, layers, shells in counts if layers > 0]
    pad_colors = [color for color, layers, _shells in counts if layers == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, np.ndarray]] = []
    for layers in sorted({layers for _color, layers, _shells in drawings}, reverse=True):
        group = [item for item in drawings if item[1] == layers]
        ordered.extend(_cover_order(g, group))

    max_layers = max(layers for _color, layers, _shells in ordered)
    bar_lengths = [_inverse_bar_length(layers, max_layers) for _color, layers, _shells in ordered]
    width = max(bar_lengths)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, ((color, _layers, _shells), bar_len) in enumerate(zip(ordered, bar_lengths)):
        out[r, :bar_len] = int(color)
    return out


def _choose_layer_counts(frame_count: int, force_tie: bool | None = None) -> list[int] | None:
    max_layers = _MAX_LAYERS_BY_FRAMES.get(frame_count, _LAYER_COUNT_RANGE[1])
    available = list(range(_LAYER_COUNT_RANGE[0], max_layers + 1))
    if force_tie is None:
        allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
    else:
        allow_tie = bool(force_tie) and frame_count >= 3
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, min(distinct_needed, len(available)))
        if len(chosen) < distinct_needed:
            return None
        counts = chosen + [random.choice(chosen)]
    else:
        if frame_count > len(available):
            return None
        counts = random.sample(available, frame_count)
    random.shuffle(counts)
    return [int(v) for v in counts]


def _random_gaps_with_sum(n: int, total: int, low: int, high: int) -> list[int] | None:
    if n == 0:
        return [] if total == 0 else None
    if total < n * low or total > n * high:
        return None

    gaps = [int(low)] * n
    remaining = int(total - n * low)
    for i in range(n):
        add_max = min(int(high - low), remaining)
        add = random.randint(0, add_max)
        gaps[i] += int(add)
        remaining -= int(add)
    idxs = list(range(n))
    random.shuffle(idxs)
    for idx in idxs:
        if remaining <= 0:
            break
        extra = min(int(high - gaps[idx]), remaining)
        if extra > 0:
            gaps[idx] += int(extra)
            remaining -= int(extra)
    if remaining != 0:
        return None
    random.shuffle(gaps)
    return gaps


def _build_nested_frame(target_layers: int, max_side: int) -> tuple[np.ndarray, np.ndarray] | None:
    gap_count = max(0, int(target_layers - 1))
    min_gap_total = int(gap_count * _LAYER_GAP_RANGE[0])
    max_gap_total = int(gap_count * _LAYER_GAP_RANGE[1])
    feasible_gap_total_max = int((max_side - _INNER_SIDE_RANGE[0]) // 2)
    if min_gap_total > feasible_gap_total_max:
        return None

    for _ in range(80):
        gap_total = random.randint(min_gap_total, min(max_gap_total, feasible_gap_total_max))
        gaps = _random_gaps_with_sum(
            gap_count,
            gap_total,
            _LAYER_GAP_RANGE[0],
            _LAYER_GAP_RANGE[1],
        )
        if gaps is None:
            continue

        inner_max = min(_INNER_SIDE_RANGE[1], max_side - 2 * gap_total)
        if inner_max < _INNER_SIDE_RANGE[0]:
            continue
        inner_side = random.randint(_INNER_SIDE_RANGE[0], inner_max)
        side = int(inner_side + 2 * gap_total)

        mask = np.zeros((side, side), dtype=bool)
        protected = np.zeros((side, side), dtype=bool)
        inset = 0
        for layer_idx in range(target_layers):
            if layer_idx > 0:
                inset += gaps[layer_idx - 1]
            _paint_square_border(mask, int(inset))
            _mark_protected_points(protected, int(inset))
        return mask, protected
    return None


def _pair_overlap_stats(
    frame_a: tuple[int, int, np.ndarray, np.ndarray],
    pos_a: tuple[int, int],
    frame_b: tuple[int, int, np.ndarray, np.ndarray],
    pos_b: tuple[int, int],
) -> tuple[bool, int, bool]:
    mask_a = frame_a[2]
    prot_a = frame_a[3]
    mask_b = frame_b[2]
    prot_b = frame_b[3]
    ra, ca = pos_a
    rb, cb = pos_b
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape

    r0 = max(int(ra), int(rb))
    c0 = max(int(ca), int(cb))
    r1 = min(int(ra + ha - 1), int(rb + hb - 1))
    c1 = min(int(ca + wa - 1), int(cb + wb - 1))
    if r0 > r1 or c0 > c1:
        return False, 0, False

    a_rs = slice(int(r0 - ra), int(r1 - ra + 1))
    a_cs = slice(int(c0 - ca), int(c1 - ca + 1))
    b_rs = slice(int(r0 - rb), int(r1 - rb + 1))
    b_cs = slice(int(c0 - cb), int(c1 - cb + 1))

    overlap = mask_a[a_rs, a_cs] & mask_b[b_rs, b_cs]
    if not bool(overlap.any()):
        return False, 0, False

    protected_hit = bool(
        (prot_a[a_rs, a_cs] & mask_b[b_rs, b_cs]).any()
        or (prot_b[b_rs, b_cs] & mask_a[a_rs, a_cs]).any()
    )
    return True, int(overlap.sum()), protected_hit


def _is_valid_overlap(
    frame_a: tuple[int, int, np.ndarray, np.ndarray],
    pos_a: tuple[int, int],
    frame_b: tuple[int, int, np.ndarray, np.ndarray],
    pos_b: tuple[int, int],
) -> bool:
    has_overlap, overlap_pixels, protected_hit = _pair_overlap_stats(
        frame_a, pos_a, frame_b, pos_b
    )
    if not has_overlap or protected_hit:
        return False

    shell_a = int(frame_a[2].sum())
    shell_b = int(frame_b[2].sum())
    allowed = max(1, math.floor(_MAX_OVERLAP_SHARE * min(shell_a, shell_b)))
    return overlap_pixels <= allowed


def _position_compatible_with_others(
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    positions: list[tuple[int, int]],
    idx: int,
    candidate: tuple[int, int],
) -> bool:
    for other_idx, other_pos in enumerate(positions):
        if other_idx == idx or other_pos[0] < 0:
            continue
        if not _is_valid_overlap(frames[idx], candidate, frames[other_idx], other_pos):
            has_overlap, _overlap_pixels, _protected_hit = _pair_overlap_stats(
                frames[idx], candidate, frames[other_idx], other_pos
            )
            if has_overlap:
                return False
    return True


def _initial_positions(
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = [(-1, -1)] * len(frames)
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].shape[0])
    for idx in order:
        side = int(frames[idx][2].shape[0])
        placed = False
        for _ in range(120):
            cand = (
                random.randint(0, grid_h - side),
                random.randint(0, grid_w - side),
            )
            if _position_compatible_with_others(frames, positions, idx, cand):
                positions[idx] = cand
                placed = True
                break
        if not placed:
            return None
    return positions


def _force_overlap(
    positions: list[tuple[int, int]],
    i: int,
    j: int,
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> bool:
    if _is_valid_overlap(frames[i], positions[i], frames[j], positions[j]) and _position_compatible_with_others(
        frames, positions, j, positions[j]
    ):
        return True

    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]

    r0_min = max(0, int(ri - fj_h + 1))
    r0_max = min(int(grid_h - fj_h), int(ri + fi_h - 1))
    c0_min = max(0, int(ci - fj_w + 1))
    c0_max = min(int(grid_w - fj_w), int(ci + fi_w - 1))
    if r0_min > r0_max or c0_min > c0_max:
        return False

    for _ in range(_OVERLAP_RETRY):
        rj = random.randint(r0_min, r0_max)
        cj = random.randint(c0_min, c0_max)
        cand = (int(rj), int(cj))
        if _is_valid_overlap(frames[i], positions[i], frames[j], cand) and _position_compatible_with_others(
            frames, positions, j, cand
        ):
            positions[j] = cand
            return True
    return False


def _render_frames(
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    positions: list[tuple[int, int]],
    render_order: list[int],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        shell_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(shell_arr)
        grid[rs + r0, cs + c0] = int(color)

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        shell_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(shell_arr)
        abs_rs = rs + r0
        abs_cs = cs + c0
        visible = grid[abs_rs, abs_cs] == int(color)
        frame_mask[abs_rs[visible], abs_cs[visible]] = True

    return grid, frame_mask


def _scatter_noise(grid: np.ndarray, frame_mask: np.ndarray, noise_color: int) -> None:
    total_cells = int(grid.size)
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)
    nonframe_rs, nonframe_cs = np.where(~frame_mask)
    if len(nonframe_rs) == 0:
        return
    chosen = np.random.permutation(len(nonframe_rs))[: min(noise_count, len(nonframe_rs))]
    grid[nonframe_rs[chosen], nonframe_cs[chosen]] = int(noise_color)


def _expected_output(
    frame_info: list[tuple[int, int, int]], noise_color: int
) -> np.ndarray:
    max_layers = max(count for count, _cover_rank, _color in frame_info)
    bar_lengths = [
        _inverse_bar_length(count, max_layers)
        for count, _cover_rank, _color in frame_info
    ]
    width = max(bar_lengths)
    expected = np.full((len(frame_info), width), noise_color, dtype=int)
    for r, ((count, _cover_rank, color), bar_len) in enumerate(zip(frame_info, bar_lengths)):
        expected[r, :bar_len] = int(color)
    return expected


def _gen_attempt(force_tie: bool | None = None) -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    layer_counts = _choose_layer_counts(frame_count, force_tie=force_tie)
    if layer_counts is None:
        return None

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    h_min, h_max, w_min, w_max, frame_max_side = _GRID_SIZE_BY_FRAMES.get(
        frame_count, _GRID_SIZE_BY_FRAMES[3]
    )
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for color, layers in zip(frame_colors, layer_counts):
        build_max_side = min(int(frame_max_side), int(grid_h - 2), int(grid_w - 2))
        built = _build_nested_frame(int(layers), build_max_side)
        if built is None:
            return None
        shell_mask, protected = built
        frames.append((int(color), int(layers), shell_mask, protected))

    positions = _initial_positions(frames, grid_h, grid_w)
    if positions is None:
        return None

    tied_pairs = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if layer_counts[i] == layer_counts[j]
    ]
    for i, j in tied_pairs:
        if not _force_overlap(positions, i, j, frames, grid_h, grid_w):
            return None

    non_tied = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if layer_counts[i] != layer_counts[j]
    ]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        placed = 0
        for i, j in non_tied:
            if _force_overlap(positions, i, j, frames, grid_h, grid_w):
                placed += 1
                if placed >= n_extra:
                    break

    render_order = list(range(frame_count))
    random.shuffle(render_order)
    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        (
            (layer_counts[i], render_order.index(i), frame_colors[i])
            for i in range(frame_count)
        ),
        key=lambda x: (-x[0], x[1]),
    )
    expected = _expected_output(frame_info, noise_color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    global _NEXT_FORCE_TIE

    desired_tie = bool(_NEXT_FORCE_TIE)
    _NEXT_FORCE_TIE = not _NEXT_FORCE_TIE

    for _ in range(_TARGETED_GEN_RETRIES):
        result = _gen_attempt(force_tie=desired_tie)
        if result is not None:
            return result

    for _ in range(_GEN_RETRIES - _TARGETED_GEN_RETRIES):
        result = _gen_attempt(force_tie=None)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
