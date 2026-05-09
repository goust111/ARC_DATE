# Combo variant of bf45cf4b (index 26)
# combo_spec: D4/3+D9/0
# Name: 标记像素触发二次变换

from __future__ import annotations

import math
import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ImportError:  # pragma: no cover - fallback for direct file execution
    from solvers.arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


"""
design_analysis:
  fused_rule_summary:
    preserved_backbone:
      - 输入仍有且仅有两个 8 连通前景对象: 一个 control sprite, 一个 polychrome square tile
      - control 的 bbox 仍作为“像素放大模板”: 非背景格对应贴一个 tile, 背景洞对应整块背景
      - 输出仍是紧凑裁剪后的拼接网格, 尺寸为 (ctrl_h * tile_h) x (ctrl_w * tile_w)
    fused_change:
      - D4/3 提供“control 内有特殊 marker 像素”这个结构信号
      - D9/0 提供“不是一步到位, 而是先做基础放大, 再做第二步额外变换”这个处理链
      - 融合后具体落地为:
        1. 先按原题规则把所有 control 非背景格都扩成原始 tile
        2. 再只对 marker 像素对应的那些放大块执行固定的二次变换: 顺时针 90 度旋转
      - 因而 marker 不再直接决定“贴哪种 tile”, 而是触发对 stage-1 结果中对应局部块的后处理

  behavior_trace:
    small_example:
      tile:
        - [1, 2, 3]
        - [4, 5, 6]
        - [7, 8, 9]
      rotated_cw:
        - [7, 4, 1]
        - [8, 5, 2]
        - [9, 6, 3]
      control:
        - [5, 0, 5]
        - [6, 5, 0]
      source_task:
        - 四个非背景位置都贴原始 tile
      fused_task_step_1:
        - 仍先在 (0,0),(0,2),(1,0),(1,1) 四处都贴原始 tile
      fused_task_step_2:
        - control 中唯一 marker 颜色 6 位于 (1,0)
        - 只把输出里该块覆盖成 rotated_cw
        - 其余三个块保持原始 tile
      visible_difference:
        - 对比原题, 输出中仅有一个局部块方向不同
        - 对比单纯 marker 直接选 tile 版本, 这里的语义明显是“先复制, 后局部旋转”

  complexity_calibration:
    anti_degeneracy_choices:
      - control 恰有两种前景色: 主色 + 少量 marker 色, 并保留背景洞
      - tile 强制为 3-5 的正方形、至少 3 种前景色且不含背景
      - tile 必须对 90 度旋转不对称, 且旋转后至少有多个像素变化
      - control 至少有多个普通填充格, 使输出里同时出现多个原始 tile 与若干旋转 tile 供肉眼对比
    why_not_more_complex:
      - 不引入多种 marker 颜色到多种旋转/翻转的映射
      - 不把第二步编码到对象相对位置里, 避免和 marker 语义竞争
      - marker 数量限制在 1-3 个且保持少数, 让解题者能稳定把 input 中的异色点对齐到 output 中对应的异常块

  implicit_assumption_audit_for_main:
    - 原 main 用 `len(cols)` 排序来区分 tile/control:
      - 融合后 control 也变成双色, 该假设失效
      - 新 solver 改为识别“带背景洞 + 恰有两种前景色且少数色为 marker”的对象作为 control
    - 原 main 用 `ctrl_cols[0]` 作为唯一填充色:
      - 融合后 control 有主色和 marker 色两种前景色, 该假设失效
      - 新 solver 先找出少数 marker 色, 再把所有非背景格都参与 stage-1 放大
    - 原 main 在贴完 tile 后直接 return:
      - 融合后必须再执行 stage-2 局部旋转覆盖, 不能提前返回
    - 新 solver 的新增假设:
      - control 恰有两种前景色, 且 marker 色出现次数严格少于主色
      - block 是正方形、无背景洞、至少 3 色
      - 二次变换固定为顺时针 90 度旋转
      - generator 会逐条保证这些前提

  placement_legality:
    - 输入网格尺寸保持在 12..18
    - tile 与 control 仅做水平或垂直分离放置, 中间至少 1 格背景, 因而始终是两个 8 连通对象
    - control 使用 transparent=bg 粘贴, 保留 bbox 内背景洞
    - tile 和 control 都不会越界, 且不重叠

  visual_observability:
    - input 中 marker 是 control 内少量异色点, 与大多数主色填充格形成清晰对比
    - output 中有 1-3 个旋转块, 同时存在多个未旋转块作参照
    - tile 被强制设计成旋转后明显不同, 避免“旋了但肉眼看不出”
    - 2-4 个样例足以归纳: marker 位置决定哪一块异常, 异常方式恒为 90 度旋转

  cleanliness_and_color_reasoning:
    - 对象不重叠, 不叠层
    - 不使用绝对颜色编号语义; marker 只是 control 中的稀有色
    - 颜色变化只发生在 tile 内部朝向, 不是额外的重映射表

  generator_non_degeneracy_metric:
    - 定义 baseline = 忽略 marker 特权时的原题放大输出
    - 要求每个 marker 对应块与 baseline 的对应块都至少有 1 个像素变化
    - 同时要求 fused_output 与 baseline 至少在 max(3, tile_size) 个像素上不同
    - 因为只有 marker 对应的 tile 会被旋转, 该指标直接量化“二次变换是否真的发生且可见”

  main_generator_contract_review:
    - `_split_objects` 假设 control 可由“有洞 + 双前景色 + 少数 marker 色”唯一识别
      - generator 保证 tile 无背景洞且至少 3 色
    - `_marker_color` 假设两种前景色计数严格不等, 较少者为 marker
      - generator 在 control 构造时把 marker 数量限制为 1-3 且严格少于主色
    - `_rotate_tile_cw` 假设 block 为正方形
      - generator 只生成方形 tile
    - stage-2 覆盖假设 marker 对应位置在输出中已有一个 stage-1 tile
      - generator 保证 marker 只落在原本 filled 的位置上
"""


def _rotate_tile_cw(tile: np.ndarray) -> np.ndarray:
    return np.rot90(tile, k=3).copy()


def _fg_counts(patch: np.ndarray, bg: int) -> dict[int, int]:
    return {int(col): int(np.sum(patch == int(col))) for col in colors_present(patch, exclude={bg})}


def _is_control_patch(patch: np.ndarray, bg: int) -> bool:
    counts = sorted(_fg_counts(patch, bg).values())
    return bool(np.any(patch == int(bg))) and len(counts) == 2 and counts[0] >= 1 and counts[1] > counts[0]


def _split_objects(grid: np.ndarray, bg: int) -> tuple[dict, dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        infos.append({"bbox": comp["bbox"], "patch": np.array(patch, dtype=int), "fg_counts": _fg_counts(patch, bg)})

    ctrl_candidates = [info for info in infos if _is_control_patch(info["patch"], bg)]
    assert len(ctrl_candidates) == 1

    ctrl_info = ctrl_candidates[0]
    block_info = next(info for info in infos if info is not ctrl_info)
    bh, bw = map(int, block_info["patch"].shape)
    assert bh == bw
    assert not np.any(block_info["patch"] == int(bg))
    assert len(block_info["fg_counts"]) >= 3
    return block_info, ctrl_info


def _marker_color(ctrl: np.ndarray, bg: int) -> int:
    counts = _fg_counts(ctrl, bg)
    assert len(counts) == 2
    marker = min(counts, key=counts.get)
    fill = max(counts, key=counts.get)
    assert counts[int(marker)] >= 1
    assert counts[int(fill)] > counts[int(marker)]
    return int(marker)


def _compose_stage1(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, block, r * bh, c * bw)
    return out


def _compose_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    out = _compose_stage1(block, ctrl, bg)
    marker_color = _marker_color(ctrl, bg)
    rotated = _rotate_tile_cw(block)
    bh, bw = map(int, block.shape)
    for r, c in np.argwhere(ctrl == int(marker_color)):
        paste(out, rotated, int(r) * bh, int(c) * bw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    First scale the control sprite exactly as in the source task, then rotate the
    tile copies at marker-colored positions by 90 degrees clockwise.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    block_info, ctrl_info = _split_objects(g, bg)
    return _compose_output(block_info["patch"], ctrl_info["patch"], bg)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _random_points(h: int, w: int, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    picks = rng.choice(h * w, size=k, replace=False)
    return [(int(v // w), int(v % w)) for v in picks]


def _make_tile(tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        size = int(rng.integers(3, 6))
        n_colors = int(rng.integers(3, min(5, len(tile_colors) + 1)))
        chosen = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]
        tile = rng.choice(chosen, size=(size, size), replace=True).astype(int)

        for (r, c), color in zip(_random_points(size, size, n_colors, rng), chosen):
            tile[r, c] = int(color)

        diff = int(np.count_nonzero(tile != _rotate_tile_cw(tile)))
        if diff < max(3, size):
            continue
        return tile
    raise RuntimeError("failed to build rotation-asymmetric tile")


def _make_ctrl(bg: int, fill_color: int, marker_color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
        area = h * w
        min_fill = max(5, int(math.ceil(0.45 * area)))
        max_fill = min(area - 1, int(math.floor(0.72 * area)))
        target = int(rng.integers(min_fill, max_fill + 1))

        sr = int(rng.integers(0, h))
        sc = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[sr, sc] = True

        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) != target or target >= area:
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(fill_color)
        filled = np.argwhere(mask)
        max_markers = min(3, max(1, target // 3))
        marker_count = int(rng.integers(1, max_markers + 1))
        if target - marker_count < 3:
            continue
        marker_indices = np.atleast_1d(rng.choice(len(filled), size=marker_count, replace=False))
        for idx in marker_indices:
            mr, mc = map(int, filled[int(idx)])
            ctrl[mr, mc] = int(marker_color)

        if not _is_control_patch(ctrl, bg):
            continue
        return ctrl
    raise RuntimeError("failed to build marked control sprite")


def _compose_baseline_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    return _compose_stage1(block, ctrl, bg)


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = int(rng.integers(1, 4))

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        H = int(rng.integers(max(12, max(lh, rh) + 2), 19))
        W = int(rng.integers(max(12, lw + rw + gap + 2), 19))
        left_c = int(rng.integers(1, W - (lw + gap + rw)))
        right_c = left_c + lw + gap
        left_r = int(rng.integers(1, H - lh))
        right_r = int(rng.integers(1, H - rh))
        if left_obj is tile:
            tile_pos = (left_r, left_c)
            ctrl_pos = (right_r, right_c)
        else:
            ctrl_pos = (left_r, left_c)
            tile_pos = (right_r, right_c)
    else:
        if rng.random() < 0.5:
            top_obj, bot_obj = tile, ctrl
        else:
            top_obj, bot_obj = ctrl, tile
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        H = int(rng.integers(max(12, th + bh + gap + 2), 19))
        W = int(rng.integers(max(12, max(tw, bw) + 2), 19))
        top_r = int(rng.integers(1, H - (th + gap + bh)))
        bot_r = top_r + th + gap
        top_c = int(rng.integers(1, W - tw))
        bot_c = int(rng.integers(1, W - bw))
        if top_obj is tile:
            tile_pos = (top_r, top_c)
            ctrl_pos = (bot_r, bot_c)
        else:
            ctrl_pos = (top_r, top_c)
            tile_pos = (bot_r, bot_c)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=7, replace=False)]
    bg = int(palette[0])
    ctrl_fill = int(palette[1])
    ctrl_marker = int(palette[2])
    tile_palette = palette[3:]

    tile = _make_tile(tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_fill, ctrl_marker, rng)
    expected = _compose_output(tile, ctrl, bg)
    baseline = _compose_baseline_output(tile, ctrl, bg)
    changed_pixels = int(np.count_nonzero(expected != baseline))

    if changed_pixels < max(3, int(tile.shape[0])):
        return None

    inp = _make_input(tile, ctrl, bg, rng)
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    try:
        block_info, ctrl_info = _split_objects(inp, bg)
    except AssertionError:
        return None

    if not np.array_equal(block_info["patch"], tile):
        return None
    if not np.array_equal(ctrl_info["patch"], ctrl):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
