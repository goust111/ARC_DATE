# Variant of bf45cf4b
# Dimension: D4 — 控制精灵的结构 (src)
# Change: 单色矩形区域内含背景色空洞（填充率0.28-0.72），作为二值掩码控制瓦片放置... → 控制精灵包含标记像素指示特殊操作

from __future__ import annotations

import math
import numpy as np
from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


"""
design_analysis:
  variant_summary:
    preserved_backbone:
      - 输入仍然只有两个 8 连通前景对象: 一个 control sprite, 一个 polychrome square tile
      - 输出仍然是把 control bbox 按 tile 尺寸放大后的紧凑拼接图
      - control 中的背景洞仍然对应输出中的整块背景区域
    changed_dimension_D4:
      - 原题 control 只有一种前景色, 只表达“放 tile / 不放 tile”
      - 变体 control 有两种前景色:
        - 主色: 放原始 tile
        - 标记色: 只出现 1 次, 在对应输出位置放 tile 的左右镜像版本
      - 这让解题者必须同时发现:
        - control 里那个异色像素不是第二个普通填充色, 而是“特殊指令”
        - 特殊指令对应的操作是 tile 的左右翻转

  concrete_pixel_behavior:
    input_objects:
      control_sprite:
        - 一个紧凑裁剪的矩形 bbox
        - bbox 内同时含有:
          - 背景洞
          - 若干主色像素
          - 恰好 1 个标记色像素
        - 标记像素总是落在原本应为“filled”的位置上, 所以它仍属于同一前景连通块
      building_block:
        - 一个 3-5 的正方形满铺彩色 tile
        - 至少 3 种前景色
        - 不含背景色, 避免与带洞的 control 混淆
        - 必须对左右翻转不对称, 否则 marker 的效果肉眼不可见
    output_rule:
      - 遍历 control bbox 的每个格子
      - 背景格 -> 留空一个 tile 大小的背景区域
      - 主色格 -> 贴原始 tile
      - 标记格 -> 贴 np.fliplr(tile)

  behavior_trace:
    small_example:
      tile:
        - [1, 2, 3]
        - [3, 1, 2]
        - [2, 3, 1]
      mirrored_tile_lr:
        - [3, 2, 1]
        - [2, 1, 3]
        - [1, 3, 2]
      control:
        - [5, 0, 5]
        - [5, 6, 0]
      original_logic:
        - 只要 control 非背景, 四个非背景位置全部贴原始 tile
      variant_logic:
        - (0,0),(0,2),(1,0) 贴原始 tile
        - (1,1) 因为是标记色 6, 改贴 mirrored_tile_lr
      difference:
        - 只有一个放大块发生变化, 但它会和同图中的多个原始 tile 并列出现
        - 解题者可以把异色 marker 的位置与输出中唯一的镜像块对齐起来

  complexity_calibration:
    anti_degeneracy_choices:
      - tile 强制至少 3 色, control 恰好 2 色, 避免两对象都“多色”后难分辨
      - tile 强制无背景洞, control 强制有背景洞, 进一步拉开对象类型
      - tile 强制左右不对称, 且与其镜像至少有多个像素不同, 保证视觉可观察
      - control 保持多个普通填充像素, 让输出中有多个原始 tile 可作为参照
    why_not_more_complex:
      - 不引入多 marker / 多种翻转轴 / 颜色到操作的映射表
      - 否则 2-4 个样例不足以让人稳定归纳规则, 会超出本维度需要

  implicit_assumption_audit_for_main:
    - find_components(..., monochromatic=False) 仍然成立, 因为 control 虽变成双色, 但仍是单个前景连通块
    - 原 main 依赖“control 只有 1 个前景色”, 通过 ctrl_cols[0] 直接做掩码
      - 该前提在变体中失效
      - 新 solver 必须显式区分 control 主色与 singleton marker 色
    - 原 main 依赖“颜色数更多的那个对象就是 block”
      - 该前提在变体中也不再稳固, 因为 control 从 1 色变成 2 色
      - 新 solver 改为识别“带洞且恰有一个 singleton 颜色”的对象作为 control
    - 新 solver 额外假设:
      - control 的前景色正好两种
      - 其中一种只出现 1 次
      - block 不含背景且是正方形
      - flip 规则固定为左右翻转
      - generator 会逐条保证这些条件

  placement_legality:
    - 输入网格仍在 12-18 范围内随机取值
    - tile 与 control 只做水平或垂直分离摆放
    - 两 bbox 之间至少保留 1 列或 1 行纯背景, 保证 8 连通下仍是两个对象
    - control 用 transparent=bg 粘贴, 使 bbox 内的洞保持为背景

  visual_observability:
    - 每个样例都只有 1 个 marker, 所以输出中也只有 1 块镜像 tile
    - 同一个输出里还会出现多个普通 tile 副本, 人眼可以直接比较“这一块怎么反了”
    - marker 色与主色不同, 在 input 中很醒目; 其位置和输出中特殊块的位置一一对应

  cleanliness_and_color_reasoning:
    - control 和 tile 空间上完全分离, 不叠加
    - 不引入绝对颜色语义; marker 只是 control 中的稀有色, 各样例颜色可变
    - 颜色变化的可推断性来自结构角色, 不是来自固定颜色编号
"""


def _flip_tile(tile: np.ndarray) -> np.ndarray:
    return np.fliplr(tile).copy()


def _fg_counts(patch: np.ndarray, bg: int) -> dict[int, int]:
    return {int(col): int(np.sum(patch == int(col))) for col in colors_present(patch, exclude={bg})}


def _is_control_patch(patch: np.ndarray, bg: int) -> bool:
    counts = sorted(_fg_counts(patch, bg).values())
    return bool(np.any(patch == int(bg))) and counts[:1] == [1] and len(counts) == 2 and counts[1] >= 2


def _split_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        infos.append({"patch": patch, "fg_counts": _fg_counts(patch, bg)})

    ctrl_candidates = [info for info in infos if _is_control_patch(info["patch"], bg)]
    assert len(ctrl_candidates) == 1

    ctrl_info = ctrl_candidates[0]
    block_info = next(info for info in infos if info is not ctrl_info)
    block = np.array(block_info["patch"], dtype=int)
    ctrl = np.array(ctrl_info["patch"], dtype=int)

    bh, bw = block.shape
    assert bh == bw
    assert not np.any(block == int(bg))
    assert len(block_info["fg_counts"]) >= 3
    return block, ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up the control sprite by stamping a colorful tile, except that the
    single marker pixel stamps a left-right mirrored copy of the tile.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    block, ctrl = _split_objects(g, bg)
    flipped = _flip_tile(block)

    ctrl_counts = _fg_counts(ctrl, bg)
    assert len(ctrl_counts) == 2
    marker_color = min(ctrl_counts, key=ctrl_counts.get)
    fill_color = max(ctrl_counts, key=ctrl_counts.get)
    assert ctrl_counts[int(marker_color)] == 1
    assert ctrl_counts[int(fill_color)] >= 2

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)

    for r in range(ch):
        for c in range(cw):
            v = int(ctrl[r, c])
            if v == int(bg):
                continue
            tile = flipped if v == int(marker_color) else block
            paste(out, tile, r * bh, c * bw)
    return out


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


def _make_tile(tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    min_diff = max(3, size)
    n_colors = int(rng.integers(3, min(5, len(tile_colors) + 1)))
    chosen = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]

    for _ in range(200):
        tile = np.array(rng.choice(chosen, size=(size, size), replace=True), dtype=int)
        if any(int(np.sum(tile == color)) == 0 for color in chosen):
            continue
        if int(np.count_nonzero(tile != _flip_tile(tile))) < min_diff:
            continue
        return tile
    raise RuntimeError("failed to build asymmetric tile")


def _make_ctrl(bg: int, fill_color: int, marker_color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(4, 7))
    w = int(rng.integers(4, 7))
    area = h * w
    min_fill = max(4, int(math.ceil(0.42 * area)))
    max_fill = min(area - 1, int(math.floor(0.72 * area)))
    target = int(rng.integers(min_fill, max_fill + 1))

    ar = int(rng.integers(0, h))
    ac = int(rng.integers(0, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, ac] = True
    mask[ar, :] = True

    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        assert frontier
        pick = frontier[int(rng.integers(len(frontier)))]
        mask[pick] = True

    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(fill_color)

    filled = np.argwhere(mask)
    mr, mc = map(int, filled[int(rng.integers(len(filled)))])
    ctrl[mr, mc] = int(marker_color)
    return ctrl


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    flipped = _flip_tile(tile)
    counts = _fg_counts(ctrl, bg)
    marker_color = min(counts, key=counts.get)

    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            v = int(ctrl[r, c])
            if v == int(bg):
                continue
            paste(out, flipped if v == int(marker_color) else tile, r * th, c * tw)
    return out


def _compose_output_without_marker(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        H = int(rng.integers(max(12, max(lh, rh)), 19))
        W = int(rng.integers(max(12, lw + rw + gap), 19))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(0, slack + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - rw + 1))
        r_left = int(rng.integers(0, H - lh + 1))
        r_right = int(rng.integers(0, H - rh + 1))
        if left_obj is tile:
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if rng.random() < 0.5:
            top_obj, bot_obj = tile, ctrl
        else:
            top_obj, bot_obj = ctrl, tile
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        H = int(rng.integers(max(12, th + bh + gap), 19))
        W = int(rng.integers(max(12, max(tw, bw)), 19))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(0, slack + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - bh + 1))
        c_top = int(rng.integers(0, W - tw + 1))
        c_bot = int(rng.integers(0, W - bw + 1))
        if top_obj is tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

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
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)
    baseline = _compose_output_without_marker(tile, ctrl, bg)

    if np.array_equal(expected, baseline):
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    try:
        block2, ctrl2 = _split_objects(inp, bg)
    except AssertionError:
        return None

    if not np.array_equal(block2, tile):
        return None
    if not np.array_equal(ctrl2, ctrl):
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
