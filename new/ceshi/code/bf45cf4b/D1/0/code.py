# Variant of bf45cf4b
# Dimension: D1 — 像素放大方式 (transform)
# Change: 控制精灵中每个填充像素替换为完整的多色方块瓦片副本，每个空洞像素替换为等大的背景区域... → 填充像素放大为瓦片、空洞像素放大为瓦片的颜色反转版
#
# Design analysis
# ===============
# spec_variant:
#   preserved_input_structure:
#     - 背景色仍为全局最常见色
#     - 输入仍恰好包含两个 8 连通前景对象:
#       1) 单色控制精灵: 一个带洞的矩形 bbox，只有一种前景色
#       2) 多色方块瓦片: 一个正方形 patch，至少两种前景色
#     - 两个对象仍至少隔 1 个背景像素，因此 `find_components(..., monochromatic=False)` 仍应得到 2 个组件
#   d1_change:
#     - 原题: ctrl 的填充像素 -> 贴正常瓦片；ctrl 的空洞像素 -> 留背景块
#     - 变体: ctrl 的填充像素 -> 贴正常瓦片；ctrl 的空洞像素 -> 贴“颜色反转瓦片”
#   concrete_inversion_rule:
#     - 使用固定全局映射 `c -> 9 - c`
#     - 只对瓦片中的前景色做映射；若瓦片内部含背景色，则背景保持不变
#     - 生成器进一步限制: 采样瓦片颜色时只从若干互补对里选一侧，因此正常瓦片与反转瓦片的颜色集合完全分离，肉眼容易看出对应关系
#   behavior_trace:
#     - 例: 若 tile =
#         [[1, 2],
#          [2, 1]]
#       则 inverted(tile) =
#         [[8, 7],
#          [7, 8]]
#     - 若 ctrl =
#         [[x, bg],
#          [bg, x]]
#       原题输出为:
#         [[tile, bg-block],
#          [bg-block, tile]]
#       本变体输出为:
#         [[tile, inv-tile],
#          [inv-tile, tile]]
#     - 差异只发生在“ctrl bbox 内原本是背景的位置”这一环节，其余对象识别、裁剪、拼接逻辑保持不变
#   complexity_calibration:
#     - 仍只需要识别 2 个对象，不引入额外物体，避免结构性歧义
#     - 但洞位不再是纯背景，解题者必须同时学会:
#       1) ctrl 充当放大掩码
#       2) 掩码的 0/1 对应两种瓦片版本
#       3) 两种瓦片版本之间存在固定颜色映射
#     - 为避免规则太隐蔽，生成器要求:
#       1) ctrl 至少有 2 个洞和 2 个填充像素
#       2) tile 使用 2-3 种颜色，并且每种颜色在 tile 中实际出现
#       3) inverted(tile) 的前景色都不等于背景色或 ctrl 色
#   main_implicit_assumption_review:
#     - `len(comps) == 2`: 仍要求输入只有两个前景组件；生成器通过分离放置保证
#     - `infos.sort(...)`: 仍依赖“tile 的前景色种类 > ctrl 的前景色种类”；生成器维持 1 vs 2/3
#     - `ctrl_cols[0]`: 仍依赖 ctrl 单色；生成器维持单色 ctrl
#     - `out = full(bg)`: 现在不再能只在填充位写入 tile；必须对每个 ctrl bbox 单元都写一个块，否则洞位会错误地退化为原题背景块
#     - `paste(out, block, ...)`: 对洞位要改成粘贴 `invert(block)`，不能沿用“什么都不做”的旧假设
#   placement_legality:
#     - 输入网格仍随机横排或竖排放置对象
#     - 两对象 bbox 之间至少隔 1 个背景像素，不重叠、不越界
#     - 控制精灵使用透明贴图放置，因此其内部洞不会覆盖背景
#   visual_observability:
#     - 在输出里，同一 tile 结构会以两套稳定颜色重复出现；洞位和填充位混排，可直接比较
#     - 反色关系不是依赖位置旋转/翻转，而是整块同形换色，因此 2-3 个样例即可看出“洞位不是空白，而是另一套固定配色”
#   cleanliness:
#     - 不增加第三种标记或额外装饰，避免输出过乱
#     - 反转 tile 与正常 tile 共享完全相同几何结构，只在颜色上变化，便于逐块比对
#   color_inference:
#     - 颜色映射固定为全局 `9-c`
#     - 生成器避免 tile 颜色与其反色同时出现在输入 tile 中，减少“交换还是替换”的歧义
from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


INV_PAIRS = ((0, 9), (1, 8), (2, 7), (3, 6), (4, 5))


def _invert_color(color: int) -> int:
    return 9 - int(color)


def _invert_tile(tile: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(tile, dtype=int, copy=True)
    mask = (out != int(bg))
    out[mask] = 9 - out[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up a monochrome control sprite by stamping a polychrome tile at each cell.

    Variant rule:
    - filled ctrl pixel -> paste the original tile
    - bg hole inside ctrl bbox -> paste the color-inverted tile
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": comp["bbox"], "patch": patch, "cols": cols})

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), bg, dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mask = (ctrl == ctrl_col)
    inv_block = _invert_tile(block, bg)

    out = np.full((ch * bh, cw * bw), bg, dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = block if bool(mask[r, c]) else inv_block
            paste(out, patch, r * bh, c * bw)
    return out


def _dihedral_orbits(size: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    s = int(size)
    for r in range(s):
        for c in range(s):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (c, s - 1 - r),
                (s - 1 - r, s - 1 - c),
                (s - 1 - c, r),
                (r, s - 1 - c),
                (s - 1 - r, c),
                (c, r),
                (s - 1 - c, s - 1 - r),
            }
            cells = sorted(orbit)
            seen.update(cells)
            out.append(cells)
    return out


def _pick_tile_colors(bg: int, ctrl_color: int, rng: np.random.Generator) -> list[int]:
    usable_pairs = [
        pair
        for pair in INV_PAIRS
        if int(bg) not in pair and int(ctrl_color) not in pair
    ]
    n_tile_colors = int(rng.integers(2, min(3, len(usable_pairs)) + 1))
    chosen_pairs_idx = rng.choice(len(usable_pairs), size=n_tile_colors, replace=False)
    tile_colors = []
    for idx in chosen_pairs_idx:
        a, b = usable_pairs[int(idx)]
        tile_colors.append(int(a if rng.integers(0, 2) == 0 else b))
    return tile_colors


def _make_tile(bg: int, ctrl_color: int, rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    tile_colors = _pick_tile_colors(bg, ctrl_color, rng)
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(bg), dtype=int)
    for i, color in enumerate(tile_colors):
        for r, c in orbits[i]:
            tile[r, c] = int(color)

    for orbit in orbits[len(tile_colors) :]:
        color = int(tile_colors[int(rng.integers(len(tile_colors)))])
        for r, c in orbit:
            tile[r, c] = color

    return tile


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


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    area = h * w
    min_fill = max(2, (area + 1) // 2)
    max_fill = min(area - 2, int(np.floor(0.72 * area)))
    if min_fill > max_fill:
        max_fill = min_fill
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
        pick_r, pick_c = frontier[int(rng.integers(len(frontier)))]
        mask[pick_r, pick_c] = True

    if area - int(mask.sum()) < 2:
        holes = np.argwhere(mask)
        rng.shuffle(holes)
        for rr, cc in holes:
            if area - int(mask.sum()) >= 2:
                break
            trial = mask.copy()
            trial[int(rr), int(cc)] = False
            if trial.any():
                comps = find_components(trial.astype(int), bg=0, connectivity=8, monochromatic=False, return_format="dict")
                if len(comps) == 1:
                    mask = trial

    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)
    return ctrl


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    inv_tile = _invert_tile(tile, bg)
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = tile if ctrl[r, c] != bg else inv_tile
            paste(out, patch, r * th, c * tw)
    return out


def _compose_original_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
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
    bg = int(rng.integers(0, 10))
    ctrl_candidates = [c for c in range(10) if c != bg]
    ctrl_color = int(rng.choice(ctrl_candidates))

    tile = _make_tile(bg, ctrl_color, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)
    old_expected = _compose_original_output(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))

    if sorted(len(cols) for cols in colors_per_comp) != [1, len(set(tile.flatten()) - {bg})]:
        return None
    if np.array_equal(expected, old_expected):
        return None
    if int(np.sum(ctrl == bg)) < 2:
        return None
    if len(set(int(x) for x in tile.flatten()) - {bg}) < 2:
        return None
    inv_tile = _invert_tile(tile, bg)
    if len((set(int(x) for x in tile.flatten()) - {bg}) & (set(int(x) for x in inv_tile.flatten()) - {bg})) != 0:
        return None
    if np.any(inv_tile == bg):
        return None
    if np.any(inv_tile == ctrl_color):
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
