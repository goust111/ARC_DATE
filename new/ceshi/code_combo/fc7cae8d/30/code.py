# Combo variant of fc7cae8d (index 30)
# combo_spec: D8/0+D9/0
# Name: 两步变换+颜色图例

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

"""
设计分析
========
a) 具体化与行为追踪
- 原题输入: 边框上两条控制边直接给出目标朝向; 内部矩形的两条引导边颜色与控制边颜色相同。
- 融合后输入分成三个视觉区:
  1. 外框: 仍有一条水平控制边和一条垂直控制边。
  2. 主体: 内部矩形仍需做 D8 对齐, 但两条引导边不再直接使用控制边颜色, 而是使用“符号色”。
  3. 图例区: 放在内角附近, 由若干分离的 2x2 色砖组成。
     - 解码砖: [[symbol, control], [symbol, control]] 表示“符号色 -> 控制边颜色”。
     - 动作砖: [[h_ctrl, v_ctrl], [src, dst]] 表示“当边框控制色对为 (h_ctrl, v_ctrl) 时, 将核心中的 src 替换成 dst”。
- 小例子:
  边框控制边是 top=2, right=8。主体引导边颜色是 6 和 4, 不再直接等于 2/8。
  图例里有解码砖 [[6,2],[6,2]] 和 [[4,8],[4,8]], 所以先知道“6 对应 top 控制色 2, 4 对应 right 控制色 8”。
  第一步把主体经某个 D8 变换摆正并裁掉两条引导边, 得到核心:
      [[3,7,3],
       [1,7,1]]
  图例里还有动作砖 [[2,8],[7,9]], 所以第二步把核心里的 7 全改成 9, 输出:
      [[3,9,3],
       [1,9,1]]
- 与原题差异:
  原题只需“看颜色相等 -> 对齐 -> 裁剪”; 现在必须先查图例做符号解码才能完成对齐, 且裁剪后还要继续查图例完成替色。

b) 复杂度校准
- 若图例只有一块动作砖, 第二步过于直白; 因此生成器放 3-4 块动作砖, 只有一块键值匹配当前边框控制色对。
- 若 src 颜色只出现 1 次, 替色不明显; 因此强制 src 在核心中至少出现 3 次。
- 若主体或带引导边的对象对某些 D8 变换不敏感, 第一步不明显; 因此强制核心视觉上非 D8 对称, 且输入使用非恒等变换。
- 颜色总类控制在少量角色色上: 两个控制色、两个符号色、fill、src、dst 与一个辅助图案色, 保持可观察而不花哨。

c) 隐式假设审查
- `_detect_controls_frame()` 仍假设边框上只有两条近乎纯色控制边; 生成器继续保留这一结构, 另外两边只放灰色噪声。
- `_extract_center_object()` 仍假设“最大的内部连通块是主体”; 图例砖每块只有 4 格, 且与主体留空隔开, 因此前提成立。
- `_guide_edges()` 原本假设引导边颜色就是目标控制色; 融合后这个前提失效, 必须先通过解码砖把 guide color 映射到 control color。
- `_extract_legend_tiles()` 新假设是“图例砖均为独立 2x2 连通块”; 生成器用 0 间隔构造 legend block, 并与主体/边框保持空隙。
- `_action_map()` 新假设是“动作砖中当前 control pair 唯一”; 生成器显式保证动作键唯一。
- 第二步替色依赖 `src != dst` 且 `dst` 不预先出现在核心中, 否则视觉上不清晰; 生成器显式保证。

d) 放置合法性
- legend block 固定放在内角, 但与外框保持至少 1 格黑色间隙, 防止与边框连通。
- 主体通过 `random_free_location(..., border=2, padding=1)` 放置, 保证不越界、不碰 frame、也不碰 legend。
- legend block 自身用 0 间隔切开每块 2x2 色砖, 因而组件提取不会误并。

e) 视觉可观察性
- 解题者会看到: 边框控制色、与之颜色不同的引导边、角落里整齐的 2x2 对照砖, 以及输出中发生的统一替色。
- 多个样例里, 同一个 guide symbol 会在图例里稳定对应某个 control color; 动作砖的顶行则稳定用控制色对作键。
- 因而人类可以归纳出“先查 symbol->control 做对齐, 再查 control-pair->src/dst 做替色”的两阶段规则。

f) 可视化整洁度
- 图例区集中放在单个角落, 用两列堆叠的 2x2 砖表达, 不与主体混杂。
- 主体和 legend 都离外框留空, 边框功能、主体功能、图例功能三者分区清楚。
- 不在同一对象上叠加多层标记: 主体只承担 guide+pattern, 图例只承担解码和动作说明。

g) 颜色变化可推断性
- 颜色变化并非任意发生: 只有动作砖顶行匹配当前 frame control pair 时, 其底行 src->dst 才生效。
- 样例中会重复出现“同一 pair 对应同一替色规则”的结构线索, 不要求人类记住抽象数字, 只需看图例配对。
- 通过禁止 dst 预先出现在核心中, 输出里的新增颜色就是被替换出的结果, 结构信号足够强。

9) 新增隐式假设与生成保障
- 求解器假设每个 guide symbol 都能在解码砖中找到唯一映射; 生成器只放唯一 decoder entry。
- 求解器假设至少存在一个 D8 变换, 使两条解码后的 guide edge 分别对上 frame 的水平/垂直控制边; 生成器从“已对齐对象”反向施加非恒等 D8 变换得到输入。
- 求解器假设动作砖键使用 `(水平控制色, 垂直控制色)` 的顺序; 生成器按该顺序构造并在指标里核验唯一命中。
"""


SEMANTIC_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    h, w = arr.shape
    sides = {
        "top": arr[0, 1 : w - 1],
        "bottom": arr[h - 1, 1 : w - 1],
        "left": arr[1 : h - 1, 0],
        "right": arr[1 : h - 1, w - 1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, vals in sides.items():
        seq = list(map(int, vals))
        non_bg = [x for x in seq if x != int(bg)]
        if not non_bg:
            continue
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(seq) - 1:
            cand.append((side, int(col), int(count), int(len(seq))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _count, _len in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, vals in edges.items():
        seq = list(map(int, vals))
        non_fill = [x for x in seq if x != int(fill)]
        if not non_fill:
            continue
        col, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(seq) - 1:
            out[side] = int(col)
    return out


def _extract_legend_tiles(arr: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    tiles: list[np.ndarray] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if int(comp["area"]) != 4:
            continue
        patch = crop_bbox(arr, (r0, c0, r1, c1))
        if patch.shape == (2, 2):
            tiles.append(np.array(patch, dtype=int))
    return tiles


def _is_decoder_tile(tile: np.ndarray) -> bool:
    return bool(
        tile.shape == (2, 2)
        and int(tile[0, 0]) == int(tile[1, 0])
        and int(tile[0, 1]) == int(tile[1, 1])
        and int(tile[0, 0]) != int(tile[0, 1])
    )


def _decoder_map(tiles: list[np.ndarray]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for tile in tiles:
        if not _is_decoder_tile(tile):
            continue
        mapping[int(tile[0, 0])] = int(tile[0, 1])
    return mapping


def _action_map(tiles: list[np.ndarray]) -> dict[tuple[int, int], tuple[int, int]]:
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for tile in tiles:
        if _is_decoder_tile(tile):
            continue
        key = (int(tile[0, 0]), int(tile[0, 1]))
        value = (int(tile[1, 0]), int(tile[1, 1]))
        mapping[key] = value
    return mapping


def _control_key(ctrls: list[tuple[str, int]]) -> tuple[int, int] | None:
    horiz = [int(col) for side, col in ctrls if side in ("top", "bottom")]
    vert = [int(col) for side, col in ctrls if side in ("left", "right")]
    if not horiz or not vert:
        return None
    return int(horiz[0]), int(vert[0])


def _decode_color(color: int, decoder: dict[int, int]) -> int | None:
    if int(color) in decoder:
        return int(decoder[int(color)])
    return None


def _align_and_crop(obj: np.ndarray, desired: dict[str, int], decoder: dict[int, int]) -> np.ndarray:
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            raw_edges = _guide_edges(t)
            decoded_edges = {side: _decode_color(col, decoder) for side, col in raw_edges.items()}
            if all(side in decoded_edges and decoded_edges[side] == int(col) for side, col in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                return t[int(top) : int(bottom), int(left) : int(right)]
    return obj.copy()


def _replace_color(arr: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[out == int(src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Phase 1: decode guide symbols through legend tiles, then D8-align the object and crop guides.
    Phase 2: use the frame control-color pair to select an action tile and recolor one pattern color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}

    tiles = _extract_legend_tiles(g, bg)
    decoder = _decoder_map(tiles)
    actions = _action_map(tiles)

    obj = _extract_center_object(g, bg)
    core = _align_and_crop(obj, desired, decoder)

    ctrl_key = _control_key(ctrls)
    if ctrl_key is None or ctrl_key not in actions:
        return core

    src, dst = actions[ctrl_key]
    return _replace_color(core, src, dst)


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int], src_color: int) -> np.ndarray:
    area = h * w
    min_nonfill = max(6, area // 8)
    max_nonfill = max(min_nonfill + 2, int(area * 0.35))
    min_src = max(3, area // 14)

    for _ in range(150):
        core = np.full((h, w), int(fill), dtype=int)
        target_nonfill = random.randint(min_nonfill, max_nonfill)
        painted = 0

        for color in pattern_colors:
            r = random.randrange(h)
            c = random.randrange(w)
            if int(core[r, c]) == int(fill):
                painted += 1
            core[r, c] = int(color)

        while painted < target_nonfill:
            color = random.choice(pattern_colors)
            shape_kind = random.choice(["dot", "line", "rect"])
            if shape_kind == "dot":
                cells = [(random.randrange(h), random.randrange(w))]
            elif shape_kind == "line":
                line_len = random.randint(2, min(5, max(h, w)))
                if random.random() < 0.5:
                    rr = random.randrange(h)
                    c0 = random.randrange(max(1, w - line_len + 1))
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(max(1, h - line_len + 1))
                    cc = random.randrange(w)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            else:
                mh = random.randint(1, min(3, h))
                mw = random.randint(1, min(3, w))
                r0 = random.randrange(h - mh + 1)
                c0 = random.randrange(w - mw + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

            remaining = target_nonfill - painted
            fill_cells = [(r, c) for r, c in cells if int(core[r, c]) == int(fill)]
            if remaining <= 0:
                break
            if len(fill_cells) > remaining:
                fill_cells = random.sample(fill_cells, remaining)
            for r, c in fill_cells:
                core[r, c] = int(color)
            painted += len(fill_cells)

        marker_color = random.choice(pattern_colors)
        corner = random.choice(
            [
                (0, 0),
                (0, max(0, w - 2)),
                (max(0, h - 2), 0),
                (max(0, h - 2), max(0, w - 2)),
            ]
        )
        mr, mc = corner
        marker_cells = [(mr, mc), (min(h - 1, mr + 1), mc), (min(h - 1, mr + 1), min(w - 1, mc + 1))]
        for r, c in marker_cells:
            core[r, c] = int(marker_color)

        src_count = int(np.sum(core == int(src_color)))
        if src_count < min_src:
            fill_cells = [tuple(map(int, rc)) for rc in np.argwhere(core == int(fill))]
            random.shuffle(fill_cells)
            for r, c in fill_cells[: min_src - src_count]:
                core[r, c] = int(src_color)

        if _is_visibly_asymmetric(core):
            return core

    raise RuntimeError("failed to build asymmetric core")


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])

    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _is_visibly_asymmetric(arr: np.ndarray) -> bool:
    for k in range(4):
        for flip in (False, True):
            if k == 0 and not flip:
                continue
            t = _transform(arr, k, flip)
            if t.shape == arr.shape and np.array_equal(t, arr):
                return False
    return True


def _decoder_tile(symbol: int, control: int) -> np.ndarray:
    return np.array([[int(symbol), int(control)], [int(symbol), int(control)]], dtype=int)


def _action_tile(ctrl_h: int, ctrl_v: int, src: int, dst: int) -> np.ndarray:
    return np.array([[int(ctrl_h), int(ctrl_v)], [int(src), int(dst)]], dtype=int)


def _build_decoder_tiles(
    actual_entries: list[tuple[int, int]], palette: list[int], forbidden_symbols: set[int]
) -> list[np.ndarray]:
    tiles = [_decoder_tile(symbol, ctrl) for symbol, ctrl in actual_entries]
    available_symbols = [c for c in palette if c not in forbidden_symbols]
    available_ctrls = [c for c in palette if c not in {ctrl for _symbol, ctrl in actual_entries}]
    if available_symbols and available_ctrls:
        extra_symbol = random.choice(available_symbols)
        extra_ctrl_choices = [c for c in available_ctrls if c != int(extra_symbol)]
        if extra_ctrl_choices:
            extra_ctrl = random.choice(extra_ctrl_choices)
            tiles.append(_decoder_tile(extra_symbol, extra_ctrl))
    random.shuffle(tiles)
    return tiles


def _build_action_tiles(
    actual_tile: tuple[int, int, int, int], palette: list[int], n_tiles: int
) -> list[np.ndarray]:
    ctrl_h, ctrl_v, src, dst = map(int, actual_tile)
    used_keys = {(ctrl_h, ctrl_v)}
    rows = [_action_tile(ctrl_h, ctrl_v, src, dst)]
    while len(rows) < int(n_tiles):
        h_ctrl = random.choice(palette)
        v_ctrl = random.choice(palette)
        if h_ctrl == v_ctrl:
            continue
        key = (int(h_ctrl), int(v_ctrl))
        if key in used_keys:
            continue
        src_color = random.choice(palette)
        dst_choices = [c for c in palette if c != int(src_color)]
        if not dst_choices:
            continue
        dst_color = random.choice(dst_choices)
        if int(h_ctrl) == int(src_color) and int(v_ctrl) == int(dst_color):
            continue
        rows.append(_action_tile(h_ctrl, v_ctrl, src_color, dst_color))
        used_keys.add(key)
    random.shuffle(rows)
    return rows


def _legend_block(decoder_tiles: list[np.ndarray], action_tiles: list[np.ndarray]) -> np.ndarray:
    left_h = 3 * len(decoder_tiles) - 1
    right_h = 3 * len(action_tiles) - 1
    h = max(left_h, right_h)
    w = 5
    block = np.zeros((h, w), dtype=int)
    for i, tile in enumerate(decoder_tiles):
        r0 = 3 * i
        block[r0 : r0 + 2, 0:2] = tile
    for i, tile in enumerate(action_tiles):
        r0 = 3 * i
        block[r0 : r0 + 2, 3:5] = tile
    return block


def _place_legend_in_corner(grid: np.ndarray, legend_block: np.ndarray) -> bool:
    n = grid.shape[0]
    h, w = legend_block.shape
    corners = [
        (2, 2),
        (2, n - 2 - w),
        (n - 2 - h, 2),
        (n - 2 - h, n - 2 - w),
    ]
    random.shuffle(corners)
    for r0, c0 in corners:
        if r0 < 0 or c0 < 0:
            continue
        if r0 + h > n or c0 + w > n:
            continue
        if np.any(grid[r0 : r0 + h, c0 : c0 + w] != 0):
            continue
        paste(grid, legend_block, r0, c0, transparent=0)
        return True
    return False


def _non_identity_d8() -> tuple[int, bool]:
    choices = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
    return random.choice(choices)


def _matching_transform_counts(
    obj: np.ndarray, desired: dict[str, int], decoder: dict[int, int]
) -> tuple[int, int]:
    direct = 0
    decoded = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            raw_edges = _guide_edges(t)
            if all(side in raw_edges and int(raw_edges[side]) == int(col) for side, col in desired.items()):
                direct += 1
            decoded_edges = {side: _decode_color(col, decoder) for side, col in raw_edges.items()}
            if all(side in decoded_edges and decoded_edges[side] == int(col) for side, col in desired.items()):
                decoded += 1
    return int(direct), int(decoded)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(30, 36)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    roles = random.sample(SEMANTIC_COLORS, 7)
    ctrl_h, ctrl_v, sym_h, sym_v, fill, src, dst = roles
    if sym_h in {ctrl_h, ctrl_v} or sym_v in {ctrl_h, ctrl_v} or dst == src:
        return None

    pattern_pool = [c for c in SEMANTIC_COLORS if c not in {ctrl_h, ctrl_v, sym_h, sym_v, fill, dst}]
    if src not in pattern_pool:
        pattern_pool.append(src)
    pattern_pool = list(dict.fromkeys(pattern_pool))
    if len(pattern_pool) < 2:
        return None
    pattern_colors = [int(src), int(random.choice([c for c in pattern_pool if c != int(src)]))]

    max_core_h = min(17, n - 14)
    max_core_w = min(17, n - 14)
    if max_core_h < 6 or max_core_w < 6:
        return None
    core_h = random.randint(6, max_core_h)
    core_w = random.randint(6, max_core_w)
    core_before = _build_core(core_h, core_w, fill, pattern_colors, src)
    if int(np.sum(core_before == int(src))) < 3:
        return None
    if int(np.sum(core_before == int(dst))) > 0:
        return None

    output = _replace_color(core_before, src, dst)
    if np.array_equal(output, core_before):
        return None

    decoder_tiles = _build_decoder_tiles(
        [(sym_h, ctrl_h), (sym_v, ctrl_v)],
        SEMANTIC_COLORS,
        forbidden_symbols={ctrl_h, ctrl_v, sym_h, sym_v, fill, src, dst},
    )
    action_tiles = _build_action_tiles((ctrl_h, ctrl_v, src, dst), SEMANTIC_COLORS, random.randint(3, 4))
    legend_block = _legend_block(decoder_tiles, action_tiles)

    guide_colors = {horiz_side: sym_h, vert_side: sym_v}
    aligned_object = _attach_guides(core_before, (horiz_side, vert_side), guide_colors)
    k, flip = _non_identity_d8()
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)
    if not _place_legend_in_corner(grid, legend_block):
        return None

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=2, padding=1)
        paste(grid, input_object, r0, c0, transparent=0)
    except ValueError:
        return None

    metrics = variant_metrics(grid)
    if metrics["decoder_tiles"] < 2:
        return None
    if metrics["action_tiles"] < 3:
        return None
    if metrics["matched_key_count"] != 1:
        return None
    if metrics["changed_cells"] < 3:
        return None
    if metrics["direct_match_count"] != 0:
        return None
    if metrics["decoded_match_count"] != 1:
        return None
    if metrics["symbolic_guides"] != 2:
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    ctrl_key = _control_key(ctrls)

    tiles = _extract_legend_tiles(g, bg)
    decoder = _decoder_map(tiles)
    actions = _action_map(tiles)
    obj = _extract_center_object(g, bg)
    phase1 = _align_and_crop(obj, desired, decoder)

    matched_key_count = 0
    changed_cells = 0
    recolored_cells = 0
    if ctrl_key is not None and ctrl_key in actions:
        matched = [key for key in actions if key == ctrl_key]
        matched_key_count = len(matched)
        src, dst = actions[ctrl_key]
        recolored_cells = int(np.sum(phase1 == int(src)))
        changed_cells = recolored_cells if int(src) != int(dst) else 0

    raw_edges = _guide_edges(obj)
    symbolic_guides = 0
    for color in raw_edges.values():
        decoded = _decode_color(int(color), decoder)
        if decoded is not None and int(decoded) != int(color):
            symbolic_guides += 1

    direct_match_count, decoded_match_count = _matching_transform_counts(obj, desired, decoder)
    phase1_direct_only = _align_and_crop(obj, desired, {})

    return {
        "decoder_tiles": int(sum(1 for tile in tiles if _is_decoder_tile(tile))),
        "action_tiles": int(sum(1 for tile in tiles if not _is_decoder_tile(tile))),
        "matched_key_count": int(matched_key_count),
        "recolored_cells": int(recolored_cells),
        "changed_cells": int(changed_cells),
        "symbolic_guides": int(symbolic_guides),
        "direct_match_count": int(direct_match_count),
        "decoded_match_count": int(decoded_match_count),
        "phase1_differs_without_decode": int(not np.array_equal(phase1, phase1_direct_only)),
    }
