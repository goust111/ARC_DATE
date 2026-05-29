# Combo variant of dbff022c (index 4)
# combo_spec: D2/2+D12/1
# Name: 嵌套方框编码调色板 + 三行调色板

# Design analysis
# ----------------
# fused_variant:
#   preserved_core:
#     - 背景仍是 0；主体仍是若干单色空心 sprite；核心任务仍是根据边缘图例，把封闭孔洞改成图例指定的样子。
#     - 仍沿用 output-first 生成：先生成带正确孔洞纹理的输出，再把所有被处理的孔洞清空成输入，最后用 main() 回放自检。
#   fusion_rule:
#     - 原题图例是 2xN / Nx2 色带，只编码 `border -> fill_color`。
#     - D2/2 把每个图例条目改成“嵌套方框”局部符号；D12/1 再要求第三层提供额外编码。
#     - 融合后每个条目固定为 5x5 三层嵌套符号：
#         1. 最外层 1 格厚外框 = sprite 边框色 border
#         2. 中间 3x3 区域的环 = 目标填充色 fill
#         3. 最中心 1 格 = 模式码 mode_token
#       其中 mode_token 不再给出新颜色表，而是直接复用可见层关系：
#         - center == fill   -> 实心填充 solid
#         - center == bg(0)  -> 竖条纹填充 stripe
#         - center == border -> 棋盘格填充 checker
#   behavior_trace:
#     - 原逻辑例子：顶边色带 `[2,4,6] / [7,8,9]` 只会得到 `2->7, 4->8, 6->9`，孔洞整块涂满。
#     - 融合后例子：顶边放三个 5x5 条目。
#         * 条目 A: 外框 2，中环 7，中心 7  => `2 -> (7, solid)`
#         * 条目 B: 外框 4，中环 8，中心 0  => `4 -> (8, stripe)`
#         * 条目 C: 外框 6，中环 9，中心 6  => `6 -> (9, checker)`
#       若一个空心 sprite 的边框是 4，则其内部不再整块变成 8，而是按孔洞 bbox 锚定的竖条纹填 8；
#       若边框是 6，则内部改成 9/0 交错的棋盘格。
#     - 差异点不再只是“换颜色”，而是要同时解读“边框色决定哪一条目”“中间层给出填充色”“最内层决定纹理模式”。
#   complexity_calibration:
#     - 图例条目数固定为 3，正好对应三种模式；这避免模式种类过多，也保证单个样例里经常能同时看到多种模式。
#     - 中心码只允许三档：同中环 / 同背景 / 同外框。这样模式空间小而离散，肉眼可见且在 2-4 个样例里有重复支撑。
#     - 生成器要求至少 2 个被实际使用的 palette key，且这些 key 的 mode 至少两种并至少包含一种非实心，
#       避免退化成“仍然只是 border->solid fill”的原题。
#   implicit_assumption_review:
#     - 原 `_best_palette()` 假设图例是连续 2 行或 2 列色带；融合后完全失效，必须改成逐个识别 5x5 嵌套符号。
#     - 原 `_palette_mapping()` 只返回 `dict[border] = fill`；融合后必须升级为 `dict[border] = (fill, mode)`。
#     - 原填洞逻辑默认“若 border 命中映射，则整块 region 全部写成 fill”；融合后这个假设不成立，
#       必须按 mode 对 region 内像素做选择性着色，未选中的洞内像素保留背景。
#     - 新 solver 还新增三条输入前提：
#         1. 图例条目是精确的 5x5 三层嵌套；
#         2. 条目只出现在一条外边缘上，彼此至少隔 1 格背景；
#         3. 中心码只会取 `fill / bg / border` 三种之一。
#       generator 会显式构造并在末端自检这些前提。
#   placement_legality:
#     - 5x5 图例比原题大，因此网格尺寸扩大到 17..21，保证三条目 palette 与 sprite 能同时放下。
#     - 图例整体 bbox 连同外侧 1 格缓冲写入 occ，sprite 同样要求与 palette 和彼此之间至少隔 1 格背景。
#     - sprite 仍禁止贴到网格外边界，避免把边缘 palette 与主体混在一起，也保持 `_bg_boundary()` 稳定识别 0。
#   visual_observability:
#     - 解题者能直接看到：边缘有三组“外框/中环/中心码”符号；输出里，匹配到外框色的 sprite 内部
#       分别出现纯色、竖条纹或棋盘格。
#     - 模式码不依赖颜色 id 排序，而依赖“中心与哪一层同色”的可见关系；这是局部结构线索，不是隐藏数值规则。
#   visual_tidy:
#     - 图例只占一条边，主体 sprite 留在内部，二者之间保留背景缝隙；hole 纹理只发生在 sprite 孔洞内部，
#       不会和 palette 重叠，也不会在同一区域叠加多套标记。
#     - 条纹与棋盘格只使用 `fill + bg` 两种像素，不额外引入第四层装饰色，画面仍然规整。
#   inference_support:
#     - 每个样例都至少包含一个 patterned sprite（条纹或棋盘格），大多数样例包含两种不同 mode；
#       palette 里还可能保留 1 个未使用条目，让解题者看到“图例是完整编码表，而非仅对应现有对象”。
#     - 模式锚定在孔洞 bbox 的左上角，保证同一 mode 在不同 sprite 上的纹理相位一致，便于跨例归纳。
#   post_edit_assumptions_to_audit:
#     - `_edge_entries()` 假设同一边上只有一串合法 palette；generator 只放一串。
#     - `_apply_pattern()` 假设 region 至少有一个点且模式在三档中；generator 只产出含封闭 hole 的 sprite，
#       palette 也只用三档 mode。
#     - `_fill_enclosed_bg_regions()` 依赖“孔洞邻接颜色中只有一个主导 border 属于 mapping”；generator 继续保证
#       每个 hole 来自单色边框 sprite，因此这一前提保持成立。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, neighbors4


_MODE_SOLID = "solid"
_MODE_STRIPE = "stripe"
_MODE_CHECKER = "checker"
_MODES = (_MODE_SOLID, _MODE_STRIPE, _MODE_CHECKER)


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _decode_center_mode(center: int, border: int, fill: int, bg: int) -> str | None:
    """Decode the innermost 1x1 token into a fill mode."""
    if center == fill:
        return _MODE_SOLID
    if center == bg:
        return _MODE_STRIPE
    if center == border:
        return _MODE_CHECKER
    return None


def _nested_triplet(sub: np.ndarray, bg: int) -> tuple[int, int, str] | None:
    """Return (border, fill, mode) if sub is a valid 5x5 three-layer legend entry."""
    if sub.shape != (5, 5):
        return None

    border = int(sub[0, 0])
    fill = int(sub[1, 1])
    if border == bg or fill == bg or fill == border:
        return None

    for r in range(5):
        for c in range(5):
            v = int(sub[r, c])
            if r in (0, 4) or c in (0, 4):
                if v != border:
                    return None
            elif (r, c) != (2, 2):
                if v != fill:
                    return None

    mode = _decode_center_mode(int(sub[2, 2]), border, fill, bg)
    if mode is None:
        return None
    return border, fill, mode


def _entry_mask(shape: tuple[int, int], side: str, start: int) -> np.ndarray:
    """Boolean mask for one 5x5 legend entry on a given edge."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0:5, start : start + 5] = True
    elif side == "bottom":
        mask[h - 5 : h, start : start + 5] = True
    elif side == "left":
        mask[start : start + 5, 0:5] = True
    else:
        mask[start : start + 5, w - 5 : w] = True
    return mask


def _mask_is_isolated(g: np.ndarray, bg: int, mask: np.ndarray) -> bool:
    """
    Non-bg palette cells should not be 4-neigh adjacent to same-colored cells outside the entry.
    """
    h, w = g.shape
    for r, c in zip(*np.where(mask), strict=False):
        r = int(r)
        c = int(c)
        v = int(g[r, c])
        if v == bg:
            continue
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                if int(g[rr, cc]) == v:
                    return False
    return True


def _edge_entries(
    g: np.ndarray, bg: int, side: str
) -> list[tuple[int, int, int, str, np.ndarray]]:
    """
    Find isolated 5x5 three-layer legend entries on one edge.

    Returns (start, border, fill, mode, mask), ordered along the edge axis.
    """
    h, w = g.shape
    if side in ("top", "bottom"):
        if h < 5:
            return []
        limit = w - 4
    else:
        if w < 5:
            return []
        limit = h - 4

    candidates: list[tuple[int, int, int, str, np.ndarray]] = []
    for start in range(limit):
        if side == "top":
            sub = g[0:5, start : start + 5]
        elif side == "bottom":
            sub = g[h - 5 : h, start : start + 5]
        elif side == "left":
            sub = g[start : start + 5, 0:5]
        else:
            sub = g[start : start + 5, w - 5 : w]

        triplet = _nested_triplet(sub, bg)
        if triplet is None:
            continue

        mask = _entry_mask((h, w), side, int(start))
        if not _mask_is_isolated(g, bg, mask):
            continue
        border, fill, mode = triplet
        candidates.append((int(start), int(border), int(fill), mode, mask))

    chosen: list[tuple[int, int, int, str, np.ndarray]] = []
    last_block = -99
    for start, border, fill, mode, mask in candidates:
        if start <= last_block:
            continue
        chosen.append((start, border, fill, mode, mask))
        last_block = start + 5
    return chosen


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, str]], np.ndarray]:
    """
    Decode the best edge legend made of 5x5 three-layer nested boxes.

    Returns (mapping border->(fill, mode), palette_mask).
    """
    h, w = g.shape
    best = None  # (score, side, entries)
    for side in ("top", "bottom", "left", "right"):
        entries = _edge_entries(g, bg, side)
        if len(entries) < 2:
            continue

        borders = [border for _, border, _, _, _ in entries]
        if len(borders) != len(set(borders)):
            continue

        fills = [fill for _, _, fill, _, _ in entries]
        modes = [mode for _, _, _, mode, _ in entries]
        starts = [start for start, *_ in entries]
        span = starts[-1] - starts[0] + 5
        score = (len(entries), len(set(modes)), len(set(fills)), -span)
        cand = (score, side, entries)
        if best is None or score > best[0]:
            best = cand

    assert best is not None, "_palette_mapping must find a valid three-layer palette"
    _, _, entries = best

    mapping: dict[int, tuple[int, str]] = {}
    pal_mask = np.zeros((h, w), dtype=bool)
    for _, border, fill, mode, mask in entries:
        mapping[int(border)] = (int(fill), mode)
        pal_mask |= mask
    return mapping, pal_mask


def _apply_pattern(
    out: np.ndarray,
    pts: np.ndarray,
    fill: int,
    mode: str,
) -> None:
    """Fill one enclosed hole region according to the decoded mode."""
    r_anchor = int(np.min(pts[:, 0]))
    c_anchor = int(np.min(pts[:, 1]))
    for r, c in pts:
        r = int(r)
        c = int(c)
        place = False
        if mode == _MODE_SOLID:
            place = True
        elif mode == _MODE_STRIPE:
            place = ((c - c_anchor) % 2) == 0
        elif mode == _MODE_CHECKER:
            place = ((r - r_anchor + c - c_anchor) % 2) == 0
        else:
            raise AssertionError(f"unknown mode: {mode}")
        if place:
            out[r, c] = int(fill)


def _fill_enclosed_bg_regions(
    out: np.ndarray,
    bg: int,
    mapping: dict[int, tuple[int, str]],
    pal_mask: np.ndarray,
) -> None:
    """
    Find non-boundary background regions outside the palette and fill them by decoded pattern.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)
    for pts in comps:
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w:
                    if pal_mask[rr, cc]:
                        continue
                    v = int(out[rr, cc])
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in mapping:
                continue
            if cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        fill, mode = mapping[border]
        _apply_pattern(out, np.asarray(pts, dtype=int), int(fill), mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites' interiors using three-layer nested-box legend symbols on the grid edge.

    Input structure:
    - bg is the dominant border color of the grid; several hollow sprites sit inside the board and each hole is initially cleared to bg.
    - Exactly one outer edge contains a legend made of separated 5x5 nested boxes. In each entry the outer ring gives a sprite border color, the middle ring gives the fill color, and the center cell encodes whether the hole should become solid, striped, or checkerboard.

    Rule:
    - Locate the edge legend, decode every nested box into `border -> (fill_color, fill_mode)`, and ignore sprites whose border color is absent from that mapping.
    - Find enclosed bg regions outside the palette area that do not touch the outer grid boundary.
    - For each such region, determine the dominant adjacent sprite-border color; if it appears in the legend, repaint the region with the mapped fill color using the mapped pattern mode.

    Key insight: The legend is no longer a simple color strip. Each local 5x5 symbol simultaneously encodes three facts: which border color to match, which color to use inside the hole, and how that interior should be textured.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 17, 21
_GRID_W_MIN, _GRID_W_MAX = 17, 21
_TINY_H_THRESHOLD = 18

# Palette
_PALETTE_N = 3
_LEGEND_SIZE = 5
_LEGEND_GAP = 1

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITES_RANGE = (2, 3)

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
_MAX_RETRIES = 400
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point-arrays for bg components not touching the sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped boolean mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole."""
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n - 1)
        sprite[br, s : s + n] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n - 1)
        sprite[s : s + n, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for cr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _legend_entry(border: int, fill: int, mode: str) -> np.ndarray:
    """Construct one 5x5 three-layer legend entry."""
    entry = np.full((_LEGEND_SIZE, _LEGEND_SIZE), int(border), dtype=int)
    entry[1:4, 1:4] = int(fill)
    if mode == _MODE_SOLID:
        entry[2, 2] = int(fill)
    elif mode == _MODE_STRIPE:
        entry[2, 2] = 0
    elif mode == _MODE_CHECKER:
        entry[2, 2] = int(border)
    else:
        raise AssertionError(f"unknown mode: {mode}")
    return entry


def _palette_span(n: int) -> int:
    """Bounding-box span along the arrangement axis for n legend entries."""
    return n * _LEGEND_SIZE + (n - 1) * _LEGEND_GAP


def _choose_colors(
    rng: random.Random,
) -> tuple[list[int], list[int], int]:
    """Pick distinct border keys, fill values, and a distractor border color."""
    colors = rng.sample(list(range(1, 10)), 7)
    border_keys = colors[:_PALETTE_N]
    fill_values = colors[_PALETTE_N : 2 * _PALETTE_N]
    distractor = colors[-1]
    return border_keys, fill_values, distractor


def _choose_mapping(
    rng: random.Random,
) -> tuple[list[int], list[int], dict[int, tuple[int, str]]]:
    """Assign each border key a distinct fill color and one of the three modes."""
    border_keys, fill_values, _ = _choose_colors(rng)
    modes = list(_MODES)
    rng.shuffle(modes)
    mapping = {
        int(border): (int(fill), str(mode))
        for border, fill, mode in zip(border_keys, fill_values, modes, strict=False)
    }
    return border_keys, fill_values, mapping


def _build_palette(side: str, entries: list[tuple[int, int, str]]) -> np.ndarray:
    """Construct the edge legend as a row/column of separated 5x5 nested boxes."""
    n = len(entries)
    span = _palette_span(n)
    if side in ("top", "bottom"):
        pal = np.zeros((_LEGEND_SIZE, span), dtype=int)
        for i, (border, fill, mode) in enumerate(entries):
            c0 = i * (_LEGEND_SIZE + _LEGEND_GAP)
            pal[:, c0 : c0 + _LEGEND_SIZE] = _legend_entry(border, fill, mode)
    else:
        pal = np.zeros((span, _LEGEND_SIZE), dtype=int)
        for i, (border, fill, mode) in enumerate(entries):
            r0 = i * (_LEGEND_SIZE + _LEGEND_GAP)
            pal[r0 : r0 + _LEGEND_SIZE, :] = _legend_entry(border, fill, mode)
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    """Place the palette on the chosen edge and reserve its bbox plus 1-cell padding."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, h - pal_h), w - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[
        max(0, pr0 - 1) : min(h, pr0 + pal_h + 1),
        max(0, pc0 - 1) : min(w, pc0 + pal_w + 1),
    ] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements where the sprite stays inside the border and away from occupied cells."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if r0 <= 0 or c0 <= 0 or (r0 + sh) >= h or (c0 + sw) >= w:
                continue
            if occ[
                max(0, r0 - 1) : min(h, r0 + sh + 1),
                max(0, c0 - 1) : min(w, c0 + sw + 1),
            ].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_patterned_holes(
    grid: np.ndarray,
    hole_points: list[np.ndarray],
    fill: int | None,
    mode: str | None,
    r0: int,
    c0: int,
) -> None:
    """Paint hole interiors for the output grid."""
    if fill is None or mode is None:
        return
    for hpts in hole_points:
        abs_pts = np.asarray([(r0 + int(hr), c0 + int(hc)) for hr, hc in hpts], dtype=int)
        _apply_pattern(grid, abs_pts, int(fill), mode)


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    mode: str | None,
    holes: list[np.ndarray],
) -> None:
    """Render one sprite and its patterned holes."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
    _paint_patterned_holes(grid, holes, fill, mode, r0, c0)


def _candidate_filled_key_sets(
    rng: random.Random, mapping: dict[int, tuple[int, str]]
) -> list[list[int]]:
    """Enumerate 2- or 3-key subsets with at least two distinct modes and one non-solid mode."""
    keys = list(mapping)
    candidates: list[list[int]] = []
    for size in range(_FILLED_SPRITES_RANGE[0], _FILLED_SPRITES_RANGE[1] + 1):
        if size > len(keys):
            continue
        for trial in range(60):
            chosen = rng.sample(keys, size)
            modes = {mapping[k][1] for k in chosen}
            if len(modes) < 2:
                continue
            if modes == {_MODE_SOLID}:
                continue
            candidates.append(chosen)
    assert candidates, "_candidate_filled_key_sets must find a non-degenerate mode set"
    return candidates


def _hole_supports_mode(holes: list[np.ndarray], mode: str) -> bool:
    """Check whether at least one hole in the shape can visibly express the mode."""
    if mode == _MODE_SOLID:
        return True
    for pts in holes:
        pts = np.asarray(pts, dtype=int)
        if mode == _MODE_CHECKER and len(pts) >= 2:
            return True
        if mode == _MODE_STRIPE and len(pts) >= 2:
            cols = {int(c) for _, c in pts}
            if len(cols) >= 2:
                return True
    return False


def _assign_filled_shapes(
    rng: random.Random,
    mapping: dict[int, tuple[int, str]],
    shapes: list[np.ndarray],
    holes_list: list[list[np.ndarray]],
) -> list[tuple[int, int]] | None:
    """
    Assign palette keys to compatible shapes and return (shape_index, border_key) pairs.
    """
    candidates = _candidate_filled_key_sets(rng, mapping)
    rng.shuffle(candidates)

    for chosen_keys in candidates:
        ordered_keys = sorted(chosen_keys, key=lambda k: mapping[k][1] == _MODE_SOLID)
        used_shapes: set[int] = set()
        pairs: list[tuple[int, int]] = []

        def dfs(i: int) -> bool:
            if i == len(ordered_keys):
                return True
            key = ordered_keys[i]
            mode = mapping[key][1]
            shape_indices = list(range(len(shapes)))
            rng.shuffle(shape_indices)
            for shape_idx in shape_indices:
                if shape_idx in used_shapes:
                    continue
                if not _hole_supports_mode(holes_list[shape_idx], mode):
                    continue
                used_shapes.add(shape_idx)
                pairs.append((shape_idx, key))
                if dfs(i + 1):
                    return True
                pairs.pop()
                used_shapes.remove(shape_idx)
            return False

        if not dfs(0):
            continue

        if sum(idx not in used_shapes for idx in range(len(shapes))) != 1:
            continue
        return pairs
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_sh = min(_SPRITE_H_MAX, h - 2)
    max_sw = min(_SPRITE_W_MAX, w - 2)
    if h <= _TINY_H_THRESHOLD:
        max_sh = min(max_sh, 5)
    if max_sh < 3 or max_sw < 3:
        return None

    border_keys, fill_values, mapping = _choose_mapping(rng)
    distractor_candidates = [c for c in range(1, 10) if c not in set(border_keys) | set(fill_values)]
    if not distractor_candidates:
        return None
    distractor_color = rng.choice(distractor_candidates)

    num_filled = rng.randint(_FILLED_SPRITES_RANGE[0], _FILLED_SPRITES_RANGE[1])
    total_sprites = num_filled + 1

    shapes: list[np.ndarray] = []
    holes_list: list[list[np.ndarray]] = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)
        holes_list.append(_get_enclosed_holes(mask))

    assignment = _assign_filled_shapes(rng, mapping, shapes, holes_list)
    if assignment is None:
        return None
    filled_pairs = assignment
    filled_keys = [key for _, key in filled_pairs]
    key_by_shape = {shape_idx: key for shape_idx, key in filled_pairs}

    palette_entries = [(border, mapping[border][0], mapping[border][1]) for border in border_keys]
    pal = _build_palette(side, palette_entries)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i in key_by_shape:
            border = key_by_shape[i]
            fill, mode = mapping[border]
        else:
            border, fill, mode = distractor_color, None, None
        infos.append((mask, border, fill, mode))

    placed = []
    for idx, (mask, color, fill, mode) in enumerate(infos):
        holes = holes_list[idx]
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, mode, holes)
        placed.append((r0, c0, mask, holes, fill, mode))
        sh, sw = mask.shape
        occ[
            max(0, r0 - 1) : min(h, r0 + sh + 1),
            max(0, c0 - 1) : min(w, c0 + sw + 1),
        ] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, _mask, holes, fill, _mode in placed:
        if fill is None:
            continue
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    # Reject degeneration back to plain solid fills.
    patterned_used = sum(mapping[k][1] != _MODE_SOLID for k in filled_keys)
    if patterned_used == 0:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        r = _gen_attempt()
        if r is not None:
            return r
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
