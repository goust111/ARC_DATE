# Combo variant of dbff022c (index 5)
# combo_spec: D3/2+D6/0
# Name: 反直觉阅读方向 + 裁剪移除调色板

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 原题: 顶边 palette 若是 2xN, 默认按 "上->下" 读, 即 row0=border, row1=fill; 解完后输出仍保留这 2 行。
#    - 本融合: 同样的顶边 palette 必须按 "下->上" 读, 即 row1=border, row0=fill; 左边同理改成 col1=border, col0=fill。
#      在底边/右边也做完全反转: bottom 改成 row(h-2)=border, row(h-1)=fill; right 改成 col(w-2)=border, col(w-1)=fill。
#    - 然后不是返回完整网格, 而是把承载 palette 的整条 2-cell 边带裁掉。例: top palette 先在完整网格上填洞, 最终返回 out[2:, :]。
#    - 差异因此是协同的: 先用反直觉方向解码颜色, 再意识到 palette 只是输入指令而不属于输出。
# b) 复杂度校准
#    - 仍保留原题的单一步颜色映射 + 空心精灵填洞主干, 只新增两个明确分支: "反向读取图例" 与 "裁掉图例边带"。
#    - 为了让 D3/2 肉眼可见而不退化, palette 的 border 集与 fill 集强制互不相交。这样若按旧方向读取, 映射键集合会错位,
#      已出现的精灵边框色通常根本查不到, 输出会明显不同。
#    - palette 长度压到 2..4, 颜色种类仍少, 但每个样例至少有 2 个真的被使用的 key, 便于人类从少量例子中重建反向规则。
# c) 隐式假设审查
#    - _bg_boundary 仍假设完整输入外边界大部分是 bg。生成器把精灵全部放在裁剪后画布内部, 外边界只留 bg 与 palette。
#    - _best_palette 仍假设 edge 上存在唯一可信的 2xN/Nx2 图例。生成器确保没有别的非 bg 物体接触完整输入外边界。
#    - 原 _palette_mapping 隐含 "outer line = border keys"; 本融合中该前提失效, 必须按 side 显式交换 inner/outer 的角色。
#    - _fill_enclosed_bg_regions 仍在裁剪前运行, 所以洞的封闭性必须相对于完整输入成立; 生成器把所有精灵与完整输入边界隔开。
#    - 新增 _crop_palette_strip 假设 palette 总位于可整条移除的 2-cell 边带内; 生成器严格按该带构造 palette。
# d) 放置合法性
#    - 先在裁剪后画布上放置精灵, 再嵌回完整输入, 可天然避免 sprite 进入待裁 strip。
#    - sprite 与 sprite、sprite 与 palette 均保留至少 1 格 bg 间隔, 且 sprite 离裁剪后画布边界至少 1 格。
#    - palette 只占据某一侧的 2-row/2-col strip 中的一段, 其 bbox 加 1 格缓冲写入 occ, 防止与主体混淆。
# e) 视觉可观察性
#    - 每个样例都同时体现两点: 输出某一维缩短 2 格; 以及输入里真正出现的边框色来自 palette 的内侧线而不是贴边外侧线。
#    - 至少两个精灵会被填色, 且旧方向读取通常会导致 "完全不填" 或 "填错色", 因而反向读法在 2-3 组样例里可归纳。
# f) 可视化整洁度
#    - palette 独占被裁边带, sprite 全在保留区域, 没有叠层竞争。
#    - 仍只用规则的单色空心精灵, 背景缓冲充足, 人眼能稳定区分图例、主体和未匹配 distractor。
# g) 变化可推断性
#    - 颜色变化不是任意重排: 精灵内部仍由围绕它的边框色决定, 只是图例中 "哪一条线才是边框键" 被反转了。
#    - 生成器拒绝 `legacy(old-direction)+crop` 与 combo 输出相同的样本, 用量化 guard 保证 D3/2 真正起作用。
# 新主函数引入的额外假设
#    - 四个 side 的两条 palette 线都必须各自拥有唯一非 bg 颜色序列; 生成器用互不相交且各自 distinct 的颜色集保证。
#    - 由于裁剪后输出不含 palette, 所有有效 sprite 都必须完全落在保留画布中; 生成器按 cropped canvas 放置保证这一点。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top", "bottom", "left", "right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """
        Palette cells (excluding bg) must stay isolated from same-colored non-palette cells.
        """
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[0, :]
        inner = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(keys)), len(union), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[h - 1, :]
        inner = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(keys)), len(union), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, 0]
        inner = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(keys)), len(union), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, w - 1]
        inner = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(keys)), len(union), -(r1 - r0), nonbg)
                cand = (score, "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _combo_palette_mapping(
    g: np.ndarray, bg: int
) -> tuple[dict[int, int], np.ndarray, tuple[str, int, int]]:
    """
    Return (mapping border->fill, palette_mask, palette_info) using the reversed read direction.
    """
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[1, start:end]
        fill_line = g[0, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 2, start:end]
        fill_line = g[h - 1, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 1]
        fill_line = g[start:end, 0]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 2]
        fill_line = g[start:end, w - 1]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f

    return mapping, pal_mask, (side, start, end)


def _legacy_palette_mapping(
    g: np.ndarray, bg: int
) -> tuple[dict[int, int], np.ndarray, tuple[str, int, int]]:
    """Original-direction mapping, used only for anti-degeneration checks."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f

    return mapping, pal_mask, (side, start, end)


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Fill non-boundary background holes outside the palette using the dominant mapped border color.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
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

        fill = int(mapping[border])
        for r, c in pts:
            out[int(r), int(c)] = fill


def _crop_palette_strip(out: np.ndarray, side: str) -> np.ndarray:
    """Remove the full 2-cell edge strip that contains the palette legend."""
    if side == "top":
        return out[2:, :].copy()
    if side == "bottom":
        return out[:-2, :].copy()
    if side == "left":
        return out[:, 2:].copy()
    return out[:, :-2].copy()


def _solve_with_mapping(
    g: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
    side: str,
) -> np.ndarray:
    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return _crop_palette_strip(out, side)


def _legacy_output(grid: np.ndarray) -> np.ndarray:
    """Reference output under the original read direction, used for generator guards."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask, (side, _start, _end) = _legacy_palette_mapping(g, bg)
    return _solve_with_mapping(g, bg, mapping, pal_mask, side)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int | bool | tuple[int, int]]:
    """Small quantitative probe used after testing to verify the fused behavior is active."""
    g = np.asarray(grid, dtype=int)
    combo = main(g)
    legacy = _legacy_output(g)
    bg = _bg_boundary(g)
    side, _start, _end = _best_palette(g, bg)
    cropped_input = _crop_palette_strip(g, side)
    delta = (int(g.shape[0] - combo.shape[0]), int(g.shape[1] - combo.shape[1]))
    return {
        "shape_delta": delta,
        "legacy_equal": bool(np.array_equal(combo, legacy)),
        "legacy_diff_cells": int(np.count_nonzero(combo != legacy)),
        "filled_cells": int(np.count_nonzero(combo != cropped_input)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites from a reversed edge legend, then remove that legend strip.

    Input structure:
    - The outer border is mostly background, while one side contains a compact 2xN or Nx2 palette inside the removable edge band.
    - Hollow single-color sprites live away from that band; some of their border colors are encoded by the legend and one sprite may be a distractor.
    - Unlike the original family, the border-key colors are on the inward line of the palette and the fill colors are on the edge-touching line.

    Rule:
    - Detect the palette side, decode each pair using the opposite reading direction for that side, and build a border-color to fill-color mapping from the inward line to the outward line.
    - Fill every enclosed background hole whose dominant surrounding border color appears in that mapping.
    - After filling on the full grid, crop away the entire 2-row or 2-column side strip that hosted the palette.

    Key insight: The palette is both misleading and disposable. Its useful color keys sit on the inner line rather than the edge line, and once those reversed instructions are used, the whole palette strip disappears from the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask, (side, _start, _end) = _combo_palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"
    return _solve_with_mapping(g, bg, mapping, pal_mask, side)


# --------------- generator (combo D3/2 + D6/0) ---------------

_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 12, 15

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MIN = 2
_FILLED_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 1000


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components that stay fully inside the sprite bbox."""
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


def _choose_colors(
    rng: random.Random, n_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """
    Choose disjoint border/fill sets so the reversed read direction is behaviorally necessary.
    """
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_pairs]
    fill_values = colors[n_pairs : 2 * n_pairs]
    distractor_color = colors[2 * n_pairs]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the reversed-direction 2xN or Nx2 palette for the chosen side."""
    n_pairs = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_pairs), dtype=int)
        if side == "top":
            pal[0] = fill_values
            pal[1] = border_keys
        else:
            pal[0] = border_keys
            pal[1] = fill_values
        return pal

    pal = np.zeros((n_pairs, 2), dtype=int)
    if side == "left":
        pal[:, 0] = fill_values
        pal[:, 1] = border_keys
    else:
        pal[:, 0] = border_keys
        pal[:, 1] = fill_values
    return pal


def _place_palette(grid: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place the palette somewhere inside the removable 2-cell edge strip."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        r0, c0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        r0, c0 = rng.randint(0, h - pal_h), 0
    else:
        r0, c0 = rng.randint(0, h - pal_h), w - pal_w
    grid[r0 : r0 + pal_h, c0 : c0 + pal_w] = pal


def _cropped_shape(side: str, h: int, w: int) -> tuple[int, int]:
    """Shape after removing the palette strip."""
    if side in ("top", "bottom"):
        return h - 2, w
    return h, w - 2


def _embed_cropped(full: np.ndarray, crop: np.ndarray, side: str) -> None:
    """Paste the retained sprite field back into the full canvas."""
    if side == "top":
        full[2:, :] = crop
    elif side == "bottom":
        full[:-2, :] = crop
    elif side == "left":
        full[:, 2:] = crop
    else:
        full[:, :-2] = crop


def _to_full_coords(side: str, r: int, c: int) -> tuple[int, int]:
    """Map cropped-canvas coordinates back to full-grid coordinates."""
    if side == "top":
        return r + 2, c
    if side == "bottom":
        return r, c
    if side == "left":
        return r, c + 2
    return r, c


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """Find placements with a 1-cell margin from border and from other sprites."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite onto the cropped canvas."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    crop_h, crop_w = _cropped_shape(side, h, w)
    if crop_h < 5 or crop_w < 8:
        return None

    max_n = min(_PALETTE_N_MAX, 4, (9 - 1) // 2)
    n_pairs = rng.randint(_PALETTE_N_MIN, max_n)
    num_filled = rng.randint(_FILLED_SPRITE_MIN, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_filled + 1

    max_sh = min(_SPRITE_H_MAX, crop_h - 2)
    max_sw = min(_SPRITE_W_MAX, crop_w - 2)
    if max_sh < 3 or max_sw < 3:
        return None

    shapes: list[np.ndarray] = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sw > max_sw:
            sw = max_sw
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n_pairs, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    crop_out = np.full((crop_h, crop_w), bg, dtype=int)
    occ = np.zeros((crop_h, crop_w), dtype=bool)

    infos: list[tuple[np.ndarray, int, int | None]] = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            key = filled_keys[i]
            infos.append((mask, key, mapping[key]))
        else:
            infos.append((mask, distractor_color, None))

    placed: list[tuple[int, int, np.ndarray, list[np.ndarray], int | None]] = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(crop_h, crop_w, occ, mask)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(crop_out, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, holes, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(crop_h, r0 + sh + 1), max(0, c0 - 1) : min(crop_w, c0 + sw + 1)] = True

    full_out = np.full((h, w), bg, dtype=int)
    _embed_cropped(full_out, crop_out, side)
    _place_palette(full_out, pal, side, rng)

    inp = full_out.copy()
    for r0, c0, _mask, holes, fill in placed:
        if fill is None:
            continue
        for hpts in holes:
            for hr, hc in hpts:
                rr, cc = _to_full_coords(side, r0 + int(hr), c0 + int(hc))
                inp[rr, cc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        combo_out = main(inp)
        if not np.array_equal(combo_out, crop_out):
            return None
        legacy_out = _legacy_output(inp)
        if np.array_equal(combo_out, legacy_out):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
