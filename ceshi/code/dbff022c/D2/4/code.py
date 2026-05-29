# Variant of dbff022c
# Dimension: D2 — 调色板图例的位置与朝向 (symbol_instructions)
# Change: 调色板为2xN或Nx2色块，位于网格边缘（上/下/左），根据所在边缘确定阅读方向：上边缘从上到下读，下边缘从下到上读，左... → 调色板为对角线排列的颜色对

# Design analysis for the D2 variant
# ----------------------------------
# a) 具体化与行为追踪
#    - 原题的图例是边缘上的 2xN / Nx2 矩形块；本变体把它改成“从上边或下边出发、沿对角线向网格内部延伸”的单像素链。
#    - 调色板长度为 2N 个像素，按读取顺序分组为
#      (border_1, fill_1), (border_2, fill_2), ...
#      其中同一对的两个像素彼此对角相邻；读取方向始终是“从贴边的第一个像素往内读”。
#    - 具体例子：若顶边起点在第 1 列，方向为右下，则像素序列是
#      (0,1)=8, (1,2)=3, (2,3)=4, (3,4)=6
#      表示 8->3、4->6 两组映射。原题会把同样的信息编码成两行矩形块；本变体要求沿斜线按偶/奇位置配对。
#    - 洞填充流程本身不变：先找空心精灵的封闭背景区域，再看该区域四邻最常见的边框颜色，最后用调色板映射到填充色。
#
# b) 复杂度校准
#    - 只保留 4 种“从边缘向内”的对角方向：`top_dr`, `top_dl`, `bottom_ur`, `bottom_ul`。
#      不再同时开放左/右边起始，避免分支过多导致 2-4 个样例里不易归纳。
#    - 调色板对数 N 取 2-4。这样可以保证 border / fill / distractor 都使用互不相同的非背景色，
#      视觉上更清楚，也避免颜色重叠削弱“对角线成对读取”的线索。
#    - 为了让图例仍然显眼，起点会靠近左右边缘，使对角线图例明显像“贴着外框伸进来的一条斜链”。
#
# c) 隐式假设审查（针对原 main）
#    - 原 `_best_palette` 假设调色板一定是 2 行或 2 列的连续矩形，并且能从一整行/列取到 border 与 fill。
#      这个前提在变体中失效，必须改成“枚举边缘起点 + 对角步进”，并在偶数位读取 border、奇数位读取 fill。
#    - 原 `_palette_mapping` 假设 `pal_mask` 是矩形区域；变体中 `pal_mask` 是稀疏的若干单元格，必须按单点遮罩。
#    - 原 `_fill_enclosed_bg_regions` 只依赖 `mapping` 和 `pal_mask`，不依赖调色板几何形状，因此可直接复用。
#    - 原 generator 用矩形 bbox+padding 给调色板留空；对角线若仍用整块 bbox，会白白占掉大面积空间，
#      所以改成按每个调色板像素各自扩 1 格的方式保留缓冲区。
#
# d) 放置合法性
#    - 调色板起点在上边或下边；整条斜链长度是 2N，因此必须满足所有对角坐标都在网格内。
#    - 调色板周围 1 格内不能放精灵，避免颜色相同的 4 邻接破坏 solver 对“孤立图例”的识别。
#    - 精灵之间仍保留至少 1 格背景分隔，洞也仍然不能接触全局外边界，保证填充判定稳定。
#
# e) 视觉可观察性
#    - 人类解题者能看到：有一串贴边的斜向单格颜色链，而输出里若干空心图形的内部被填成链上成对出现的另一种颜色。
#    - 排序完全由可见的空间次序决定（沿斜线从边界往内），没有使用颜色 id 大小、坐标数值大小等不可见因素。
#
# f) 可视化整洁度
#    - 图例固定为单条斜链，不与精灵重叠，也不会被大矩形留白包住，因而不会在角落产生杂乱大块。
#    - 精灵仍然是原题的单色空心轮廓，视觉主次清晰：斜链给出规则，空心精灵承载变换对象。
#
# g) 颜色变化可推断性
#    - border / fill / distractor 颜色彼此不重合，避免“某种颜色既像键又像值”的额外歧义。
#    - 每题至少有 1 个 distractor 精灵不被填色，强调“不是所有空心图形都填，只有边框色出现在斜链奇偶配对中的才填”。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, neighbors4


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


_PALETTE_KINDS = ("top_dr", "top_dl", "bottom_ur", "bottom_ul")


def _diag_positions(shape: tuple[int, int], kind: str, start_col: int, length: int) -> list[tuple[int, int]] | None:
    """Return `length` coordinates along the chosen edge-anchored inward diagonal."""
    h, w = map(int, shape)
    if kind == "top_dr":
        r, c, dr, dc = 0, int(start_col), 1, 1
    elif kind == "top_dl":
        r, c, dr, dc = 0, int(start_col), 1, -1
    elif kind == "bottom_ur":
        r, c, dr, dc = h - 1, int(start_col), -1, 1
    else:  # bottom_ul
        r, c, dr, dc = h - 1, int(start_col), -1, -1

    pts: list[tuple[int, int]] = []
    for _ in range(int(length)):
        if not (0 <= r < h and 0 <= c < w):
            return None
        pts.append((int(r), int(c)))
        r += dr
        c += dc
    return pts


def _start_cols(w: int, length: int, kind: str) -> range:
    """All edge start columns that keep the diagonal of `length` cells in-bounds."""
    if kind in ("top_dr", "bottom_ur"):
        hi = w - length
        if hi < 0:
            return range(0)
        return range(0, hi + 1)
    lo = length - 1
    if lo > w - 1:
        return range(0)
    return range(lo, w)


def _isolated_points(g: np.ndarray, bg: int, pts: list[tuple[int, int]]) -> bool:
    """
    Palette cells should not be 4-neigh adjacent to same-colored non-palette cells.
    The diagonal palette itself has no internal 4-neigh contacts, so this check focuses
    entirely on keeping the legend separate from sprites.
    """
    h, w = g.shape
    pal = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        pal[r, c] = True

    for r, c in pts:
        v = int(g[r, c])
        if v == bg:
            return False
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                if int(g[rr, cc]) == v:
                    return False
    return True


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the diagonal palette legend.

    Returns `(kind, start_col, num_pairs)` where `kind` determines the edge and diagonal
    direction, `start_col` is the boundary column of the first palette cell, and
    `num_pairs` is the number of (border, fill) pairs along that diagonal.
    """
    h, w = g.shape
    max_pairs = min(4, h // 2, w // 2)
    best = None  # (score, kind, start_col, num_pairs)

    for kind in _PALETTE_KINDS:
        for num_pairs in range(2, max_pairs + 1):
            length = 2 * num_pairs
            for start_col in _start_cols(w, length, kind):
                pts = _diag_positions(g.shape, kind, start_col, length)
                if pts is None:
                    continue
                vals = [int(g[r, c]) for r, c in pts]
                if any(v == bg for v in vals):
                    continue
                border_keys = vals[0::2]
                fill_values = vals[1::2]
                if len(set(border_keys)) != num_pairs:
                    continue
                if len(set(fill_values)) != num_pairs:
                    continue
                if len(set(vals)) < 3:
                    continue
                if not _isolated_points(g, bg, pts):
                    continue
                score = (num_pairs, len(set(vals)))
                cand = (score, kind, int(start_col), int(num_pairs))
                if best is None or cand[0] > best[0]:
                    best = cand

    assert best is not None, "_best_palette must find a valid diagonal palette"
    _, kind, start_col, num_pairs = best
    return kind, start_col, num_pairs


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Returns (mapping border->fill, palette_mask).
    """
    h, w = g.shape
    kind, start_col, num_pairs = _best_palette(g, bg)
    pts = _diag_positions(g.shape, kind, start_col, 2 * num_pairs)
    assert pts is not None

    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}
    vals = []
    for r, c in pts:
        pal_mask[r, c] = True
        vals.append(int(g[r, c]))

    for border, fill in zip(vals[0::2], vals[1::2], strict=False):
        mapping[int(border)] = int(fill)

    return mapping, pal_mask


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Find background regions (excluding the palette area) that do NOT touch the grid boundary,
    and fill each region based on the most common adjacent border color (mapped via palette).
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

        fill = int(mapping[border])
        for r, c in pts:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading a diagonal color legend that starts on the outer border.

    Input structure:
    - Background is the dominant border color. Several hollow sprites sit on that background.
    - A sparse palette legend appears as a single diagonal chain of colored cells beginning on
      the top or bottom edge; consecutive diagonal cell pairs encode (border color, fill color).

    Rule:
    - Detect the diagonal legend, read it from the boundary inward, and interpret every two
      successive diagonal cells as one border→fill mapping.
    - Find enclosed background regions that do not touch the global grid boundary.
    - For each such region, measure the most common adjacent border color; if that color is a
      palette key, fill the region with the mapped palette value.

    Key insight:
    - The legend is no longer a horizontal or vertical block. The crucial cue is the visible
      diagonal ordering: parity along the inward diagonal determines which color is the key and
      which is the fill value.
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
_GRID_H_MIN, _GRID_H_MAX = 6, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4
_PALETTE_EDGE_SHIFT_MAX = 2

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 300
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return list of point-arrays for bg components not touching sprite boundary."""
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
    """Create hollow sprite (h x w) with at least 1 enclosed bg hole, cell count <= limit."""
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n)
        sprite[br, s : s + n] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n)
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


def _choose_colors(rng: random.Random, num_pairs: int, num_filled: int) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick disjoint border/fill/distractor colors and choose which keys appear in sprites."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    need = 2 * num_pairs + 1
    assert need <= len(all_c)
    border_keys = all_c[:num_pairs]
    fill_values = all_c[num_pairs : 2 * num_pairs]
    distractor_color = all_c[2 * num_pairs]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _pick_palette_start(w: int, length: int, kind: str, rng: random.Random) -> int:
    """Bias the start near one side so the edge-anchored diagonal reads like a compact legend."""
    starts = list(_start_cols(w, length, kind))
    assert starts
    if kind in ("top_dr", "bottom_ur"):
        hi = min(starts[-1], _PALETTE_EDGE_SHIFT_MAX)
        return rng.randint(starts[0], hi)
    lo = max(starts[0], starts[-1] - _PALETTE_EDGE_SHIFT_MAX)
    return rng.randint(lo, starts[-1])


def _build_palette_cells(h: int, w: int, kind: str, start_col: int, border_keys: list[int], fill_values: list[int]) -> list[tuple[tuple[int, int], int]]:
    """Construct the diagonal palette as individual colored cells."""
    pts = _diag_positions((h, w), kind, start_col, 2 * len(border_keys))
    assert pts is not None
    seq = []
    for idx, border in enumerate(border_keys):
        seq.append((pts[2 * idx], int(border)))
        seq.append((pts[2 * idx + 1], int(fill_values[idx])))
    return seq


def _place_palette(grid: np.ndarray, occ: np.ndarray, cells: list[tuple[tuple[int, int], int]]) -> None:
    """Place the sparse diagonal palette and reserve a 1-cell moat around each palette cell."""
    h, w = grid.shape
    for (r, c), color in cells:
        grid[r, c] = int(color)
        occ[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _find_valid_positions(H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]) -> list[tuple[int, int]]:
    """Find all (r0, c0) where mask fits without overlap and holes don't touch grid boundary."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            if occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == H - 1 or c0 + hc == W - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, fill: int | None, holes: list[np.ndarray]) -> None:
    """Render a sprite onto the grid: border cells in color, hole cells in fill (if given)."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = int(fill)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    max_pairs = min(_PALETTE_N_MAX, W // 2, max(2, (H - 2) // 2))
    if max_pairs < _PALETTE_N_MIN:
        return None
    num_pairs = rng.randint(_PALETTE_N_MIN, max_pairs)

    kind = rng.choice(_PALETTE_KINDS)
    length = 2 * num_pairs
    start_col = _pick_palette_start(W, length, kind, rng)

    max_sh = min(_SPRITE_H_MAX, H)
    max_sw = min(_SPRITE_W_MAX, W)

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, num_pairs))
    total_sprites = num_filled + 1

    shapes = []
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

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, num_pairs, num_filled)
    pal_cells = _build_palette_cells(H, W, kind, start_col, border_keys, fill_values)

    grid = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    _place_palette(grid, occ, pal_cells)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            c, f = filled_keys[i], mapping[filled_keys[i]]
        else:
            c, f = distractor_color, None
        infos.append((mask, c, f))

    placed = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(H, W, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill in placed:
        if fill is not None:
            for hpts in _get_enclosed_holes(mask):
                for hr, hc in hpts:
                    inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
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
