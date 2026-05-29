# Combo variant of dbff022c (index 12)
# combo_spec: D2/4+D3/2
# Name: 对角线调色板 + 反直觉阅读方向

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 原题把 palette 放成边缘上的 2xN / Nx2 矩形块，并且按“贴边那一侧先读”的直觉顺序取
#      (border, fill) 配对。D2/4 把几何形态改成单条对角线链；本融合版进一步要求按对角线反向读。
#    - 具体例子：若顶边向右下延伸的链在边缘到内部的可见顺序是
#      (0,1)=6, (1,2)=8, (2,3)=4, (3,4)=3
#      那么直觉正向读会得到 6->8、4->3；融合规则却要求从内部反向读成
#      3->4、8->6。也就是说，同一条链不仅要沿对角线读取，还要从“远离边缘的一端”
#      反向配对，key/value 角色一起翻转。
#    - 洞填充主体不变：先找不接触全局边界的背景洞，再看其四邻中最常见的边框色，最后用
#      反向解码出的 palette 映射决定填充色。
#
# b) 复杂度校准
#    - 保留 4 种边缘锚定的 inward diagonal：`top_dr`, `top_dl`, `bottom_ur`, `bottom_ul`。
#      这样样例既能明显展示“沿对角线排列”，又不至于因为再开放左右边起点而让归纳分支过多。
#    - palette 对数限制在 2-4，对应链长 4-8；足够看出反向读取的规律，也给 sprites 留出空间。
#    - palette 的 border 集和 fill 集完全不重合，并且 filled sprites 只使用“反向读取后”的 key。
#      这样错误地按正向读取时，通常不会碰巧得到相同输出，融合效应可量化验证。
#
# c) 隐式假设审查
#    - 原 `_best_palette` 假设图例一定是二维矩形，并能整行/整列取出 key/value；这里失效，
#      必须枚举边缘起点 + 对角步进，并在候选链上先反转序列再做配对。
#    - 原 `_palette_mapping` 假设 `pal_mask` 是一个紧凑矩形；这里改成稀疏单点遮罩。
#    - 原 `_fill_enclosed_bg_regions` 只依赖 `mapping` 与 `pal_mask`，与图例几何无关，可复用。
#    - 新 solver 的额外假设是：palette 链颜色都非背景、两端顺序可由“整条链反转后奇偶位”唯一解码；
#      generator 会通过颜色互异、链隔离和只放一条合法对角线来保证这一点。
#
# d) 放置合法性
#    - 对角线链长度是 2N，因此生成时必须满足整条链都在网格内。
#    - palette 每个像素周围 1 格都标为占用，避免 sprites 与图例发生 4 邻接或视觉缠绕。
#    - sprites 之间仍保持至少 1 格背景间隔，且洞本身不能触碰全局边界，从而保证填充判定稳定。
#
# e) 视觉可观察性
#    - 人类可以直接看到：边缘有一串斜向单格链，且输出里的填色并不对应“靠边那一端先读”的配对，
#      而是对应“从链尾往回读”的配对。这个规律完全由可见位置顺序给出，不依赖颜色编号大小。
#    - 因为每题至少有 1 个会被填的 sprite，且错误正向读取会得到不同映射，所以 2-4 个样例足以
#      暴露“对角线 + 反向”这两个同时生效的变化。
#
# f) 可视化整洁度
#    - palette 仍是一条稀疏单链，不会像大矩形留白那样挤占角落；其周围保留背景缓冲带，和 sprites
#      视觉上明显分层。
#    - sprites 保持原题的单色空心轮廓，输出只在洞内加色，整体布局仍然清楚。
#
# g) 变化的可推断性
#    - palette 中所有 key/value 颜色两两不同，distractor 颜色也与二者都不同，避免出现“既像 key
#      又像 value”的二义性。
#    - 生成器始终让可填 sprites 使用反向解码后的 key，这给人类提供了稳定证据：只有按反方向读，
#      才能解释哪些边框色会触发哪些填充色。

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
    """All boundary columns that keep the full diagonal in bounds."""
    if kind in ("top_dr", "bottom_ur"):
        hi = w - length
        if hi < 0:
            return range(0)
        return range(0, hi + 1)
    lo = length - 1
    if lo > w - 1:
        return range(0)
    return range(lo, w)


def _reverse_pairs(vals: list[int]) -> tuple[list[int], list[int]]:
    """
    Decode a visible edge→interior palette chain by reversing the chain first.

    Example visible vals [f2, b2, f1, b1] become reversed [b1, f1, b2, f2],
    so keys are even positions of that reversed sequence.
    """
    rev = list(reversed([int(v) for v in vals]))
    return rev[0::2], rev[1::2]


def _isolated_points(g: np.ndarray, bg: int, pts: list[tuple[int, int]]) -> bool:
    """
    Palette cells should not be 4-neigh adjacent to same-colored non-palette cells.
    This prevents a sprite edge from masquerading as part of the palette chain.
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
    direction, `start_col` is the boundary column of the visible chain start, and
    `num_pairs` is the number of reverse-read (border, fill) pairs.
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
                border_keys, fill_values = _reverse_pairs(vals)
                if len(set(vals)) != length:
                    continue
                if len(set(border_keys)) != num_pairs:
                    continue
                if len(set(fill_values)) != num_pairs:
                    continue
                if set(border_keys) & set(fill_values):
                    continue
                if not _isolated_points(g, bg, pts):
                    continue
                score = (num_pairs, length)
                cand = (score, kind, int(start_col), int(num_pairs))
                if best is None or cand[0] > best[0]:
                    best = cand

    assert best is not None, "_best_palette must find a valid reverse-read diagonal palette"
    _, kind, start_col, num_pairs = best
    return kind, start_col, num_pairs


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return `(mapping border->fill, palette_mask)` for the reverse-read diagonal legend."""
    h, w = g.shape
    kind, start_col, num_pairs = _best_palette(g, bg)
    pts = _diag_positions(g.shape, kind, start_col, 2 * num_pairs)
    assert pts is not None

    pal_mask = np.zeros((h, w), dtype=bool)
    vals = []
    for r, c in pts:
        pal_mask[r, c] = True
        vals.append(int(g[r, c]))

    border_keys, fill_values = _reverse_pairs(vals)
    mapping = {
        int(border): int(fill)
        for border, fill in zip(border_keys, fill_values, strict=False)
    }
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
    Fill hollow sprites by decoding a diagonal edge legend in the opposite direction.

    Input structure:
    - Background is the dominant border color. Several hollow sprites sit on that background.
    - A sparse palette legend appears as a single diagonal chain of colored cells starting on
      the top or bottom edge. The visible edge-to-interior order is intentionally misleading:
      consecutive displayed cells are not the intended `(border, fill)` pairs.

    Rule:
    - Detect the diagonal legend, reverse the chain order, and then read the reversed sequence
      two cells at a time as `(border color, fill color)` pairs.
    - Find enclosed background regions that do not touch the global grid boundary.
    - For each such region, measure the most common adjacent border color; if that color is a
      decoded palette key, fill the region with the mapped palette value.

    Key insight:
    - The legend must be read along a diagonal, but not in the intuitive boundary-inward
      direction. Reversing the entire diagonal sequence is what reveals which colors are keys
      and which are fills.
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


def _choose_colors(
    rng: random.Random, num_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
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
    """Bias the visible edge cell near a side so the diagonal reads like a compact legend."""
    starts = list(_start_cols(w, length, kind))
    assert starts
    if kind in ("top_dr", "bottom_ur"):
        hi = min(starts[-1], _PALETTE_EDGE_SHIFT_MAX)
        return rng.randint(starts[0], hi)
    lo = max(starts[0], starts[-1] - _PALETTE_EDGE_SHIFT_MAX)
    return rng.randint(lo, starts[-1])


def _visible_palette_values(border_keys: list[int], fill_values: list[int]) -> list[int]:
    """
    Encode the palette so that reverse-reading recovers the intended mapping.

    Logical pairs are `(border_1, fill_1), (border_2, fill_2), ...`.
    Visible chain is the full logical sequence reversed, e.g.
    [b1, f1, b2, f2] -> [f2, b2, f1, b1].
    """
    logical = []
    for border, fill in zip(border_keys, fill_values, strict=False):
        logical.extend((int(border), int(fill)))
    return list(reversed(logical))


def _build_palette_cells(
    h: int, w: int, kind: str, start_col: int, border_keys: list[int], fill_values: list[int]
) -> list[tuple[tuple[int, int], int]]:
    """Construct the visible diagonal palette as individual colored cells."""
    pts = _diag_positions((h, w), kind, start_col, 2 * len(border_keys))
    assert pts is not None
    vals = _visible_palette_values(border_keys, fill_values)
    return [((int(r), int(c)), int(v)) for (r, c), v in zip(pts, vals, strict=False)]


def _place_palette(grid: np.ndarray, occ: np.ndarray, cells: list[tuple[tuple[int, int], int]]) -> None:
    """Place the sparse diagonal palette and reserve a 1-cell moat around each palette cell."""
    h, w = grid.shape
    for (r, c), color in cells:
        grid[r, c] = int(color)
        occ[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _find_valid_positions(
    H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find all `(r0, c0)` where mask fits without overlap and holes avoid the global border."""
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


def _paint_sprite(
    grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, fill: int | None, holes: list[np.ndarray]
) -> None:
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

    max_pairs = min(_PALETTE_N_MAX, H // 2, W // 2)
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
        if fill is None:
            continue
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
