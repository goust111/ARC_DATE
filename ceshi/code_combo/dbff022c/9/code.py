# Combo variant of dbff022c (index 9)
# combo_spec: D6/0+D9/3
# Name: 裁剪移除调色板 + 两遍处理

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# yaml:
#   concretization:
#     fused_rule:
#       - 输入仍是完整大网格: 某一侧边界内含 2xN / Nx2 palette, 其余区域是若干空心单色 sprite。
#       - solver 在完整输入上先读取 palette 的 border->fill 映射, 第一遍填满所有可匹配洞。
#       - 第二遍不是立刻输出, 而是把 palette 精确占据的那一小块改写为背景色。
#       - 最后再按 palette 所在 side 裁掉整条 2-cell 边带, 输出尺寸比输入恰少 2。
#     traced_example:
#       - 例: top palette 占 rows 0..1, cols 3..6; 某个红色空心框在 rows 4..8, cols 5..9, 红->黄。
#       - 原题: 读 top palette, 把框内部填黄, 输出仍保留 rows 0..1 上的 palette。
#       - 本变体: 先同样填黄; 再把 rows 0..1, cols 3..6 写回 bg; 最后整体返回 out[2:, :]。
#       - 差异: palette 既是“先读后删”的临时指令, 又决定裁掉哪一侧, 最终输出更小。
#   complexity_calibration:
#     - 保留原题 palette 方向、颜色映射和 hollow sprite 机制, 避免额外排序或隐藏条件。
#     - 输出差异固定且肉眼明显: 每个样例都会同时出现“洞被填色”与“网格某一轴缩短 2 格”。
#     - 为避免裁剪和删除把 sprite 一并伤到, generator 只在裁剪后保留的画布中放置 sprite。
#   implicit_assumptions_audit:
#     - _bg_boundary 依赖输入外边界大多数仍是 bg; palette 只占边带的一段, 其余边界保持 bg。
#     - _best_palette 假设 palette 是唯一可信的边缘 2-cell 图例; generator 禁止 sprite 触达完整输入外边界。
#     - _fill_enclosed_bg_regions 假设“可填洞”是相对完整输入的非边界封闭 bg 连通块; generator 让 sprite 洞远离完整输入边界。
#     - 新增的第二遍擦除依赖 pal_mask 精确标出 palette 小块; 因此 generator 保证 palette 内全是非 bg。
#     - 新增的裁剪依赖 palette 始终位于某条完整 2-row / 2-col 边带中; generator 直接在那条边带安放 palette。
#     - 删除与裁剪共用同一个 side 推断结果, 因而不会出现“删的是一边, 裁的是另一边”的冲突。
#   placement_legality:
#     - 先构造裁剪后画布 crop_out, 所有 sprite 都放在其内部并与边界保持至少 1 格余量。
#     - 再把 crop_out 嵌回完整输入/中间态, 仅在被裁边带放 palette。
#     - occ 仍给 sprite 之间和 sprite-vs-palette 预留 1 格背景缓冲, 防止接触、重叠和洞外泄到边界。
#   visual_observability:
#     - 人类可从输入看到 palette 是边缘图例, 从输出看到两件稳定现象: 某些空心框被填色、palette 整侧被裁掉。
#     - 第二遍擦除本身在最终裁剪后不直接可见, 但它与裁剪共享同一 palette 对象, 共同表达“palette 仅作输入指令”的语义。
#   cleanliness:
#     - sprite 全部位于保留画布, 不与 palette 边带叠加; 输出只剩 sprite 区域, 视觉更整洁。
#   color_inference:
#     - 颜色变化仍完全由 palette 的双行/双列对应关系提供线索, 颜色种类保持在原题范围内。
#   generator_guardrails:
#     - 至少 1 个可匹配 sprite, 且其洞面积 > 0, 保证样例不退化为“只裁剪不填色”。
#     - 被擦除的 palette 非背景格数始终 > 0, 证明第二遍确实作用于一个真实对象。
#     - output 固定等于“完整已擦除中间态再裁边带”, shape delta 必有且仅有一个维度差为 2。


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
      - for top/bottom: start/end are column indices [start, end)
      - for left/right: start/end are row indices [start, end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
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
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(union), -(c1 - c0), nonbg), "top", c0, c1)
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
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(union), -(c1 - c0), nonbg), "bottom", c0, c1)
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
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(union), -(r1 - r0), nonbg), "left", r0, r1)
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
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(border_keys)), len(union), -(r1 - r0), nonbg), "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(
    g: np.ndarray, bg: int
) -> tuple[dict[int, int], np.ndarray, tuple[str, int, int]]:
    """Return (mapping border->fill, palette_mask, palette_info)."""
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
        border = int(b)
        fill = int(f)
        if border == bg:
            continue
        mapping[border] = fill

    return mapping, pal_mask, (side, start, end)


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Fill each non-palette background component that stays away from the full-grid boundary,
    using the mapped color of its dominant adjacent border.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v != bg:
                        adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col in mapping and cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        out[tuple(pts.T)] = int(mapping[border])


def _crop_palette_strip(arr: np.ndarray, side: str) -> np.ndarray:
    """Remove the full 2-cell edge strip that hosts the palette."""
    if side == "top":
        return arr[2:, :].copy()
    if side == "bottom":
        return arr[:-2, :].copy()
    if side == "left":
        return arr[:, 2:].copy()
    return arr[:, :-2].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites from an edge legend, erase that legend, then crop away its strip.

    Input structure:
    - The outer border is mostly background, and exactly one side carries a 2xN or Nx2 palette legend inside the 2-cell edge band.
    - Hollow single-color sprites live in the remaining field; some sprite border colors appear on the palette's outer line and one sprite color may be a distractor.

    Rule:
    - Detect the palette, read its outer line as border-color keys and its inner line as fill colors, then fill every enclosed non-boundary background hole whose surrounding border matches a palette key.
    - After those fills are done, overwrite the palette block itself with background, and finally remove the entire 2-row or 2-column edge strip that carried that palette.

    Key insight: The same edge object has three roles in sequence: it provides the color mapping, it is explicitly deleted in a second pass, and it also tells you which side of the full grid must be cropped away.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask, pal_info = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    out[pal_mask] = bg

    side, _start, _end = pal_info
    return _crop_palette_strip(out, side)


# --------------- generator (combo variant) ---------------

# Full input dimensions; one axis shrinks by exactly 2 in the output.
_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 12, 15

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return background components that stay fully inside the sprite bbox."""
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
    """Pick palette pairs, a distractor border color, and the keys used by filled sprites."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:n_pairs]
    rng.shuffle(all_colors)
    fill_values = all_colors[:n_pairs]
    distractor_color = rng.choice([c for c in range(1, 10) if c not in border_keys])
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the chosen edge."""
    n_pairs = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_pairs), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n_pairs, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(
    grid: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> np.ndarray:
    """Place the palette inside the removable strip and return its exact mask."""
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
    pal_mask = np.zeros_like(grid, dtype=bool)
    pal_mask[r0 : r0 + pal_h, c0 : c0 + pal_w] = True
    return pal_mask


def _cropped_shape(side: str, h: int, w: int) -> tuple[int, int]:
    """Shape after removing the palette strip."""
    if side in ("top", "bottom"):
        return h - 2, w
    return h, w - 2


def _embed_cropped(full: np.ndarray, crop: np.ndarray, side: str) -> None:
    """Paste the cropped sprite field into the full pre-crop canvas."""
    if side == "top":
        full[2:, :] = crop
    elif side == "bottom":
        full[:-2, :] = crop
    elif side == "left":
        full[:, 2:] = crop
    else:
        full[:, :-2] = crop


def _crop_to_full_coords(side: str, r: int, c: int) -> tuple[int, int]:
    """Map a coordinate from the retained crop canvas back to the full input canvas."""
    if side == "top":
        return r + 2, c
    if side == "left":
        return r, c + 2
    return r, c


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """Find placements with 1-cell margin from the cropped canvas border and other sprites."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
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
    """Render a sprite onto a cropped canvas."""
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

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, crop_h - 2)
    max_sw = min(_SPRITE_W_MAX, crop_w - 2)
    if max_sh < 3 or max_sw < 3:
        return None

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_filled + 1

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

    crop_solved = np.full((crop_h, crop_w), bg, dtype=int)
    occ = np.zeros((crop_h, crop_w), dtype=bool)

    sprite_infos: list[tuple[np.ndarray, int, int | None]] = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            key = filled_keys[i]
            sprite_infos.append((mask, key, mapping[key]))
        else:
            sprite_infos.append((mask, distractor_color, None))

    placed: list[tuple[int, int, np.ndarray, int | None, list[np.ndarray]]] = []
    total_fill_cells = 0
    for mask, color, fill in sprite_infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(crop_h, crop_w, occ, mask)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(crop_solved, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill, holes))
        if fill is not None:
            total_fill_cells += sum(len(hpts) for hpts in holes)
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(crop_h, r0 + sh + 1), max(0, c0 - 1) : min(crop_w, c0 + sw + 1)] = True

    if total_fill_cells <= 0:
        return None

    full_solved = np.full((h, w), bg, dtype=int)
    _embed_cropped(full_solved, crop_solved, side)
    pal_mask = _place_palette(full_solved, pal, side, rng)

    full_erased = full_solved.copy()
    erased_nonbg = int(np.count_nonzero(full_erased[pal_mask] != bg))
    if erased_nonbg <= 0:
        return None
    full_erased[pal_mask] = bg
    output = _crop_palette_strip(full_erased, side)

    inp = full_solved.copy()
    for r0, c0, _mask, fill, holes in placed:
        if fill is None:
            continue
        for hpts in holes:
            for hr, hc in hpts:
                rr, cc = _crop_to_full_coords(side, r0 + int(hr), c0 + int(hc))
                inp[rr, cc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if not np.all(inp[pal_mask] != bg):
        return None
    if not np.all(full_erased[pal_mask] == bg):
        return None
    if output.shape == inp.shape:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
