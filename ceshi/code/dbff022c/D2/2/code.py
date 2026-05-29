# Variant of dbff022c
# Dimension: D2 — 调色板图例的位置与朝向 (symbol_instructions)
# Change: 调色板为2xN或Nx2色块，位于网格边缘（上/下/左），根据所在边缘确定阅读方向：上边缘从上到下读，下边缘从下到上读，左... → 调色板使用嵌套方框编码——外框色为边框色，内部色为填充色

# Design analysis
# ----------------
# variant_spec:
#   preserved_core:
#     - 背景仍为 0，主体仍是若干单色空心精灵，任务仍是把封闭孔洞按 border->fill 映射填色。
#     - 输出先行生成、再把已填孔洞挖空回输入、最后用 main() 自检的骨架保持不变。
#   legend_change:
#     - 原题的图例是边缘上的两条色带，需要靠“在哪一条边上”决定读图方向。
#     - 变体把每个映射条目改成固定 3x3 的嵌套方框：
#         b b b
#         b f b
#         b b b
#       其中外环颜色 b 表示某类精灵的边框色，中心颜色 f 表示该边框对应的目标填充色。
#     - 顶边/底边时，这些 3x3 图例横向排成一列；左边/右边时纵向排成一列；条目之间留 1 格背景。
#     - 这使“方向”从读取顺序问题，变成“图例放在哪一侧、按哪一条轴排布”的问题；每个符号本身已经局部编码了映射。
#   behavior_trace:
#     - 原逻辑示例：顶边色带第一行 [2,4,6]、第二行 [7,8,9]，读成 2->7, 4->8, 6->9。
#     - 新逻辑示例：顶边放三个 3x3 方框，外环分别是 2/4/6，中心分别是 7/8/9；
#       solver 不再“按边缘方向读两行”，而是逐个识别 3x3 方框，直接得到同样的 2->7, 4->8, 6->9。
#     - 后续对精灵孔洞的处理不变：仍按孔洞邻接到的主导边框色决定填充值。
#   complexity_calibration:
#     - 图例条目数仍保持少量离散档位：至少 2 个，通常 2-4 个；这样样例里会出现“有用条目 + 未使用条目”的重复支撑。
#     - 每个条目要求中心色 != 外环色，否则会退化成纯色 3x3 块，肉眼看不出“嵌套方框”结构。
#     - 边框键仍要求互不重复，避免多个图例争夺同一 sprite border 的歧义。
#   implicit_assumption_review:
#     - 原 solver 的 _best_palette / _palette_mapping 假设图例一定是 2xN 或 Nx2 的连续色带；该假设在变体中失效，必须整体替换。
#     - 原填洞逻辑只依赖“能得到稳定的 border->fill 映射”和“孔洞是封闭背景区域”，这两点在变体中仍成立，因此保留。
#     - 新 solver 的 _edge_entries 假设：
#         1. 图例条目是精确的 3x3 嵌套方框；
#         2. 它们只会出现在外边缘贴边处；
#         3. 相邻条目至少隔 1 格背景，故起点相差至少 4。
#       新 generator 会显式保证这三点。
#   placement_legality:
#     - 图例整体 bbox 及其外侧 1 格缓冲都会记入 occ，避免 sprite 与图例粘连。
#     - sprite 额外限制为不贴外边界，既保持边界背景占优，也避免输入中的 sprite 与“边缘图例”在视觉上混淆。
#     - 左右边图例因为更高，条目数会按网格高度自动收缩，防止不可放置。
#   visual_observability:
#     - 人类从 2-4 组样例中能直接看到：边缘有若干“外环+中心”的小方框，输出里凡是外框颜色匹配的空心精灵，其内部都被涂成图例中心色。
#     - 规则不依赖颜色 id 的大小或排序，也不依赖不可见的读取方向；线索完全由可见的局部结构给出。
#   anti_regression_check_plan:
#     - 测试后额外统计图例条目数、边缘分布，并与原 solver 对同一输入的行为比较；
#       若原 solver 经常也能在同一输入上读出等价结果，则说明变体退化，需要回调图例生成。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, neighbors4


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _nested_box_pair(sub: np.ndarray, bg: int) -> tuple[int, int] | None:
    """Return (border, fill) if sub is a 3x3 nested-box legend entry."""
    if sub.shape != (3, 3):
        return None
    border = int(sub[0, 0])
    fill = int(sub[1, 1])
    if border == bg or fill == bg or fill == border:
        return None
    outer = [
        int(sub[0, 0]), int(sub[0, 1]), int(sub[0, 2]),
        int(sub[1, 0]), int(sub[1, 2]),
        int(sub[2, 0]), int(sub[2, 1]), int(sub[2, 2]),
    ]
    if any(v != border for v in outer):
        return None
    return border, fill


def _entry_mask(shape: tuple[int, int], side: str, start: int) -> np.ndarray:
    """Boolean mask for one 3x3 legend entry on a given edge."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0:3, start : start + 3] = True
    elif side == "bottom":
        mask[h - 3 : h, start : start + 3] = True
    elif side == "left":
        mask[start : start + 3, 0:3] = True
    else:
        mask[start : start + 3, w - 3 : w] = True
    return mask


def _mask_is_isolated(g: np.ndarray, bg: int, mask: np.ndarray) -> bool:
    """
    Palette cells should not be 4-neigh adjacent to same-colored cells outside the entry.
    This mirrors the original palette-isolation safeguard.
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


def _edge_entries(g: np.ndarray, bg: int, side: str) -> list[tuple[int, int, int, np.ndarray]]:
    """
    Find all isolated 3x3 nested-box legend entries on one edge.

    Returns a list of (start, border, fill, mask), ordered along the edge axis.
    """
    h, w = g.shape
    if side in ("top", "bottom"):
        if h < 3:
            return []
        limit = w - 2
    else:
        if w < 3:
            return []
        limit = h - 2

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for start in range(limit):
        if side == "top":
            sub = g[0:3, start : start + 3]
        elif side == "bottom":
            sub = g[h - 3 : h, start : start + 3]
        elif side == "left":
            sub = g[start : start + 3, 0:3]
        else:
            sub = g[start : start + 3, w - 3 : w]

        pair = _nested_box_pair(sub, bg)
        if pair is None:
            continue

        mask = _entry_mask((h, w), side, start)
        if not _mask_is_isolated(g, bg, mask):
            continue
        candidates.append((int(start), int(pair[0]), int(pair[1]), mask))

    chosen: list[tuple[int, int, int, np.ndarray]] = []
    last_end = -99
    for start, border, fill, mask in candidates:
        if start <= last_end:
            continue
        chosen.append((start, border, fill, mask))
        last_end = start + 3
    return chosen


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Decode the best edge legend made of 3x3 nested boxes.

    Returns (mapping border->fill, palette_mask).
    """
    h, w = g.shape
    best = None  # (score, entries)
    for side in ("top", "bottom", "left", "right"):
        entries = _edge_entries(g, bg, side)
        if len(entries) < 2:
            continue
        border_keys = [border for _, border, _, _ in entries]
        if len(border_keys) != len(set(border_keys)):
            continue
        fills = [fill for _, _, fill, _ in entries]
        starts = [start for start, _, _, _ in entries]
        span = starts[-1] - starts[0] + 3
        score = (len(entries), len(set(fills)), -span)
        cand = (score, side, entries)
        if best is None or score > best[0]:
            best = cand

    assert best is not None, "_palette_mapping must find a valid nested-box palette"
    _, _, entries = best
    mapping: dict[int, int] = {}
    pal_mask = np.zeros((h, w), dtype=bool)
    for _, border, fill, mask in entries:
        mapping[int(border)] = int(fill)
        pal_mask |= mask
    return mapping, pal_mask


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
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
    Fill hollow sprites' interiors using nested-box legend symbols on the grid edge.

    Input structure:
    - bg is the dominant border color of the grid; several hollow sprites sit away from the edge.
    - A legend occupies exactly one outer edge: each entry is a 3x3 nested box whose outer ring is a sprite border color and whose center is the corresponding fill color.

    Rule:
    - Locate the edge legend entries and decode each nested box as a border->fill pair.
    - Find enclosed bg regions that belong to hollow sprites and do not touch the outer grid boundary.
    - For each such region, detect the dominant adjacent border color and, if that color appears in the legend, fill the region with the mapped center color.

    Key insight: The mapping is carried by each local symbol itself, so unlike the original strip legend there is no edge-dependent reading direction to infer.
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
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_TINY_H_THRESHOLD = 5

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5
_LEGEND_SIZE = 3
_LEGEND_GAP = 1

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
_MAX_RETRIES = 300
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


def _legend_entry(border: int, fill: int) -> np.ndarray:
    """Construct one 3x3 nested-box legend entry."""
    entry = np.full((_LEGEND_SIZE, _LEGEND_SIZE), int(border), dtype=int)
    entry[1, 1] = int(fill)
    return entry


def _palette_span(n: int) -> int:
    """Bounding-box span along the arrangement axis for n legend entries."""
    return n * _LEGEND_SIZE + (n - 1) * _LEGEND_GAP


def _palette_capacity(side: str, H: int, W: int) -> int:
    """Maximum legend entry count that fits on the chosen edge."""
    axis_len = W if side in ("top", "bottom") else H
    return min(_PALETTE_N_MAX, (axis_len + _LEGEND_GAP) // (_LEGEND_SIZE + _LEGEND_GAP))


def _choose_colors(
    rng: random.Random, N: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick border keys, fill values, a distractor border color, and used keys."""
    colors = list(range(1, 10))
    border_keys = rng.sample(colors, N)

    fill_values: list[int] = []
    for _ in range(100):
        trial = rng.sample(colors, N)
        if all(f != b for f, b in zip(trial, border_keys, strict=False)):
            fill_values = trial
            break
    if not fill_values:
        fill_values = border_keys[1:] + border_keys[:1]

    blocked = set(border_keys) | set(fill_values)
    dist_cands = [c for c in colors if c not in blocked]
    if not dist_cands:
        dist_cands = [c for c in colors if c not in border_keys]
    distractor_color = rng.choice(dist_cands)

    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the edge legend array as a row/column of separated nested boxes."""
    N = len(border_keys)
    span = _palette_span(N)
    if side in ("top", "bottom"):
        pal = np.zeros((_LEGEND_SIZE, span), dtype=int)
        for i, (border, fill) in enumerate(zip(border_keys, fill_values, strict=False)):
            c0 = i * (_LEGEND_SIZE + _LEGEND_GAP)
            pal[:, c0 : c0 + _LEGEND_SIZE] = _legend_entry(border, fill)
    else:
        pal = np.zeros((span, _LEGEND_SIZE), dtype=int)
        for i, (border, fill) in enumerate(zip(border_keys, fill_values, strict=False)):
            r0 = i * (_LEGEND_SIZE + _LEGEND_GAP)
            pal[r0 : r0 + _LEGEND_SIZE, :] = _legend_entry(border, fill)
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    """Place palette on the chosen edge and mark its bbox plus 1-cell padding as occupied."""
    H, W = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, W - pal_w)
    elif side == "bottom":
        pr0, pc0 = H - pal_h, rng.randint(0, W - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, H - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, H - pal_h), W - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[
        max(0, pr0 - 1) : min(H, pr0 + pal_h + 1),
        max(0, pc0 - 1) : min(W, pc0 + pal_w + 1),
    ] = True


def _find_valid_positions(
    H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find all placements where the sprite stays inside the border and away from occupied cells."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            if r0 <= 0 or c0 <= 0 or (r0 + sh) >= H or (c0 + sw) >= W:
                continue
            if occ[
                max(0, r0 - 1) : min(H, r0 + sh + 1),
                max(0, c0 - 1) : min(W, c0 + sw + 1),
            ].any():
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
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite onto the grid."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    feasible_sides = [side for side in ("top", "bottom", "left", "right") if _palette_capacity(side, H, W) >= _PALETTE_N_MIN]
    if not feasible_sides:
        return None
    side = rng.choice(feasible_sides)
    max_N = _palette_capacity(side, H, W)
    N = rng.randint(_PALETTE_N_MIN, max_N)

    free_h = H - (4 if side in ("top", "bottom") else 2)
    free_w = W - (4 if side in ("left", "right") else 2)
    max_sh = min(_SPRITE_H_MAX, free_h)
    max_sw = min(_SPRITE_W_MAX, free_w)
    if H <= _TINY_H_THRESHOLD:
        max_sh = min(max_sh, 3)
    if max_sh < 3 or max_sw < 3:
        return None

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, N))
    total_sprites = num_filled + 1

    shapes: list[np.ndarray] = []
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

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, N, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            border = filled_keys[i]
            fill = mapping[border]
        else:
            border = distractor_color
            fill = None
        infos.append((mask, border, fill))

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
        occ[
            max(0, r0 - 1) : min(H, r0 + sh + 1),
            max(0, c0 - 1) : min(W, c0 + sw + 1),
        ] = True

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
