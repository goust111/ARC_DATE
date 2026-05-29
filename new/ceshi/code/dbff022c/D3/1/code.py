# Variant of dbff022c
# Dimension: D3 — 调色板阅读方向的推断规则 (symbol_instructions)
# Change: 阅读方向由调色板所在边缘决定：上边缘→从上到下，下边缘→从下到上，左边缘→从左到右；第一行/列始终是边框颜色... → 阅读方向由调色板内的特殊标记像素指示

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the D3 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     - 调色板仍是贴着边缘的 2xM 或 Mx2 小块，但其中恰好有一个“标记槽”。
#     - 标记槽由一格背景色和一格非背景标记色组成；标记色落在哪一条线（外侧线或内侧线），就表示那条线是 border-key 线。
#     - 其余槽位都还是正常的 key/value 对，因此解题者需要先识别标记，再决定哪条线提供边框色。
#   concrete_behavior:
#     - 例：顶部调色板若是两行宽度为 5，左端一列为 [0, 7]，其余四列为 [fill, border]，
#       则虽然它在顶部，真正要读取的 key 线是第二行而不是第一行。
#     - 原任务在这里会固定把第一行当作 border keys；本变体则会把包含标记色的那一行当作 border keys。
#   complexity_calibration:
#     - 标记只允许出现一次，且只占一个槽位，避免自由度过大。
#     - 真实映射对数保持在 2-4，对应槽位数是 3-5，视觉上足够明显，也给颜色预算留出空间。
#     - fill 颜色与 border keys 生成时强制不相交，这样当“标记方向”与旧规则不一致时，旧规则会明显失效而不是偶然同解。
#   implicit_assumptions_review:
#     - 原 _best_palette 假设调色板内部没有背景色；这里不成立，因为标记槽半边就是背景。
#     - 原 _palette_mapping 直接按边缘 side 决定哪条线是 border_line；这里不成立，必须改为由标记槽决定。
#     - 原填洞逻辑只依赖最终的 border->fill 映射；这部分仍然成立，因此保留。
#   placement_legality:
#     - palette 仍与所有 sprite 保持至少 1 格背景间隔，避免标记槽的背景色与其他背景孔洞粘连成歧义结构。
#     - 标记槽放在调色板长轴的一端，但 solver 不依赖端点位置，只要求存在且唯一。
#   observability:
#     - 训练样例里会同时出现“外侧线是 key”和“内侧线是 key”的情况；同样的贴边位置不再足以判断方向。
#     - 标记槽是一眼可见的半空异形列/行，能支撑人类从少量样例归纳“看标记，而不是看边”。
#   new_main_assumptions:
#     - 恰好存在一个符合上述结构的调色板。
#     - 标记槽恰好有一格背景和一格非背景。
#     - 其他槽位两格都非背景，且 border keys 唯一；generator 会构造并用 oracle 校验这些条件。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0:2, start:end] = True
    elif side == "bottom":
        mask[h - 2 : h, start:end] = True
    elif side == "left":
        mask[start:end, 0:2] = True
    else:
        mask[start:end, w - 2 : w] = True
    return mask


def _palette_lines(g: np.ndarray, side: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (outer_line, inner_line) for a candidate edge palette segment.
    The outer line is the one touching the grid boundary.
    """
    h, w = g.shape
    if side == "top":
        return g[0, start:end], g[1, start:end]
    if side == "bottom":
        return g[h - 1, start:end], g[h - 2, start:end]
    if side == "left":
        return g[start:end, 0], g[start:end, 1]
    return g[start:end, w - 1], g[start:end, w - 2]


def _is_isolated_palette(g: np.ndarray, bg: int, side: str, start: int, end: int) -> bool:
    """
    Palette cells (excluding bg marker half) should not 4-touch same-colored cells outside
    the palette region, preventing accidental absorption of edge-touching sprites.
    """
    h, w = g.shape
    pal = _palette_mask((h, w), side, start, end)
    for r, c in zip(*np.where(pal), strict=False):
        r = int(r)
        c = int(c)
        v = int(g[r, c])
        if v == bg:
            continue
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                if int(g[rr, cc]) == v:
                    return False
    return True


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int, bool, int]:
    """
    Find the edge palette and decode its reading direction.

    Returns (side, start, end, key_is_outer, marker_index) where:
      - side in {"top","bottom","left","right"}
      - start/end span the palette along its long axis
      - key_is_outer says whether the border-key line is the outer edge line
      - marker_index is the slot containing the half-bg direction marker
    """
    h, w = g.shape
    best = None  # (score, side, start, end, key_is_outer, marker_index)

    def consider(side: str) -> None:
        nonlocal best
        axis_len = w if side in ("top", "bottom") else h
        if axis_len < 3:
            return
        for start in range(axis_len):
            for end in range(start + 3, axis_len + 1):
                outer_line, inner_line = _palette_lines(g, side, start, end)
                marker_slots: list[int] = []
                bad = False
                for idx, (outer, inner) in enumerate(
                    zip(outer_line.tolist(), inner_line.tolist(), strict=False)
                ):
                    outer = int(outer)
                    inner = int(inner)
                    outer_bg = outer == bg
                    inner_bg = inner == bg
                    if outer_bg and inner_bg:
                        bad = True
                        break
                    if outer_bg ^ inner_bg:
                        marker_slots.append(idx)
                if bad or len(marker_slots) != 1:
                    continue

                marker_index = marker_slots[0]
                key_is_outer = int(outer_line[marker_index]) != bg

                border_keys: list[int] = []
                fill_values: list[int] = []
                all_nonbg: set[int] = set()
                for idx, (outer, inner) in enumerate(
                    zip(outer_line.tolist(), inner_line.tolist(), strict=False)
                ):
                    outer = int(outer)
                    inner = int(inner)
                    if idx == marker_index:
                        if outer == bg and inner == bg:
                            bad = True
                            break
                        marker_color = outer if outer != bg else inner
                        all_nonbg.add(int(marker_color))
                        continue
                    if outer == bg or inner == bg:
                        bad = True
                        break
                    border = outer if key_is_outer else inner
                    fill = inner if key_is_outer else outer
                    border_keys.append(int(border))
                    fill_values.append(int(fill))
                    all_nonbg.add(int(outer))
                    all_nonbg.add(int(inner))
                if bad:
                    continue
                if len(border_keys) < 2:
                    continue
                if len(border_keys) != len(set(border_keys)):
                    continue
                if len(fill_values) != len(set(fill_values)):
                    continue
                if not _is_isolated_palette(g, bg, side, start, end):
                    continue

                score = (len(border_keys), len(all_nonbg), -(end - start))
                cand = (score, side, start, end, key_is_outer, marker_index)
                if best is None or cand[0] > best[0]:
                    best = cand

    if h >= 2:
        consider("top")
        consider("bottom")
    if w >= 2:
        consider("left")
        consider("right")

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end, key_is_outer, marker_index = best
    return side, int(start), int(end), bool(key_is_outer), int(marker_index)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Returns (mapping border->fill, palette_mask).
    """
    h, w = g.shape
    side, start, end, key_is_outer, marker_index = _best_palette(g, bg)
    pal_mask = _palette_mask((h, w), side, start, end)
    outer_line, inner_line = _palette_lines(g, side, start, end)
    border_line = outer_line if key_is_outer else inner_line
    fill_line = inner_line if key_is_outer else outer_line

    mapping: dict[int, int] = {}
    for idx, (border, fill) in enumerate(
        zip(border_line.tolist(), fill_line.tolist(), strict=False)
    ):
        if idx == marker_index:
            continue
        mapping[int(border)] = int(fill)

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
    Fill hollow sprites by decoding an edge palette whose direction is marked inside the palette.

    Input structure:
    - bg is the dominant border color. Several hollow sprites sit on the background.
    - One edge contains a 2xM or Mx2 palette block. Exactly one slot is a direction marker:
      one half of that slot is bg and the other half is a non-bg marker color.
    - The line containing the marker color is the border-key line; the opposite line holds fill colors.

    Rule:
    - Locate the edge palette and find the half-bg marker slot.
    - Use the marker-bearing row/column as the border-color legend, pair it with the opposite row/column,
      and build a border->fill mapping from all non-marker slots.
    - Fill each enclosed background hole whose surrounding border color appears in that mapping.

    Key insight:
    - Edge position no longer tells you how to read the legend. The solver must first identify the unique
      marker slot, infer which palette line comes first, and only then apply the same hole-filling logic.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

# Palette
_PAIR_COUNT_MIN, _PAIR_COUNT_MAX = 2, 4
_MARKER_AT_FRONT_PROB = 0.5

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
    """Create hollow sprite (h x w) with at least 1 enclosed bg hole, cell count <= cap."""
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
    rng: random.Random, pair_count: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int], int]:
    """
    Pick disjoint border/fill sets so the marker-controlled direction is visually consequential.
    """
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:pair_count]
    remaining = [c for c in all_c if c not in border_keys]
    rng.shuffle(remaining)
    fill_values = remaining[:pair_count]
    distractor_pool = [c for c in remaining if c not in fill_values]
    distractor_color = rng.choice(distractor_pool)
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)

    marker_pool = [c for c in range(1, 10) if c not in border_keys and c not in fill_values]
    if distractor_color in marker_pool:
        marker_pool.remove(distractor_color)
    marker_color = rng.choice(marker_pool) if marker_pool else rng.choice(fill_values)
    return border_keys, fill_values, distractor_color, mapping, filled_keys, marker_color


def _build_palette(
    side: str,
    border_keys: list[int],
    fill_values: list[int],
    marker_color: int,
    key_is_outer: bool,
    marker_at_front: bool,
) -> np.ndarray:
    """Construct the edge palette with one half-bg direction marker slot."""
    pair_count = len(border_keys)
    slots = pair_count + 1
    marker_idx = 0 if marker_at_front else slots - 1
    real_slots = [i for i in range(slots) if i != marker_idx]

    if side in ("top", "bottom"):
        pal = np.zeros((2, slots), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        key_line, fill_line = (outer, inner) if key_is_outer else (inner, outer)
        pal[key_line, marker_idx] = marker_color
        for slot, border, fill in zip(real_slots, border_keys, fill_values, strict=False):
            pal[key_line, slot] = border
            pal[fill_line, slot] = fill
    else:
        pal = np.zeros((slots, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        key_line, fill_line = (outer, inner) if key_is_outer else (inner, outer)
        pal[marker_idx, key_line] = marker_color
        for slot, border, fill in zip(real_slots, border_keys, fill_values, strict=False):
            pal[slot, key_line] = border
            pal[slot, fill_line] = fill
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    """Place palette on the chosen edge and mark its occupancy (with 1-cell padding)."""
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
    occ[max(0, pr0 - 1) : min(h, pr0 + pal_h + 1), max(0, pc0 - 1) : min(w, pc0 + pal_w + 1)] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find all (r0, c0) where mask fits without overlap and holes don't touch grid boundary."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
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
    """Render a sprite onto the grid: border cells in color, hole cells in fill (if given)."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_pairs = (h - 1) if side in ("left", "right") else (w - 1)
    pair_count = rng.randint(_PAIR_COUNT_MIN, min(_PAIR_COUNT_MAX, max_pairs))
    key_is_outer = rng.random() < 0.5
    marker_at_front = rng.random() < _MARKER_AT_FRONT_PROB

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, pair_count))
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

    border_keys, fill_values, distractor_color, mapping, filled_keys, marker_color = _choose_colors(
        rng, pair_count, num_filled
    )
    pal = _build_palette(side, border_keys, fill_values, marker_color, key_is_outer, marker_at_front)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
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
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

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
