# Variant of dbff022c
# Dimension: D1 — 颜色映射规则——边框颜色到填充颜色的对应方式 (match_cond)
# Change: 调色板直接给出边框颜色→填充颜色的一一对应关系，精灵的边框颜色在调色板中查找得到唯一填充颜色... → 精灵边框颜色在调色板中的位置索引决定填充色

"""
design_analysis:
  variant_core:
    original_palette_rule: >
      原题把边框色放在外侧一行/一列，把填充色放在相邻的内侧一行/一列；
      solver 逐格 zip，这意味着“同列/同行的两个像素天然成对”。
    new_palette_rule: >
      变体保留“图例位于网格边缘、由两条线组成”的整体外观，但两条线变成两条独立有序色序。
      外侧线给出 border-key 序列，内侧线给出 fill-value 序列；二者不再按同坐标配对，
      而是按扫描顺序的第 1 个、第 2 个……建立索引对应。
    concrete_visualization: >
      palette 仍是 2xL 或 Lx2 的边缘带，但允许其中有 bg 空隙。
      两条线采用交错排布：同一个列/行上最多只有一条线出现非 bg 颜色。
      例如 top 图例可长成：
        row0: 2 0 8 0 4 0
        row1: 0 7 0 3 0 6
      那么映射是 2->7, 8->3, 4->6，因为它们分别是两条色序中的第 1/2/3 个元素。
  behavior_trace:
    original_logic:
      - 原题读取 top palette 时执行 zip(border_line, fill_line)。
      - 上例若套用原题逻辑，会得到 2->0, 8->0, 4->0，显然错误。
    variant_logic:
      - 先分别过滤非 bg，得到 border_seq=[2,8,4], fill_seq=[7,3,6]。
      - 再按索引 zip(border_seq, fill_seq)，得到 2->7, 8->3, 4->6。
      - 最后用洞周围主导边框色查 mapping，填充对应颜色。
    observable_difference: >
      原题依赖“上下正对/左右正对”，变体依赖“同一方向上的先后顺序”。
      肉眼可以直接看到两条线是错位交错的，因此不能再按局部相邻配对。
  complexity_calibration:
    why_not_keep_dense_pairs: >
      如果仍用无空隙的 2xN / Nx2 实心 palette，那么“按索引配对”和“按同坐标配对”视觉上等价，
      变体不可观察，会退化成原题。
    controlled_freedom:
      - 仍只保留一处 palette，且仍位于单一边缘。
      - 两条色序长度相等，颜色各自唯一，避免多重解释。
      - palette 长度上限从 5 调整到 4，这样 border 序列、fill 序列、distractor 都可取互不相同颜色，
        避免颜色重叠削弱“独立色序”的可读性。
      - filled sprites 至少放 2 个，保证单个样例里能同时看到多个索引映射结果。
  implicit_assumption_audit:
    old_main_dependencies:
      - "_best_palette 假设 palette 区域里两条线各自至少有 2 个非 bg 颜色。这个前提保留。"
      - "_palette_mapping 旧版逐格 zip(border_line, fill_line)。这个前提失效，必须改成分别提取非 bg 序列后再按索引 zip。"
      - "_fill_enclosed_bg_regions 假设每个洞相邻边框色能在 mapping 中找到或故意找不到（distractor）。这个前提保留。"
      - "palette 检测旧版默认可以吸收整块实心 2xN / Nx2；变体里若不限制错位结构，就可能把别的边缘对象误判成 palette。"
    new_solver_assumptions:
      - "palette 的外侧线与内侧线非 bg 个数相同，且都至少为 2。generator 保证。"
      - "同一列/行不会同时出现 outer 和 inner 的非 bg 颜色，即交错排布。generator 保证。"
      - "outer 序列颜色唯一，作为边框 key；inner 序列颜色唯一，便于清晰读取。generator 保证。"
      - "palette 与 sprite 之间仍隔着至少 1 格背景，isolated 检测继续成立。generator 保证。"
  placement_legality:
    palette_bbox: >
      虽然 palette 中含 bg 空格，但 generator 仍把整个 2xL / Lx2 包围盒视为占用区，并加 1 格 padding，
      防止 sprite 落到 palette 空隙里造成视觉混淆。
    sprite_rules:
      - hollow sprite 的生成、洞不接触网格边界、彼此与 palette 之间至少 1 格间距等原约束保留。
      - distractor sprite 的边框色不出现在 border_seq 中，因此其洞保持背景色。
  visual_observability:
    cues_for_humans:
      - "图例仍在边缘，仍明显分成“外侧一排”和“内侧一排”，所以解题者知道这两排都参与规则。"
      - "两排交错，消除了逐格配对的误导；只能按从左到右/从上到下的次序理解。"
      - "至少 2 个被填充的 sprite 提供重复证据，避免只靠一次颜色变化猜规则。"
  cleanliness:
    visual_layout: >
      交错 palette 比完全随机散点更整洁，且仍保留顺序信息。sprite 仍是空心单色边框，
      输出只在洞内上色，不会叠加更多标记层。
  color_inference:
    rule_visibility: >
      映射只依赖可见顺序，不依赖颜色 id 大小、频次排序或不可见元数据。
      颜色数被控制在 9 个非 bg ARC 颜色内，且一例中不会出现大量重复角色争夺解释权。
"""

import random

import numpy as np

from arc_utils import components_mask, neighbors4


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    h, w = shape
    pal = np.zeros((h, w), dtype=bool)
    if side == "top":
        pal[0:2, start:end] = True
    elif side == "bottom":
        pal[h - 2 : h, start:end] = True
    elif side == "left":
        pal[start:end, 0:2] = True
    else:
        pal[start:end, w - 2 : w] = True
    return pal


def _extract_sequences(
    outer_line: np.ndarray, inner_line: np.ndarray, bg: int
) -> tuple[list[int], list[int]] | None:
    outer = [int(x) for x in outer_line.tolist() if int(x) != bg]
    inner = [int(x) for x in inner_line.tolist() if int(x) != bg]
    if len(outer) < 2 or len(inner) < 2 or len(outer) != len(inner):
        return None
    if len(outer) != len(set(outer)) or len(inner) != len(set(inner)):
        return None

    # The two sequences are independent: no column/row contains both entries.
    if not np.all((outer_line == bg) | (inner_line == bg)):
        return None
    return outer, inner


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the sparse 2xL or Lx2 palette band on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        pal = _palette_mask(g.shape, side, start, end)
        for r, c in zip(*np.where(pal), strict=False):
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
        return True

    def consider(side: str, outer: np.ndarray, inner: np.ndarray, limit: int) -> None:
        nonlocal best
        for start in range(limit):
            for end in range(start + 3, limit + 1):
                seqs = _extract_sequences(outer[start:end], inner[start:end], bg)
                if seqs is None:
                    continue
                outer_seq, inner_seq = seqs
                if not isolated(side, start, end):
                    continue
                distinct = len(set(outer_seq) | set(inner_seq))
                span = end - start
                nonbg = len(outer_seq) + len(inner_seq)
                score = (len(outer_seq), distinct, -span, nonbg)
                cand = (score, side, start, end)
                if best is None or cand[0] > best[0]:
                    best = cand

    if h >= 2:
        consider("top", g[0, :], g[1, :], w)
        consider("bottom", g[h - 1, :], g[h - 2, :], w)
    if w >= 2:
        consider("left", g[:, 0], g[:, 1], h)
        consider("right", g[:, w - 1], g[:, w - 2], h)

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Returns (mapping border->fill, palette_mask).
    Mapping is read by matching the kth outer entry with the kth inner entry.
    """
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = _palette_mask((h, w), side, start, end)

    if side == "top":
        outer_line = g[0, start:end]
        inner_line = g[1, start:end]
    elif side == "bottom":
        outer_line = g[h - 1, start:end]
        inner_line = g[h - 2, start:end]
    elif side == "left":
        outer_line = g[start:end, 0]
        inner_line = g[start:end, 1]
    else:
        outer_line = g[start:end, w - 1]
        inner_line = g[start:end, w - 2]

    seqs = _extract_sequences(outer_line, inner_line, bg)
    assert seqs is not None, "_palette_mapping: palette must contain two valid sparse sequences"
    border_seq, fill_seq = seqs
    mapping = {int(b): int(f) for b, f in zip(border_seq, fill_seq, strict=False)}
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
    Fill hollow sprites' interiors using two ordered color sequences on a grid edge.

    Input structure:
    - bg = most common border color. Several hollow sprites use single-color borders and enclose bg holes.
    - One edge contains a 2xL or Lx2 sparse palette band: the outer line lists border colors, the inner line lists fill colors.
    - The two palette lines are staggered with bg gaps, so they form two independent ordered sequences rather than cell-by-cell pairs.

    Rule:
    - Locate the edge palette and read the outer non-bg colors in scan order as border keys.
    - Read the inner non-bg colors in the same scan order as fill values.
    - Match the kth border key to the kth fill value, then fill each enclosed hole according to its surrounding border color.

    Key insight: The palette encodes correspondence by ordinal position inside two separate sequences, not by immediate vertical or horizontal adjacency.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator (D1 variant) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

# Palette: reduced max N so border keys, fill values, and distractor can stay distinct.
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

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
    """Create hollow sprite (h x w) with at least 1 enclosed bg hole, cell count <= _SPRITE_CELL_MAX."""
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
    rng: random.Random, n: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick distinct border colors, fill colors, distractor color, and filled-sprite keys."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    needed = 2 * n + 1
    assert needed <= len(all_c), "_choose_colors requires enough distinct non-bg colors"
    picked = all_c[:needed]
    border_keys = picked[:n]
    fill_values = picked[n : 2 * n]
    distractor_color = picked[-1]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(
    side: str, border_keys: list[int], fill_values: list[int], rng: random.Random
) -> np.ndarray:
    """Construct an interleaved 2xL or Lx2 palette band for the chosen edge."""
    n = len(border_keys)
    span = 2 * n
    outer_first = rng.choice([True, False])
    outer_positions = list(range(0 if outer_first else 1, span, 2))
    inner_positions = list(range(1 if outer_first else 0, span, 2))

    if side in ("top", "bottom"):
        pal = np.zeros((2, span), dtype=int)
        outer_row, inner_row = (0, 1) if side == "top" else (1, 0)
        for idx, col in enumerate(outer_positions):
            pal[outer_row, col] = border_keys[idx]
        for idx, col in enumerate(inner_positions):
            pal[inner_row, col] = fill_values[idx]
    else:
        pal = np.zeros((span, 2), dtype=int)
        outer_col, inner_col = (0, 1) if side == "left" else (1, 0)
        for idx, row in enumerate(outer_positions):
            pal[row, outer_col] = border_keys[idx]
        for idx, row in enumerate(inner_positions):
            pal[row, inner_col] = fill_values[idx]
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

    edge_len = h if side in ("left", "right") else w
    max_n = min(_PALETTE_N_MAX, edge_len // 2)
    if max_n < _PALETTE_N_MIN:
        return None
    n = rng.randint(_PALETTE_N_MIN, max_n)

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    num_filled = rng.randint(2, min(_FILLED_SPRITE_MAX, n))
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

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    pal = _build_palette(side, border_keys, fill_values, rng)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            color = filled_keys[i]
            fill = mapping[color]
        else:
            color = distractor_color
            fill = None
        infos.append((mask, color, fill))

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
