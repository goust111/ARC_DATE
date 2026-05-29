# Combo variant of dbff022c (index 14)
# combo_spec: D1/4+D12/0
# Name: 位置索引映射 + 分隔调色板

"""
design_analysis:
  variant_core:
    original_rule: >
      原题把 palette 做成紧凑的 2xN / Nx2 色条，外侧格与内侧格在同一列/同行天然成对，
      所以 solver 只要逐格 zip 就能得到 border->fill 映射。
    fused_rule: >
      融合后把 palette 拆成两条独立序列：外侧线只放 border-key 序列，内侧线只放 fill-value 序列。
      每条序列内部用同一个 divider 颜色隔开，例如 top 侧可出现：
        row0: 2 5 8 5 4 0 0 0 0 0
        row1: 0 0 0 0 0 7 5 3 5 6
      这里 5 是分隔符，真正的序列是 border=[2,8,4]、fill=[7,3,6]，映射按第 1/2/3 个位置建立。
  behavior_trace:
    small_example: >
      若一个空心精灵边框色为 8，则原题会去找与 8 同列/同行的内侧颜色；在这个融合版里那样做会失败，
      因为 8 周围只直接邻接 divider 或背景。正确过程是：
      1. 找到边缘 palette 带；
      2. 在外侧线中解析出被 divider 分隔的条目 [2,8,4]；
      3. 在内侧线中解析出另一条被同一 divider 分隔的条目 [7,3,6]；
      4. 由 8 在 border 序列中的索引 2，得到 fill 序列第 2 项 3；
      5. 用 3 填充该精灵内部孔洞。
    difference_from_original: >
      原题依赖“同坐标配对”，D1/4 改成“同序号配对”，D12/0 再要求先识别 divider 才能读出序列。
      因此 solver 不能再把 palette 当作密集色条逐列/逐行读取。
  complexity_calibration:
    why_this_is_not_too_simple: >
      如果只把 palette 稀疏化成带 bg 缝隙的单像素序列，solver 只需过滤非 bg，D12/0 的分隔结构几乎不起作用。
      这里改用重复 divider 颜色，把“条目”和“分隔符”都显式画出来；如果不先解析分隔符，就会把 divider 当成普通映射色。
    controlled_degrees_of_freedom:
      - palette 仍只有一处，仍位于单一边缘，避免多图例歧义。
      - divider 颜色在两条序列中共用，肉眼可把它识别为“结构标记”而非映射值。
      - border 与 fill 序列长度限制为 2-3，保证颜色预算足够，同时每个样例里至少有 2 个可验证映射。
      - 至少 2 个 filled sprites + 1 个 distractor sprite，既能提供重复证据，也能保留“非 palette 边框色不填充”的对照。
  implicit_assumption_audit:
    solver_lines_affected:
      - "_best_palette 原先默认 palette 是稠密 block；现在必须改成识别两条各自只有一段编码序列的 edge band。"
      - "_palette_mapping 原先直接 zip 相邻行/列；现在要先在每条线中剥离 divider，提取条目序列，再按索引 zip。"
      - "_fill_enclosed_bg_regions 仍然依赖“孔洞周围主导边框色可唯一决定填充值”；这个前提保留。"
    new_assumptions_and_generator_guards:
      - "每条 palette 线恰有一段连续非 bg 编码串，格式为 value,divider,value,divider,...,value；generator 保证。"
      - "两条编码串使用同一个 divider，且彼此不重叠，中间至少留 1 格 bg；generator 保证。"
      - "divider 不会拿去当 sprite 颜色，也不会出现在 border/fill 序列里；generator 保证。"
      - "sprite 不接触网格边界，避免其它边缘非 bg 像素干扰 edge-band 识别；generator 保证。"
  placement_legality:
    palette_placement: >
      palette 仍按完整 2xL / Lx2 包围盒占位，并在 occupancy 中额外包 1 格 padding，
      防止 sprite 靠进 divider/gap 区域让视觉结构混乱。
    sprite_placement: >
      所有 sprite bbox 都被限制在非边界区域内，且彼此与 palette 至少隔 1 格背景。
      这样内部孔洞天然不会碰到网格边界，也不会和图例结构混叠。
  visual_observability:
    cues_for_humans:
      - "边缘带中有一眼可见的重复 divider 颜色，说明并非普通稠密色条。"
      - "外侧线和内侧线各自形成一串独立条目，中间还分成前后两个区段，提示要分别读取两条序列。"
      - "至少两个被成功填充的精灵提供重复证据，能看出“第几个边框色对应第几个填充值”。"
  cleanliness:
    layout: >
      palette 只占单一边缘的窄带，divider 颜色只服务于图例；主体区域仍是空心单色精灵，
      输出变化只发生在洞内，视觉层次清楚，不会出现同一区域叠加多种标记。
  inference_risk_check:
    anti_hidden_signal: >
      规则只依赖可见的顺序与分隔结构，不依赖颜色编号大小、频次排序或其他人类不可见属性。
      若误用原题的逐列配对，生成样本会得到不同映射，因此不会退化回原始行为。
"""

import random

import numpy as np

from arc_utils import components_mask, neighbors4


_PALETTE_GAP_CHOICES = (1,)
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3

# Grid dimensions: keep within the original envelope but bias upward so
# palette + multiple interior sprites fit without touching the boundary.
_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 11, 15

_SPRITE_H_MAX = 5
_SPRITE_W_MAX = 5
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 14
_FILLED_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 800


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_span(n: int, gap: int) -> int:
    return 2 * (2 * n - 1) + gap


def _palette_capacity(edge_len: int, gap: int) -> int:
    return min(_PALETTE_N_MAX, (edge_len - gap + 2) // 4)


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


def _extract_divided_sequence(line: np.ndarray, bg: int) -> tuple[int, int, list[int], int] | None:
    """
    Parse one separated palette sequence from a single line.

    Expected form within the line:
      bg* [value, divider, value, divider, ..., value] bg*
    """
    nonbg = np.flatnonzero(line != bg)
    if len(nonbg) == 0:
        return None
    if not np.all(np.diff(nonbg) == 1):
        return None

    start = int(nonbg[0])
    end = int(nonbg[-1]) + 1
    run = [int(x) for x in line[start:end].tolist()]
    if len(run) < 3 or len(run) % 2 == 0:
        return None

    divider = run[1]
    if divider == bg:
        return None

    values: list[int] = []
    for idx, val in enumerate(run):
        if idx % 2 == 0:
            if val == bg or val == divider:
                return None
            values.append(val)
        else:
            if val != divider:
                return None

    if len(values) < 2 or len(values) != len(set(values)):
        return None
    return start, end, values, divider


def _edge_lines(g: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = g.shape
    if side == "top":
        return g[0, :], g[1, :]
    if side == "bottom":
        return g[h - 1, :], g[h - 2, :]
    if side == "left":
        return g[:, 0], g[:, 1]
    return g[:, w - 1], g[:, w - 2]


def _palette_candidate(
    g: np.ndarray, bg: int, side: str
) -> tuple[tuple[int, int], dict[int, int], np.ndarray] | None:
    outer_line, inner_line = _edge_lines(g, side)
    union = np.flatnonzero((outer_line != bg) | (inner_line != bg))
    if len(union) == 0:
        return None

    start = int(union[0])
    end = int(union[-1]) + 1
    outer_info = _extract_divided_sequence(outer_line[start:end], bg)
    inner_info = _extract_divided_sequence(inner_line[start:end], bg)
    if outer_info is None or inner_info is None:
        return None

    o0, o1, border_seq, divider = outer_info
    i0, i1, fill_seq, divider2 = inner_info
    if divider != divider2:
        return None
    if len(border_seq) != len(fill_seq):
        return None

    abs_o0, abs_o1 = start + o0, start + o1
    abs_i0, abs_i1 = start + i0, start + i1
    if not (abs_o1 < abs_i0 or abs_i1 < abs_o0):
        return None

    pal_mask = _palette_mask(g.shape, side, start, end)
    h, w = g.shape
    for r, c in zip(*np.where(pal_mask), strict=False):
        v = int(g[int(r), int(c)])
        if v == bg:
            continue
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc] and int(g[rr, cc]) == v:
                return None

    mapping = {int(b): int(f) for b, f in zip(border_seq, fill_seq, strict=False)}
    score = (len(border_seq), -(end - start))
    return score, mapping, pal_mask


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Returns (mapping border->fill, palette_mask).
    Mapping is determined by matching the kth separated outer entry to the kth separated inner entry.
    """
    best = None
    for side in ("top", "bottom", "left", "right"):
        cand = _palette_candidate(g, bg, side)
        if cand is None:
            continue
        score, mapping, pal_mask = cand
        packed = (score, mapping, pal_mask)
        if best is None or score > best[0]:
            best = packed

    assert best is not None, "_palette_mapping: must find a valid separated palette"
    _, mapping, pal_mask = best
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
    Fill hollow sprites by reading two divider-separated color sequences on a grid edge.

    Input structure:
    - Background is the dominant border color, and several hollow single-color sprites sit away from the grid boundary.
    - One edge contains a 2xL or Lx2 palette band. The outer line encodes border colors; the inner line encodes fill colors.
    - Each line stores its colors as a separated sequence `value, divider, value, divider, ...`, and the two sequences appear in different positions within the band.

    Rule:
    - Locate the edge palette, parse the outer sequence by removing its repeated divider cells, and do the same for the inner sequence.
    - Use scan order within each sequence to pair the 1st outer color with the 1st inner color, the 2nd with the 2nd, and so on.
    - For every enclosed background hole whose surrounding border color appears in that index-based mapping, fill the hole with the mapped color.

    Key insight: The palette no longer provides local border/fill pairs. The repeated divider marks entry boundaries, and the true correspondence is indirect: it comes from matching positions across two separately read sequences.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


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
    rng: random.Random, n: int, num_filled: int
) -> tuple[list[int], list[int], int, int, dict[int, int], list[int]]:
    """Pick distinct border colors, fill colors, divider, distractor, and filled-sprite keys."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    needed = 2 * n + 2
    assert needed <= len(colors), "_choose_colors requires enough distinct non-bg colors"
    picked = colors[:needed]
    border_keys = picked[:n]
    fill_values = picked[n : 2 * n]
    divider = picked[2 * n]
    distractor_color = picked[2 * n + 1]
    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, divider, distractor_color, mapping, filled_keys


def _build_encoded_sequence(values: list[int], divider: int) -> np.ndarray:
    seq = np.empty(2 * len(values) - 1, dtype=int)
    seq[0::2] = np.array(values, dtype=int)
    seq[1::2] = divider
    return seq


def _build_palette(
    side: str,
    border_keys: list[int],
    fill_values: list[int],
    divider: int,
    gap: int,
    border_first: bool,
) -> np.ndarray:
    """Construct the separated 2xL or Lx2 palette band for the chosen edge."""
    border_seq = _build_encoded_sequence(border_keys, divider)
    fill_seq = _build_encoded_sequence(fill_values, divider)
    total_span = len(border_seq) + gap + len(fill_seq)

    if side in ("top", "bottom"):
        pal = np.zeros((2, total_span), dtype=int)
        outer_row, inner_row = (0, 1) if side == "top" else (1, 0)
        if border_first:
            pal[outer_row, : len(border_seq)] = border_seq
            pal[inner_row, len(border_seq) + gap : total_span] = fill_seq
        else:
            pal[inner_row, : len(fill_seq)] = fill_seq
            pal[outer_row, len(fill_seq) + gap : total_span] = border_seq
    else:
        pal = np.zeros((total_span, 2), dtype=int)
        outer_col, inner_col = (0, 1) if side == "left" else (1, 0)
        if border_first:
            pal[: len(border_seq), outer_col] = border_seq
            pal[len(border_seq) + gap : total_span, inner_col] = fill_seq
        else:
            pal[: len(fill_seq), inner_col] = fill_seq
            pal[len(fill_seq) + gap : total_span, outer_col] = border_seq
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    """Place palette on the chosen edge and mark its occupancy with 1-cell padding."""
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


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """Find all interior placements where the sprite fits with a 1-cell margin."""
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
    """Render a sprite onto the grid: border cells in color, hole cells in fill (if given)."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _direct_dense_mapping_from_palette(inp: np.ndarray, bg: int) -> dict[int, int]:
    """
    Approximate the original dense-pair reading on the detected palette side.
    Used only as a generation-time anti-degeneracy check.
    """
    best = None
    best_side = None
    for side in ("top", "bottom", "left", "right"):
        outer_line, inner_line = _edge_lines(inp, side)
        union = np.flatnonzero((outer_line != bg) | (inner_line != bg))
        if len(union) == 0:
            continue
        span = int(union[-1] - union[0] + 1)
        score = -span
        if best is None or score > best:
            best = score
            best_side = side
    if best_side is None:
        return {}

    outer_line, inner_line = _edge_lines(inp, best_side)
    union = np.flatnonzero((outer_line != bg) | (inner_line != bg))
    start, end = int(union[0]), int(union[-1]) + 1
    mapping: dict[int, int] = {}
    for b, f in zip(outer_line[start:end].tolist(), inner_line[start:end].tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f
    return mapping


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    feasible_sides: list[tuple[str, int]] = []
    for side in ("top", "bottom", "left", "right"):
        edge_len = w if side in ("top", "bottom") else h
        side_max = 0
        for gap in _PALETTE_GAP_CHOICES:
            side_max = max(side_max, _palette_capacity(edge_len, gap))
        if side_max >= _PALETTE_N_MIN:
            if side in ("left", "right") and h < 10:
                continue
            feasible_sides.append((side, side_max))
    if not feasible_sides:
        return None

    horiz = [item for item in feasible_sides if item[0] in ("top", "bottom")]
    vert = [item for item in feasible_sides if item[0] in ("left", "right")]
    if horiz and (not vert or rng.random() < 0.8):
        side, side_max_n = rng.choice(horiz)
    else:
        side, side_max_n = rng.choice(vert)

    gap_choices = [
        gap
        for gap in _PALETTE_GAP_CHOICES
        if _palette_capacity(w if side in ("top", "bottom") else h, gap) >= _PALETTE_N_MIN
    ]
    gap = rng.choice(gap_choices)
    max_n = min(side_max_n, _palette_capacity(w if side in ("top", "bottom") else h, gap))
    n = rng.randint(_PALETTE_N_MIN, max_n)

    num_filled = 2
    total_sprites = num_filled + 1

    max_sh = min(_SPRITE_H_MAX, h - 2)
    max_sw = min(_SPRITE_W_MAX, w - 2)
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

    border_keys, fill_values, divider, distractor_color, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    border_first = rng.choice([True, False])
    pal = _build_palette(side, border_keys, fill_values, divider, gap, border_first)

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
        positions = _find_valid_positions(h, w, occ, mask)
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

    # Anti-degeneracy check: a dense original-style read must disagree with the intended mapping.
    dense_mapping = _direct_dense_mapping_from_palette(inp, bg)
    if all(dense_mapping.get(key) == mapping[key] for key in filled_keys):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        r = _gen_attempt()
        if r is not None:
            return r
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
