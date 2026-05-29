# Combo variant of dbff022c (index 34)
# combo_spec: D1/4+D9/1
# Name: 位置索引映射 + 链式填充

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, neighbors4

# Design analysis for combo D1/4 + D9/1
# a) 具体化与行为追踪
#    - 原题：边缘 2-line palette 逐格配对；每个空心精灵独立处理，直接用自己的边框色查对应填充色。
#    - D1/4：palette 仍是 2xN / Nx2，但变成稀疏交错带。外侧线只给 key 序列，内侧线只给 fill 序列，
#      两条线按扫描顺序的第 1/2/3 个位置配对，而不是按同列/同行配对。
#    - D9/1：洞不再并行独立填。先按空间顺序排序；第一个洞用自身边框色起链，之后每个洞都用“前一个洞刚填出的颜色”
#      作为新的 lookup key。
#    - 融合例子：top palette 若是
#        row0: 2 0 8 0 4 0
#        row1: 0 8 0 4 0 2
#      则 index mapping 是 2->8, 8->4, 4->2。若三个精灵按左到右排列，边框色分别是 2, 9, 7：
#      第 1 个洞用 border 2，填成 8；第 2 个洞即使边框是 9，也不用 9，而用上一洞的 8，填成 4；
#      第 3 个洞再用上一洞的 4，填成 2。原题只能解释第一个精灵，后两个必须依赖链式传递。
# b) 复杂度校准
#    - palette 固定为 3 个 key，形成长度 3 的可见颜色链；fill 序列是 key 序列的非平凡循环置换，
#      因此前一步 fill 一定还能作为下一步 key。
#    - 只放 3 个带单洞的精灵，顺序清晰但不至于太短；后两个精灵的边框色强制不在 key 集内，
#      避免链式规则退化成“仍用当前边框直接查表”。
#    - 交错 palette 保证 D1/4 在视觉上可观察：若按原题逐格配对，outer 的 key 会只对到 bg，映射失效。
# c) 隐式假设审查
#    - 原 `_palette_mapping()` 假设外侧和内侧同坐标就是一对；融合后这个前提失效，必须先分别过滤非 bg 序列。
#    - 原 `_fill_enclosed_bg_regions()` 假设每个洞只依赖自己的相邻边框色；融合后只有第一个洞如此，
#      后续洞都依赖上一洞输出，所以不能再无序遍历组件，必须先收集并排序后串行推进。
#    - 原实现默认“若边框色不在 mapping 中，该洞就保持 bg”；融合后后两个精灵故意满足这一条件，
#      solver 只能把它们的边框色用于识别外壳，不能再把它们当作 lookup key。
# d) 放置合法性
#    - top/bottom palette 时，3 个精灵排成一条水平链；left/right palette 时排成垂直链。
#    - palette 与精灵固定隔 1 格背景；精灵之间也隔 1 格背景，避免组件粘连或排序歧义。
#    - 每个精灵恰有 1 个封闭洞，洞不触边，因此“内部 bg 连通块”识别稳定。
# e) 视觉可观察性
#    - 交错 palette 明确提示“两条线都是有序色序”，而不是逐格对齐的双色条带。
#    - 后两个精灵的边框色根本不在 key 集里，但输出仍被填色，这会强迫解题者注意到颜色是沿空间顺序传递的。
#    - 颜色传递完全由可见的前一步输出暴露，不依赖颜色编号大小、频次排序或其它不可见元信息。
# f) 可视化整洁度
#    - 画面只有一块边缘图例和一条精灵链，没有重叠标记层，也没有多洞对象互相干扰。
#    - 精灵都保持单色空心轮廓；输出只改洞内颜色，前后差异集中、清楚。
# g) 变化可推断性
#    - 样例中实际起作用的颜色种类很少：3 个 key、3 个循环 fill、2 个非 key 边框色。
#    - D1/4 给出“如何从 palette 读映射”，D9/1 给出“何时用哪个 key”；两者缺一不可，且都能从 2-3 组样例中观察到。
# h) 新 solver 的额外假设审查
#    - 新 solver 假设所有可处理洞经 row-major 排序后就是唯一处理链；generator 固定构造纯水平或纯垂直单链来保证。
#    - 新 solver 假设链的起点边框色在 key 集中，且每一步 fill 仍可继续查表；generator 通过循环置换 fill 序列保证。
#    - 新 solver 假设 palette 的 outer/inner 非 bg 计数相等、且同一列/行不会同时出现 outer 和 inner 颜色；
#      generator 用严格交错布局保证这一点。

_PALETTE_N = 3
_MAX_RETRIES = 500
_SHAPE_SPECS = [(3, 3), (3, 4), (4, 3), (4, 4)]
_P_CORNER_NOTCH = 0.35


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    """Boolean mask covering the full 2xL or Lx2 palette footprint."""
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


def _extract_sparse_sequences(
    outer_line: np.ndarray, inner_line: np.ndarray, bg: int
) -> tuple[list[int], list[int]] | None:
    """Read outer/inner non-bg entries as two independent ordered sequences."""
    outer = [int(x) for x in outer_line.tolist() if int(x) != bg]
    inner = [int(x) for x in inner_line.tolist() if int(x) != bg]
    if len(outer) < 2 or len(inner) < 2 or len(outer) != len(inner):
        return None
    if len(outer) != len(set(outer)) or len(inner) != len(set(inner)):
        return None
    if not np.all((outer_line == bg) | (inner_line == bg)):
        return None
    return outer, inner


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the sparse 2xL or Lx2 palette band on a grid edge.

    Returns (side, start, end) where:
      - side in {"top", "bottom", "left", "right"}
      - for top/bottom: start/end are column indices [start, end)
      - for left/right: start/end are row indices [start, end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        pal = _palette_mask(g.shape, side, start, end)
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

    def consider(side: str, outer: np.ndarray, inner: np.ndarray, limit: int) -> None:
        nonlocal best
        for start in range(limit):
            for end in range(start + 3, limit + 1):
                seqs = _extract_sparse_sequences(outer[start:end], inner[start:end], bg)
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


def _palette_lines(g: np.ndarray, side: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (outer_line, inner_line) in edge-inward order."""
    h, w = g.shape
    if side == "top":
        return g[0, start:end], g[1, start:end]
    if side == "bottom":
        return g[h - 1, start:end], g[h - 2, start:end]
    if side == "left":
        return g[start:end, 0], g[start:end, 1]
    return g[start:end, w - 1], g[start:end, w - 2]


def _palette_mapping(
    g: np.ndarray, bg: int, *, use_index: bool
) -> tuple[dict[int, int], np.ndarray]:
    """
    Returns (mapping, palette_mask).

    If use_index=True, pair the kth outer non-bg entry with the kth inner non-bg entry.
    If use_index=False, use the original same-coordinate pairing as an ablation.
    """
    side, start, end = _best_palette(g, bg)
    pal_mask = _palette_mask(g.shape, side, start, end)
    outer_line, inner_line = _palette_lines(g, side, start, end)

    if use_index:
        seqs = _extract_sparse_sequences(outer_line, inner_line, bg)
        assert seqs is not None, "_palette_mapping: palette must contain two valid sparse sequences"
        outer_seq, inner_seq = seqs
        mapping = {int(b): int(f) for b, f in zip(outer_seq, inner_seq, strict=False)}
        return mapping, pal_mask

    mapping: dict[int, int] = {}
    for b, f in zip(outer_line.tolist(), inner_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f
    return mapping, pal_mask


def _hole_infos(out: np.ndarray, bg: int, pal_mask: np.ndarray) -> list[dict]:
    """Collect enclosed holes together with their border color and row-major anchor."""
    h, w = out.shape
    infos: list[dict] = []
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
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = max(adj_counts.items(), key=lambda item: (item[1], -item[0]))[0]
        r0 = int(pts[:, 0].min())
        c0 = int(pts[:, 1].min())
        infos.append({"pts": pts, "border": int(border), "anchor": (r0, c0)})

    infos.sort(key=lambda info: info["anchor"])
    return infos


def _fill_ordered_holes(
    out: np.ndarray, mapping: dict[int, int], hole_infos: list[dict], *, use_chain: bool
) -> None:
    """Fill holes in row-major order, optionally carrying the previous fill forward."""
    prev_fill: int | None = None
    for info in hole_infos:
        border = int(info["border"])
        lookup_key = border if (not use_chain or prev_fill is None) else int(prev_fill)
        if lookup_key not in mapping:
            if use_chain:
                prev_fill = None
            continue
        fill = int(mapping[lookup_key])
        out[tuple(info["pts"].T)] = fill
        if use_chain:
            prev_fill = fill


def _solve_modes(grid: np.ndarray, *, use_chain: bool, use_index: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg, use_index=use_index)
    assert mapping, "_solve_modes: palette mapping must be non-empty"

    out = g.copy()
    infos = _hole_infos(out, bg, pal_mask)
    _fill_ordered_holes(out, mapping, infos, use_chain=use_chain)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading a staggered edge palette and passing colors along a spatial chain.

    Input structure:
    - The background is the dominant outer-border color.
    - One grid edge contains a 2xL or Lx2 sparse palette band. The outer line shows border-key colors and the inner line shows
      fill colors, but the two lines are staggered with bg gaps so they form two separate ordered sequences.
    - Several single-color hollow sprites each contain one enclosed bg hole and are arranged in a clear left-to-right or top-to-bottom chain.

    Rule:
    - Locate the palette, read the outer non-bg colors in scan order as the key sequence, and read the inner non-bg colors in scan order as
      the value sequence. Match them by ordinal position.
    - Find the enclosed holes, sort them by spatial position, and process them in that order.
    - The first hole uses its own surrounding border color as the lookup key. Every later hole ignores its own border color and instead uses
      the previous hole's fill color as the next lookup key.

    Key insight:
    - The palette is not read by local adjacency, and the sprites are not solved independently. The answer comes from combining two visible
      structures: index-based pairing inside the staggered legend, then chain-based color propagation across the ordered sprites.
    """
    return _solve_modes(grid, use_chain=True, use_index=True)


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components that stay away from the sprite boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_ring(h: int, w: int) -> np.ndarray:
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False
    return sprite


def _make_chain_sprite(h: int, w: int, rng: random.Random) -> np.ndarray:
    sprite = _make_ring(h, w)
    if rng.random() < _P_CORNER_NOTCH:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        cr, cc = rng.choice(corners)
        test = sprite.copy()
        test[cr, cc] = False
        if _get_enclosed_holes(test):
            sprite = test
    assert len(_get_enclosed_holes(sprite)) == 1, "_make_chain_sprite must keep exactly one hole"
    return sprite


def _sample_shapes(side: str, rng: random.Random) -> list[np.ndarray]:
    """
    Sample 3 ring-like sprites while keeping the chain axis compact enough for the original grid limits.
    """
    chain_vertical = side in ("left", "right")
    for _ in range(200):
        dims = [rng.choice(_SHAPE_SPECS) for _ in range(3)]
        axis_sum = sum(h if chain_vertical else w for h, w in dims)
        if chain_vertical:
            if axis_sum > 9:
                continue
        else:
            if axis_sum > 11:
                continue
        return [_make_chain_sprite(h, w, rng) for h, w in dims]
    raise RuntimeError("_sample_shapes: failed to sample compact shapes")


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
        return pal

    pal = np.zeros((span, 2), dtype=int)
    outer_col, inner_col = (0, 1) if side == "left" else (1, 0)
    for idx, row in enumerate(outer_positions):
        pal[row, outer_col] = border_keys[idx]
    for idx, row in enumerate(inner_positions):
        pal[row, inner_col] = fill_values[idx]
    return pal


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _chain_fill_sequence(first_key: int, mapping: dict[int, int], length: int) -> list[int]:
    fills: list[int] = []
    key = int(first_key)
    for _ in range(length):
        fill = int(mapping[key])
        fills.append(fill)
        key = fill
    return fills


def _layout_horizontal(
    side: str, shapes: list[np.ndarray], rng: random.Random
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]]:
    widths = [int(shape.shape[1]) for shape in shapes]
    heights = [int(shape.shape[0]) for shape in shapes]
    total_w = int(sum(widths) + 2)
    max_h = max(heights)

    w_lo = max(10, total_w + 2)
    h_lo = max(9, max_h + 5)
    if w_lo > 15 or h_lo > 13:
        raise RuntimeError("_layout_horizontal: shapes exceed target grid bounds")

    h = rng.randint(h_lo, 13)
    w = rng.randint(w_lo, 15)

    if side == "top":
        row = rng.randint(3, h - max_h - 1)
        pal_r0 = 0
    else:
        row = rng.randint(1, h - max_h - 4)
        pal_r0 = h - 2

    col = rng.randint(1, w - total_w - 1)
    positions = []
    cur_c = col
    for shape in shapes:
        positions.append((row, cur_c))
        cur_c += int(shape.shape[1] + 1)

    pal_c0 = rng.randint(0, w - 2 * _PALETTE_N)
    return h, w, positions, (pal_r0, pal_c0)


def _layout_vertical(
    side: str, shapes: list[np.ndarray], rng: random.Random
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]]:
    widths = [int(shape.shape[1]) for shape in shapes]
    heights = [int(shape.shape[0]) for shape in shapes]
    total_h = int(sum(heights) + 2)
    max_w = max(widths)

    h_lo = max(10, total_h + 2)
    w_lo = max(10, max_w + 5)
    if h_lo > 13 or w_lo > 15:
        raise RuntimeError("_layout_vertical: shapes exceed target grid bounds")

    h = rng.randint(h_lo, 13)
    w = rng.randint(w_lo, 15)

    row = rng.randint(1, h - total_h - 1)
    if side == "left":
        col = rng.randint(3, w - max_w - 1)
        pal_c0 = 0
    else:
        col = rng.randint(1, w - max_w - 4)
        pal_c0 = w - 2

    positions = []
    cur_r = row
    for shape in shapes:
        positions.append((cur_r, col))
        cur_r += int(shape.shape[0] + 1)

    pal_r0 = rng.randint(0, h - 2 * _PALETTE_N)
    return h, w, positions, (pal_r0, pal_c0)


def _choose_colors(
    rng: random.Random,
) -> tuple[list[int], list[int], dict[int, int], list[int]]:
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    keys = all_c[:_PALETTE_N]
    shift = rng.choice([1, 2])
    fills = keys[shift:] + keys[:shift]
    mapping = dict(zip(keys, fills, strict=False))
    distractors = all_c[_PALETTE_N : _PALETTE_N + 2]
    return keys, fills, mapping, distractors


def _boundary_bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()]) == int(bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    side = rng.choice(["top", "bottom", "left", "right"])

    try:
        shapes = _sample_shapes(side, rng)
        if side in ("top", "bottom"):
            h, w, positions, pal_pos = _layout_horizontal(side, shapes, rng)
        else:
            h, w, positions, pal_pos = _layout_vertical(side, shapes, rng)
    except RuntimeError:
        return None

    keys, fills, mapping, distractors = _choose_colors(rng)
    palette = _build_palette(side, keys, fills, rng)
    first_key = keys[0]
    fill_chain = _chain_fill_sequence(first_key, mapping, len(shapes))
    sprite_borders = [first_key, distractors[0], distractors[1]]

    output = np.full((h, w), bg, dtype=int)
    pr0, pc0 = pal_pos
    output[pr0 : pr0 + palette.shape[0], pc0 : pc0 + palette.shape[1]] = palette

    placed: list[tuple[int, int, np.ndarray, list[np.ndarray]]] = []
    for (r0, c0), shape, border, fill in zip(positions, shapes, sprite_borders, fill_chain, strict=False):
        holes = _get_enclosed_holes(shape)
        _paint_sprite(output, int(r0), int(c0), shape, int(border), int(fill), holes)
        placed.append((int(r0), int(c0), shape, holes))

    inp = output.copy()
    for r0, c0, _shape, holes in placed:
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

    if not _boundary_bg_is_dominant(inp, bg):
        return None

    try:
        combo = main(inp)
        no_chain = _solve_modes(inp, use_chain=False, use_index=True)
        no_index = _solve_modes(inp, use_chain=True, use_index=False)
    except Exception:
        return None

    if not np.array_equal(combo, output):
        return None
    if np.array_equal(combo, no_chain):
        return None
    if np.array_equal(combo, no_index):
        return None

    keyset = set(keys)
    if sum(int(border not in keyset) for border in sprite_borders[1:]) != 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input for dbff022c combo D1/4+D9/1")
