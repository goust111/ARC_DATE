# Combo variant of dbff022c (index 11)
# combo_spec: D1/3+D9/1
# Name: 条件映射（按面积分组） + 链式填充

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, neighbors4

# Design analysis for combo D1/3 + D9/1
# a) 具体化与行为追踪
#    - 原题：边缘只有 1 组 2-line palette；每个空心精灵独立处理，直接用“自己的边框色”查表填洞。
#    - D1/3：palette 变成 4xN / Nx4，两组 key 完全相同；大精灵用靠外的第一组，小精灵用第二组。
#    - D9/1：精灵按空间位置顺序处理；第一个精灵仍用自己的边框色起链，之后每个精灵都用“前一个洞刚填出的颜色”
#      作为下一次查表 key，而不是再看自己的边框色。
#    - 融合例子：若 key 行是 [2,4,6]，large 行给 [4,6,2]，small 行给 [6,2,4]。三个精灵按阅读顺序排列。
#      第 1 个 large 边框色是 2，所以填成 4；第 2 个 small 即使边框色是 9，也不用 9，而是用上一洞的 4，
#      因而填成 2；第 3 个 large 再用上一洞的 2，填成 4。和原题相比，后两个精灵的边框色不再决定答案。
# b) 复杂度校准
#    - 只保留两档面积：small 使用 3x3 / 3x4 / 4x3 环，large 使用 4x5 / 5x4 环，面积带有明确间隔。
#    - palette 固定 3 列，便于形成长度 3 的颜色传递链，同时颜色种类仍然可控。
#    - 两个 fill 行都限制为 key 集的置换，这样“前一步 fill 可作为下一步 key”在视觉上和逻辑上都成立。
#    - 生成器强制至少一个 large 和一个 small，并强制后续精灵边框色不在 key 集里，避免退化成原题。
# c) 隐式假设审查
#    - 原 `_best_palette` 只接受 2-line legend；这里必须扫描 4-line legend，并保持 edge-inward 的行序。
#    - 原 `_palette_mapping` 只返回单映射；这里需要 large/small 两张表。
#    - 原 `_fill_enclosed_bg_regions` 假设每个洞的 lookup key 来自自身边框色；融合后只有第一个洞满足，
#      后续洞的 key 来自上一步 fill，所以不能再并行独立填充，必须先排序再串行推进。
#    - 原实现默认“边框色必须在 mapping 中”；融合后后续精灵的边框色可以完全不在 key 集里，因此 solver
#      只能把边框色用于识别外壳和测面积，不能把它当作后续查表的必要条件。
# d) 放置合法性
#    - top/bottom palette 时把三个精灵摆成同一行的水平链；left/right palette 时摆成同一列的垂直链。
#    - palette 与精灵之间固定保留 1 格背景 moat；精灵彼此之间也保留 1 格背景，避免组件粘连。
#    - 每个精灵都只有 1 个封闭洞，且洞不接触外边界，因此 solver 的“内部 bg 连通块”识别稳定。
# e) 视觉可观察性
#    - 双 palette 明确展示“同一 key 有两套答案”，large/small 由形状尺寸直接可见。
#    - 读取顺序由明显的空间链给出：横向样例按左到右，纵向样例按上到下。无需依赖颜色 id 或隐藏排序。
#    - 后续精灵边框色故意不出现在 key 行里，但输出仍被填色；这迫使解题者注意到颜色传递链。
# f) 可视化整洁度
#    - 对象布局只有一条主链和一块边缘图例，没有重叠层，也没有多个洞互相干扰。
#    - palette 只占 4x3 / 3x4，小而规则；三个环形精灵尺寸差异清楚，画面不会杂乱。
# g) 变化可推断性
#    - 颜色变化并非无提示发生：key 行和两条 fill 行给出结构化图例，面积档位给出第二个控制变量，
#      顺序链则由前一步输出颜色直接暴露。2-3 个样例足以让人观察到“看前一个填充色，而不是看当前边框色”。
# h) 新 solver 的额外假设审查
#    - 新 solver 假设所有可处理洞按 row-major 排序后形成单链；生成器固定构造水平或垂直单链保证这一点。
#    - 新 solver 用全局 min/max area 的中点区分大小档；生成器只使用两组相隔明显的模板，避免中间值歧义。
#    - 新 solver 假设第一个精灵的边框色在 key 集内、后续链传递颜色也都在 key 集内；生成器通过
#      “fill 行是 key 集的置换”以及首精灵 border=key 来保证。

_PALETTE_N = 3
_SMALL_SPECS = [(3, 3), (3, 4), (4, 3)]
_LARGE_SPECS = [(4, 5), (5, 4)]
_ROLE_PATTERNS = [
    ("large", "small", "large"),
    ("small", "large", "small"),
]
_P_CORNER_NOTCH = 0.35
_MAX_RETRIES = 500


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    """Boolean mask covering the full 4xN / Nx4 palette footprint."""
    h, w = shape
    pal_mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        pal_mask[0:4, start:end] = True
    elif side == "bottom":
        pal_mask[h - 4 : h, start:end] = True
    elif side == "left":
        pal_mask[start:end, 0:4] = True
    else:
        pal_mask[start:end, w - 4 : w] = True
    return pal_mask


def _palette_lines(g: np.ndarray, side: str, start: int, end: int) -> list[np.ndarray]:
    """Return the palette's four lines in edge-inward order."""
    h, w = g.shape
    if side == "top":
        return [g[0, start:end], g[1, start:end], g[2, start:end], g[3, start:end]]
    if side == "bottom":
        return [g[h - 1, start:end], g[h - 2, start:end], g[h - 3, start:end], g[h - 4, start:end]]
    if side == "left":
        return [g[start:end, 0], g[start:end, 1], g[start:end, 2], g[start:end, 3]]
    return [g[start:end, w - 1], g[start:end, w - 2], g[start:end, w - 3], g[start:end, w - 4]]


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 4xN or Nx4 palette block on a grid edge.

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

    def valid_candidate(side: str, start: int, end: int) -> tuple[bool, tuple[int, int, int, int] | None]:
        lines = _palette_lines(g, side, start, end)
        block = np.stack(lines, axis=0)
        if np.any(block == bg):
            return False, None

        keys_a = [int(x) for x in lines[0].tolist()]
        keys_b = [int(x) for x in lines[2].tolist()]
        fills_a = [int(x) for x in lines[1].tolist()]
        fills_b = [int(x) for x in lines[3].tolist()]

        if len(keys_a) < 2:
            return False, None
        if len(keys_a) != len(set(keys_a)):
            return False, None
        if keys_a != keys_b:
            return False, None
        if len(set(fills_a)) != len(fills_a):
            return False, None
        if len(set(fills_b)) != len(fills_b):
            return False, None
        if not any(a != b for a, b in zip(fills_a, fills_b, strict=False)):
            return False, None
        if not isolated(side, start, end):
            return False, None

        distinct = len(set(keys_a) | set(fills_a) | set(fills_b))
        score = (len(keys_a), distinct, -(end - start), int(block.size))
        return True, score

    def consider_top() -> None:
        nonlocal best
        if h < 4:
            return
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                ok, score = valid_candidate("top", c0, c1)
                if ok:
                    cand = (score, "top", c0, c1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 4:
            return
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                ok, score = valid_candidate("bottom", c0, c1)
                if ok:
                    cand = (score, "bottom", c0, c1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 4:
            return
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                ok, score = valid_candidate("left", r0, r1)
                if ok:
                    cand = (score, "left", r0, r1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 4:
            return
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                ok, score = valid_candidate("right", r0, r1)
                if ok:
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


def _palette_mappings(g: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    """Return (large_mapping, small_mapping, palette_mask)."""
    side, start, end = _best_palette(g, bg)
    pal_mask = _palette_mask(g.shape, side, start, end)
    lines = _palette_lines(g, side, start, end)

    large_keys = [int(x) for x in lines[0].tolist()]
    large_fills = [int(x) for x in lines[1].tolist()]
    small_keys = [int(x) for x in lines[2].tolist()]
    small_fills = [int(x) for x in lines[3].tolist()]

    assert large_keys == small_keys, "_palette_mappings: both groups must use the same keys"
    large_map = dict(zip(large_keys, large_fills, strict=False))
    small_map = dict(zip(small_keys, small_fills, strict=False))
    return large_map, small_map, pal_mask


def _component_area_from_seed(out: np.ndarray, pal_mask: np.ndarray, seed: tuple[int, int]) -> int:
    """Area of the monochrome non-palette component containing seed."""
    h, w = out.shape
    sr, sc = seed
    color = int(out[sr, sc])
    stack = [(int(sr), int(sc))]
    seen = {(int(sr), int(sc))}
    area = 0

    while stack:
        r, c = stack.pop()
        area += 1
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen:
                if pal_mask[rr, cc]:
                    continue
                if int(out[rr, cc]) != color:
                    continue
                seen.add((rr, cc))
                stack.append((rr, cc))
    return area


def _hole_infos(out: np.ndarray, bg: int, pal_mask: np.ndarray) -> list[dict]:
    """Collect enclosed holes together with their surrounding border color, area, and sort anchor."""
    h, w = out.shape
    infos: list[dict] = []
    bg_mask = (out == bg) & (~pal_mask)

    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        adj_seed: dict[int, tuple[int, int]] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
                    adj_seed.setdefault(v, (int(rr), int(cc)))
        if not adj_counts:
            continue

        border = max(adj_counts.items(), key=lambda item: (item[1], -item[0]))[0]
        area = _component_area_from_seed(out, pal_mask, adj_seed[border])
        r0 = int(pts[:, 0].min())
        c0 = int(pts[:, 1].min())
        infos.append(
            {
                "pts": pts,
                "border": int(border),
                "area": int(area),
                "anchor": (r0, c0),
            }
        )

    infos.sort(key=lambda info: info["anchor"])
    return infos


def _fill_ordered_holes(
    out: np.ndarray,
    large_map: dict[int, int],
    small_map: dict[int, int],
    hole_infos: list[dict],
    *,
    use_chain: bool,
    use_area: bool,
) -> None:
    """Shared engine for combo behavior and ablations used by the generator."""
    if not hole_infos:
        return

    areas = [int(info["area"]) for info in hole_infos]
    small_area = min(areas)
    large_area = max(areas)
    pivot = (small_area + large_area) / 2.0

    prev_fill: int | None = None
    for info in hole_infos:
        border = int(info["border"])
        area = int(info["area"])
        lookup_key = border if (not use_chain or prev_fill is None) else int(prev_fill)

        if use_area and large_area != small_area:
            mapping = large_map if area > pivot else small_map
        else:
            mapping = large_map

        if lookup_key not in mapping:
            prev_fill = None if use_chain else prev_fill
            continue

        fill = int(mapping[lookup_key])
        pts = info["pts"]
        out[tuple(pts.T)] = fill
        if use_chain:
            prev_fill = fill


def _solve_modes(grid: np.ndarray, *, use_chain: bool, use_area: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    large_map, small_map, pal_mask = _palette_mappings(g, bg)
    assert large_map and small_map, "_solve_modes: palette mappings must be non-empty"

    out = g.copy()
    infos = _hole_infos(out, bg, pal_mask)
    _fill_ordered_holes(out, large_map, small_map, infos, use_chain=use_chain, use_area=use_area)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites from a doubled edge legend, but carry the previous fill color forward as the next lookup key.

    Input structure:
    - The background is the dominant border color.
    - One 4xN or Nx4 palette block sits on a grid edge and consists of two consecutive 2-line legends read from the edge inward.
      The first legend is for larger sprites, the second legend is for smaller sprites, and both legends share the same key colors.
    - Several hollow mono-color sprites contain enclosed background holes and are arranged in a clear spatial sequence.

    Rule:
    - Detect the palette and recover two border-key -> fill mappings: one for large sprites and one for small sprites.
    - Find every enclosed background hole, sort the holes by spatial position, and process them in that order.
    - The first hole uses its own surrounding border color as the lookup key. Each later hole instead uses the fill color produced for the previous hole.
    - At each step, decide between the large or small mapping by the enclosing sprite's area, then fill the hole with that mapped color.

    Key insight:
    - Solving the task requires combining two dependencies at once: size decides which palette group is active, while order decides which color key is active.
      After the first sprite, the current border color is no longer the color being looked up.
    """
    return _solve_modes(grid, use_chain=True, use_area=True)


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


def _make_band_sprite(role: str, rng: random.Random) -> np.ndarray:
    specs = _LARGE_SPECS if role == "large" else _SMALL_SPECS
    h, w = rng.choice(specs)
    sprite = _make_ring(h, w)

    if rng.random() < _P_CORNER_NOTCH:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        cr, cc = rng.choice(corners)
        test = sprite.copy()
        test[cr, cc] = False
        if _get_enclosed_holes(test):
            sprite = test

    assert _get_enclosed_holes(sprite), "_make_band_sprite must preserve an enclosed hole"
    return sprite


def _build_palette(side: str, border_keys: list[int], large_fills: list[int], small_fills: list[int]) -> np.ndarray:
    edge_lines = [
        np.array(border_keys, dtype=int),
        np.array(large_fills, dtype=int),
        np.array(border_keys, dtype=int),
        np.array(small_fills, dtype=int),
    ]
    n = len(border_keys)

    if side == "top":
        pal = np.zeros((4, n), dtype=int)
        for i, line in enumerate(edge_lines):
            pal[i] = line
        return pal
    if side == "bottom":
        pal = np.zeros((4, n), dtype=int)
        for i, line in enumerate(edge_lines[::-1]):
            pal[i] = line
        return pal
    if side == "left":
        pal = np.zeros((n, 4), dtype=int)
        for i, line in enumerate(edge_lines):
            pal[:, i] = line
        return pal

    pal = np.zeros((n, 4), dtype=int)
    for i, line in enumerate(edge_lines[::-1]):
        pal[:, i] = line
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


def _role_fill_sequence(
    roles: tuple[str, ...],
    first_border: int,
    large_map: dict[int, int],
    small_map: dict[int, int],
) -> list[int]:
    fills = []
    key = int(first_border)
    for role in roles:
        mapping = large_map if role == "large" else small_map
        fill = int(mapping[key])
        fills.append(fill)
        key = fill
    return fills


def _layout_horizontal(
    side: str,
    shapes: list[np.ndarray],
    rng: random.Random,
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]]:
    widths = [int(shape.shape[1]) for shape in shapes]
    heights = [int(shape.shape[0]) for shape in shapes]
    total_w = int(sum(widths) + (len(shapes) - 1))
    h = max(10, int(max(heights) + 6 + rng.randint(0, 2)))
    w = max(10, int(total_w + 2 + rng.randint(0, 2)))

    if side == "top":
        row = int(rng.randint(5, h - max(heights) - 1))
        pal_r0 = 0
    else:
        row = int(rng.randint(1, h - 5 - max(heights)))
        pal_r0 = h - 4

    col = int(rng.randint(1, w - total_w - 1))
    positions = []
    cur_c = col
    for shape in shapes:
        positions.append((row, cur_c))
        cur_c += int(shape.shape[1] + 1)

    pal_c0 = int(rng.randint(0, w - _PALETTE_N))
    return h, w, positions, (pal_r0, pal_c0)


def _layout_vertical(
    side: str,
    shapes: list[np.ndarray],
    rng: random.Random,
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]]:
    widths = [int(shape.shape[1]) for shape in shapes]
    heights = [int(shape.shape[0]) for shape in shapes]
    total_h = int(sum(heights) + (len(shapes) - 1))
    h = max(10, int(total_h + 2 + rng.randint(0, 2)))
    w = max(10, int(max(widths) + 6 + rng.randint(0, 2)))

    if side == "left":
        col = int(rng.randint(5, w - max(widths) - 1))
        pal_c0 = 0
    else:
        col = int(rng.randint(1, w - 5 - max(widths)))
        pal_c0 = w - 4

    row = int(rng.randint(1, h - total_h - 1))
    positions = []
    cur_r = row
    for shape in shapes:
        positions.append((cur_r, col))
        cur_r += int(shape.shape[0] + 1)

    pal_r0 = int(rng.randint(0, h - _PALETTE_N))
    return h, w, positions, (pal_r0, pal_c0)


def _choose_colors(
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int], tuple[str, ...]]:
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    keys = all_c[:_PALETTE_N]
    nonkeys = all_c[_PALETTE_N : _PALETTE_N + 2]

    large_fills = keys[1:] + keys[:1]
    small_fills = keys[-1:] + keys[:-1]
    if rng.random() < 0.5:
        large_fills, small_fills = small_fills, large_fills

    roles = rng.choice(_ROLE_PATTERNS)
    return keys, large_fills, small_fills, nonkeys, roles


def _boundary_bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()]) == int(bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    side = rng.choice(["top", "bottom", "left", "right"])
    keys, large_fills, small_fills, nonkeys, roles = _choose_colors(rng)
    large_map = dict(zip(keys, large_fills, strict=False))
    small_map = dict(zip(keys, small_fills, strict=False))

    shapes = [_make_band_sprite(role, rng) for role in roles]
    if side in ("top", "bottom"):
        h, w, positions, pal_pos = _layout_horizontal(side, shapes, rng)
    else:
        h, w, positions, pal_pos = _layout_vertical(side, shapes, rng)

    fills = _role_fill_sequence(roles, keys[0], large_map, small_map)
    sprite_borders = [keys[0], nonkeys[0], nonkeys[1]]
    palette = _build_palette(side, keys, large_fills, small_fills)

    output = np.full((h, w), bg, dtype=int)
    pr0, pc0 = pal_pos
    output[pr0 : pr0 + palette.shape[0], pc0 : pc0 + palette.shape[1]] = palette

    placed: list[tuple[int, int, np.ndarray, list[np.ndarray]]] = []
    for (r0, c0), shape, border, fill in zip(positions, shapes, sprite_borders, fills, strict=False):
        holes = _get_enclosed_holes(shape)
        _paint_sprite(output, int(r0), int(c0), shape, int(border), int(fill), holes)
        placed.append((int(r0), int(c0), shape, holes))

    inp = output.copy()
    for r0, c0, shape, holes in placed:
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

    if not _boundary_bg_is_dominant(inp, bg):
        return None

    try:
        combo = main(inp)
        area_only = _solve_modes(inp, use_chain=False, use_area=True)
        chain_only = _solve_modes(inp, use_chain=True, use_area=False)
    except Exception:
        return None

    if not np.array_equal(combo, output):
        return None
    if np.array_equal(combo, area_only):
        return None
    if np.array_equal(combo, chain_only):
        return None

    keyset = set(keys)
    if sum(int(border not in keyset) for border in sprite_borders[1:]) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input for dbff022c combo D1/3+D9/1")
