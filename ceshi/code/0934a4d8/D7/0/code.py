# Variant of 0934a4d8
# Dimension: D7 — 网格形状 (meta)
# Change: 固定 30x30 正方形网格... → 非正方形矩形网格

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis
# a) 具体化与行为追踪
#    - 变体后的输入不再是 30x30，而是 18x38 / 20x36 / 36x20 / 38x18 之一。
#      这些尺寸都保留了原题的“下方大条带 + 右方大条带”双轴镜像，但行轴和列轴
#      分别按各自维度定义：行反射使用 [lo_r..H-1]，列反射使用 [lo_c..W-1]。
#    - 标记仍是唯一的实心矩形，输出仍是该矩形下方被遮挡的真实子图。
#    - 原题里某个被遮挡格若整个 LR+UD 轨道都落在标记中，就直接读 g[c, r]。
#      这依赖 H == W。现在若 H < W，顶部条带里靠近横向中心的格子在交换坐标后会落到
#      行边界外；变体把它改成“先交换，再把越界坐标按短边边界镜面折回”。
#      例：18x38 网格中，(0, 20) 的纯转置是 (20, 0) 越界，新逻辑读折返后的 (14, 0)。
#      这正是矩形版与正方形版的关键差异。
# b) 复杂度校准
#    - 仅修改网格形状相关机制，颜色集、唯一矩形标记、输出目标都保留。
#    - 为了让 D7 变化可观察，尺寸只在少数强非正方形档位中取值，并强制每题至少出现
#      一个“折返 transpose”格子；否则视觉上容易退化成原题的普通转置。
# c) 隐式假设审查
#    - 原 `_find_reflected_strip_start` 假设一条 lo 同时适用于行/列，且 n == m。
#      变体中必须拆成独立的 `lo_r` 与 `lo_c`。
#    - 原 `_reflection_orbit_values` 用同一个 `n` 同时做行/列上界；矩形下会把较长维
#      的坐标截错，因此必须分别用 `H-1` 和 `W-1`。
#    - 原 fallback `g[c, r]` 也默认转置后仍在网格内；矩形下该前提不成立，必须显式
#      做边界折返。
# d) 放置合法性
#    - 生成器只接受这样的标记放置：若某个标记格的完整反射轨道都被盖住，则它必须位于
#      顶部外带或左侧外带中，并且它的折返 transpose 参考格在网格内且不在标记里。
#    - 这样 solver 在“轨道恢复失败 -> fallback”这条路径上不会撞到隐藏或越界格。
# e) 视觉可观察性
#    - 输入始终是明显的长方形，不再是正方形；同时标记会压到外带并跨到镜像中心附近，
#      从而让矩形专属的折返 transpose 真正参与恢复。
# f) 可视化整洁度
#    - 仍然只有一块矩形标记，没有引入新对象层；主图案继续是规则的双轴镜像纹理，
#      视觉上和原题同风格，但长宽比明显不同。
# g) 颜色变化可推断性
#    - 不引入新的颜色映射。fallback 只是从另一个已可见位置取同色，颜色规律仍由镜像
#      结构支撑，不依赖不可见排序或颜色编号。


def _solid_rectangle_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append((area, int(color), b))

    assert len(candidates) == 1
    _area, color, b = candidates[0]
    return color, b


def _find_reflected_strip_starts(g: np.ndarray, marker: int) -> tuple[int, int]:
    h, w = map(int, g.shape)
    lo_r = -1
    lo_c = -1

    for cand in range(h // 2 + 1):
        strip = g[cand:h, :]
        strip_reflected = strip[::-1, :]
        known = (strip != marker) & (strip_reflected != marker)
        if not bool(np.any((strip != strip_reflected) & known)):
            lo_r = cand
            break

    for cand in range(w // 2 + 1):
        strip = g[:, cand:w]
        strip_reflected = strip[:, ::-1]
        known = (strip != marker) & (strip_reflected != marker)
        if not bool(np.any((strip != strip_reflected) & known)):
            lo_c = cand
            break

    assert lo_r >= 0 and lo_c >= 0
    return lo_r, lo_c


def _reflection_orbit_values(
    g: np.ndarray, r: int, c: int, lo_r: int, lo_c: int, marker: int
) -> set[int]:
    h, w = map(int, g.shape)
    hi_r = h - 1
    hi_c = w - 1
    seen: set[tuple[int, int]] = set()
    values: set[int] = set()
    q: deque[tuple[int, int]] = deque([(int(r), int(c))])

    while q:
        rr, cc = q.popleft()
        if (rr, cc) in seen:
            continue
        seen.add((rr, cc))

        value = int(g[rr, cc])
        if value != marker:
            values.add(value)

        if lo_c <= cc <= hi_c:
            q.append((rr, lo_c + hi_c - cc))
        if lo_r <= rr <= hi_r:
            q.append((lo_r + hi_r - rr, cc))

    return values


def _reflect_into_range(idx: int, length: int) -> int:
    idx = int(idx)
    length = int(length)
    assert idx >= 0 and length >= 1
    if length == 1:
        return 0
    period = 2 * length - 2
    folded = idx % period
    return folded if folded < length else period - folded


def _transpose_fallback_coord(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    return _reflect_into_range(c, h), _reflect_into_range(r, w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rectangle from a non-square mirrored grid using two independent reflection axes.

    Input structure:
    - A rectangular MxN color grid with M != N and one solid rectangular marker color hiding a patch.
    - Rows [lo_r..M-1] obey up/down reflection, while columns [lo_c..N-1] obey left/right reflection;
      the two strip starts are inferred independently because the canvas is not square.
    - Cells in the top and left outer bands may provide transpose-style witnesses for marker cells whose
      full reflection orbit is also hidden.

    Rule:
    - Locate the unique color that forms a completely filled rectangle, and treat that rectangle as the missing patch.
    - Find the smallest valid reflected row strip and reflected column strip separately, ignoring marker pixels when
      checking mirror consistency.
    - For each hidden cell, take the unique visible value in its LR+UD reflection orbit; if the orbit has no visible
      member, swap coordinates and fold any out-of-bounds index back across the shorter side's boundary before reading
      the fallback witness.

    Key insight:
    The clean pattern is still determined by bilateral reflection, but on a rectangle the row axis and column axis are
    no longer interchangeable; when plain transpose would leave the grid, the missing value is encoded by a boundary-
    reflected transpose copy instead.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo_r, lo_c = _find_reflected_strip_starts(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _reflection_orbit_values(g, r, c, lo_r, lo_c, marker)
            if values:
                assert len(values) == 1
                out[rr, cc] = values.pop()
            else:
                assert r < lo_r or c < lo_c
                fr, fc = _transpose_fallback_coord(g.shape, r, c)
                value = int(g[fr, fc])
                assert value != marker
                out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_LO_R = 2
_LO_C = 2
_SHAPES = [
    (18, 38),
    (20, 36),
    (36, 20),
    (38, 18),
]

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[
    tuple[tuple[int, int], int, int],
    list[tuple[int, int, list[tuple[int, int]]]],
] = {}
_SIZE_CACHE: dict[tuple[int, int], list[tuple[int, int]]] = {}


def _orbit(shape: tuple[int, int], r: int, c: int) -> set[tuple[int, int]]:
    h, w = shape
    rows = {int(r)}
    cols = {int(c)}
    if _LO_R <= r < h:
        rows.add(_LO_R + h - 1 - r)
    if _LO_C <= c < w:
        cols.add(_LO_C + w - 1 - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(
    shape: tuple[int, int], r0: int, c0: int, h: int, w: int
) -> list[tuple[int, int]] | None:
    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    adjusted = 0
    for r in range(r0, r1):
        for c in range(c0, c1):
            orb = _orbit(shape, r, c)
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in orb):
                if r >= _LO_R and c >= _LO_C:
                    return None
                fr, fc = _transpose_fallback_coord(shape, r, c)
                if r0 <= fr < r1 and c0 <= fc < c1:
                    return None
                cells.append((r, c))
                if (fr, fc) != (c, r):
                    adjusted += 1

    if not cells or adjusted == 0:
        return None
    return cells


def _valid_placements(shape: tuple[int, int], h: int, w: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = (shape, h, w)
    if key not in _PLACEMENT_CACHE:
        grid_h, grid_w = shape
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(grid_h - h + 1):
            for c0 in range(grid_w - w + 1):
                fallback = _fallback_cells(shape, r0, c0, h, w)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _eligible_size_pairs(shape: tuple[int, int]) -> list[tuple[int, int]]:
    if shape not in _SIZE_CACHE:
        _SIZE_CACHE[shape] = [(h, w) for h, w in _SIZE_PAIRS if _valid_placements(shape, h, w)]
    return _SIZE_CACHE[shape]


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]]) -> None:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groups:
        coords = list(group)
        if not coords:
            continue
        for coord in coords:
            parent.setdefault(coord, coord)
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        color = int(_RNG.choice(_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _build_clean(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    inner_h = h - _LO_R
    inner_w = w - _LO_C
    assert inner_h % 2 == 0 and inner_w % 2 == 0
    half_h = inner_h // 2
    half_w = inner_w // 2
    clean = np.empty((h, w), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half_h, half_w))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO_R:h, _LO_C:w] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO_R):
        vals = _RNG.choice(_PALETTE, size=half_w)
        clean[r, _LO_C : _LO_C + half_w] = vals
        clean[r, _LO_C + half_w : w] = vals[::-1]

    for c in range(_LO_C):
        vals = _RNG.choice(_PALETTE, size=half_h)
        clean[_LO_R : _LO_R + half_h, c] = vals
        clean[_LO_R + half_h : h, c] = vals[::-1]

    clean[:_LO_R, :_LO_C] = _RNG.choice(_PALETTE, size=(_LO_R, _LO_C))
    return clean


def _gen_attempt() -> np.ndarray | None:
    shape = _SHAPES[int(_RNG.integers(len(_SHAPES)))]
    clean = _build_clean(shape)

    size_pairs = _eligible_size_pairs(shape)
    if not size_pairs:
        return None
    h, w = size_pairs[int(_RNG.integers(len(size_pairs)))]

    placements = _valid_placements(shape, h, w)
    if not placements:
        return None
    r0, c0, fallback = placements[int(_RNG.integers(len(placements)))]

    groups = []
    for r, c in fallback:
        fr, fc = _transpose_fallback_coord(shape, r, c)
        groups.append(_orbit(shape, r, c) | _orbit(shape, fr, fc))
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangular 0934a4d8 input")
