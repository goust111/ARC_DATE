# Variant of 0934a4d8
# Dimension: D6 — 对称轴偏移量（lo 参数） (meta)
# Change: lo=2，固定值；前2行/列为自由边界，不参与主体对称约束
#         → 行方向和列方向有不同的 lo 值，形成非对称自由边界

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis:
#   a) 具体化与行为追踪:
#      - 原题把 [lo..N-1] 同时当作“可左右镜像的列带”和“可上下镜像的行带”，因此
#        顶边和左边的自由边界宽度完全一样，都是 lo。
#      - 本变体改成 lo_row != lo_col。于是输入中的自由区域变成一个 lo_row x lo_col
#        的左上角矩形，顶边高度和左边宽度不同；主体对称块仍然覆盖右下大区域，但
#        行镜像只从 lo_row 开始，列镜像只从 lo_col 开始。
#      - 小例子追踪:
#        原题若 lo=2，格子 (4,7) 的反射轨道会用行轴 2..29 和列轴 2..29；本变体若
#        lo_row=2, lo_col=4，则 (4,7) 的上下镜像仍围绕 2..29，而左右镜像围绕
#        4..29。原 solver 把两者绑成同一个 lo，会把列镜像位置算错。
#   b) 复杂度校准:
#      - lo_row / lo_col 只从 {2,3,4} 中取值且强制不相等，避免自由度过大，同时让
#        边界差异在 2-4 个样例内肉眼可见。
#      - 标记矩形尺寸、颜色集合、输出形式全部保持原题，避免把变化稀释到无关维度。
#   c) 隐式假设审查:
#      - `_find_reflected_strip_start()` 依赖“行列共享同一 lo”，变体下必须拆成两个独立
#        的推断函数。
#      - `_reflection_orbit_values()` 依赖“同一 lo 同时控制行镜像和列镜像”，必须改成
#        lo_row / lo_col 双参数。
#      - `main()` 的转置回退原先默认 `g[c, r]` 总在对应自由边界里；变体下只有当
#        转置落点仍位于顶边或左边自由区时，这个回退才符合题意，因此 generator 要
#        保证此条件。
#   d) 放置合法性:
#      - 标记仍可放在任意位置，但若某个被遮挡格子的整条反射轨道都被标记完全覆盖，
#        则其转置格 (c,r) 必须不在标记内，且必须落在可见自由边界（`c < lo_row`
#        或 `r < lo_col`）中；否则该放置无效。
#   e) 视觉可观察性:
#      - 解题者能直接看到“上方自由边界高度”和“左侧自由边界宽度”不同，这是本变体
#        的主要可见信号。
#      - 行列镜像轴不同会改变右下主体块的对称中心，相同输入不能再用单一 lo 解释。
#   f) 可视化整洁度:
#      - 仍保留一个纯色实心矩形标记，其他区域维持原任务的清晰分区：右下主体块、
#        顶边条带、左边条带、左上自由角。
#   g) 颜色变化可推断性:
#      - 不引入新的颜色映射；颜色仅作为被复制的像素值，推理核心仍是空间对称。
#   h) 反退化策略:
#      - generator 只接受能被新 solver 推断回真实 `(lo_row, lo_col)` 的样本，防止
#        偶然出现“更小 lo 也成立”的退化情况。
#      - 统计时还会检查 lo_row != lo_col 始终成立，且旧版“单 lo”推断无法解释大多数
#        样本，从而确认变体真的生效。
#   i) main() 的新假设:
#      - 输入仍是正方形网格。
#      - 仍存在且只存在一个纯色实心矩形标记。
#      - 所有标记格子的反射轨道要么能在可见格子里得到唯一颜色，要么能安全使用
#        转置回退；generator 会按这两个条件筛选放置。


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


def _find_col_reflected_strip_start(g: np.ndarray, marker: int) -> int:
    m = int(g.shape[1])
    hi = m - 1
    for lo_col in range(m):
        strip = g[:, lo_col : hi + 1]
        strip_reflected = strip[:, ::-1]
        known = (strip != marker) & (strip_reflected != marker)
        if not bool(np.any((strip != strip_reflected) & known)):
            return lo_col
    raise AssertionError("no column reflection strip found")


def _find_row_reflected_strip_start(g: np.ndarray, marker: int) -> int:
    n = int(g.shape[0])
    hi = n - 1
    for lo_row in range(n):
        strip = g[lo_row : hi + 1, :]
        strip_reflected = strip[::-1, :]
        known = (strip != marker) & (strip_reflected != marker)
        if not bool(np.any((strip != strip_reflected) & known)):
            return lo_row
    raise AssertionError("no row reflection strip found")


def _find_reflected_strip_starts(g: np.ndarray, marker: int) -> tuple[int, int]:
    n, m = map(int, g.shape)
    assert n == m
    return _find_row_reflected_strip_start(g, marker), _find_col_reflected_strip_start(g, marker)


def _reflection_orbit_values(
    g: np.ndarray, r: int, c: int, lo_row: int, lo_col: int, marker: int
) -> set[int]:
    n, m = map(int, g.shape)
    hi_row = n - 1
    hi_col = m - 1
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

        if lo_col <= cc <= hi_col:
            q.append((rr, lo_col + hi_col - cc))
        if lo_row <= rr <= hi_row:
            q.append((lo_row + hi_row - rr, cc))

    return values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the contents hidden by the solid rectangle using separate row and column mirror bands.

    Input structure:
    - A square color grid contains one solid rectangular marker covering the answer patch.
    - Rows [lo_row..N-1] obey up/down reflection, while columns [lo_col..N-1] obey left/right reflection.
    - Because lo_row and lo_col differ, the free top strip and free left strip have different thicknesses.

    Rule:
    - Detect the unique solid-rectangle color and treat its bounding box as the output window.
    - Infer lo_row from the smallest lower strip with vertical mirror consistency, and infer lo_col from the smallest right strip with horizontal mirror consistency, always ignoring marker pixels.
    - For each covered cell, read the unique non-marker value in its row/column reflection orbit; if the whole orbit is hidden, use the visible transposed cell at (c, r).

    Key insight:
    The marker only hides data. Even with asymmetric free borders, every missing cell is still determined either by the two independent reflection axes or by a transpose clue exposed in the unmatched border area.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo_row, lo_col = _find_reflected_strip_starts(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _reflection_orbit_values(g, r, c, lo_row, lo_col, marker)
            if values:
                assert len(values) == 1
                out[rr, cc] = values.pop()
            else:
                assert c < lo_row or r < lo_col
                assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
                value = int(g[c, r])
                assert value != marker
                out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_HI = _N - 1
_LO_CHOICES = (2, 3, 4)
_LO_PAIRS = [(lo_row, lo_col) for lo_row in _LO_CHOICES for lo_col in _LO_CHOICES if lo_row != lo_col]

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
_PLACEMENT_CACHE: dict[tuple[int, int, int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}


def _axis_seed_len(lo: int) -> int:
    return (_N - lo + 1) // 2


def _canonical_axis(i: int, lo: int) -> int:
    return min(i, lo + _HI - i) - lo


def _orbit(r: int, c: int, lo_row: int, lo_col: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if lo_row <= r <= _HI:
        rows.add(lo_row + _HI - r)
    if lo_col <= c <= _HI:
        cols.add(lo_col + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(r0: int, c0: int, h: int, w: int, lo_row: int, lo_col: int) -> list[tuple[int, int]] | None:
    """
    Find marker cells whose entire reflection orbit is hidden and therefore need transpose fallback.

    Valid fallback requires:
    - the transpose cell (c, r) is outside the marker, and
    - that transpose cell still lies in the visible free top/left border (`c < lo_row` or `r < lo_col`).
    """
    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            orbit = _orbit(r, c, lo_row, lo_col)
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in orbit):
                if c >= lo_row and r >= lo_col:
                    return None
                if r0 <= c < r1 and c0 <= r < c1:
                    return None
                cells.append((r, c))
    return cells


def _valid_placements(lo_row: int, lo_col: int, h: int, w: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = (lo_row, lo_col, h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                fallback = _fallback_cells(r0, c0, h, w, lo_row, lo_col)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


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
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        color = int(_RNG.choice(_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _build_clean(lo_row: int, lo_col: int) -> np.ndarray:
    clean = np.empty((_N, _N), dtype=int)

    row_seed = _axis_seed_len(lo_row)
    col_seed = _axis_seed_len(lo_col)

    base = _RNG.choice(_PALETTE, size=(row_seed, col_seed))
    for r in range(lo_row, _N):
        rr = _canonical_axis(r, lo_row)
        for c in range(lo_col, _N):
            cc = _canonical_axis(c, lo_col)
            clean[r, c] = int(base[rr, cc])

    for r in range(lo_row):
        vals = _RNG.choice(_PALETTE, size=col_seed)
        for c in range(lo_col, _N):
            clean[r, c] = int(vals[_canonical_axis(c, lo_col)])

    for c in range(lo_col):
        vals = _RNG.choice(_PALETTE, size=row_seed)
        for r in range(lo_row, _N):
            clean[r, c] = int(vals[_canonical_axis(r, lo_row)])

    clean[:lo_row, :lo_col] = _RNG.choice(_PALETTE, size=(lo_row, lo_col))
    return clean


def _gen_attempt() -> tuple[np.ndarray, tuple[int, int]] | None:
    lo_row, lo_col = _LO_PAIRS[int(_RNG.integers(len(_LO_PAIRS)))]
    clean = _build_clean(lo_row, lo_col)
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(lo_row, lo_col, h, w)
    if not placements:
        return None
    r0, c0, fallback = placements[int(_RNG.integers(len(placements)))]

    groups = [_orbit(r, c, lo_row, lo_col) | _orbit(c, r, lo_row, lo_col) for r, c in fallback]
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        found_row, found_col = _find_reflected_strip_starts(inp, _MARKER)
        if (found_row, found_col) != (lo_row, lo_col):
            return None
        if np.array_equal(main(inp), expected):
            return inp, (lo_row, lo_col)
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _offsets = result
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 D6 input")
