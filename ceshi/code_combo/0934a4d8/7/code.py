# Combo variant of 0934a4d8 (index 7)
# combo_spec: D1/2+D6/2
# Name: 对角线对称+行列不同偏移

from __future__ import annotations

import numpy as np

from arc_utils import bbox


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪:
#    - 原题在同一个 lo 上同时做 LR/UD 反射，遮挡格会沿四元轨道恢复；D1 单变体则改成
#      整张棋盘的单一对角镜像，每个遮挡格只需看它的对角伙伴。
#    - 本融合版不做“整张棋盘全局对角对称”，而是只要求右下激活块
#      D(lo_row, lo_col) = {(r, c) | r >= lo_row and c >= lo_col}
#      满足主对角转置约束 g[r, c] == g[c, r]。因为 lo_row != lo_col，转置后的伙伴
#      可能落到顶边或左边的非对称边界带里，这正是融合点。
#    - 小例子: 若 lo_row=2, lo_col=4，则格子 (2, 9) 位于激活块中，因为 2>=2 且 9>=4。
#      它的对角伙伴是 (9, 2)，后者落在左边界带（列 2 < 4）。原 D1 全局对角版会把整张
#      图都当成对角对称；本融合版只承诺这种“至少一端在激活块里”的成对关系。原题的
#      LR/UD solver 更完全不适用，因为这里没有水平/垂直反射轨道。
#    - 对遮挡恢复而言，若标记盖住 (2, 9)，solver 直接读可见的 (9, 2)。若标记盖住
#      左边界带里的 (9, 2)，也仍可通过其伙伴 (2, 9) 恢复。这使得“内区映到边界”
#      真正参与输出，而不是只停留在背景外观上。
# b) 复杂度校准:
#    - lo_row / lo_col 只从 {2, 3, 4} 选，并进一步固定为 lo_row < lo_col。原因是主对角
#      转置约束只让“较小起点 / 较大起点”的无序组合可观测；若同时采样交换后的顺序，会
#      产生人眼与 solver 都无法区分的隐性二义性。
#    - 在这个有序子族里，左右边界厚度仍然明显不同，自由度也仍然很小，避免样例过散。
#    - 仍保留 30x30 画布、颜色集合、实心矩形标记和 output-first 生成方式，让变化集中在
#      对称几何本身。
#    - generator 强制标记矩形至少覆盖一个“恰有一端落在激活块”的对角对，避免退化成
#      单纯的全局对角遮挡题。
# c) 隐式假设审查:
#    - 原 `_find_reflected_strip_start()` 假设存在同一个 lo 控制 LR 与 UD；融合后彻底失效，
#      必须改成寻找能解释“右下激活块转置约束”的 `(lo_row, lo_col)`。
#    - D1 单变体的 `_diagonal_modes()` 假设整张图都服从主/副对角镜像；融合后不再成立，
#      因为顶边和左边保留一块不受约束的自由区域。
#    - `main()` 若简单对所有标记格读取 `g[c, r]`，隐含前提是该格与转置格确实属于受约束
#      的同一对角类，且转置格不被标记遮住。generator 需要保证:
#      1. 每个标记格 `(r, c)` 至少有一端 `(r, c)` 或 `(c, r)` 落在激活块里；
#      2. 标记矩形与其转置矩形不相交，因此转置伙伴始终可见。
# d) 放置合法性:
#    - 标记仍是任意位置的实心矩形，但必须满足:
#      1. 不与自己的转置矩形相交，否则会出现“隐藏格映到另一个隐藏格”；
#      2. 矩形内每个格都可恢复，即 `(r, c)` 和 `(c, r)` 这对里至少有一端属于激活块；
#      3. 至少有一个格是跨边界恢复格，即正向在激活块而转置落到边界，或反之。
# e) 视觉可观察性:
#    - 解题者能直接看到主对角成对关系，但并不是整张图都对称: 左上角附近存在明显的
#      非对称自由区；同时 top/left 的可约束区域宽度不同，对应 lo_row != lo_col。
#    - 标记矩形会切到“内区 <-> 边界”的成对带，因此从 2-4 个例子里能观察到: 有些输出像素
#      并非从对角线另一侧的内部大块恢复，而是从边界条带恢复。
# f) 可视化整洁度:
#    - 图中对象仍只有一个纯色标记矩形和若干颜色纹理区，没有叠加多层对象。
#    - 受约束区域由大量转置配对组成，自由区域只保留在左上角附近的 L 形区域，布局稳定。
# g) 变化的可推断性:
#    - 不引入颜色置换；颜色只是通过转置配对被复制。人类需要识别的是“哪些位置与哪些
#      位置成对”，不是学习新的颜色映射表。
# h) 反退化策略:
#    - generator 只接受能被 `_find_diagonal_offsets()` 唯一回推到真实 `(lo_row, lo_col)`
#      的样本，防止更小偏移也碰巧成立。
#    - 另外统计两个指标验证融合确实发生:
#      1. `cross_boundary_hidden_count > 0`，表示输出里确实用到了“内区映边界”的恢复；
#      2. `global_visible_mismatches > 0`，表示若把整张图错当成全局主对角对称，会在可见区
#         直接发现矛盾，因此该任务没有退化成纯 D1。
# i) main() 的新假设:
#    - 输入仍是正方形网格。
#    - 仍存在且只存在一个纯色实心矩形标记。
#    - 最小可行的 `(lo_row, lo_col)`（且 lo_row < lo_col）唯一。
#    - 每个标记格的转置伙伴可见且与其值相同；generator 用放置约束和求解回验保证这一点。


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


def _diagonal_consistent(g: np.ndarray, marker: int, lo_row: int, lo_col: int) -> bool:
    n, m = map(int, g.shape)
    assert n == m
    for r in range(lo_row, n):
        for c in range(lo_col, n):
            a = int(g[r, c])
            b = int(g[c, r])
            if a != marker and b != marker and a != b:
                return False
    return True


def _find_diagonal_offsets(g: np.ndarray, marker: int) -> tuple[int, int]:
    n, m = map(int, g.shape)
    assert n == m
    for lo_row in range(n):
        for lo_col in range(lo_row + 1, n):
            if lo_row == lo_col:
                continue
            if _diagonal_consistent(g, marker, lo_row, lo_col):
                return lo_row, lo_col
    raise AssertionError("no diagonal offset pair found")


def _pair_active(r: int, c: int, lo_row: int, lo_col: int) -> bool:
    return (r >= lo_row and c >= lo_col) or (c >= lo_row and r >= lo_col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the marker-covered rectangle using a main-diagonal symmetry that only starts after separate row/column offsets.

    Input structure:
    - The input is a square color grid with one solid rectangular marker hiding the answer patch.
    - There is one active lower-right block determined by `lo_row` and `lo_col`, and every cell in that block matches its transpose partner across the main diagonal.
    - Because `lo_row != lo_col`, some cells in that active block transpose into the visible top or left boundary strips, while a smaller left-upper L-shaped region stays unconstrained.

    Rule:
    - Locate the unique color whose full footprint is a filled rectangle; its bounding box is the output window.
    - Infer the smallest `(lo_row, lo_col)` with `lo_row < lo_col` for which all visible cells in rows `lo_row..` and cols `lo_col..` agree with their transposed counterparts.
    - For each marker cell `(r, c)`, copy the value from `(c, r)`; the generator guarantees that every hidden cell belongs to one of these valid transpose-linked pairs and that the partner is visible.

    Key insight:
    This is not a global diagonal mirror. The hidden rectangle sits inside a grid where only an offset lower-right block enforces transpose equality, so solving requires noticing both the main-diagonal pairing and the unequal row/column starts that push some recoverable partners onto the border.
    """
    g = np.asarray(grid, dtype=int)
    n, m = map(int, g.shape)
    assert n == m

    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo_row, lo_col = _find_diagonal_offsets(g, marker)
    assert lo_row < lo_col

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            assert _pair_active(r, c, lo_row, lo_col)
            value = int(g[c, r])
            assert value != marker
            out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO_CHOICES = (2, 3, 4)
_LO_PAIRS = [(lo_row, lo_col) for lo_row in _LO_CHOICES for lo_col in _LO_CHOICES if lo_row < lo_col]

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 500

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int, int, int], list[tuple[int, int, int]]] = {}
_LO_PAIR_CURSOR = 0


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _transpose_rect(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return c0, r0, c0 + w, r0 + h


def _cell_in_active_block(r: int, c: int, lo_row: int, lo_col: int) -> bool:
    return r >= lo_row and c >= lo_col


def _cross_boundary_pair(r: int, c: int, lo_row: int, lo_col: int) -> bool:
    forward = _cell_in_active_block(r, c, lo_row, lo_col)
    backward = _cell_in_active_block(c, r, lo_row, lo_col)
    return forward != backward


def _placement_cross_boundary_count(r0: int, c0: int, h: int, w: int, lo_row: int, lo_col: int) -> int:
    count = 0
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if _cross_boundary_pair(r, c, lo_row, lo_col):
                count += 1
    return count


def _valid_placements(lo_row: int, lo_col: int, h: int, w: int) -> list[tuple[int, int, int]]:
    key = (lo_row, lo_col, h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                rect = (r0, c0, r0 + h, c0 + w)
                if _rect_overlap(rect, _transpose_rect(r0, c0, h, w)):
                    continue
                recoverable = True
                for r in range(r0, r0 + h):
                    for c in range(c0, c0 + w):
                        if not _pair_active(r, c, lo_row, lo_col):
                            recoverable = False
                            break
                    if not recoverable:
                        break
                if not recoverable:
                    continue
                cross_boundary = _placement_cross_boundary_count(r0, c0, h, w, lo_row, lo_col)
                if cross_boundary == 0:
                    continue
                placements.append((r0, c0, cross_boundary))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean(lo_row: int, lo_col: int) -> np.ndarray:
    clean = _RNG.choice(_PALETTE, size=(_N, _N))
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for r in range(lo_row, _N):
        for c in range(lo_col, _N):
            a = (int(r), int(c))
            b = (int(c), int(r))
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            color = int(_RNG.choice(_PALETTE))
            clean[a] = color
            clean[b] = color

    return clean


def _global_visible_mismatches(g: np.ndarray, marker: int) -> int:
    n = int(g.shape[0])
    mismatches = 0
    for r in range(n):
        for c in range(n):
            a = int(g[r, c])
            b = int(g[c, r])
            if a != marker and b != marker and a != b:
                mismatches += 1
    return mismatches


def _next_lo_pair() -> tuple[int, int]:
    global _LO_PAIR_CURSOR
    pair = _LO_PAIRS[_LO_PAIR_CURSOR % len(_LO_PAIRS)]
    _LO_PAIR_CURSOR += 1
    return pair


def _gen_attempt_for_pair(lo_row: int, lo_col: int) -> tuple[np.ndarray, dict[str, int]] | None:
    clean = _build_clean(lo_row, lo_col)
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(lo_row, lo_col, h, w)
    if not placements:
        return None
    r0, c0, cross_boundary = placements[int(_RNG.integers(len(placements)))]

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        found_row, found_col = _find_diagonal_offsets(inp, _MARKER)
        if (found_row, found_col) != (lo_row, lo_col):
            return None
        if np.array_equal(main(inp), expected):
            metrics = {
                "lo_row": lo_row,
                "lo_col": lo_col,
                "cross_boundary_hidden_count": cross_boundary,
                "global_visible_mismatches": _global_visible_mismatches(inp, _MARKER),
            }
            if metrics["global_visible_mismatches"] == 0:
                return None
            return inp, metrics
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    lo_row, lo_col = _next_lo_pair()
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt_for_pair(lo_row, lo_col)
        if result is not None:
            grid, _metrics = result
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
