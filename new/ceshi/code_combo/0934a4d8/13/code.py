# Combo variant of 0934a4d8 (index 13)
# combo_spec: D8/0+D9/2
# Name: 频率统计识别标记+标记色可变

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import bbox


# Design analysis for the fused variant
#
# a) Concrete behavior and trace:
# - 原题先在整张图里找“唯一一个自身外接框被完全填满的颜色”，因此即便 marker 颜色换了，
#   只要它仍是唯一实心矩形，旧推理路径也能工作。
# - 本融合版要求先做频率统计：出现次数最多的颜色就是 marker，然后再把该颜色的整块矩形
#   当作遮挡区域。恢复机制本身仍沿用原题的 LR+UD 反射轨道，必要时再用转置回退。
# - 具体小例子：若输入里颜色 3 以 12x11=132 次出现，其他颜色最多只出现 117 次，
#   那么 solver 必须先锁定 3，而不是去遍历“哪个颜色像矩形”。之后再对 3 的 bbox
#   中每个格子做轨道恢复；若某格子的反射同轨道全被遮住，仍回到 g[c, r] 取值。
# - 与原题差异在于“哪一块是 marker”的识别入口完全改了：先计数，再定位该色矩形。
#
# b) Complexity calibration:
# - 若仍保留原题 16-36 面积的小矩形，那么在 30x30 画布上它几乎不可能是全局最高频颜色，
#   D8/0 会退化成假规则。因此生成器把 marker 扩大到 120-168 格，使“最高频颜色 = marker”
#   在数值上可实现且肉眼可见。
# - 仅让 marker 变大还不够，因为旧版“唯一实心矩形检测”仍可能直接成功。为此生成器固定在
#   左上角放一个 2x2 诱饵实心方块，故输入里至少有两个实心矩形颜色，形状启发式不再唯一。
# - 其余可见像素只在 7 种 bulk 颜色间分配，并对每种颜色的可见计数设上限 `< marker_area`，
#   从而最高频条件稳定成立，同时颜色数量仍足够少，样例视觉不会过乱。
#
# c) Hidden assumptions broken by the fusion:
# - 原 `_solid_rectangle_marker` 假设“唯一实心矩形颜色”存在；融合后这不再成立，必须改成
#   “唯一最高频颜色”，再仅对该颜色做矩形完整性校验。
# - 原生成器默认 marker 固定为 8，且 clean 网格只用其它 8 色。融合后 marker 每例随机，
#   所以 palette 必须改成 {1..9} 中任选一个作 marker，剩余颜色承担底图与诱饵。
# - 原随机对称底图不控制频次；融合后若继续这样做，某个普通颜色可能比 marker 更常见。
#   因此生成器需要按“可见像素配额”而不是简单独立采样来着色。
# - 原 solver 的反射恢复与转置回退逻辑本身仍成立，但它新增了“最高频颜色的支撑必须恰好是
#   一个实心矩形，且最高频唯一”的输入假设；生成器必须显式保证。
#
# d) Placement legality:
# - marker 仍是单个实心矩形，放置时不能越界。
# - 为了保住左上角 2x2 诱饵矩形的独立性，marker 禁止覆盖同时满足 `r < lo` 且 `c < lo`
#   的 corner 区域；这样转置回退也不会把 corner 绑定进 marker 轨道。
# - 对所有会触发“整条反射轨道都被 marker 吃掉”的格子，仍沿用原题合法性检查：这类格子
#   必须位于 top/left 外带，且其转置位置不能也落在 marker 内。
#
# e) Visual observability:
# - 人类从样例中能直接看到一个明显更大的纯色矩形，且该颜色在整张图里出现最多。
# - 左上角始终还有一个小 2x2 纯色实心块，提示“不能只靠找唯一实心矩形”。
# - marker 颜色跨样例变化；真正稳定的线索是“它既最大频，又盖住一块需要按对称恢复的区域”。
#
# f) Visual cleanliness:
# - 诱饵只占左上角 2x2，不和主对称纹理叠层，也不参与恢复来源，噪声很低。
# - 其余区域只有一层双轴对称纹理与一个大矩形 marker，结构关系仍然清晰。
#
# g) Color-rule inferability:
# - 颜色 id 本身不编码角色；marker 可以是任意 1..9 中的一色。
# - 可推断的规则是“统计频率最高的颜色”，这是人类可见的全局属性，不依赖 0/黑色等不可见约定。
#
# Post-change audit of main() assumptions:
# - `main()` 假设最高频颜色唯一；generator 对 bulk 颜色的可见配额做上限控制，并在产物上复验。
# - `main()` 假设最高频颜色只出现在一个实心矩形里；clean 从不使用 marker 色，输入覆盖后复验其 bbox。
# - `main()` 假设反射一致偏移仍是 `lo=2`；generator 仍按原题对称结构造 clean，并复验检测结果。
# - `main()` 假设 corner 诱饵不会被转置回退引用；placement 过滤掉覆盖左上 2x2 corner 的矩形。


def _solid_rectangle_candidates(g: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], int]]:
    candidates: list[tuple[int, tuple[int, int, int, int], int]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append((int(color), b, area))
    return candidates


def _most_frequent_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    colors, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    marker_colors = [int(color) for color, count in zip(colors, counts) if int(count) == max_count]
    assert len(marker_colors) == 1

    marker = marker_colors[0]
    mask = g == marker
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    assert area > 1
    assert int(mask.sum()) == area
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
    return marker, b


def _find_reflected_strip_start(g: np.ndarray, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    for lo in range(n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        return lo

    raise AssertionError("no reflected strip found")


def _reflection_orbit_values(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> set[int]:
    n = int(g.shape[0])
    hi = n - 1
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

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the hidden rectangle by first identifying the marker through color frequency, then using the original reflection logic.

    Input structure:
    - A square colored grid contains one solid rectangular marker, but the marker color is not fixed across examples.
    - The marker color is the unique most frequent color in the whole input.
    - Away from that rectangle, the underlying clean pattern still obeys the original LR+UD reflection structure, with transpose fallback available on the top/left outer strips.

    Rule:
    - Count color frequencies and select the unique highest-frequency color as the marker color.
    - Take the full solid rectangle occupied by that color as the hidden output window, and infer the smallest reflection-strip offset `lo` exactly as in the base task, ignoring marker pixels.
    - For each hidden cell, copy the unique non-marker value from its horizontal/vertical reflection orbit; if the whole orbit is covered, use the transposed fallback cell `g[c, r]`.

    Key insight:
    The geometric recovery rule is unchanged, but the way to locate the occluder is no longer “find the unique filled rectangle”. The decisive clue is global color frequency: the marker is the color that dominates the input.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _most_frequent_marker(g)
    lo = _find_reflected_strip_start(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _reflection_orbit_values(g, r, c, lo, marker)
            if values:
                assert len(values) == 1
                out[rr, cc] = values.pop()
            else:
                assert r < lo or c < lo
                assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
                value = int(g[c, r])
                assert value != marker
                out[rr, cc] = value

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1
_CORNER_CELLS = frozenset((r, c) for r in range(_LO) for c in range(_LO))

# --- CFG: colors ---
_ALL_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 10
_MARKER_DIM_HI = 14
_MARKER_AREA_LO = 120
_MARKER_AREA_HI = 168

# --- CFG: generation retries ---
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {int(r)}
    cols = {int(c)}
    if _LO <= r <= _HI:
        rows.add(int(_LO + _HI - r))
    if _LO <= c <= _HI:
        cols.add(int(_LO + _HI - c))
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c)):
                if r >= _LO and c >= _LO:
                    return None
                if r0 <= c < r1 and c0 <= r < c1:
                    return None
                cells.append((r, c))
    return cells


def _valid_placements(h: int, w: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, list[tuple[int, int]]]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                if r0 < _LO and c0 < _LO:
                    continue
                fallback = _fallback_cells(r0, c0, h, w)
                if fallback is not None:
                    placements.append((r0, c0, fallback))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_components(fallback: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
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

    def merge_group(group: set[tuple[int, int]]) -> None:
        coords = [coord for coord in group if coord not in _CORNER_CELLS]
        if not coords:
            return
        first = coords[0]
        find(first)
        for coord in coords[1:]:
            union(first, coord)

    for r in range(_N):
        for c in range(_N):
            if (r, c) in _CORNER_CELLS:
                continue
            merge_group(_orbit(r, c))

    for r, c in fallback:
        merge_group(_orbit(r, c) | _orbit(c, r))

    components: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(_N):
        for c in range(_N):
            if (r, c) in _CORNER_CELLS:
                continue
            root = find((r, c))
            components.setdefault(root, []).append((r, c))
    return list(components.values())


def _marker_mask(r0: int, c0: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((_N, _N), dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _assign_component_colors(
    components: list[list[tuple[int, int]]],
    visible_mask: np.ndarray,
    bulk_palette: np.ndarray,
    marker_area: int,
) -> dict[tuple[int, int], int] | None:
    visible_components: list[tuple[int, list[tuple[int, int]]]] = []
    hidden_only: list[list[tuple[int, int]]] = []
    for coords in components:
        visible_weight = sum(1 for r, c in coords if bool(visible_mask[r, c]))
        if visible_weight == 0:
            hidden_only.append(coords)
        else:
            visible_components.append((visible_weight, coords))

    visible_total = int(visible_mask.sum() - len(_CORNER_CELLS))
    assert visible_total == sum(weight for weight, _coords in visible_components)

    colors = [int(color) for color in bulk_palette]
    base = visible_total // len(colors)
    extra = visible_total % len(colors)
    assert base + (1 if extra else 0) < marker_area

    for _ in range(200):
        order = list(_RNG.permutation(len(colors)))
        remaining = {colors[idx]: base + (1 if rank < extra else 0) for rank, idx in enumerate(order)}
        assignments: dict[tuple[int, int], int] = {}
        ok = True

        shuffled = list(_RNG.permutation(len(visible_components)))
        weighted = [visible_components[idx] for idx in shuffled]
        weighted.sort(key=lambda item: item[0], reverse=True)

        for weight, coords in weighted:
            options = [color for color in colors if remaining[color] >= weight]
            if not options:
                ok = False
                break
            max_remaining = max(remaining[color] for color in options)
            best = [color for color in options if remaining[color] == max_remaining]
            color = int(_RNG.choice(best))
            remaining[color] -= weight
            for coord in coords:
                assignments[coord] = color

        if not ok:
            continue
        if any(value != 0 for value in remaining.values()):
            continue

        for coords in hidden_only:
            color = int(_RNG.choice(colors))
            for coord in coords:
                assignments[coord] = color
        return assignments

    return None


def _build_clean(
    r0: int,
    c0: int,
    h: int,
    w: int,
    fallback: list[tuple[int, int]],
    decoy_color: int,
    bulk_palette: np.ndarray,
) -> np.ndarray | None:
    visible_mask = ~_marker_mask(r0, c0, h, w)
    components = _build_components(fallback)
    assignments = _assign_component_colors(components, visible_mask, bulk_palette, h * w)
    if assignments is None:
        return None

    clean = np.empty((_N, _N), dtype=int)
    clean[:_LO, :_LO] = int(decoy_color)
    for (rr, cc), color in assignments.items():
        clean[rr, cc] = int(color)
    return clean


def _choose_marker_and_palettes() -> tuple[int, int, np.ndarray]:
    marker = int(_RNG.choice(_ALL_COLORS))
    remaining = [int(color) for color in _ALL_COLORS if int(color) != marker]
    _RNG.shuffle(remaining)
    decoy = remaining[0]
    bulk_palette = np.array(remaining[1:], dtype=int)
    assert len(bulk_palette) == 7
    return marker, decoy, bulk_palette


def _marker_is_unique_highest(inp: np.ndarray, marker: int) -> bool:
    colors, counts = np.unique(inp, return_counts=True)
    max_count = int(counts.max())
    winners = [int(color) for color, count in zip(colors, counts) if int(count) == max_count]
    return len(winners) == 1 and winners[0] == int(marker)


def _pick_placement(h: int, w: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    placements = _valid_placements(h, w)
    if not placements:
        return None

    with_fallback = [item for item in placements if item[2]]
    without_fallback = [item for item in placements if not item[2]]
    if with_fallback and without_fallback:
        pool = with_fallback if bool(_RNG.integers(2)) else without_fallback
    else:
        pool = placements
    return pool[int(_RNG.integers(len(pool)))]


def _gen_attempt() -> np.ndarray | None:
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    picked = _pick_placement(h, w)
    if picked is None:
        return None
    r0, c0, fallback = picked
    marker, decoy, bulk_palette = _choose_marker_and_palettes()
    clean = _build_clean(r0, c0, h, w, fallback, decoy, bulk_palette)
    if clean is None:
        return None

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = marker

    try:
        if not np.array_equal(main(inp), expected):
            return None
        if not _marker_is_unique_highest(inp, marker):
            return None
        if _find_reflected_strip_start(inp, marker) != _LO:
            return None
        candidates = _solid_rectangle_candidates(inp)
        if len(candidates) < 2:
            return None
        if not any(color == marker and area == h * w for color, _bbox, area in candidates):
            return None
        if not any(color == decoy and box == (0, 0, _LO - 1, _LO - 1) for color, box, _area in candidates):
            return None
        return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
