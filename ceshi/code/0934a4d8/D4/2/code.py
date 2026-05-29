# Variant of 0934a4d8
# Dimension: D4 — 恢复变换机制 (transform)
# Change: 通过 LR/UD 反射轨道找到非标记值恢复；若轨道全被遮挡，使用转置回退 g[c,r]... → 多层级回退恢复

from __future__ import annotations

import numpy as np

from arc_utils import bbox

# Design analysis:
# a) 具体化与行为追踪
#    - 变体仍然是 30x30 方阵、一个实心矩形 marker、输出该矩形下方原始子图。
#    - 关键变化不再是“把 LR/UD 轨道上的所有可见值并起来”，而是对每个被遮挡像素按固定优先级找单一来源：
#      1. 左右镜像 `(r, mc)`
#      2. 上下镜像 `(mr, c)`
#      3. 一条偏移 1 格的主对角线反射 `(c + 1, r - 1)`
#      4. 主对角转置 `(c, r)`
#    - 小例子：若 marker 覆盖了 `(r, c)`、`(r, mc)`、`(mr, c)`，但偏移对角对应 `(c + 1, r - 1)` 仍可见，
#      本变体会在前两层失败后继续走第三层；若该位置也被挡住，才会退到最终的 `(c, r)`。
# b) 复杂度校准
#    - 仍保留“唯一实心矩形 marker + 大网格对称纹理”的原任务视觉骨架，不额外引入新对象或按颜色 id 排序。
#    - 直接使用穿过网格中心的对角反射会和“实心矩形 + 最终转置”产生不可恢复的固定点，因此第三层改成紧邻主对角线的偏移反射轴。
#    - 生成器挑选能同时触发多层分支的矩形；底图仍维持原题的 LR/UD 对称骨架，只在确需第三/第四层的位置补充一致性约束。
# c) 原 solver 的隐式假设审查
#    - 原 `_reflection_orbit_values()` 假设 LR/UD 任意可见成员并集后只有一个值；新规则要求保留来源优先级，这个假设不再可用。
#    - 原 `else: assert r < lo or c < lo` 假设只有外边框像素才会用转置；新变体允许内层像素在前三层都被挡住时再用转置。
#    - 原生成器只需保证“整条 LR/UD 轨道全被挡住”的像素有转置备份；新生成器必须逐像素保证优先级链中最早可见来源存在。
# d) 放置合法性
#    - 合法放置的定义改成：marker 内每个像素都必须存在一个最早可用的恢复分支，且该来源不在 marker 内。
#    - 额外要求分支覆盖：优先挑选同时包含 `lr / ud / diag / transpose` 四类像素的放置，避免退化成原题。
# e) 视觉可观察性
#    - 人类能在样例中看到：同一个 marker 内，不同像素会分别从左右、上下、偏移对角、转置四类位置取值；
#      第三层来源会出现在一条斜向偏移带上，不会和第四层主对角转置重合。
# f) 可视化整洁度
#    - 仍只有背景纹理与一个纯色矩形 marker 两层；没有额外叠加物，视觉负担与原题接近。
# g) 颜色变化可推断性
#    - 不做颜色映射，只做位置变换；8 种普通颜色重复出现在同类对称位置上，规则来自空间关系而非颜色 id。
# h) 新的 solver 假设与生成器保证
#    - solver 假设 `[lo..hi]` 仍可由 LR/UD 一致性推回；生成器继续对所有可见行列保持该双轴镜像。
#    - solver 假设第三层偏移对角源和第四层转置源在需要时都可用；生成器通过把相应像素的 LR/UD 轨道与对应来源轨道做颜色合并来保证这一点。


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


def _mirror(x: int, lo: int, hi: int) -> int:
    return int(lo + hi - x)


def _diag_reflect(r: int, c: int) -> tuple[int, int]:
    # Reflection across the offset diagonal r = c + 1.
    return int(c + 1), int(r - 1)


def _recover_from_priority_chain(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> int:
    n, m = map(int, g.shape)
    hi = n - 1
    assert n == m

    dr, dc = _diag_reflect(int(r), int(c))
    candidates: list[tuple[bool, tuple[int, int]]] = [
        (lo <= c <= hi, (int(r), _mirror(int(c), lo, hi))),
        (lo <= r <= hi, (_mirror(int(r), lo, hi), int(c))),
        (0 <= dr < n and 0 <= dc < m, (dr, dc)),
        (0 <= c < n and 0 <= r < m, (int(c), int(r))),
    ]

    for enabled, (rr, cc) in candidates:
        if not enabled:
            continue
        value = int(g[rr, cc])
        if value != marker:
            return value

    raise AssertionError("no recovery source found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the rectangle hidden by the solid marker using a prioritized chain of square symmetries.

    Input structure:
    - A square colored grid contains exactly one solid rectangular marker color covering the answer patch.
    - The large lower/right square strip still keeps the same horizontal and vertical mirror axes as the original task.
    - The clean pattern still keeps the original LR/UD mirror structure, and selected cells also expose a diagonal-style backup chain when the simpler mirrors are blocked.

    Rule:
    - Detect the unique solid-rectangle color and output that rectangle's original contents.
    - Infer the mirrored strip start as before from the smallest lower/right strip that is LR- and UD-consistent when marker pixels are ignored.
    - Restore each covered pixel by trying, in order: its LR mirror, its UD mirror, a fixed one-step-offset diagonal reflection, and finally its transposed cell.

    Key insight:
    The marker hides data but does not destroy the backup chain; each missing pixel is determined by the first still-visible witness along a fixed priority ladder rather than by an unordered LR/UD orbit.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            out[rr, cc] = _recover_from_priority_chain(g, int(r0 + rr), int(c0 + cc), lo, marker)

    return out


# --------------- generator ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 300

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACING_CACHE: dict[
    tuple[int, int],
    list[tuple[int, int, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]],
] = {}


def _inside_marker(r: int, c: int, r0: int, c0: int, h: int, w: int) -> bool:
    return r0 <= r < r0 + h and c0 <= c < c0 + w


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_mirror(r, _LO, _HI))
    if _LO <= c <= _HI:
        cols.add(_mirror(c, _LO, _HI))
    return {(rr, cc) for rr in rows for cc in cols}


def _branch_for_cell(r: int, c: int, r0: int, c0: int, h: int, w: int) -> str | None:
    mr = _mirror(r, _LO, _HI)
    mc = _mirror(c, _LO, _HI)
    dr, dc = _diag_reflect(r, c)
    candidates: list[tuple[str, bool, tuple[int, int]]] = [
        ("lr", _LO <= c <= _HI, (r, mc)),
        ("ud", _LO <= r <= _HI, (mr, c)),
        ("diag", 0 <= dr < _N and 0 <= dc < _N, (dr, dc)),
        ("transpose", 0 <= c < _N and 0 <= r < _N, (c, r)),
    ]
    for name, enabled, (rr, cc) in candidates:
        if enabled and not _inside_marker(rr, cc, r0, c0, h, w):
            return name
    return None


def _placement_branch_counts(
    h: int, w: int, r0: int, c0: int
) -> tuple[dict[str, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    counts = {"lr": 0, "ud": 0, "diag": 0, "transpose": 0}
    diag_cells: list[tuple[int, int]] = []
    transpose_cells: list[tuple[int, int]] = []
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            branch = _branch_for_cell(r, c, r0, c0, h, w)
            if branch is None:
                return None
            counts[branch] += 1
            if branch == "diag":
                diag_cells.append((r, c))
            if branch == "transpose":
                transpose_cells.append((r, c))
    return counts, diag_cells, transpose_cells


def _valid_placements(
    h: int, w: int
) -> list[tuple[int, int, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]]:
    key = (h, w)
    if key not in _PLACING_CACHE:
        placements: list[
            tuple[int, int, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]
        ] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                data = _placement_branch_counts(h, w, r0, c0)
                if data is None:
                    continue
                counts, diag_cells, transpose_cells = data
                if all(counts[name] > 0 for name in ("lr", "ud", "diag", "transpose")):
                    placements.append((r0, c0, counts, diag_cells, transpose_cells))
        _PLACING_CACHE[key] = placements
    return _PLACING_CACHE[key]


def _build_clean() -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    # Inner block: quarter tile mirrored LR then UD.
    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    # Top border rows: LR-symmetric.
    half = (_N - _LO) // 2
    for r in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    # Left border cols: UD-symmetric.
    for c in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    # Top-left corner: free.
    clean[:_LO, :_LO] = _RNG.choice(_PALETTE, size=(_LO, _LO))
    return clean


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


def _pick_shape_and_placement(
) -> tuple[int, int, int, int, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    choices: list[
        tuple[int, int, int, int, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]
    ] = []
    for h, w in _SIZE_PAIRS:
        for r0, c0, counts, diag_cells, transpose_cells in _valid_placements(h, w):
            choices.append((h, w, r0, c0, counts, diag_cells, transpose_cells))
    if not choices:
        return None

    weights = np.array(
        [
            1
            + 3 * int(item[4]["diag"] > 0)
            + 3 * int(item[4]["transpose"] > 0)
            + 2 * min(item[4]["diag"], 3)
            + 2 * min(item[4]["transpose"], 3)
            for item in choices
        ],
        dtype=float,
    )
    weights /= weights.sum()
    idx = int(_RNG.choice(len(choices), p=weights))
    return choices[idx]


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    picked = _pick_shape_and_placement()
    if picked is None:
        return None

    h, w, r0, c0, _counts, diag_cells, transpose_cells = picked
    groups = [_orbit(r, c) | _orbit(*_diag_reflect(r, c)) for r, c in diag_cells]
    groups.extend(_orbit(r, c) | _orbit(c, r) for r, c in transpose_cells)
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
    raise RuntimeError("failed to generate a valid 0934a4d8 variant input")
