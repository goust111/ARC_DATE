# Variant of 0934a4d8
# Dimension: D1 — 对称类型 (match_cond)
# Change: 双轴对称（LR + UD），反射轴位于 [lo..N-1] 区间的中心... → 四重旋转对称（C4）

from __future__ import annotations

import numpy as np

from arc_utils import bbox


# Design analysis for the D1 variant
#
# a) Concrete behavior:
# - 原题的“可恢复依据”是 [lo..N-1] 大方块上的左右镜像 + 上下镜像；如果矩形压到上/左自由边框，
#   还可借助转置格做回退。
# - 这个变体把核心约束改成同一大方块上的 90 度旋转对称。遮挡矩形仍然是唯一的实心色块，
#   输出仍是该矩形下方的原始像素。若被遮挡格位于旋转方块内部，就从其 4 点旋转轨道
#   {(r,c), (c, lo+hi-r), (lo+hi-r, lo+hi-c), (lo+hi-c, r)} 恢复；若它落在上边框或左边框，
#   则改用边框上的 90 度对应点恢复。
# - 小例子：若 lo=2, hi=7，则内部点 (3,4) 的 C4 轨道是 {(3,4),(4,6),(6,5),(5,3)}；
#   上边框点 (0,5) 的回退点是 (4,0)，左边框点 (6,1) 的回退点是 (1,3)。
#   原题会看镜像点或转置点；变体必须看旋转轨道或 90 度边框对应点。
#
# b) Complexity calibration:
# - 由于矩形最大只有 9x9，不可能同时覆盖某内部格子及其 180 度对位；如果只放在内部，
#   求解会退化成“看半周对称也能做”。因此生成器要求矩形必须压到上边框或左边框，
#   使一部分输出像素必须通过边框上的 90 度对应点恢复。
# - 内部图案由一个 14x14 随机种子经 0/90/180/270 度拼成 28x28 大方块，视觉上会明显表现为
#   旋转而不是镜像；边框则只承担“旋转回退副本”的角色。
#
# c) Hidden assumptions in the original main():
# - `_find_reflected_strip_start` 假设“对称”可分别沿列反射和沿行反射检查；C4 中此假设失效，
#   必须改成对整个 [lo..N-1]^2 方块做 90 度旋转一致性检查。
# - `_reflection_orbit_values` 通过两条反射边生成最多 4 个同值点；C4 中 orbit 生成规则完全不同。
# - 原 `else: value = g[c, r]` 的回退依赖“边框上的转置副本一定可用”；C4 版本里这不成立，
#   必须改成“上边框和左边框互为 90 度副本”的新回退规则。
#
# d) Placement legality:
# - 旋转对称只定义在 [lo..N-1]^2 内；但矩形允许压到上边框或左边框，因为这些边框被额外构造为
#   彼此的 90 度对应副本。
# - 矩形禁止覆盖左上 lo x lo 角块，因为那一块没有可用的旋转回退。
# - 对每个落在内部方块的矩形格子，4 点旋转轨道不能全部落在矩形内，否则无法恢复。
#
# e) Visual observability:
# - 人类在样例里能看到：右下大方块不是左右/上下镜像，而是四象限围绕中心逐次旋转。
# - 输出仍是一个普通矩形切片，因此从 input->output 可直接观察到“矩形内像素来自旋转同位点”。
#
# f) Visual cleanliness:
# - 网格中只有一个异常实体：纯色矩形标记。其余区域是统一的旋转对称纹理；
#   上边框与左边框只保留一条简单的 90 度对应关系，仍然足够整洁。
#
# g) Color inference:
# - 颜色不做任何基于 id 的映射；所有恢复仅基于位置旋转关系。
#
# Post-change audit of new main() assumptions:
# - `main()` 假设唯一实心矩形仍可唯一定位；generator 继续用固定 marker 色并在末端自检。
# - `main()` 假设最小旋转一致 lo 就是真实 lo=2；generator 随机化外边框并用 `_find_rotated_strip_start`
#   复验，避免 lo=0/1 偶然也成立。
# - `main()` 假设边框遮挡格一定能在另一条边框上找到 90 度对应点；generator 的放置过滤保证矩形
#   不会覆盖左上角，也不会遮掉内部格子的整个 4 点轨道。


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


def _find_rotated_strip_start(g: np.ndarray, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m

    for lo in range(n // 2 + 1):
        block = g[lo:n, lo:n]
        rotated = np.rot90(block, -1)
        known = (block != marker) & (rotated != marker)
        if bool(np.any((block != rotated) & known)):
            continue
        return lo

    raise AssertionError("no rotated strip found")


def _rotate90(r: int, c: int, lo: int, hi: int) -> tuple[int, int]:
    return int(c), int(lo + hi - r)


def _rotation_orbit(r: int, c: int, lo: int, hi: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    for _ in range(4):
        coords.append((rr, cc))
        rr, cc = _rotate90(rr, cc, lo, hi)
    return coords


def _rotation_orbit_values(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> set[int]:
    hi = int(g.shape[0] - 1)
    values: set[int] = set()
    for rr, cc in _rotation_orbit(r, c, lo, hi):
        value = int(g[rr, cc])
        if value != marker:
            values.add(value)
    return values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the rectangular patch hidden by the solid marker using quarter-turn rotational symmetry.

    Input structure:
    - A square colored grid contains one solid rectangular marker color that hides a patch.
    - The lower-right square block [lo..N-1] x [lo..N-1] is 90-degree rotation-symmetric about its center.
    - The thin top border and left border outside that block carry auxiliary 90-degree counterparts of each other, while the top-left corner remains unconstrained.

    Rule:
    - Find the unique color whose full extent is a solid rectangle; that rectangle is the output window.
    - Infer the smallest lo for which the square block [lo..N-1] x [lo..N-1] is consistent with a 90-degree rotation, ignoring marker pixels.
    - For each marker pixel inside the rotational square, collect the visible non-marker values in its 0/90/180/270 rotation orbit and copy the unique surviving value.
    - For each marker pixel on the top border or left border, copy the value from its paired 90-degree counterpart on the other border.

    Key insight:
    The marker does not define the pattern. It only occludes cells from a larger C4-symmetric construction, so every hidden pixel is determined either by its quarter-turn orbit in the main square or by a border cell placed as a 90-degree auxiliary copy.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_rotated_strip_start(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            if lo <= r < g.shape[0] and lo <= c < g.shape[1]:
                values = _rotation_orbit_values(g, r, c, lo, marker)
                assert values
                assert len(values) == 1
                out[rr, cc] = values.pop()
            elif r < lo <= c:
                value = int(g[lo + g.shape[0] - 1 - c, r])
                assert value != marker
                out[rr, cc] = value
            elif c < lo <= r:
                value = int(g[c, lo + g.shape[0] - 1 - r])
                assert value != marker
                out[rr, cc] = value
            else:
                raise AssertionError("marker overlaps unsupported top-left corner")

    return out


# --------------- generator (o5x) ---------------

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
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, int]]] = {}


def _orbit(r: int, c: int) -> list[tuple[int, int]]:
    return _rotation_orbit(r, c, _LO, _HI)


def _placement_stats(r0: int, c0: int, h: int, w: int) -> tuple[bool, int]:
    border_recovery_cells = 0
    r1, c1 = r0 + h, c0 + w
    for r in range(r0, r1):
        for c in range(c0, c1):
            if r < _LO and c < _LO:
                return False, 0
            if r < _LO or c < _LO:
                border_recovery_cells += 1
                continue
            orbit = _orbit(r, c)
            inside = [r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in orbit]
            if all(inside):
                return False, 0
    return True, border_recovery_cells


def _valid_placements(h: int, w: int) -> list[tuple[int, int, int]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, int]] = []
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                ok, border_recovery_cells = _placement_stats(r0, c0, h, w)
                if ok and border_recovery_cells > 0:
                    placements.append((r0, c0, border_recovery_cells))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean() -> np.ndarray:
    """Build a clean NxN grid with a C4 inner square and 90-degree border fallback."""
    half = (_N - _LO) // 2
    clean = _RNG.choice(_PALETTE, size=(_N, _N))

    seed = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([seed, np.rot90(seed, -1)], axis=1)
    bottom = np.concatenate([np.rot90(seed, 1), np.rot90(seed, 2)], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, bottom], axis=0)

    clean[:_LO, _LO:_N] = _RNG.choice(_PALETTE, size=(_LO, _N - _LO))
    for r in range(_LO, _N):
        for c in range(_LO):
            clean[r, c] = clean[c, _LO + _HI - r]
    clean[:_LO, :_LO] = _RNG.choice(_PALETTE, size=(_LO, _LO))
    return clean


def _visible_reflection_mismatches(clean: np.ndarray) -> int:
    block = clean[_LO:_N, _LO:_N]
    lr = int(np.count_nonzero(block != block[:, ::-1]))
    ud = int(np.count_nonzero(block != block[::-1, :]))
    return lr + ud


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    if _visible_reflection_mismatches(clean) == 0:
        return None

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None
    r0, c0, _border_recovery_cells = placements[int(_RNG.integers(len(placements)))]

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if _find_rotated_strip_start(inp, _MARKER) != _LO:
            return None
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
    raise RuntimeError("failed to generate a valid 0934a4d8 D1 input")
