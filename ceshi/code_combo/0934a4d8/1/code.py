# Combo variant of 0934a4d8 (index 1)
# combo_spec: D1/1+D4/3
# Name: 四重旋转对称+旋转180度恢复

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import bbox


# Design analysis for the combo variant
#
# a) Concrete behavior:
# - 原题在 [lo..N-1]^2 上依赖 LR+UD 双轴镜像，并在压到上/左边框时用转置回退。
# - 这个融合版把“可见全局结构”改成 C4：右下 28x28 主方块由一个 14x14 种子经
#   0/90/180/270 度旋转拼成，视觉上应当明显像四象限逐次转动，而不是镜像。
# - 但被遮挡像素的恢复机制不再使用整条 4 点轨道，而只看它的 180 度对径：
#   (r, c) -> (lo+hi-r, lo+hi-c)。例如 lo=2, hi=9 时，(3, 4) 只从 (8, 7)
#   取值；90 度邻点 (4, 8) 与 270 度邻点 (7, 3) 只是背景中可观察到的更强对称证据，
#   不是 solver 的恢复来源。
#
# b) Complexity calibration:
# - 如果仅要求“矩形在 C4 图案里”，那么 D1 单变体的 4 点轨道恢复与本融合版的
#   180 度恢复在输出上会完全一致，D4 改动就会失去作用。
# - 为避免这一退化，生成器强制矩形里至少存在一些“quarter-hidden”格子：这些格子的
#   某个 90 度邻点也被矩形一起遮住，但 180 度对径仍然露在外面。这样样例中会
#   真正出现“可见结构是 4 点 C4 轨道，但 solver 明确只读对径”的情况。
#
# c) Hidden assumptions in the original main():
# - `_find_reflected_strip_start` 默认对称可拆成横向镜像和纵向镜像；这里必须改为
#   整块 `[lo..N-1]^2` 的 90 度旋转一致性检测。
# - 原 `_reflection_orbit_values` 会从反射闭包里收集多个可见值；融合后不允许这样做，
#   因为恢复规则被限定成单一的 180 度对径拷贝。
# - 原 `else: value = g[c, r]` 依赖边框转置副本；融合后不再存在该副本，因此
#   `main()` 新增了“矩形必须完全位于旋转主方块内部”的输入假设。
#
# d) Placement legality:
# - 矩形只能放在 `[lo..N-1]^2` 内，不能碰上/左自由边框，更不能进入左上 lo x lo 角块。
# - 对矩形内每个格子，其 180 度对径必须落在矩形外，否则 solver 会看到 marker 而无法恢复。
# - 生成器还要求至少一个 quarter-hidden 格子，从而让 D4 的“只看对径”机制在样本里实际发生。
#
# e) Visual observability:
# - 人类能直接看到主方块的四象限是逐次旋转关系，而不是左右/上下镜像。
# - 输出仍是矩形切片；跨样例观察时，可以看出被遮挡内容与对面的 180 度区域一致。
# - quarter-hidden 格子通常出现在靠近旋转中心的矩形位置，能提供“为什么看对面半周而不是看镜像”
#   的额外支撑。
#
# f) Visual cleanliness:
# - 网格里仍只有一个异常对象：唯一的纯色矩形 marker。
# - 其余区域只有一层统一的 C4 纹理；外边框保持随机，不再叠加第二套回退结构，避免视觉噪声。
#
# g) Color inference:
# - 颜色 id 不参与任何规则判断；所有恢复都纯粹来自几何位置关系。
#
# Post-change audit of new main() assumptions:
# - `main()` 假设唯一实心矩形可定位；generator 继续用固定 marker 色并用 oracle 自检。
# - `main()` 假设最小 C4 一致偏移就是 `_LO=2`；generator 随机化外边框并显式复验
#   `_find_rotated_strip_start(inp, _MARKER) == _LO`。
# - `main()` 假设 marker 全在旋转主方块内，且每个遮挡格的 180 度对径不是 marker；
#   `_placement_stats()` 逐格过滤保证这一点。


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


def _rotate180(r: int, c: int, lo: int, hi: int) -> tuple[int, int]:
    return int(lo + hi - r), int(lo + hi - c)


def _rotation_orbit(r: int, c: int, lo: int, hi: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    for _ in range(4):
        coords.append((rr, cc))
        rr, cc = _rotate90(rr, cc, lo, hi)
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the rectangular patch hidden by the solid marker using a C4 layout but 180-degree recovery.

    Input structure:
    - A square colored grid contains one solid rectangular marker color that hides a patch.
    - The large square block [lo..N-1] x [lo..N-1] is invariant under 90-degree rotation about its center.
    - The outer border above and left of that block is just distracting free context; it does not supply recovery values.

    Rule:
    - Find the unique color whose whole support is a solid rectangle; that rectangle is the output window.
    - Infer the smallest lo for which the block [lo..N-1] x [lo..N-1] is quarter-turn consistent, ignoring marker pixels.
    - For each marker pixel, recover its value from the single cell obtained by a 180-degree turn around that same center: g[lo+hi-r, lo+hi-c].

    Key insight:
    The visible grid advertises a stronger C4 symmetry than the solver actually uses. Once the rotational center is found, every hidden cell is restored by its diametrically opposite mate, not by mirror axes and not by the full four-point orbit.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_rotated_strip_start(g, marker)
    hi = int(g.shape[0] - 1)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            assert lo <= r <= hi and lo <= c <= hi
            mate_r, mate_c = _rotate180(r, c, lo, hi)
            value = int(g[mate_r, mate_c])
            assert value != marker
            out[rr, cc] = value

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


def _placement_stats(r0: int, c0: int, h: int, w: int) -> tuple[bool, int]:
    """Check legality and count cells whose quarter-turn orbit is partially hidden."""
    r1, c1 = r0 + h, c0 + w
    quarter_hidden_cells = 0

    for r in range(r0, r1):
        for c in range(c0, c1):
            mate_r, mate_c = _rotate180(r, c, _LO, _HI)
            if r0 <= mate_r < r1 and c0 <= mate_c < c1:
                return False, 0

            orbit = _rotation_orbit(r, c, _LO, _HI)
            inside = [r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in orbit]
            if (inside[1] or inside[3]) and not inside[2]:
                quarter_hidden_cells += 1

    return True, quarter_hidden_cells


def _valid_placements(h: int, w: int) -> list[tuple[int, int, int]]:
    """Return all valid (r0, c0, quarter_hidden_cells) placements for an h x w marker."""
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, int]] = []
        for r0 in range(_LO, _N - h + 1):
            for c0 in range(_LO, _N - w + 1):
                ok, quarter_hidden_cells = _placement_stats(r0, c0, h, w)
                if ok and quarter_hidden_cells > 0:
                    placements.append((r0, c0, quarter_hidden_cells))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean() -> np.ndarray:
    """Build a clean NxN grid with a visibly C4-symmetric inner square."""
    half = (_N - _LO) // 2
    clean = _RNG.choice(_PALETTE, size=(_N, _N))

    seed = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([seed, np.rot90(seed, -1)], axis=1)
    bottom = np.concatenate([np.rot90(seed, 1), np.rot90(seed, 2)], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, bottom], axis=0)
    return clean


def _visible_reflection_mismatches(clean: np.ndarray) -> int:
    block = clean[_LO:_N, _LO:_N]
    lr = int(np.count_nonzero(block != block[:, ::-1]))
    ud = int(np.count_nonzero(block != block[::-1, :]))
    return lr + ud


def _sample_size_and_placement() -> tuple[int, int, int, int, int] | None:
    order = _RNG.permutation(len(_SIZE_PAIRS))
    for idx in order:
        h, w = _SIZE_PAIRS[int(idx)]
        placements = _valid_placements(h, w)
        if not placements:
            continue
        r0, c0, quarter_hidden_cells = placements[int(_RNG.integers(len(placements)))]
        return int(h), int(w), int(r0), int(c0), int(quarter_hidden_cells)
    return None


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    if _visible_reflection_mismatches(clean) == 0:
        return None

    sampled = _sample_size_and_placement()
    if sampled is None:
        return None
    h, w, r0, c0, _quarter_hidden_cells = sampled

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
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
