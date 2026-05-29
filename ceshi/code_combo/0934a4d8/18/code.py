# Combo variant of 0934a4d8 (index 18)
# combo_spec: D4/0+D12/0
# Name: 禁止回退+噪声像素

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox

# Design analysis:
# a) 具体化与行为追踪
#    - 原题是 30x30 方阵，存在唯一一个纯色实心矩形 marker，输出该矩形下方被遮挡的原始子图。
#    - 原 solver 对每个 marker 像素先收集 LR/UD 反射轨道中的可见值；若整条轨道都被 marker 挡住，
#      才退到转置 `g[c, r]`。这次 D4/0 明确禁止该回退，因此 marker 只能放在“每个像素都能从反射轨道恢复”的区域。
#    - D12/0 又让可见区域出现少量单像素噪声。具体小例子：若某个被挡像素的 3 个可见镜像见证原本都是颜色 4，
#      现在其中 1 个被翻成 7，则原题的“值集合必须唯一”会失败；融合后应改成“用轨道内多数值恢复真值”。
#    - 因此融合后的典型输入外观是：marker 位于某一象限内部，不跨对称轴；每个被挡像素在另外三个对称位置都有可见副本，
#      但其中少数副本被单点噪声破坏。solver 需要同时利用“禁止回退带来的放置约束”和“多数一致性去噪”。
# b) 复杂度校准
#    - 若 marker 仍允许靠近上/左边框，则很多像素只剩 1 个反射见证，单点噪声就无法和真值区分，这会把题目变成不可解。
#      所以生成器把 marker 限制在四个镜像象限之一的内部区域，保证每个隐藏像素都有 3 个可见反射见证。
#    - 若噪声完全落在与 marker 无关的区域，人类几乎感觉不到它参与了恢复。因此噪声专门打在 marker 相关轨道的见证位置上，
#      让“多数投票”成为必要步骤，而不是装饰。
#    - 噪声数量保持在 2..4 个、且每条相关轨道最多 1 个噪声点，既能显著破坏局部对称，又不会压倒原题的整体双轴骨架。
# c) 隐式假设审查
#    - 原 `_find_reflected_strip_start()` 假设忽略 marker 后可见区域完全对称；噪声打破了这个假设，所以需要改成
#      “在所有 lo 中选残差最小者”，并由生成器保证真 lo=2 仍是最优。
#    - 原 `_reflection_orbit_values()` 与 `assert len(values) == 1` 假设所有可见见证完全一致；噪声使其失效，
#      新 solver 必须允许少量离群值并取严格多数。
#    - 原 `else: value = g[c, r]` 的转置回退在融合后是非法路径；新的 `main()` 不能再读取转置来源。
#    - 新 solver 新增假设：marker 内每个像素至少有 3 个可见反射见证，且其中真值严格占多数。生成器会显式构造并校验这点。
# d) 放置合法性
#    - marker 仍是唯一实心矩形，尺寸保持 3..9 且面积 16..36，完全在边界内。
#    - 合法放置要求 marker 完全处于某个镜像象限内部，既不碰上/左自由边框，也不跨越水平/垂直中心轴；
#      这样每个 marker 像素的 LR/UD 轨道恰有 3 个可见见证，零个像素需要转置回退。
#    - 噪声像素只放在 marker 外、且每个被选中的轨道只改 1 个见证，因此不会让多数规则失效。
# e) 视觉可观察性
#    - 人类看 2-4 个样例时，会看到：大图整体仍近似四向对称，但在 marker 的其余对称副本位置附近出现少量“坏点”。
#      这些坏点破坏了“直接照抄一个副本”的做法，却保留了“三份里两份一致”的可视线索。
#    - 同时 marker 不再靠近上/左边框，训练样例中不会出现原题那种需要转置补救的边界特例，这正是 D4/0 的可见后果。
# f) 可视化整洁度
#    - 画面元素仍只有对称底图、一个纯色矩形 marker、以及 2..4 个孤立噪声像素，不存在多层叠框或复杂装饰。
#    - 噪声是单像素、数量少，视觉上清楚是局部瑕疵而非第二对象，不会盖过主规律。
# g) 变化的可推断性
#    - 没有颜色映射、颜色排序或依赖颜色 id 的隐藏逻辑；颜色只通过“同一反射轨道本应相同”这一可见空间关系发挥作用。
#    - 由于 marker 每个隐藏像素都有 3 个可见见证，解题者可以在少量样例中归纳出“忽略孤立噪声，取一致的镜像值”。
# h) 对修改后 `main()` 的逐段假设复查
#    - `_solid_rectangle_marker()` 仍依赖“唯一实心矩形颜色”成立；生成器若意外制造出第二个实心矩形，会在验算阶段重试。
#    - `_best_reflected_strip_start()` 依赖“虽然有噪声，但真 lo 的总残差最小”；生成器会验证这一点。
#    - `_orbit_visible_values()` 依赖 marker 像素不需要转置、且有足够多的可见见证；象限放置保证每个目标像素都有 3 个见证。
#    - `_majority_value()` 依赖真值在每条相关轨道上严格多数；生成器限制每条轨道至多 1 个噪声点并在成功前再次校验。


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


def _mirror(x: int, lo: int, hi: int) -> int:
    return int(lo + hi - x)


def _symmetry_score(g: np.ndarray, lo: int, ignore: np.ndarray) -> int:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    score = 0

    strip = g[:, lo : hi + 1]
    strip_reflected = strip[:, ::-1]
    ignore_strip = ignore[:, lo : hi + 1]
    ignore_reflected = ignore_strip[:, ::-1]
    score += int(np.sum((strip != strip_reflected) & (~ignore_strip) & (~ignore_reflected)))

    strip = g[lo : hi + 1, :]
    strip_reflected = strip[::-1, :]
    ignore_strip = ignore[lo : hi + 1, :]
    ignore_reflected = ignore_strip[::-1, :]
    score += int(np.sum((strip != strip_reflected) & (~ignore_strip) & (~ignore_reflected)))

    return score


def _best_reflected_strip_start(g: np.ndarray, ignore: np.ndarray) -> tuple[int, int]:
    n, m = map(int, g.shape)
    assert n == m

    best_lo = -1
    best_score: int | None = None
    for lo in range(n // 2 + 1):
        score = _symmetry_score(g, lo, ignore)
        if best_score is None or score < best_score:
            best_lo = lo
            best_score = score

    assert best_lo >= 0 and best_score is not None
    return best_lo, best_score


def _orbit_coords(r: int, c: int, lo: int, n: int) -> list[tuple[int, int]]:
    hi = n - 1
    rows = {int(r)}
    cols = {int(c)}
    if lo <= r <= hi:
        rows.add(_mirror(int(r), lo, hi))
    if lo <= c <= hi:
        cols.add(_mirror(int(c), lo, hi))
    return sorted((rr, cc) for rr in rows for cc in cols)


def _orbit_visible_values(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> list[int]:
    values: list[int] = []
    for rr, cc in _orbit_coords(int(r), int(c), lo, int(g.shape[0])):
        value = int(g[rr, cc])
        if value != marker:
            values.append(value)
    return values


def _majority_value(values: list[int]) -> int:
    counts = Counter(values).most_common()
    assert counts
    top_value, top_count = counts[0]
    next_count = counts[1][1] if len(counts) > 1 else 0
    assert top_count > next_count
    assert top_count * 2 > len(values)
    return int(top_value)


def combo_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the fused behavior is active on a given input."""
    g = np.asarray(grid, dtype=int)
    marker, marker_bbox = _solid_rectangle_marker(g)
    marker_mask = g == marker
    lo, score = _best_reflected_strip_start(g, marker_mask)
    r0, c0, r1, c1 = marker_bbox

    fallback_needed_cells = 0
    noisy_marker_orbits = 0
    min_visible_witnesses: int | None = None

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            visible_values = _orbit_visible_values(g, r, c, lo, marker)
            visible_count = len(visible_values)
            if min_visible_witnesses is None or visible_count < min_visible_witnesses:
                min_visible_witnesses = visible_count
            if not visible_values:
                fallback_needed_cells += 1
                continue
            if len(set(visible_values)) > 1:
                noisy_marker_orbits += 1

    assert min_visible_witnesses is not None
    return {
        "lo": int(lo),
        "symmetry_score": int(score),
        "fallback_needed_cells": int(fallback_needed_cells),
        "noisy_marker_orbits": int(noisy_marker_orbits),
        "min_visible_witnesses": int(min_visible_witnesses),
        "marker_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the marker-covered rectangle from noisy reflection copies without any transpose fallback.

    Input structure:
    - A square color grid contains exactly one solid rectangular marker hiding the output patch.
    - The clean underlying picture still follows the original left-right and up-down reflection rules on the large [lo..N-1] strip.
    - A few visible single-pixel defects are injected into that reflective area, so some mirror witnesses disagree locally even though the hidden patch itself is untouched.

    Rule:
    - Detect the unique solid rectangle and treat its bounding box as the answer window.
    - Infer the reflective strip start by choosing the offset with the smallest LR/UD mismatch count after ignoring the marker.
    - For each hidden pixel, look only at its LR/UD reflection orbit and take the strict majority among the visible non-marker witnesses; do not use any transpose backup.

    Key insight:
    The generator now guarantees that every covered pixel has enough clean reflection witnesses to outvote the few noise pixels, so the correct recovery comes from denoising the orbit consensus rather than from a border-case transpose escape hatch.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    marker_mask = g == marker
    lo, _score = _best_reflected_strip_start(g, marker_mask)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            visible_values = _orbit_visible_values(g, r, c, lo, marker)
            assert len(visible_values) >= 2
            out[rr, cc] = _majority_value(visible_values)

    return out


# --------------- generator ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1
_AXIS_TOP = (_LO + _HI) // 2

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: noise ---
_NOISE_COUNT_LO = 2
_NOISE_COUNT_HI = 4

# --- CFG: generation retries ---
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int]]] = {}


def _orbit(r: int, c: int) -> list[tuple[int, int]]:
    return _orbit_coords(int(r), int(c), _LO, _N)


def _build_clean() -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    for c in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    clean[:_LO, :_LO] = _RNG.choice(_PALETTE, size=(_LO, _LO))
    return clean


def _mirror_rect_start(start: int, size: int) -> int:
    return _mirror(int(start + size - 1), _LO, _HI)


def _valid_placements(h: int, w: int) -> list[tuple[int, int]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int]] = []
        max_top_r0 = _AXIS_TOP - h + 1
        max_left_c0 = _AXIS_TOP - w + 1
        if max_top_r0 >= _LO and max_left_c0 >= _LO:
            for base_r0 in range(_LO, max_top_r0 + 1):
                for base_c0 in range(_LO, max_left_c0 + 1):
                    for row_flip in (False, True):
                        for col_flip in (False, True):
                            r0 = _mirror_rect_start(base_r0, h) if row_flip else base_r0
                            c0 = _mirror_rect_start(base_c0, w) if col_flip else base_c0
                            placements.append((int(r0), int(c0)))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _marker_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _visible_witnesses(r: int, c: int, r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for rr, cc in _orbit(r, c):
        if r0 <= rr < r0 + h and c0 <= cc < c0 + w:
            continue
        out.append((rr, cc))
    return out


def _paint_noise(inp: np.ndarray, clean: np.ndarray, r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    marker_cells = _marker_cells(r0, c0, h, w)
    _RNG.shuffle(marker_cells)

    noise_goal = int(_RNG.integers(_NOISE_COUNT_LO, min(_NOISE_COUNT_HI, len(marker_cells)) + 1))
    noise_cells: list[tuple[int, int]] = []

    for r, c in marker_cells:
        witnesses = _visible_witnesses(r, c, r0, c0, h, w)
        assert len(witnesses) == 3
        rr, cc = witnesses[int(_RNG.integers(len(witnesses)))]
        if (rr, cc) in noise_cells:
            continue
        base_value = int(clean[rr, cc])
        choices = [color for color in _PALETTE.tolist() if color != base_value]
        inp[rr, cc] = int(_RNG.choice(choices))
        noise_cells.append((rr, cc))
        if len(noise_cells) >= noise_goal:
            break

    assert len(noise_cells) == noise_goal
    return noise_cells


def _clean_output(clean: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    return clean[r0 : r0 + h, c0 : c0 + w].copy()


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None

    r0, c0 = placements[int(_RNG.integers(len(placements)))]
    expected = _clean_output(clean, r0, c0, h, w)

    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER
    noise_cells = _paint_noise(inp, clean, r0, c0, h, w)

    try:
        result = main(inp)
        if not np.array_equal(result, expected):
            return None
        metrics = combo_metrics(inp)
        if metrics["lo"] != _LO:
            return None
        if metrics["fallback_needed_cells"] != 0:
            return None
        if metrics["min_visible_witnesses"] != 3:
            return None
        if metrics["noisy_marker_orbits"] < len(noise_cells):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
