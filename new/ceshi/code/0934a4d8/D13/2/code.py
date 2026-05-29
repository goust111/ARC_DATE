# Variant of 0934a4d8
# Dimension: D13 — 多步处理序列 (multi_step)
# Change: 单步处理：识别标记 -> 确定 lo -> 反射恢复 -> 裁剪输出... → 先裁剪再恢复

from __future__ import annotations

import numpy as np

from arc_utils import bbox

# Design analysis
# ---
# variant_spec:
#   core_change:
#     original: "整张 30x30 网格共享一个全局 LR+UD 对称框架，solver 直接在全局坐标系恢复矩形。"
#     variant: "整张网格不再要求全局对称。唯一实心矩形标记只是某个局部 2h x 2w 对称面板的一个角象限；solver 必须先由标记尺寸裁出候选面板，再只在该局部面板内恢复。"
#   concrete_shape:
#     - "网格仍为 30x30，颜色仍使用 {1,2,3,4,5,6,7,9}，标记色仍为 8。"
#     - "先生成一个 h x w 的非对称纹样 q，再镜像成 2h x 2w 面板：左上=q，右上=q 的左右翻转，左下=q 的上下翻转，右下=q 的双翻转。"
#     - "从四个角象限 {tl,tr,bl,br} 中随机选一个，用实心矩形色 8 覆盖该角象限。输出是被覆盖象限的原始 h x w 内容。"
#     - "面板放进 30x30 的随机背景中；背景不再服从原任务的全局轴对称，因此旧 solver 的全局 strip 搜索会失效。"
#   behavior_trace:
#     - "原逻辑示例：若标记在全局网格左上外带，solver 会先找全局 lo，再把某个标记像素映射到整张图里的水平/垂直反射轨道。"
#     - "新逻辑示例：若标记大小为 4x6 且位于局部面板右下角，solver 先枚举 4 个由该 4x6 矩形导出的 8x12 候选裁剪框，只保留那个在局部上镜像一致的面板；随后在该 8x12 子图内部用局部四点轨道恢复右下角。"
#     - "差异：恢复参考系从整图坐标改为局部裁剪坐标；如果不先裁剪，背景噪声会破坏全局对称假设。"
#   complexity_calibration:
#     - "角象限位置保留 4 个离散分支，避免生成器长期固定一种情况。"
#     - "q 被强制为多色且不自带水平/垂直对称，这样四个象限的取向差异肉眼可见，恢复结果不会退化成纯色块。"
#   old_main_assumption_audit:
#     - "_find_reflected_strip_start 依赖整图存在统一 lo；变体中此假设失效，必须删除。"
#     - "_reflection_orbit_values 依赖全局 strip 内的反射闭包；变体中改为局部 panel 的四点镜像轨道。"
#     - "原 fallback 到 g[c,r] 依赖全局外带暴露转置副本；变体中局部面板总有 3 个可见镜像副本，因此不再需要 transpose fallback。"
#   placement_legality:
#     - "要求 2h x 2w 面板完整落在 30x30 内，所以 h,w 仍取 3..9 且面积 16..36，面板最大仅 18x18。"
#     - "标记是面板四角之一，因此必与面板边界对齐，裁剪尺寸由标记尺寸唯一决定。"
#     - "生成后用 main(input)==expected 自检，并额外确认唯一实心矩形确实就是 marker。"
#   observability:
#     - "人类可见线索是：大网格里只有一块局部区域呈现明显双轴镜像，而 marker 只遮住该区域四分之一。"
#     - "不同样例中角象限可变化，能支撑从 2-4 个样例归纳‘先锁定局部面板，再用局部对称补全’。"
#   cleanliness:
#     - "背景随机但不使用 marker 色，局部面板是唯一强结构区域；没有额外边框、排序或颜色映射规则叠加。"
#   new_solver_assumptions:
#     - "marker 的 bbox 尺寸就是缺失象限尺寸。"
#     - "存在且仅存在一个把 marker 放进四角之一后得到的 2h x 2w 候选面板满足局部 LR+UD 镜像一致。"
#     - "每个 marker 像素的局部四点轨道中至少有一个、实际通常有三个非 marker 副本；generator 通过完整镜像面板保证该条件。"

# --------------- shared helpers ---------------

_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8
_CORNERS = ("tl", "tr", "bl", "br")


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


def _panel_bbox_from_marker(
    marker_bbox: tuple[int, int, int, int], corner: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = marker_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    if corner == "tl":
        return r0, c0, r0 + 2 * h - 1, c0 + 2 * w - 1
    if corner == "tr":
        return r0, c1 - 2 * w + 1, r0 + 2 * h - 1, c1
    if corner == "bl":
        return r1 - 2 * h + 1, c0, r1, c0 + 2 * w - 1
    if corner == "br":
        return r1 - 2 * h + 1, c1 - 2 * w + 1, r1, c1
    raise AssertionError(f"unknown corner: {corner}")


def _in_bounds(shape: tuple[int, int], b: tuple[int, int, int, int]) -> bool:
    n, m = map(int, shape)
    r0, c0, r1, c1 = b
    return 0 <= r0 <= r1 < n and 0 <= c0 <= c1 < m


def _mirror_consistent(panel: np.ndarray, marker: int) -> bool:
    lr = panel[:, ::-1]
    known = (panel != marker) & (lr != marker)
    if bool(np.any((panel != lr) & known)):
        return False

    ud = panel[::-1, :]
    known = (panel != marker) & (ud != marker)
    if bool(np.any((panel != ud) & known)):
        return False

    return True


def _find_local_panel(
    g: np.ndarray, marker: int, marker_bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int, int, int]]:
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []
    for corner in _CORNERS:
        panel_bbox = _panel_bbox_from_marker(marker_bbox, corner)
        if not _in_bounds(g.shape, panel_bbox):
            continue
        pr0, pc0, pr1, pc1 = panel_bbox
        panel = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        if _mirror_consistent(panel, marker):
            candidates.append((corner, panel_bbox))

    assert len(candidates) == 1
    return candidates[0]


def _quadrant_origin(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w
    if corner == "bl":
        return h, 0
    if corner == "br":
        return h, w
    raise AssertionError(f"unknown corner: {corner}")


def _orbit_values(panel: np.ndarray, r: int, c: int, marker: int) -> set[int]:
    h, w = map(int, panel.shape)
    coords = {
        (int(r), int(c)),
        (int(r), w - 1 - int(c)),
        (h - 1 - int(r), int(c)),
        (h - 1 - int(r), w - 1 - int(c)),
    }
    return {int(panel[rr, cc]) for rr, cc in coords if int(panel[rr, cc]) != marker}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden quadrant by first cropping the marker's local symmetric panel.

    Input structure:
    - A 30x30 colored grid contains one solid rectangular marker in color 8.
    - The whole grid is not globally symmetric; instead, the marker occupies one corner quadrant of a local 2h x 2w panel.
    - Inside that panel, the visible three quadrants are horizontal/vertical mirror copies of the missing quadrant.

    Rule:
    - Find the unique solid rectangle marker and use its h x w size to enumerate the four possible 2h x 2w crops where it could be a corner quadrant.
    - Keep the unique crop whose contents are mirror-consistent after ignoring marker pixels.
    - Within that cropped panel, recover each marker cell from its local four-point LR+UD reflection orbit and output the recovered h x w quadrant.

    Key insight:
    The symmetry no longer lives at the whole-grid level. The marker is a locator for a smaller symmetric panel, so the solver must switch reference frames: crop first, then restore inside the crop.
    """
    g = np.asarray(grid, dtype=int)
    marker, marker_bbox = _solid_rectangle_marker(g)
    corner, panel_bbox = _find_local_panel(g, marker, marker_bbox)
    pr0, pc0, pr1, pc1 = panel_bbox
    panel = g[pr0 : pr1 + 1, pc0 : pc1 + 1]

    h = int(marker_bbox[2] - marker_bbox[0] + 1)
    w = int(marker_bbox[3] - marker_bbox[1] + 1)
    qr0, qc0 = _quadrant_origin(h, w, corner)

    out = np.empty((h, w), dtype=int)
    for rr in range(h):
        for cc in range(w):
            values = _orbit_values(panel, qr0 + rr, qc0 + cc, marker)
            assert len(values) == 1
            out[rr, cc] = values.pop()
    return out


# --------------- generator ---------------

_N = 30
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36
_MAX_RETRIES = 400

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]


def _make_patch(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        patch = _RNG.choice(_PALETTE, size=(h, w))
        if len(np.unique(patch)) < 3:
            continue
        if np.array_equal(patch, patch[:, ::-1]):
            continue
        if np.array_equal(patch, patch[::-1, :]):
            continue
        return patch
    raise RuntimeError("failed to make asymmetric patch")


def _build_panel(h: int, w: int) -> np.ndarray:
    q = _make_patch(h, w)
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


def _marker_bbox_from_panel(pr0: int, pc0: int, h: int, w: int, corner: str) -> tuple[int, int, int, int]:
    qr0, qc0 = _quadrant_origin(h, w, corner)
    return pr0 + qr0, pc0 + qc0, pr0 + qr0 + h - 1, pc0 + qc0 + w - 1


def _quadrant_view(panel: np.ndarray, h: int, w: int, corner: str) -> np.ndarray:
    qr0, qc0 = _quadrant_origin(h, w, corner)
    return panel[qr0 : qr0 + h, qc0 : qc0 + w].copy()


def _gen_attempt() -> np.ndarray | None:
    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    panel = _build_panel(h, w)
    corner = _CORNERS[int(_RNG.integers(len(_CORNERS)))]

    pr0 = int(_RNG.integers(0, _N - 2 * h + 1))
    pc0 = int(_RNG.integers(0, _N - 2 * w + 1))
    r0, c0, r1, c1 = _marker_bbox_from_panel(pr0, pc0, h, w, corner)

    grid = _RNG.choice(_PALETTE, size=(_N, _N))
    grid[pr0 : pr0 + 2 * h, pc0 : pc0 + 2 * w] = panel

    expected = _quadrant_view(panel, h, w, corner)
    inp = grid.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = _MARKER

    try:
        marker, found_bbox = _solid_rectangle_marker(inp)
        if marker != _MARKER or found_bbox != (r0, c0, r1, c1):
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
    raise RuntimeError("failed to generate a valid 0934a4d8 D13 input")
