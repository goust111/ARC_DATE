# Variant of aa4ec2a5
# Dimension: D8 — 对象与洞的重着色颜色方案 (transform)
# Change: 有洞对象：蓝色→青色，洞：背景→品红；无洞对象保持蓝色不变
#   → 有洞对象重着色为洞的形状编码色：矩形洞→绿色，L形洞→黄色，不规则洞→橙色
#
# Design analysis:
#   source_task: none
#   adaptation:
#     - 原题背景色是黄色(4)，但本变体要求某类有洞对象也变成黄色；若不改背景，黄色对象会与背景融在一起，
#       导致 D8 的颜色变化完全不可见。因此只对生成器做最小配套调整：背景改为灰色(5)。
#   concretization:
#     - 输入仍是 3-4 个蓝色(1)四连通对象，1-2 个对象含恰好 1 个内部背景洞。
#     - 输出仍给每个对象画 1 像素红色(2)八邻域外轮廓，洞仍填成品红(6)。
#     - 唯一核心变化是：有洞对象本体颜色不再统一为青色，而是取决于洞的几何类别：
#       rectangle -> green(3), L-shape -> yellow(4), irregular -> orange(7)。
#   behavior_trace:
#     - 原逻辑：若某组件存在洞，则无条件执行 out[mask] = 8。
#     - 新逻辑：先提取该洞的紧包围盒形状；若洞恰好铺满其 bbox，则 out[mask] = 3；
#       若洞等于某个角上的“横条 ∪ 竖条”L 形，则 out[mask] = 4；否则 out[mask] = 7。
#     - 例如 2x3 的满洞原题会把外壳染成青色，现在染成绿色；2x3 的 L 洞原题也染青色，
#       现在染成黄色；T 形洞则染成橙色。
#   complexity_calibration:
#     - 只保留 3 个类别，避免自由度过大。
#     - 生成器对 2 个 hollow 组件时强制选择不同类别，提高少量样例中的对照性。
#     - irregular 洞从小型模板库中采样，避免和矩形/L 形混淆。
#   implicit_assumption_review:
#     - _interior_holes 仍假设“洞”定义为 bbox 内且不接触 bbox 边界的 4 连通背景区域；该前提不变。
#     - 原 main 中 out[mask] = 8 依赖“所有有洞对象同色”这一旧前提；变体下必须改成按洞分类后赋不同颜色。
#     - 原代码把所有洞都并入 solid 再描边；这在变体中仍成立，因为描边规则未改。
#     - 新 solver 对“每个 hollow 组件恰好一个洞”有弱假设；若出现多个洞，则回退为 irregular 颜色。
#       生成器显式保证每个 hollow 组件只有一个洞。
#   placement_legality:
#     - 继续使用对象间 Chebyshev >= 3 和边界留白 >= 1，因此红色外轮廓不会互相覆盖，也不会越界写入。
#     - 洞模板先嵌入带 padding 的实心外壳，再可选切角/外扩；切角尺寸受 padding 限制，保证洞不泄露到外部。
#   visual_observability:
#     - 矩形洞、L 洞、不规则洞在输入上肉眼可区分，输出颜色也分别稳定映射到绿/黄/橙。
#     - 灰背景与黄对象、橙对象、红轮廓、品红洞都能清晰分离，避免视觉脏乱。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


RECT_HOLLOW_COLOR = 3
L_HOLLOW_COLOR = 4
IRREG_HOLLOW_COLOR = 7
OUTLINE_COLOR = 2
HOLE_COLOR = 6


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords.copy()
    shifted[:, 0] -= r0
    shifted[:, 1] -= c0
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == sub.shape[0] - 1)
            | (hole[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = hole.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _is_l_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2 or bool(mask.all()):
        return False

    for arm_h in range(1, h):
        for arm_w in range(1, w):
            if (h - arm_h) * (w - arm_w) < 2:
                continue

            expected = np.zeros((h, w), dtype=bool)
            expected[:arm_h, :] = True
            expected[:, :arm_w] = True
            if np.array_equal(mask, expected):
                return True
            if np.array_equal(mask, np.fliplr(expected)):
                return True
            if np.array_equal(mask, np.flipud(expected)):
                return True
            if np.array_equal(mask, np.flipud(np.fliplr(expected))):
                return True
    return False


def _hole_kind(hole: np.ndarray) -> str:
    tight = _tight_mask(hole)
    if bool(tight.all()):
        return "rectangle"
    if _is_l_hole(tight):
        return "lshape"
    return "irregular"


def _hollow_object_color(holes: list[np.ndarray]) -> int:
    if len(holes) != 1:
        return IRREG_HOLLOW_COLOR
    kind = _hole_kind(holes[0])
    if kind == "rectangle":
        return RECT_HOLLOW_COLOR
    if kind == "lshape":
        return L_HOLLOW_COLOR
    return IRREG_HOLLOW_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around objects, and recolor hollow ones by the geometry of their holes.

    Input structure:
    - A mostly uniform background contains several blue 4-connected foreground objects.
    - Some objects enclose a 4-connected background hole fully inside their bounding box.

    Rule:
    - Draw a one-cell 8-neighborhood red outline around every foreground component.
    - If a component has an enclosed hole, fill that hole with magenta (6).
    - Recolor the surrounding object by hole shape before outlining:
      rectangle hole -> green (3), L-shaped hole -> yellow (4), other hole -> orange (7).

    Key insight: The outline still treats the hole as part of a single solid silhouette, but
    the object's interior color now depends on classifying the hole's tight shape rather than
    merely checking whether a hole exists.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = _hollow_object_color(holes)
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = HOLE_COLOR
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 5
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

RECT_HOLE_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

GEN_OUTER_RETRIES = 300


IRREGULAR_HOLE_TEMPLATES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _rect_dims(area: int) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= HOLE_DIM_MAX and area // i <= HOLE_DIM_MAX
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _has_interior_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            return True
    return False


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE))
        for _ in range(num_rects)
    ]

    canvas_h = sum(r[0] for r in rects) + POLY_CANVAS_PAD
    canvas_w = sum(r[1] for r in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for i in range(1, num_rects):
        rh, rw = rects[i]
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), r0 + rects[0][0] + 2)
            nc = random.randint(max(0, c0 - rw + 1), c0 + rects[0][1] + 2)
            if nr + rh > canvas_h or nc + rw > canvas_w or nr < 0 or nc < 0:
                continue
            if mask[nr : nr + rh, nc : nc + rw].any():
                mask[nr : nr + rh, nc : nc + rw] = True
                break

    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)

    if side in ("top", "bottom"):
        ext_w = min(random.randint(2, max(2, base_w)), base_w)
        offset = random.randint(0, base_w - ext_w)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[:ext_h, offset : offset + ext_w] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h : base_h + ext_h, offset : offset + ext_w] = True
    else:
        ext_h_actual = min(ext_h, base_h)
        ext_w_actual = random.randint(*HOLLOW_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, :ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        if random.random() < 0.5:
            return np.ones((1, 2), dtype=bool)
        return np.ones((2, 1), dtype=bool)
    if size == 3:
        return random.choice(
            [
                np.array([[1, 1], [1, 0]], dtype=bool),
                np.array([[1, 0], [1, 1]], dtype=bool),
                np.array([[1, 1, 1]], dtype=bool),
                np.array([[1], [1], [1]], dtype=bool),
            ]
        )
    return np.ones((2, 2), dtype=bool)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _make_rect_hole_mask() -> np.ndarray:
    dims = _rect_dims(random.choice(RECT_HOLE_AREAS))
    if dims is None:
        raise RuntimeError("Failed to pick rectangle-hole dimensions")
    return np.ones(dims, dtype=bool)


def _make_l_hole_mask() -> np.ndarray:
    for _ in range(50):
        h = random.randint(2, HOLE_DIM_MAX)
        w = random.randint(2, HOLE_DIM_MAX)
        arm_h = random.randint(1, h - 1)
        arm_w = random.randint(1, w - 1)
        if (h - arm_h) * (w - arm_w) < 2:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[:arm_h, :] = True
        mask[:, :arm_w] = True
        return _transform_mask(mask)
    raise RuntimeError("Failed to build L-hole mask")


def _make_irregular_hole_mask() -> np.ndarray:
    return _transform_mask(random.choice(IRREGULAR_HOLE_TEMPLATES))


def _make_hole_mask(kind: str) -> np.ndarray:
    if kind == "rectangle":
        return _make_rect_hole_mask()
    if kind == "lshape":
        return _make_l_hole_mask()
    if kind == "irregular":
        return _make_irregular_hole_mask()
    raise ValueError(f"Unknown hole kind: {kind}")


def _carve_hole(mask: np.ndarray, hole_mask: np.ndarray, top: int, left: int) -> np.ndarray:
    carved = mask.copy()
    sub = carved[top : top + hole_mask.shape[0], left : left + hole_mask.shape[1]]
    sub[hole_mask] = False
    carved[top : top + hole_mask.shape[0], left : left + hole_mask.shape[1]] = sub
    return carved


def _make_hollow_component(kind: str) -> np.ndarray | None:
    hole_mask = _make_hole_mask(kind)
    hh, hw = hole_mask.shape

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)
    mask = _carve_hole(mask, hole_mask, pad_top, pad_left)

    num_notches = random.randint(0, HOLLOW_MAX_NOTCHES)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        max_h = (pad_top - 1) if corner[0] == "t" else (pad_bot - 1)
        max_w = (pad_left - 1) if corner[1] == "l" else (pad_right - 1)
        if max_h <= 0 or max_w <= 0:
            continue
        nh = random.randint(1, max_h)
        nw = random.randint(1, max_w)
        if corner == "tl":
            mask[:nh, :nw] = False
        elif corner == "tr":
            mask[:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, :nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w)
    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    if not _has_interior_hole(mask):
        return None

    holes = _interior_holes(mask, np.argwhere(mask))
    if len(holes) != 1 or _hole_kind(holes[0]) != kind:
        return None
    if int(mask.sum()) < HOLLOW_MIN_AREA:
        return None
    return mask


def _choose_hollow_kinds(num_hollow: int) -> list[str]:
    kinds = ["rectangle", "lshape", "irregular"]
    if num_hollow == 1:
        return [random.choice(kinds)]
    return random.sample(kinds, k=num_hollow)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])
    hollow_kinds = iter(_choose_hollow_kinds(num_hollow))

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            target_kind = next(hollow_kinds)
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(target_kind)
                if mask is not None:
                    components.append(mask)
                    found = True
                    break
            if not found:
                return None
        elif random.random() < SMALL_SHAPE_PROB:
            components.append(_make_small_shape())
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append(mask)
                    break
            else:
                return None

    if len(components) != num_components:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, pm in placed:
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
