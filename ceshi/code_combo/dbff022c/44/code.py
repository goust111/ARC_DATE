# Combo variant of dbff022c (index 44)
# combo_spec: D11/0+D13/1
# Name: 形状决定填充色 + 共享边框精灵

# Design analysis
# ===============
# spec:
#   preserved_core:
#     - 仍然是在背景色 0 上寻找“未接触整张图边界”的背景连通块，并只修改这些孔洞。
#     - 输出仍然只在空腔内部发生变化，外部背景和边框像素保持不变。
#   d13_change_shape_controls_fill:
#     - 删除原题边缘 2xN / Nx2 调色板，solver 不再读取任何 border->fill 映射。
#     - 只保留两类孔洞并赋固定颜色：
#       - 矩形孔洞 -> 3
#       - L 形孔洞（允许旋转） -> 7
#   d11_change_shared_borders:
#     - 不再要求每个精灵彼此隔开；至少有一对逻辑精灵共享一段 1 像素厚边框。
#     - 在像素层面，这表现为一个连通的单色边框团块里包含两个独立孔洞，
#       两个孔洞之间只隔着一列或一行公共边框像素。
#   concrete_pixel_realization:
#     - 每个样例固定含 3 个孔洞：
#       1. 一个共享边框团块中的矩形孔
#       2. 同一共享边框团块中的 L 形孔
#       3. 一个独立精灵中的重复孔型（矩形或 L），其边框颜色与共享团块不同
#     - 这样可以同时展示：
#       - 同一边框色下，不同孔形会得到不同填充色
#       - 不同边框色下，相同孔形仍得到相同填充色
#   behavior_trace_with_small_example:
#     - 原题小流程：
#       1. 找边缘调色板。
#       2. 看到某个孔洞周围主导边框色，例如 4。
#       3. 用 4 去调色板查出填充值，再填进去。
#     - 融合后小流程：
#       1. 图上没有调色板可读。
#       2. 先直接枚举所有封闭背景孔洞；哪怕两个孔洞属于同一个连通边框团块，也分别处理。
#       3. 若某孔洞裁到自身 bbox 后是全 1 矩形，则填 3；若是“整行+整列”的旋转 L，则填 7。
#     - 关键差异：
#       - 原题把“边框颜色”当钥匙；
#       - 这里把“孔洞几何”当钥匙，而且孔洞的归属不依赖把共享边框拆回独立精灵。
#   complexity_calibration:
#     - 只用两类孔形，避免规则档位太多，保证 2-4 个样例可归纳。
#     - 每个样例里共享团块固定同时出现矩形与 L，避免退化成“每个边框色只配一个填色”。
#     - 第三个独立孔洞重复其中一类并换边框色，给出跨颜色重复支撑。
#   implicit_assumption_audit_for_main:
#     - 原 `_best_palette()` / `_palette_mapping()` 假设边缘一定有图例；融合后彻底失效，必须移除。
#     - 原 `_fill_enclosed_bg_regions()` 假设可以凭邻接边框色查表；现在边框色不再决定结果，必须改为孔洞形状分类。
#     - 原任务常可把每个孔洞视作各自独立精灵；共享边框后，“按非背景连通块分割对象再读对象属性”会把两洞误并到同一对象，
#       所以 solver 必须直接以背景孔洞连通块为核心单位。
#   placement_legality:
#     - 生成器仍用 output-first：先画已填色输出，再把孔洞清回背景得到输入。
#     - 共享团块内部允许共享边框，但不同团块之间仍保留至少 1 格背景，避免整图过乱。
#     - 所有孔洞都必须不接触整张图外边界，这样 solver 的“边界接触=外部背景”前提仍成立。
#   visual_observability:
#     - 共享团块中同色边框包住两个不同孔形，输出里会出现同边框色对应两种填色，足以否定原题式颜色查表。
#     - 独立重复孔型提供“相同孔形跨边框色同填色”的第二条证据，强化“看形状而不是看颜色”的规律。
#   cleanliness_and_inference:
#     - 不引入噪声、标记或额外装饰；视觉信息只来自边框团块结构和孔洞形状。
#     - 填色只用 3 和 7 两种固定颜色，且都由肉眼可见的结构条件触发，不依赖颜色编号大小、排序或计数。
#   audit_of_new_main_assumptions:
#     - `_hole_fill_color()` 假设每个待处理孔洞只有“矩形”或“L”两类；生成器据此只产生这两类。
#     - `_fill_enclosed_bg_regions()` 假设共享边框不会让两个孔洞的背景连成一个大孔；生成器用 1 像素实墙隔开它们。
#     - `_bg_boundary()` 假设外边界仍以背景为主；生成器放置时禁止孔洞贴图外边界并保持大部分边框为空背景。

import random

import numpy as np

from arc_utils import components_mask


_RECT_FILL = 3
_L_FILL = 7
_BORDER_COLOR_POOL = [1, 2, 4, 5, 6, 8, 9]

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 11, 15

# Hole families
_RECT_HOLE_SIZES = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
_L_HOLE_SIZES = [(2, 2), (2, 3), (3, 2)]

# Generation
_MAX_RETRIES = 500


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _component_to_mask(pts: np.ndarray) -> np.ndarray:
    """Crop component coordinates to their bbox and return a boolean mask."""
    rs = pts[:, 0]
    cs = pts[:, 1]
    r0 = int(rs.min())
    c0 = int(cs.min())
    mask = np.zeros((int(rs.max()) - r0 + 1, int(cs.max()) - c0 + 1), dtype=bool)
    mask[rs - r0, cs - c0] = True
    return mask


def _is_l_shape(mask: np.ndarray) -> bool:
    """True iff the mask is a rotated L made from one full row plus one full column."""
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    for k in range(4):
        rot = np.rot90(mask, k)
        if np.all(rot[0, :]) and np.all(rot[:, 0]) and not np.any(rot[1:, 1:]):
            return True
    return False


def _hole_fill_color(mask: np.ndarray) -> int | None:
    """Map a hole shape mask to its fixed fill color."""
    if mask.all():
        return _RECT_FILL
    if _is_l_shape(mask):
        return _L_FILL
    return None


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int) -> None:
    """Fill each enclosed background component by its local geometry."""
    h, w = out.shape
    for pts in components_mask(out == bg, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        fill = _hole_fill_color(_component_to_mask(pts))
        if fill is None:
            continue
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed cavities from their own geometry, even when neighboring sprites share border pixels.

    Input structure:
    - The outer border is mostly background, and the non-background cells form hollow border structures.
    - There is no palette on any grid edge; the informative signal is each enclosed bg cavity's shape.
    - At least one connected border cluster contains two cavities that touch the same 1-cell wall segment, so object ownership is ambiguous.

    Rule:
    - Find every background component that is fully enclosed by non-background cells and does not touch the grid boundary.
    - Crop each enclosed cavity to its own bbox and classify that local mask: solid rectangle cavities get one fixed fill, rotated L cavities get another.
    - Fill each cavity independently, without trying to split shared-border clusters back into separate sprites.

    Key insight: the correct unit is the enclosed hole itself, not the surrounding border object; shared walls make sprite segmentation unreliable, but the cavity geometry still cleanly determines the fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    out = g.copy()
    _fill_enclosed_bg_regions(out, bg)
    return out


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point arrays for background components enclosed by a sprite mask."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_rect_hole(rng: random.Random) -> np.ndarray:
    """Return a filled rectangular hole mask."""
    h, w = rng.choice(_RECT_HOLE_SIZES)
    return np.ones((h, w), dtype=bool)


def _make_l_hole(rng: random.Random, touch_side: str | None = None) -> np.ndarray:
    """Return an L hole mask, optionally forced to touch a specific side with its long arm."""
    h, w = rng.choice(_L_HOLE_SIZES)
    base = np.zeros((h, w), dtype=bool)
    base[0, :] = True
    base[:, 0] = True

    if touch_side is None:
        return np.rot90(base, rng.randint(0, 3)).copy()
    if touch_side == "left":
        return np.flipud(base).copy() if rng.random() < 0.5 else base.copy()
    if touch_side == "right":
        right = np.fliplr(base)
        return np.flipud(right).copy() if rng.random() < 0.5 else right.copy()
    if touch_side == "top":
        return np.fliplr(base).copy() if rng.random() < 0.5 else base.copy()
    if touch_side == "bottom":
        bottom = np.flipud(base)
        return np.fliplr(bottom).copy() if rng.random() < 0.5 else bottom.copy()
    raise ValueError(f"Unknown touch side: {touch_side}")


def _make_hole(kind: str, rng: random.Random, touch_side: str | None = None) -> np.ndarray:
    """Return a hole mask of the requested family."""
    if kind == "rect":
        return _make_rect_hole(rng)
    if kind == "L":
        return _make_l_hole(rng, touch_side=touch_side)
    raise ValueError(f"Unknown hole kind: {kind}")


def _sprite_from_hole(hole: np.ndarray) -> np.ndarray:
    """Build a hollow border mask whose only cavity is the given hole."""
    h, w = hole.shape
    sprite = np.ones((h + 2, w + 2), dtype=bool)
    sprite[1 : h + 1, 1 : w + 1][hole] = False
    return sprite


def _compose_shared_pair(hole_a: np.ndarray, hole_b: np.ndarray, horizontal: bool) -> np.ndarray:
    """Combine two hollow sprites so they share one wall column or row."""
    sprite_a = _sprite_from_hole(hole_a)
    sprite_b = _sprite_from_hole(hole_b)
    ha, wa = sprite_a.shape
    hb, wb = sprite_b.shape

    if horizontal:
        out_h = max(ha, hb)
        out_w = wa + wb - 1
        combo = np.zeros((out_h, out_w), dtype=bool)
        combo[0:ha, 0:wa] |= sprite_a
        combo[0:hb, wa - 1 : wa - 1 + wb] |= sprite_b
    else:
        out_h = ha + hb - 1
        out_w = max(wa, wb)
        combo = np.zeros((out_h, out_w), dtype=bool)
        combo[0:ha, 0:wa] |= sprite_a
        combo[ha - 1 : ha - 1 + hb, 0:wb] |= sprite_b

    return combo


def _find_valid_positions(
    H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements with 1-cell separation from other objects and internal holes away from the outer frame."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            if occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == H - 1 or c0 + hc == W - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_object(
    grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, border_color: int, holes: list[np.ndarray]
) -> list[tuple[np.ndarray, int]]:
    """Render one border object and return hole point-arrays with their fill colors."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    painted: list[tuple[np.ndarray, int]] = []
    for hpts in holes:
        fill = _hole_fill_color(_component_to_mask(hpts))
        if fill is None:
            raise ValueError("Unexpected hole shape in generator")
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = fill
        painted.append((hpts, fill))
    return painted


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    shared_color, solo_color = rng.sample(_BORDER_COLOR_POOL, 2)
    repeated_kind = rng.choice(["rect", "L"])

    horizontal = rng.random() < 0.5
    if horizontal:
        if rng.random() < 0.5:
            shared_hole_a = _make_hole("rect", rng)
            shared_hole_b = _make_hole("L", rng, touch_side="left")
        else:
            shared_hole_a = _make_hole("L", rng, touch_side="right")
            shared_hole_b = _make_hole("rect", rng)
    else:
        if rng.random() < 0.5:
            shared_hole_a = _make_hole("rect", rng)
            shared_hole_b = _make_hole("L", rng, touch_side="top")
        else:
            shared_hole_a = _make_hole("L", rng, touch_side="bottom")
            shared_hole_b = _make_hole("rect", rng)
    shared_mask = _compose_shared_pair(shared_hole_a, shared_hole_b, horizontal=horizontal)
    shared_holes = _get_enclosed_holes(shared_mask)
    if len(shared_holes) != 2:
        return None
    shared_fills = {_hole_fill_color(_component_to_mask(hpts)) for hpts in shared_holes}
    if shared_fills != {_RECT_FILL, _L_FILL}:
        return None

    solo_mask = _sprite_from_hole(_make_hole(repeated_kind, rng))
    solo_holes = _get_enclosed_holes(solo_mask)
    if len(solo_holes) != 1:
        return None
    solo_fill = _hole_fill_color(_component_to_mask(solo_holes[0]))
    if solo_fill is None:
        return None

    output = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    placed: list[tuple[int, int, list[np.ndarray]]] = []

    objects = [
        (shared_mask, shared_color, shared_holes),
        (solo_mask, solo_color, solo_holes),
    ]
    objects.sort(key=lambda item: int(item[0].shape[0] * item[0].shape[1]), reverse=True)

    repeated_seen = False
    same_border_both_shapes = False

    for mask, border_color, holes in objects:
        positions = _find_valid_positions(H, W, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        painted = _paint_object(output, r0, c0, mask, border_color, holes)
        placed.append((r0, c0, [hpts for hpts, _fill in painted]))

        fills_here = {fill for _hpts, fill in painted}
        if border_color == shared_color and fills_here == {_RECT_FILL, _L_FILL}:
            same_border_both_shapes = True
        if border_color == solo_color and len(painted) == 1:
            repeated_seen = (solo_fill == _RECT_FILL and repeated_kind == "rect") or (
                solo_fill == _L_FILL and repeated_kind == "L"
            )

        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)] = True

    if not (same_border_both_shapes and repeated_seen):
        return None

    inp = output.copy()
    for r0, c0, holes in placed:
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
