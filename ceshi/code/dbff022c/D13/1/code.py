# Variant of dbff022c
# Dimension: D13 — 填充颜色的确定来源 (transform)
# Change: 填充颜色完全由调色板图例显式给出... → 填充颜色由精灵形状决定——矩形用一种色，L形用另一种

# Design analysis
# ===============
# spec:
#   preserved_core:
#     - 仍然是黑色背景上的若干单色空心精灵。
#     - 仍然通过寻找“不接触整张网格边界”的背景连通块来定位待填充孔洞。
#     - 输入到输出的唯一变化仍然发生在这些孔洞内部。
#   d13_change:
#     - 删除原题的 2xN / Nx2 边缘调色板，不再从边缘读取 border->fill 映射。
#     - 每个孔洞的填充色只由孔洞本身的几何类别决定：
#       - 实心矩形孔洞 -> 固定填充为绿色(3)
#       - L 形孔洞（允许旋转） -> 固定填充为橙色(7)
#   concrete_pixel_realization:
#     - 生成器只产生两类内部空腔：
#       1. 矩形孔：孔洞在自身 bbox 内全部为背景。
#       2. L 形孔：孔洞等于“某一整行 + 某一整列”的并集，允许四种旋转。
#     - 精灵本体通过“在一个实心矩形块内部挖去该孔洞”得到，所以边框颜色与孔形独立。
#     - 每个样例至少包含一个矩形孔和一个 L 形孔，并且第三个精灵重复其中一类，
#       这样人眼能看到“不同边框色但相同孔形会得到相同填充色”。
#   behavior_trace:
#     - 原题局部流程：
#       1. 在边缘找到调色板。
#       2. 读取某个边框色 -> 某个填充色。
#       3. 找到孔洞后，靠邻接边框色查表填色。
#     - 本变体局部流程：
#       1. 根本没有调色板可读。
#       2. 找到孔洞后直接观察孔洞几何。
#       3. 若孔洞覆盖其 bbox 的全部格子，则填 3；若孔洞是旋转后的 L，则填 7。
#     - 差异：
#       - 原题依赖“边框颜色”和“边缘图例”；
#       - 变体依赖“孔洞形状”，边框颜色只负责描边，不再决定结果。
#   complexity_calibration:
#     - 只保留两类孔形，避免类别过多导致 2-4 个样例无法归纳。
#     - 同时要求每个样例中两类都出现，且至少一类重复一次，避免退化成
#       “每个对象各填一个偶然颜色”的弱规律。
#     - 边框色从固定的非填充色池中抽取，防止把“边框色=填充值”误读成规则。
#   implicit_assumption_audit_for_main:
#     - 原 `_best_palette()` / `_palette_mapping()` 假设边缘存在隔离的 2 格厚图例块；
#       该前提在变体中失效，因此整段必须删除，否则会直接 assert。
#     - 原 `_fill_enclosed_bg_regions()` 假设“孔洞邻接的主导边框色”一定是查表 key；
#       变体中无查表，所以必须改为“把孔洞自身裁成局部 mask 并分类”。
#     - 原 `main()` 中 `mapping, pal_mask = _palette_mapping(...)` 和
#       `assert mapping` 都依赖旧属性，现已不成立。
#   placement_legality:
#     - 生成器继续用 output-first：先构造带填色的输出，再把孔洞抹回背景得到输入。
#     - 放置时要求对象之间保留至少 1 格背景，避免相互粘连影响连通域判断。
#     - 孔洞不允许接触整张网格边界，确保 solver 仍可用“边界接触”排除外部背景。
#   visual_observability:
#     - 取消调色板后，图面更干净，解题者会自然把注意力放到“孔洞形状”。
#     - 输出中同类孔洞总是同色，不同类孔洞颜色不同，且这种规律跨不同边框色重复出现。
#   cleanliness:
#     - 不叠加额外标记，不引入噪声，不让多个机制共用同一区域。
#   color_inference:
#     - 填充色只使用两种固定颜色，且它们由可见结构（矩形/L）决定，
#       不依赖颜色 id 排序、计数大小或任何不可见元信息。

import random

import numpy as np

from arc_utils import components_mask


_RECT_FILL = 3
_L_FILL = 7
_BORDER_COLOR_POOL = [1, 2, 4, 5, 6, 8, 9]

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15

# Shape families
_RECT_HOLE_SIZES = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
_L_HOLE_SIZES = [(2, 2), (2, 3), (3, 2)]

# Generation
_MAX_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _component_to_mask(pts: np.ndarray) -> np.ndarray:
    """Crop a component's coordinates to its own bbox and return a boolean mask."""
    rs = pts[:, 0]
    cs = pts[:, 1]
    r0 = int(rs.min())
    c0 = int(cs.min())
    mask = np.zeros((int(rs.max()) - r0 + 1, int(cs.max()) - c0 + 1), dtype=bool)
    mask[rs - r0, cs - c0] = True
    return mask


def _is_l_shape(mask: np.ndarray) -> bool:
    """True iff the mask is an L polyomino of one full row plus one full column, up to rotation."""
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    for k in range(4):
        rot = np.rot90(mask, k)
        if np.all(rot[0, :]) and np.all(rot[:, 0]) and not np.any(rot[1:, 1:]):
            return True
    return False


def _hole_fill_color(mask: np.ndarray) -> int | None:
    """Map a hole shape to its fill color."""
    if mask.all():
        return _RECT_FILL
    if _is_l_shape(mask):
        return _L_FILL
    return None


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int) -> None:
    """Fill each enclosed background component according to its local shape class."""
    for pts in components_mask(out == bg, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == out.shape[0] - 1 or c == out.shape[1] - 1 for r, c in pts)
        if touches_boundary:
            continue
        fill = _hole_fill_color(_component_to_mask(pts))
        if fill is None:
            continue
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites' interiors from cavity geometry instead of any explicit legend.

    Input structure:
    - The border background is the dominant color, and several single-color hollow sprites sit on it.
    - There is no palette block on a grid edge; the only informative structure is each sprite's enclosed bg cavity.
    - Every enclosed cavity is generated as either a solid rectangle or an L shape, possibly rotated.

    Rule:
    - Find bg regions that are fully enclosed by non-bg cells and therefore do not touch the grid boundary.
    - Classify each enclosed cavity by its own shape: a full rectangular bbox gets one fill color, an L-shaped cavity gets another.
    - Fill every recognized cavity while leaving sprite borders and exterior background unchanged.

    Key insight: identical cavity shapes receive identical fills even when their border colors differ, so the answer is controlled by geometry, not by any edge legend or border-color lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    out = g.copy()
    _fill_enclosed_bg_regions(out, bg)
    return out


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point-arrays for background components inside a sprite mask."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_rect_hole(rng: random.Random) -> np.ndarray:
    """Return a filled rectangle hole mask."""
    h, w = rng.choice(_RECT_HOLE_SIZES)
    return np.ones((h, w), dtype=bool)


def _make_l_hole(rng: random.Random) -> np.ndarray:
    """Return a rotated L-shaped hole mask."""
    h, w = rng.choice(_L_HOLE_SIZES)
    hole = np.zeros((h, w), dtype=bool)
    hole[0, :] = True
    hole[:, 0] = True
    return np.rot90(hole, rng.randint(0, 3)).copy()


def _sprite_from_hole(hole: np.ndarray) -> np.ndarray:
    """Build a solid sprite with a single enclosed hole of the requested shape."""
    h, w = hole.shape
    sprite = np.ones((h + 2, w + 2), dtype=bool)
    inner = sprite[1 : h + 1, 1 : w + 1]
    inner[hole] = False
    return sprite


def _find_valid_positions(H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]) -> list[tuple[int, int]]:
    """Find placements with 1-cell separation and no hole touching the outer grid boundary."""
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


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    border_color: int,
    fill_color: int,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite border and its already-filled output hole."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = fill_color


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    # Always include both classes; the third sprite repeats one class for visible support.
    hole_kinds = ["rect", "L", rng.choice(["rect", "L"])]
    rng.shuffle(hole_kinds)

    border_colors = rng.sample(_BORDER_COLOR_POOL, len(hole_kinds))
    specs = []
    for kind, border_color in zip(hole_kinds, border_colors, strict=False):
        hole = _make_rect_hole(rng) if kind == "rect" else _make_l_hole(rng)
        fill_color = _hole_fill_color(hole)
        if fill_color is None:
            return None
        sprite = _sprite_from_hole(hole)
        holes = _get_enclosed_holes(sprite)
        if len(holes) != 1:
            return None
        specs.append((sprite, holes, border_color, fill_color))

    # Larger sprites go first to reduce placement failure.
    specs.sort(key=lambda item: int(item[0].shape[0] * item[0].shape[1]), reverse=True)

    output = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    placed: list[tuple[int, int, list[np.ndarray], int]] = []

    for sprite, holes, border_color, fill_color in specs:
        positions = _find_valid_positions(H, W, occ, sprite, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(output, r0, c0, sprite, border_color, fill_color, holes)
        placed.append((r0, c0, holes, fill_color))

        sh, sw = sprite.shape
        occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)] = True

    inp = output.copy()
    for r0, c0, holes, _fill_color in placed:
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
