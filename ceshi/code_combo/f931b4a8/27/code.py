# Combo variant of f931b4a8 (index 27)
# combo_spec: D4/2+D5/1+D15/2
# Name: 嵌套框架+旋转铺贴+边界镜像

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color

# Design analysis
# ===============
# a) 具体化与行为追踪
#    原题输入仍是 2x2 象限：左上/右上只数非黑像素个数来决定输出高宽，左下提供 coarse
#    颜色，右下提供 stencil。融合后真正改变的是“右下如何在输出中展开”。
#    这里把右下固定成嵌套矩形框架，但每一层框架在同一侧开一个门洞；这样它仍然是
#    同心框架结构，同时具备明确朝向。输出按 qxq 粗块拼接时，四邻块朝向按 2x2 周期
#    轮换：原始、顺时针 90 度、180 度、逆时针 90 度。
#    小例子：q=8 时，左上块的开口朝上，右上块的开口朝右，左下块的开口朝左，右下块
#    的开口朝下，四块围成风车式同心图案。若输出宽度是 13，则最右列块只显示 5 列：
#    原题会直接取该朝向模板的前 5 列；本融合版改为取该朝向模板的最后 5 列并左右镜像，
#    也就是从靠近内部拼接边的纹理反射出去，因此边缘不会像“硬截断”那样突然断开。
#
# b) 复杂度校准
#    纯同心框架对 90 度旋转完全不敏感，会让 D5 退化，所以模板必须带一个稳定的方向性
#    缺口；但缺口不能破坏“嵌套框架”这个主体，于是选择“每层框架同侧开门”而不是换成
#    L 形或随机散点。为了让 D15 也稳定可见，生成器强制输出至少跨 2x2 个粗块，并且
#    至少有一个边界块是不完整的。这样 rotation 与 boundary-reflection 都会在大多数样例
#    中同时出现，而自由度仍然只有少数档位：固定 4 个旋转、固定右/下边界镜像。
#
# c) 对原 main() 隐式假设的审查
#    - 原 `out_h/out_w = count_nonzero(...)` 与融合无冲突，保持不变。
#    - 原空 stencil 分支会直接平铺左下；融合语义主要针对非空模板，因此生成器不再产出
#      空 stencil，但 main() 仍保留该退化分支以保证健壮性。
#    - 原 `stencil[r % q, c % q]` 假设模板全局只有一种朝向；融合后失效，必须改成按粗块
#      奇偶选择旋转版本。
#    - 原边界处理默认“最后一个块直接裁左上角子矩形”；融合后这个前提失效，最后一行/列
#      的不完整块要改成从模板底边/右边反射取样。
#    - 原黑洞回填 `fill_tile[r // q, c // q]` 依赖的是 coarse block 坐标，而不是模板局部
#      坐标；旋转与边界镜像都不改变 coarse block 编号，因此这一点仍然成立。
#
# d) 放置合法性
#    没有自由摆放对象，合法性来自尺寸约束。生成器固定 N=16, q=8，输出宽高限制在 9..16，
#    因此 coarse block 行列数始终是 2，不会越过左下 8x8 的索引范围。右下模板由程序直接
#    构造，不存在重叠冲突；上方控制象限仍用精确计数填充，保证 main() 的断言成立。
#
# e) 视觉可观察性
#    解题者需要从 2-4 个样例中同时看出三件事：右下是同心框架模板、相邻块发生 90 度轮转、
#    最右/最下的残缺块不是简单裁切。为此生成器只用几何差异做约束：要求四个旋转版本互异，
#    要求融合输出同时区别于“仅旋转无边界镜像”和“仅边界镜像无旋转”，并要求至少一个
#    维度有余数，避免 D15 被隐藏。
#
# f) 可视化整洁度
#    结构仍是清晰的四象限输入和规则网格输出。左下只提供 coarse 颜色，右下只提供细纹理，
#    上方控制颜色不出现在输出。模板是规则框架而非噪声纹理，边界镜像也只发生在最后一行/
#    列块，不会把整个画面变成难以阅读的杂乱拼接。
#
# g) 颜色变化的可推断性
#    没有新的抽象颜色映射。输出颜色只来自左下 coarse 颜色和右下模板前景色；上方控制色
#    仍然与输出无关。模板前景色数量保持很少，左下颜色也限制为 1-3 种，并强制与模板色
#    分离，使人类更容易把“颜色来自哪个象限”与“几何变换如何发生”区分开。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)
LL_SOLID_PROB = 0.25
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.55
STENCIL_MULTI_COLOR_RANGE = (2, 3)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 9
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all (r,c) in a qxq grid in a layout-dependent order."""
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0 = random.randint(0, q - 1)
        c0 = random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar = random.randrange(q)
        ac = random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (all non-black) using stripe/solid/diagonal patterns."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _ring_depths(q: int) -> np.ndarray:
    rr, cc = np.mgrid[:q, :q]
    return np.minimum.reduce([rr, cc, q - 1 - rr, q - 1 - cc])


def _doorway_columns(q: int, depth: int) -> tuple[int, int]:
    """Return an inclusive-exclusive doorway interval on the top side of one ring."""
    span = q - 2 * depth
    width = 1 if span <= 5 else 2
    start = depth + max(1, (span - width) // 2)
    stop = min(q - depth - 1, start + width)
    if stop <= start:
        stop = start + 1
    return start, stop


def _make_nested_frame_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a nested-frame stencil with one oriented doorway per foreground frame."""
    assert q >= 8
    depths = _ring_depths(q)
    tile = np.zeros((q, q), dtype=int)

    for depth in range(int(depths.max()) + 1):
        if depth % 2 == 1:
            continue
        color = palette[(depth // 2) % len(palette)]
        tile[depths == depth] = color
        start, stop = _doorway_columns(q, depth)
        tile[depth, start:stop] = BLACK

    # Reinforce the opening with a short centered stem so all rotations stay distinct.
    stem_col = q // 2
    tile[: q // 2, stem_col] = BLACK

    assert np.any(tile != BLACK)
    assert np.any(tile == BLACK)
    return tile


def _rotation_signature_count(tile: np.ndarray) -> int:
    return len({np.rot90(tile, k).tobytes() for k in range(4)})


def _rotations_for_tiling(tile: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Precompute the four parity-based stencil orientations."""
    return {
        (0, 0): tile,
        (0, 1): np.rot90(tile, 3),
        (1, 1): np.rot90(tile, 2),
        (1, 0): np.rot90(tile, 1),
    }


def _extract_block(
    tile: np.ndarray,
    block_h: int,
    block_w: int,
    *,
    mirror_bottom: bool,
    mirror_right: bool,
) -> np.ndarray:
    """Take one possibly partial tile block, reflecting from the interior-facing edges."""
    q = int(tile.shape[0])
    rows = np.arange(block_h, dtype=int)
    cols = np.arange(block_w, dtype=int)
    if mirror_bottom and block_h < q:
        rows = q - 1 - rows
    if mirror_right and block_w < q:
        cols = q - 1 - cols
    return tile[np.ix_(rows, cols)].copy()


def _render_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    use_rotation: bool,
    use_boundary_mirror: bool,
) -> np.ndarray:
    """Shared renderer used by the solver and generator-side degeneracy checks."""
    q = int(fill_tile.shape[0])
    assert fill_tile.shape == (q, q)
    assert stencil.shape == (q, q)

    if not np.any(stencil != BLACK):
        rr, cc = np.mgrid[:out_h, :out_w]
        out = fill_tile[rr % q, cc % q].copy()
        assert np.all(out != BLACK)
        return out

    out = np.empty((out_h, out_w), dtype=int)
    tile_rows = (out_h + q - 1) // q
    tile_cols = (out_w + q - 1) // q
    assert tile_rows <= q and tile_cols <= q
    rotations = _rotations_for_tiling(stencil)

    for block_r in range(tile_rows):
        for block_c in range(tile_cols):
            rs = block_r * q
            cs = block_c * q
            block_h = min(q, out_h - rs)
            block_w = min(q, out_w - cs)
            oriented = rotations[(block_r % 2, block_c % 2)] if use_rotation else stencil
            block = _extract_block(
                oriented,
                block_h,
                block_w,
                mirror_bottom=use_boundary_mirror and block_r == tile_rows - 1,
                mirror_right=use_boundary_mirror and block_c == tile_cols - 1,
            )
            bg_mask = block == BLACK
            if np.any(bg_mask):
                fill_value = int(fill_tile[block_r, block_c])
                assert fill_value != BLACK
                block[bg_mask] = fill_value
            out[rs : rs + block_h, cs : cs + block_w] = block

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the two upper control counts and expand the lower quadrants into a rotated nested-frame wallpaper with mirrored edge completion.

    Input structure:
    - The input is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control panels whose non-black counts encode output height and width.
    - The lower-left quadrant is an all-non-black coarse color tile, and the lower-right quadrant is a nested rectangular-frame stencil with black gaps.

    Rule:
    - Output height equals the number of non-black cells in the upper-left quadrant.
    - Output width equals the number of non-black cells in the upper-right quadrant.
    - If the stencil is empty, repeat the lower-left tile directly as a fallback.
    - Otherwise, split the output into qxq blocks. The four neighboring blocks follow a rotation cycle:
      original, 90-degree clockwise, 180-degree, 90-degree counterclockwise.
      Within each block, black stencil cells are replaced by the single coarse color at lower-left[block_row, block_col].
      If the last block row or last block column is only partially visible, do not crop the rotated stencil from its top-left corner; instead, reflect the needed rows from the tile's bottom edge and the needed columns from its right edge.

    Key insight:
    - The top half still determines only the output size, while the bottom half now defines a two-level pattern whose geometry depends on both coarse-block parity (rotation) and whether a boundary block is incomplete (mirror completion).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
    q = qh

    upper_left = g[:q, :q]
    upper_right = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    return _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_rotation=True,
        use_boundary_mirror=True,
    )


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0 : r0 + h, c0 : c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate combo sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    assert q >= 8

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    remaining_for_stencil = [c for c in COLORS if c not in ll_palette]
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
        *STENCIL_MULTI_COLOR_RANGE
    )
    if len(remaining_for_stencil) < st_k:
        return None
    stencil_palette = _palette(st_k, remaining_for_stencil)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    valid_dims = [
        (out_h, out_w)
        for out_h in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        for out_w in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        and (out_h % q != 0 or out_w % q != 0)
    ]
    if not valid_dims:
        return None

    double_partial = [(h, w) for h, w in valid_dims if h % q != 0 and w % q != 0]
    out_h, out_w = random.choice(double_partial or valid_dims)

    tile_rows = (out_h + q - 1) // q
    tile_cols = (out_w + q - 1) // q
    if tile_rows < 2 or tile_cols < 2:
        return None

    used_fill = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = _make_nested_frame_stencil(q, stencil_palette)
    if _rotation_signature_count(stencil) != 4:
        return None

    combo = _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_rotation=True,
        use_boundary_mirror=True,
    )
    rot_only = _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_rotation=True,
        use_boundary_mirror=False,
    )
    mirror_only = _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_rotation=False,
        use_boundary_mirror=True,
    )
    original = _render_output(
        fill_tile,
        stencil,
        out_h,
        out_w,
        use_rotation=False,
        use_boundary_mirror=False,
    )

    if np.count_nonzero(combo != rot_only) == 0:
        return None
    if np.count_nonzero(combo != mirror_only) == 0:
        return None
    if np.count_nonzero(combo != original) == 0:
        return None

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, combo):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
