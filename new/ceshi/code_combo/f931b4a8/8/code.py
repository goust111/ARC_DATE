# Combo variant of f931b4a8 (index 8)
# combo_spec: D4/2+D5/0
# Name: 嵌套框架模板+镜像铺贴

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color

# Design analysis
# ===============
# a) 具体化与行为追踪
#    保留原题“四象限”骨架：左上/右上仍然只通过非黑像素个数控制输出高宽，左下仍然是粗粒度颜色源，
#    右下仍然是细粒度模板。融合后的变化全部发生在“模板长什么样”和“模板如何铺贴”这两个环节。
#    这里把右下模板固定成嵌套矩形框架：最外层是一圈前景框，里面留一圈黑洞，再往里再来一圈前景框。
#    但如果做成完全中心对称的单色框架，镜像铺贴会退化成普通周期铺贴，因为横翻/竖翻后完全一样。
#    所以每一层前景框都保持“框架”几何不变，同时令上边+左边用一色，下边+右边用另一色。
#    小例子：Q=8 时，外框可设为“上左=7，下右=6”，内框设为“上左=8，下右=4”，黑洞由左下粗色填入。
#    原题若直接周期平铺，相邻块横向接缝会出现“右边=6，下一块左边=7”的断裂；本变体在相邻块上做镜像，
#    下一块的左边来自原模板右边，于是接缝两侧都变成 6，竖向接缝也同理。于是人眼看到的是连续的万花筒式
#    框架环带，而不是简单重复。
#
# b) 复杂度校准
#    这个组合最容易退化的地方是 D4 与 D5 相互抵消：如果模板过于对称，D5 根本不可见；如果模板太乱，
#    又看不出 D4 的“嵌套框架”主体。因此模板几何固定为少量同心矩形框，变化只放在每层框架的侧边配色上。
#    生成器还强制：
#    - 右下模板非空；
#    - 输入网格固定为 16x16，使每个象限是 8x8，能稳定容纳至少两层框架；
#    - 输出高宽都在 9..16，使输出至少跨 2x2 个模板块，镜像必定发生；
#    - 样本必须让镜像输出与原题的普通周期输出存在显著像素差异。
#
# c) 对原 main() 的隐式假设审查
#    - “上方两个象限只负责计数”仍成立，不需要改。
#    - “左下没有黑色，且黑洞回填使用 coarse block 坐标 (r//Q, c//Q)” 仍成立，也必须保持不变；
#      镜像只影响右下模板的局部读取朝向，不能连带改 coarse 填色索引。
#    - 原题默认非空模板的局部索引是 (r % Q, c % Q)。这一前提在 D5 下失效，必须改成：
#      先按块编号 br=r//Q, bc=c//Q 判断奇偶，再决定局部坐标是否翻转。
#    - 新增前提是：若想让 D5 可观察，模板必须在水平和垂直翻转下都发生变化。solver 本身不强制这一点，
#      但 generator 必须保证。
#
# d) 放置合法性
#    生成器直接构造四个象限，没有对象重叠问题。新的合法性重点是：
#    - 右下 8x8 模板必须真的是嵌套框架，且含黑洞；
#    - 输出尺寸要让 ceil(out_h/Q)=ceil(out_w/Q)=2，保证只访问左下 tile 的前 2x2 coarse 颜色且不越界；
#    - 左下所有像素都非黑，保证黑洞替换永远合法。
#
# e) 视觉可观察性
#    解题者需要从少量样例中同时看出“两层框架模板”和“相邻块镜像”。为了让这两件事都能被肉眼抓到：
#    - 框架层数固定为少量、规则的矩形环，避免散点噪声；
#    - 模板配色用少量颜色，且上左 vs 下右的侧边颜色不同，使镜像后的接缝连续性明显可见；
#    - 左下颜色与右下模板颜色分离，黑洞填色不会和框架边界混淆。
#
# f) 可视化整洁度
#    输出中只有两类元素：模板框架前景，以及由黑洞暴露出的粗粒度块色。二者职责分明，没有额外标记层。
#    控制象限颜色不会出现在输出中，因此上方计数信息和下方生成信息视觉上完全分区。
#
# g) 变化的可推断性
#    本组合不引入任意颜色映射，颜色只承担“来自模板前景”与“来自左下粗填色”两种语义。
#    规则真正变化的是空间关系：右下是嵌套框架模板，相邻重复块按共享边镜像。人类可以通过框架边在接缝处
#    是否连续、以及黑洞层是否按 coarse block 着色，推断整套变换。
#
# h) main() 的新隐式假设与 generator 保证
#    main() 新增的实际假设只有两条：
#    - 只要 stencil 非空，就按镜像块奇偶读取 stencil；
#    - stencil 中的 0 表示回填 coarse 颜色，而不是“跳过不填”。
#    generator 通过构造非空嵌套框架模板、非黑左下 tile、以及至少 2x2 的输出块布局，保证这些假设成立并且可见。


BLACK = int(Color.BLACK)
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)
LL_SOLID_PROB = 0.20
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_COLOR_RANGE = (2, 3)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 9
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300
MIN_MIRROR_DIFF = 16


def _palette(k: int, pool: tuple[int, ...] | list[int] = COLORS) -> list[int]:
    pool_list = list(pool)
    random.shuffle(pool_list)
    return pool_list[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
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


def _mirrored_local_index(idx: int, tile_size: int, block_idx: int) -> int:
    local = int(idx % tile_size)
    if int(block_idx) % 2 == 1:
        return int(tile_size - 1 - local)
    return local


def _mirrored_stencil_lookup(stencil: np.ndarray, r: int, c: int) -> int:
    q = int(stencil.shape[0])
    block_r = int(r // q)
    block_c = int(c // q)
    local_r = _mirrored_local_index(int(r), q, block_r)
    local_c = _mirrored_local_index(int(c), q, block_c)
    return int(stencil[local_r, local_c])


def _frame_margins(q: int) -> list[int]:
    margins: list[int] = []
    margin = 0
    while q - 2 * margin >= 4:
        margins.append(margin)
        margin += 2
    return margins


def _paint_frame(tile: np.ndarray, margin: int, color_tl: int, color_br: int) -> None:
    q = int(tile.shape[0])
    lo = int(margin)
    hi = int(q - 1 - margin)
    tile[lo, lo : hi + 1] = int(color_tl)
    tile[lo : hi + 1, lo] = int(color_tl)
    tile[hi, lo : hi + 1] = int(color_br)
    tile[lo : hi + 1, hi] = int(color_br)


def _make_nested_frame_stencil(q: int, palette: list[int]) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    margins = _frame_margins(q)
    assert len(margins) >= 2
    assert len(palette) >= 2
    for idx, margin in enumerate(margins):
        color_tl = int(palette[idx % len(palette)])
        color_br = int(palette[(idx + 1) % len(palette)])
        if idx % 2 == 1:
            color_tl, color_br = color_br, color_tl
        _paint_frame(tile, margin, color_tl, color_br)
    return tile


def _foreground_frame_layers(stencil: np.ndarray) -> int:
    q = int(stencil.shape[0])
    count = 0
    for margin in _frame_margins(q):
        lo = int(margin)
        hi = int(q - 1 - margin)
        mask = np.zeros_like(stencil, dtype=bool)
        mask[lo, lo : hi + 1] = True
        mask[hi, lo : hi + 1] = True
        mask[lo : hi + 1, lo] = True
        mask[lo : hi + 1, hi] = True
        if np.all(stencil[mask] != BLACK):
            count += 1
    return int(count)


def _coarse_fill_colors(fill_tile: np.ndarray, tile_rows: int, tile_cols: int) -> set[int]:
    return {
        int(fill_tile[r, c])
        for r in range(tile_rows)
        for c in range(tile_cols)
    }


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = int(palette[0])
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = int(palette[(r + offset) % k])
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = int(palette[(c + offset) % k])
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = int(color)

    existing = set(map(int, np.unique(tile)))
    missing = [int(c) for c in palette if int(c) not in existing]
    if missing:
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = int(color)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2,
        min(max_colors, count, len(pool)),
    )
    palette = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0 : r0 + h, c0 : c0 + w] = _random_color_array((h, w), palette)
    else:
        coords = _ordered_coords(q, random.choice(["rect", "compact", "rows"]))[:count]
        for r, c in coords:
            tile[r, c] = int(random.choice(palette))
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = int(value)
    return out


def _periodic_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _variant_signal(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    q = int(g.shape[0] // 2)
    upper_left = g[:q, :q]
    upper_right = g[:q, q:]
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]
    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    mirrored = _expected_output(fill_tile, stencil, out_h, out_w)
    periodic = _periodic_output(fill_tile, stencil, out_h, out_w)
    return {
        "frame_layers": int(_foreground_frame_layers(stencil)),
        "mirror_diff_pixels": int(np.count_nonzero(mirrored != periodic)),
        "coarse_colors_used": int(len(_coarse_fill_colors(fill_tile, int(np.ceil(out_h / q)), int(np.ceil(out_w / q))))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size a mirrored output built from a nested-frame template.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants encode output height and width by non-black pixel count.
    - The lower-left quadrant is a non-black coarse color tile, and the lower-right quadrant is a nested rectangular-frame stencil.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - Repeat the lower-right stencil over the output with mirror tiling: odd repeated block rows flip vertically and odd repeated block columns flip horizontally.
    - Wherever the mirrored stencil is black, replace that cell with the lower-left coarse color at the repeated block coordinate `(r // Q, c // Q)`.

    Key insight:
    - The top half still supplies only dimensions, but the lower-right tile is now an asymmetric nested-frame template whose side colors only line up correctly when neighboring repeated blocks are reflected rather than copied in a fixed orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    qh, qw = h // 2, w // 2
    assert qh == qw
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                fill_r = int(r // qh)
                fill_c = int(c // qw)
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != BLACK
            out[r, c] = int(value)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(random.choice(GRID_SIZES))
    q = int(n // 2)

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    remaining_for_stencil = [c for c in COLORS if c not in ll_palette]
    st_k = int(random.randint(*STENCIL_COLOR_RANGE))
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
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)
    tile_rows = int(np.ceil(out_h / q))
    tile_cols = int(np.ceil(out_w / q))
    if tile_rows < 2 or tile_cols < 2:
        return None

    used_fill = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    fill_tile = _make_lower_left(q, ll_palette, required_coords=used_fill)
    stencil = _make_nested_frame_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    if np.any(fill_tile == BLACK):
        return None
    if _foreground_frame_layers(stencil) < 2:
        return None

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    periodic = _periodic_output(fill_tile, stencil, out_h, out_w)
    mirror_diff = int(np.count_nonzero(expected != periodic))
    if mirror_diff < MIN_MIRROR_DIFF:
        return None

    if len(_coarse_fill_colors(fill_tile, tile_rows, tile_cols)) < 2 and ll_k > 1:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
