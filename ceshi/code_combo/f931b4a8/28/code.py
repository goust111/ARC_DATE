# Combo variant of f931b4a8 (index 28)
# combo_spec: D5/0+D6/0+D12/0
# Name: 镜像铺贴+局部坐标替换+模板重着色

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 输入仍是偶数边长正方形，按四象限分割。UL/UR 仍通过非黑像素计数给出输出高宽。
#    - 为了让 D12 的“配对色”可推断，两个上方象限的第 0 行最左侧连续 k 个格子组成图例：
#      UL 放左下颜色源的颜色序列，UR 放与之配对的替换颜色序列；其余上方非黑像素只负责计数。
#    - LR 不再提供最终颜色，只提供一个前景/背景模板；输出中所有颜色都来自 LL。
#    - D5 作用在“怎样重复 LR 模板”上：块坐标 br=r//Q, bc=c//Q；若 br 为奇数则局部行翻转，
#      若 bc 为奇数则局部列翻转。于是相邻重复块互为镜像。
#    - D6 作用在“LL 从哪里取色”上：不再用粗粒度 fill_tile[br,bc]，而是用镜像后的模板局部坐标
#      fill_tile[lr,lc]。这样每个 QxQ 单元内部都会出现 LL 的细粒度颜色变化。
#    - D12 作用在“模板前景怎么着色”上：若镜像后的 stencil[lr,lc] 非黑，输出不是 stencil 自己的
#      颜色，而是 recolor_map[fill_tile[lr,lc]]；若 stencil[lr,lc] 为黑，则输出 fill_tile[lr,lc]。
#    - 小例子：Q=2，LL=[[2,3],[4,5]]，图例给 2->3, 3->4, 4->5, 5->2，stencil=[[8,0],[0,8]]。
#      原题在右上重复块的左上角会继续读 stencil[0,0]=8，黑洞才读粗粒度 LL[0,1]。
#      融合后该块被水平镜像，所以局部点变成 (0,1)；此处 stencil 为黑，于是输出 LL[0,1]=3。
#      同一块右上角局部点变成 (0,0)，stencil 为前景，于是输出 recolor_map[2]=3，而不是模板色 8。
#      差异同时来自镜像朝向、局部取色粒度、以及前景重着色。
#
# b) 复杂度校准
#    - 若 stencil 为空、输出只覆盖一个重复块、LL 单色、或 LL/stencil 对翻转完全对称，三项改动
#      都会退化。生成器因此强制：stencil 非空且同时含前景/背景；输出在横纵两个方向都跨越至少
#      两个 QxQ 重复块；LL 至少 2 色；LL 与 stencil 几何都破坏活跃翻转轴上的镜像对称。
#    - 颜色映射只用 2 或 3 个 LL 颜色，且使用循环置换，避免颜色自由度过大。
#
# c) 对 main() 的隐式假设审查
#    - `qh == qw` 仍成立，因为输入还是偶数正方形四等分。
#    - 空 stencil 分支仍可直接周期铺贴 LL；这个分支与 D5/D6/D12 无关，保留作健壮性分支。
#    - 原题默认“模板颜色就是输出颜色”；D12 使该前提失效，所以 LR 的非黑值只能当作布尔前景。
#    - 原题默认黑洞回填用粗粒度块坐标 `(r//Q,c//Q)`；D6 使该前提失效，必须改为镜像后的局部坐标。
#    - D5 让“局部坐标=简单取模”失效；要先根据重复块奇偶求镜像后的 `(lr,lc)`，再同时用于 LR 与 LL。
#    - 新增假设是：上方第 0 行左侧存在一段连续合法图例，并且恰好覆盖 LL 的全部颜色集合。
#      生成器通过保留图例槽位、且让图例后续 filler 颜色不属于 LL 调色板来保证这一点。
#
# d) 放置合法性
#    - 图例固定在 UL/UR 的第 0 行前 k 列，k<=Q，不越界；额外控制像素只放在未保留位置。
#    - LL 始终是 QxQ 且全非黑；LR 始终是 QxQ 且同时有黑/非黑。
#    - 输出尺寸满足 `out_h,out_w <= Q*Q`，因此任何需要粗粒度对照的对比 oracle 中 `(r//Q,c//Q)`
#      也不会越界；真正 solver 的局部坐标访问天然合法。
#
# e) 视觉可观察性
#    - 解题者能同时看到三条线索：相邻块的镜像对称、每个块内部来自 LL 的局部色纹理、以及同一
#      局部源色在模板前景处会变成图例指定的配对色。
#    - 生成器显式拒绝三类退化样本：融合输出与“无镜像版”相同、与“粗粒度取色版”相同、
#      或与“模板色直接输出版”相同。这样 2-4 个样例里三项机制都会真实出现。
#
# f) 可视化整洁度
#    - 上方图例集中在顶行，和控制计数功能共用象限但不与下方纹理混杂。
#    - LL 只用 2-3 色的条纹/对角/棋盘并加少量破对称点；LR 只用单一“墨水色”表示前景，
#      让观察者容易区分“模板形状”和“真正输出颜色来源”。
#
# g) 变化的可推断性
#    - 颜色变化不是按颜色编号做算术，而是由输入中显式图例给出。
#    - 输出只使用 LL 调色板中的颜色；模板前景与背景的区别是“保留源色”还是“映射到配对色”，
#      并且这种差异会在多个重复块与多个局部坐标上重复出现，足以从少量样例归纳。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
LL_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 500
PATTERN_MAX_RETRIES = 60
MIN_MIRROR_DELTA = 2
MIN_LOCAL_DELTA = 2
MIN_RECOLOR_DELTA = 2


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
    if mode == "rows":
        rows, cols = list(range(q)), list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar, ac = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _mirrored_local_index(idx: int, tile_size: int, block_idx: int) -> int:
    """Return the local coordinate inside one repeated block under mirror tiling."""
    local = int(idx % tile_size)
    if int(block_idx) % 2 == 1:
        return int(tile_size - 1 - local)
    return local


def _mirrored_local_coords(q: int, r: int, c: int) -> tuple[int, int]:
    """Map one output coordinate to the mirrored local coordinate inside the repeated tile."""
    block_r = int(r // q)
    block_c = int(c // q)
    local_r = _mirrored_local_index(int(r), q, block_r)
    local_c = _mirrored_local_index(int(c), q, block_c)
    return local_r, local_c


def _cyclic_recolor_map(fill_palette: list[int]) -> dict[int, int]:
    """Create a visible non-identity permutation on the LL palette."""
    k = len(fill_palette)
    assert k >= 2
    return {
        int(fill_palette[i]): int(fill_palette[(i + 1) % k])
        for i in range(k)
    }


def _infer_recolor_map(
    upper_left: np.ndarray,
    upper_right: np.ndarray,
    fill_tile: np.ndarray,
    bg: int,
) -> dict[int, int]:
    """Read the top-row legend: UL lists source colors, UR lists paired replacement colors."""
    q = int(upper_left.shape[1])
    fill_colors = {int(v) for v in np.unique(fill_tile)}
    mapping: dict[int, int] = {}
    for c in range(q):
        src = int(upper_left[0, c])
        dst = int(upper_right[0, c])
        if src in fill_colors and dst in fill_colors:
            mapping[src] = dst
        else:
            break
    assert len(mapping) >= 2
    assert set(mapping) == fill_colors
    assert set(mapping.values()) == fill_colors
    assert all(src != bg and dst != bg for src, dst in mapping.items())
    assert all(mapping[src] != src for src in mapping)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top quadrants as size controls and a color legend, then render the mirrored lower template entirely from the lower-left palette.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the background color.
    - The upper-left and upper-right quadrants still set output height and width by counting non-black cells.
    - Their top rows also contain a left-to-right legend: matching columns pair each lower-left source color with its replacement color.
    - The lower-left quadrant is a fully colored source tile, and the lower-right quadrant is a stencil tile or an empty tile.

    Rule:
    - Output height equals the number of non-black cells in the upper-left quadrant.
    - Output width equals the number of non-black cells in the upper-right quadrant.
    - If the lower-right stencil is empty, repeat the lower-left tile directly over the output.
    - Otherwise, repeat the stencil with mirror tiling: odd repeated block rows are vertically flipped and odd block columns are horizontally flipped.
    - At each output cell, use that mirrored local coordinate inside both lower tiles. If the mirrored stencil cell is black, output the lower-left source color there; if it is non-black, output the paired replacement color from the legend for that same lower-left source color.

    Key insight:
    - The lower-right tile contributes only mirrored geometry, while every visible output color comes from the lower-left palette: plain source colors show through stencil holes, and stencil foreground cells apply the legend's color permutation to those same mirrored local source colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % qh, c % qw])
        return out

    recolor_map = _infer_recolor_map(upper_left, upper_right, fill_tile, bg)
    for r in range(out_h):
        for c in range(out_w):
            local_r, local_c = _mirrored_local_coords(qh, r, c)
            source_color = int(fill_tile[local_r, local_c])
            assert source_color in recolor_map
            stencil_value = int(stencil[local_r, local_c])
            out[r, c] = source_color if stencil_value == bg else int(recolor_map[source_color])

    return out


def _fill_is_informative(tile: np.ndarray, *, need_v: bool, need_h: bool) -> bool:
    """Require the LL source texture to visibly change under every active reflection."""
    if len(np.unique(tile)) < 2:
        return False
    if need_v and np.array_equal(tile, tile[::-1, :]):
        return False
    if need_h and np.array_equal(tile, tile[:, ::-1]):
        return False
    return True


def _make_lower_left_base(q: int, palette: list[int]) -> np.ndarray:
    """Build a structured non-black local source tile."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "diagonal", "checker"])

    if mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    else:
        rr, cc = np.mgrid[:q, :q]
        if mode == "diagonal":
            idx = (rr + cc + random.randrange(k)) % k
        else:
            idx = (rr + 2 * cc + random.randrange(k)) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    # A couple of accents are enough to break unwanted symmetries without making LL noisy.
    accent_coords = [(r, c) for r in range(q) for c in range(q)]
    random.shuffle(accent_coords)
    accent_count = min(len(accent_coords), max(2, k))
    for i, (r, c) in enumerate(accent_coords[:accent_count]):
        tile[r, c] = palette[(i + 1) % k]

    missing = [color for color in palette if color not in set(map(int, np.unique(tile)))]
    if missing:
        fix_coords = list(accent_coords)
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_lower_left(q: int, palette: list[int], *, need_v: bool, need_h: bool) -> np.ndarray:
    """Build an LL tile whose reflected copies remain visibly different."""
    for _ in range(PATTERN_MAX_RETRIES):
        tile = _make_lower_left_base(q, palette)
        if _fill_is_informative(tile, need_v=need_v, need_h=need_h):
            return tile

    tile = np.full((q, q), palette[0], dtype=int)
    for r in range(q):
        for c in range(q):
            tile[r, c] = palette[(r + 2 * c) % len(palette)]
    if q > 1:
        tile[0, 0] = palette[0]
        tile[0, q - 1] = palette[1 % len(palette)]
        tile[q - 1, 0] = palette[(2 if len(palette) > 2 else 0)]
    return tile


def _stencil_is_informative(tile: np.ndarray, *, need_v: bool, need_h: bool) -> bool:
    """Require the stencil mask to change under every active reflection."""
    fg = tile != BLACK
    if not np.any(fg) or np.all(fg):
        return False
    if need_v and np.array_equal(fg, fg[::-1, :]):
        return False
    if need_h and np.array_equal(fg, fg[:, ::-1]):
        return False
    return True


def _make_stencil_base(q: int, ink_color: int) -> np.ndarray:
    """Build a candidate stencil tile; its non-black color is decorative only."""
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = ink_color
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t : q - t, t : q - t] = False
        tile[mask] = ink_color
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = ink_color
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = ink_color
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = min(area - 1, max(lo, int(STENCIL_FG_FRAC[1] * area)))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = ink_color
    return tile


def _make_stencil(q: int, ink_color: int, *, need_v: bool, need_h: bool) -> np.ndarray:
    """Build an LR stencil whose mirrored repetitions visibly differ."""
    for _ in range(PATTERN_MAX_RETRIES):
        tile = _make_stencil_base(q, ink_color)
        if _stencil_is_informative(tile, need_v=need_v, need_h=need_h):
            return tile

    tile = np.zeros((q, q), dtype=int)
    tile[0, 0] = ink_color
    if q > 1:
        tile[0, 1] = ink_color
        tile[q - 1, 0] = ink_color
    return tile


def _make_control_with_legend(
    q: int,
    count: int,
    legend_colors: list[int],
    filler_pool: list[int],
) -> np.ndarray:
    """Build a control quadrant with exact foreground count while reserving row 0 for the legend."""
    k = len(legend_colors)
    assert 2 <= k <= q
    assert k <= count <= q * q

    tile = np.full((q, q), BLACK, dtype=int)
    tile[0, :k] = np.array(legend_colors, dtype=int)

    remaining = count - k
    if remaining == 0:
        return tile

    filler_k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else min(2, len(filler_pool), remaining)
    filler_palette = _palette(filler_k, filler_pool)
    mode = random.choice(["compact", "rows", "random"])
    coords = [rc for rc in _ordered_coords(q, mode) if not (rc[0] == 0 and rc[1] < k)]
    for r, c in coords[:remaining]:
        tile[r, c] = random.choice(filler_palette)
    return tile


def _expected_output_combo(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    recolor_map: dict[int, int],
) -> np.ndarray:
    """Oracle for the fused D5+D6+D12 rule."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % q, c % q])
        return out

    for r in range(out_h):
        for c in range(out_w):
            local_r, local_c = _mirrored_local_coords(q, r, c)
            source_color = int(fill_tile[local_r, local_c])
            out[r, c] = (
                source_color
                if int(stencil[local_r, local_c]) == BLACK
                else int(recolor_map[source_color])
            )
    return out


def _expected_output_no_mirror(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    recolor_map: dict[int, int],
) -> np.ndarray:
    """Ablation oracle: D6+D12 only, without D5 mirror tiling."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            local_r = int(r % q)
            local_c = int(c % q)
            source_color = int(fill_tile[local_r, local_c])
            out[r, c] = (
                source_color
                if int(stencil[local_r, local_c]) == BLACK
                else int(recolor_map[source_color])
            )
    return out


def _expected_output_coarse_fill(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    recolor_map: dict[int, int],
) -> np.ndarray:
    """Ablation oracle: D5+D12 only, using the original coarse LL lookup."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            local_r, local_c = _mirrored_local_coords(q, r, c)
            coarse_r = int(r // q)
            coarse_c = int(c // q)
            source_color = int(fill_tile[coarse_r, coarse_c])
            out[r, c] = (
                source_color
                if int(stencil[local_r, local_c]) == BLACK
                else int(recolor_map[source_color])
            )
    return out


def _expected_output_template_color(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Ablation oracle: D5+D6 only, keeping the mirrored stencil's own foreground color."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            local_r, local_c = _mirrored_local_coords(q, r, c)
            stencil_value = int(stencil[local_r, local_c])
            out[r, c] = (
                int(fill_tile[local_r, local_c])
                if stencil_value == BLACK
                else stencil_value
            )
    return out


def _metric_deltas(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    recolor_map: dict[int, int],
) -> tuple[int, int, int]:
    """Return how many cells are changed by mirror tiling, local lookup, and recoloring."""
    combo = _expected_output_combo(fill_tile, stencil, out_h, out_w, recolor_map)
    no_mirror = _expected_output_no_mirror(fill_tile, stencil, out_h, out_w, recolor_map)
    coarse = _expected_output_coarse_fill(fill_tile, stencil, out_h, out_w, recolor_map)
    template_color = _expected_output_template_color(fill_tile, stencil, out_h, out_w)
    return (
        int(np.count_nonzero(combo != no_mirror)),
        int(np.count_nonzero(combo != coarse)),
        int(np.count_nonzero(combo != template_color)),
    )


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate fused sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    if random.random() < STENCIL_EMPTY_PROB:
        return None

    ll_hi = min(LL_COLOR_RANGE[1], q)
    ll_k = random.randint(LL_COLOR_RANGE[0], ll_hi)
    fill_palette = _palette(ll_k)
    recolor_map = _cyclic_recolor_map(fill_palette)

    ink_pool = [c for c in COLORS if c not in fill_palette]
    if not ink_pool:
        return None
    ink_color = random.choice(ink_pool)

    control_pool = [c for c in COLORS if c not in set(fill_palette) | {ink_color}]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1, ll_k)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        and ceil(out_h / q) >= 2
        and ceil(out_w / q) >= 2
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)
    tile_rows = int(ceil(out_h / q))
    tile_cols = int(ceil(out_w / q))

    fill_tile = _make_lower_left(q, fill_palette, need_v=tile_rows >= 2, need_h=tile_cols >= 2)
    stencil = _make_stencil(q, ink_color, need_v=tile_rows >= 2, need_h=tile_cols >= 2)
    if not _fill_is_informative(fill_tile, need_v=tile_rows >= 2, need_h=tile_cols >= 2):
        return None
    if not _stencil_is_informative(stencil, need_v=tile_rows >= 2, need_h=tile_cols >= 2):
        return None

    mirror_delta, local_delta, recolor_delta = _metric_deltas(
        fill_tile,
        stencil,
        out_h,
        out_w,
        recolor_map,
    )
    if mirror_delta < MIN_MIRROR_DELTA:
        return None
    if local_delta < MIN_LOCAL_DELTA:
        return None
    if recolor_delta < MIN_RECOLOR_DELTA:
        return None

    upper_left = _make_control_with_legend(q, out_h, fill_palette, control_pool)
    upper_right = _make_control_with_legend(
        q,
        out_w,
        [recolor_map[color] for color in fill_palette],
        control_pool,
    )

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output_combo(fill_tile, stencil, out_h, out_w, recolor_map)
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
    raise RuntimeError("failed to generate a valid fused input")
