# Variant of f931b4a8
# Dimension: D12 — 输出中模板像素的颜色来源 (transform)
# Change: 模板非零像素在输出中保持其原始颜色值不变... → 模板像素重着色为颜色源的互补色

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis
# a) 具体化与行为追踪:
#    - 仍然是四象限输入: 左上/右上给输出高宽, 左下给粗粒度颜色源, 右下给重复模板。
#    - 新变化只发生在“模板前景像素如何着色”这一环:
#      原题里, 模板前景直接把右下象限自己的颜色抄到输出。
#      变体里, 右下象限只提供“哪里是模板前景”的遮罩; 真正的颜色来自左下象限的粗粒度颜色块,
#      并通过一个可见图例做互补色替换。
#    - 为了让映射可见, 两个上方控制象限的第 0 行最左侧连续 k 个格子组成图例:
#      左上给源颜色序列, 右上给与之同列配对的互补颜色序列。
#      例如 UL 顶行 [2,5,8], UR 顶行 [5,8,2] 表示 2->5, 5->8, 8->2。
#    - 小例子:
#      若 q=4, 某个输出位置落在粗粒度颜色块 2 上, 且模板该位置是前景,
#      原题会输出模板色(例如 7); 变体会输出 5, 因为图例给出 2->5。
#      若模板该位置是背景, 则仍输出 2。差异正好集中在 D12。
# b) 复杂度校准:
#    - 左下颜色数限制为 2 或 3, 避免单色导致“互补色”不可观察。
#    - 右下模板保持非空且同时含前景/背景, 让“直接颜色”和“互补颜色”在同一输出中反复出现。
#    - 图例只占上方一小段, 其余控制像素仍主要承担计数作用, 没有引入额外对象匹配难题。
# c) 隐式假设审查:
#    - 原 solver 在模板前景处直接读 stencil[r%q,c%q]; 这依赖“模板颜色就是答案颜色”。
#      变体里这个前提失效, 所以必须改成: 先找粗粒度源色 fill_tile[r//q,c//q], 再查图例映射。
#    - 原 solver 对上方颜色完全不关心, 只数非黑像素。变体新增假设: 顶行左侧有完整图例,
#      且图例中出现的所有源色恰好覆盖左下会被用到的颜色集合。generator 需要保证这一点。
# d) 放置合法性:
#    - 图例固定在两个上方象限的第 0 行, 不越界; 剩余控制像素只填未保留位置, 不覆盖图例。
#    - 左下仍然全非黑; 右下仍在自身象限内生成模板; 输出尺寸仍由非黑计数给出。
# e) 视觉可观察性:
#    - 模板颜色故意选成不在左下调色板中的“墨水色”, 这样观察者能直接看出输出没有沿用模板原色。
#    - 输出只含左下调色板中的颜色; 模板前景和背景的差别体现在“同一粗粒度块上是原色还是配对色”。
# f) 可视化整洁度:
#    - 图例集中在上方顶行, 不与随机控制像素混在一起。
#    - 左下维持简单条纹/对角线模式, 右下维持清晰模板, 不叠加额外标记。
# g) 颜色变化可推断性:
#    - 映射不是按颜色 id 计算, 而是由输入中显式图例给出。
#    - 每个样例都强制至少 2 个源颜色参与实际输出, 并在模板前景位置重复出现, 便于从少量样例归纳。


def _infer_recolor_map(
    upper_left: np.ndarray,
    upper_right: np.ndarray,
    fill_tile: np.ndarray,
    bg: int,
) -> dict[int, int]:
    """Read the top-row legend: UL lists source colors, UR lists their complements."""
    q = upper_left.shape[1]
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
    assert all(color != bg for color in mapping)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size an output, then color the lower-right template from the lower-left source.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants still control output height and width by foreground count.
    - Their top rows also contain a color legend: matching columns pair a lower-left source color with its complementary replacement color.
    - The lower-left quadrant is a non-black color source tile, and the lower-right quadrant is a repeated template or an empty tile.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - If the lower-right template is empty, tile the lower-left quadrant directly over the output.
    - Otherwise, repeat the lower-right template over the output. For each output cell, look up the coarse lower-left source color at (r // Q, c // Q):
      template background keeps that source color, while template foreground is recolored to the paired complementary color from the top-row legend.

    Key insight:
    - The right-lower quadrant contributes only a binary stencil when it is non-empty; the actual visible colors come from the lower-left coarse source tile and the legend encoded across the two upper quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
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
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    recolor_map = _infer_recolor_map(upper_left, upper_right, fill_tile, bg)
    for r in range(out_h):
        for c in range(out_w):
            fill_r = r // qh
            fill_c = c // qw
            assert fill_r < qh and fill_c < qw
            source_color = int(fill_tile[fill_r, fill_c])
            assert source_color in recolor_map
            value = source_color if int(stencil[r % qh, c % qw]) == bg else int(recolor_map[source_color])
            out[r, c] = value

    return out


# --------------- generator (o5x) ---------------

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
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
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


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the non-black coarse color source tile."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "rows":
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

    used = list(required_coords) if required_coords is not None else [(r, c) for r in range(q) for c in range(q)]
    random.shuffle(used)
    for color, (r, c) in zip(palette, used):
        tile[r, c] = color
    return tile


def _make_stencil(q: int, color: int) -> np.ndarray:
    """Build the template tile; its foreground color is decorative and will be ignored by the solver."""
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = color
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t : q - t, t : q - t] = False
        tile[mask] = color
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = color
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = color
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = color
    assert np.any(tile == BLACK)
    assert np.any(tile != BLACK)
    return tile


def _make_control_with_legend(
    q: int,
    count: int,
    legend_colors: list[int],
    filler_pool: list[int],
) -> np.ndarray:
    """Build a control quadrant with exact count while reserving row 0 for the legend."""
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
    coords = [
        rc
        for rc in _ordered_coords(q, mode)
        if not (rc[0] == 0 and rc[1] < k)
    ]
    for r, c in coords[:remaining]:
        tile[r, c] = random.choice(filler_palette)
    return tile


def _legend_mapping(fill_palette: list[int]) -> dict[int, int]:
    k = len(fill_palette)
    return {
        int(fill_palette[i]): int(fill_palette[(i + 1) % k])
        for i in range(k)
    }


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    recolor_map: dict[int, int],
) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    coarse = fill_tile[rr // q, cc // q]
    out = coarse.copy()
    fg_mask = stencil[rr % q, cc % q] != BLACK
    for src, dst in recolor_map.items():
        out[fg_mask & (coarse == int(src))] = int(dst)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_hi = min(LL_COLOR_RANGE[1], q)
    ll_k = random.randint(LL_COLOR_RANGE[0], ll_hi)
    fill_palette = _palette(ll_k)
    recolor_map = _legend_mapping(fill_palette)

    if stencil_empty:
        stencil_palette = []
    else:
        stencil_palette = _palette(1, [c for c in COLORS if c not in fill_palette])

    output_colors = set(fill_palette)
    template_colors = set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors | template_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w)
        for h in range(lo, hi + 1)
        for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX and (stencil_empty or ceil(h / q) * ceil(w / q) >= ll_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, fill_palette, used_fill)
    used_fill_colors = {int(fill_tile[r, c]) for r, c in used_fill}
    if used_fill_colors != set(fill_palette):
        return None

    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette[0])
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

    expected = _expected_output(fill_tile, stencil, out_h, out_w, recolor_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not stencil_empty and np.array_equal(actual, _original_output(fill_tile, stencil, out_h, out_w)):
        return None
    return grid


def _original_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Oracle for the unmodified task; used only to reject degenerate samples."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
