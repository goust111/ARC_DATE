# Combo variant of f931b4a8 (index 6)
# combo_spec: D3/3+D5/0
# Name: 渐变颜色源+镜像铺贴

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# a) 具体化与行为追踪
#    - 输入仍是偶数边长正方形，分成四个象限。上方两个象限只负责计数，左下提供颜色序列，
#      右下提供 stencil。
#    - D3/3 把左下从“粗粒度二维取色表”改成“单轴渐变序列”：要么每一行纯色、颜色沿行号推进，
#      要么每一列纯色、颜色沿列号推进。
#    - D5/0 把右下从普通周期平铺改成镜像平铺：块坐标 block_r = r // Q, block_c = c // Q；
#      若 block_r 为奇数则局部行索引反向，若 block_c 为奇数则局部列索引反向。
#    - 融合后，黑洞填色也要跟随镜像块的朝向，否则相邻块里的渐变仍会一直正向重复，无法形成
#      “暖→冷→暖”的折返过渡。于是若左下是列渐变 [2,5] 且 Q=2，右下某黑洞在前两块列中的
#      颜色不是 [2,5 | 2,5]，而是 [2,5 | 5,2]；与此同时 stencil 的前景纹理也会镜像翻转。
#    - 相比原题，改变的不是控制象限，也不是输出尺寸，而是“底部信息怎样被投影到输出”：
#      stencil 的几何朝向和 gradient 的相位都要看重复块奇偶性。
# b) 复杂度校准
#    - 左下只允许一种清晰的渐变方向（rows 或 cols），不引入任意二维彩噪声。
#    - 渐变颜色数限制在 2-4 个，并按循环序列扩展到 Q 长度，保证有重复支撑。
#    - 生成器强制 stencil 非空、且同时含前景和黑洞；输出至少跨越 2x2 个重复块，避免镜像退化。
#    - 生成器还会拒绝对称 stencil，避免镜像结果肉眼上等同于普通平铺。
# c) 隐式假设审查
#    - 原 main() 里 out_h/out_w 的计数前提未变，仍然成立。
#    - 原题对非空 stencil 使用 stencil[r % Q, c % Q]；融合后这个假设失效，必须换成按块奇偶翻转
#      后的局部索引。
#    - D3 单变体里黑洞填色使用 sequence[r % Q] 或 sequence[c % Q]；融合后若保留这个写法，
#      颜色会在每个块里都同向推进，无法体现镜像拼接，因此这里也必须改成 mirrored local index。
#    - 新 solver 额外假设左下能被无歧义地识别为“行渐变”或“列渐变”，且至少有两种颜色。
#      generator 通过构造保证这一点。
#    - 空 stencil 分支在语义上仍可定义为直接重复左下渐变源；但为了让训练样本稳定体现融合行为，
#      generator 会把空 stencil 情况禁掉。
# d) 放置合法性
#    - 四个象限仍是直接拼装，没有重叠问题。
#    - 控制象限的非黑像素数始终落在 1..Q*Q 内，所以输出尺寸合法。
#    - 输出尺寸额外要求至少覆盖 2 个块行和 2 个块列，这样镜像与渐变折返都能真实出现。
#    - stencil 构造时保证同时有黑洞和前景，并在需要的翻转方向上非对称。
# e) 视觉可观察性
#    - 人眼先能在左下看出“这是按行变色”或“按列变色”，再在输出里看到该方向上的颜色在相邻块间
#      折返；同时 stencil 的前景纹理也会左右/上下翻转。
#    - 控制象限颜色从输出调色板中排除，避免把计数色误认为规则颜色。
# f) 可视化整洁度
#    - 左下是整齐条带，右下是稀疏或规则掩模，输出只叠两层信息：镜像 stencil 纹理 + 渐变底色。
#      不同职责分区明确，不需要额外标记。
# g) 变化的可推断性
#    - 颜色变化只依赖可见的位置关系与镜像方向，不依赖颜色编号大小。
#    - 通过限制颜色种类、保证黑洞重复出现、并拒绝与 D3-only / D5-only 输出相同的样本，
#      generator 让解题者能从少量样例中同时归纳出“渐变方向”和“镜像块奇偶”两条规律。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
GRADIENT_DISTINCT_RANGE = (2, 4)
STENCIL_SINGLE_COLOR_PROB = 0.45
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.72)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400
STENCIL_ATTEMPTS = 60


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
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
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


def _mirrored_local_index(idx: int, tile_size: int) -> int:
    """Return the local coordinate inside a mirrored repeated block."""
    block_idx = int(idx // tile_size)
    local = int(idx % tile_size)
    if block_idx % 2 == 1:
        return int(tile_size - 1 - local)
    return local


def _mirrored_stencil_lookup(stencil: np.ndarray, r: int, c: int) -> int:
    """Read the mirrored-repeat stencil value at output position (r,c)."""
    q = int(stencil.shape[0])
    local_r = _mirrored_local_index(int(r), q)
    local_c = _mirrored_local_index(int(c), q)
    return int(stencil[local_r, local_c])


def _extract_gradient(fill_tile: np.ndarray) -> tuple[str, np.ndarray]:
    """Return ('rows'|'cols', 1D sequence) for a valid LL gradient source."""
    qh, qw = fill_tile.shape
    assert qh == qw and qh >= 2

    row_uniform = bool(np.all(fill_tile == fill_tile[:, :1]))
    col_uniform = bool(np.all(fill_tile == fill_tile[:1, :]))
    row_seq = fill_tile[:, 0].astype(int)
    col_seq = fill_tile[0, :].astype(int)

    if row_uniform and not col_uniform and len(set(map(int, row_seq))) >= 2:
        return "rows", row_seq
    if col_uniform and not row_uniform and len(set(map(int, col_seq))) >= 2:
        return "cols", col_seq
    raise AssertionError("lower-left quadrant must be a non-uniform row or column gradient")


def _gradient_lookup(sequence: np.ndarray, axis: str, r: int, c: int, q: int, *, mirrored: bool) -> int:
    """Read the gradient color at output position (r,c)."""
    if axis == "rows":
        idx = _mirrored_local_index(r, q) if mirrored else int(r % q)
    else:
        idx = _mirrored_local_index(c, q) if mirrored else int(c % q)
    value = int(sequence[idx])
    assert value != BLACK
    return value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size an output whose stencil texture and gradient background are both mirrored across repeated blocks.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants encode output height and width by their non-black pixel counts.
    - The lower-left quadrant is a non-black gradient source with either solid rows or solid columns, and the lower-right quadrant is a stencil tile or empty.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - Read the lower-left quadrant as a 1D color sequence: one color per row or one color per column.
    - If the lower-right stencil is empty, repeat the lower-left gradient source directly.
    - Otherwise, tile the stencil with mirror repetition: odd block rows flip vertically and odd block columns flip horizontally. Whenever the mirrored stencil cell is black, fill it from the lower-left sequence using the same mirrored local row or column phase, matching the gradient orientation.

    Key insight:
    - The top half still sets only the output size, but the bottom half now encodes two coupled rules: the stencil repeats by reflection and the gradient phase reflects with it, so adjacent blocks show both flipped texture and forward-then-back color transitions.
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

    axis, sequence = _extract_gradient(fill_tile)

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == bg:
                value = _gradient_lookup(sequence, axis, r, c, qh, mirrored=True)
            out[r, c] = value
    return out


def _make_gradient_source(q: int, palette: list[int]) -> tuple[np.ndarray, str]:
    """Build LL as a row-wise or column-wise solid-color gradient source."""
    axis = random.choice(["rows", "cols"])
    k = len(palette)
    offset = random.randrange(k)
    sequence = [palette[(offset + i) % k] for i in range(q)]
    tile = np.empty((q, q), dtype=int)

    if axis == "rows":
        for r, color in enumerate(sequence):
            tile[r, :] = color
    else:
        for c, color in enumerate(sequence):
            tile[:, c] = color
    return tile, axis


def _make_stencil_base(q: int, palette: list[int]) -> np.ndarray:
    """Build a candidate stencil tile (black holes reveal the gradient)."""
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t:q - t, t:q - t] = False
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = _random_color_array((q,), palette)
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = _random_color_array((q,), palette)
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = min(area - 1, max(lo, int(ceil(STENCIL_FG_FRAC[1] * area))))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _stencil_is_informative(stencil: np.ndarray, *, need_v: bool, need_h: bool) -> bool:
    """Require the mirrored-repeat geometry to actually alter the stencil."""
    if not np.any(stencil != BLACK):
        return False
    if not np.any(stencil == BLACK):
        return False
    if need_v and np.array_equal(stencil, stencil[::-1, :]):
        return False
    if need_h and np.array_equal(stencil, stencil[:, ::-1]):
        return False
    return True


def _fallback_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Construct a guaranteed asymmetric stencil with both holes and foreground."""
    tile = np.zeros((q, q), dtype=int)
    tile[0, 0] = palette[0]
    if q > 1:
        tile[min(1, q - 1), q - 1] = palette[min(1, len(palette) - 1)]
        tile[q - 1, min(1, q - 1)] = palette[min(2, len(palette) - 1)]
    return tile


def _make_stencil(q: int, palette: list[int], *, need_v: bool, need_h: bool) -> np.ndarray:
    """Build a stencil whose mirrored repeats visibly differ along active axes."""
    for _ in range(STENCIL_ATTEMPTS):
        tile = _make_stencil_base(q, palette)
        if _stencil_is_informative(tile, need_v=need_v, need_h=need_h):
            return tile
    tile = _fallback_stencil(q, palette)
    assert _stencil_is_informative(tile, need_v=need_v, need_h=need_h)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    max_k = min(max_colors, count, len(pool))
    k = 1 if max_k == 1 or random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, max_k)
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _fused_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Oracle for the fused mirrored-gradient rule."""
    q = int(fill_tile.shape[0])
    axis, sequence = _extract_gradient(fill_tile)
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % q, c % q])
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = _gradient_lookup(sequence, axis, r, c, q, mirrored=True)
            out[r, c] = value
    return out


def _periodic_gradient_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """D3-only ablation: periodic stencil + non-mirrored gradient phase."""
    q = int(fill_tile.shape[0])
    axis, sequence = _extract_gradient(fill_tile)
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % q, c % q])
            if value == BLACK:
                value = _gradient_lookup(sequence, axis, r, c, q, mirrored=False)
            out[r, c] = value
    return out


def _mirrored_coarse_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """D5-only ablation: mirrored stencil + original coarse fill lookup."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _mirrored_stencil_periodic_gradient_output(
    fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int
) -> np.ndarray:
    """Ablation used to confirm that mirrored gradient phase matters on top of mirrored stencil."""
    q = int(fill_tile.shape[0])
    axis, sequence = _extract_gradient(fill_tile)
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = _gradient_lookup(sequence, axis, r, c, q, mirrored=False)
            out[r, c] = value
    return out


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify how strongly the sample differs from simpler ablations."""
    g = np.asarray(grid, dtype=int)
    q = g.shape[0] // 2
    out_h = int(np.count_nonzero(g[:q, :q] != BLACK))
    out_w = int(np.count_nonzero(g[:q, q:] != BLACK))
    fill_tile = g[q:, :q]
    stencil = g[q:, q:]

    fused = _fused_output(fill_tile, stencil, out_h, out_w)
    d3_only = _periodic_gradient_output(fill_tile, stencil, out_h, out_w)
    d5_only = _mirrored_coarse_output(fill_tile, stencil, out_h, out_w)
    no_gradient_mirror = _mirrored_stencil_periodic_gradient_output(fill_tile, stencil, out_h, out_w)
    return {
        "diff_vs_d3_only": int(np.count_nonzero(fused != d3_only)),
        "diff_vs_d5_only": int(np.count_nonzero(fused != d5_only)),
        "diff_vs_no_gradient_mirror": int(np.count_nonzero(fused != no_gradient_mirror)),
    }


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate fused input."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB
    if stencil_empty:
        return None

    grad_k_lo = min(max(2, GRADIENT_DISTINCT_RANGE[0]), q)
    grad_k_hi = min(GRADIENT_DISTINCT_RANGE[1], q)
    if grad_k_lo > grad_k_hi:
        return None
    ll_k = random.randint(grad_k_lo, grad_k_hi)
    ll_palette = _palette(ll_k)

    remaining_for_stencil = [c for c in COLORS if c not in ll_palette]
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
    if len(remaining_for_stencil) < st_k:
        return None
    stencil_palette = _palette(st_k, remaining_for_stencil)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1)
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

    fill_tile, _axis = _make_gradient_source(q, ll_palette)
    stencil = _make_stencil(q, stencil_palette, need_v=tile_rows >= 2, need_h=tile_cols >= 2)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    fused = _fused_output(fill_tile, stencil, out_h, out_w)
    metrics = {
        "diff_vs_d3_only": int(np.count_nonzero(fused != _periodic_gradient_output(fill_tile, stencil, out_h, out_w))),
        "diff_vs_d5_only": int(np.count_nonzero(fused != _mirrored_coarse_output(fill_tile, stencil, out_h, out_w))),
        "diff_vs_no_gradient_mirror": int(
            np.count_nonzero(fused != _mirrored_stencil_periodic_gradient_output(fill_tile, stencil, out_h, out_w))
        ),
    }
    if min(metrics.values()) == 0:
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
    if not np.array_equal(actual, fused):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
