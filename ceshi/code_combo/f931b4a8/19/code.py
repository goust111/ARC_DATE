# Combo variant of f931b4a8 (index 19)
# combo_spec: D5/0+D13/2
# Name: 先铺贴再裁剪+镜像铺贴

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis
# ===============
# a) 具体化与行为追踪
#    输入仍是四象限：UL/UR 只编码控制，LL 提供粗粒度颜色，LR 提供细粒度 stencil。
#    融合后的执行顺序不再是“从原点开始直接生成输出”，而是：
#    1. 先把 LR 作为一个 QxQ 模板，在无限平面上做镜像铺贴。块坐标为 br = abs_r // Q、
#       bc = abs_c // Q；若 br 为奇数则该块做上下翻转，若 bc 为奇数则做左右翻转。
#    2. UL 非黑像素数给出 out_h，UR 非黑像素数给出 out_w。
#    3. UL 前景包围盒的 top 给出 row_offset，UR 前景包围盒的 left 给出 col_offset。
#    4. 最终输出是这个无限镜像平面上窗口 [row_offset:row_offset+out_h,
#       col_offset:col_offset+out_w] 的裁剪结果；若 stencil 某处为黑，则回填 LL 中由
#       绝对粗块坐标 ((abs_r // Q) % Q, (abs_c // Q) % Q) 选出的颜色。
#    小例子：Q=4，row_offset=1，col_offset=2，输出 6x7。原题读取的是绝对区域 [0:6,0:7]，
#    融合后读取 [1:7,2:9]。因此输出第一列可能已经落在“水平翻转后的第二块”里，
#    第一行也可能不是模板的第 0 行相位；求解者既要恢复偏移，也要识别相邻块镜像。
#
# b) 复杂度校准
#    若允许空 stencil、零偏移、单块输出、或关于横/纵轴对称的 stencil，组合会退化成原题、
#    纯 D5、或纯 D13。solver 仍保留原题的“空 stencil 时直接平铺 LL”分支，但为避免训练样例
#    过弱，生成器强制：
#    - LR 非空；
#    - row_offset > 0 且 col_offset > 0；
#    - 裁剪窗口在两个方向上都跨过至少一个块边界；
#    - stencil 同时破坏水平与垂直镜像对称；
#    - 组合输出必须同时区别于原题、只镜像不偏移、只偏移不镜像三种消融输出。
#
# c) 隐式假设审查
#    - 原 `out_h/out_w` 的读取逻辑仍成立，因为控制象限的计数语义未变。
#    - 原空 stencil 分支依赖 `fill_tile[r % Q, c % Q]`；组合后它仍存在，但要改用绝对坐标
#      `fill_tile[(row_offset+r) % Q, (col_offset+c) % Q]`。
#    - 原 `stencil[r % Q, c % Q]` 隐含“窗口从原点开始且每块朝向固定”；组合后失效，
#      必须改成对绝对坐标先判块奇偶再决定局部索引。
#    - 原 `fill_r = r // Q`, `fill_c = c // Q` 隐含“裁剪窗口起点为 0”；组合后也失效，
#      必须基于绝对坐标 `abs_r`, `abs_c` 计算粗块编号，并按 LL 的 QxQ 周期取模。
#    - 镜像只影响 stencil 的局部朝向，不改变粗块编号，所以黑洞回填不能跟着翻转。
#
# d) 放置合法性
#    控制区仍与底部图案分离，没有遮挡问题。新的合法性约束在于：
#    - 控制象限里，所有 UL 前景都放在 `r >= row_offset`，且至少一个在 `r == row_offset`；
#      UR 同理放在 `c >= col_offset` 且至少一个在 `c == col_offset`，这样 bbox top/left 唯一。
#    - 因为可用控制格数量分别是 `(Q-row_offset)*Q` 与 `Q*(Q-col_offset)`，必须筛掉超出这些
#      数量的 out_h/out_w。
#    - 裁剪窗口使用的粗块坐标数为 `ceil((row_offset+out_h)/Q)` 与 `ceil((col_offset+out_w)/Q)`；
#      这些值不超过 Q，故 `((abs_r // Q) % Q, (abs_c // Q) % Q)` 总能落回 LL 范围。
#
# e) 视觉可观察性
#    人类应能从少量样例看到两层规律同时存在：
#    - 上方控制区除计数外，还把输出窗口往右下推移，导致模板相位不是从左上角开始；
#    - 下方 stencil 在跨块处不是简单重复，而是镜像翻转。
#    因此样本必须跨 2x2 个块，并让 stencil 在横翻和竖翻后都明显改变。
#
# f) 可视化整洁度
#    任务仍保持原题的清爽四分区：上方控制、下方图案。没有新增标记层，也不靠颜色 id 编码。
#    控制区只通过前景像素的数量和边界位置传递信息，底部两块继续各司其职，视觉层次清晰。
#
# g) 变化的可推断性
#    本组合不引入任意颜色映射。颜色只承担“模板前景”和“黑洞回填”的原始职责；新增信息全部来自
#    几何关系：bbox 的 top/left、镜像块的奇偶、以及裁剪窗口的绝对相位。生成器还让 LL 与 LR 调色板
#    尽量不重叠，避免求解者误把颜色混同为新的映射规则。
#
# h) 新隐式假设与生成器对应保证
#    - `main()` 假设 UL/UR 至少各有一个前景，生成器用 out_h/out_w >= 3 保证。
#    - `main()` 假设 LL 全非黑，生成器始终保持这一点。
#    - `main()` 假设 LR 非空且镜像后可见差异；生成器强制非空并拒绝镜像对称 stencil。
#    - `main()` 假设偏移可从 bbox top/left 唯一读出；生成器通过受限放置保证。
#    - `main()` 假设窗口真的跨过镜像块边界；生成器显式筛选满足该条件的尺寸。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)
STENCIL_EMPTY_PROB = 0.0
LL_SOLID_PROB = 0.35
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.45
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.80
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 500
STENCIL_ATTEMPTS = 50
ABLATION_DIFF_MIN = 3


def _control_offsets(upper_left: np.ndarray, upper_right: np.ndarray, bg: int) -> tuple[int, int]:
    """Recover the shifted crop origin from the control quadrants."""
    ul_coords = np.argwhere(upper_left != bg)
    ur_coords = np.argwhere(upper_right != bg)
    assert ul_coords.size > 0 and ur_coords.size > 0
    return int(ul_coords[:, 0].min()), int(ur_coords[:, 1].min())


def _mirrored_local_index(abs_idx: int, tile_size: int) -> int:
    """Map an absolute coordinate to a mirrored-repeat local coordinate."""
    block_idx = int(abs_idx // tile_size)
    local = int(abs_idx % tile_size)
    if block_idx % 2 == 1:
        return int(tile_size - 1 - local)
    return local


def _mirrored_stencil_lookup(stencil: np.ndarray, abs_r: int, abs_c: int) -> int:
    """Read from an infinitely mirrored stencil plane at absolute coordinates."""
    q = int(stencil.shape[0])
    local_r = _mirrored_local_index(int(abs_r), q)
    local_c = _mirrored_local_index(int(abs_c), q)
    return int(stencil[local_r, local_c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a shifted crop window from the top half, then sample it from a mirrored stencil plane built from the bottom half.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the background in the control regions and the stencil holes.
    - The upper-left and upper-right quadrants still determine output height and width by foreground count, but they also reveal the crop origin through their foreground bbox.
    - The lower-left quadrant is a non-black coarse color tile, and the lower-right quadrant is a stencil tile repeated across the plane; it may also be empty.

    Rule:
    - Output height is the number of non-black cells in the upper-left quadrant, and output width is the number of non-black cells in the upper-right quadrant.
    - The top edge of the upper-left foreground bbox gives `row_offset`; the left edge of the upper-right foreground bbox gives `col_offset`.
    - If the stencil is empty, crop directly from the shifted periodic repetition of the lower-left tile.
    - Otherwise, conceptually mirror-tile the lower-right stencil over the infinite plane: odd block rows flip vertically and odd block columns flip horizontally.
    - Crop the `out_h x out_w` window starting at `(row_offset, col_offset)` from that infinite pattern; whenever the sampled stencil cell is black, replace it with the lower-left coarse color indexed by the absolute block coordinate `((abs_r // Q) % Q, (abs_c // Q) % Q)`.

    Key insight:
    - The solver must combine two geometric cues at once: the top half no longer just sizes the answer, it shifts the crop phase, and the lower-right pattern no longer repeats rigidly, it alternates by reflection across neighboring blocks.
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

    row_offset, col_offset = _control_offsets(upper_left, upper_right, bg)

    out = np.empty((out_h, out_w), dtype=int)
    stencil_empty = not np.any(stencil != bg)
    for r in range(out_h):
        abs_r = int(row_offset + r)
        for c in range(out_w):
            abs_c = int(col_offset + c)
            if stencil_empty:
                value = int(fill_tile[abs_r % qh, abs_c % qw])
                assert value != bg
                out[r, c] = value
                continue

            value = _mirrored_stencil_lookup(stencil, abs_r, abs_c)
            if value == bg:
                fill_r = int((abs_r // qh) % qh)
                fill_c = int((abs_c // qw) % qw)
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


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


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the coarse fill tile; all cells must stay non-black."""
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
        coords = list(required_coords) if required_coords is not None else [
            (r, c) for r in range(q) for c in range(q)
        ]
        random.shuffle(coords)
        for color, (r, c) in zip(missing, coords):
            tile[r, c] = color
    return tile


def _make_stencil_base(q: int, palette: list[int]) -> np.ndarray:
    """Build a candidate stencil tile with black holes and colored foreground."""
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
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _stencil_is_informative(stencil: np.ndarray) -> bool:
    """Require the tile to visibly change under both horizontal and vertical reflection."""
    if not np.any(stencil != BLACK):
        return False
    if np.array_equal(stencil, stencil[::-1, :]):
        return False
    if np.array_equal(stencil, stencil[:, ::-1]):
        return False
    return True


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a stencil whose mirrored repeats differ in both axes."""
    for _ in range(STENCIL_ATTEMPTS):
        tile = _make_stencil_base(q, palette)
        if _stencil_is_informative(tile):
            return tile

    tile = np.zeros((q, q), dtype=int)
    tile[0, 0] = palette[0]
    tile[0, q - 1] = palette[min(1, len(palette) - 1)]
    tile[q - 1, 0] = palette[min(2, len(palette) - 1)]
    return tile


def _compact_order(coords: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coords by distance from an anchor with mild randomness."""
    random.shuffle(coords)
    return sorted(
        coords,
        key=lambda rc: (abs(rc[0] - anchor[0]) + abs(rc[1] - anchor[1]), random.random()),
    )


def _make_control(
    q: int,
    count: int,
    max_colors: int,
    pool: list[int],
    *,
    fixed_top: int | None = None,
    fixed_left: int | None = None,
) -> np.ndarray:
    """Build a control quadrant with exact count and a readable bbox top/left."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    allowed = [
        (r, c)
        for r in range(q)
        for c in range(q)
        if (fixed_top is None or r >= fixed_top) and (fixed_left is None or c >= fixed_left)
    ]
    assert count <= len(allowed)

    rects: list[tuple[int, int, int, int]] = []
    for h in range(1, q + 1):
        for w in range(1, q + 1):
            if h * w != count:
                continue
            row_starts = [fixed_top] if fixed_top is not None else list(range(q - h + 1))
            col_starts = [fixed_left] if fixed_left is not None else list(range(q - w + 1))
            for r0 in row_starts:
                for c0 in col_starts:
                    if r0 + h <= q and c0 + w <= q:
                        rects.append((r0, c0, h, w))

    if rects and random.random() < CONTROL_RECT_PROB:
        r0, c0, h, w = random.choice(rects)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
        return tile

    if fixed_top is not None:
        boundary = [(fixed_top, c) for c in range(q)]
        anchor = random.choice(boundary)
    else:
        assert fixed_left is not None
        boundary = [(r, fixed_left) for r in range(q)]
        anchor = random.choice(boundary)

    ordered = [anchor]
    ordered.extend(_compact_order([coord for coord in allowed if coord != anchor], anchor))
    for r, c in ordered[:count]:
        tile[r, c] = random.choice(pal)
    return tile


def _render_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    row_offset: int,
    col_offset: int,
    *,
    mirrored: bool,
) -> np.ndarray:
    """Render a crop from either mirrored or ordinary periodic tiling."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    stencil_empty = not np.any(stencil != BLACK)
    for r in range(out_h):
        abs_r = int(row_offset + r)
        for c in range(out_w):
            abs_c = int(col_offset + c)
            if stencil_empty:
                out[r, c] = int(fill_tile[abs_r % q, abs_c % q])
                continue
            if mirrored:
                value = _mirrored_stencil_lookup(stencil, abs_r, abs_c)
            else:
                value = int(stencil[abs_r % q, abs_c % q])
            if value == BLACK:
                fill_r = int((abs_r // q) % q)
                fill_c = int((abs_c // q) % q)
                value = int(fill_tile[fill_r, fill_c])
            out[r, c] = value
    return out


def _valid_window_configs(q: int) -> list[tuple[int, int, int, int]]:
    """Enumerate shifted windows that keep controls legal and expose mirror boundaries."""
    hi = min(OUTPUT_DIM_MAX, q * q)
    configs: list[tuple[int, int, int, int]] = []
    for row_offset in range(1, q):
        max_h = min(hi, q * (q - row_offset))
        if max_h < OUTPUT_DIM_MIN:
            continue
        for col_offset in range(1, q):
            max_w = min(hi, q * (q - col_offset))
            if max_w < OUTPUT_DIM_MIN:
                continue
            for out_h in range(OUTPUT_DIM_MIN, max_h + 1):
                if row_offset + out_h <= q:
                    continue
                for out_w in range(OUTPUT_DIM_MIN, max_w + 1):
                    if col_offset + out_w <= q:
                        continue
                    if out_h * out_w > OUTPUT_AREA_MAX:
                        continue
                    configs.append((row_offset, col_offset, out_h, out_w))
    return configs


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid, visibly fused sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    if random.random() < STENCIL_EMPTY_PROB:
        return None

    configs = _valid_window_configs(q)
    if not configs:
        return None
    row_offset, col_offset, out_h, out_w = random.choice(configs)
    block_rows = int(ceil((row_offset + out_h) / q))
    block_cols = int(ceil((col_offset + out_w) / q))
    assert block_rows >= 2 and block_cols >= 2

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    available_for_stencil = [c for c in COLORS if c not in ll_palette]
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
    if len(available_for_stencil) < st_k:
        return None
    stencil_palette = _palette(st_k, available_for_stencil)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    used_fill = [(r, c) for r in range(block_rows) for c in range(block_cols)]
    if len(used_fill) < ll_k:
        return None
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool, fixed_top=row_offset)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool, fixed_left=col_offset)

    combo = _render_output(fill_tile, stencil, out_h, out_w, row_offset, col_offset, mirrored=True)
    original = _render_output(fill_tile, stencil, out_h, out_w, 0, 0, mirrored=False)
    mirrored_no_shift = _render_output(fill_tile, stencil, out_h, out_w, 0, 0, mirrored=True)
    shifted_periodic = _render_output(fill_tile, stencil, out_h, out_w, row_offset, col_offset, mirrored=False)

    diff_vs_original = int(np.count_nonzero(combo != original))
    diff_vs_mirrored_only = int(np.count_nonzero(combo != mirrored_no_shift))
    diff_vs_shifted_only = int(np.count_nonzero(combo != shifted_periodic))
    if min(diff_vs_original, diff_vs_mirrored_only, diff_vs_shifted_only) < ABLATION_DIFF_MIN:
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
    if not np.array_equal(actual, combo):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
