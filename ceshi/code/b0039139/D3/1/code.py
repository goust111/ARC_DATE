# Variant of b0039139
# Dimension: D3 — 连通分量计数方式 (src)
# Change: 计算区段2中色3像素的4-连通分量数量，该数量决定模板副本的重复次数... → 计数不同颜色的像素种类数

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, random_sprite


# Design analysis for the D3 variant
# ----------------------------------
# a) 具体化与行为追踪
# - 仍然保持四段式输入：模板区、计数区、前景色色板、背景色色板，分隔线仍是蓝色 1。
# - 模板区、两个色板、输出的平铺方向和模板重着色规则全部不变。
# - 唯一核心变化发生在区段2：它不再是“若干个同色连通块”，而是“若干个断开的双色元件集合”，
#   其中重复次数 N = 区段2里非黑色的不同颜色种类数。
# - 小例子：若区段2里有红色两块、绿色一块、黄色三块，那么原题会得到 6 次重复，
#   本变体必须得到 3 次重复。差异点不是空间上有几块，而是颜色上有几类。
#
# b) 复杂度校准
# - 为了避免退化回原题，生成器强制至少一种颜色在区段2中出现多次且互不接触，
#   因而“组件数 > 颜色数”在大多数样本中都显式可见。
# - 颜色种类数仍保持在 2-6 这一小范围内，保证人眼可直接数出来，不会引入新型复杂映射。
#
# c) 原 solver 的隐式假设审查
# - 原 `n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))`
#   假设“所有前景像素的组件数”就是控制变量。这个前提在变体中失效，因为同色多块必须合并为同一类。
# - 模板裁剪、双色重着色、按分隔方向拼接都不依赖组件计数方式，因此可原样保留。
# - 色板仍为纯色矩形，所以 `_solid_color()` 的多数值提取假设继续成立。
#
# d) 放置合法性
# - 区段2中的每个对象仍使用原题风格的 2 格连通块（1x2 或 2x1），放在黑底上。
# - 任意两个块之间保留至少 1 格黑色缓冲，保证它们在 4 邻接与 8 邻接下都不会误并。
# - 水平布局时所有区段共享高度；垂直布局时所有区段共享宽度；这与原题一致，因此 `_join_sections()`
#   与输出平铺逻辑不需要改。
#
# e) 视觉可观察性
# - 解题者在样例里会看到：区段2往往有比输出重复次数更多的离散小块，但这些小块只使用少量重复颜色。
# - 这样可以直接支持“按颜色类别计数，而不是按块数计数”的归纳。
#
# f) 可视化整洁度
# - 仍使用黑底、蓝色分隔线和独立色板，区段职责清晰，不在同一区域叠加不同编码。
# - 区段2只编码数量，不承载颜色映射到输出的职责，避免与区段3/4 的功能冲突。
#
# g) 颜色变化可推断性
# - 输出颜色仍只来自后两个色板，区段2的颜色只用于计数类别，不参与输出着色。
# - 生成器优先让区段2颜色与 fg/bg 不同，减少“区段2颜色决定输出颜色”的误导。
#
# h) main() 的新隐式假设
# - 新 solver 假设区段2背景为黑色且所有有效计数信号都是非黑像素。
# - 生成器通过黑底计数板、非黑色 2 格块、以及至少 2 种颜色的构造来保证该假设始终成立。


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = 1

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and repeat it according to a color-kind stencil.

    Input structure:
    - Solid blue (1) separator rows or columns split the grid into four sections.
    - Section 1 is a black-background shape template; section 2 is a multi-color counting panel.
    - Sections 3 and 4 are solid color swatches for output foreground and background.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels and recolor it with the two swatches.
    - Count how many distinct non-black colors appear anywhere in section 2.
    - Repeat the recolored crop that many times, inserting one background-colored row/column between copies.

    Key insight: The second section now encodes only a category count — duplicate blobs of the same
    color do not increase the output length, and the repeat direction still follows the separator layout.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = len(colors_present(repetition_pattern, exclude={0}))
    if n_repeats <= 1:
        return tile

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), background, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), background, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


# --------------- generator (variant) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # distinct color count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # fg/bg/stencil color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 300             # per-stencil placement attempts
EXTRA_BLOBS_RANGE = (1, 3)           # force repeated colors so components > color kinds
GEN_RETRIES = 300                    # outer generate_input retry limit


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg_color separator."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    """Create a solid color swatch. shared_dim is the fixed axis; varying axis is random."""
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    """Join sections with single-pixel separator lines of SEPARATOR_COLOR."""
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for s in sections[1:]:
            parts.extend([sep, s])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for s in sections[1:]:
        parts.extend([sep, s])
    return np.vstack(parts)


def _build_stencil(blob_colors: list[int], shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place disconnected 2-cell blobs using the requested colors on a black grid."""
    n_blobs = len(blob_colors)
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    shuffled_colors = blob_colors[:]
    random.shuffle(shuffled_colors)
    placed = 0
    for color in shuffled_colors:
        success = False
        for _ in range(BLOB_PLACE_RETRIES):
            br, bc = (1, 2) if random.choice([True, False]) else (2, 1)

            if h - 1 - br < 1 or w - 1 - bc < 1:
                continue
            r = random.randint(1, h - 1 - br)
            c = random.randint(1, w - 1 - bc)

            # Check blob cells + 1-cell neighborhood are all black.
            r0, c0 = max(0, r - 1), max(0, c - 1)
            r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
            if np.any(grid[r0:r1, c0:c1] != 0):
                continue

            grid[r : r + br, c : c + bc] = color
            success = True
            placed += 1
            break
        if not success:
            return None

    return grid if placed == n_blobs else None


def _sample_stencil_blob_colors(n_colors: int, fg_color: int, bg_color: int) -> list[int] | None:
    """Create a blob-color list with exactly n_colors distinct colors and at least one duplicate."""
    available_colors = [c for c in COLOR_POOL if c not in {fg_color, bg_color}]
    if len(available_colors) < n_colors:
        return None

    distinct_colors = random.sample(available_colors, n_colors)
    extra_blobs = random.randint(*EXTRA_BLOBS_RANGE)
    duplicate_colors = [random.choice(distinct_colors) for _ in range(extra_blobs)]
    blob_colors = distinct_colors + duplicate_colors
    random.shuffle(blob_colors)
    return blob_colors


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_colors = random.randint(*REPEAT_COUNT_RANGE)

    # Generate connected tile shape.
    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    density = random.uniform(*DENSITY_RANGE)
    shape = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )

    # Pick distinct fg/bg colors, then choose stencil colors separately.
    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    blob_colors = _sample_stencil_blob_colors(n_colors, fg_color, bg_color)
    if blob_colors is None:
        return None

    # Build output.
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_colors, bg_color, horizontal)

    # Build template: shape in (crop_h+2) x (crop_w+2) black border.
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    # Build multi-color stencil.
    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(blob_colors, shared_dim, horizontal)
    if stencil is None:
        return None

    if len(colors_present(stencil, exclude={0})) != n_colors:
        return None
    if int(np.count_nonzero(stencil)) <= 0:
        return None

    # Assemble: template | stencil | fg_swatch | bg_swatch.
    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    # Oracle check.
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after %d attempts" % GEN_RETRIES)
