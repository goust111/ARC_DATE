# Combo variant of b0039139 (index 5)
# combo_spec: D3/1+D7/2
# Name: 多色计数板：颜色种类数决定副本数且各色决定副本前景

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 保留原题的四段式输入和蓝色分隔线：区段1模板、区段2计数板、区段3遗留前景色板、区段4背景色板。
# - 融合后，区段2不再是“同色若干块只负责计数”，也不再是“每个块唯一着色且一块对应一份输出”。
#   它变成：若干个彼此断开的 2 格小块，允许颜色重复；输出副本数 = 非黑颜色种类数；每种颜色只产生一份副本，
#   其前景色就是该颜色本身。
# - 具体例子：若水平布局的区段2从左到右依次看到橙、绿、橙、灰四个小块，
#   原题会输出 4 份且都用区段3的统一颜色；D7 单变体会输出 4 份，颜色依次橙/绿/橙/灰；
#   本融合变体应输出 3 份，颜色依次橙/绿/灰。第二个橙块只证明“橙色已经出现过”，不会新增副本。
# - 因此差异同时发生在两个环节：计数从“组件数”改成“颜色种类数”，着色从“统一前景”改成“按保留下来的颜色种类”。
#
# b) 复杂度校准
# - 如果区段2所有块颜色都不同，那么“颜色种类数 = 组件数”，D3/1 会退化失效；因此生成器强制至少一个颜色重复出现。
# - 如果所有输出副本仍共用一个颜色，那么 D7/2 会退化失效；因此输出必然含至少 2 种前景色，并直接复制自区段2。
# - 颜色种类数仍限制在 2-6，重复块数只额外增加 1-3 个，既能肉眼看到“组件更多但副本更少”，又不会把计数板堆得过满。
#
# c) 隐式假设审查
# - 原 `foreground = _solid_color(foreground_block)` 的前提是“所有副本共用一个前景色”；融合后失效。
#   区段3保留为遗留纯色板，但 solver 不再依赖它决定输出颜色。
# - 原 `len(components_mask(repetition_pattern != 0, ...))` 假设区段2只提供一个标量组件数；融合后失效。
#   现在必须先按空间顺序读取组件，再按“首次出现的颜色”去重。
# - 仅按颜色排序也不够，因为那会引入人类不可见的数字顺序规则。必须使用可见的空间顺序：
#   水平布局按左到右（行作次级键），垂直布局按上到下（列作次级键）。
# - 模板裁剪、背景色板、多份副本之间插入一条背景色分隔带，这些原前提仍然成立。
#
# d) 放置合法性
# - 生成器继续使用原题风格的 2 格连通块（1x2 或 2x1），并在每块周围保留至少 1 格黑色缓冲，
#   因而不同颜色的块、同颜色的重复块都不会连成一个组件。
# - 水平布局时四个区段共享高度，垂直布局时共享宽度；模板、计数板和两个色板因此仍能被精确拼接。
# - 颜色列表先决定“有哪些颜色、哪些颜色会重复”，再随机放置，避免任何越界和覆盖。
#
# e) 视觉可观察性
# - 人类能从样例中同时看到两件事：区段2的小块总数通常大于输出副本数；输出中每份副本的前景色都能在区段2里找到。
# - “首次出现顺序去重”是可观察的，因为区段2沿主轴的扫描方向与输出拼接方向一致，重复颜色会在更后面再次出现，
#   但输出不会再增加同色副本。
# - 这里没有使用颜色编号大小规律、面积大小规律或黑色特殊排序等不可视因素。
#
# f) 可视化整洁度
# - 区段1仍是单个黄色模板，区段2仍是稀疏黑底小块，区段4仍是纯色背景板；每个区域只承担一种主要语义。
# - 区段3保留为单色遗留板，但只取自已出现的区段2颜色之一，不额外制造第二套颜色编码。
# - 计数板中的重复色块彼此断开且数量有限，因此不会形成杂乱的大面积彩色噪声。
#
# g) 变化的可推断性
# - 输出颜色不做任何隐式变换，直接等于区段2中“第一次出现该颜色”的颜色类别集合。
# - 生成器保证至少两个不同颜色进入输出，因此解题者能看到“颜色映射”而不是单色偶然现象。
# - 生成器还保证至少一个重复颜色块，使“按颜色种类计数”与“按组件个数计数”在同一输入上给出不同答案。
#
# h) main() 的新隐式假设
# - solver 假设区段2中的每个组件是单色组件，且组件间有足够空隙可被 4 连通分割；生成器通过纯色 2 格块加黑色缓冲保证。
# - solver 假设输出顺序由区段2沿布局主轴的空间顺序决定；生成器对所有样例都保持同一读序。
# - solver 假设区段4是纯色共享背景；生成器继续使用纯色背景板。


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


def _ordered_unique_stencil_colors(stencil: np.ndarray, horizontal: bool) -> list[int]:
    """Read stencil blobs in layout order and keep only the first occurrence of each color."""
    entries: list[tuple[int, int, int]] = []
    for coords in components_mask(stencil != 0, connectivity=4):
        mins = coords.min(axis=0)
        r0, c0 = int(mins[0]), int(mins[1])
        color = int(stencil[r0, c0])
        entries.append((c0, r0, color) if horizontal else (r0, c0, color))

    entries.sort()
    ordered_colors: list[int] = []
    seen: set[int] = set()
    for _, _, color in entries:
        if color in seen:
            continue
        seen.add(color)
        ordered_colors.append(color)
    return ordered_colors


def _stack_tiles(tiles: list[np.ndarray], background: int, horizontal: bool) -> np.ndarray:
    if not tiles:
        return np.empty((0, 0), dtype=int)
    if len(tiles) == 1:
        return tiles[0]

    if horizontal:
        sep = np.full((tiles[0].shape[0], 1), background, dtype=int)
        pieces = [tiles[0]]
        for tile in tiles[1:]:
            pieces.extend([sep, tile])
        return np.hstack(pieces)

    sep = np.full((1, tiles[0].shape[1]), background, dtype=int)
    pieces = [tiles[0]]
    for tile in tiles[1:]:
        pieces.extend([sep, tile])
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a template once and replay it for the first occurrence of each stencil color.

    Input structure:
    - Solid blue (1) separator rows or columns divide the grid into four sections.
    - Section 1 is a black-background template; section 2 is a black counting board with
      isolated non-black blobs, and some blob colors may repeat across multiple components.
    - Section 3 is a legacy solid swatch, while section 4 is the solid background swatch.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels to get the binary template mask.
    - Read section 2's 4-connected blobs in layout order and keep only the first blob of
      each distinct non-black color.
    - For every retained color, recolor the cropped mask with that color over section 4's
      background, then concatenate the copies along the layout axis with one background
      row or column between neighbors.

    Key insight: section 2 now fuses counting and coloring: duplicate blobs of an already
    seen color prove the "count color kinds, not components" rule, while the surviving
    first-seen colors directly determine the foreground sequence of the output copies.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, _legacy_foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    horizontal = orientation == "horizontal"
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)

    colors_in_order = _ordered_unique_stencil_colors(repetition_pattern, horizontal)
    tiles = [np.where(tile_mask, color, background).astype(int) for color in colors_in_order]
    return _stack_tiles(tiles, background, horizontal)


# --------------- generator (combo) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # number of distinct stencil colors = output copies
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # usable non-black colors
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 300             # per-blob placement attempts in stencil
EXTRA_BLOBS_RANGE = (1, 3)           # repeated-color blobs; enforces components > color kinds
GEN_RETRIES = 300                    # outer generate_input retry limit


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for section in sections[1:]:
            parts.extend([sep, section])
        return np.hstack(parts)

    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for section in sections[1:]:
        parts.extend([sep, section])
    return np.vstack(parts)


def _build_stencil(component_colors: list[int], shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place one isolated 2-cell blob per entry in component_colors."""
    n_blobs = len(component_colors)
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    shuffled_colors = component_colors[:]
    random.shuffle(shuffled_colors)
    for color in shuffled_colors:
        placed = False
        for _ in range(BLOB_PLACE_RETRIES):
            br, bc = (1, 2) if random.choice([True, False]) else (2, 1)
            if h - 1 - br < 1 or w - 1 - bc < 1:
                continue

            r = random.randint(1, h - 1 - br)
            c = random.randint(1, w - 1 - bc)
            r0, c0 = max(0, r - 1), max(0, c - 1)
            r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
            if np.any(grid[r0:r1, c0:c1] != 0):
                continue

            grid[r : r + br, c : c + bc] = color
            placed = True
            break
        if not placed:
            return None

    return grid


def _sample_component_colors(n_colors: int, bg_color: int) -> list[int] | None:
    """Return a blob-color multiset with exactly n_colors distinct colors and at least one duplicate."""
    available_colors = [c for c in COLOR_POOL if c != bg_color]
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

    bg_color = random.choice(COLOR_POOL)
    blob_colors = _sample_component_colors(n_colors, bg_color)
    if blob_colors is None:
        return None

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(blob_colors, shared_dim, horizontal)
    if stencil is None:
        return None

    distinct_stencil_colors = colors_present(stencil, exclude={0})
    if len(distinct_stencil_colors) != n_colors:
        return None

    component_count = len(components_mask(stencil != 0, connectivity=4))
    if component_count <= n_colors:
        return None

    ordered_colors = _ordered_unique_stencil_colors(stencil, horizontal)
    if len(ordered_colors) != n_colors:
        return None

    tile_mask = shape != 0
    tiles = [np.where(tile_mask, color, bg_color).astype(int) for color in ordered_colors]
    output = _stack_tiles(tiles, bg_color, horizontal)

    legacy_fg_color = random.choice(ordered_colors)
    fg_swatch = _make_swatch(shared_dim, legacy_fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

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
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
