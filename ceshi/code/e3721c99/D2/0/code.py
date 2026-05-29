# Variant of e3721c99
# Dimension: D2 — 源对象：灰色精灵的属性与结构 (src)
# Change: 灰色连通精灵散布在网格中，每个精灵含有不同数量的洞（内部封闭背景区域）... → 多色精灵（非统一灰色）

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis
# ----------------
# variant_spec:
#   core_change:
#     - 原题把 workspace 中所有待处理对象都涂成固定灰色 5，solver 可以直接用 `g == 5`
#       取出全部源精灵。
#     - 本变体把 workspace 精灵改成“每个对象各自是单色，但不同对象之间颜色多样”，
#       且这些颜色与 legend 颜色池有重叠；因此不能再用某个固定颜色区分“源对象”。
#   concrete_pixel_behavior:
#     - divider、legend panel、legend 形状的几何结构保持原题不变。
#     - 输入仍包含一个接触外边框的 divider，它把整张图切成较小的 legend-side 和较大的
#       workspace-side。
#     - legend-side 中放置 2-4 个彩色图例形状，每个图例仍用“洞数”编码一种目标颜色。
#     - workspace-side 中放置 8-11 个互不接触的源精灵；它们不再统一为灰色，而是从小色板中
#       选取多种颜色。输出时，这些精灵按自身洞数被重涂为对应 legend 颜色；无匹配洞数者擦除。
#   behavior_trace:
#     - 原始流程:
#       1. `main()` 从所有 `g == 5` 的连通块中取出源精灵。
#       2. 其余非背景且非灰色对象被当作 legend 候选，建立 `洞数 -> 颜色` 映射。
#     - 变体流程:
#       1. `main()` 先找“唯一接触外边框的非背景连通块”，把它视为 divider。
#       2. 去掉 divider 后，整张图恰好分成两个区域：较小区域是 legend-side，较大区域是
#          workspace-side。
#       3. legend-side 的非背景连通块用于建立 `洞数 -> 颜色` 映射；workspace-side 的
#          非背景连通块才是待处理源精灵。
#     - 差异:
#       原逻辑在“用颜色先分 legend / source”这一步失效；新逻辑改为“先按 divider 分区，
#       再在分区内部找对象”。
#   complexity_calibration:
#     - 源精灵仍只依赖一个稳定特征：洞数。没有加入额外排序或多步交互，避免过度复杂。
#     - 但为了让 D2 变化肉眼可见，generator 强制每个样例中 source 至少使用 3 种颜色，
#       且匹配精灵的输入颜色总是不同于其输出目标颜色。
#     - source 颜色池与 legend 颜色池重叠，避免退化成“看颜色集合就知道谁是 source”的
#       新捷径。
#   implicit_assumption_audit:
#     - 原 `_legend_mapping(..., grey)` 假设：所有非 `(bg, grey)` 颜色都来自 legend。
#       变体后不成立，因为 workspace sprite 也可能是这些颜色。
#     - 原 `for pts in components_mask(g == grey)` 假设：所有待处理对象都恰好是灰色。
#       变体后完全失效。
#     - 原 `out[r, c] = newc` 之所以安全，是因为被写入的格子只来自灰色 source；变体后若
#       误把 legend 也纳入 source，就会破坏正确输出。因此必须先分区，再写回。
#   placement_legality:
#     - generator 保证 divider 是唯一接触外边框的非背景对象。
#     - legend 与 source 都不会碰到 divider 或外边框，因此“按 divider 切成两区”的假设成立。
#     - source 之间保持至少 1 格背景间隔，避免布尔连通块合并。
#   visual_observability:
#     - 从 input 到 output，workspace 中会看到多色对象被统一改写成 legend 给出的颜色，
#       与原题“整片灰色物体变彩色”相比差异明显。
#     - legend 仍集中在被 divider 隔开的较小区域，source 仍集中在工作区，版面整洁。
#   color_inference:
#     - 颜色变化不依赖颜色 id 的排序或算术，只依赖可见的 legend-side 拓扑样式。
#     - 由于 source 自身颜色和输出颜色无固定对应，人类可从少量样例归纳：
#       “决定输出颜色的是洞数 + divider 分区”，而不是输入颜色本身。


BG = 0
DIV = 1
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SOURCE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
SPRITE_RETRIES = 30
SPRITE_AREA_0 = (5, 25)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 12
SPRITE_MAX_W = 18
SPRITE_MAX_ASPECT = 2.5

GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding box border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _component_mask(pts: np.ndarray) -> np.ndarray:
    """Convert component coords to a tight boolean mask."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _touches_border(pts: np.ndarray, h: int, w: int) -> bool:
    """Whether any component cell lies on the outer frame."""
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _divider_and_regions(g: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Find the unique border-touching divider component and split the grid into panel/workspace sides.

    Returns:
      divider_mask, side_labels, panel_label, workspace_label
    """
    h, w = g.shape
    comps = components_mask(g != bg, connectivity=4)
    border_comps = [pts for pts in comps if _touches_border(pts, h, w)]
    assert len(border_comps) == 1, "grid must contain exactly one border-touching divider component"

    divider_mask = np.zeros_like(g, dtype=bool)
    for r, c in border_comps[0]:
        divider_mask[r, c] = True

    side_labels, n_side = label_mask(~divider_mask, connectivity=4)
    assert n_side == 2, "divider must split the grid into exactly two regions"

    sizes = {lab: int(np.sum(side_labels == lab)) for lab in range(1, n_side + 1)}
    workspace_label = max(sizes, key=sizes.get)
    panel_label = min(sizes, key=sizes.get)
    return divider_mask, side_labels, int(panel_label), int(workspace_label)


def _legend_mapping(
    g: np.ndarray, bg: int, divider_mask: np.ndarray, side_labels: np.ndarray, panel_label: int
) -> dict[int, int]:
    """Build hole_count -> legend color from components that live on the legend side."""
    mapping: dict[int, int] = {}
    non_divider_fg = (g != bg) & ~divider_mask
    for pts in components_mask(non_divider_fg, connectivity=4):
        r, c = pts[0]
        if int(side_labels[r, c]) != panel_label:
            continue
        hc = _hole_count(_component_mask(pts))
        color = int(g[r, c])
        assert hc not in mapping, f"legend hole count {hc} must map to exactly one color"
        mapping[hc] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor workspace sprites by matching their hole count to the legend side.

    Input structure:
    - bg is the most common color, and one border-touching divider isolates a smaller legend region
      from a larger workspace region.
    - The legend region contains 2-4 colored example shapes with distinct hole counts.
    - The workspace contains separated source sprites with multiple input colors; these colors are not
      themselves the clue for which objects should change.

    Rule:
    - Identify the divider, split the grid into legend-side and workspace-side, and read the legend-side
      components as a mapping from hole count to color.
    - For every non-divider component in the workspace, count its enclosed holes.
    - Recolor that entire sprite to the legend color with the same hole count, or erase it to background
      if its hole count is absent from the legend.

    Key insight:
    - The crucial change is role detection by spatial partition, not by a fixed source color; topology
      (hole count) still determines the final color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    divider_mask, side_labels, panel_label, workspace_label = _divider_and_regions(g, bg)
    legend = _legend_mapping(g, bg, divider_mask, side_labels, panel_label)
    assert legend, "legend mapping must be non-empty: legend-side must contain colored shapes"

    out = g.copy()
    non_divider_fg = (g != bg) & ~divider_mask
    for pts in components_mask(non_divider_fg, connectivity=4):
        r, c = pts[0]
        if int(side_labels[r, c]) != workspace_label:
            continue
        hc = _hole_count(_component_mask(pts))
        newc = int(legend[hc]) if hc in legend else bg
        for rr, cc in pts:
            out[rr, cc] = newc

    return out


def _legend_span(hole_count: int) -> int:
    """Width or height of one legend patch along the arrangement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan or Spanx3 legend patch."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if hole_count:
            patch[1, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if hole_count:
            patch[1::2, 1] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus one-cell gaps."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 2-4 legends with distinct hole counts that fit within max_span."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row horizontal divider with the legend panel above or below."""
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - 4)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-column vertical divider with the legend panel left or right."""
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - 4)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend placement bounds for an L-corner."""
    n = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim

    return int(dr), int(dc), int(r_min), int(r_max), int(c_min), int(c_max)


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place an L-shaped divider with the legend panel in one corner."""
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, arrangement, total
    )

    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Randomly choose and apply a divider layout."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    """Sample a random sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Find well-separated seed positions for holes inside the canvas."""
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
    positions = [
        (r, c)
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
        if margin <= r < csz - margin and margin <= c < csz - margin
    ]
    random.shuffle(positions)
    seeds: list[tuple[int, int]] = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(
    seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int
) -> set[tuple[int, int]]:
    """Grow irregular blob regions around each hole seed."""
    margin = HOLE_SEED_MARGIN
    max_hole = (
        max(2, area // (hole_count + 2))
        if hole_count <= 2
        else max(1, area // (hole_count * 2 + 1))
    )
    forbidden: set[tuple[int, int]] = set()
    for sr, sc in seeds:
        forbidden.add((sr, sc))
        blob = {(sr, sc)}
        target = random.randint(1, max(1, max_hole))
        while len(blob) < target:
            cands = []
            for r, c in blob:
                for nr, nc in neighbors4(r, c):
                    if not (margin <= nr < csz - margin and margin <= nc < csz - margin):
                        continue
                    if (nr, nc) in forbidden:
                        continue
                    if any(
                        (nnr, nnc) in forbidden and (nnr, nnc) not in blob
                        for nnr, nnc in neighbors4(nr, nc)
                    ):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            p = random.choice(cands)
            forbidden.add(p)
            blob.add(p)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    """Find the nearest non-forbidden cell to the canvas center."""
    if (center, center) not in forbidden:
        return center, center
    for d in range(1, csz):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = center + dr, center + dc
                if 0 <= nr < csz and 0 <= nc < csz and (nr, nc) not in forbidden:
                    return nr, nc
    return center, center


def _grow_mask(
    csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]
) -> np.ndarray:
    """Grow a random connected blob of target area while avoiding hole cells."""
    mask = np.zeros((csz, csz), dtype=bool)
    sr, sc = start
    mask[sr, sc] = True
    frontier = list(neighbors4(sr, sc))
    filled = 1
    while filled < area and frontier:
        idx = random.randrange(len(frontier))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        r, c = frontier.pop()
        if not (0 <= r < csz and 0 <= c < csz) or mask[r, c] or (r, c) in forbidden:
            continue
        mask[r, c] = True
        filled += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc]:
                frontier.append((nr, nc))
    return mask


def _seal_and_connect(mask: np.ndarray, forbidden: set[tuple[int, int]]) -> np.ndarray:
    """Seal gaps around forbidden cells and keep only the largest connected component."""
    csz = mask.shape[0]
    for _ in range(SEAL_ITERS):
        sealed = False
        for r, c in sorted(forbidden):
            for nr, nc in neighbors4(r, c):
                if (
                    0 <= nr < csz
                    and 0 <= nc < csz
                    and not mask[nr, nc]
                    and (nr, nc) not in forbidden
                ):
                    mask[nr, nc] = True
                    sealed = True
        if not sealed:
            break
    if np.any(mask):
        labeled, n_comp = label_mask(mask)
        if n_comp > 1:
            best = max(range(1, n_comp + 1), key=lambda lab: int(np.sum(labeled == lab)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop a sprite mask to its tight bbox and validate its topology."""
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    for r, c in forbidden:
        rmin, cmin = min(rmin, r), min(cmin, c)
        rmax, cmax = max(rmax, r), max(cmax, c)
    h, w = rmax - rmin + 1, cmax - cmin + 1
    if h > SPRITE_MAX_H or w > SPRITE_MAX_W or max(h, w) > SPRITE_MAX_ASPECT * min(h, w):
        return None
    cropped = mask[rmin : rmax + 1, cmin : cmax + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return cropped


def _make_sprite_mask(hole_count: int) -> np.ndarray | None:
    """Generate a random amorphous sprite mask with the specified hole count."""
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2

        forbidden: set[tuple[int, int]] = set()
        if hole_count > 0:
            seeds = _place_hole_seeds(hole_count, center, csz, area)
            if seeds is None:
                continue
            forbidden = _grow_hole_blobs(seeds, hole_count, area, csz)

        start = _find_start(center, csz, forbidden)
        mask = _grow_mask(csz, area, start, forbidden)
        mask = _seal_and_connect(mask, forbidden)

        result = _crop_and_validate(mask, forbidden, hole_count)
        if result is not None:
            return result
    return None


def _choose_source_palette() -> list[int]:
    """Pick the palette used by source sprites in one sample."""
    size = random.randint(3, min(5, len(SOURCE_COLORS)))
    return random.sample(SOURCE_COLORS, size)


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None, int]]:
    """Build shuffled (hole_count, target_color_or_None, source_color) entries."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors

    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    base_items: list[tuple[int, int | None]] = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            base_items.append((hole_count, color))
    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        base_items.append((random.choice(distractor_holes), None))
    random.shuffle(base_items)

    palette = _choose_source_palette()
    need_distinct = min(3, len(base_items), len(palette))
    for _ in range(20):
        plan: list[tuple[int, int | None, int]] = []
        used_colors: list[int] = []
        reserved = palette.copy()
        random.shuffle(reserved)
        ok = True
        for i, (hole_count, target_color) in enumerate(base_items):
            choices = [c for c in palette if target_color is None or c != target_color]
            assert choices, "source palette must provide a color distinct from the target"
            if i < need_distinct:
                distinct_choices = [c for c in reserved if c in choices]
                if not distinct_choices:
                    ok = False
                    break
                source_color = distinct_choices[0]
                reserved.remove(source_color)
            else:
                source_color = random.choice(choices)
            plan.append((hole_count, target_color, source_color))
            used_colors.append(source_color)
        if ok and len(set(used_colors)) >= 3:
            return plan
    raise AssertionError("source sprites must use at least three colors")


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite_mask: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement for a sprite mask."""
    h, w = sprite_mask.shape
    occupied_pad = _dilate_mask(occupied)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if r == 0 or c == 0 or r + h == n or c + w == n:
            continue
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _paint_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    """Convert a boolean sprite mask to a colored patch."""
    return np.where(mask, int(color), BG).astype(int)


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all workspace sprites."""
    items = []
    for hole_count, target_color, source_color in _sprite_plan(legends):
        sprite_mask = _make_sprite_mask(hole_count)
        if sprite_mask is None:
            return False
        items.append((sprite_mask.size, sprite_mask, target_color, source_color))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid, dtype=bool)
    for _, sprite_mask, target_color, source_color in items:
        loc = _random_sprite_location(blocked, occupied, sprite_mask)
        if loc is None:
            return False
        r, c = loc
        in_patch = _paint_sprite(sprite_mask, source_color)
        paste(grid, in_patch, r, c, transparent=BG)
        occupied[r : r + sprite_mask.shape[0], c : c + sprite_mask.shape[1]] |= sprite_mask
        if target_color is not None:
            out_patch = _paint_sprite(sprite_mask, target_color)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build a sample, then verify it with the solver."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid != BG)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 D2 variant input")
