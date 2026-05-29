# Combo variant of e3721c99 (index 27)
# combo_spec: D2/0+D4/3+D12/1
# Name: 多色精灵+删除未匹配+渐变色阶

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis
# ----------------
# a) 具体化与行为追踪:
#    - 原题 workflow 依赖两条捷径:
#      1. `g == 5` 直接把所有 source sprite 抓出来。
#      2. legend 的颜色就是输出颜色，因此 `_legend_mapping()` 返回 `holes -> legend_color`。
#    - 融合后两条都失效。workspace 里的 source sprite 本身就是多色，且这些颜色与 legend
#      调色板重叠；同时 legend 颜色只负责声明“哪些洞数有效”，真正输出颜色来自固定色阶。
#    - 一个具体 trace:
#      panel 里出现 1/2/3 洞 legend，颜色分别是粉/黄/青；workspace 有一个红色 2 洞 sprite
#      和一个灰色 5 洞 sprite。
#      原题: 只能处理灰色 sprite，2 洞若存在会被涂成对应 legend 的黄色。
#      融合后: 先找唯一贴边 divider，把图分成小 panel 与大 workspace；panel 中 1/2/3 洞
#      仅声明合法洞数集合 `{1,2,3}`。红色 2 洞 sprite 被改成固定梯度色 3(绿)，5 洞 sprite
#      因不在集合中被擦成背景。
#    - 差异的本质是: “谁要处理” 改成空间分区识别；“涂成什么颜色” 改成洞数到全局色阶函数；
#      “未匹配怎么办” 明确为删除而不是保留原色。
# b) 复杂度校准:
#    - source 颜色控制在一个小色板内，并强制每题至少 3 种输入色，保证 D2 的视觉差异明显。
#    - legend 洞数改成连续 2-4 档，保证输出里会同时出现 2-4 个相邻梯度色，便于从少量样例
#      推断“颜色随洞数递增变化”。
#    - 每个 legend 洞数至少有 1 个匹配 sprite，同时至少有 1 个未匹配 distractor，确保“重着色”
#      与“删除”两个行为都在单题中出现，不会退化成只看其中一种。
# c) 隐式假设审查:
#    - 原 `components_mask(g == grey)` 假设 source 颜色统一；融合后完全错误。
#    - 原 `_legend_mapping(..., grey)` 假设所有非 `(bg, grey)` 前景都来自 legend；融合后也错误，
#      因为 workspace sprite 可能和 legend 共用颜色。
#    - 原 `newc = legend[hc]` 假设 legend 颜色承载答案；融合后必须改为 `gradient(hc)`。
#    - 新 solver 新增的关键假设是:
#      1. 存在且仅存在一个贴边 divider 非背景连通块；
#      2. divider 将整图切成恰好两个区域；
#      3. legend 全在小区域，source 全在大区域。
#      generator 会显式构造这些条件。
# d) 放置合法性:
#    - divider、panel 与 workspace 的几何仍沿用原题布局器，blocked 区域禁止 sprite 落入。
#    - source 之间维持至少 1 格背景间隙，避免 4 连通合并；也不接触外边框，避免被误判为 divider。
#    - sprite 仍用先造 mask 再放置的方式，并用 oracle `main(grid)==expected` 回验。
# e) 视觉可观察性:
#    - 人类能看到: panel 一侧有 2-4 个不同洞数示例，workspace 一侧有多色对象；输出里这些
#      多色对象不再复制 legend 的色，而是按洞数落到固定阶梯色，未列出的洞数则消失。
#    - 连续洞数 + 重复出现的匹配 sprite 让梯度规律比“任意洞数稀疏采样”更容易归纳。
# f) 可视化整洁度:
#    - 仍只保留 divider / legend / source 三层元素，不额外叠标记。
#    - source 不用 divider 色 1，保持分隔线始终一眼可辨；legend 和 source 靠空间分区区分。
# g) 变化可推断性:
#    - 固定梯度使用 6 档低基数映射: 0->8, 1->1, 2->3, 3->4, 4->7, 5->2。
#    - generator 避免 legend 某洞数恰好用到该洞数的梯度目标色，否则视觉上会退回“像原题那样
#      复制 legend 色”的错觉。


BG = 0
DIV = 1
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SOURCE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
HOLE_COLOR_SCALE = (8, 1, 3, 4, 7, 2)  # 0->teal, 1->blue, 2->green, 3->yellow, 4->orange, 5->red

LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = len(HOLE_COLOR_SCALE)
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
    """Whether a component touches the outer frame."""
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


def _legend_holes(
    g: np.ndarray, bg: int, divider_mask: np.ndarray, side_labels: np.ndarray, panel_label: int
) -> set[int]:
    """Read the legend-side components as the set of valid hole counts."""
    holes: set[int] = set()
    non_divider_fg = (g != bg) & ~divider_mask
    for pts in components_mask(non_divider_fg, connectivity=4):
        r, c = pts[0]
        if int(side_labels[r, c]) != panel_label:
            continue
        hc = _hole_count(_component_mask(pts))
        assert hc not in holes, f"legend hole count {hc} must appear exactly once"
        holes.add(hc)
    return holes


def _gradient_color(hole_count: int) -> int:
    """Map hole count to the fixed output gradient color."""
    assert 0 <= hole_count < len(HOLE_COLOR_SCALE), f"unsupported hole count {hole_count}"
    return int(HOLE_COLOR_SCALE[hole_count])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor workspace sprites by reading valid hole counts from a legend side, then applying a fixed gradient.

    Input structure:
    - The most common color is background, and one border-touching divider splits the grid into a smaller
      legend side and a larger workspace side.
    - The legend side contains 2-4 colored example shapes with distinct hole counts; their colors are only
      display colors and do not directly determine the output.
    - The workspace side contains separated source sprites in multiple input colors, plus some unmatched
      distractors that share the same visual style.

    Rule:
    - Identify the divider, partition the grid into legend side and workspace side, and extract the set of
      hole counts shown by the legend components.
    - For each workspace sprite, count its enclosed holes inside its own bounding box.
    - If that hole count is present in the legend set, recolor the whole sprite with the fixed hole-count
      gradient; otherwise erase the sprite to background.

    Key insight:
    - Neither the input color of a workspace sprite nor the display color of a legend patch is the answer.
      Spatial role comes from the divider split, validity comes from matching hole counts, and the final color
      comes from a global count-to-color scale.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    divider_mask, side_labels, panel_label, workspace_label = _divider_and_regions(g, bg)
    legend_holes = _legend_holes(g, bg, divider_mask, side_labels, panel_label)
    assert legend_holes, "legend-side must contain at least one legend component"

    out = g.copy()
    non_divider_fg = (g != bg) & ~divider_mask
    for pts in components_mask(non_divider_fg, connectivity=4):
        r, c = pts[0]
        if int(side_labels[r, c]) != workspace_label:
            continue
        hc = _hole_count(_component_mask(pts))
        newc = _gradient_color(hc) if hc in legend_holes else bg
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
    """
    Pick 2-4 legends whose hole counts form a consecutive run.

    Consecutive hole counts make the fixed gradient visible as an ordered ramp.
    Legend display colors are forced to differ from the gradient targets for the same holes.
    """
    del arrangement
    options = []
    for count in range(LEGEND_COUNT_RANGE[0], LEGEND_COUNT_RANGE[1] + 1):
        for start in range(MAX_HOLE_COUNT - count + 1):
            holes = tuple(range(start, start + count))
            if sum(_legend_span(h) for h in holes) + count - 1 <= max_span:
                options.append(holes)
    assert options, "legend run options must be non-empty"

    holes = list(random.choice(options))
    random.shuffle(holes)
    used_gradient = {_gradient_color(h) for h in holes}
    available_colors = [c for c in LEGEND_COLORS if c not in used_gradient]
    assert len(available_colors) >= len(holes), "need enough non-gradient legend colors"
    colors = random.sample(available_colors, len(holes))
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


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, bool, int]]:
    """
    Build shuffled (hole_count, is_match, source_color) entries.

    Assumptions encoded here:
    - every legend hole count appears at least once in the workspace,
    - at least one distractor exists,
    - source colors use at least three distinct values,
    - matched sprite input colors differ from their gradient output colors.
    """
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    assert valid_d, "sprite plan must allow at least one distractor"
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors

    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    base_items: list[tuple[int, bool]] = []
    for i, (hole_count, _) in enumerate(legends):
        for _ in range(1 + extras[i]):
            base_items.append((hole_count, True))
    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    assert distractor_holes, "legend holes must not cover the full hole range"
    for _ in range(n_distractors):
        base_items.append((random.choice(distractor_holes), False))
    random.shuffle(base_items)

    palette = _choose_source_palette()
    need_distinct = min(3, len(base_items), len(palette))
    for _ in range(20):
        plan: list[tuple[int, bool, int]] = []
        used_colors: list[int] = []
        reserved = palette.copy()
        random.shuffle(reserved)
        ok = True
        for i, (hole_count, is_match) in enumerate(base_items):
            target = _gradient_color(hole_count) if is_match else None
            choices = [c for c in palette if target is None or c != target]
            if not choices:
                ok = False
                break
            if i < need_distinct:
                distinct_choices = [c for c in reserved if c in choices]
                if not distinct_choices:
                    ok = False
                    break
                source_color = distinct_choices[0]
                reserved.remove(source_color)
            else:
                source_color = random.choice(choices)
            plan.append((hole_count, is_match, source_color))
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
    for hole_count, is_match, source_color in _sprite_plan(legends):
        sprite_mask = _make_sprite_mask(hole_count)
        if sprite_mask is None:
            return False
        items.append((sprite_mask.size, sprite_mask, hole_count, is_match, source_color))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid, dtype=bool)
    for _, sprite_mask, hole_count, is_match, source_color in items:
        loc = _random_sprite_location(blocked, occupied, sprite_mask)
        if loc is None:
            return False
        r, c = loc
        in_patch = _paint_sprite(sprite_mask, source_color)
        paste(grid, in_patch, r, c, transparent=BG)
        occupied[r : r + sprite_mask.shape[0], c : c + sprite_mask.shape[1]] |= sprite_mask
        if is_match:
            out_patch = _paint_sprite(sprite_mask, _gradient_color(hole_count))
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
    raise RuntimeError("failed to generate e3721c99 combo input")
