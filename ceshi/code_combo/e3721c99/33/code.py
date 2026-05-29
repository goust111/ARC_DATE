# Combo variant of e3721c99 (index 33)
# combo_spec: D2/0+D3/1+D14/2
# Name: 图例嵌入内部+多色精灵+非黑色背景

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - 原题依赖三条强线索：source 固定灰色、legend 靠边集中摆放、黑色背景让洞一眼可见。
#     - 本融合版同时去掉这三条线索：source 改成多色，legend 也散布在内部，背景固定为蓝色，
#       因而 source 的洞与整张图外部空间都同色。
#     - 为了仍然可解，legend 保留“刚性模板几何”：它们总是 3 格厚的规则条形/柱形，内部蓝色孔洞
#       以交替节奏出现；source 则保持原题那种不规则连通 blob。角色区分改由几何类别承担。
#   concrete_pixel_behavior:
#     - 30x30 网格，背景不是黑色而是蓝色 1。所有孔洞都显示为蓝色，与外部背景完全一致。
#     - 图中有 2-4 个 legend 组件，全部位于内部，不接触边框；每个 legend 是单色、规范模板，
#       洞数 0-4 中互不重复，颜色互不重复。
#     - 图中还有 7-9 个 source sprite，它们也是单色连通块，但不同 sprite 的输入颜色多样，
#       且颜色池与 legend 颜色池重叠。输出时，洞数命中的 sprite 被重涂成对应 legend 颜色，
#       未命中的 distractor 擦成蓝色背景。
#   behavior_trace:
#     - 原始流程:
#       1. 用 `g == 5` 直接抓所有 source。
#       2. 用“非背景且非灰色”的彩色组件建立 legend 映射。
#     - 融合后流程:
#       1. 枚举所有非背景单色组件。
#       2. 若组件的布尔形状与某个规范 legend 模板完全一致，则把它记作 `洞数 -> 颜色` 示例。
#       3. 其余组件一律视为 source，按自身洞数去 legend 映射中查目标颜色。
#     - 差异:
#       颜色和位置都不再能先验地区分对象角色；solver 必须先做“模板类 vs blob 类”分类，
#       再做原题的洞数匹配。
#   complexity_calibration:
#     - 只保留一个新的稳定判别维度：legend 的几何模板是刚性的，source 的几何是有机的。
#       这样变化足够明显，但没有再叠加排序或多步交互。
#     - legend 洞数仍限制在 0-4、数量仍为 2-4，source 仍然 7-9 个，并保留每个 legend 至少有
#       一个匹配 sprite 的支撑，保证少量样例里能观察到重复规律。
#     - generator 强制 source 至少使用 3 种输入颜色，且至少有一种 source 颜色与某个 legend
#       颜色重叠，从而避免退化成“颜色集合自动分组”。
#   implicit_assumption_audit:
#     - 原 `_legend_mapping(g, bg, grey)` 假设：只要不是 `(bg, grey)` 就是 legend。现在失效，
#       因为 source 也可能使用这些颜色。
#     - 原 `components_mask(g == grey)` 假设：所有待处理对象都是固定灰色。现在完全失效。
#     - 原代码默认 legend 靠边摆放，因此“非边框大组件 + 彩色”足以当 legend。现在也失效，
#       因为 legend/source 都在内部混放。
#     - 新 solver 的隐式前提改为：legend 组件恰好等于规范模板，source 组件不等于该模板。
#       generator 会显式拒绝任何偶然长成 legend 模板的 source mask。
#   placement_legality:
#     - 所有组件都与边框保留至少 1 格背景，legend 额外保留更大的内边距，确保“内部嵌入”可见。
#     - 放置时只要求前景之间保持 1 格缓冲，但会额外记录每个已放置组件的真实封闭洞区，防止后续
#       对象被塞进洞里从而破坏拓扑。
#     - source mask 在生成后会重新验洞数，并且再次过滤掉任何与 legend 模板同形的偶然结果。
#   visual_observability:
#     - 人类可见到两类明显不同的对象：细长规则模板是示例，较大有机 blob 是待处理目标。
#     - 背景与洞同为蓝色，所以“洞数”仍然是可见拓扑，但不再依赖黑色对比；同时 source 多色且
#       内嵌混放，肉眼能明显看出和原题的版式差异。
#   visual_cleanliness:
#     - 虽然 legend/source 混在同一区域，但每个组件都有至少 1 格蓝色缓冲，且 legend 尺寸小而
#       规则、source 尺寸更大更不规则，局部阅读顺序仍清楚。
#     - 不允许对象进入别的对象 bbox，因此不会出现“洞内再嵌一个对象”的脏布局。
#   inferability:
#     - 颜色变化只由可见的 legend 例子提供，不依赖颜色 id 的数值语义。
#     - 同一张图里通常会出现多个 source 共享某个洞数、多个 source 颜色彼此不同但输出相同，
#       这足以支撑解题者从 2-4 组样例归纳“看洞数，不看输入色；看模板几何，不看位置”。


BG = 1
GRID_SIZE = 30
OBJECT_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 5
LEGEND_BORDER_MARGIN = 2
SPRITE_BORDER_MARGIN = 1

TOTAL_SPRITES_RANGE = (7, 9)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 1
SOURCE_PALETTE_SIZE_RANGE = (3, 5)

SPRITE_RETRIES = 80
SPRITE_AREA_0 = (5, 20)
SPRITE_AREA_LOW_MIN = 7
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

GEN_RETRIES = 400
_STRUCT8 = generate_binary_structure(2, 2)


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding-box border.
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


def _legend_span(hole_count: int) -> int:
    """Width or height of one legend patch along its long axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_mask(hole_count: int, arrangement: str) -> np.ndarray:
    """Boolean geometry of one canonical legend component."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        mask = np.ones((3, span), dtype=bool)
        if hole_count:
            mask[1, 1::2] = False
    else:
        mask = np.ones((span, 3), dtype=bool)
        if hole_count:
            mask[1::2, 1] = False
    return mask


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Colored patch for one canonical legend component."""
    return np.where(_legend_mask(hole_count, arrangement), int(color), BG).astype(int)


def _detect_legend_hole_count(mask: np.ndarray) -> int | None:
    """Return the encoded hole count if mask exactly matches a legend template."""
    for hole_count in range(MAX_HOLE_COUNT):
        for arrangement in ("horizontal", "vertical"):
            templ = _legend_mask(hole_count, arrangement)
            if mask.shape == templ.shape and np.array_equal(mask, templ):
                return hole_count
    return None


def _monochrome_components(g: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    """Enumerate all non-background monochrome components as (color, coords)."""
    comps: list[tuple[int, np.ndarray]] = []
    for color in sorted(c for c in set(map(int, np.unique(g))) if c != bg):
        for pts in components_mask(g == color, connectivity=4):
            comps.append((int(color), pts))
    return comps


def _find_legends(g: np.ndarray, bg: int) -> tuple[dict[int, int], set[int]]:
    """
    Detect embedded legend components by exact template geometry.

    Returns:
      hole_count -> legend color mapping, and indices of components that are legends.
    """
    legend: dict[int, int] = {}
    legend_ids: set[int] = set()
    comps = _monochrome_components(g, bg)
    for idx, (color, pts) in enumerate(comps):
        hole_count = _detect_legend_hole_count(_component_mask(pts))
        if hole_count is None:
            continue
        assert hole_count not in legend, f"legend hole count {hole_count} must be unique"
        legend[hole_count] = int(color)
        legend_ids.add(idx)
    return legend, legend_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor interior multi-color sprites by matching their hole count to embedded legend templates.

    Input structure:
    - The most common color is a non-black background, and in generated samples it is blue; both the outer
      field and every enclosed hole share that same background color.
    - A few small legend components are embedded inside the grid instead of sitting on an edge panel; each
      legend is a rigid 3-cell-thick template whose blue gaps encode a distinct hole count.
    - The remaining foreground components are separated blob-like sprites with varied input colors, and
      those colors may overlap the legend palette.

    Rule:
    - Find the components whose masks exactly match one of the canonical legend templates and read them as
      a mapping from hole count to output color.
    - For every other foreground component, count the enclosed background regions inside its tight bbox.
    - Recolor that whole sprite to the legend color with the same hole count, or erase it to background if
      that hole count is absent from the legend set.

    Key insight:
    - After removing the old cues of fixed grey sources, edge-positioned legends, and black holes, the
      stable distinction is geometric role: exact bar templates are examples, irregular blobs are targets,
      and topology still determines the final color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = _monochrome_components(g, bg)
    legend, legend_ids = _find_legends(g, bg)
    assert legend, "legend mapping must be non-empty"

    out = g.copy()
    for idx, (_, pts) in enumerate(comps):
        if idx in legend_ids:
            continue
        sub = _component_mask(pts)
        hole_count = _hole_count(sub)
        newc = int(legend[hole_count]) if hole_count in legend else bg
        for r, c in pts:
            out[r, c] = newc
    return out


def _sprite_area(hole_count: int) -> int:
    """Sample a target sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 20 + 8 * hole_count)
    return random.randint(9 + 2 * hole_count, 16 + 3 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Pick separated interior seeds for hole blobs."""
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
    """Grow irregular forbidden regions that will become sprite holes."""
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
    """Find the nearest allowed start cell to the canvas center."""
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
    """Grow one connected foreground blob while avoiding hole cells."""
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
    """Seal gaps around holes and keep only the largest connected foreground part."""
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
        labeled, n_comp = label_mask(mask, connectivity=4)
        if n_comp > 1:
            best = max(range(1, n_comp + 1), key=lambda lab: int(np.sum(labeled == lab)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop a generated mask, validate its size/topology, and reject legend-shaped accidents."""
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
    if _detect_legend_hole_count(cropped) is not None:
        return None
    return cropped


def _make_sprite_mask(hole_count: int) -> np.ndarray | None:
    """Generate one amorphous sprite mask with the requested hole count."""
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


def _enclosed_hole_mask(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the enclosed background cells inside one component bbox."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = np.zeros((h, w), dtype=bool)
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            cells = []
            touches = False
            while st:
                r, c = st.pop()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                for r, c in cells:
                    holes[r, c] = True
    return holes


def _choose_legends() -> list[tuple[int, int, str]]:
    """Choose distinct hole-count/color legend entries with independent orientations."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    holes = random.sample(range(MAX_HOLE_COUNT), count)
    colors = random.sample(OBJECT_COLORS, count)
    return [
        (hole_count, color, random.choice(["horizontal", "vertical"]))
        for hole_count, color in zip(holes, colors)
    ]


def _choose_source_palette(legend_colors: set[int]) -> list[int]:
    """Choose a source palette that overlaps the legend palette but is not identical to it."""
    size = random.randint(*SOURCE_PALETTE_SIZE_RANGE)
    overlap_n = random.randint(1, min(2, len(legend_colors), size - 1))
    overlap = random.sample(sorted(legend_colors), overlap_n)
    others_pool = [c for c in OBJECT_COLORS if c not in overlap]
    others = random.sample(others_pool, size - overlap_n)
    palette = overlap + others
    random.shuffle(palette)
    return palette


def _sprite_plan(legends: list[tuple[int, int, str]]) -> list[tuple[int, int | None, int]]:
    """Build shuffled (hole_count, target_color_or_None, source_color) items."""
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
    for i, (hole_count, color, _) in enumerate(legends):
        for _ in range(1 + extras[i]):
            base_items.append((hole_count, color))
    legend_holes = {hole_count for hole_count, _, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        base_items.append((random.choice(distractor_holes), None))
    random.shuffle(base_items)

    legend_colors = {color for _, color, _ in legends}
    palette = _choose_source_palette(legend_colors)
    need_distinct = min(3, len(base_items), len(palette))

    for _ in range(40):
        plan: list[tuple[int, int | None, int]] = []
        used_colors: list[int] = []
        reserved = palette.copy()
        random.shuffle(reserved)
        ok = True
        for i, (hole_count, target_color) in enumerate(base_items):
            choices = [c for c in palette if target_color is None or c != target_color]
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
            used_colors.append(source_color)
            plan.append((hole_count, target_color, source_color))
        if ok and len(set(used_colors)) >= 3 and set(used_colors) & legend_colors:
            return plan
    raise AssertionError("source sprites must stay multi-color and overlap legend colors")


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    """Convert a boolean mask into a colored patch on the blue background."""
    return np.where(mask, int(color), BG).astype(int)


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """One-cell 8-neighborhood dilation."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_location(
    occupied: np.ndarray, blocked_holes: np.ndarray, mask: np.ndarray, border_margin: int
) -> tuple[int, int] | None:
    """Find a location with no foreground contact and no overlap with preserved holes."""
    h, w = mask.shape
    occupied_pad = _dilate_mask(occupied)
    candidates = [
        (r, c)
        for r in range(border_margin, GRID_SIZE - border_margin - h + 1)
        for c in range(border_margin, GRID_SIZE - border_margin - w + 1)
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if np.any(occupied_pad[r : r + h, c : c + w] & mask):
            continue
        if np.any(blocked_holes[r : r + h, c : c + w] & mask):
            continue
        return int(r), int(c)
    return None


def _register_component(
    occupied: np.ndarray, blocked_holes: np.ndarray, mask: np.ndarray, r: int, c: int
) -> None:
    """Record a placed component's foreground and its enclosed hole cells."""
    h, w = mask.shape
    occupied[r : r + h, c : c + w] |= mask
    hole_mask = _enclosed_hole_mask(mask)
    blocked_holes[r : r + h, c : c + w] |= hole_mask


def _place_legends(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    blocked_holes: np.ndarray,
    legends: list[tuple[int, int, str]],
) -> bool:
    """Place all legend components in interior positions."""
    items = []
    for hole_count, color, arrangement in legends:
        mask = _legend_mask(hole_count, arrangement)
        patch = _legend_patch(hole_count, color, arrangement)
        items.append((patch.size, mask, patch))
    items.sort(key=lambda x: x[0], reverse=True)

    for _, mask, patch in items:
        loc = _random_location(occupied, blocked_holes, mask, LEGEND_BORDER_MARGIN)
        if loc is None:
            return False
        r, c = loc
        paste(grid, patch, r, c, transparent=BG)
        paste(expected, patch, r, c, transparent=BG)
        _register_component(occupied, blocked_holes, mask, r, c)
    return True


def _place_sprites(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    blocked_holes: np.ndarray,
    legends: list[tuple[int, int, str]],
) -> bool:
    """Generate and place all source sprites."""
    items = []
    for hole_count, target_color, source_color in _sprite_plan(legends):
        sprite_mask = _make_sprite_mask(hole_count)
        if sprite_mask is None:
            return False
        items.append((sprite_mask.size, sprite_mask, target_color, source_color))
    items.sort(key=lambda x: x[0], reverse=True)

    for _, sprite_mask, target_color, source_color in items:
        loc = _random_location(occupied, blocked_holes, sprite_mask, SPRITE_BORDER_MARGIN)
        if loc is None:
            return False
        r, c = loc
        paste(grid, _paint_mask(sprite_mask, source_color), r, c, transparent=BG)
        if target_color is not None:
            paste(expected, _paint_mask(sprite_mask, target_color), r, c, transparent=BG)
        _register_component(occupied, blocked_holes, sprite_mask, r, c)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Build one sample and verify it against the solver."""
    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    expected = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    blocked_holes = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    legends = _choose_legends()
    if not _place_legends(grid, expected, occupied, blocked_holes, legends):
        return None
    if not _place_sprites(grid, expected, occupied, blocked_holes, legends):
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
    raise RuntimeError("failed to generate e3721c99 combo variant input")
