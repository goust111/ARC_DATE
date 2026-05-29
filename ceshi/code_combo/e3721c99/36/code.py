# Combo variant of e3721c99 (index 36)
# combo_spec: D4/4+D11/1+D12/1
# Name: 迭代消除+删除匹配+删除未匹配
#
# Design analysis
# ---
# concretization_and_trace:
#   input_visual:
#     - "30x30 黑底网格，边缘/角落有分隔线，图例条上按洞数从小到大摆放 3-4 个彩色模板。"
#     - "工作区散布 8-11 个灰色精灵；有些洞数属于图例前缀中的若干值，有些是完全不在图例里的干扰项。"
#   small_trace:
#     - "例：图例洞数按顺序为 [0,2,4,5]，灰精灵只有 2-hole 匹配项和 1-hole 干扰项。"
#     - "原题：2-hole 灰精灵会被重涂成对应颜色，1-hole 干扰项被删，所有图例都保留。"
#     - "本融合版第 1 轮处理 0-hole：删掉所有当前未列出的干扰项（这里是 1-hole），同时移除 0-hole 图例。"
#     - "第 2 轮处理 2-hole：删掉 2-hole 灰精灵，同时移除 2-hole 图例。此时工作区清空，迭代停止，4-hole/5-hole 图例保留。"
#     - "因此最终输出不是简单的'删光灰精灵'，而是'删光灰精灵 + 删除从最小洞数起、直到工作区清空为止的图例前缀'。"
# complexity_calibration:
#   - "保留 3-4 个图例，且至少保留 1 个高洞数幸存图例，避免退化成整条图例全空或全留。"
#   - "强制至少 1 个前缀图例在没有直接匹配精灵的情况下也被删除，这样才能把 D11 的迭代顺序和简单的'按是否匹配删图例'区分开。"
#   - "继续保留 1 个以上未匹配干扰项，让 D12 的删除未匹配分支在每个样本中都可见。"
# implicit_assumption_review:
#   - "原 solver 用 hole_count -> color 字典即可，因为图例只负责着色；本变体需要逐个擦掉图例组件，因此改为显式提取每个图例组件的像素集合。"
#   - "灰精灵仍假设 4 连通且彼此分离；生成器沿用 8 邻域留白放置，因此逐轮重算组件不会把多个精灵粘成一块。"
#   - "洞数统计仍基于各组件自身 bbox 内的 4 连通背景洞；删除其他组件不会改变剩余组件的洞数定义，因此迭代影响的是'删到哪一轮就停止'，不是洞数计算本身。"
#   - "新的停止条件是'工作区没有灰精灵后立即停止'；因此 generator 必须保证至少一个高洞数图例上方没有对应精灵，才能留下图例后缀。"
# placement_legality:
#   - "图例区仍由 divider 与 blocked 掩码隔开，灰精灵只放在工作区。"
#   - "精灵放置仍禁止越界、禁止与 divider/图例接触、禁止与其他精灵 8 邻接，避免视觉粘连和 solver 歧义。"
# visual_observability:
#   - "解题者能看到：所有灰精灵都会消失，但图例并非全保留也非全删除，而是按洞数从小到大删掉一段前缀。"
#   - "图例沿条带按洞数升序摆放，输出常表现为前半段图例被抹空、后半段仍在，便于肉眼把'顺序'与'洞数'对应起来。"
# cleanliness_and_inference:
#   - "输出只包含背景、divider 和残留图例，不会出现重涂后的混色叠加。"
#   - "颜色本身不参与排序；可推断线索完全来自洞数拓扑与图例的空间顺序。"

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


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


def _component_mask(pts: list[tuple[int, int]]) -> np.ndarray:
    """Build a tight boolean mask for one component."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _legend_entries(g: np.ndarray, bg: int, grey: int) -> list[dict[str, object]]:
    """
    Extract one legend component per non-(bg, grey) color.

    Each legend component:
    - does not touch the outer grid border,
    - is at least 2x2,
    - maps to a unique hole count.
    """
    h, w = g.shape
    entries: list[dict[str, object]] = []
    seen_holes: set[int] = set()
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates: list[list[tuple[int, int]]] = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append(pts)
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        pts = candidates[0]
        hc = _hole_count(_component_mask(pts))
        assert hc not in seen_holes, f"legend hole count {hc} must be unique"
        seen_holes.add(hc)
        entries.append({"hole_count": int(hc), "color": int(color), "pts": pts})
    entries.sort(key=lambda item: int(item["hole_count"]))
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively erase grey sprites and consume a low-to-high prefix of the legend.

    Input structure:
    - bg is the dominant background; grey (5) sprites occupy only the workspace.
    - A divider separates the workspace from a legend strip containing 3-4 colored hole templates,
      each a distinct non-bg, non-grey component with a distinct hole count.
    - The legend entries are interpretable independently of color value: topology (number of holes) is the key feature.

    Rule:
    - Read the legend as an ordered list of hole counts from smallest to largest.
    - Repeatedly process the smallest remaining legend count: erase every grey sprite with that count,
      erase every grey sprite whose count is absent from the still-remaining legend set,
      then erase that legend entry itself.
    - Stop as soon as no grey sprites remain; any higher-hole legend entries that were never reached stay visible.

    Key insight: This is not ordinary recoloring and not even a one-shot erase rule: all grey sprites disappear eventually,
    but the surviving colored evidence is the suffix of legend entries that the low-to-high elimination never had to reach.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    grey = 5

    legends = _legend_entries(g, bg, grey)
    assert legends, "grid must contain at least one legend entry"

    out = g.copy()
    remaining = list(legends)
    while remaining:
        grey_comps = components_mask(out == grey, connectivity=4)
        if not grey_comps:
            break
        current_hole = int(remaining[0]["hole_count"])
        listed_holes = {int(item["hole_count"]) for item in remaining}
        for pts in grey_comps:
            hc = _hole_count(_component_mask(pts))
            if hc == current_hole or hc not in listed_holes:
                for r, c in pts:
                    out[r, c] = bg
        for r, c in remaining[0]["pts"]:
            out[r, c] = bg
        remaining = remaining[1:]

    if np.any(out == grey):
        out[out == grey] = bg
    return out


# --------------- generator (holes-first organic blob) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (3, 4)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
SPRITE_RETRIES = 30
SPRITE_AREA_0 = (5, 25)
SPRITE_AREA_LOW_MIN = 8
# 3+ hole sprites: min = 10+2*h, max = 20+4*h
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

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


def _legend_span(hole_count: int) -> int:
    """Width (or height) of one legend patch along the arrangement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xspan or spanx3 legend patch."""
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
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(max_span: int) -> list[tuple[int, int]]:
    """Pick 3-4 distinct legend hole counts, kept in ascending order for visibility."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(max_span + 2)
    holes = list(random.choice(combos))
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> list[dict[str, object]]:
    """Paste legend patches in order and record their global colored pixels."""
    entries: list[dict[str, object]] = []
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        pts = [
            (int(rr + pr), int(cc + pc))
            for pr, pc in np.argwhere(patch == int(color))
        ]
        entries.append(
            {
                "hole_count": int(hole_count),
                "color": int(color),
                "pts": pts,
            }
        )
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1
    return entries


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[dict[str, object]]:
    """Place a full-row horizontal divider with legend panel above or below."""
    legends = _choose_legends(FULL_LAYOUT_MAX_SPAN)
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
    return _place_legend_line(grid, legends, "horizontal", r0, c0)


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[dict[str, object]]:
    """Place a full-column vertical divider with legend panel left or right."""
    legends = _choose_legends(FULL_LAYOUT_MAX_SPAN)
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
    return _place_legend_line(grid, legends, "vertical", r0, c0)


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
    else:  # br
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[dict[str, object]]:
    """Place an L-shaped divider with legend panel in one corner."""
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(L_LAYOUT_MAX_SPAN)
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
    else:  # br
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    return _place_legend_line(grid, legends, arrangement, r0, c0)


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[dict[str, object]]:
    """Randomly choose and apply one divider layout."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    """Sample a sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Find well-separated hole seeds inside the sprite canvas."""
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area**0.5))
    positions = [
        (r, c)
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
        if margin <= r < csz - margin and margin <= c < csz - margin
    ]
    random.shuffle(positions)
    seeds = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(
    seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int
) -> set[tuple[int, int]]:
    """Grow irregular forbidden blobs around hole seeds."""
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
    """Find the nearest non-forbidden start cell to the canvas center."""
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
    """Grow a connected blob of target area, avoiding forbidden cells."""
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
    """Seal around hole blobs and keep only the largest connected component."""
    csz = mask.shape[0]
    for _ in range(SEAL_ITERS):
        sealed = False
        for r, c in sorted(forbidden):
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc] and (nr, nc) not in forbidden:
                    mask[nr, nc] = True
                    sealed = True
        if not sealed:
            break
    if np.any(mask):
        labeled, n_comp = label_mask(mask)
        if n_comp > 1:
            best = max(range(1, n_comp + 1), key=lambda lbl: int(np.sum(labeled == lbl)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop to the tight bbox, then validate aspect and hole count."""
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
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    """Generate one amorphous grey sprite with the requested hole count."""
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area**0.5 * CANVAS_SCALE) + CANVAS_PAD)
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


def _sprite_plan(
    legend_entries: list[dict[str, object]]
) -> tuple[list[tuple[int, bool]], int, set[int]]:
    """
    Build sprite plan plus the highest legend index that will be reached.

    Returns:
    - plan of (hole_count, is_matched)
    - active_end index: legends[:active_end+1] are consumed
    - matched_prefix_indices: which consumed legends actually have direct matches
    """
    legend_holes = [int(item["hole_count"]) for item in legend_entries]
    active_end = random.randint(1, len(legend_holes) - 2)

    matched_prefix_indices = {active_end}
    for idx in range(active_end):
        if random.random() < 0.5:
            matched_prefix_indices.add(idx)
    if len(matched_prefix_indices) == active_end + 1:
        matched_prefix_indices.remove(random.randint(0, active_end - 1))

    matched_holes = [legend_holes[idx] for idx in sorted(matched_prefix_indices)]
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in set(legend_holes)]
    total = random.randint(*TOTAL_SPRITES_RANGE)

    plan: list[tuple[int, bool]] = []
    for hole_count in matched_holes:
        copies = random.randint(1, 2)
        for _ in range(copies):
            plan.append((int(hole_count), True))

    plan.append((int(random.choice(distractor_holes)), False))
    while len(plan) < total:
        add_distractor = (
            len([1 for _, flag in plan if not flag]) < MAX_DISTRACTORS
            and random.random() < 0.55
        )
        if add_distractor:
            plan.append((int(random.choice(distractor_holes)), False))
        else:
            plan.append((int(random.choice(matched_holes)), True))

    random.shuffle(plan)
    return plan, active_end, matched_prefix_indices


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connected 1-cell dilation."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a legal location for one sprite."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, blocked: np.ndarray, plan: list[tuple[int, bool]]
) -> bool:
    """Generate and place all grey sprites onto the workspace."""
    items = []
    for hole_count, _is_matched in plan:
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite))
    items.sort(key=lambda item: item[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
    return True


def _erase_entry(grid: np.ndarray, entry: dict[str, object]) -> None:
    """Erase one legend entry from a grid copy."""
    for r, c in entry["pts"]:
        grid[int(r), int(c)] = BG


def _non_iterative_baseline(grid: np.ndarray) -> np.ndarray:
    """
    Simpler baseline for degeneration checks:
    erase all grey sprites and erase only legends whose hole count is directly matched.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legends = _legend_entries(g, bg, GREY)
    matched_holes = set()
    for pts in components_mask(g == GREY, connectivity=4):
        hc = _hole_count(_component_mask(pts))
        if any(int(entry["hole_count"]) == hc for entry in legends):
            matched_holes.add(int(hc))

    out = g.copy()
    out[out == GREY] = bg
    for entry in legends:
        if int(entry["hole_count"]) in matched_holes:
            _erase_entry(out, entry)
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, plan iterative behavior, place sprites, and validate."""
    n = GRID_SIZE
    grid = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=int)

    legend_entries = _add_layout(grid, blocked)
    plan, active_end, matched_prefix_indices = _sprite_plan(legend_entries)

    expected = grid.copy()
    for idx in range(active_end + 1):
        _erase_entry(expected, legend_entries[idx])

    if not _place_sprites(grid, blocked, plan):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.any(actual == GREY):
        return None
    if active_end >= len(legend_entries) - 1:
        return None
    if not any(idx < active_end for idx in range(active_end) if idx not in matched_prefix_indices):
        return None
    if np.array_equal(actual, _non_iterative_baseline(grid)):
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
