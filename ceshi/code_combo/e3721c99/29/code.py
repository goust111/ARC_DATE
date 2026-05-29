# Combo variant of e3721c99 (index 29)
# combo_spec: D9/2+D13/0
# Name: 按洞数分区排列+分组匹配

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, label_mask
from scipy.ndimage import binary_dilation, generate_binary_structure


# Design analysis
# a) Concrete behavior:
#    The board is split by one full divider row and one full divider column, giving four rectangles.
#    Two adjacent rectangles are legend panels; the opposite two are workspace regions. Each legend panel
#    has exactly two entries. The two legend panels reuse the same two hole counts but assign different
#    colors. Example: with legends on the left, the top-right region may use {1->red, 3->green} while the
#    bottom-right region uses {1->yellow, 3->teal}. Inside each workspace region, all 1-hole sprites are
#    clustered in one band and all 3-hole sprites are clustered in another band. The original task would
#    read one global hole_count -> color table; the fused variant must first choose the local legend by
#    region and then match the sprite's hole count within that local table.
# b) Complexity calibration:
#    Each workspace region contains exactly two repeated hole-count clusters, with 2-3 sprites per cluster.
#    Using the same two hole counts in both regions keeps the pattern compact, but the color assignments
#    differ across regions so location genuinely matters. Repetition inside a cluster makes the D13 signal
#    visually obvious without introducing too many categories.
# c) Old assumptions that break:
#    The original solver assumed hole counts are globally unique across all legends; that is false now,
#    because each legend panel reuses the same hole counts with different colors. The original generator
#    also scattered sprites freely; that no longer holds because sprite placement now carries information:
#    same-hole sprites are intentionally grouped in one row-band or column-band inside each workspace region.
#    The solver must therefore identify the sprite's workspace rectangle before consulting a legend, and the
#    generator must guarantee every grey component stays wholly inside one workspace region.
# d) Placement legality:
#    The generator draws a full cross divider, places legends with a 1-cell inset inside their own panels,
#    and places sprites only inside the opposite workspace rectangles. Occupancy is dilated by 8-neighborhood
#    during placement so blobs keep a visible gap. Cluster placement uses preferred halves of a region, but
#    still checks bounds against each actual sprite size, so large blobs do not cross dividers or overlap.
# e) Visual observability:
#    A human can see two cues at once: the cross partition says there are local legend groups, and repeated
#    same-hole blobs appear in the same band inside a workspace region. Because the same topology appears in
#    both regions but recolors differently there, the local-legend dependency is visible from examples rather
#    than hidden in color ids or generator-only metadata.
# f) Visual cleanliness:
#    Legends occupy only the legend side; grey sprites occupy only the workspace side. Within each workspace
#    region there are only two loose bands, one per hole count, so the grid stays readable instead of mixing
#    many unrelated sprites in the same small area.
# g) Color-change inferability:
#    Only four legend colors appear per sample, two per local legend. The same two hole counts are duplicated
#    across both local legends, so topology alone is insufficient; at the same time each hole count repeats in
#    a spatial cluster, giving a visible structural reason for why similarly holed sprites may recolor
#    differently in different regions.
#
# Variant spec:
#   grid: 30x30, bg=0, grey=5, divider=1
#   partition: exactly one full divider row and one full divider column
#   legend_side: one adjacent pair of rectangles on left/right/top/bottom
#   workspace_side: the opposite adjacent pair of rectangles
#   local_legends:
#     count: exactly 2 entries per legend rectangle
#     relation: both legend rectangles use the same two hole counts, but with different colors
#   workspace_clusters:
#     each workspace rectangle contains 2 clusters, one per local hole count
#     each cluster has 2-3 grey sprites and occupies one row-band or column-band
#   sprite_rule:
#     sprites recolor by matching hole count against the legend of their own workspace rectangle


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed 4-connected background holes inside a binary mask."""
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


def _legend_mapping_in_rect(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Read legend components from one local rectangle.
    Border-touching components are ignored so the divider line does not count as a legend entry.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {c} must have exactly one local legend component"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == c
        hc = _hole_count(sub)
        assert hc not in mapping, f"local legend hole count {hc} must be unique within a panel"
        mapping[hc] = int(c)
    return mapping


def _quadrant_rects(h: int, w: int, div_r: int, div_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (0, 0, div_r - 1, div_c - 1),
        "tr": (0, div_c + 1, div_r - 1, w - 1),
        "bl": (div_r + 1, 0, h - 1, div_c - 1),
        "br": (div_r + 1, div_c + 1, h - 1, w - 1),
    }


def _subgrid(g: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    assert r0 <= r1 and c0 <= c1, "all partition rectangles must be non-empty"
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _region_maps(
    g: np.ndarray, bg: int, grey: int
) -> tuple[int, int, dict[str, tuple[int, int, int, int]], dict[str, dict[int, int]]]:
    h, w = g.shape
    full_rows = [r for r in range(h) if np.all(g[r, :] == DIV)]
    full_cols = [c for c in range(w) if np.all(g[:, c] == DIV)]
    assert len(full_rows) == 1 and len(full_cols) == 1, "grid must contain one full divider row and one full divider column"
    div_r, div_c = int(full_rows[0]), int(full_cols[0])

    rects = _quadrant_rects(h, w, div_r, div_c)
    local = {k: _legend_mapping_in_rect(_subgrid(g, rect), bg, grey) for k, rect in rects.items()}

    sides = {
        "left": ("tl", "bl"),
        "right": ("tr", "br"),
        "top": ("tl", "tr"),
        "bottom": ("bl", "br"),
    }
    legend_sides = [side for side, keys in sides.items() if all(local[k] for k in keys)]
    assert len(legend_sides) == 1, "legend panels must occupy exactly one side of the cross"
    side = legend_sides[0]

    if side == "left":
        region_legend = {"tr": local["tl"], "br": local["bl"]}
    elif side == "right":
        region_legend = {"tl": local["tr"], "bl": local["br"]}
    elif side == "top":
        region_legend = {"bl": local["tl"], "br": local["tr"]}
    else:
        region_legend = {"tl": local["bl"], "tr": local["br"]}

    assert len(region_legend) == 2, "there must be exactly two workspace regions"
    return div_r, div_c, rects, region_legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching hole counts against the local legend of their region.

    Input structure:
    - The board is split by one full row and one full column of divider color 1.
    - Two adjacent rectangles contain colored legend patches; the opposite two rectangles contain grey sprites.
    - Inside each workspace rectangle, sprites with the same hole count are visually clustered in one band.

    Rule:
    - Build a separate hole_count -> color mapping from each legend rectangle.
    - For every grey sprite, determine which workspace rectangle contains it, count its holes,
      and consult only that rectangle's local legend.
    - Recolor matched sprites with the corresponding local color; erase sprites whose hole count
      is absent from that local legend.

    Key insight:
    - Position chooses the relevant sublegend, and topology chooses the entry within that sublegend:
      the same hole count can map to different colors in different regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY

    div_r, div_c, _rects, region_legend = _region_maps(g, bg, grey)
    out = g.copy()

    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)

        pr, pc = map(int, pts[0])
        assert pr != div_r and pc != div_c, "sprite must not touch the divider cross"
        quad = ("t" if pr < div_r else "b") + ("l" if pc < div_c else "r")
        assert quad in region_legend, "grey sprite must lie in a workspace region"

        legend = region_legend[quad]
        newc = int(legend[hc]) if hc in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator (clustered local-legend blobs) ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

PARTITION_CHOICES = [14, 15]
PANEL_SIZE_RANGE = (6, 7)
MAX_HOLE_COUNT = 6
ACTIVE_HOLES = [0, 1, 2, 3]
GEN_RETRIES = 300

SPRITE_RETRIES = 30
SPRITES_PER_CLUSTER = (2, 3)
SPRITE_AREA_0 = (5, 20)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 11
SPRITE_MAX_W = 16
SPRITE_MAX_ASPECT = 2.5

_STRUCT8 = generate_binary_structure(2, 2)


def _legend_span(hole_count: int) -> int:
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
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
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _rect_span(rect: tuple[int, int, int, int], arrangement: str) -> int:
    r0, c0, r1, c1 = rect
    return (c1 - c0 + 1) if arrangement == "horizontal" else (r1 - r0 + 1)


def _place_legends_in_rect(
    grid: np.ndarray, rect: tuple[int, int, int, int], legends: list[tuple[int, int]], arrangement: str
) -> None:
    r0, c0, r1, c1 = rect
    total = _line_span(legends)
    if arrangement == "horizontal":
        assert r1 - r0 + 1 >= 5
        assert c1 - c0 + 1 >= total + 2
        rr = random.randint(r0 + 1, r1 - 3)
        cc = random.randint(c0 + 1, c1 - total)
    else:
        assert c1 - c0 + 1 >= 5
        assert r1 - r0 + 1 >= total + 2
        rr = random.randint(r0 + 1, r1 - total)
        cc = random.randint(c0 + 1, c1 - 3)
    _place_legend_line(grid, legends, arrangement, rr, cc)


def _choose_shared_holes(max_span: int) -> tuple[int, int]:
    pairs = [
        (h1, h2)
        for i, h1 in enumerate(ACTIVE_HOLES)
        for h2 in ACTIVE_HOLES[i + 1 :]
        if _line_span([(h1, 0), (h2, 0)]) <= max_span
    ]
    assert pairs, "must have at least one legend hole pair that fits all panels"
    return random.choice(pairs)


def _cross_layout(
    grid: np.ndarray,
) -> tuple[
    dict[str, tuple[int, int, int, int]],
    dict[str, str],
    dict[str, list[tuple[int, int]]],
]:
    kind = random.choice(["vertical_panel", "horizontal_panel"])
    if kind == "vertical_panel":
        div_r = random.choice(PARTITION_CHOICES)
        panel_w = random.randint(*PANEL_SIZE_RANGE)
        side = random.choice(["left", "right"])
        div_c = panel_w if side == "left" else GRID_SIZE - 1 - panel_w
        arrangement = "vertical"
    else:
        div_c = random.choice(PARTITION_CHOICES)
        panel_h = random.randint(*PANEL_SIZE_RANGE)
        side = random.choice(["top", "bottom"])
        div_r = panel_h if side == "top" else GRID_SIZE - 1 - panel_h
        arrangement = "horizontal"

    grid[div_r, :] = DIV
    grid[:, div_c] = DIV
    rects = _quadrant_rects(GRID_SIZE, GRID_SIZE, div_r, div_c)

    if kind == "vertical_panel" and side == "left":
        region_to_legend_rect = {"tr": "tl", "br": "bl"}
    elif kind == "vertical_panel" and side == "right":
        region_to_legend_rect = {"tl": "tr", "bl": "br"}
    elif kind == "horizontal_panel" and side == "top":
        region_to_legend_rect = {"bl": "tl", "br": "tr"}
    else:
        region_to_legend_rect = {"tl": "bl", "tr": "br"}

    legend_rects = list(region_to_legend_rect.values())
    max_span = min(_rect_span(rects[k], arrangement) - 2 for k in legend_rects)
    h1, h2 = _choose_shared_holes(max_span)
    colors = random.sample(LEGEND_COLORS, 4)

    legends_by_rect = {
        legend_rects[0]: [(h1, colors[0]), (h2, colors[1])],
        legend_rects[1]: [(h1, colors[2]), (h2, colors[3])],
    }
    for rect_key, legends in legends_by_rect.items():
        random.shuffle(legends)
        _place_legends_in_rect(grid, rects[rect_key], legends, arrangement)

    return rects, region_to_legend_rect, legends_by_rect


def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 22 + 8 * hole_count)
    return random.randint(14, 28)


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
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
    margin = HOLE_SEED_MARGIN
    max_hole = max(1, area // (hole_count + 2))
    forbidden = set()
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
                    if any((nnr, nnc) in forbidden and (nnr, nnc) not in blob for nnr, nnc in neighbors4(nr, nc)):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            p = random.choice(cands)
            forbidden.add(p)
            blob.add(p)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    if (center, center) not in forbidden:
        return center, center
    for d in range(1, csz):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = center + dr, center + dc
                if 0 <= nr < csz and 0 <= nc < csz and (nr, nc) not in forbidden:
                    return nr, nc
    return center, center


def _grow_mask(csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]) -> np.ndarray:
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
            best = max(range(1, n_comp + 1), key=lambda l: int(np.sum(labeled == l)))
            mask = labeled == best
    return mask


def _crop_and_validate(mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int) -> np.ndarray | None:
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
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2

        forbidden = set()
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


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _cluster_axis(rect: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return "cols" if w >= h else "rows"


def _random_sprite_location_in_cluster(
    rect: tuple[int, int, int, int], occupied: np.ndarray, sprite: np.ndarray, axis: str, slot: int
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = rect
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)

    r_lo = r0 + 1
    c_lo = c0 + 1
    r_hi = r1 - h
    c_hi = c1 - w
    if r_lo > r_hi or c_lo > c_hi:
        return None

    candidates = []
    if axis == "rows":
        min_ctr = 2 * r_lo + h - 1
        max_ctr = 2 * r_hi + h - 1
        mid_ctr = (min_ctr + max_ctr) // 2
        ideal = min_ctr + (max_ctr - min_ctr) * (1 if slot == 0 else 3) // 4
    else:
        min_ctr = 2 * c_lo + w - 1
        max_ctr = 2 * c_hi + w - 1
        mid_ctr = (min_ctr + max_ctr) // 2
        ideal = min_ctr + (max_ctr - min_ctr) * (1 if slot == 0 else 3) // 4

    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
                continue
            ctr = 2 * r + h - 1 if axis == "rows" else 2 * c + w - 1
            if slot == 0 and ctr > mid_ctr:
                continue
            if slot == 1 and ctr <= mid_ctr:
                continue
            candidates.append((abs(ctr - ideal), random.random(), r, c))

    if not candidates:
        return None
    candidates.sort()
    top = candidates[: min(24, len(candidates))]
    _, _, r, c = random.choice(top)
    return int(r), int(c)


def _place_region_sprites(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    rects: dict[str, tuple[int, int, int, int]],
    region_to_legend_rect: dict[str, str],
    legends_by_rect: dict[str, list[tuple[int, int]]],
) -> bool:
    local_maps = {rk: {h: c for h, c in legends} for rk, legends in legends_by_rect.items()}
    region_maps = {region: local_maps[legend_rect] for region, legend_rect in region_to_legend_rect.items()}

    items = []
    for region, local_map in region_maps.items():
        hole_counts = list(local_map.keys())
        random.shuffle(hole_counts)
        axis = _cluster_axis(rects[region])
        hole_to_slot = {hole_counts[0]: 0, hole_counts[1]: 1}
        for hole_count in hole_counts:
            for _ in range(random.randint(*SPRITES_PER_CLUSTER)):
                sprite = _make_sprite(hole_count)
                if sprite is None:
                    return False
                items.append((sprite.size, region, axis, hole_to_slot[hole_count], sprite, int(local_map[hole_count])))

    items.sort(key=lambda x: x[0], reverse=True)
    for _, region, axis, slot, sprite, color in items:
        loc = _random_sprite_location_in_cluster(rects[region], occupied, sprite, axis, slot)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        out_patch = np.where(sprite == GREY, color, BG)
        paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    occupied = np.zeros_like(grid)

    rects, region_to_legend_rect, legends_by_rect = _cross_layout(grid)
    expected[:, :] = grid
    if not _place_region_sprites(grid, expected, occupied, rects, region_to_legend_rect, legends_by_rect):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
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
