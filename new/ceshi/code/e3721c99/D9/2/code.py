# Variant of e3721c99
# Dimension: D9 — 精灵与图例的基数关系 (meta)
# Change: 多对一关系——多个灰色精灵可能有相同洞数，匹配同一个图例形状，获得相同颜色... → 分组匹配（精灵按区域分组，各组对应不同图例）

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, label_mask
from scipy.ndimage import binary_dilation, generate_binary_structure


# Design analysis
# a) Concrete behavior:
#    The grid is split by one full divider row and one full divider column, yielding four rectangles.
#    One adjacent pair of rectangles forms the legend side; the opposite pair is the workspace side.
#    Each legend rectangle contains its own 2-entry sublegend, and the workspace rectangle aligned with it
#    uses only that local sublegend. Example: with a left legend panel, the top-right sprites read the
#    top-left legend, while the bottom-right sprites read the bottom-left legend. A 1-hole sprite can turn
#    red in the upper region and green in the lower region because the shared hole count appears in both
#    sublegends with different colors.
# b) Complexity calibration:
#    Each region has exactly two legend entries. The two regions share one hole count and each has one
#    exclusive hole count. This keeps the rule compact while forcing visible region dependence:
#    same topology can recolor differently across regions, and a hole count valid in one region can be
#    erased in the other.
# c) Old assumptions that break:
#    The original solver built one global hole_count -> color table and asserted hole counts are unique
#    globally. That is invalid now because the same hole count may reappear in the other legend group.
#    The recoloring loop also assumed every grey component should consult the same table. The new solver
#    must first identify the component's workspace region, then consult only that region's table.
# d) Placement legality:
#    The generator draws a cross of full-span dividers, keeps legends inside their own subpanels with
#    a 1-cell inset, and places sprites fully inside one workspace rectangle with a 1-cell margin.
#    This guarantees no sprite crosses a partition line and every sprite has an unambiguous local legend.
# e) Visual observability:
#    Region dependence is carried by visible geometry only: the cross partition and the legend subpanels.
#    No sorting or transformation uses hidden color-id ordering. Every sample contains at least one shared
#    hole count across regions plus cross-region distractors so the local rule is inferable from 2-3 grids.
# f) Visual cleanliness:
#    The legend side contains only colored legend patches; the workspace side contains only grey sprites.
#    The full row/column partition creates clean rectangular zones instead of cluttered overlays.
# g) Color-change inferability:
#    Only four legend colors are used per grid, two per region. The same hole count is duplicated across
#    regions with different colors, giving a direct structural cue that color depends on both topology and
#    location.
#
# Variant spec:
#   grid: 30x30, bg=0, grey=5, divider=1
#   partition: exactly one full row and one full column of divider color, forming a cross
#   legend_side: left or right pair of rectangles, OR top or bottom pair of rectangles
#   workspace_side: the opposite pair of rectangles
#   sublegend_per_region:
#     count: exactly 2 entries per region
#     arrangement: vertical in left/right panels, horizontal in top/bottom panels
#     relation: the two region hole sets share one hole count and differ on the other
#   sprite_rule:
#     each grey sprite uses the legend mapping of its own workspace rectangle only
#     unmatched hole counts in that rectangle erase to bg


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
    Read legend components from a local rectangle only.
    Components must be interior to the rectangle so border-touching divider lines are ignored.
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
        sub = (g[r0 : r1 + 1, c0 : c1 + 1] == c)
        hc = _hole_count(sub)
        assert hc not in mapping, f"local legend hole count {hc} must map to exactly one color"
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
    assert r0 <= r1 and c0 <= c1, "all quadrants must be non-empty"
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _region_maps(g: np.ndarray, bg: int, grey: int) -> dict[str, dict[int, int]]:
    h, w = g.shape
    divider = 1
    full_rows = [r for r in range(h) if np.all(g[r, :] == divider)]
    full_cols = [c for c in range(w) if np.all(g[:, c] == divider)]
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
    assert len(legend_sides) == 1, "legend groups must occupy exactly one side of the cross"
    side = legend_sides[0]

    if side == "left":
        return {"tr": local["tl"], "br": local["bl"]}
    if side == "right":
        return {"tl": local["tr"], "bl": local["br"]}
    if side == "top":
        return {"bl": local["tl"], "br": local["tr"]}
    return {"tl": local["bl"], "tr": local["br"]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by matching hole counts to the legend of their own region.

    Input structure:
    - bg is the most common color, grey sprites use color 5, and divider lines use color 1.
    - One full row and one full column divide the board into four rectangles.
    - Two adjacent rectangles contain colored legend patches; the opposite two rectangles contain grey sprites.

    Rule:
    - Read a separate hole_count -> color mapping from each legend rectangle.
    - For every grey sprite, determine which workspace rectangle contains it, count its holes,
      and consult only that rectangle's local legend.
    - Recolor matched sprites with the local legend color; erase sprites whose hole count is absent there.

    Key insight:
    - Topology still identifies the sprite, but topology alone is no longer sufficient globally:
      the sprite's region selects which legend table applies.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    region_legend = _region_maps(g, bg, grey)
    assert region_legend, "region legend mapping must be non-empty"

    h, w = g.shape
    div_r = next(r for r in range(h) if np.all(g[r, :] == 1))
    div_c = next(c for c in range(w) if np.all(g[:, c] == 1))

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)

        pr, pc = map(int, pts[0])
        assert pr != div_r and pc != div_c, "sprite must not lie on the divider cross"
        quad = ("t" if pr < div_r else "b") + ("l" if pc < div_c else "r")
        assert quad in region_legend, "grey sprite must lie in a workspace rectangle"

        legend = region_legend[quad]
        newc = int(legend[hc]) if hc in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator (region-partitioned holes-first blobs) ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

PARTITION_CHOICES = [14, 15]
PANEL_SIZE_RANGE = (6, 7)
MAX_HOLE_COUNT = 6
GEN_RETRIES = 300

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


def _choose_region_holes(max_span: int) -> tuple[int, int, int]:
    choices = []
    for shared in range(MAX_HOLE_COUNT):
        for h1 in range(MAX_HOLE_COUNT):
            for h2 in range(MAX_HOLE_COUNT):
                if len({shared, h1, h2}) != 3:
                    continue
                if _line_span([(shared, 0), (h1, 0)]) > max_span:
                    continue
                if _line_span([(shared, 0), (h2, 0)]) > max_span:
                    continue
                choices.append((shared, h1, h2))
    assert choices, "must have at least one valid region-hole assignment"
    return random.choice(choices)


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


def _cross_layout(
    grid: np.ndarray,
) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, str], dict[str, list[tuple[int, int]]], str]:
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
    shared, h1, h2 = _choose_region_holes(max_span)
    colors = random.sample(LEGEND_COLORS, 4)

    legends_by_rect = {
        legend_rects[0]: [(shared, colors[0]), (h1, colors[1])],
        legend_rects[1]: [(shared, colors[2]), (h2, colors[3])],
    }
    for rect_key in legends_by_rect:
        random.shuffle(legends_by_rect[rect_key])
        _place_legends_in_rect(grid, rects[rect_key], legends_by_rect[rect_key], arrangement)

    return rects, region_to_legend_rect, legends_by_rect, arrangement


def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


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
    max_hole = (max(2, area // (hole_count + 2)) if hole_count <= 2 else max(1, area // (hole_count * 2 + 1)))
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


def _random_sprite_location_in_rect(
    rect: tuple[int, int, int, int], occupied: np.ndarray, sprite: np.ndarray
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

    candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _region_sprite_specs(local_holes: set[int], other_unique: int, local_map: dict[int, int]) -> list[tuple[int, int | None]]:
    h_a, h_b = sorted(local_holes)
    # Generation only needs one guaranteed match per local legend entry plus one cross-region distractor.
    specs = [
        (h_a, local_map[h_a]),
        (h_b, local_map[h_b]),
        (other_unique, None),
    ]
    extras = [
        (h_a, local_map[h_a]),
        (h_b, local_map[h_b]),
    ]
    absent = [h for h in range(MAX_HOLE_COUNT) if h not in local_holes and h != other_unique]
    if absent:
        extras.append((random.choice(absent), None))
    specs.append(random.choice(extras))
    random.shuffle(specs)
    return specs


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

    region_keys = list(region_maps)
    assert len(region_keys) == 2
    local_holes = {region: set(region_maps[region]) for region in region_keys}
    uniques = {region: next(iter(local_holes[region] - set.intersection(*local_holes.values()))) for region in region_keys}

    items = []
    for region in region_keys:
        other = next(r for r in region_keys if r != region)
        specs = _region_sprite_specs(local_holes[region], uniques[other], region_maps[region])
        for hole_count, color in specs:
            sprite = _make_sprite(hole_count)
            if sprite is None:
                return False
            items.append((sprite.size, region, sprite, color))

    items.sort(key=lambda x: x[0], reverse=True)
    for _, region, sprite, color in items:
        loc = _random_sprite_location_in_rect(rects[region], occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if color is not None:
            out_patch = np.where(sprite == GREY, int(color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    occupied = np.zeros_like(grid)

    rects, region_to_legend_rect, legends_by_rect, _arrangement = _cross_layout(grid)
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
    raise RuntimeError("failed to generate e3721c99 D9 variant input")
