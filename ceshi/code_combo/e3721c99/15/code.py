# Combo variant of e3721c99 (index 15)
# combo_spec: D9/2+D14/0
# Name: 分组匹配+双灰度区分
#
# Design analysis
# a) Concrete behavior and trace:
#    The board is cut by one full divider row and one full divider column, so the visible geometry
#    already partitions the workspace into two active rectangles and two legend rectangles. Each legend
#    rectangle contains its own 2-entry sublegend. A sprite is decoded in two stages:
#    first choose the local legend by region, then apply the shade-priority rule inside that region.
#    Example: the shared hole-count 2 appears in both legends with different colors. In the upper
#    workspace, a dark 2-hole sprite and a light 2-hole sprite coexist, so only the dark one survives
#    and turns, say, red. In the lower workspace, only a light 2-hole sprite exists, so it survives
#    and turns, say, green. Original task: both 2-hole sprites would have consulted one global legend
#    and both would have recolored the same way. Fused task: region chooses the table, and shade breaks
#    ties only within that region and hole-count.
# b) Complexity calibration:
#    Each active region still has exactly two legend entries. The two regions share one hole-count and
#    differ on the other, which keeps the puzzle compact while making region dependence obvious. The
#    generator also forces at least one same-region light-vs-dark conflict and one different-region
#    light-only winner for the shared hole-count, so neither sub-rule can disappear silently.
# c) Hidden assumptions that break from the seed solver:
#    The seed solver assumed one global hole_count -> color mapping and one sprite color. Both fail here.
#    Reusing the old mapping would be wrong because the same hole-count is intentionally reused across the
#    two legend groups with different colors. Reusing the old component loop would also be wrong because
#    dark and light sprites must be collected separately and priority must be resolved per region.
#    A new fusion-specific assumption appears too: dark priority is local, not global. If a dark shared-hole
#    sprite exists in the top region, it must not suppress a light shared-hole sprite in the bottom region.
# d) Placement legality:
#    The generator uses a cross partition, keeps legends inside their own rectangles with a 1-cell inset,
#    and places every sprite fully inside one workspace rectangle with a 1-cell margin plus 8-neighborhood
#    spacing. That prevents overlap, border crossing, and ambiguous region assignment.
# e) Visual observability:
#    Background is fixed to teal (8), so dark sprites can safely use black (0) without becoming invisible.
#    The cross partition, duplicated shared hole-count, and two sprite shades are all directly visible.
#    Across 2-3 samples a human can see that identical topology can change color by region, and that within
#    one region dark beats light only when both compete for the same local hole-count.
# f) Visual cleanliness:
#    Legend panels and workspaces remain separated exactly as rectangles. Each rectangle contains only one
#    object family: colored legends on the legend side, grey shades on the workspace side. No overlay marks
#    or hidden ordering are introduced.
# g) Color-change inferability:
#    Only four legend colors appear per grid, two per region. The shared hole-count reappears with different
#    colors across regions, providing a concrete cue for local matching. The light/dark distinction uses only
#    two source shades, so the new color behavior is still structurally learnable from few examples.
# h) Generator guarantees matched to solver assumptions:
#    `main()` assumes exactly one full divider row and one full divider column, exactly one legend side,
#    and sprites that do not cross the divider. The generator constructs the cross directly and places each
#    sprite inside one chosen workspace rectangle. `main()` also assumes legend colors exclude BG, dark grey,
#    and light grey; the generator enforces that palette restriction.

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


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
            stack = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        stack.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _component_mask(pts: np.ndarray) -> np.ndarray:
    """Convert component coordinates into a tight boolean mask."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _legend_mapping_in_rect(g: np.ndarray, bg: int, sprite_colors: tuple[int, int]) -> dict[int, int]:
    """
    Read legend components from one local rectangle.
    Border-touching components are ignored so the divider line cannot be mistaken for a legend.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    excluded = {int(bg), *map(int, sprite_colors), DIV}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in excluded]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
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
        assert len(candidates) == 1, f"color {color} must have exactly one local legend component"
        r0, c0, r1, c1 = candidates[0]
        hc = _hole_count(g[r0 : r1 + 1, c0 : c1 + 1] == color)
        assert hc not in mapping, f"local legend hole count {hc} must map to exactly one color"
        mapping[hc] = int(color)
    return mapping


def _quadrant_rects(h: int, w: int, div_r: int, div_c: int) -> dict[str, tuple[int, int, int, int]]:
    """Return inclusive rectangles for the four quadrants around the divider cross."""
    return {
        "tl": (0, 0, div_r - 1, div_c - 1),
        "tr": (0, div_c + 1, div_r - 1, w - 1),
        "bl": (div_r + 1, 0, h - 1, div_c - 1),
        "br": (div_r + 1, div_c + 1, h - 1, w - 1),
    }


def _subgrid(g: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    assert r0 <= r1 and c0 <= c1, "quadrants must be non-empty"
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _cross_region_maps(
    g: np.ndarray, bg: int, sprite_colors: tuple[int, int]
) -> tuple[int, int, dict[str, tuple[int, int, int, int]], dict[str, dict[int, int]]]:
    """Infer the cross partition and return workspace-region legend tables."""
    h, w = g.shape
    full_rows = [r for r in range(h) if np.all(g[r, :] == DIV)]
    full_cols = [c for c in range(w) if np.all(g[:, c] == DIV)]
    assert len(full_rows) == 1 and len(full_cols) == 1, "grid must contain one full divider row and one full divider column"
    div_r, div_c = int(full_rows[0]), int(full_cols[0])
    rects = _quadrant_rects(h, w, div_r, div_c)
    local_maps = {k: _legend_mapping_in_rect(_subgrid(g, rect), bg, sprite_colors) for k, rect in rects.items()}

    sides = {
        "left": ("tl", "bl"),
        "right": ("tr", "br"),
        "top": ("tl", "tr"),
        "bottom": ("bl", "br"),
    }
    legend_sides = [side for side, rect_keys in sides.items() if all(local_maps[k] for k in rect_keys)]
    assert len(legend_sides) == 1, "legend panels must occupy exactly one adjacent pair of quadrants"
    side = legend_sides[0]

    if side == "left":
        region_maps = {"tr": local_maps["tl"], "br": local_maps["bl"]}
    elif side == "right":
        region_maps = {"tl": local_maps["tr"], "bl": local_maps["br"]}
    elif side == "top":
        region_maps = {"bl": local_maps["tl"], "br": local_maps["tr"]}
    else:
        region_maps = {"tl": local_maps["bl"], "tr": local_maps["br"]}
    return div_r, div_c, rects, region_maps


def _component_region(pts: np.ndarray, div_r: int, div_c: int) -> str:
    """Assign a sprite component to one workspace quadrant."""
    r0, c0, r1, c1 = bbox_coords(pts)
    if r1 < div_r:
        row_key = "t"
    elif r0 > div_r:
        row_key = "b"
    else:
        raise AssertionError("sprite must not cross the divider row")

    if c1 < div_c:
        col_key = "l"
    elif c0 > div_c:
        col_key = "r"
    else:
        raise AssertionError("sprite must not cross the divider column")
    return row_key + col_key


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor region-grouped two-shade sprites by local hole legends, with dark priority inside each region.

    Input structure:
    - The background is the most common color and is chosen so dark sprites remain visible.
    - One full divider row and one full divider column split the board into four rectangles.
    - Two adjacent rectangles contain colored legend patches; the opposite two rectangles contain sprites
      drawn in light grey and dark grey.

    Rule:
    - Read one hole_count -> color mapping from each legend rectangle, then assign each workspace rectangle
      the legend of its aligned partner.
    - For every sprite, identify its workspace region and count holes inside its own bounding box.
    - A matching dark sprite always recolors to that region's legend color.
    - A matching light sprite recolors only if the same region does not also contain any dark sprite with
      that same hole count; otherwise it is erased. Any locally unmatched sprite is erased as well.

    Key insight:
    - The answer depends on two visible selectors at once: topology chooses a legend entry, region chooses
      which legend table applies, and source shade resolves same-region conflicts for the same hole count.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sprite_colors = (LIGHT_GREY, DARK_GREY)

    div_r, div_c, _rects, region_maps = _cross_region_maps(g, bg, sprite_colors)
    assert region_maps, "workspace legend mapping must be non-empty"

    sprite_info: list[tuple[str, int, np.ndarray, int]] = []
    dark_holes_by_region = {region: set() for region in region_maps}
    for shade in sprite_colors:
        for pts in components_mask(g == shade, connectivity=4):
            region = _component_region(pts, div_r, div_c)
            assert region in region_maps, "sprite must lie in a workspace region"
            hc = _hole_count(_component_mask(pts))
            sprite_info.append((region, shade, pts, hc))
            if shade == DARK_GREY:
                dark_holes_by_region[region].add(hc)

    out = g.copy()
    for region, shade, pts, hc in sprite_info:
        local_map = region_maps[region]
        if hc not in local_map:
            newc = int(bg)
        elif shade == DARK_GREY:
            newc = int(local_map[hc])
        elif hc in dark_holes_by_region[region]:
            newc = int(bg)
        else:
            newc = int(local_map[hc])
        for r, c in pts:
            out[r, c] = newc
    return out


# --------------- generator (cross-partitioned dual-grey blobs) ---------------

BG = 8
DIV = 1
DARK_GREY = 0
LIGHT_GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 9]

PARTITION_CHOICES = [14, 15]
PANEL_SIZE_RANGE = (6, 7)
MAX_HOLE_COUNT = 6
GEN_RETRIES = 300

TOTAL_SPRITES_RANGE = (8, 11)
MAX_EXTRA_PER_REGION = 2
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
    """Width or height of one legend patch along its main axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a horizontal or vertical legend patch with the requested hole count."""
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
    """Total legend span including one background gap between adjacent entries."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


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


def _choose_region_holes(max_span: int) -> tuple[int, int, int]:
    """Choose one shared hole-count plus one unique hole-count per region."""
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
    assert choices, "must have at least one valid hole assignment"
    return random.choice(choices)


def _rect_span(rect: tuple[int, int, int, int], arrangement: str) -> int:
    r0, c0, r1, c1 = rect
    return (c1 - c0 + 1) if arrangement == "horizontal" else (r1 - r0 + 1)


def _place_legends_in_rect(
    grid: np.ndarray, rect: tuple[int, int, int, int], legends: list[tuple[int, int]], arrangement: str
) -> None:
    """Place one 2-entry legend line with a 1-cell inset inside a rectangle."""
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
) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, dict[int, int]]]:
    """Construct the cross partition and two aligned local legends."""
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
    max_span = min(_rect_span(rects[key], arrangement) - 2 for key in legend_rects)
    shared, unique_a, unique_b = _choose_region_holes(max_span)
    colors = random.sample(LEGEND_COLORS, 4)

    legends_by_rect = {
        legend_rects[0]: [(shared, colors[0]), (unique_a, colors[1])],
        legend_rects[1]: [(shared, colors[2]), (unique_b, colors[3])],
    }
    for rect_key, legends in legends_by_rect.items():
        random.shuffle(legends)
        _place_legends_in_rect(grid, rects[rect_key], legends, arrangement)

    region_maps = {
        region: {hole_count: color for hole_count, color in legends_by_rect[legend_rect]}
        for region, legend_rect in region_to_legend_rect.items()
    }
    return rects, region_maps


def _sprite_area(hole_count: int) -> int:
    """Sample a target sprite area based on the requested hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
    """Find well-separated interior seed positions for hole blobs."""
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
    """Grow irregular forbidden blobs that will become holes."""
    margin = HOLE_SEED_MARGIN
    max_hole = max(2, area // (hole_count + 2)) if hole_count <= 2 else max(1, area // (hole_count * 2 + 1))
    forbidden: set[tuple[int, int]] = set()
    for sr, sc in seeds:
        forbidden.add((sr, sc))
        blob = {(sr, sc)}
        target = random.randint(1, max(1, max_hole))
        while len(blob) < target:
            candidates = []
            for r, c in blob:
                for nr, nc in neighbors4(r, c):
                    if not (margin <= nr < csz - margin and margin <= nc < csz - margin):
                        continue
                    if (nr, nc) in forbidden:
                        continue
                    if any((nnr, nnc) in forbidden and (nnr, nnc) not in blob for nnr, nnc in neighbors4(nr, nc)):
                        continue
                    candidates.append((nr, nc))
            if not candidates:
                break
            choice = random.choice(candidates)
            forbidden.add(choice)
            blob.add(choice)
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


def _grow_mask(csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]) -> np.ndarray:
    """Grow a connected blob while avoiding the forbidden cells."""
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
    """Seal gaps near holes, then keep only the largest connected component."""
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


def _crop_and_validate(mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int) -> np.ndarray | None:
    """Crop to a tight bbox, then reject illegal shapes or wrong hole counts."""
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
    return np.where(cropped, LIGHT_GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    """Generate a random sprite template using LIGHT_GREY as the foreground color."""
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


def _colorize_sprite(sprite: np.ndarray, shade: int) -> np.ndarray:
    """Repaint a sprite template into the requested input shade."""
    return np.where(sprite != BG, int(shade), BG).astype(int)


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connected dilation for one-cell exclusion padding."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location_in_rect(
    rect: tuple[int, int, int, int], occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Sample a valid sprite location inside one workspace rectangle."""
    r0, c0, r1, c1 = rect
    h, w = sprite.shape
    sprite_mask = sprite != BG
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


def _base_region_specs(region_maps: dict[str, dict[int, int]]) -> dict[str, list[tuple[int, int, int | None]]]:
    """
    Build a non-degenerate sprite plan.

    Guarantees:
    - shared hole-count appears in both regions with different colors,
    - region A has a dark-vs-light conflict on the shared hole-count,
    - region B has a light-only winner for that same shared hole-count,
    - each region also contains a hole-count that is valid only in the other region, acting as a local distractor.
    """
    region_keys = sorted(region_maps)
    assert len(region_keys) == 2, "cross layout must expose exactly two workspace regions"
    region_a, region_b = region_keys

    holes_a = set(region_maps[region_a])
    holes_b = set(region_maps[region_b])
    shared = next(iter(holes_a & holes_b))
    unique_a = next(iter(holes_a - {shared}))
    unique_b = next(iter(holes_b - {shared}))

    specs = {
        region_a: [
            (shared, DARK_GREY, region_maps[region_a][shared]),
            (shared, LIGHT_GREY, None),
            (unique_a, LIGHT_GREY, region_maps[region_a][unique_a]),
            (unique_b, random.choice([LIGHT_GREY, DARK_GREY]), None),
        ],
        region_b: [
            (shared, LIGHT_GREY, region_maps[region_b][shared]),
            (unique_b, DARK_GREY, region_maps[region_b][unique_b]),
            (unique_b, LIGHT_GREY, None),
            (unique_a, random.choice([LIGHT_GREY, DARK_GREY]), None),
        ],
    }
    return specs


def _extra_region_specs(region: str, region_maps: dict[str, dict[int, int]]) -> list[tuple[int, int, int | None]]:
    """Sample optional extras without removing the core fused behavior."""
    local_holes = sorted(region_maps[region])
    other_region = next(key for key in region_maps if key != region)
    other_only = next(iter(set(region_maps[other_region]) - set(local_holes)))
    extras = []
    for hole_count in local_holes:
        color = region_maps[region][hole_count]
        extras.append((hole_count, random.choice([LIGHT_GREY, DARK_GREY]), color))
    extras.append((other_only, random.choice([LIGHT_GREY, DARK_GREY]), None))
    return extras


def _place_region_sprites(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    rects: dict[str, tuple[int, int, int, int]],
    region_maps: dict[str, dict[int, int]],
) -> bool:
    """Generate and place all region-constrained sprites onto the grid."""
    total_target = random.randint(*TOTAL_SPRITES_RANGE)
    base_specs = _base_region_specs(region_maps)
    items: list[tuple[int, str, np.ndarray, int | None]] = []

    for region, specs in base_specs.items():
        for hole_count, shade, out_color in specs:
            sprite_template = _make_sprite(hole_count)
            if sprite_template is None:
                return False
            sprite = _colorize_sprite(sprite_template, shade)
            items.append((sprite.size, region, sprite, out_color))

    extra_budget = total_target - len(items)
    extra_counts = {region: 0 for region in region_maps}
    while extra_budget > 0:
        region = random.choice(sorted(region_maps))
        if extra_counts[region] >= MAX_EXTRA_PER_REGION:
            if all(extra_counts[r] >= MAX_EXTRA_PER_REGION for r in extra_counts):
                break
            continue
        hole_count, shade, out_color = random.choice(_extra_region_specs(region, region_maps))
        sprite_template = _make_sprite(hole_count)
        if sprite_template is None:
            return False
        sprite = _colorize_sprite(sprite_template, shade)
        items.append((sprite.size, region, sprite, out_color))
        extra_counts[region] += 1
        extra_budget -= 1

    items.sort(key=lambda x: x[0], reverse=True)
    for _, region, sprite, out_color in items:
        loc = _random_sprite_location_in_rect(rects[region], occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if out_color is not None:
            out_patch = np.where(sprite != BG, int(out_color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build cross layout, place sprites, validate against the solver."""
    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    expected = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    occupied = np.full_like(grid, BG)

    rects, region_maps = _cross_layout(grid)
    expected[:, :] = grid
    if not _place_region_sprites(grid, expected, occupied, rects, region_maps):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(np.isin(grid, [LIGHT_GREY, DARK_GREY]))):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate one valid fused-variant input grid."""
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
