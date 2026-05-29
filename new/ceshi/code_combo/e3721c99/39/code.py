# Combo variant of e3721c99 (index 39)
# combo_spec: D3/0+D5/2
# Name: 图例分散四边+颜色对表示

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis:
#   a) concretization_and_trace:
#     - Original e3721c99 let one legend shape directly encode a hole count, so the solver could stay local:
#       inspect one colored blob, count its holes, then reuse that count on grey sprites.
#     - This combo removes that locality twice. A 30x30 sample now has four tiny 1x2 legend pairs, one on each
#       side of the outer border. Example: top [2,8], bottom [7,4], left [3,9], right [6,1].
#       The left cell is the output color; the right cell is the marker color to look for around sprite holes.
#     - A workspace sprite can have 1, 2, or 4 holes; that number is now incidental. What matters is the unique
#       non-grey color painted on the rim cells touching every enclosed hole. If that rim color is 8, the sprite
#       turns color 2, regardless of hole count or shape geometry.
#     - Difference from the original: "count holes -> recolor" becomes "collect edge pairs -> read rim marker ->
#       recolor". Hole count is preserved only as visible variation so the new rule is not a trivial direct recolor.
#   b) complexity_calibration:
#     - Exactly four legend entries keeps the four-edge distribution unmistakable while staying within ARC scale.
#     - Sprite count stays close to the original scale at 7-9, which is enough repeated evidence for the mapping
#       while keeping the four-edge layout placeable and visually clean.
#     - The generator forces at least one hole count to appear under two different output colors, and at least one
#       legend marker to appear on sprites with different hole counts. This prevents accidental collapse back to the
#       original topology-only rule.
#   c) implicit_assumption_review:
#     - Original _legend_mapping assumed sizeable interior monochrome legend components; that is invalid here because
#       legends are now border-touching 1x2 pairs. The solver therefore parses only isolated non-grey pairs on edges.
#     - Original sprite loop assumed relevant components were exactly the grey cells. That also breaks here because
#       marker cells are part of the sprite, so the solver must process all non-bg, non-legend component cells.
#     - Matching no longer assumes a bijection hole_count -> color. The generator must therefore guarantee readable
#       hole rims and unique marker colors in the legend.
#   d) placement_legality:
#     - Legend pairs are isolated on rows 0 / H-1 and columns 0-1 / W-2:W, with blank guard cells around them.
#     - A blocked border band keeps sprites away from legends, so components never fuse with edge instructions.
#     - Sprite placement still uses padded occupancy checks, and every sample is validated by main(input) == output.
#   e) visual_observability:
#     - Humans can visibly inspect all four borders, collect four [fill, marker] pairs, then look for the same
#       marker color hugging holes inside sprites. The marker sits exactly where the legend says to look.
#     - Because different sprites can share a hole count yet receive different fills, examples visibly refute the
#       original rule and support the new color-pair rule.
#   f) neatness:
#     - The edge legend is one-cell tall and isolated; sprites live in a clean interior workspace with background
#       padding. No divider or overlaid decoration is needed, so object roles stay visually separate.
#   g) inferability_of_color_change:
#     - Only four marker->fill pairs exist in a sample, each repeated across multiple sprites. The mapping is not
#       hidden in numeric color ids; it is explicitly shown by left/right position within each border pair.
#
# Solver assumption audit after the change:
#   - _find_edge_legend_pairs assumes exactly one isolated horizontal pair on each side; generate_input() creates
#     one per side and reserves guard cells so sprites cannot create false positives on the border.
#   - _component_marker assumes each sprite has at least one enclosed hole and all its hole-rim marker cells share
#     one non-grey color; the generator enforces hole_count >= 1 and paints every hole rim uniformly.
#   - main() treats every non-bg, non-legend cell as part of a sprite component; the generator keeps legends on the
#     border and all sprites in the interior, so this partition is valid.


def _hole_regions(mask: np.ndarray) -> list[np.ndarray]:
    """Return enclosed background regions inside a binary mask using 4-connectivity."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            pts = []
            touches = False
            while st:
                r, c = st.pop()
                pts.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes.append(np.array(pts, dtype=int))
    return holes


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed background components inside a binary mask."""
    return len(_hole_regions(mask))


def _find_edge_legend_pairs(g: np.ndarray, bg: int, grey: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Read the four isolated border legend pairs.
    Each pair is horizontal and means [fill_color, marker_color].
    """
    h, w = g.shape

    def good(v: int) -> bool:
        return int(v) not in (bg, grey)

    def find_top() -> list[tuple[int, int, int, int]]:
        out = []
        r = 0
        for c in range(1, w - 2):
            fill = int(g[r, c])
            marker = int(g[r, c + 1])
            if not (good(fill) and good(marker) and fill != marker):
                continue
            if int(g[r, c - 1]) != bg or int(g[r, c + 2]) != bg:
                continue
            if int(g[1, c]) != bg or int(g[1, c + 1]) != bg:
                continue
            out.append((r, c, fill, marker))
        return out

    def find_bottom() -> list[tuple[int, int, int, int]]:
        out = []
        r = h - 1
        for c in range(1, w - 2):
            fill = int(g[r, c])
            marker = int(g[r, c + 1])
            if not (good(fill) and good(marker) and fill != marker):
                continue
            if int(g[r, c - 1]) != bg or int(g[r, c + 2]) != bg:
                continue
            if int(g[h - 2, c]) != bg or int(g[h - 2, c + 1]) != bg:
                continue
            out.append((r, c, fill, marker))
        return out

    def find_left() -> list[tuple[int, int, int, int]]:
        out = []
        c = 0
        for r in range(1, h - 1):
            fill = int(g[r, c])
            marker = int(g[r, c + 1])
            if not (good(fill) and good(marker) and fill != marker):
                continue
            if int(g[r, 2]) != bg:
                continue
            if int(g[r - 1, c]) != bg or int(g[r - 1, c + 1]) != bg:
                continue
            if int(g[r + 1, c]) != bg or int(g[r + 1, c + 1]) != bg:
                continue
            out.append((r, c, fill, marker))
        return out

    def find_right() -> list[tuple[int, int, int, int]]:
        out = []
        c = w - 2
        for r in range(1, h - 1):
            fill = int(g[r, c])
            marker = int(g[r, c + 1])
            if not (good(fill) and good(marker) and fill != marker):
                continue
            if int(g[r, w - 3]) != bg:
                continue
            if int(g[r - 1, c]) != bg or int(g[r - 1, c + 1]) != bg:
                continue
            if int(g[r + 1, c]) != bg or int(g[r + 1, c + 1]) != bg:
                continue
            out.append((r, c, fill, marker))
        return out

    side_entries = {
        "top": find_top(),
        "bottom": find_bottom(),
        "left": find_left(),
        "right": find_right(),
    }
    for side, entries in side_entries.items():
        assert len(entries) == 1, f"{side} side must contain exactly one isolated legend pair"

    legend_mask = np.zeros_like(g, dtype=bool)
    mapping: dict[int, int] = {}
    for entries in side_entries.values():
        r, c, fill, marker = entries[0]
        assert marker not in mapping, "legend marker colors must be unique"
        mapping[int(marker)] = int(fill)
        legend_mask[r, c] = True
        legend_mask[r, c + 1] = True
    return mapping, legend_mask


def _component_marker(subgrid: np.ndarray, comp_mask: np.ndarray, grey: int) -> int | None:
    """Read the unique non-grey color touching the component's enclosed holes."""
    markers: set[int] = set()
    holes = _hole_regions(comp_mask)
    if not holes:
        return None
    h, w = comp_mask.shape
    for hole in holes:
        for r, c in hole:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and comp_mask[nr, nc]:
                    color = int(subgrid[nr, nc])
                    if color != grey:
                        markers.add(color)
    if len(markers) == 1:
        return next(iter(markers))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hole-marked sprites by combining border-scattered legend pairs from all four sides.

    Input structure:
    - The background is dominant, and the legend is no longer a single panel: one isolated 1x2 color pair appears on
      each of the four outer edges.
    - In every pair, the left cell gives the target output color and the right cell gives the marker color to match.
    - Interior sprites are connected blobs made of grey plus one marker color painted on cells that touch enclosed holes.

    Rule:
    - Search all four borders and collect the four marker_color -> output_color pairs.
    - For each interior component, find its enclosed hole(s) and read the unique non-grey color on the hole rim.
    - Recolor the whole component to the mapped output color; if its rim marker is not listed in the edge legend,
      erase the component to background.

    Key insight: The instructions are fragmented across the border, and the actual lookup key is the colored hole rim,
    not the number of holes or the sprite's geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend, legend_mask = _find_edge_legend_pairs(g, bg, grey)
    assert legend, "legend must contain at least one marker->fill mapping"

    out = g.copy()
    sprite_mask = (g != bg) & (~legend_mask)
    for pts in components_mask(sprite_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub_mask[r - r0, c - c0] = True
        marker = _component_marker(g[r0 : r1 + 1, c0 : c1 + 1], sub_mask, grey)
        newc = int(legend[marker]) if marker in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator (four-edge color-pair legend) ---------------

BG = 0
GREY = 5
GRID_SIZE = 30
ALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_COUNT = 4

TOTAL_SPRITES_RANGE = (7, 9)
MAX_DISTRACTORS = 3
MAX_EXTRA_PER_LEGEND = 2
MAX_SPRITE_HOLES = 4

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


def _choose_legends() -> list[tuple[int, int]]:
    """Pick four disjoint [fill, marker] pairs so role is visually unambiguous."""
    fills = random.sample(ALL_COLORS, LEGEND_COUNT)
    marker_pool = [c for c in ALL_COLORS if c not in fills]
    random.shuffle(marker_pool)
    return list(zip(fills, marker_pool))


def _place_edge_legends(grid: np.ndarray, legends: list[tuple[int, int]], blocked: np.ndarray) -> None:
    """Place one isolated horizontal pair on each border side."""
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    side_map = dict(zip(sides, legends))

    top_c = random.randint(4, GRID_SIZE - 7)
    bottom_c = random.randint(4, GRID_SIZE - 7)
    left_r = random.randint(4, GRID_SIZE - 6)
    right_r = random.randint(4, GRID_SIZE - 6)

    fill, marker = side_map["top"]
    grid[0, top_c] = int(fill)
    grid[0, top_c + 1] = int(marker)

    fill, marker = side_map["bottom"]
    grid[GRID_SIZE - 1, bottom_c] = int(fill)
    grid[GRID_SIZE - 1, bottom_c + 1] = int(marker)

    fill, marker = side_map["left"]
    grid[left_r, 0] = int(fill)
    grid[left_r, 1] = int(marker)

    fill, marker = side_map["right"]
    grid[right_r, GRID_SIZE - 2] = int(fill)
    grid[right_r, GRID_SIZE - 1] = int(marker)

    blocked[0, :] = 9
    blocked[-1, :] = 9
    blocked[:, :2] = 9
    blocked[:, -2:] = 9


def _sprite_area(hole_count: int) -> int:
    """Sample a random sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
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
    """Grow a random connected blob of target area, avoiding forbidden cells."""
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
                if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc] and (nr, nc) not in forbidden:
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


def _crop_and_validate(mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int) -> np.ndarray | None:
    """Crop mask to tight bbox, validate size/aspect, and emit a grey sprite."""
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
    """Generate a random amorphous grey sprite with the specified number of holes."""
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2

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


def _mark_hole_rims(sprite: np.ndarray, marker: int) -> np.ndarray | None:
    """Recolor all hole-adjacent sprite cells with the marker color."""
    out = sprite.copy()
    mask = sprite == GREY
    holes = _hole_regions(mask)
    if not holes:
        return None
    rim = np.zeros_like(mask, dtype=bool)
    h, w = sprite.shape
    for hole in holes:
        for r, c in hole:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    rim[nr, nc] = True
    if not np.any(rim):
        return None
    out[rim] = int(marker)
    return out


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int, int | None]]:
    """Build a shuffled list of (hole_count, marker_color, fill_or_None) items."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining - 2)
    n_distractors = random.randint(1, max_d)
    extra_matches = remaining - n_distractors

    extras = [0] * len(legends)
    vary_idx = random.randrange(len(legends))
    extras[vary_idx] = 1
    for _ in range(extra_matches - 1):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    shared_hole = random.randint(1, MAX_SPRITE_HOLES)
    shared_idxs = set(random.sample(range(len(legends)), 2))
    base_holes = []
    for i in range(len(legends)):
        if i in shared_idxs:
            base_holes.append(shared_hole)
        else:
            base_holes.append(random.randint(1, MAX_SPRITE_HOLES))

    plan: list[tuple[int, int, int | None]] = []
    for i, (fill, marker) in enumerate(legends):
        plan.append((base_holes[i], marker, fill))
        extra_slots = extras[i]
        if i == vary_idx:
            choices = [h for h in range(1, MAX_SPRITE_HOLES + 1) if h != base_holes[i]]
            plan.append((random.choice(choices), marker, fill))
            extra_slots -= 1
        for _ in range(extra_slots):
            plan.append((random.randint(1, MAX_SPRITE_HOLES), marker, fill))

    legend_markers = {marker for _, marker in legends}
    distractor_markers = [c for c in ALL_COLORS if c not in legend_markers]
    for _ in range(n_distractors):
        plan.append((random.randint(1, MAX_SPRITE_HOLES), random.choice(distractor_markers), None))
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement for a sprite, avoiding blocked and occupied areas."""
    h, w = sprite.shape
    sprite_mask = sprite != BG
    occupied_pad = _dilate_mask(occupied != BG)
    candidates = [(r, c) for r in range(GRID_SIZE - h + 1) for c in range(GRID_SIZE - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(GRID_SIZE, r + h + 1), min(GRID_SIZE, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all sprites (matches + distractors) onto the grid."""
    items = []
    for hole_count, marker, fill in _sprite_plan(legends):
        base = _make_sprite(hole_count)
        if base is None:
            return False
        sprite = _mark_hole_rims(base, marker)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, fill))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, fill in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if fill is not None:
            out_patch = np.where(sprite != BG, int(fill), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build the four-edge legend, place sprites, and validate via the solver."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _choose_legends()
    _place_edge_legends(grid, legends, blocked)
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
    raise RuntimeError("failed to generate e3721c99 combo variant input")
