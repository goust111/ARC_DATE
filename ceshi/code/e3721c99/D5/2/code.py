# Variant of e3721c99
# Dimension: D5 — 符号指令：图例的编码方式 (symbol_instructions)
# Change: 图例形状直接用自身的洞数编码匹配关系——计算图例形状的洞数，灰色精灵与之洞数相同则获得该图例颜色... → 图例用颜色对表示（左色=图例色，右色=匹配特征色）

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis:
#   variant_spec:
#     grid:
#       size: 30x30
#       persistent_regions:
#         - divider color 1 still separates legend panel from workspace
#         - legend remains inside the blocked panel, sprites remain in workspace
#     legend:
#       encoding:
#         - each legend entry is one horizontal row of two cells
#         - left cell = target recolor color
#         - right cell = marker color that must appear on a sprite's hole rim
#       arrangement:
#         - 2-4 rows stacked vertically
#         - inset by one bg cell from the outer border and from the divider
#       palette:
#         - legend colors exclude bg=0, divider=1, grey=5 so the pair table is visually isolated
#     sprites:
#       structure:
#         - still connected amorphous blobs derived from the original generator
#         - every sprite now has at least one enclosed hole, because the new instruction lives on hole boundaries
#         - cells adjacent to hole interiors are recolored from grey to one marker color; outer mass stays grey
#       matching:
#         - marker color on the hole rim, not hole count, selects the output color
#         - marker colors absent from the legend create distractor sprites that are erased to bg
#   behavior_trace:
#     original_local_step:
#       - read a colored legend shape
#       - compute that shape's hole count
#       - compute each grey sprite's hole count
#       - use hole_count -> color mapping
#     variant_local_step:
#       - read a legend row [fill, marker]
#       - find each sprite component and inspect the non-grey color touching its enclosed holes
#       - use marker_color -> fill_color mapping
#     visible_difference:
#       - two sprites may share one hole count but get different outputs because their rim markers differ
#       - one legend row can recolor sprites with different hole counts to the same output color
#   complexity_calibration:
#     - the task still has only 2-4 legend entries and 8-11 sprites, matching the original cognitive scale
#     - marker colors are few and repeated across multiple sprites, so the mapping can be inferred from 2-4 examples
#     - hole counts still vary, preventing accidental reinterpretation as a trivial direct recolor task
#   implicit_assumption_review:
#     original_main_dependencies:
#       - _legend_mapping assumed each legend color appeared as exactly one sizeable interior component
#       - the recolor loop assumed every relevant sprite cell was grey
#       - matching assumed topology alone encoded identity
#     variant_fixes:
#       - legend parsing now looks for one compact stacked 2-column table instead of per-color blobs
#       - sprite processing works on full non-bg workspace components, so colored rim cells are recolored too
#       - hole counting is no longer used for matching; holes are only used to locate the marker-bearing boundary
#   placement_legality:
#     - legend rows are placed inside the blocked panel with explicit bbox checks
#     - sprites use the original non-overlap placement strategy, but occupancy now treats marker cells as part of the sprite
#     - generator validates every sample with main(input) == expected_output
#   visual_observability:
#     - the marker sits exactly around visible holes, so the right-hand legend color has a concrete place to look at
#     - the generator forces at least one shared hole-count across different legend markers, so the new rule cannot collapse back to the original topology rule
#   neatness:
#     - legend colors never share divider/grey/bg ids
#     - legend table stays in its own panel; sprites remain separated by bg padding
#   color_inference:
#     - every legend row is an explicit [output, marker] pair
#     - marker colors are unique within a sample, limiting ambiguity
#
# Solver assumption audit after the change:
#   - _find_legend_pairs assumes exactly one vertical stack of adjacent distinct color pairs exists; generate_input() creates exactly one such stack and keeps workspace sprites from forming pair-like rows.
#   - _component_marker assumes each sprite has at least one enclosed hole and all hole-adjacent non-grey cells share one marker color; generator enforces hole_count >= 1 and colors every hole rim uniformly.
#   - main() recolors every non-bg cell of a sprite component; generator ensures marker cells belong only to sprites, never to the legend/work divider mask.


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


def _find_legend_pairs(g: np.ndarray, bg: int, grey: int, div: int) -> tuple[dict[int, int], np.ndarray]:
    """
    Find the unique vertical stack of legend rows.
    Each row is two adjacent non-(bg,grey,div) cells: [fill_color, marker_color].
    """
    h, w = g.shape

    def good(v: int) -> bool:
        return int(v) not in (bg, grey, div)

    candidates: list[tuple[int, int, list[tuple[int, int, int]]]] = []
    for c in range(w - 1):
        r = 0
        while r < h:
            left = int(g[r, c])
            right = int(g[r, c + 1])
            side_ok = (c == 0 or int(g[r, c - 1]) in (bg, div)) and (
                c + 2 == w or int(g[r, c + 2]) in (bg, div)
            )
            if good(left) and good(right) and left != right and side_ok:
                start = r
                entries: list[tuple[int, int, int]] = []
                while r < h:
                    left = int(g[r, c])
                    right = int(g[r, c + 1])
                    side_ok = (c == 0 or int(g[r, c - 1]) in (bg, div)) and (
                        c + 2 == w or int(g[r, c + 2]) in (bg, div)
                    )
                    if not (good(left) and good(right) and left != right and side_ok):
                        break
                    entries.append((r, left, right))
                    r += 1
                if len(entries) >= 2:
                    candidates.append((start, c, entries))
            else:
                r += 1

    assert len(candidates) == 1, "legend must be a unique stacked 2-column color table"
    _, c, entries = candidates[0]

    mapping: dict[int, int] = {}
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, fill, marker in entries:
        assert marker not in mapping, "marker colors must be unique within the legend"
        mapping[int(marker)] = int(fill)
        legend_mask[r, c] = True
        legend_mask[r, c + 1] = True
    return mapping, legend_mask


def _component_marker(subgrid: np.ndarray, comp_mask: np.ndarray, grey: int) -> int | None:
    """Read the unique non-grey color that borders the component's enclosed holes."""
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
    Recolor hole-marked sprites by matching their rim-marker color to a legend pair table.

    Input structure:
    - bg is the dominant color; color 1 is still a divider that fences off a legend panel.
    - Inside that panel sits a vertical stack of 2-cell rows: left cell = output color, right cell = marker color.
    - Workspace sprites are mostly grey (5), but every enclosed hole has its boundary tinted with one marker color.

    Rule:
    - Read the legend as marker_color -> output_color from the right and left cells of each row.
    - For every workspace component, find its enclosed hole(s) and inspect the non-grey color touching those hole interiors.
    - Recolor the entire sprite to the mapped output color; if its marker color is absent from the legend, erase the sprite to bg.

    Key insight: The instruction is carried by the colored hole rim, while hole count becomes incidental variation rather than the matching key.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5
    div = 1

    legend, legend_mask = _find_legend_pairs(g, bg, grey, div)
    assert legend, "legend must contain at least one marker->fill mapping"

    out = g.copy()
    sprite_mask = (g != bg) & (g != div) & (~legend_mask)
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


# --------------- generator (rim-marker sprites) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
PANEL_SIZE_RANGE = (5, 8)

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
MAX_SPRITE_HOLES = 5
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

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


def _legend_patch(legends: list[tuple[int, int]]) -> np.ndarray:
    """Build a vertical stack of [fill, marker] legend rows."""
    patch = np.zeros((len(legends), 2), dtype=int)
    for r, (fill, marker) in enumerate(legends):
        patch[r, 0] = int(fill)
        patch[r, 1] = int(marker)
    return patch


def _choose_legends() -> list[tuple[int, int]]:
    """Pick unique fill colors and unique marker colors for the legend."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    fills = random.sample(LEGEND_COLORS, count)
    markers: list[int] = []
    for fill in fills:
        choices = [c for c in LEGEND_COLORS if c != fill and c not in markers]
        preferred = [c for c in choices if c not in fills]
        markers.append(random.choice(preferred or choices))
    return list(zip(fills, markers))


def _place_legend_stack(grid: np.ndarray, legends: list[tuple[int, int]], panel_bbox: tuple[int, int, int, int]) -> None:
    """Place the stacked legend rows inside the free panel bbox."""
    rmin, cmin, rmax, cmax = panel_bbox
    patch = _legend_patch(legends)
    h, w = patch.shape
    assert rmax - rmin + 1 >= h and cmax - cmin + 1 >= w, "legend panel must fit the stack"
    r0 = random.randint(rmin, rmax - h + 1)
    c0 = random.randint(cmin, cmax - w + 1)
    paste(grid, patch, r0, c0)


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray, legend_h: int) -> tuple[int, int, int, int]:
    """Place a full-row divider with the legend panel above or below it."""
    panel_h = random.randint(max(PANEL_SIZE_RANGE[0], legend_h + 2), PANEL_SIZE_RANGE[1])
    top = random.choice([True, False])
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        return 1, 1, div_r - 2, GRID_SIZE - 2
    div_r = GRID_SIZE - 1 - panel_h
    grid[div_r, :] = DIV
    blocked[div_r:, :] = 9
    return div_r + 2, 1, GRID_SIZE - 2, GRID_SIZE - 2


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray, legend_h: int) -> tuple[int, int, int, int]:
    """Place a full-column divider with the legend panel on the left or right."""
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        return 1, 1, GRID_SIZE - 2, div_c - 2
    div_c = GRID_SIZE - 1 - panel_w
    grid[:, div_c] = DIV
    blocked[:, div_c:] = 9
    return 1, div_c + 2, GRID_SIZE - 2, GRID_SIZE - 2


def _layout_l(grid: np.ndarray, blocked: np.ndarray, legend_h: int) -> tuple[int, int, int, int]:
    """Place an L-shaped divider with a corner panel."""
    panel_h = random.randint(max(PANEL_SIZE_RANGE[0], legend_h + 2), PANEL_SIZE_RANGE[1])
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    corner = random.choice(["tl", "tr", "bl", "br"])

    if corner == "tl":
        dr, dc = panel_h, panel_w
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
        return 1, 1, dr - 2, dc - 2
    if corner == "tr":
        dr, dc = panel_h, GRID_SIZE - 1 - panel_w
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
        return 1, dc + 2, dr - 2, GRID_SIZE - 2
    if corner == "bl":
        dr, dc = GRID_SIZE - 1 - panel_h, panel_w
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
        return dr + 2, 1, GRID_SIZE - 2, dc - 2
    dr, dc = GRID_SIZE - 1 - panel_h, GRID_SIZE - 1 - panel_w
    grid[dr, dc:] = DIV
    grid[dr:, dc] = DIV
    blocked[dr + 1 :, dc + 1 :] = 9
    blocked[dr, dc:] = 9
    blocked[dr:, dc] = 9
    return dr + 2, dc + 2, GRID_SIZE - 2, GRID_SIZE - 2


def _add_layout(grid: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]) -> None:
    """Draw one divider layout and place the legend stack inside its panel."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        panel_bbox = _layout_full_horizontal(grid, blocked, len(legends))
    elif kind == "vertical":
        panel_bbox = _layout_full_vertical(grid, blocked, len(legends))
    else:
        panel_bbox = _layout_l(grid, blocked, len(legends))
    _place_legend_stack(grid, legends, panel_bbox)


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
    """Crop mask to tight bbox (including holes), validate size/aspect, and emit a grey sprite."""
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
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int, int | None]] = []
    shared_holes = random.randint(1, MAX_SPRITE_HOLES)
    shared_idxs = set(random.sample(range(len(legends)), 2))
    for i, (fill, marker) in enumerate(legends):
        count = 1 + extras[i]
        first_holes = shared_holes if i in shared_idxs else random.randint(1, MAX_SPRITE_HOLES)
        plan.append((first_holes, marker, fill))
        for _ in range(count - 1):
            plan.append((random.randint(1, MAX_SPRITE_HOLES), marker, fill))

    legend_markers = {marker for _, marker in legends}
    distractor_markers = [c for c in LEGEND_COLORS if c not in legend_markers]
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
    """Single attempt: build layout, place sprites, and validate via the solver."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _choose_legends()
    _add_layout(grid, blocked, legends)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum((grid != BG) & (grid != DIV))):
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
    raise RuntimeError("failed to generate e3721c99 D5 variant input")
