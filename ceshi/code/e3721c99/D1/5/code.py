# Variant of e3721c99
# Dimension: D1 — 匹配条件：洞数匹配 vs 其他匹配逻辑 (match_cond)
# Change: s.hole_count == d.hole_count — 灰色精灵的洞数与图例形状的洞数一致时匹配... → 边缘颜色匹配

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
# variant_spec:
#   core_change:
#     from: grey sprite is identified by its own hole count
#     to: grey sprite is identified by colored marker pixels touching its outer edge
#   preserved_structure:
#     - 30x30 grid, bg=0, grey sprites remain irregular 4-connected blobs
#     - divider and legend panel layouts remain horizontal / vertical / L-shaped
#     - legend colors are still expressed by one larger interior component per color
#     - distractor sprites still disappear in the output
#   concretization:
#     legends:
#       - keep the original 3-by-span or span-by-3 patches with distinct colors
#       - their hole counts are still varied for visual continuity, but no longer used for matching
#     sprite_markers:
#       - each grey sprite gets 1-2 orthogonally adjacent single-cell colored markers
#       - matched sprites use a marker color that appears in the legend panel
#       - distractors use a non-legend marker color, so they still get erased
#       - markers are placed just outside the sprite bbox, never inside holes
#   behavior_trace:
#     original:
#       - find legend hole_count -> color mapping
#       - crop each grey component, count its holes, recolor by that lookup
#     variant:
#       - find the set of valid legend colors from the large interior legend patches
#       - for each grey component, inspect the 4-neighbor cells touching its boundary
#       - if exactly one legend color appears there, recolor to that color; otherwise erase
#     concrete_difference:
#       - two sprites can have identical hole counts yet end up with different output colors
#       - one sprite can keep the same shape while changing output color when only its edge marker changes
#   complexity_calibration:
#     - legend colors stay limited to 2-4, so humans can infer the palette from few examples
#     - all sprites still have varied hole counts, avoiding a too-trivial "marker only" look
#     - generator forces at least one pair of matched sprites with the same hole count but different marker colors,
#       so the new rule is visibly different from the original topology-based one
#   implicit_assumption_audit:
#     old_legend_mapping:
#       dependency: every relevant color could be keyed by hole count
#       status_now: false; large legend components are still identifiable, but only their colors matter
#     old_sprite_loop:
#       dependency: each sprite carries all information needed for recoloring inside its own mask
#       status_now: false; solver must inspect external neighboring cells in the original grid
#     old_assignment:
#       dependency: unmatched hole count means erase, and matching is never ambiguous
#       status_now: adapted; sprites with zero or multiple adjacent legend colors are erased
#   placement_legality:
#     - markers are reserved in the occupancy grid, so later sprites cannot overlap or touch them
#     - blocked legend/divider zones are still respected
#     - markers are chosen from valid in-bounds edge-adjacent cells only
#   visual_observability:
#     - matched sprites repeat legend colors at their edges, making the cause of recoloring local and visible
#     - same-hole/different-color witness sprites prevent accidental interpretation by hole count
#   visual_cleanliness:
#     - markers are single cells outside the sprite, so they do not cover the sprite body or legend panel
#     - objects remain separated by at least one background cell
#   color_inference:
#     - only legend colors can trigger recoloring
#     - distractor markers use non-legend colors, giving a clear negative example


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


def _legend_colors(g: np.ndarray, bg: int, grey: int) -> set[int]:
    """
    Pick one legend component per non-(bg,grey) color:
      - component is at least 2x2,
      - component does not touch the outer grid border (filters dividers),
      - marker pixels stay too small to qualify.
    Return the set of valid legend colors.
    """
    h, w = g.shape
    legend_colors: set[int] = set()
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
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        legend_colors.add(int(c))
    return legend_colors


def _adjacent_legend_colors(
    g: np.ndarray, pts: np.ndarray, legend_colors: set[int], grey: int
) -> set[int]:
    """Collect legend colors present in the 4-neighborhood around a grey component."""
    h, w = g.shape
    found: set[int] = set()
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w:
                col = int(g[rr, cc])
                if col != grey and col in legend_colors:
                    found.add(col)
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by reading the colored tags attached to their edges.

    Input structure:
    - bg is the most common color; grey (5) sprites are scattered through the workspace.
    - A divider still separates the workspace from a legend panel made of larger colored patches.
    - Near each grey sprite there are one or two single-cell colored markers touching the sprite orthogonally.

    Rule:
    - The large interior legend patches define which colors are valid output colors.
    - For each grey sprite, inspect the cells in its 4-neighborhood and collect adjacent legend colors.
    - If exactly one legend color is present on that boundary, repaint the whole sprite with it; otherwise erase the sprite.

    Key insight: The deciding cue sits outside the sprite, on its boundary, so topology no longer determines the match.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend_colors = _legend_colors(g, bg, grey)
    assert legend_colors, "legend colors must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        adjacent = _adjacent_legend_colors(g, pts, legend_colors, grey)
        newc = next(iter(adjacent)) if len(adjacent) == 1 else bg
        for r, c in pts:
            out[r, c] = int(newc)

    return out


# --------------- generator (edge-tagged organic blob variant) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NONLEGEND_MARKER_COLORS = [2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
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
MATCH_MARKER_COUNT_RANGE = (1, 2)
DISTRACTOR_MARKER_COUNT_RANGE = (1, 2)

# -- generation --
GEN_RETRIES = 300

# 8-connectivity structuring element for dilation
_STRUCT8 = generate_binary_structure(2, 2)


# ---------- legend helpers ----------

def _legend_span(hole_count: int) -> int:
    """Width (or height) of a single legend patch along the arrangement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan (horizontal) or Spanx3 (vertical) legend patch."""
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


# ---------- layout helpers ----------

def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row horizontal divider with legend panel above or below."""
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
    """Place a full-column vertical divider with legend panel left or right."""
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

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place an L-shaped divider with legend panel in one corner."""
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

    n = GRID_SIZE
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
    """Randomly choose and apply a divider layout (horizontal/vertical/L)."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


# ---------- sprite generation ----------

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
    """Grow irregular blob regions around each hole seed. Returns the forbidden set."""
    margin = HOLE_SEED_MARGIN
    max_hole = (
        max(2, area // (hole_count + 2))
        if hole_count <= 2
        else max(1, area // (hole_count * 2 + 1))
    )
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


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop mask to tight bbox (including forbidden cells), validate shape and holes."""
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
        csz = max(CANVAS_MIN, int(area**0.5 * CANVAS_SCALE) + CANVAS_PAD)
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


# ---------- sprite planning & placement ----------

def _prefer_mismatched_hole(target_hole: int) -> int:
    """Choose a sprite hole count, usually different from the legend's decorative hole count."""
    options = list(range(MAX_HOLE_COUNT))
    mismatched = [h for h in options if h != target_hole]
    if mismatched and random.random() < 0.8:
        return int(random.choice(mismatched))
    return int(random.choice(options))


def _sprite_plan(legends: list[tuple[int, int]]) -> list[dict[str, int | None]]:
    """Build a shuffled plan of sprites with independent hole counts and marker colors."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d
        for d in range(1, max_d + 1)
        if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[dict[str, int | None]] = []
    for i, (legend_hole, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append(
                {
                    "hole_count": _prefer_mismatched_hole(legend_hole),
                    "target_color": int(color),
                    "marker_color": int(color),
                    "legend_hole": int(legend_hole),
                }
            )

    if len(legends) >= 2:
        shared_hole = int(legends[0][0])
        first = next(i for i, item in enumerate(plan) if item["target_color"] == int(legends[0][1]))
        second = next(i for i, item in enumerate(plan) if item["target_color"] == int(legends[1][1]))
        plan[first]["hole_count"] = shared_hole
        plan[second]["hole_count"] = shared_hole

    distractor_colors = [c for c in NONLEGEND_MARKER_COLORS if c not in {col for _, col in legends}]
    for _ in range(n_distractors):
        plan.append(
            {
                "hole_count": random.randrange(MAX_HOLE_COUNT),
                "target_color": None,
                "marker_color": int(random.choice(distractor_colors)),
                "legend_hole": None,
            }
        )

    if all(item["target_color"] is None or item["hole_count"] == item["legend_hole"] for item in plan):
        for item in plan:
            if item["target_color"] is not None and item["legend_hole"] is not None:
                item["hole_count"] = int((int(item["legend_hole"]) + 1) % MAX_HOLE_COUNT)
                break

    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask (1-cell padding in all directions)."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _edge_marker_candidates(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray, r0: int, c0: int
) -> list[tuple[int, int]]:
    """Candidate marker cells: 4-neighbors just outside the sprite bbox, not inside holes."""
    h, w = sprite.shape
    mask = sprite == GREY
    out: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(mask):
        rr = int(rr)
        cc = int(cc)
        if rr == 0:
            out.add((int(r0 - 1), int(c0 + cc)))
        if rr == h - 1:
            out.add((int(r0 + h), int(c0 + cc)))
        if cc == 0:
            out.add((int(r0 + rr), int(c0 - 1)))
        if cc == w - 1:
            out.add((int(r0 + rr), int(c0 + w)))
    valid = []
    for r, c in out:
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and static_blocked[r, c] == BG and occupied[r, c] == BG:
            valid.append((int(r), int(c)))
    return valid


def _pick_marker_positions(
    candidates: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    """Pick non-touching marker cells from the candidate edge positions."""
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - pr) + abs(c - pc) > 1 for pr, pc in picked):
            picked.append((int(r), int(c)))
            if len(picked) == count:
                return picked
    return picked if len(picked) >= count else None


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray, min_markers: int
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Find a placement whose sprite and edge marker slots fit without collisions."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
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
        marker_cands = _edge_marker_candidates(static_blocked, occupied, sprite, r, c)
        if len(marker_cands) >= min_markers:
            return int(r), int(c), marker_cands
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all sprites with their edge markers."""
    items = []
    for spec in _sprite_plan(legends):
        sprite = _make_sprite(int(spec["hole_count"]))
        if sprite is None:
            return False
        items.append({"bbox_area": int(sprite.size), "sprite": sprite, **spec})
    items.sort(key=lambda item: int(item["bbox_area"]), reverse=True)

    occupied = np.zeros_like(grid)
    for item in items:
        sprite = item["sprite"]
        marker_range = MATCH_MARKER_COUNT_RANGE if item["target_color"] is not None else DISTRACTOR_MARKER_COUNT_RANGE
        marker_count = random.randint(*marker_range)
        loc = _random_sprite_location(blocked, occupied, sprite, marker_count)
        if loc is None:
            return False
        r, c, marker_cands = loc
        marker_positions = _pick_marker_positions(marker_cands, marker_count)
        if marker_positions is None:
            return False

        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        for mr, mc in marker_positions:
            grid[mr, mc] = int(item["marker_color"])
            expected[mr, mc] = int(item["marker_color"])
            occupied[mr, mc] = int(item["marker_color"])

        if item["target_color"] is not None:
            out_patch = np.where(sprite == GREY, int(item["target_color"]), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


# ---------- top-level generation ----------

def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, validate via oracle."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
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
    raise RuntimeError("failed to generate e3721c99 D1 edge-color variant input")
