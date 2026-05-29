# Combo variant of e3721c99 (index 9)
# combo_spec: D1/5+D2/3
# Name: 边缘颜色匹配+对角线标记

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# combo_design:
#   summary:
#     - Preserve the original 30x30 divider + legend-panel layout and the organic grey sprites.
#     - Replace hole-count matching with color matching through external marker pixels.
#     - Specialize those markers into a diagonal pair: exactly two same-colored cells sit at
#       opposite diagonal sides of a sprite, and both are edge-adjacent to the sprite.
#   concretization:
#     legends:
#       - Keep the original colored legend patches near the border.
#       - Their hole patterns remain decorative; solver now only needs the legend colors.
#     sprite_markers:
#       - Each grey sprite gets exactly two single-cell markers of one color.
#       - The pair is diagonal across the sprite: either top-left-ish with bottom-right-ish,
#         or top-right-ish with bottom-left-ish, using exposed edge cells of the tight bbox.
#       - Each marker remains orthogonally adjacent to the sprite, so D1's edge-color cue
#         is still the active matching signal.
#   behavior_trace:
#     small_example:
#       - Suppose the legend contains red and teal patches.
#       - Sprite A has one hole and receives two red diagonal markers.
#       - Sprite B also has one hole but receives two teal diagonal markers on the other diagonal.
#       - Original task: A and B would share one output color because their hole counts match.
#       - Combo task: A turns red, B turns teal, because the solver reads the diagonal marker pair,
#         not the internal topology.
#     solver_delta:
#       - Original solver built hole_count -> color from legend shapes, then recolored each sprite
#         by its own hole count.
#       - Combo solver first extracts the valid legend colors, then checks a sprite's four possible
#         diagonal marker templates around its bbox. A color is accepted only when both cells of one
#         template are present and equal to a legend color.
#   complexity_calibration:
#     - Legend color count stays at 2-4, so the hypothesis space is still small.
#     - Sprites keep varied hole counts, including deliberate same-hole / different-color witnesses,
#       making it visually obvious that hole topology no longer explains the output.
#     - Marker structure has only two diagonal modes, which is enough to be visible without adding
#       too many cases for the human solver.
#   implicit_assumption_audit:
#     old_legend_mapping:
#       dependency: every relevant legend color is keyed by a unique hole count
#       status_now: removed; only the presence of a large legend patch of that color matters
#     old_sprite_locality:
#       dependency: all information needed to recolor a sprite lies inside the sprite mask
#       status_now: false; solver must inspect cells just outside the grey component
#     old_matching_uniqueness:
#       dependency: hole count produces at most one matching legend entry
#       status_now: replaced by "exactly one legend color forms a valid diagonal pair"
#     new_solver_assumptions:
#       - a true cue is encoded by two same-colored cells in one of the supported diagonal templates
#       - those marker cells are singletons or small isolated cues, not parts of large same-colored regions
#       - no sprite receives two different valid legend-colored diagonal pairs
#       - generator enforces all of the above
#   placement_legality:
#     - The generator reserves both sprite bodies and marker cells in the occupancy grid.
#     - Candidate placements require the sprite's 1-cell halo to avoid the divider / legend panel.
#     - Marker cells must be in-bounds, unoccupied, and outside blocked regions before placement.
#   visual_observability:
#     - The only colored cells near a grey sprite are the two diagonal markers, so their role is legible.
#     - At least one pair of matched sprites shares the same hole count but ends with different colors,
#       which gives a direct counterexample to the original rule.
#     - Distractors carry a diagonal pair in a non-legend color, showing that "any marker" is insufficient;
#       the marker color must appear in the legend panel.
#   visual_cleanliness:
#     - Markers stay outside the grey body and are only one cell each.
#     - Objects remain separated by at least one background cell, including marker occupancy.
#     - The legend panel is unchanged, so the new cue is localized in the workspace rather than layered
#       on top of the legend itself.


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


def _component_mask(pts: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Build a tight boolean mask for one grey component and return (r0, c0, mask)."""
    r0, c0, r1, c1 = bbox_coords(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[int(r) - r0, int(c) - c0] = True
    return int(r0), int(c0), mask


def _local_diagonal_pairs(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Candidate marker pairs relative to a tight sprite mask.

    Each pair uses exposed edge cells and lands one step outside the sprite, so the markers are
    edge-adjacent to the sprite body while still forming a diagonal pair across it.
    """
    h, w = mask.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    top_cols = np.flatnonzero(mask[0, :])
    bottom_cols = np.flatnonzero(mask[h - 1, :])
    left_rows = np.flatnonzero(mask[:, 0])
    right_rows = np.flatnonzero(mask[:, w - 1])

    if top_cols.size and bottom_cols.size:
        pairs.append(((-1, int(top_cols[0])), (h, int(bottom_cols[-1]))))
        pairs.append(((-1, int(top_cols[-1])), (h, int(bottom_cols[0]))))
    if left_rows.size and right_rows.size:
        pairs.append(((int(left_rows[0]), -1), (int(right_rows[-1]), w)))
        pairs.append(((int(left_rows[-1]), -1), (int(right_rows[0]), w)))

    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen = set()
    for a, b in pairs:
        if abs(a[0] - b[0]) == 0 or abs(a[1] - b[1]) == 0:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def _diagonal_marker_colors(
    g: np.ndarray, pts: np.ndarray, legend_colors: set[int], grey: int
) -> set[int]:
    """Return legend colors that appear as a valid diagonal edge-marker pair for this component."""
    h, w = g.shape
    r0, c0, mask = _component_mask(pts)
    found: set[int] = set()
    for (dr1, dc1), (dr2, dc2) in _local_diagonal_pairs(mask):
        r1, c1 = r0 + dr1, c0 + dc1
        r2, c2 = r0 + dr2, c0 + dc2
        if not (0 <= r1 < h and 0 <= c1 < w and 0 <= r2 < h and 0 <= c2 < w):
            continue
        col1 = int(g[r1, c1])
        col2 = int(g[r2, c2])
        if col1 == col2 and col1 != grey and col1 in legend_colors:
            found.add(col1)
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by reading the color encoded by their diagonal edge markers.

    Input structure:
    - The grid still contains a border-side legend panel of larger colored patches plus grey (5) sprites in the workspace.
    - Each grey sprite has exactly two single-cell colored markers outside its body.
    - Those two markers sit on opposite diagonal sides of the sprite and each one touches the sprite along an edge.

    Rule:
    - First identify which colors are legitimate legend colors by locating the large interior legend patches.
    - For every grey sprite, test the supported diagonal marker templates around its tight bounding box.
    - If one diagonal pair is present and both marker cells share a legend color, repaint the whole sprite with that color.
    - If no such legend-colored diagonal pair exists, erase the grey sprite to background.

    Key insight: The recoloring clue is the external diagonal marker pair; the legend only tells you which marker colors are meaningful.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend_colors = _legend_colors(g, bg, grey)
    assert legend_colors, "legend colors must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        colors = _diagonal_marker_colors(g, pts, legend_colors, grey)
        newc = next(iter(colors)) if len(colors) == 1 else bg
        for r, c in pts:
            out[int(r), int(c)] = int(newc)

    return out


# --------------- generator (diagonal edge-marker organic blob variant) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NONLEGEND_MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

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
MARKER_READY_SPRITE_RETRIES = 80
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


def _make_marker_ready_sprite(hole_count: int) -> np.ndarray | None:
    """Generate a sprite that supports at least one diagonal marker template."""
    for _ in range(MARKER_READY_SPRITE_RETRIES):
        sprite = _make_sprite(hole_count)
        if sprite is not None and _local_diagonal_pairs(sprite == GREY):
            return sprite
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
                    "hole_count": _prefer_mismatched_hole(int(legend_hole)),
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


def _absolute_diagonal_pairs(sprite: np.ndarray, r0: int, c0: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Shift local diagonal marker templates into grid coordinates."""
    return [
        ((int(r0 + dr1), int(c0 + dc1)), (int(r0 + dr2), int(c0 + dc2)))
        for (dr1, dc1), (dr2, dc2) in _local_diagonal_pairs(sprite == GREY)
    ]


def _valid_diagonal_pairs(
    static_blocked: np.ndarray, occupied: np.ndarray, occupied_pad: np.ndarray, sprite: np.ndarray, r0: int, c0: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All diagonal marker pairs that fit at this sprite placement."""
    valid = []
    for a, b in _absolute_diagonal_pairs(sprite, r0, c0):
        okay = True
        for r, c in (a, b):
            if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
                okay = False
                break
            if static_blocked[r, c] != BG or occupied[r, c] != BG or occupied_pad[r, c]:
                okay = False
                break
        if okay:
            valid.append((a, b))
    return valid


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int, tuple[tuple[int, int], tuple[int, int]]] | None:
    """Find a placement whose sprite and diagonal marker pair fit without collisions."""
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
        pairs = _valid_diagonal_pairs(static_blocked, occupied, occupied_pad, sprite, r, c)
        if pairs:
            return int(r), int(c), random.choice(pairs)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all sprites with diagonal edge-marker pairs."""
    items = []
    for spec in _sprite_plan(legends):
        sprite = _make_marker_ready_sprite(int(spec["hole_count"]))
        if sprite is None:
            return False
        items.append({"bbox_area": int(sprite.size), "sprite": sprite, **spec})
    items.sort(key=lambda item: int(item["bbox_area"]), reverse=True)

    occupied = np.zeros_like(grid)
    for item in items:
        sprite = item["sprite"]
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c, marker_pair = loc

        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)

        for mr, mc in marker_pair:
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
    raise RuntimeError("failed to generate e3721c99 combo variant input")
