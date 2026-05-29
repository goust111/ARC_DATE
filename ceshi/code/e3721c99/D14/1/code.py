# Variant of e3721c99
# Dimension: D14 — 图例与精灵的颜色语义 (meta)
# Change: 灰色统一标记所有源精灵，图例使用互不相同的彩色标记目标颜色... → 图例颜色对映射（两色为一组）

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the D14 variant
# a) Concrete behavior and trace:
#    The original task exposed 2-4 legend shapes, each color standing alone for one hole count.
#    Here the legend is always four shapes arranged as two visual pairs. Within each pair the two
#    shapes sit close together, while the gap between pairs is larger. A grey sprite still matches
#    by hole count, but the final color is no longer always "the color of the legend shape with the
#    same hole count". Instead, the partner relation matters:
#      - odd hole-count difference inside a pair  -> each count keeps its own legend color
#      - even hole-count difference inside a pair -> the two legend colors are swapped
#    Example: a pair has (holes=1,color=red) and (holes=3,color=green). The gap is 2, so a 1-hole
#    grey sprite becomes green and a 3-hole grey sprite becomes red. In the original task these two
#    sprites would have become red and green respectively.
# b) Complexity calibration:
#    The rule stays close to the original topology-matching task because hole count is still the
#    only sprite descriptor. The added reasoning is limited to two legend pairs, each with a binary
#    branch (keep/swap), so it is visibly different without becoming combinatorially noisy.
# c) Old assumptions that break:
#    The old _legend_mapping() assumed each colored legend component independently defines a final
#    color. That is no longer valid. main() must now recover legend order, group entries into pairs,
#    inspect partner hole-count gaps, and only then derive hole->output-color mapping.
# d) Placement legality:
#    The generator keeps the original divider/panel/workspace layout and sprite non-overlap logic.
#    Only legend spacing changes: 1-cell inside a pair, 3-cells between pairs. This gives the
#    solver a stable ordering cue without changing workspace placement constraints.
# e) Visual observability:
#    Every sample contains exactly two pairs, with one odd-gap pair and one even-gap pair. That
#    guarantees at least one pair behaves like the original rule and at least one pair visibly swaps
#    colors, so the variant is readable from a few examples.
# f) Visual cleanliness:
#    The panel remains a single line of four legend patches separated by blank cells; sprites stay
#    isolated in the workspace. No extra marks are added on top of existing objects.
# g) Color-change inferability:
#    Only four legend colors are active at once, and they are grouped by spacing. The larger
#    inter-pair gap is the structural cue that tells the human solver which two colors belong to
#    one rule.
# New solver assumptions introduced by this variant, and how generation guarantees them:
#    - there are exactly four non-border legend components;
#    - they lie on one row/column ordering axis;
#    - adjacent entries in that order form the two pairs;
#    - all four hole counts are distinct, so pair membership is unambiguous.


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


def _legend_component_records(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Extract the four internal colored legend components and order them along the legend line.
    """
    h, w = g.shape
    recs = []
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        for pts in components_mask(g == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
            recs.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "holes": int(_hole_count(sub)),
                    "center_r": (r0 + r1) / 2,
                    "center_c": (c0 + c1) / 2,
                }
            )

    assert len(recs) == LEGEND_ENTRY_COUNT, "grid must contain exactly four legend components"
    holes = [int(rec["holes"]) for rec in recs]
    assert len(set(holes)) == len(holes), "legend hole counts must be distinct"

    span_r = max(rec["center_r"] for rec in recs) - min(rec["center_r"] for rec in recs)
    span_c = max(rec["center_c"] for rec in recs) - min(rec["center_c"] for rec in recs)
    if span_c >= span_r:
        recs.sort(key=lambda rec: (rec["center_c"], rec["center_r"]))
    else:
        recs.sort(key=lambda rec: (rec["center_r"], rec["center_c"]))
    return recs


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Build hole_count -> output_color using pair semantics.

    The four legend components are read in spatial order and grouped as (0,1) and (2,3).
    Within a pair:
      - odd hole-count gap  -> keep each legend's own color
      - even hole-count gap -> swap the two colors
    """
    recs = _legend_component_records(g, bg, grey)
    mapping: dict[int, int] = {}
    for i in range(0, len(recs), 2):
        a = recs[i]
        b = recs[i + 1]
        if abs(int(a["holes"]) - int(b["holes"])) % 2 == 0:
            mapping[int(a["holes"])] = int(b["color"])
            mapping[int(b["holes"])] = int(a["color"])
        else:
            mapping[int(a["holes"])] = int(a["color"])
            mapping[int(b["holes"])] = int(b["color"])
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching hole counts, then resolving colors through legend pairs.

    Input structure:
    - bg is the most common color; grey (5) sprites are scattered across the workspace.
    - The legend area contains four colored shapes arranged as two close pairs, with a larger gap
      between the pairs. Each legend shape has a distinct number of enclosed holes.

    Rule:
    - Read the four legend shapes in order along the legend line and split them into adjacent pairs.
    - For each pair, compare the two hole counts: odd difference means "use each shape's own color",
      even difference means "swap the two colors within the pair".
    - Recolor every grey sprite using the color assigned to its hole count by that pair rule; erase
      grey distractors whose hole counts are absent from the legend.

    Key insight: Topology still identifies which legend entry a sprite belongs to, but the final
    color is determined by the relationship between paired legend entries rather than by a direct
    one-shape-one-color lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY

    legend = _legend_mapping(g, bg, grey)

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        newc = int(legend[hc]) if hc in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator (holes-first organic blob) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_ENTRY_COUNT = 4
LEGEND_PAIR_COUNT = 2
MAX_HOLE_COUNT = 6
PAIR_INNER_GAP = 1
PAIR_OUTER_GAP = 3
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 25
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
MIN_EXTRA_MATCHES = 2
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


def _flatten_pairs(legend_pairs: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [entry for pair in legend_pairs for entry in pair]


def _line_span(legend_pairs: list[list[tuple[int, int]]]) -> int:
    """Total span with 1-cell gaps inside pairs and 3-cell gaps between pairs."""
    total = 0
    for i, pair in enumerate(legend_pairs):
        total += sum(_legend_span(hole_count) for hole_count, _ in pair)
        total += PAIR_INNER_GAP
        if i < len(legend_pairs) - 1:
            total += PAIR_OUTER_GAP
    return total


def _pairings_for_four(holes: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    a, b, c, d = holes
    return [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]


def _choose_legends(_arrangement: str, max_span: int) -> list[list[tuple[int, int]]]:
    """Pick two legend pairs whose gaps are one odd and one even, and that fit within max_span."""
    candidates = []
    for holes in itertools.combinations(range(MAX_HOLE_COUNT), LEGEND_ENTRY_COUNT):
        for pair0, pair1 in _pairings_for_four(tuple(holes)):
            gaps = [abs(pair0[0] - pair0[1]), abs(pair1[0] - pair1[1])]
            if sorted(gap % 2 for gap in gaps) != [0, 1]:
                continue
            pair_holes = [list(pair0), list(pair1)]
            dummy = [[(pair_holes[0][0], 0), (pair_holes[0][1], 0)],
                     [(pair_holes[1][0], 0), (pair_holes[1][1], 0)]]
            if _line_span(dummy) <= max_span:
                candidates.append(pair_holes)
    if not candidates:
        return _choose_legends(_arrangement, max_span + 1)

    chosen = random.choice(candidates)
    random.shuffle(chosen)
    colors = random.sample(LEGEND_COLORS, LEGEND_ENTRY_COUNT)
    color_idx = 0
    legend_pairs = []
    for pair_holes in chosen:
        random.shuffle(pair_holes)
        pair = []
        for hole_count in pair_holes:
            pair.append((int(hole_count), int(colors[color_idx])))
            color_idx += 1
        legend_pairs.append(pair)
    return legend_pairs


def _color_mapping_from_pairs(legend_pairs: list[list[tuple[int, int]]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for pair in legend_pairs:
        (h0, c0), (h1, c1) = pair
        assert h0 not in mapping and h1 not in mapping, "legend hole counts must be unique"
        if abs(h0 - h1) % 2 == 0:
            mapping[h0] = c1
            mapping[h1] = c0
        else:
            mapping[h0] = c0
            mapping[h1] = c1
    return mapping


def _place_legend_line(
    grid: np.ndarray, legend_pairs: list[list[tuple[int, int]]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially, using tighter spacing inside pairs."""
    rr, cc = int(r), int(c)
    for i, pair in enumerate(legend_pairs):
        for j, (hole_count, color) in enumerate(pair):
            patch = _legend_patch(hole_count, color, arrangement)
            paste(grid, patch, rr, cc)
            if arrangement == "horizontal":
                cc += patch.shape[1]
                if j == 0:
                    cc += PAIR_INNER_GAP
            else:
                rr += patch.shape[0]
                if j == 0:
                    rr += PAIR_INNER_GAP
        if i < len(legend_pairs) - 1:
            if arrangement == "horizontal":
                cc += PAIR_OUTER_GAP
            else:
                rr += PAIR_OUTER_GAP


# ---------- layout helpers ----------

def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    """Place a full-row horizontal divider with legend panel above or below."""
    legend_pairs = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legend_pairs)
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
    _place_legend_line(grid, legend_pairs, "horizontal", r0, c0)
    return legend_pairs


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    """Place a full-column vertical divider with legend panel left or right."""
    legend_pairs = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legend_pairs)
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
    _place_legend_line(grid, legend_pairs, "vertical", r0, c0)
    return legend_pairs


def _l_corner_geometry(corner: str, panel_h: int, panel_w: int,
                       arrangement: str, total: int) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend placement bounds for an L-corner."""
    N = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, N - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, N - 1 - along_dim
    elif corner == "bl":
        dr, dc = N - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, N - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:  # br
        dr, dc = N - 1 - panel_h, N - 1 - panel_w
        r_min, r_max = dr + 2, N - 1 - cross_dim
        c_min, c_max = dc + 2, N - 1 - along_dim

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    """Place an L-shaped divider with legend panel in one corner."""
    arrangement = random.choice(["horizontal", "vertical"])
    legend_pairs = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legend_pairs)
    extra = max(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    along = total + 2 + random.randint(0, extra)
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, arrangement, total)

    N = GRID_SIZE
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
    _place_legend_line(grid, legend_pairs, arrangement, r0, c0)
    return legend_pairs


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[list[tuple[int, int]]]:
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


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
    """Find well-separated seed positions for holes inside the canvas."""
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
    positions = [(r, c) for r in range(center - radius, center + radius + 1)
                 for c in range(center - radius, center + radius + 1)
                 if margin <= r < csz - margin and margin <= c < csz - margin]
    random.shuffle(positions)
    seeds = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(seeds: list[tuple[int, int]], hole_count: int,
                     area: int, csz: int) -> set[tuple[int, int]]:
    """Grow irregular blob regions around each hole seed. Returns the full forbidden set."""
    margin = HOLE_SEED_MARGIN
    max_hole = (max(2, area // (hole_count + 2)) if hole_count <= 2
                else max(1, area // (hole_count * 2 + 1)))
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
                    if any((nnr, nnc) in forbidden and (nnr, nnc) not in blob
                           for nnr, nnc in neighbors4(nr, nc)):
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


def _grow_mask(csz: int, area: int, start: tuple[int, int],
               forbidden: set[tuple[int, int]]) -> np.ndarray:
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
            best = max(range(1, n_comp + 1), key=lambda l: int(np.sum(labeled == l)))
            mask = labeled == best
    return mask


def _crop_and_validate(mask: np.ndarray, forbidden: set[tuple[int, int]],
                       hole_count: int) -> np.ndarray | None:
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
    cropped = mask[rmin:rmax + 1, cmin:cmax + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    """Generate a random amorphous grey sprite with the specified number of holes."""
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


# ---------- sprite planning & placement ----------

def _sprite_plan(legend_pairs: list[list[tuple[int, int]]]) -> list[tuple[int, int | None]]:
    """Build a shuffled list of (hole_count, output_color_or_None) for all sprites to generate."""
    flat = _flatten_pairs(legend_pairs)
    targets = _color_mapping_from_pairs(legend_pairs)

    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(flat)
    max_d = min(MAX_DISTRACTORS, remaining - MIN_EXTRA_MATCHES)
    valid_d = [d for d in range(1, max_d + 1)
               if MIN_EXTRA_MATCHES <= remaining - d <= MAX_EXTRA_PER_LEGEND * len(flat)]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors

    extras = [0] * len(flat)
    for pair_idx in range(LEGEND_PAIR_COUNT):
        base = 2 * pair_idx
        extras[base + random.randint(0, 1)] += 1
    for _ in range(extra_matches - LEGEND_PAIR_COUNT):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, (hole_count, _color) in enumerate(flat):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, targets[hole_count]))
    legend_holes = {h for h, _ in flat}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None))
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask (1-cell padding in all directions)."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement for sprite, avoiding blocked and occupied areas."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    N = GRID_SIZE
    candidates = [(r, c) for r in range(N - h + 1) for c in range(N - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(N, r + h + 1), min(N, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legend_pairs: list[list[tuple[int, int]]]
) -> bool:
    """Generate and place all sprites (matches + distractors) onto the grid."""
    items = []
    for hole_count, color in _sprite_plan(legend_pairs):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, color))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, color in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if color is not None:
            out_patch = np.where(sprite == GREY, int(color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


# ---------- top-level generation ----------

def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, validate via oracle."""
    N = GRID_SIZE
    grid = np.zeros((N, N), dtype=int)
    expected = np.zeros((N, N), dtype=int)
    blocked = np.zeros((N, N), dtype=int)

    legend_pairs = _add_layout(grid, blocked)
    mapping = _color_mapping_from_pairs(legend_pairs)
    direct = {hole_count: color for hole_count, color in _flatten_pairs(legend_pairs)}
    if sum(int(mapping[h] != direct[h]) for h in mapping) < 2:
        return None

    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legend_pairs):
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
    raise RuntimeError("failed to generate e3721c99 D14 input")
