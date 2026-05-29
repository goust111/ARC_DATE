# Variant of e3721c99
# Dimension: D5 — 符号指令：图例的编码方式 (symbol_instructions)
# Change: 图例形状直接用自身的洞数编码匹配关系——计算图例形状的洞数，灰色精灵与之洞数相同则获得该图例颜色... → 图例用点阵编码目标洞数

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste

# Design analysis for the D5 variant
# a) Concrete behavior and trace:
#    The workspace still contains irregular grey sprites whose hole count decides
#    their output color. The change is only in the legend panel. Instead of each
#    legend entry being a perforated shape, each entry is now a solid 3x3 color
#    block plus 0-5 isolated same-color dots. In horizontal panels the dots sit
#    to the right of the block; in vertical panels they sit below the block. A
#    sample transition is:
#      old: a red legend patch with two enclosed holes -> grey sprite with two
#      holes becomes red.
#      new: a red 3x3 block with exactly two red dots beside it -> grey sprite
#      with two holes becomes red.
#    The sprite-processing stage is unchanged; only legend decoding moves from
#    topological analysis of the legend object to explicit dot counting.
# b) Complexity calibration:
#    The legend still uses 2-4 entries and hole counts still range over 0-5, so
#    multiple grey sprites support each mapping. Dot layouts use only five fixed
#    slots, which keeps counting direct but visually distinct from the original
#    perforated bars.
# c) Hidden assumptions from the original main():
#    The old solver assumed one non-border legend component per color and that
#    the legend component's own holes carried the code. Both fail here because a
#    legend color now intentionally appears in several disconnected components
#    (one 3x3 block plus singleton dots), while the block itself has zero holes.
#    The new solver therefore identifies the unique 3x3 non-border component for
#    each legend color and counts same-color singleton markers in fixed nearby
#    slots.
# d) Placement legality:
#    Generator placement keeps the old divider/panel/workspace split. Legend
#    entries remain inset from the outer border and divider by at least one cell,
#    so the solver can safely ignore border-touching components as divider noise.
#    Dots are separated from the block and from each other by background cells,
#    making them unambiguous singleton markers.
# e) Visual observability:
#    A human can see that legend patches are now solid blocks with visible count
#    markers, not holey objects. Across 2-3 examples, the repeated relation
#    "number of dots beside color X equals hole-count of grey sprites recolored
#    to X" is directly observable.
# f) Visual cleanliness:
#    Legend blocks and dots stay confined to the panel and do not overlap the
#    workspace. The fixed five-slot marker geometry prevents clutter even for
#    counts 4-5, while still leaving a 1-cell gap between legend entries.
# g) Color-change inferability:
#    Colors are few (2-4 legend colors per sample), and each color's cue is
#    local: a block and its nearby dots. No ordering by color id is used.

LEGEND_BLOCK = 3
LEGEND_THICKNESS = 5
_HORIZONTAL_DOT_SLOTS = [(0, 4), (2, 4), (4, 4), (0, 6), (2, 6)]
_VERTICAL_DOT_SLOTS = [(4, 0), (4, 2), (4, 4), (6, 0), (6, 2)]


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


def _touches_outer_border(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> bool:
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _legend_slots(r0: int, c0: int, arrangement: str) -> list[tuple[int, int]]:
    slots = _HORIZONTAL_DOT_SLOTS if arrangement == "horizontal" else _VERTICAL_DOT_SLOTS
    return [(r0 + dr, c0 + dc) for dr, dc in slots]


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one solid 3x3 legend block per non-(bg,grey) color, then decode the
    hole count from isolated same-color dots placed in fixed slots near it.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        inner_large = []
        singletons = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if _touches_outer_border(r0, c0, r1, c1, h, w):
                continue
            if len(pts) == 1:
                singletons.append((r0, c0))
            else:
                inner_large.append((pts, r0, c0, r1, c1))
        if not inner_large:
            continue

        blocks = []
        for pts, r0, c0, r1, c1 in inner_large:
            if (r1 - r0 + 1) == LEGEND_BLOCK and (c1 - c0 + 1) == LEGEND_BLOCK and len(pts) == 9:
                blocks.append((r0, c0))
        assert len(blocks) == 1, f"color {c} must have exactly one 3x3 legend block"
        assert len(inner_large) == 1, f"color {c} has unexpected extra non-border components"

        r0, c0 = blocks[0]
        singletons_set = set(singletons)
        horizontal_slots = set(_legend_slots(r0, c0, "horizontal"))
        vertical_slots = set(_legend_slots(r0, c0, "vertical"))
        horizontal_ok = singletons_set.issubset(horizontal_slots)
        vertical_ok = singletons_set.issubset(vertical_slots)

        if horizontal_ok and not vertical_ok:
            hc = len(singletons)
        elif vertical_ok and not horizontal_ok:
            hc = len(singletons)
        elif not singletons:
            hc = 0
        else:
            raise AssertionError(f"color {c} has stray dots outside legend slots")

        assert hc not in mapping, f"legend hole count {hc} must map to exactly one color"
        mapping[hc] = c
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by matching their hole count to a dot-coded color legend.

    Input structure:
    - bg = most common color; grey (5) sprites are scattered through the workspace.
    - Near an edge or corner, each legend entry is a solid 3x3 colored block with
      zero or more isolated same-color dots placed beside it inside the legend panel.

    Rule:
    - For each legend color, count the nearby marker dots; that count defines the
      hole-count target for that color.
    - For every grey sprite, count its enclosed holes inside its own bounding box.
    - Recolor the grey sprite if its hole count appears in the legend; erase grey
      distractors whose hole count is not listed.

    Key insight: The sprites are still classified by topology, but the legend no
    longer encodes topology visually inside its own shape. The legend uses an
    explicit symbol count, so the solver must count dots around each color block.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

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

from arc_utils import label_mask
from scipy.ndimage import binary_dilation, generate_binary_structure

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)         # number of legend entries
MAX_HOLE_COUNT = 6                  # hole values drawn from range(MAX_HOLE_COUNT)
PANEL_SIZE_RANGE = (6, 9)           # panel thickness (cross-axis)
FULL_LAYOUT_MAX_SPAN = 28          # max legend line span for full-row/col layouts
L_LAYOUT_MAX_SPAN = 24            # max legend line span for L-shaped layouts
L_LAYOUT_EXTRA_PADDING = 3        # max random padding along the L arm

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)       # total sprites (matches + distractors)
MAX_DISTRACTORS = 7                 # max distractor sprites
MAX_EXTRA_PER_LEGEND = 2            # max extra match sprites per legend entry
SPRITE_RETRIES = 30                 # retries per sprite generation attempt
SPRITE_AREA_0 = (5, 25)              # (min, max) area for 0-hole sprites
SPRITE_AREA_LOW_MIN = 8              # min area for 1-2 hole sprites (max = 25+10*h)
# 3+ hole sprites: min = 10+2*h, max = 20+4*h
CANVAS_SCALE = 1.2                  # sqrt(area) multiplier for canvas size
CANVAS_MIN = 5                      # minimum canvas dimension
CANVAS_PAD = 2                      # padding added to scaled sqrt for canvas
HOLE_SEED_RADIUS_FACTOR = 0.45      # radius = factor * sqrt(area)
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2               # minimum Chebyshev distance between hole seeds
HOLE_SEED_MARGIN = 2                # margin from canvas edge for hole seeds
SEAL_ITERS = 5                      # max iterations for sealing forbidden cells
SPRITE_MAX_H = 12                   # max sprite bbox height
SPRITE_MAX_W = 18                   # max sprite bbox width
SPRITE_MAX_ASPECT = 2.5             # max aspect ratio (max_dim / min_dim)

# -- generation --
GEN_RETRIES = 300                   # outer retries for generate_input()

# 8-connectivity structuring element for dilation
_STRUCT8 = generate_binary_structure(2, 2)


# ---------- legend helpers ----------

def _legend_span(hole_count: int) -> int:
    """Width (or height) of a single legend patch along the arrangement axis."""
    if hole_count == 0:
        return LEGEND_BLOCK
    if hole_count <= 3:
        return 5
    return 7


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a dot-coded legend patch around a solid 3x3 color block."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((LEGEND_THICKNESS, span), BG, dtype=int)
        patch[:LEGEND_BLOCK, :LEGEND_BLOCK] = int(color)
        for rr, cc in _HORIZONTAL_DOT_SLOTS[:hole_count]:
            if cc < span:
                patch[rr, cc] = int(color)
    else:
        patch = np.full((span, LEGEND_THICKNESS), BG, dtype=int)
        patch[:LEGEND_BLOCK, :LEGEND_BLOCK] = int(color)
        for rr, cc in _VERTICAL_DOT_SLOTS[:hole_count]:
            if rr < span:
                patch[rr, cc] = int(color)
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 2-4 legends with distinct hole counts that fit within max_span."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [holes for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
              if sum(_legend_span(h) for h in holes) + count - 1 <= max_span]
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
        r0 = random.randint(1, panel_h - LEGEND_THICKNESS)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 1, GRID_SIZE - LEGEND_THICKNESS - 1)
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
        c0 = random.randint(1, panel_w - LEGEND_THICKNESS)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 1, GRID_SIZE - LEGEND_THICKNESS - 1)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(corner: str, panel_h: int, panel_w: int,
                       arrangement: str, total: int) -> tuple:
    """Compute divider pivot and legend placement bounds for an L-corner.

    Returns (dr, dc, r_min, r_max, c_min, c_max) where (dr, dc) is the corner
    pivot of the L-divider and the remaining values bound the legend origin.
    """
    N = GRID_SIZE
    cross_dim = LEGEND_THICKNESS if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else LEGEND_THICKNESS

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim
        c_min, c_max = 1, panel_w - along_dim
    elif corner == "tr":
        dr, dc = panel_h, N - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim
        c_min, c_max = dc + 1, N - 1 - along_dim
    elif corner == "bl":
        dr, dc = N - 1 - panel_h, panel_w
        r_min, r_max = dr + 1, N - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim
    else:  # br
        dr, dc = N - 1 - panel_h, N - 1 - panel_w
        r_min, r_max = dr + 1, N - 1 - cross_dim
        c_min, c_max = dc + 1, N - 1 - along_dim

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
        corner, panel_h, panel_w, arrangement, total)

    # Draw divider arms and mark blocked regions per corner
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
    elif hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    else:
        return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
    """Find well-separated seed positions for holes inside the canvas.

    Returns a list of seed coordinates, or None if insufficient seeds found.
    """
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
    return center, center  # fallback (shouldn't happen)


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

def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Build a shuffled list of (hole_count, color_or_None) for all sprites to generate."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [d for d in range(1, max_d + 1)
               if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color))
    legend_holes = {h for h, _ in legends}
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
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all sprites (matches + distractors) onto the grid."""
    items = []
    for hole_count, color in _sprite_plan(legends):
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
    raise RuntimeError("failed to generate e3721c99 input")
