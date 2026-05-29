# Variant of e3721c99
# Dimension: D4 — 变换：重新着色的具体规则 (transform)
# Change: s.color = d.color — 灰色精灵被直接重新着色为匹配图例的颜色... → 渐变着色（按洞数递增色阶）
#
# Design analysis for the D4 variant:
# a) Concrete behavior and trace:
#    - The divider, legend panel, grey sprites, hole-count matching, and distractor erasure stay intact.
#    - The only semantic change is what color a matched grey sprite receives after its hole count is read.
#    - Input still shows 2-4 legend patches near an edge/corner. Each legend patch is identified by its hole
#      topology, not by its display color. Grey sprites remain scattered in the workspace.
#    - Small trace:
#      input legend holes = {1, 2, 3}, legend colors = {red, yellow, cyan}; one grey sprite has 2 holes.
#      Old rule: 2-hole sprite becomes the color of the 2-hole legend, so it would become yellow.
#      New rule: 2-hole sprite ignores legend color and becomes the fixed gradient color for hole count 2,
#      namely green. A 4-hole grey sprite is still erased if 4 is not listed by the legends.
#    - Therefore the workflow split becomes:
#      legends declare which hole counts are valid; the gradient function decides the output color.
#
# b) Complexity calibration:
#    - To make the new rule visible, legend hole counts are sampled as a consecutive run of length 2-4.
#      This yields several nearby hole values per instance, so the output shows an actual color ramp rather
#      than isolated unrelated recolorings.
#    - The gradient itself uses only six fixed colors for hole counts 0..5. Each grid uses only 2-4 of them,
#      and every listed hole count appears on at least one sprite, usually several, so the mapping has repeated
#      support and is learnable from a few examples.
#
# c) Implicit-assumption audit against the original main():
#    - Original `_legend_mapping()` assumed legend color was the payload, so it grouped by color and returned
#      `hole_count -> legend_color`. That assumption is no longer valid.
#    - The new solver must not read legend colors at all for recoloring, only for locating legend components.
#    - The old line `newc = int(legend[hc]) if hc in legend else bg` would silently revert the task back to the
#      original behavior if kept. It is replaced with `newc = gradient(hc)` for listed holes.
#    - Because legend colors are now decoys, the generator must avoid accidentally choosing a legend color equal
#      to the gradient color for the same hole count; otherwise some samples would look like the original task.
#
# d) Placement legality:
#    - The original blocked-panel logic, sprite dilation gap, and bbox limits remain valid and unchanged.
#    - Legends are still inset from the divider and border, so the solver can isolate them by component bbox.
#    - Grey sprites are still generated in the workspace only, non-overlapping, with at least one background
#      cell of clearance, so hole counting on each sprite bbox stays well-defined.
#
# e) Visual observability:
#    - A human can still count holes on legends and sprites exactly as before.
#    - The visible difference is that output sprite colors form a fixed cool-to-warm ramp by hole count even
#      when legend colors vary arbitrarily between examples.
#    - Consecutive hole counts inside one grid make the ramp easier to notice, while shuffled legend order keeps
#      position from becoming a shortcut.
#
# f) Visual cleanliness:
#    - No extra overlays or markers are added. The scene still has just divider, legend panel, and sprites.
#    - Legend colors are distinct and the gradient colors are distinct, so workspace recolorings are easy to
#      separate from the legend panel by both region and appearance.
#
# g) Color-rule inferability:
#    - The gradient is fixed and low-cardinality:
#        0 holes -> cyan, 1 -> blue, 2 -> green, 3 -> yellow, 4 -> orange, 5 -> red.
#    - Legend colors are chosen to avoid these same colors for the listed hole counts, making it clear from
#      examples that the legend color is not the answer key anymore.
#    - Consecutive legend hole values ensure the output often displays several adjacent gradient steps together.
#
# Review of new main() assumptions:
#    - Every non-border non-grey non-background component of size at least 2x2 is a legend patch.
#    - Legend hole counts are distinct.
#    - Any grey sprite whose hole count is listed by the legends should survive and take `gradient(hc)`.
#    - The generator explicitly constructs inputs to satisfy all three assumptions above.

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


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


def _legend_holes(g: np.ndarray, bg: int, grey: int) -> set[int]:
    """
    Find all legend components and return the set of declared hole counts.

    Legend components are the non-(bg,grey) connected components that:
    - are at least 2x2,
    - do not touch the outer grid border.
    """
    h, w = g.shape
    holes: set[int] = set()
    mask = (g != bg) & (g != grey)
    for pts in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
            continue
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        assert hc not in holes, f"legend hole count {hc} must appear exactly once"
        holes.add(hc)
    return holes


# --------------- fixed gradient palette ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
HOLE_COLOR_SCALE = (8, 1, 3, 4, 7, 2)  # 0->cyan, 1->blue, 2->green, 3->yellow, 4->orange, 5->red


def _gradient_color(hole_count: int) -> int:
    """Map hole count to the fixed output color ramp."""
    assert 0 <= hole_count < len(HOLE_COLOR_SCALE), f"unsupported hole count {hole_count}"
    return int(HOLE_COLOR_SCALE[hole_count])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by reading hole counts from the legends, then applying a fixed gradient.

    Input structure:
    - Background is the dominant color; grey (5) sprites are scattered in the workspace.
    - A legend panel near one edge/corner contains colored patches whose shapes encode distinct hole counts.
      Those patch colors are decorative here; only the hole counts they display matter.

    Rule:
    - Extract the set of hole counts shown by the legend patches.
    - For each grey sprite, count its enclosed holes inside its own bounding box.
    - If that hole count is listed by the legends, recolor the sprite with the fixed hole-count gradient;
      otherwise erase it to background.

    Key insight:
    - The legends still tell you which topologies are valid, but output color now comes from a global
      count-to-color scale rather than copying any legend color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY

    legend_holes = _legend_holes(g, bg, grey)
    assert legend_holes, "legend set must be non-empty: grid must contain legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        newc = _gradient_color(hc) if hc in legend_holes else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator (holes-first organic blob) ---------------

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)       # number of legend entries
MAX_HOLE_COUNT = len(HOLE_COLOR_SCALE)
PANEL_SIZE_RANGE = (5, 8)         # panel thickness (cross-axis)
FULL_LAYOUT_MAX_SPAN = 28         # max legend line span for full-row/col layouts
L_LAYOUT_MAX_SPAN = 24            # max legend line span for L-shaped layouts
L_LAYOUT_EXTRA_PADDING = 3        # max random padding along the L arm

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)     # total sprites (matches + distractors)
MAX_DISTRACTORS = 7               # max distractor sprites
MAX_EXTRA_PER_LEGEND = 2          # max extra match sprites per legend entry
SPRITE_RETRIES = 30               # retries per sprite generation attempt
SPRITE_AREA_0 = (5, 25)           # (min, max) area for 0-hole sprites
SPRITE_AREA_LOW_MIN = 8           # min area for 1-2 hole sprites (max = 25+10*h)
# 3+ hole sprites: min = 10+2*h, max = 20+4*h
CANVAS_SCALE = 1.2                # sqrt(area) multiplier for canvas size
CANVAS_MIN = 5                    # minimum canvas dimension
CANVAS_PAD = 2                    # padding added to scaled sqrt for canvas
HOLE_SEED_RADIUS_FACTOR = 0.45    # radius = factor * sqrt(area)
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2             # minimum Chebyshev distance between hole seeds
HOLE_SEED_MARGIN = 2              # margin from canvas edge for hole seeds
SEAL_ITERS = 5                    # max iterations for sealing forbidden cells
SPRITE_MAX_H = 12                 # max sprite bbox height
SPRITE_MAX_W = 18                 # max sprite bbox width
SPRITE_MAX_ASPECT = 2.5           # max aspect ratio (max_dim / min_dim)

# -- generation --
GEN_RETRIES = 300                 # outer retries for generate_input()

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
    """
    Pick 2-4 legends whose hole counts form a consecutive run.

    The consecutive run makes the fixed gradient visibly act like a ramp.
    Legend display colors are forced to differ from the gradient colors for the same holes.
    """
    del arrangement
    options = []
    for count in range(LEGEND_COUNT_RANGE[0], LEGEND_COUNT_RANGE[1] + 1):
        for start in range(MAX_HOLE_COUNT - count + 1):
            holes = tuple(range(start, start + count))
            if sum(_legend_span(h) for h in holes) + count - 1 <= max_span:
                options.append(holes)
    assert options, "legend run options must be non-empty"

    holes = list(random.choice(options))
    random.shuffle(holes)
    used_gradient = {_gradient_color(h) for h in holes}
    available_colors = [c for c in LEGEND_COLORS if c not in used_gradient]
    assert len(available_colors) >= len(holes), "need enough non-gradient legend colors"
    colors = random.sample(available_colors, len(holes))
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
    else:  # br
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
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
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
    """Grow irregular blob regions around each hole seed. Returns the full forbidden set."""
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
            best = max(range(1, n_comp + 1), key=lambda l: int(np.sum(labeled == l)))
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

def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, bool]]:
    """Build a shuffled list of (hole_count, is_match) for all sprites to generate."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, (hole_count, _) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, True))
    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), False))
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
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all sprites (matches + distractors) onto the grid."""
    items = []
    for hole_count, is_match in _sprite_plan(legends):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, hole_count, is_match))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, hole_count, is_match in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if is_match:
            out_patch = np.where(sprite == GREY, _gradient_color(hole_count), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


# ---------- top-level generation ----------

def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, validate via oracle."""
    n = GRID_SIZE
    grid = np.zeros((n, n), dtype=int)
    expected = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=int)

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
