# Combo variant of e3721c99 (index 4)
# combo_spec: D3/2+D5/0
# Name: 纯色块图例+点阵编码洞数
#
# Design analysis
# a) Concrete form and behavior trace
#    - Original task: each legend object was itself a colored shape with holes, so
#      the solver could count holes on the legend object and reuse the same
#      topological measurement on each grey sprite.
#    - Combo task: each legend entry is split into two cooperating parts:
#        * a solid 2x2 color swatch that says "this is the output color",
#        * a nearby fixed-color dot matrix whose population says "this color stands
#          for sprites with N holes".
#      The legend therefore stops being a topological object and becomes an
#      explicit symbolic record.
#    - Example trace:
#        old: red legend shape has 3 holes -> grey sprite with 3 holes becomes red.
#        new: red 2x2 swatch sits next to exactly 3 marker dots -> grey sprite with
#             3 holes becomes red.
#      The sprite side of the task still depends on hole counting, but the legend
#      side now requires decoding a swatch-plus-dots entry instead of counting
#      holes on the legend itself.
# b) Complexity calibration
#    - Hole values stay in 0..5, matching the original sprite family. This keeps
#      the symbolic alphabet small and repeatedly supported across 2-4 legend
#      entries and 8-11 total sprites.
#    - The dot code uses five fixed slots only. That is enough to make the new
#      visual system obvious while avoiding overfitting to arbitrary patterns.
#    - Color and number are separated into different objects: swatch carries color,
#      marker dots carry count. This makes the D3 and D5 changes visibly stronger
#      than a same-color block-with-dots variant.
# c) Implicit assumption audit for `main()`
#    - The original `_legend_mapping` assumed one non-border multi-cell colored
#      component per legend color and assumed that component's holes encoded the
#      mapping. That is no longer true because the swatch is hole-free.
#    - The new parser assumes:
#        * legend colors appear only as one solid 2x2 swatch each,
#        * marker dots use one dedicated code color, not the swatch color,
#        * all entries in a sample share one arrangement direction
#          (horizontal strip entries or vertical strip entries),
#        * every code pixel lies in one of the fixed slots relative to some swatch.
#      The generator enforces all of these constraints.
#    - Grey-sprite processing still assumes 4-connected components and tight-bbox
#      hole counting. That part remains compatible with the original task.
# d) Placement legality
#    - Divider layouts remain full-row, full-column, or L-shaped, so the legend
#      panel stays separated from the workspace.
#    - Swatches and code dots are placed with at least one background cell between
#      adjacent legend entries, and sprite placement still uses an 8-neighborhood
#      padding check so workspace objects do not touch each other or the panel.
# e) Visual observability
#    - Across a few examples, a human can see that legend objects are now plain
#      squares and that the only varying cue beside them is the number of fixed
#      marker dots. Matching recolors consistently follow "same hole count as the
#      dot count next to this swatch".
# f) Visual cleanliness
#    - The code color is reserved for markers only, so the legend decomposes into
#      two easily distinguishable layers instead of one cluttered colored cluster.
#    - The dot slots are sparse and fixed, which prevents crowded panel geometry
#      even for values 4-5.
# g) Change inferability
#    - Output colors are directly visible in the swatches, while the hole-count
#      targets are directly visible in the marker population. No hidden ordering by
#      color id, component size, or scan order is used.

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _hole_count(mask: np.ndarray) -> int:
    """Count holes in a binary mask using 4-connectivity."""
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


# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
CODE = 8
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
LEGEND_SWATCH = 2
LEGEND_THICKNESS = 5
PANEL_SIZE_RANGE = (6, 9)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3
_HORIZONTAL_DOT_SLOTS = [(0, 3), (2, 3), (4, 3), (0, 5), (2, 5)]
_VERTICAL_DOT_SLOTS = [(3, 0), (3, 2), (3, 4), (5, 0), (5, 2)]

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

# -- generation --
GEN_RETRIES = 300
_STRUCT8 = generate_binary_structure(2, 2)


def _touches_outer_border(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> bool:
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _slot_positions(r0: int, c0: int, arrangement: str) -> list[tuple[int, int]]:
    slots = _HORIZONTAL_DOT_SLOTS if arrangement == "horizontal" else _VERTICAL_DOT_SLOTS
    return [(r0 + dr, c0 + dc) for dr, dc in slots]


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Decode a legend made of solid color swatches plus separate dot-code markers.

    Each legend color appears as exactly one 2x2 swatch. The matching hole count is
    the number of CODE pixels placed in fixed slots either to the right (horizontal
    panel) or below (vertical panel) of that swatch.
    """
    h, w = g.shape
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey, DIV, CODE)]
    swatches: list[tuple[int, int, int]] = []
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if _touches_outer_border(r0, c0, r1, c1, h, w):
                continue
            if (r1 - r0 + 1, c1 - c0 + 1) != (LEGEND_SWATCH, LEGEND_SWATCH):
                continue
            if len(pts) != LEGEND_SWATCH * LEGEND_SWATCH:
                continue
            candidates.append((r0, c0))
        assert len(candidates) == 1, f"color {color} must have exactly one legend swatch"
        swatches.append((candidates[0][0], candidates[0][1], int(color)))

    assert 2 <= len(swatches) <= 4, "legend must contain 2-4 swatches"

    horizontal_evidence = False
    vertical_evidence = False
    for r0, c0, _ in swatches:
        hcount = sum(g[r, c] == CODE for r, c in _slot_positions(r0, c0, "horizontal"))
        vcount = sum(g[r, c] == CODE for r, c in _slot_positions(r0, c0, "vertical"))
        if hcount and vcount:
            raise AssertionError("legend swatch cannot support both dot directions")
        horizontal_evidence |= hcount > 0
        vertical_evidence |= vcount > 0

    assert horizontal_evidence ^ vertical_evidence, "legend direction must be inferable from nonzero entries"
    arrangement = "horizontal" if horizontal_evidence else "vertical"

    mapping: dict[int, int] = {}
    expected_code: set[tuple[int, int]] = set()
    for r0, c0, color in swatches:
        slots = _slot_positions(r0, c0, arrangement)
        assert all(0 <= r < h and 0 <= c < w for r, c in slots), "legend slots must stay in bounds"
        hole_count = sum(g[r, c] == CODE for r, c in slots)
        assert hole_count not in mapping, f"hole count {hole_count} must map to exactly one color"
        mapping[int(hole_count)] = int(color)
        expected_code.update(slots[:hole_count])

    actual_code = {tuple(map(int, pt)) for pt in np.argwhere(g == CODE)}
    assert actual_code == expected_code, "all code pixels must belong to legend dot slots"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by decoding a swatch-and-dots legend and matching hole counts.

    Input structure:
    - The dominant color is the background, and grey (5) sprites are scattered in the
      workspace as 4-connected amorphous shapes with 0-5 enclosed holes.
    - The legend sits inside a divider-separated panel. Each legend entry is a solid
      2x2 color swatch, and nearby fixed-color marker dots encode the hole count for
      that swatch's color.

    Rule:
    - Read each swatch together with its adjacent dot matrix to build a mapping
      `hole_count -> output_color`.
    - Count holes inside every grey sprite's tight bounding box.
    - Recolor sprites whose hole count appears in the legend, and erase grey
      distractors whose hole count is absent.

    Key insight:
    - Color and number are no longer bundled inside one holey legend shape. The
      swatch tells you which color to paint, while the separate marker dots tell you
      which sprite topology that color applies to.
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


def _legend_span(hole_count: int) -> int:
    """Length of one legend patch along the panel's main axis."""
    if hole_count == 0:
        return LEGEND_SWATCH
    if hole_count <= 3:
        return 4
    return 6


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Render one legend entry as a color swatch plus fixed-color dot markers."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((LEGEND_THICKNESS, span), BG, dtype=int)
        patch[:LEGEND_SWATCH, :LEGEND_SWATCH] = int(color)
        for rr, cc in _HORIZONTAL_DOT_SLOTS[:hole_count]:
            if cc < span:
                patch[rr, cc] = CODE
    else:
        patch = np.full((span, LEGEND_THICKNESS), BG, dtype=int)
        patch[:LEGEND_SWATCH, :LEGEND_SWATCH] = int(color)
        for rr, cc in _VERTICAL_DOT_SLOTS[:hole_count]:
            if rr < span:
                patch[rr, cc] = CODE
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total line length of all legend patches plus one-cell gaps."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick distinct hole-count/color pairs that fit in the chosen panel span."""
    del arrangement
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends("horizontal", max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along a horizontal or vertical strip."""
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row divider with a horizontal legend strip above or below it."""
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
    """Place a full-column divider with a vertical legend strip left or right."""
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


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legal legend origin bounds for one L-corner."""
    n = GRID_SIZE
    cross_dim = LEGEND_THICKNESS if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else LEGEND_THICKNESS

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim
        c_min, c_max = 1, panel_w - along_dim
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim
        c_min, c_max = dc + 1, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 1, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 1, n - 1 - cross_dim
        c_min, c_max = dc + 1, n - 1 - along_dim

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place an L-shaped divider with the legend panel tucked into one corner."""
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
        blocked[:dr, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Randomly choose one divider layout and place the legend inside its panel."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    """Sample a sprite area with size loosely increasing with hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Find separated seeds for interior holes."""
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
    """Grow irregular forbidden blobs that will become enclosed holes."""
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
    """Pick the nearest non-forbidden growth start to the canvas center."""
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
    """Grow one connected blob of approximately the requested area."""
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
    """Seal around intended holes, then keep only the largest connected component."""
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
    """Crop the sprite, then validate size, aspect ratio, and final hole count."""
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
    """Generate one amorphous grey sprite with the requested number of holes."""
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


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Plan matched and distractor sprites, guaranteeing every legend is represented."""
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
    """Dilate a boolean occupancy mask by one 8-neighborhood step."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a legal placement that avoids the panel, divider, and other sprites."""
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
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all grey sprites, then color matched ones in `expected`."""
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


def _gen_attempt() -> np.ndarray | None:
    """Build one input grid and validate it against the solver."""
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
    """Retry generation until a valid combo-variant input is found."""
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
