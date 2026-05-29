# Combo variant of e3721c99 (index 30)
# combo_spec: D6/1+D11/0
# Name: 原位洞数标记+两阶段着色

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis
# ---------------
# a) Concrete behavior:
#    - The base task recolors each matching grey sprite directly according to
#      the legend's hole-count mapping.
#    - This combo keeps the same legend lookup, but matched sprites are handled
#      in two explicit passes: first they are classified as "confirmed"
#      matches, then they are repainted to the legend color and decorated with
#      as many outside witness dots as the sprite has holes.
#    - Example: if a grey sprite has 3 holes and the 3-hole legend is green,
#      the old output was just a green sprite; the combo output is a green
#      sprite plus 3 grey dots on the 1-cell ring around its bbox, read
#      clockwise from the top-left corner.
#
# b) Complexity calibration:
#    - The only new visual degree of freedom is the count of outside dots.
#      Their color is fixed (grey) and their placement rule is fixed
#      (clockwise outer-ring prefix), so the transformation stays structured.
#    - Every grid still contains multiple legend entries and at least one
#      matching sprite per legend entry, so the new marker behavior repeats
#      several times in one example instead of appearing as a singleton quirk.
#
# c) Assumption review:
#    - The old solver assumed output changes were confined to the sprite body.
#      That no longer holds, so the new solver must also write into adjacent bg.
#    - The old one-pass recolor is no longer a faithful model.  The code now
#      separates a confirmation pass from a rendering pass, mirroring the
#      two-stage description even though the intermediate state is not shown.
#    - The old generator only needed sprite-body clearance.  The combo also
#      needs legal witness-dot cells, so matched sprites reserve their actual
#      future dot positions during placement.
#    - Legend extraction remains valid because witness dots never appear in the
#      input that `main()` sees, only in the output it produces.
#
# d) Placement legality:
#    - All sprite bodies remain at least one cell away from other reserved
#      content by using an 8-neighborhood dilation mask.
#    - For matched sprites, the exact future witness-dot cells must also be bg,
#      in-bounds, and unused before placement is accepted.
#
# e) Visual observability and neatness:
#    - The added dots sit outside the sprite instead of inside its holes, so the
#      topological feature that drives the rule remains visible.
#    - Grey witness dots around recolored sprites make the combo visibly
#      different from the base task while keeping body color, legend color, and
#      count evidence easy to separate by eye.


BG = 0
DIV = 1
GREY = 5
TEMP_CONFIRM = 10

GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

TOTAL_SPRITES_RANGE = (7, 9)
MAX_DISTRACTORS = 5
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

GEN_RETRIES = 600

_STRUCT8 = generate_binary_structure(2, 2)


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed 4-connected holes inside a binary sprite mask."""
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


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """Map legend hole-count -> legend color using non-border colored components."""
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
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == c
        hc = _hole_count(sub)
        assert hc not in mapping, f"legend hole count {hc} must map to exactly one color"
        mapping[hc] = c
    return mapping


def _outer_ring_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Clockwise order on the 1-cell outer ring around an inclusive bbox."""
    pos: list[tuple[int, int]] = []
    for c in range(c0 - 1, c1 + 2):
        pos.append((r0 - 1, c))
    for r in range(r0, r1 + 2):
        pos.append((r, c1 + 1))
    for c in range(c1, c0 - 2, -1):
        pos.append((r1 + 1, c))
    for r in range(r1, r0 - 1, -1):
        pos.append((r, c0 - 1))
    return pos


def _marker_positions_for_bbox(
    bbox: tuple[int, int, int, int], hole_count: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    """Return the first `hole_count` in-bounds cells on the bbox outer ring."""
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    ring = [
        (r, c)
        for r, c in _outer_ring_positions(r0, c0, r1, c1)
        if 0 <= r < h and 0 <= c < w
    ]
    assert len(ring) >= hole_count, "generator must keep enough ring cells in bounds"
    return ring[:hole_count]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor matched grey sprites via hole-count legends and show count evidence.

    Input structure:
    - Background is the most common color.
    - Grey sprites (5) are scattered in the workspace.
    - Colored legend patches near the divider encode hole-count -> final color.

    Rule:
    - Read the legend exactly as in the base task.
    - Count holes inside each grey sprite's tight bbox.
    - If the hole count appears in the legend, first treat the sprite as a
      confirmed match, then recolor it to the legend color and add that many
      grey witness dots on the sprite's outside ring.
    - Grey sprites whose hole counts are absent from the legend are erased.

    Key insight:
    - Topology still decides which color a sprite deserves, but the final output
      also visualizes the counted topology itself: outside witness dots encode
      the same hole count that drove the hidden confirmation stage.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend = _legend_mapping(g, bg, GREY)
    assert legend, "grid must contain at least one legend entry"

    stage = g.copy()
    out = g.copy()
    matched: list[tuple[list[tuple[int, int]], tuple[int, int, int, int], int, int]] = []

    for pts_arr in components_mask(g == GREY, connectivity=4):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hole_count = _hole_count(sub)
        if hole_count in legend:
            for r, c in pts:
                stage[r, c] = TEMP_CONFIRM
            matched.append((pts, (r0, c0, r1, c1), hole_count, int(legend[hole_count])))
        else:
            for r, c in pts:
                out[r, c] = bg

    for pts, bbox, hole_count, color in matched:
        for r, c in pts:
            assert stage[r, c] == TEMP_CONFIRM
            out[r, c] = color
        for mr, mc in _marker_positions_for_bbox(bbox, hole_count, g.shape):
            assert out[mr, mc] == bg, "generator must reserve witness-dot cells"
            out[mr, mc] = GREY

    return out


def _legend_span(hole_count: int) -> int:
    """Width or height of one legend patch along its layout axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a horizontal or vertical legend patch with the correct hole pattern."""
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
    """Total legend span including 1-cell gaps between patches."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 2-4 legend entries with distinct hole counts that fit the panel."""
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
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row divider with a horizontal legend strip."""
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
    """Place a full-column divider with a vertical legend strip."""
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
    """Return divider pivot and allowed legend-origin bounds for one L-corner."""
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
    """Place an L-shaped divider with a legend panel in one corner."""
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    panel_h, panel_w = (cross, along) if arrangement == "horizontal" else (along, cross)
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
    """Randomly choose one divider/legend layout."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    """Sample a sprite area from the original hole-count-dependent regime."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Pick well-separated seeds for interior hole blobs."""
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
    positions = [
        (r, c)
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
        if margin <= r < csz - margin and margin <= c < csz - margin
    ]
    random.shuffle(positions)
    seeds: list[tuple[int, int]] = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(
    seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int
) -> set[tuple[int, int]]:
    """Grow irregular forbidden blobs that later become sprite holes."""
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
    """Grow a connected blob of the target area while avoiding forbidden cells."""
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
    """Seal around hole seeds and keep only the largest connected component."""
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
    """Crop to a tight bbox, then validate bbox size, aspect, and hole count."""
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
    """Generate one random connected grey sprite with the requested hole count."""
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
    """Build a shuffled list of sprites as (hole_count, matching_color_or_None)."""
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

    plan: list[tuple[int, int | None]] = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color))
    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None))
    random.shuffle(plan)
    return plan


def _matched_output_patch(sprite: np.ndarray, color: int, hole_count: int) -> np.ndarray:
    """Build the output patch for one matched sprite plus its witness dots."""
    h, w = sprite.shape
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[1 : h + 1, 1 : w + 1] = np.where(sprite == GREY, int(color), BG)
    for r, c in _outer_ring_positions(1, 1, h, w)[:hole_count]:
        patch[r, c] = GREY
    return patch


def _random_sprite_location(
    static_blocked: np.ndarray,
    reserved: np.ndarray,
    sprite: np.ndarray,
    hole_count: int,
    is_match: bool,
) -> tuple[int, int] | None:
    """Find a legal body origin for one sprite."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    candidates = [(r, c) for r in range(1, GRID_SIZE - h) for c in range(1, GRID_SIZE - w)]
    random.shuffle(candidates)
    reserved_pad = binary_dilation(reserved, structure=_STRUCT8).astype(bool)
    for r, c in candidates:
        if np.any(static_blocked[r - 1 : r + h + 1, c - 1 : c + w + 1] != BG):
            continue
        if np.any(reserved_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        if is_match:
            bbox = (r, c, r + h - 1, c + w - 1)
            marker_ok = True
            for mr, mc in _marker_positions_for_bbox(bbox, hole_count, static_blocked.shape):
                if static_blocked[mr, mc] != BG or reserved[mr, mc]:
                    marker_ok = False
                    break
            if not marker_ok:
                continue
        return r, c
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate all sprites, place them, and write the expected combo output."""
    items = []
    for hole_count, color in _sprite_plan(legends):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((int(np.sum(sprite == GREY)), sprite, hole_count, color))
    items.sort(key=lambda x: x[0], reverse=True)

    reserved = np.zeros_like(grid, dtype=bool)
    for _, sprite, hole_count, color in items:
        loc = _random_sprite_location(blocked, reserved, sprite, hole_count, color is not None)
        if loc is None:
            return False
        body_r, body_c = loc
        paste(grid, sprite, body_r, body_c, transparent=BG)
        reserved[body_r : body_r + sprite.shape[0], body_c : body_c + sprite.shape[1]] |= sprite == GREY

        if color is not None:
            bbox = (body_r, body_c, body_r + sprite.shape[0] - 1, body_c + sprite.shape[1] - 1)
            for mr, mc in _marker_positions_for_bbox(bbox, hole_count, grid.shape):
                reserved[mr, mc] = True
            out_patch = _matched_output_patch(sprite, int(color), hole_count)
            paste(expected, out_patch, body_r - 1, body_c - 1, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Build one input grid and validate it against the combo solver."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if int(np.sum(expected == GREY)) == 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Retry until a valid combo instance is produced."""
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo variant input")
