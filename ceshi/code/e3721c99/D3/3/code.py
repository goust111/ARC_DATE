# Variant of e3721c99
# Dimension: D3 — 目标结构：图例的设计与位置 (dst)
# Change: 彩色图例形状位于网格左上角，每个图例形状有特定颜色和洞数... → 图例颜色与位置编码洞数

"""
design_analysis:
  variant_summary:
    original_task: Each colored legend object advertised its hole count directly through its own geometry.
    new_task: Every legend object is the same 2x2 solid block; only the order of those blocks inside a
      top-left legend strip encodes the hole count. The first block means 0 holes, the second means 1,
      and so on.

  concretization_and_behavior_trace:
    input_look:
      - The legend is always anchored near the top-left corner.
      - It is either a horizontal row of same-shaped colored blocks above a full-width divider,
        or a vertical column of same-shaped colored blocks left of a full-height divider.
      - Grey sprites remain scattered in the workspace and still differ by topology only.
    output_look:
      - Grey sprites with hole count k are recolored by the k-th legend block color.
      - Grey sprites whose hole count is not represented by a legend position are erased.
    traced_example:
      - Old logic: a red 3x5 legend with one cavity means "1 hole -> red" because the shape itself has 1 hole.
      - New logic: three identical blocks [red, teal, yellow] mean
        "0 holes -> red, 1 hole -> teal, 2 holes -> yellow" because their positions are 0, 1, 2.
      - Difference: the solver can no longer inspect legend topology. It must read an ordered sequence.

  complexity_calibration:
    - Legend count stays in 2..4 so the sequence is long enough to show ordering but short enough to read.
    - Matched sprites still repeat per legend color, so each index->hole mapping gets visual support.
    - Distractors still exist and use larger hole counts outside the represented prefix, preventing a trivial
      "recolor all grey objects" interpretation.

  implicit_assumption_audit_for_main:
    - Old `_legend_mapping` assumed each legend color owned one non-border component whose hole count was unique.
      This fails immediately here because all legend blocks have 0 holes.
    - The new solver instead assumes:
      1. divider color is fixed and not reused by legends;
      2. each legend color appears in exactly one small component;
      3. legend components form a single horizontal or vertical ordered strip.
    - The generator is responsible for enforcing those assumptions.

  placement_legality:
    - The legend panel is reserved by a divider exactly as before, so sprites never overlap the panel.
    - Horizontal layout blocks rows above the divider; vertical layout blocks columns left of the divider.
    - Sprite placement still uses occupancy dilation, so sprites do not touch each other or the divider.

  visual_observability:
    - All legend items share the same 2x2 geometry, making the D3 change visible at a glance.
    - The only varying property inside the legend is color and position order, so order must be the cue.
    - Using both horizontal and vertical strips keeps "left-to-right or top-to-bottom" explicit without
      introducing extra ambiguous layouts.

  cleanliness_and_color_inference:
    - The legend is isolated in one corner and never mixed with workspace sprites.
    - Divider color is excluded from legend colors, avoiding accidental confusion between separator and key.
    - The color mapping remains sparse (2..4 legend colors per sample), which is still easy to infer.
"""

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


def _ordered_legend_colors(g: np.ndarray, bg: int, grey: int, divider: int) -> dict[int, int]:
    """Read identical legend blocks as an ordered sequence: index -> color."""
    blocks: list[tuple[int, int, int]] = []
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey, divider)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert len(comps) == 1, f"legend color {color} must appear in exactly one component"
        r0, c0, r1, c1 = bbox_coords(comps[0])
        blocks.append((r0, c0, color))

    assert len(blocks) >= 2, "grid must contain an ordered legend strip with at least two entries"
    span_r = max(r for r, _, _ in blocks) - min(r for r, _, _ in blocks)
    span_c = max(c for _, c, _ in blocks) - min(c for _, c, _ in blocks)
    if span_c >= span_r:
        blocks.sort(key=lambda x: (x[1], x[0]))
    else:
        blocks.sort(key=lambda x: (x[0], x[1]))
    return {idx: color for idx, (_, _, color) in enumerate(blocks)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by matching their hole count to an ordered legend strip.

    Input structure:
    - Background is the dominant color; grey (5) sprites are scattered in the workspace.
    - The legend lives in the top-left panel, separated by a blue divider line, and consists of
      identical solid color blocks laid out either left-to-right or top-to-bottom.

    Rule:
    - Read the legend blocks in sequence order; the first block stands for 0 holes, the second for 1,
      and so on.
    - For each grey sprite, count the number of enclosed holes inside its bounding box.
    - Recolor the sprite with the color at the matching legend position; if no such position exists,
      erase the sprite to background.

    Key insight:
    - The legend no longer describes hole counts with shape geometry; the count is encoded only by block order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5
    divider = 1

    legend = _ordered_legend_colors(g, bg, grey, divider)

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


# --------------- generator (ordered legend strip + holes-first sprites) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
LEGEND_BLOCK = 2
LEGEND_GAP = 1
LEGEND_START = 1
DIVIDER_RANGE = (4, 6)

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
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

_STRUCT8 = generate_binary_structure(2, 2)


def _legend_entries() -> list[tuple[int, int]]:
    """Choose a prefix of hole counts so index k always means hole count k."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    colors = random.sample(LEGEND_COLORS, count)
    return [(hole_count, color) for hole_count, color in enumerate(colors)]


def _legend_patch(color: int) -> np.ndarray:
    """All legend entries share the same appearance."""
    return np.full((LEGEND_BLOCK, LEGEND_BLOCK), int(color), dtype=int)


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place the ordered legend strip in the top-left panel."""
    legends = _legend_entries()
    if random.choice(["horizontal", "vertical"]) == "horizontal":
        div_r = random.randint(*DIVIDER_RANGE)
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = LEGEND_START
        c0 = LEGEND_START
        for _, color in legends:
            patch = _legend_patch(color)
            paste(grid, patch, r0, c0)
            c0 += LEGEND_BLOCK + LEGEND_GAP
    else:
        div_c = random.randint(*DIVIDER_RANGE)
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        r0 = LEGEND_START
        c0 = LEGEND_START
        for _, color in legends:
            patch = _legend_patch(color)
            paste(grid, patch, r0, c0)
            r0 += LEGEND_BLOCK + LEGEND_GAP
    return legends


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


def _grow_mask(csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]) -> np.ndarray:
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
            best = max(range(1, n_comp + 1), key=lambda label: int(np.sum(labeled == label)))
            mask = labeled == best
    return mask


def _crop_and_validate(mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int) -> np.ndarray | None:
    """Crop mask to a tight bbox, then validate aspect ratio and hole count."""
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
    """Build a shuffled list of (hole_count, color_or_None) for all sprites."""
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
        choices = [i for i, value in enumerate(extras) if value < MAX_EXTRA_PER_LEGEND]
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
    """8-connectivity dilation of a boolean mask."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement for sprite, avoiding blocked and occupied areas."""
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
    raise RuntimeError("failed to generate e3721c99 D3 input")
