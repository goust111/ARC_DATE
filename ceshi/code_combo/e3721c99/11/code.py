# Combo variant of e3721c99 (index 11)
# combo_spec: D3/3+D5/3
# Name: 图例位置隐含排序+颜色位置编码洞数

"""
design_analysis:
  variant_summary:
    original_task: Each legend color was attached to a distinct holed shape, so the solver could read
      hole-count -> color directly from legend topology.
    combo_task: The legend is reduced to a single vertical stack of identical solid color squares.
      The stack position is the only code: top block means 1 hole, next means 2 holes, and so on.
      There is no visible hole signal inside the legend itself.

  concretization_and_behavior_trace:
    input_look:
      - A divider still isolates one side or one corner panel from the workspace.
      - Inside that panel, 2-4 same-sized solid squares form one top-to-bottom column.
      - Grey sprites in the workspace remain organic connected blobs whose identity is still their hole count.
    traced_example:
      - Original behavior: if a red legend object visibly had 2 holes, every grey 2-hole sprite turned red.
      - Combo behavior: if the legend stack is [yellow, teal, red], then yellow means 1 hole,
        teal means 2 holes, red means 3 holes, purely because of vertical order.
      - A 0-hole grey blob is now a useful counterexample: it disappears because the stack starts at 1,
        not at 0, even though all legend squares themselves also have 0 holes.
    difference_from_original:
      - Legend topology becomes irrelevant and deliberately uninformative.
      - The solver must first decode ordinal position, then apply topology only to the grey sprites.

  complexity_calibration:
    - Legend count stays in 2..4, so the ordinal code is short and readable.
    - Every represented hole count 1..k appears on at least one sprite, often more than once.
    - Every sample also contains at least one 0-hole distractor, which makes the 1-based indexing visible
      instead of letting the task collapse into a 0-based ordered-strip variant.
    - Legend colors are limited to 2,3,4,6,7,8,9 so the divider color never doubles as a legend entry.

  implicit_assumption_audit_for_main:
    - The old legend reader assumed legend geometry carried hole information; that breaks completely because
      all legend blocks now have hole count 0.
    - The new legend reader assumes:
      1. all legend entries are solid rectangles of the same size;
      2. each legend color appears in exactly one component;
      3. legend entries form one strictly ordered vertical stack with shared column overlap;
      4. the top entry stands for 1 hole, not 0.
    - Grey-sprite processing still assumes each grey component can be isolated, cropped, and hole-counted
      independently; the generator preserves that by padded non-overlap placement.

  placement_legality:
    - The divider blocks off the whole legend panel before any sprite placement happens.
    - Full vertical and L-corner layouts both leave enough room for a 2-4 item vertical stack with
      1-cell margins and 1-cell gaps.
    - Workspace sprites are placed with 8-neighborhood padding so they do not touch each other,
      the divider, or the legend panel.

  visual_observability:
    - D3 is visible because all legend entries are literally the same square shape.
    - D5 is visible because only the top-to-bottom order distinguishes meanings.
    - The mandatory erased 0-hole distractor gives human solvers evidence that the first square means
      1 hole rather than 0 holes.

  visual_cleanliness_and_inference:
    - The legend is confined to a clean side or corner panel, separated from the workspace by the divider.
    - The legend column uses identical 3x3 blocks with a 1-cell gap, so it reads as a compact ordered list.
    - Because color changes are explained by one structural cue only (vertical position), 2-4 examples are
      enough to infer the mapping without relying on hidden color-id semantics.

  generator_guarantees_for_new_main:
    - The legend contains 2-4 solid 3x3 squares in one column.
    - Represented hole counts are exactly 1..k for that column length.
    - Each sample includes at least one 0-hole distractor and at least one matched sprite for every legend row.
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


def _legend_mapping(g: np.ndarray, bg: int, grey: int, divider: int) -> dict[int, int]:
    """
    Decode a single vertical stack of identical solid legend blocks.

    The top block means 1 hole, the next means 2 holes, etc.
    """
    h, w = g.shape
    blocks: list[tuple[int, int, int, int, int]] = []
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey, divider)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert len(comps) == 1, f"legend color {color} must appear in exactly one component"
        r0, c0, r1, c1 = bbox_coords(comps[0])
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "legend blocks must avoid outer border"
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        assert len(comps[0]) == hh * ww, "legend entries must be solid rectangles"
        blocks.append((r0, c0, r1, c1, color))

    assert 2 <= len(blocks) <= 4, "legend must contain 2-4 ordered blocks"
    block_sizes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1, _ in blocks}
    assert len(block_sizes) == 1, "all legend blocks must share the same size"

    blocks.sort(key=lambda item: (item[0], item[1]))
    for i in range(1, len(blocks)):
        assert blocks[i][0] > blocks[i - 1][2], "legend blocks must be vertically separated"

    overlap_left = max(c0 for _r0, c0, _r1, _c1, _color in blocks)
    overlap_right = min(c1 for _r0, _c0, _r1, c1, _color in blocks)
    assert overlap_left <= overlap_right, "legend blocks must form a single vertical column"

    return {i + 1: color for i, (_r0, _c0, _r1, _c1, color) in enumerate(blocks)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching their hole count to the vertical order of identical legend blocks.

    Input structure:
    - The background is dominant, grey (5) sprites are scattered in the workspace, and a divider line
      isolates a legend panel on one side or in one corner.
    - The legend panel contains 2-4 identical solid color squares stacked from top to bottom; these
      squares differ only by color and position, not by geometry or hole pattern.

    Rule:
    - Read the legend column from top to bottom: the first square stands for 1 hole, the second for 2,
      and so on.
    - For each grey sprite, count enclosed holes inside its own bounding box.
    - Recolor the sprite with the color of the legend square at that ordinal hole count; erase sprites
      whose hole count is not represented in the column.

    Key insight:
    - The legend no longer tells you anything through its own topology. Position alone encodes the number,
      while topology is used only on the grey workspace sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5
    divider = 1

    legend = _legend_mapping(g, bg, grey, divider)

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


# --------------- generator (vertical ordinal legend + organic holes-first sprites) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
LEGEND_BLOCK_SIZE = 3
LEGEND_GAP = 1
PANEL_SIZE_RANGE = (5, 8)
L_LAYOUT_EXTRA_PADDING = 3

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


def _legend_block(color: int) -> np.ndarray:
    """Build one identical solid legend square."""
    return np.full((LEGEND_BLOCK_SIZE, LEGEND_BLOCK_SIZE), int(color), dtype=int)


def _stack_height(legends: list[tuple[int, int]]) -> int:
    """Total height of the vertical legend stack."""
    return len(legends) * LEGEND_BLOCK_SIZE + (len(legends) - 1) * LEGEND_GAP


def _choose_legends() -> list[tuple[int, int]]:
    """Pick 2-4 colors; row index i implies hole count i+1."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(range(1, count + 1), colors))


def _place_legend_stack(grid: np.ndarray, legends: list[tuple[int, int]], r: int, c: int) -> None:
    """Paste a single top-to-bottom column of solid legend squares."""
    rr = int(r)
    cc = int(c)
    for _hole_count, color in legends:
        paste(grid, _legend_block(color), rr, cc)
        rr += LEGEND_BLOCK_SIZE + LEGEND_GAP


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-column divider with the legend stack on the left or right."""
    legends = _choose_legends()
    total_h = _stack_height(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - total_h - 1)

    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - LEGEND_BLOCK_SIZE - 1)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - LEGEND_BLOCK_SIZE - 1)

    _place_legend_stack(grid, legends, r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, total_h: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and valid legend-origin bounds for a corner panel."""
    n = GRID_SIZE
    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - total_h - 1
        c_min, c_max = 1, panel_w - LEGEND_BLOCK_SIZE - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - total_h - 1
        c_min, c_max = dc + 2, n - LEGEND_BLOCK_SIZE - 1
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - total_h - 1
        c_min, c_max = 1, panel_w - LEGEND_BLOCK_SIZE - 1
    else:  # br
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - total_h - 1
        c_min, c_max = dc + 2, n - LEGEND_BLOCK_SIZE - 1
    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place an L-shaped divider with the legend stack in one corner panel."""
    legends = _choose_legends()
    total_h = _stack_height(legends)
    panel_h = total_h + 2 + random.randint(0, L_LAYOUT_EXTRA_PADDING)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(corner, panel_h, panel_w, total_h)

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
    _place_legend_stack(grid, legends, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Choose a layout that preserves one clean vertical legend column."""
    if random.choice(["vertical", "l"]) == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


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
    """Grow irregular blob regions around each hole seed. Returns the full forbidden set."""
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
    """Crop mask to tight bbox, then validate aspect ratio and hole count."""
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    for r, c in forbidden:
        rmin, cmin = min(rmin, r), min(cmin, c)
        rmax, cmax = max(rmax, r), max(cmax, c)
    h = rmax - rmin + 1
    w = cmax - cmin + 1
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
        d
        for d in range(1, max_d + 1)
        if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, value in enumerate(extras) if value < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int | None]] = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color))

    legend_holes = {hole_count for hole_count, _color in legends}
    distractor_holes = [hole_count for hole_count in range(MAX_HOLE_COUNT) if hole_count not in legend_holes]
    assert 0 in distractor_holes, "combo requires 0-hole distractors to keep the 1-based code visible"

    plan.append((0, None))
    for _ in range(n_distractors - 1):
        plan.append((random.choice(distractor_holes), None))
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
    items.sort(key=lambda item: item[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _size, sprite, color in items:
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
    raise RuntimeError("failed to generate e3721c99 combo variant input")
