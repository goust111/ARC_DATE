# Variant of e3721c99
# Dimension: D3 — 目标结构：图例的设计与位置 (dst)
# Change: 彩色图例形状位于网格左上角，每个图例形状有特定颜色和洞数... → 图例为纯色块（无洞信息）配文字式编码
#
# Design analysis
# a) Concrete form and behavior trace
#    - The old task encoded `hole_count -> color` directly inside each colored legend
#      shape: the legend object itself had holes, so the solver could count holes on the
#      colored component and reuse the same topological test on the grey sprites.
#    - This variant moves all legend information into a fixed top-left corner panel.
#      Each legend entry is now:
#        * a 2x2 solid color swatch (target output color),
#        * a 5x3 digit made from fixed-color marker pixels immediately to its right.
#      The digit is a visible symbolic code for numbers 0..5, so the legend no longer
#      contains any holes. Grey sprites still live in the workspace and are still matched
#      by their own hole counts.
#    - Trace on one entry:
#        old: colored legend shape with 3 holes -> solver counts 3 holes on that shape
#             -> grey sprite with 3 holes becomes that color.
#        new: color swatch is always solid 2x2, so direct hole-count comparison is gone.
#             Solver must first decode the adjacent 5x3 digit `3`, build mapping 3->color,
#             then recolor every 3-hole grey sprite.
#      The change is therefore localized to legend parsing, while sprite topology logic
#      remains the same.
# b) Complexity calibration
#    - Digits are restricted to 0..5 because grey sprites already use 0..5 holes in the
#      original task. This preserves the original reasoning target and avoids adding extra
#      categories.
#    - Legend count stays 2..4 so each grid has repeated evidence but not too many colors.
#    - Digits use a fixed marker color and a fixed 5x3 font, which adds one symbolic
#      decoding step without making the representation noisy or ambiguous.
# c) Implicit assumption audit for `main()`
#    - Old `_legend_mapping` assumed every non-bg/non-grey color has a unique multi-cell
#      component whose own holes define the mapping. That assumption is false now because
#      swatches are hole-free 2x2 blocks.
#    - The new parser instead assumes:
#        * divider color 1 forms a top-left L with pivot `(div_r, div_c)`,
#        * legend swatches are the only non-bg/non-grey/non-divider/non-code colors,
#        * each swatch is exactly one 2x2 component inside the panel,
#        * the digit box is always in the 5x3 window right of the swatch.
#      The generator explicitly enforces all of these.
#    - Sprite recoloring still assumes grey objects are 4-connected components and their
#      hole count is computed inside their tight bbox; that remains unchanged.
# d) Placement legality
#    - The legend panel occupies the top-left rectangle and is fenced by an L-divider.
#      Sprites are placed outside the panel and also not adjacent to divider cells.
#    - Sprite placement keeps the old non-overlap + one-cell padding behavior via an
#      8-neighborhood dilation check, so objects stay visually separated.
# e) Visual observability
#    - Across 2-4 examples, a human can see that all color hints now come from the same
#      corner panel, and that the colored items there are plain squares rather than holey
#      shapes. The only varying structure next to them is the repeated digit font, which
#      makes the “decode symbol, then match sprite holes” rule visible.
# f) Visual cleanliness
#    - The panel uses one entry per row with fixed spacing, a single code color, and a
#      narrow corner footprint. The workspace remains mostly open, so the panel and the
#      sprites do not visually tangle.
# g) Color-change inferability
#    - Output colors still come directly from visible swatches, keeping the color mapping
#      grounded. Only the number source changes: it is read from the adjacent digit rather
#      than from holes in the legend object itself.

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _hole_count(mask: np.ndarray) -> int:
    """Count 4-connected holes inside a binary object mask."""
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
ENTRY_H = 5
ENTRY_W = 6
ENTRY_GAP = 1
PANEL_MARGIN = 1
SWATCH_H = 2
SWATCH_W = 2
SWATCH_ROW_OFFSET = 1
SWATCH_COL_OFFSET = 0
DIGIT_COL_OFFSET = 3
DIGIT_PATTERNS = {
    0: ("111", "101", "101", "101", "111"),
    1: ("010", "110", "010", "010", "111"),
    2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"),
    4: ("101", "101", "111", "001", "001"),
    5: ("111", "100", "111", "001", "111"),
}

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


def _digit_patch(value: int) -> np.ndarray:
    """Render one 5x3 code digit using the fixed marker color."""
    rows = DIGIT_PATTERNS[int(value)]
    patch = np.full((5, 3), BG, dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "1":
                patch[r, c] = CODE
    return patch


def _decode_digit(patch: np.ndarray) -> int:
    """Decode a 5x3 marker patch back into a number 0..5."""
    mask = patch == CODE
    for value, rows in DIGIT_PATTERNS.items():
        target = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
        if patch.shape == target.shape and np.array_equal(mask, target):
            return int(value)
    raise AssertionError("unrecognized legend digit")


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Parse the fixed top-left legend panel.

    Each legend entry consists of:
    - a 2x2 colored swatch,
    - a 5x3 digit patch of color CODE immediately to the right.
    """
    divider_pts = np.argwhere(g == DIV)
    assert divider_pts.size > 0, "legend divider must exist"
    div_r = int(divider_pts[:, 0].max())
    div_c = int(divider_pts[:, 1].max())
    assert np.all(g[div_r, : div_c + 1] == DIV), "divider row must run from left edge to pivot"
    assert np.all(g[: div_r + 1, div_c] == DIV), "divider col must run from top edge to pivot"

    panel = g[:div_r, :div_c]
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(panel)))) if c not in (bg, grey, DIV, CODE)]
    for color in colors:
        comps = components_mask(panel == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if (r1 - r0 + 1, c1 - c0 + 1) != (SWATCH_H, SWATCH_W):
                continue
            if np.sum(panel[r0 : r1 + 1, c0 : c1 + 1] == color) != SWATCH_H * SWATCH_W:
                continue
            candidates.append((r0, c0, r1, c1))
        assert len(candidates) == 1, f"color {color} must have exactly one legend swatch"
        r0, c0, r1, c1 = candidates[0]
        digit = panel[r0 - SWATCH_ROW_OFFSET : r0 - SWATCH_ROW_OFFSET + 5, c1 + 2 : c1 + 5]
        assert digit.shape == (5, 3), "digit window must be 5x3"
        value = _decode_digit(digit)
        assert value not in mapping, f"hole count {value} must map to exactly one color"
        mapping[int(value)] = int(color)
    assert 2 <= len(mapping) <= 4, "legend must contain 2-4 entries"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by decoding a corner legend and matching sprite hole counts.

    Input structure:
    - bg is the dominant color; grey (5) sprites are scattered in the workspace.
    - The top-left corner contains a legend panel fenced by an L-shaped divider.
      Each legend entry is a solid colored swatch plus a fixed-color 5x3 digit code.

    Rule:
    - Read each legend entry as digit value -> swatch color.
    - For every grey sprite, count its enclosed holes inside its tight bounding box.
    - Recolor sprites whose hole count appears in the legend; erase grey distractors
      whose hole count is not listed.

    Key insight:
    - The color legend no longer exposes topology directly. You must first decode the
      symbolic number beside each swatch, then use that number to classify grey sprites
      by their hole count.
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


def _choose_legends() -> list[tuple[int, int]]:
    """Pick distinct hole-count/color legend entries."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    holes = list(random.choice(list(itertools.combinations(range(MAX_HOLE_COUNT), count))))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _entry_origin(idx: int) -> tuple[int, int]:
    """Top-left coordinate of legend entry `idx` inside the panel."""
    return PANEL_MARGIN + idx * (ENTRY_H + ENTRY_GAP), PANEL_MARGIN


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place the fixed top-left legend panel and its L-shaped divider."""
    legends = _choose_legends()
    panel_h = PANEL_MARGIN + len(legends) * ENTRY_H + (len(legends) - 1) * ENTRY_GAP + PANEL_MARGIN
    panel_w = PANEL_MARGIN + ENTRY_W + PANEL_MARGIN
    div_r = panel_h
    div_c = panel_w

    grid[div_r, : div_c + 1] = DIV
    grid[: div_r + 1, div_c] = DIV
    blocked[:div_r, :div_c] = 9
    blocked[div_r, : div_c + 1] = 9
    blocked[: div_r + 1, div_c] = 9

    for idx, (hole_count, color) in enumerate(legends):
        er, ec = _entry_origin(idx)
        sr = er + SWATCH_ROW_OFFSET
        sc = ec + SWATCH_COL_OFFSET
        grid[sr : sr + SWATCH_H, sc : sc + SWATCH_W] = int(color)
        paste(grid, _digit_patch(hole_count), er, ec + DIGIT_COL_OFFSET)
    return legends


def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
    """Find separated seed positions for interior holes."""
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


def _grow_hole_blobs(seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int) -> set[tuple[int, int]]:
    """Grow irregular forbidden regions that will become sprite holes."""
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
                    if any((nnr, nnc) in forbidden and (nnr, nnc) not in blob for nnr, nnc in neighbors4(nr, nc)):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            p = random.choice(cands)
            forbidden.add(p)
            blob.add(p)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    """Pick a central non-forbidden start cell for blob growth."""
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
    """Grow a connected blob of the requested area."""
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
    """Seal around planned holes, then keep only the largest connected component."""
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


def _crop_and_validate(mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int) -> np.ndarray | None:
    """Crop the sprite to its bbox and verify size/aspect/hole count."""
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
    """Build a shuffled list of matched and distractor sprites."""
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
    """Dilate a boolean mask by one cell in the 8-neighborhood."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray) -> tuple[int, int] | None:
    """Find a valid random location outside the legend panel and away from other sprites."""
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


def _place_sprites(grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]) -> bool:
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
    """Single output-first attempt validated against `main()`."""
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
