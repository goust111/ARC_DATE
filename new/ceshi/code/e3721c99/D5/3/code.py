# Variant of e3721c99
# Dimension: D5 — 符号指令：图例的编码方式 (symbol_instructions)
# Change: 图例形状直接用自身的洞数编码匹配关系——计算图例形状的洞数，灰色精灵与之洞数相同则获得该图例颜色... → 图例位置隐含排序（从上到下对应洞数1,2,3...）
#
# Design analysis
# a) Concrete behavior:
#    - The workspace still contains scattered grey sprites whose identity is their hole count.
#    - The legend panel no longer shows holed shapes. Instead it shows a vertical stack of solid 3x3
#      color blocks. The top block means "1 hole", the next means "2 holes", and so on.
#    - Example of the changed execution point:
#      old task: a red legend patch with two visible voids meant "2 holes -> red";
#      new task: the second block from the top is red, so "2 holes -> red" even though the red block
#      itself has zero holes. The solver must sort legend blocks by row before recoloring sprites.
# b) Complexity calibration:
#    - Legend count stays small (2-4) so the ordinal code is inferable from a few examples.
#    - Every sample contains repeated matches because each legend color is used by at least one sprite
#      and may receive extras. Distractors remain present, so the task is not reduced to simple recolor-all.
#    - The legend blocks are identical solid squares, preventing accidental recovery from geometry.
# c) Original-main assumptions that break under D5:
#    - The old `_legend_mapping` measured holes inside each legend component. That now fails because all
#      legend components have hole count 0, so topology no longer distinguishes legend entries.
#    - Grey sprite hole counting itself is still valid, so only the legend decoding stage must change.
#    - The old "one legend component per color" assumption remains safe because colors stay distinct.
# d) Placement legality:
#    - Legend blocks are kept 1 cell away from borders and from the divider, so they are detectable as
#      non-border components and never merge with divider color 1 even when a legend block is also color 1.
#    - Sprites are still placed only in the workspace, with padded non-overlap and generator-side oracle
#      verification `main(input) == expected`.
# e) Visual observability:
#    - The top-to-bottom order is directly visible because legend blocks are in one clean vertical stack.
#    - The legend blocks carry no internal hole signal, so the new rule is genuinely about position.
#    - Horizontal-strip layouts were removed because they either compress the ordinal stack too much or
#      force oversized panels; side panels and L-corner panels preserve the original divider-based framing.
# f) Visual cleanliness:
#    - The legend uses uniform 3x3 squares with a 1-cell gap, yielding a compact readable column.
#    - The divider still separates legend from workspace, so colored instructions and grey sprites do not
#      visually interfere.
# g) Color inference:
#    - Colors remain few (2-4 legend colors). Their meaning is supported by repeated sprite instances and
#      by the shared vertical ordering cue across examples.
# New main() assumptions introduced by the variant and guaranteed by generate_input():
#    - There are 2-4 non-border solid legend squares.
#    - These squares form one strict top-to-bottom stack with a shared column overlap.
#    - The stack position encodes consecutive hole counts 1..k.

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


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Find solid legend blocks and decode them by vertical order.

    The top legend block means 1 hole, the next means 2 holes, etc.
    """
    h, w = g.shape
    legends: list[tuple[int, int, int, int, int]] = []
    seen_colors: set[int] = set()

    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        for pts in components_mask(g == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 2 or ww < 2:
                continue
            area = len(pts)
            if area != hh * ww:
                continue
            assert color not in seen_colors, f"color {color} must have exactly one legend block"
            legends.append((r0, c0, r1, c1, color))
            seen_colors.add(color)

    legends.sort(key=lambda item: (item[0], item[1]))
    assert 2 <= len(legends) <= 4, "legend must contain 2-4 solid blocks"

    for i in range(1, len(legends)):
        assert legends[i][0] > legends[i - 1][2], "legend blocks must be stacked from top to bottom"

    overlap_left = max(c0 for _r0, c0, _r1, _c1, _color in legends)
    overlap_right = min(c1 for _r0, _c0, _r1, c1, _color in legends)
    assert overlap_left <= overlap_right, "legend blocks must share a vertical column"

    return {i + 1: color for i, (_r0, _c0, _r1, _c1, color) in enumerate(legends)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching their hole count to the vertical order of solid legend blocks.

    Input structure:
    - bg is the dominant color, grey (5) sprites are scattered in the workspace, and a divider isolates
      a legend panel near one side or one corner.
    - The legend panel contains 2-4 solid colored blocks stacked vertically; these blocks have no holes
      of their own and are separated from the outer border.

    Rule:
    - Read the legend from top to bottom: the first block stands for 1 hole, the second for 2 holes, etc.
    - For each grey sprite, count its enclosed holes inside its own bounding box.
    - Recolor the sprite with the color of the legend block at that ordinal position; erase sprites whose
      hole count is not represented in the legend.

    Key insight:
    - The legend no longer encodes values through topology. The numeric code is implicit in vertical order,
      while topology is still used only on the grey sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

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


# --------------- generator (ordinal legend + organic holes-first sprites) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

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

# 8-connectivity structuring element for dilation
_STRUCT8 = generate_binary_structure(2, 2)


# ---------- legend helpers ----------

def _legend_block(color: int) -> np.ndarray:
    """Build one solid legend square."""
    return np.full((LEGEND_BLOCK_SIZE, LEGEND_BLOCK_SIZE), int(color), dtype=int)


def _stack_height(legends: list[tuple[int, int]]) -> int:
    """Total height of the vertical legend stack."""
    return len(legends) * LEGEND_BLOCK_SIZE + (len(legends) - 1) * LEGEND_GAP


def _choose_legends() -> list[tuple[int, int]]:
    """Pick 2-4 colors; their ordinal positions imply hole counts 1..k."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(range(1, count + 1), colors))


def _place_legend_stack(grid: np.ndarray, legends: list[tuple[int, int]], r: int, c: int) -> None:
    """Paste solid legend blocks in a single top-to-bottom stack."""
    rr = int(r)
    cc = int(c)
    for _hole_count, color in legends:
        paste(grid, _legend_block(color), rr, cc)
        rr += LEGEND_BLOCK_SIZE + LEGEND_GAP


# ---------- layout helpers ----------

def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-column divider with a legend strip on the left or right."""
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


def _l_corner_geometry(corner: str, panel_h: int, panel_w: int, total_h: int) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and valid legend origin bounds for a corner panel."""
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
    """Choose a divider layout that leaves enough vertical room for ordinal legends."""
    kind = random.choice(["vertical", "l"])
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
    """Crop mask to tight bbox (including forbidden cells), validate shape and holes."""
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


# ---------- sprite planning & placement ----------

def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Build a shuffled list of (hole_count, color_or_None) for all sprites to generate."""
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
    legend_holes = {h for h, _color in legends}
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
    raise RuntimeError("failed to generate e3721c99 D5 variant input")
