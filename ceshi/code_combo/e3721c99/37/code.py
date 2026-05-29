# Combo variant of e3721c99 (index 37)
# combo_spec: D5/0+D8/3
# Name: 洞数取反+点阵编码

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis for combo D5/0 + D8/3
# a) Concrete behavior and trace:
#    The original task used colored legend shapes whose own visible hole counts
#    directly named the target class. The fused task changes both sides of that
#    mapping chain. Each legend entry is now a solid 3x3 color block plus 0-3
#    isolated same-color dots in fixed nearby slots; the dot count is the target
#    lookup value. Grey workspace sprites are still irregular connected blobs, but
#    they no longer match by their actual holes. Instead the solver computes the
#    complementary key `3 - actual_holes`.
#    Example:
#      legend: red block + 0 dots, green block + 1 dot, yellow block + 3 dots
#      grey sprite actual holes = 3
#      original task would need a 3-hole legend shape and would color it yellow
#      D5-only would read 3 dots and still color it yellow
#      fused task reads sprite key = 3 - 3 = 0, so it becomes red
#    The human-visible difference is therefore two-step: the legend is decoded by
#    explicit marker counting, and the sprite-topology comparison is reversed.
# b) Complexity calibration:
#    We restrict the operative hole-value range to 0..3. That keeps the inversion
#    legible as two opposite pairs (0<->3 and 1<->2), avoids overly large legends,
#    and still provides four classes. Each puzzle uses exactly three legend entries,
#    so one class is absent and distractors can still be erased. Multiple sprites
#    per legend color remain, giving repeated support for the fused rule.
# c) Hidden assumptions review:
#    The original `_legend_mapping` assumed one non-border legend component per
#    color and that the component's own holes encode the key. D5/0 breaks both:
#    each legend color intentionally occupies several disconnected components, and
#    the main 3x3 block has zero holes regardless of the target value. We replace
#    that logic with: find the unique interior 3x3 block for each color, then count
#    same-color singleton markers in fixed horizontal or vertical slots.
#    The original grey-sprite loop also assumed `actual_holes` is the lookup key.
#    D8/3 invalidates that, so the loop must transform the measured hole count with
#    `3 - actual_holes` before consulting the legend mapping.
# d) Placement legality:
#    The generator preserves the original divider/panel/workspace split and keeps
#    legends inset from borders and divider by at least one cell. Marker dots are
#    singleton components with background gaps from both the block and each other.
#    Workspace sprites are placed with 1-cell padded non-overlap via dilation, so
#    neither sprite-sprite nor sprite-legend collisions can create ambiguous holes.
# e) Visual observability:
#    Across examples a solver can see that legend objects are no longer holey bars
#    but solid color blocks with countable dots. They can also compare outputs and
#    notice that the most hollow sprites do not take the color of the largest dot
#    count, but rather the opposite dot count under a fixed 0..3 scheme.
# f) Visual cleanliness:
#    Limiting marker counts to 0..3 lets every legend entry fit in a compact 3x3 or
#    5x5 patch. The panel stays sparse, dots never touch their anchor block, and the
#    workspace remains visually separated by the divider, so the two information
#    channels (legend markers vs. sprite holes) stay easy to parse.
# g) Color-change inferability:
#    Colors are few (exactly three legend colors per sample), and each color carries
#    its own local cue: one colored block plus nearby dots. No ordering by numeric
#    color id is used. The generator guarantees at least one sprite for each legend
#    color and at least one absent class, so both recoloring and erasure are visible.

HOLE_VALUE_MAX = 3
LEGEND_BLOCK = 3
LEGEND_THICKNESS = 5
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_COUNT_RANGE = (3, 3)
MAX_HOLE_COUNT = HOLE_VALUE_MAX + 1
PANEL_SIZE_RANGE = (6, 9)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3
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
GEN_RETRIES = 300

_HORIZONTAL_DOT_SLOTS = [(0, 4), (2, 4), (4, 4)]
_VERTICAL_DOT_SLOTS = [(4, 0), (4, 2), (4, 4)]
_STRUCT8 = generate_binary_structure(2, 2)


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
                    if 0 <= rr < h and 0 <= cc < w and bg[r, c] and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _inverted_hole_value(hole_count: int) -> int:
    """Convert an actual hole count in 0..3 into its complementary lookup key."""
    return HOLE_VALUE_MAX - int(hole_count)


def _touches_outer_border(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> bool:
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _legend_slots(r0: int, c0: int, arrangement: str) -> list[tuple[int, int]]:
    slots = _HORIZONTAL_DOT_SLOTS if arrangement == "horizontal" else _VERTICAL_DOT_SLOTS
    return [(r0 + dr, c0 + dc) for dr, dc in slots]


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one solid 3x3 legend block per non-(bg,grey) color, then decode the
    complementary-match target value from isolated same-color dots nearby.
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
            target = len(singletons)
        elif vertical_ok and not horizontal_ok:
            target = len(singletons)
        elif not singletons:
            target = 0
        else:
            raise AssertionError(f"color {c} has stray dots outside legend slots")

        assert 0 <= target <= HOLE_VALUE_MAX
        assert target not in mapping, f"legend target value {target} must map to exactly one color"
        mapping[target] = c
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by matching the complement of their hole count to a dot-coded color legend.

    Input structure:
    - The dominant background fills the grid; grey (5) sprites are scattered in the workspace region.
    - A divider carves out a legend panel near an edge or corner. Each legend entry is a solid 3x3
      colored block plus 0-3 isolated same-color dots in fixed slots beside it.

    Rule:
    - For each legend color, count its nearby dots to obtain that color's target value.
    - For every grey sprite, count its actual enclosed holes inside its own bounding box, then convert
      that count with the 0..3 complement rule `3 - holes`.
    - Recolor the grey sprite to the legend color whose dot count matches that complementary value;
      erase grey distractors whose complementary value is absent from the legend.

    Key insight:
    - The puzzle now requires two abstractions in sequence: the legend must be decoded by explicit
      marker counting, and the grey sprites must be matched by the inverse of their visible topology,
      not by the topology itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain dot-coded legend entries"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        actual_holes = _hole_count(sub)
        key = _inverted_hole_value(actual_holes)
        newc = int(legend[key]) if key in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


def _legend_span(target_value: int) -> int:
    """Width or height of a single legend patch along the arrangement axis."""
    return LEGEND_BLOCK if target_value == 0 else LEGEND_THICKNESS


def _legend_patch(target_value: int, color: int, arrangement: str) -> np.ndarray:
    """Build a dot-coded legend patch around a solid 3x3 color block."""
    span = _legend_span(target_value)
    if arrangement == "horizontal":
        patch = np.full((LEGEND_THICKNESS, span), BG, dtype=int)
        patch[:LEGEND_BLOCK, :LEGEND_BLOCK] = int(color)
        for rr, cc in _HORIZONTAL_DOT_SLOTS[:target_value]:
            patch[rr, cc] = int(color)
    else:
        patch = np.full((span, LEGEND_THICKNESS), BG, dtype=int)
        patch[:LEGEND_BLOCK, :LEGEND_BLOCK] = int(color)
        for rr, cc in _VERTICAL_DOT_SLOTS[:target_value]:
            patch[rr, cc] = int(color)
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(target) for target, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick three target values in 0..3 with distinct colors that fit in the panel."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        values
        for values in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(v) for v in values) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    targets = list(random.choice(combos))
    random.shuffle(targets)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(targets, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for target_value, color in legends:
        patch = _legend_patch(target_value, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


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


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend placement bounds for an L-corner."""
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
    """Randomly choose and apply a divider layout."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    """Sample a random sprite area based on actual hole count."""
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
    """Grow irregular forbidden blobs that will become holes."""
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
            best = max(range(1, n_comp + 1), key=lambda lab: int(np.sum(labeled == lab)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop mask to a tight bbox, then validate size and actual hole count."""
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
    """Generate a random amorphous grey sprite with the specified actual hole count."""
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
    """Build a shuffled list of (sprite_actual_holes, color_or_None) pairs."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)

    legend_targets = {target for target, _ in legends}
    distractor_holes = [
        holes for holes in range(MAX_HOLE_COUNT) if _inverted_hole_value(holes) not in legend_targets
    ]
    min_d = 1 if distractor_holes and max_d >= 1 else 0
    valid_d = [d for d in range(min_d, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)]
    n_distractors = random.choice(valid_d) if valid_d else 0

    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, (target_value, color) in enumerate(legends):
        actual_holes = _inverted_hole_value(target_value)
        for _ in range(1 + extras[i]):
            plan.append((actual_holes, color))
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
    """Find a random valid placement for a sprite."""
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
    """Generate and place all sprites onto the workspace."""
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
    """Single attempt: build layout, place sprites, validate with the solver."""
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
    raise RuntimeError("failed to generate e3721c99 combo input")
