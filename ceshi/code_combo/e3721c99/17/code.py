# Combo variant of e3721c99 (index 17)
# combo_spec: D4/3+D12/2
# Name: 渐变色阶着色+未匹配精灵标记色

"""
Design analysis for the combo variant.

a_concretization_and_trace:
  - The scene geometry stays close to the source task: a 30x30 grid, one divider that carves out a legend panel,
    2-4 colored legend glyphs whose shapes encode distinct hole counts, and many grey sprites in the workspace.
  - The combo changes two semantic branches at once.
    1. Matched sprites no longer copy legend colors. They use a fixed hole-count gradient:
       0->teal, 1->blue, 2->green, 3->yellow, 4->orange, 5->red.
    2. Unmatched sprites no longer disappear. They become a dedicated marker color, pink.
  - Small trace:
      legends show hole counts {1, 2, 3}
      legend display colors are {yellow, teal, maroon}
      grey sprites have hole counts [1, 3, 5]
      source task output would be [yellow, maroon, background]
      combo output becomes [blue, yellow, pink]
    The first two sprites prove that legend colors are decoys for matched recoloring, while the third proves that
    "not in legend" is now a visible state instead of erasure.

b_complexity_calibration:
  - To make the gradient discoverable from only a few examples, the generator uses a consecutive run of 2-4 hole
    counts in the legend. This tends to expose several adjacent gradient steps in the same sample.
  - Every legend hole count gets at least one matching sprite, so the output repeats each active gradient color.
  - Every sample also contains at least two unmatched sprites, so the marker branch is visible and cannot be missed.
  - The marker color is outside the gradient palette, so the three output categories remain visually separable:
    background, gradient-colored matches, and marker-colored mismatches.

c_implicit_assumption_audit_main:
  - The original solver treated legend color as payload through `_legend_mapping(...) -> hole_count -> color`.
    That breaks under this combo because legend color no longer controls matched recoloring.
  - The combo solver therefore extracts only the set of declared legend hole counts.
  - The original unmatched branch `else bg` is invalid now because unmatched sprites must stay visible.
    It is replaced with `else MARKER`.
  - `_gradient_color(hc)` assumes hole counts stay in 0..5. The generator preserves the original hole-count range.
  - The loop over grey connected components still assumes sprites are separated well enough to be processed one by
    one from their own bounding boxes. Placement logic remains responsible for that guarantee.

d_placement_legality:
  - Divider and legend placement still reserve a blocked panel region exactly as in the source task.
  - Workspace sprites are generated as separate 4-connected grey components, with 8-neighborhood padding to prevent
    touching or overlap, so component extraction and hole counting stay stable.
  - Sprite bbox, aspect-ratio, and canvas-size constraints stay within the original physical envelope, so the combo
    does not rely on any new geometry the generator cannot realize.

e_visual_observability:
  - A human can still read the task by counting holes on legends and sprites.
  - The gradient is not arbitrary per grid: the same hole count always maps to the same output color across samples.
  - Consecutive legend hole runs make the gradient look like a coherent scale rather than unrelated recolorings.
  - Pink unmatched sprites are repeatedly present, making it clear they represent "hole count absent from legend"
    instead of "this one specific topology".

f_visual_cleanliness:
  - The combo adds no overlays or extra symbols; all information is expressed by recoloring whole sprites in place.
  - Pink is reserved away from the legend panel, so unmatched markers cannot be mistaken for legend entries.
  - Legend colors are also forced to differ from the matched gradient color of the same hole count, which prevents
    the samples from visually collapsing back to the original "copy the legend color" behavior.

g_change_inferability:
  - The output color rule has only two branches and both are structurally grounded by the legend:
      listed hole count -> fixed gradient color
      unlisted hole count -> fixed pink marker
  - Because each sample shows multiple matched counts plus multiple unmatched sprites, the solver can infer both the
    hole-validity test and the global color-scale rule from 2-4 examples.

h_behavioral_guardrails:
  - Oracle validation remains `main(input) == expected`.
  - The generator additionally enforces three non-degeneracy conditions:
    1. at least one sprite per legend hole count,
    2. at least two unmatched sprites,
    3. legend display color for a hole count never equals that hole count's gradient color.
  - These ensure the combo differs from both the source task and the D4-only variant on every valid sample.

i_new_main_assumptions_after_change:
  - Every interior non-grey non-background component of size at least 2x2 is treated as a legend glyph.
  - Legend hole counts must be unique.
  - Any grey sprite with a listed hole count should survive and take `_gradient_color(hc)`.
  - Any grey sprite with an unlisted hole count should survive and take `MARKER`.
  - The generator explicitly constructs inputs that satisfy all of these assumptions.
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


BG = 0
DIV = 1
MARKER = 6
GREY = 5
GRID_SIZE = 30
HOLE_COLOR_SCALE = (8, 1, 3, 4, 7, 2)  # 0->teal, 1->blue, 2->green, 3->yellow, 4->orange, 5->red
LEGEND_COLORS = [1, 2, 3, 4, 7, 8, 9]  # marker color 6 is reserved away from legends


def _gradient_color(hole_count: int) -> int:
    """Map hole count to the fixed output color ramp."""
    assert 0 <= hole_count < len(HOLE_COLOR_SCALE), f"unsupported hole count {hole_count}"
    return int(HOLE_COLOR_SCALE[hole_count])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by using legend topology as a validity test, then applying a gradient-or-marker rule.

    Input structure:
    - The most common color is background; grey (5) sprites occupy the workspace.
    - A separated legend panel near one edge or corner contains colored glyphs whose shapes encode distinct hole counts.
      Those legend colors identify the glyphs visually but do not determine matched output colors.

    Rule:
    - Read the set of hole counts declared by the legend glyphs.
    - For each grey sprite, count the enclosed holes inside its own bounding box.
    - If the sprite's hole count appears in the legend, recolor it with the fixed hole-count gradient.
      Otherwise recolor the whole sprite with the dedicated marker color.

    Key insight:
    - The legend still decides which topologies are valid, but output color no longer comes from the legend itself:
      valid counts use a global gradient, while absent counts collapse to one shared fallback state.
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
        newc = _gradient_color(hc) if hc in legend_holes else MARKER
        for r, c in pts:
            out[r, c] = newc

    return out


# --------------- generator (holes-first organic blob) ---------------

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = len(HOLE_COLOR_SCALE)
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MIN_UNMATCHED = 2
MAX_UNMATCHED = 7
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


def _legend_span(hole_count: int) -> int:
    """Width or height of a single legend patch along its placement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan horizontal patch or a Spanx3 vertical patch."""
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


def _assign_legend_colors(holes: list[int]) -> list[int]:
    """Choose distinct legend colors, avoiding the gradient color of the same hole count."""
    for _ in range(200):
        colors = random.sample(LEGEND_COLORS, len(holes))
        if all(color != _gradient_color(hole_count) for hole_count, color in zip(holes, colors)):
            return colors

    ordered = sorted(holes, key=lambda hole_count: _gradient_color(hole_count))
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(ordered):
            return True
        hole_count = ordered[idx]
        candidates = [c for c in LEGEND_COLORS if c not in used and c != _gradient_color(hole_count)]
        random.shuffle(candidates)
        for color in candidates:
            assignment[hole_count] = color
            used.add(color)
            if backtrack(idx + 1):
                return True
            used.remove(color)
            del assignment[hole_count]
        return False

    assert backtrack(0), "legend colors must admit a non-gradient assignment"
    return [assignment[hole_count] for hole_count in holes]


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """
    Pick 2-4 legends whose hole counts form a consecutive run.

    Consecutive runs make the fixed gradient easier to infer as a scale, while legend colors remain decoys.
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
    colors = _assign_legend_colors(holes)
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
    else:
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

    n = GRID_SIZE
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
            best = max(range(1, n_comp + 1), key=lambda label: int(np.sum(labeled == label)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop mask to a tight bbox, then validate size, aspect ratio, and topology."""
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


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, bool]]:
    """Build a shuffled list of (hole_count, is_match) requests for all sprites."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_unmatched = min(MAX_UNMATCHED, remaining)
    valid_unmatched = [
        count
        for count in range(MIN_UNMATCHED, max_unmatched + 1)
        if remaining - count <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_unmatched = random.choice(valid_unmatched)
    extra_matches = remaining - n_unmatched
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [idx for idx, value in enumerate(extras) if value < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for idx, (hole_count, _) in enumerate(legends):
        for _ in range(1 + extras[idx]):
            plan.append((hole_count, True))
    legend_holes = {hole_count for hole_count, _ in legends}
    unmatched_holes = [hole_count for hole_count in range(MAX_HOLE_COUNT) if hole_count not in legend_holes]
    for _ in range(n_unmatched):
        plan.append((random.choice(unmatched_holes), False))
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
    """Generate and place all sprites, painting expected output with gradient or marker colors."""
    items = []
    for hole_count, is_match in _sprite_plan(legends):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, hole_count, is_match))
    items.sort(key=lambda item: item[0], reverse=True)

    occupied = np.zeros_like(grid)
    unmatched_seen = 0
    for _, sprite, hole_count, is_match in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)

        out_color = _gradient_color(hole_count) if is_match else MARKER
        unmatched_seen += int(not is_match)
        out_patch = np.where(sprite == GREY, out_color, BG)
        paste(expected, out_patch, r, c, transparent=BG)

    return unmatched_seen >= MIN_UNMATCHED


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, validate against the solver."""
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
    if int(np.sum(expected == MARKER)) == 0:
        return None
    if int(np.sum(expected == GREY)) != 0:
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
