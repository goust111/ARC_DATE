# Combo variant of e3721c99 (index 10)
# combo_spec: D4/0+D10/1
# Name: 部分重着色（洞边框着色）+高度相似图例

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   summary:
#     - Keep the original topological matching rule: a legend color still stands for a
#       specific enclosed-hole count.
#     - Fuse two changes:
#       1) D4/0: a matched grey sprite is no longer repainted wholesale; only the grey
#          pixels that touch an enclosed hole are recolored.
#       2) D10/1: all legend shapes use the same 5x5 outer silhouette, differing only
#          in how many tiny interior holes are opened.
#   behavior_trace:
#     - Concrete legend example:
#         * red legend = solid 5x5 block with only the top-left inner slot opened -> 1 hole
#         * green legend = same 5x5 block, but three fixed inner slots opened -> 3 holes
#       These two legends look nearly identical because the border and almost all filled
#       pixels coincide; only the count of 1x1 interior voids changes.
#     - Concrete sprite example:
#         * a workspace grey blob encloses 3 holes
#         * original task: every pixel of that blob turns green
#         * combo variant: only the inner rim pixels touching the three holes turn green;
#           the outer body stays grey
#         * a 2-hole distractor, if 2 is not listed by the legends, still disappears
#     - Net effect:
#         * solver must first distinguish very similar legend exemplars by careful hole
#           counting, then apply a per-pixel partial recolor rather than a flat recolor
#   complexity_calibration:
#     - Legend hole counts are restricted to 1..4.
#       This avoids 0-hole matches, which would look unchanged under the partial-color rule.
#     - Every sample has 2-4 legend entries chosen from those four counts, so multiple
#       highly similar exemplars appear side by side.
#     - Each legend count gets 1-3 matching sprites, creating repeated evidence for the
#       mapping instead of a single fragile example.
#     - Matching sprites are required to keep a visible grey core after recoloring, so the
#       output cannot collapse back into full recoloring.
#   implicit_assumptions_review:
#     - `_legend_mapping()` originally tolerated legends with visibly different widths.
#       In this combo, width/height no longer help; only hole counting may be used.
#     - The original whole-object assignment `out[r, c] = newc` is invalid now because a
#       matched sprite becomes two-tone. The solver must classify pixels into hole-boundary
#       versus remaining body.
#     - `_hole_count()` on a tight local bbox still works unchanged because enclosed-hole
#       topology is the same invariant for both legend glyphs and workspace sprites.
#     - Restricting legends to positive hole counts introduces a new solver-side assumption:
#       any matched sprite must actually contain at least one enclosed hole. The generator
#       guarantees this by never placing 0-hole legends.
#   placement_legality:
#     - Legends stay inside the panel, one cell away from edges and dividers, with a fixed
#       5x5 bbox and one-cell spacing between neighbors.
#     - Sprites are still placed only in the workspace, remain 4-connected, never overlap,
#       and keep at least one background cell of separation via 8-neighborhood dilation.
#     - Panel thickness is raised so a 5x5 legend can fit cleanly in horizontal, vertical,
#       and L-corner layouts without clipping.
#   visual_observability:
#     - Humans can compare nearly identical legend blocks and notice that only the number
#       of tiny internal holes changes.
#     - The same hole count reappears across several workspace sprites, and those matched
#       sprites all show the same legend color specifically on their inner rims.
#     - No hidden ordering, numeric color arithmetic, or non-visual metadata is used.
#   cleanliness:
#     - Divider and legend panel remain spatially separated from the workspace.
#     - The legend family is visually consistent because every exemplar shares the same
#       outer frame; the workspace remains readable because matched sprites become only
#       mildly two-tone instead of heavily overpainted.
#   generator_guarantees_for_new_main_assumptions:
#     - All legend components share identical 5x5 outer support after holes are filled.
#     - Every matched sprite has positive hole count and at least two non-boundary grey
#       pixels that survive in the output.
#     - `_combo_signal_strength()` rejects generated inputs whose combo output would be
#       indistinguishable from the original full-recolor behavior.


def _enclosed_hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected background regions fully enclosed by a binary mask."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            cells = []
            while st:
                r, c = st.pop()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes.append(cells)
    return holes


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed holes in a binary mask using 4-connectivity."""
    return len(_enclosed_hole_regions(mask))


def _hole_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return mask pixels that touch at least one enclosed hole by 4-neighborhood."""
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for hole in _enclosed_hole_regions(mask):
        for r, c in hole:
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    boundary[rr, cc] = True
    return boundary


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg, grey) color and map hole_count -> color.

    Candidate legend components must:
      - be interior to the grid (to exclude the divider),
      - have area at least 2x2.
    """
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey sprites to very similar legend glyphs, then color only their hole rims.

    Input structure:
    - The background is the most common color. Grey (5) sprites lie in the workspace.
    - Near an edge or corner, a legend panel contains 2-4 colored components that all share
      the same outer 5x5 silhouette and differ only by how many enclosed holes they contain.

    Rule:
    - Read the legend as a mapping from enclosed-hole count to legend color.
    - For each grey sprite, count its enclosed holes inside its own tight bounding box.
    - If that count appears in the legend, recolor only the sprite pixels that touch a hole;
      leave the rest of the sprite grey. If the count is absent, erase the sprite to background.

    Key insight:
    - Both matching and the visible output depend on hole topology: first count holes in very
      similar legend exemplars, then transfer only the legend color to the matched sprite's
      inner boundary around those holes.
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
        if hc not in legend:
            for r, c in pts:
                out[r, c] = bg
            continue
        boundary = _hole_boundary_mask(sub)
        newc = int(legend[hc])
        for r, c in pts:
            out[r, c] = newc if boundary[r - r0, c - c0] else grey

    return out


# --------------- generator (holes-first organic blob) ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
LEGEND_SIZE = 5
LEGEND_HOLE_SLOTS = ((1, 1), (1, 3), (3, 1), (3, 3))
LEGEND_HOLE_VALUES = tuple(range(1, len(LEGEND_HOLE_SLOTS) + 1))
PANEL_SIZE_RANGE = (7, 10)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 23
L_LAYOUT_EXTRA_PADDING = 2

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
MAX_HOLE_COUNT = 5
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
MIN_VISIBLE_CORE = 2

# -- generation --
GEN_RETRIES = 300
MIN_COMBO_SIGNAL = 4

_STRUCT8 = generate_binary_structure(2, 2)


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a fixed-size legend patch; arrangement only affects line placement later."""
    del arrangement
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), int(color), dtype=int)
    for i in range(hole_count):
        r, c = LEGEND_HOLE_SLOTS[i]
        patch[r, c] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of fixed-size legend patches plus 1-cell gaps."""
    return len(legends) * LEGEND_SIZE + max(0, len(legends) - 1)


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 2-4 distinct positive hole counts that fit in the chosen panel span."""
    del arrangement
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(LEGEND_HOLE_VALUES, count)
        if len(holes) * LEGEND_SIZE + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends("horizontal", max_span + 1)
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
            cc += LEGEND_SIZE + 1
        else:
            rr += LEGEND_SIZE + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-row divider with a horizontal legend line above or below it."""
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - total - 1)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - LEGEND_SIZE - 1)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - LEGEND_SIZE - 1)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a full-column divider with a vertical legend line on one side."""
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - total - 1)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - LEGEND_SIZE - 1)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - LEGEND_SIZE - 1)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend-origin bounds for an L-corner panel."""
    n = GRID_SIZE
    cross_dim = LEGEND_SIZE if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else LEGEND_SIZE

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
    """Place an L-shaped divider with a legend line tucked into one corner."""
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
    """Sample a random sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Find well-separated interior seeds for the requested number of holes."""
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
    """Grow irregular forbidden regions that later become holes."""
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
    """Find a starting growth cell nearest to the canvas center."""
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
    """Grow a connected blob of roughly the requested area while avoiding holes."""
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
    """Seal ring gaps around holes and keep only the largest 4-connected component."""
    csz = mask.shape[0]
    for _ in range(SEAL_ITERS):
        sealed = False
        for r, c in sorted(forbidden):
            for nr, nc in neighbors4(r, c):
                if (
                    0 <= nr < csz
                    and 0 <= nc < csz
                    and not mask[nr, nc]
                    and (nr, nc) not in forbidden
                ):
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


def _sprite_has_visible_core(mask: np.ndarray, hole_count: int) -> bool:
    """Require matched sprites to keep some grey interior after hole-rim recoloring."""
    if hole_count <= 0:
        return True
    boundary = _hole_boundary_mask(mask)
    area = int(np.sum(mask))
    boundary_area = int(np.sum(boundary))
    core_area = area - boundary_area
    return 0 < boundary_area < area and core_area >= MIN_VISIBLE_CORE


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop to the tight bbox (including holes), then validate size, aspect, and topology."""
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
    if not _sprite_has_visible_core(cropped, hole_count):
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    """Generate a random amorphous grey sprite with the requested hole count."""
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


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Build a shuffled list of (hole_count, color_or_None) sprite requests."""
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
    """8-connected 1-cell dilation."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a valid placement that avoids blocked regions and neighboring sprites."""
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


def _matched_output_patch(sprite: np.ndarray, color: int) -> np.ndarray:
    """Apply the combo transform to a single matching sprite patch."""
    mask = sprite == GREY
    boundary = _hole_boundary_mask(mask)
    out = sprite.copy()
    out[boundary] = int(color)
    return out


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all matching and distractor sprites."""
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
            out_patch = _matched_output_patch(sprite, int(color))
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _full_recolor_output(grid: np.ndarray) -> np.ndarray:
    """Reference output for the original whole-sprite recolor rule on the same input."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _legend_mapping(g, bg, GREY)
    out = g.copy()
    for pts in components_mask(g == GREY, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        newc = int(legend[hc]) if hc in legend else bg
        for r, c in pts:
            out[r, c] = newc
    return out


def _combo_signal_strength(grid: np.ndarray, combo_out: np.ndarray) -> int:
    """Count cells that stay grey here but would be recolored in the original task."""
    full_out = _full_recolor_output(grid)
    return int(np.sum((combo_out == GREY) & (full_out != GREY)))


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build layout, place sprites, validate oracle and combo signal."""
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
    if _combo_signal_strength(grid, expected) < MIN_COMBO_SIGNAL:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a valid combo-variant input grid."""
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
