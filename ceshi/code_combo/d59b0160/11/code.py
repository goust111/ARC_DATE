# Combo variant of d59b0160 (index 11)
# combo_spec: D3/0+D7/2
# Name: 无显式边框+图例位置不固定
#
# Design analysis for the fused variant
# a) Concrete trace:
#    - Original: read a 4x4 legend fixed at the top-left corner; a green L border marks the
#      legend and the three interior colors form the deletion palette.
#    - Fused: the legend is now a tiny 3-cell L-shaped component with no green border. It sits
#      flush against exactly one grid edge and is separated from the main scene by a 1-2 cell
#      background moat on the inward side. Example: if the legend is on the right edge, the
#      solver must scan edge-touching mini-components and pick the one whose left side is followed
#      by a background strip, instead of reading g[:4, :4].
#    - Difference: both key anchors used by the original solver disappear: no fixed corner and
#      no border color. The legend must be localized by edge position plus the empty-space pattern.
# b) Complexity calibration:
#    - The free variables are deliberately small and repeated: legend side in {top,bottom,left,right}
#      and inward gap depth in {1,2}. The legend shape stays a small L triomino so humans can
#      recognize it across examples, while ordinary objects remain much larger scattered rectangles.
# c) Assumption audit against the original solver:
#    - `_find_legend` originally assumed the legend starts at (0,0), is square, and exposes a
#      colored bottom/right border. None of these remain valid.
#    - `main()` originally treated every non-bg component alike. Now it must exclude the legend
#      component itself from deletion and detect it before evaluating the larger objects.
# d) Placement legality:
#    - The generator reserves the legend footprint plus its inward separation using a placement
#      occupancy canvas. Ordinary components are then placed with padding, so nothing overlaps,
#      nothing leaves the board, and the moat remains intact.
# e) Visual observability:
#    - In examples, the only edge-touching non-bg object is a tiny isolated triad of three colors.
#      A human can infer that this edge micro-object acts as the legend because deleted objects are
#      exactly those that contain all three of its colors.
# f) Visual cleanliness:
#    - Ordinary components do not touch the border and remain separated from one another. The legend
#      is small, isolated, and never layered with another marker, so the grid stays readable.
# g) New solver assumptions guaranteed by the generator:
#    - There is exactly one edge-touching component of area 3 and bbox at most 2x2.
#    - That component has at least one inward background strip.
#    - All ordinary components are larger and stay away from the border, so legend detection is unique.

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color on the outer frame."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touching_side(box: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = box
    hits = []
    if r0 == 0:
        hits.append("top")
    if r1 == h - 1:
        hits.append("bottom")
    if c0 == 0:
        hits.append("left")
    if c1 == w - 1:
        hits.append("right")
    return hits[0] if len(hits) == 1 else None


def _inward_gap_depth(g: np.ndarray, box: tuple[int, int, int, int], side: str, bg: int) -> int:
    """Count up to 2 fully-background strips immediately behind an edge-touching bbox."""
    h, w = g.shape
    r0, c0, r1, c1 = box
    depth = 0

    if side == "top":
        rr = r1 + 1
        while rr < h and depth < 2 and np.all(g[rr, c0 : c1 + 1] == bg):
            depth += 1
            rr += 1
        return depth
    if side == "bottom":
        rr = r0 - 1
        while rr >= 0 and depth < 2 and np.all(g[rr, c0 : c1 + 1] == bg):
            depth += 1
            rr -= 1
        return depth
    if side == "left":
        cc = c1 + 1
        while cc < w and depth < 2 and np.all(g[r0 : r1 + 1, cc] == bg):
            depth += 1
            cc += 1
        return depth
    cc = c0 - 1
    while cc >= 0 and depth < 2 and np.all(g[r0 : r1 + 1, cc] == bg):
        depth += 1
        cc -= 1
    return depth


def _find_legend_component(g: np.ndarray, bg: int) -> tuple[int, set[int]]:
    """
    Identify the unique legend component:
    - touches exactly one edge
    - consists of exactly 3 cells with 3 distinct colors
    - fits in a 2x2 bbox
    - has a 1-2 cell inward background moat
    """
    h, w = g.shape
    candidates: list[tuple[int, int, set[int]]] = []
    comps = components_mask(g != bg, connectivity=4)
    for idx, pts in enumerate(comps):
        if len(pts) != 3:
            continue
        box = _bbox(pts)
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 > 2 or c1 - c0 + 1 > 2:
            continue
        side = _touching_side(box, h, w)
        if side is None:
            continue
        colors = {int(g[r, c]) for r, c in pts}
        if bg in colors or len(colors) != 3:
            continue
        gap = _inward_gap_depth(g, box, side, bg)
        if gap >= 1:
            candidates.append((gap, idx, colors))

    assert len(candidates) == 1, "legend detection failed or was ambiguous"
    _gap, idx, colors = candidates[0]
    return idx, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove every non-legend object that contains all colors shown by a small edge legend.

    Input structure:
    - Background color fills most of the board.
    - One grid edge contains a tiny 3-cell legend component with no explicit border.
    - That legend sits against the border and is buffered from the larger objects by 1-2 rows or
      columns of background cells on the inward side.
    - Elsewhere, larger 4-connected multicolor components occupy rectangular regions.

    Rule:
    - Locate the legend by scanning edge-touching mini-components and selecting the one backed by
      an inward background gap.
    - Read the legend's three colors from that component.
    - For every other non-background connected component, erase it iff it contains all three legend colors.
    - Leave the legend itself and every incomplete component unchanged.

    Key insight:
    - The borderless legend is discoverable through spatial structure rather than a marker color:
      it is the only tiny component attached to an edge and separated from the main scene by empty space.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend_idx, legend_colors = _find_legend_component(g, bg)

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for idx, pts in enumerate(comps):
        if idx == legend_idx:
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

LEGEND_GAPS = [1, 2]
LEGEND_BASE = np.array([[1, 1], [1, 0]], dtype=int)
SIDES = ["top", "bottom", "left", "right"]

SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (6, 10)
TALL_W = (2, 3)
WIDE_H = (3, 4)
WIDE_W = (4, 7)
MED_H = (3, 6)
MED_W = (2, 6)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5]
COMP_WEIGHTS = [3, 2]
NUM_KEPT = 3

MAX_RETRIES = 1600


def _legend_mask(side: str) -> np.ndarray:
    if side == "top":
        return LEGEND_BASE.astype(bool)
    if side == "right":
        return np.rot90(LEGEND_BASE, -1).astype(bool)
    if side == "bottom":
        return np.rot90(LEGEND_BASE, 2).astype(bool)
    return np.rot90(LEGEND_BASE, 1).astype(bool)


def _legend_position(side: str) -> tuple[int, int]:
    if side in {"top", "bottom"}:
        c0 = random.randint(1, GRID_W - 3)
        r0 = 0 if side == "top" else GRID_H - 2
        return r0, c0
    r0 = random.randint(1, GRID_H - 3)
    c0 = 0 if side == "left" else GRID_W - 2
    return r0, c0


def _place_legend(
    inp: np.ndarray, out: np.ndarray, blocked: np.ndarray, targets: list[int]
) -> tuple[str, int]:
    side = random.choice(SIDES)
    gap = random.choice(LEGEND_GAPS)
    r0, c0 = _legend_position(side)
    mask = _legend_mask(side)

    patch = np.full(mask.shape, BG, dtype=int)
    colors = targets[:]
    random.shuffle(colors)
    coords = list(zip(*np.where(mask)))
    for color, (rr, cc) in zip(colors, coords):
        patch[rr, cc] = color

    # Reserve the legend's full 2x2 bbox, not only its colored cells, so the inward
    # gap is visibly aligned with the whole legend footprint.
    occ_patch = np.zeros_like(patch)
    paste(inp, patch, r0, c0, transparent=BG)
    paste(out, patch, r0, c0, transparent=BG)
    paste(blocked, occ_patch, r0, c0, transparent=BG)

    if gap == 2:
        if side == "top":
            extra = np.zeros((1, patch.shape[1]), dtype=int)
            paste(blocked, extra, r0 + patch.shape[0], c0)
        elif side == "bottom":
            extra = np.zeros((1, patch.shape[1]), dtype=int)
            paste(blocked, extra, r0 - 1, c0)
        elif side == "left":
            extra = np.zeros((patch.shape[0], 1), dtype=int)
            paste(blocked, extra, r0, c0 + patch.shape[1])
        else:
            extra = np.zeros((patch.shape[0], 1), dtype=int)
            paste(blocked, extra, r0, c0 - 1)

    return side, gap


def _component_dims() -> tuple[int, int]:
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
        colors = list(targets) + [random.choice(PALETTE) for _ in range(n_colored - NUM_TARGETS)]
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        colors = [random.choice(allowed) for _ in range(n_colored)]

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()
    blocked = np.full_like(inp, BG)

    targets = random.sample(PALETTE, NUM_TARGETS)
    _side, _gap = _place_legend(inp, out, blocked, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted)
        occ_patch = np.zeros_like(patch)
        try:
            r, c = random_free_location(
                blocked, occ_patch, transparent=BG, border=1, padding=1, padding_connectivity=4
            )
        except ValueError:
            return None
        paste(blocked, occ_patch, r, c)
        paste(inp, patch, r, c)
        if not deleted:
            paste(out, patch, r, c)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
