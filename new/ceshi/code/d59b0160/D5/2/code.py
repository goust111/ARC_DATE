# Variant of d59b0160
# Dimension: D5 — 图例颜色集合的提取方式 (legend_lookup)
# Change: 图例区域内所有非背景(7)、非边框(3)颜色的集合作为目标颜色集... → 图例中的颜色排列定义有序序列

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis:
# spec_variant:
#   core_change:
#     - The legend no longer specifies an unordered color set.
#     - Read the 3x3 legend interior in row-major order and keep the non-bg/non-border cells
#       as an ordered 3-color sequence.
#     - A component is erased only when it contains all legend colors and the first time those
#       colors appear in the component's own bbox row-major scan matches the legend sequence.
#   concrete_visualization:
#     - The 4x4 top-left legend keeps the same bottom/right border color 3 and bg 7.
#     - The three legend colors sit in one of a few non-4-adjacent templates so the legend
#       itself never collapses into one deletable component.
#     - Outside the legend, objects remain solid rectangles of color 0 with sprinkled colors.
#       Deleted objects contain the full legend sequence in matching scan order; kept objects
#       either miss at least one legend color or contain all three in a different order.
#   behavior_trace:
#     - Original rule: legend {2,5,8}, component first-seen order [5,2,8] was still deleted
#       because set inclusion ignored order.
#     - Variant rule: the same component is kept; only components whose first-seen order is
#       exactly [2,5,8] are deleted.
#   complexity_calibration:
#     - Only 3 legend colors are used, so the hypothesis space stays small.
#     - Every sample contains at least one "full palette but wrong order" kept component so the
#       new branch is visible and the task does not degenerate to the original set-based rule.
#     - The legend uses a small template family rather than arbitrary placements, giving repeated
#       support across examples without fixing everything to a single picture.
#   implicit_assumptions_review:
#     - Original `_find_legend(...)` returned a set; that loses order and must become a sequence.
#     - Original `issubset` deletion check is no longer sufficient; component scan order must be
#       computed explicitly.
#     - Extra copies of target colors would make first-occurrence order unstable, so the generator
#       only places each target color once inside each non-legend component.
#     - The solver still processes all non-bg connected components, so the generator must keep the
#       legend colors split across legend subcomponents; otherwise the legend itself could match and
#       be erased. The fixed legend templates guarantee that.
#   placement_legality:
#     - Components are still pasted with `random_free_location(..., padding=1)` so they do not
#       overlap, do not touch each other, and stay separated from the legend.
#     - Components also stay one cell away from the outer frame, preserving the original
#       frame-based background detection assumption used by `_bg_boundary`.
#     - Anchor positions for ordered colors are chosen inside the component bbox, so all forced
#       sequence cells are in-bounds even for the smallest allowed rectangles.
#   visual_observability:
#     - A human can now compare two components that both contain the legend colors and see that one
#       disappears only when those colors appear in the same reading order as the legend.
#     - The rule uses visible reading order only; it never relies on color ids or hidden numeric
#       sorting.
#   cleanliness_and_color_inference:
#     - Non-target sprinkles use only non-legend colors, avoiding ambiguous extra target markers.
#     - The base rectangle remains color 0, so each object is still a single clean 4-connected
#       block and the ordered target cells are easy to inspect inside it.
#   generator_guarantees_for_main:
#     - Legend sequence length is always 3.
#     - Every deleted component has all 3 legend colors in matching order.
#     - Every kept component is either missing a legend color or has a non-matching full order.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, list[int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend sequence is read from the interior (0:sz-1, 0:sz-1) in row-major order,
    keeping only non-bg and non-border cells.
    """
    h, w = g.shape
    sizes = [LEGEND_SIZE] + [sz for sz in range(2, min(h, w) + 1) if sz != LEGEND_SIZE]
    for sz in sizes:
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            seq: list[int] = []
            for r in range(inside.shape[0]):
                for c in range(inside.shape[1]):
                    color = int(inside[r, c])
                    if color not in (bg, x):
                        seq.append(color)
            return int(sz), int(x), seq
    assert False, "no legend border found in grid"


def _component_sequence(g: np.ndarray, pts: np.ndarray, targets: set[int]) -> list[int]:
    """Row-major first-occurrence sequence of target colors inside the component bbox."""
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    local = np.zeros((h, w), dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        rr = int(r - r0)
        cc = int(c - c0)
        local[rr, cc] = int(g[r, c])
        mask[rr, cc] = True

    seq: list[int] = []
    seen: set[int] = set()
    for rr in range(h):
        for cc in range(w):
            if not mask[rr, cc]:
                continue
            color = int(local[rr, cc])
            if color in targets and color not in seen:
                seen.add(color)
                seq.append(color)
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose ordered target sequence matches the legend.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a small square legend region bordered on its bottom and right edges by a
      single color; the interior places three target colors whose reading order defines a sequence.
    - Elsewhere: multicolored 4-connected objects on the bg.

    Rule:
    - Read the legend sequence by scanning the legend interior from top to bottom, left to right,
      skipping bg and border cells.
    - For each connected component, require two things: it must contain every legend color, and the
      first time those legend colors appear in the component's bbox scan must follow the same order.
    - Erase (fill with bg) every component satisfying both conditions; leave all others unchanged.
    - Output grid is the same size.

    Key insight: The legend is now an ordered filter rather than an unordered palette, so color
    presence alone is not enough; spatial reading order decides which full-palette components vanish.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_seq = _find_legend(g, bg)
    assert legend_seq, "legend interior contains no target colors"
    target_set = set(legend_seq)
    assert len(legend_seq) == len(target_set), "legend sequence must not repeat colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if not target_set.issubset(cols):
            continue
        if _component_sequence(g, pts, target_set) == legend_seq:
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (o5x) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

LEGEND_TEMPLATES = [
    [(0, 0), (1, 2), (2, 1)],
    [(0, 1), (1, 0), (2, 2)],
    [(0, 2), (1, 0), (2, 1)],
]

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [6, 2, 2]
TALL_H = (7, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of colored non-zero pixels inside each rectangle) --
DENSITY_LO = 0.10
DENSITY_HI = 0.45

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [2, 5, 10]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box and place the ordered target sequence in a safe template."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    positions = random.choice(LEGEND_TEMPLATES)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample component (h, w) according to shape-mode weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_anchor_templates(h: int, w: int) -> list[list[tuple[int, int]]]:
    """Three increasing row-major anchor templates inside an h x w rectangle."""
    mid_r = h // 2
    mid_c = w // 2
    return [
        [(0, 0), (mid_r, w - 1), (h - 1, mid_c)],
        [(0, mid_c), (mid_r, 0), (h - 1, w - 1)],
        [(0, w - 1), (mid_r, mid_c), (h - 1, 0)],
    ]


def _wrong_permutation(targets: list[int]) -> list[int]:
    """Random non-identity permutation of the target sequence."""
    perms = [
        [targets[0], targets[2], targets[1]],
        [targets[1], targets[0], targets[2]],
        [targets[1], targets[2], targets[0]],
        [targets[2], targets[0], targets[1]],
        [targets[2], targets[1], targets[0]],
    ]
    return random.choice(perms)


def _ordered_subset(targets: list[int]) -> list[int]:
    """Choose a proper ordered subsequence of the targets."""
    k = random.randint(0, NUM_TARGETS - 1)
    if k == 0:
        return []
    indices = sorted(random.sample(range(NUM_TARGETS), k))
    return [targets[i] for i in indices]


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """
    Build an h x w rectangle with base color 0 plus colored markers.

    Roles:
      - delete: full legend sequence in matching scan order
      - permute: full legend colors in a different order
      - missing: only a proper subset of legend colors
    """
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))

    anchors = random.choice(_component_anchor_templates(h, w))
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "delete":
        forced_colors = list(targets)
    elif role == "permute":
        forced_colors = _wrong_permutation(targets)
    else:
        forced_colors = _ordered_subset(targets)

    lo = max(lo, len(forced_colors))
    hi = max(hi, lo)
    n_colored = random.randint(lo, hi)
    extra_count = n_colored - len(forced_colors)

    patch = np.zeros((h, w), dtype=int)
    used_coords: set[tuple[int, int]] = set()

    for color, (r, c) in zip(forced_colors, anchors):
        patch[r, c] = color
        used_coords.add((r, c))

    if extra_count > 0:
        free_coords = [(r, c) for r in range(h) for c in range(w) if (r, c) not in used_coords]
        extra_coords = random.sample(free_coords, extra_count)
        for r, c in extra_coords:
            patch[r, c] = random.choice(non_targets)

    return patch


def _component_roles() -> list[str]:
    """Build one role list with exactly 3 kept components and at least one wrong-order full match."""
    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    kept_roles = ["permute", "missing", random.choice(["permute", "missing"])]
    roles = ["delete"] * deleted_count + kept_roles
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose expected output matches `main`."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    for role in _component_roles():
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, role)
        try:
            r, c = random_free_location(
                inp, np.zeros_like(patch), transparent=BG, border=1, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if role != "delete":
            paste(out, patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
