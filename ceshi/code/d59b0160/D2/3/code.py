# Variant of d59b0160
# Dimension: D2 — 核心变换操作：删除 vs 其他处理 (transform)
# Change: 匹配的连通分量被删除（所有像素替换为背景色7）... → 匹配分量进行镜像翻转

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location

# Design analysis for the D2 variant
# ----------------------------------
# source_task_baseline:
#   reference_task: 7ed72f31
#   borrowed_idea:
#     - A visible reflection should happen in-place rather than deleting pixels.
#     - The reflection mechanism should stay simple and local so the rule is readable from a few examples.
#   complexity_choice:
#     - We keep the original d59b0160 selection rule unchanged.
#     - We use only one fixed transform: horizontal flip inside the matched component's own bbox.
#
# variant_spec:
#   unchanged_structure:
#     - grid is 16x16 with bg=7
#     - legend is a 4x4 top-left region with border color 3 on bottom row and right column
#     - legend interior still encodes exactly 3 target colors
#     - non-legend objects are still 4-connected multicolor components on bg
#   changed_behavior:
#     - original: if a component contains all legend colors, replace the whole component with bg
#     - variant: if a component contains all legend colors, isolate that component inside its bbox,
#       mirror the bbox left-right, and write the mirrored patch back in the same place
#     - unmatched components remain unchanged
#
# behavior_trace:
#   example:
#     - Suppose a matched component occupies rows 5..8 and cols 10..13.
#     - Original logic: once the component color set contains every legend color, every cell in that
#       component becomes 7, so the whole rectangle disappears.
#     - Variant logic: the same component is copied into a local bbox patch and that patch is reversed
#       along columns. Colors remain the same, but their left/right arrangement swaps.
#   difference:
#     - Selection is still palette-based.
#     - Only the action after selection changes, from erase to reflection.
#
# complexity_calibration:
#   risk_if_unchanged:
#     - Random sprinkled rectangles can be accidentally mirror-symmetric, making the new transform
#       visually indistinguishable from "do nothing".
#   fix:
#     - The generator forces every matched component to be horizontally asymmetric.
#     - It also requires at least one visibly changing matched component in every sample; in practice
#       all matched components are forced to change.
#   controlled_freedom:
#     - Only one transform exists, always horizontal, never vertical and never rotation.
#     - Match count remains 1..3 as in the original deleted-component count.
#
# implicit_assumption_audit:
#   _bg_boundary:
#     - assumes the outer frame is mostly bg; still valid because output transform never changes bbox size
#   _find_legend:
#     - assumes the top-left legend format is unchanged; generator preserves it exactly
#   components_mask(g != bg):
#     - assumes all non-bg object cells define the components; still valid
#   bbox reflection:
#     - new solver assumes that cells inside a matched component bbox but outside the component are bg
#       or belong to the same component-shaped hole pattern
#     - generator guarantees stronger conditions: every object is a solid rectangle patch, and placement
#       uses padding=1 so no other component can enter that bbox
#
# placement_legality:
#   - Each component is placed by random_free_location on the input canvas with 1-cell 8-neighbor padding.
#   - Because mirrored output stays inside the same bbox, any valid input placement remains a valid output.
#   - Legend placement remains fixed and separated from objects by the same occupancy check as before.
#
# visual_observability:
#   - The visible cue is not based on hidden color ids or ordering.
#   - Matching objects keep all colors but reverse their left/right internal arrangement.
#   - Kept objects do not move, so the contrast between matched and unmatched objects is directly visible.
#
# visual_cleanliness:
#   - Objects remain disjoint rectangles with bg gaps.
#   - The legend remains compact in the top-left and is never overwritten.
#   - No extra markers or overlays are introduced.
#
# new_main_assumptions_and_generator_guarantees:
#   - main flips an entire matched-component bbox.
#   - generate_input guarantees matched components have width >= 3 and are not equal to their horizontal
#     mirror, so the transform is observable.
#   - generate_input also guarantees kept components miss at least one legend color, preserving the
#     original selection logic.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend color set is the set of colors in the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    assert False, "no legend border found in grid"


def _component_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    rs = pts[:, 0]
    cs = pts[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _mirror_component_in_bbox(out: np.ndarray, g: np.ndarray, pts: np.ndarray, bg: int) -> None:
    r0, c0, r1, c1 = _component_bbox(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in pts:
        patch[int(r) - r0, int(c) - c0] = int(g[int(r), int(c)])
    paste(out, np.fliplr(patch), r0, c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Horizontally reflect selected connected components inside their own bounding boxes.

    Input structure:
    - The outer frame determines the background color.
    - The top-left corner contains a square legend whose bottom edge and right edge share one border color.
    - Outside that legend, the non-background cells form 4-connected multicolor components on the background.

    Rule:
    - Read the target color set from the legend interior.
    - For each non-background 4-connected component, test whether its color set contains every target color.
    - If it does, isolate that component inside its bounding rectangle and mirror that rectangle left-to-right in place.
    - If it does not, leave the component unchanged.

    Key insight: The legend still acts as a palette filter, but the matched objects are transformed rather than erased, so the important signal is which components keep their colors while reversing their internal left-right pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend = _find_legend(g, bg)
    _sz, _x, legend_colors = legend
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[int(r), int(c)]) for r, c in pts}
        if legend_colors.issubset(cols):
            _mirror_component_in_bbox(out, g, pts, bg)
    return out


# --------------- generator ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of colored pixels among the base-0 rectangle) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/flipped split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Variant observability controls --
PATCH_RETRIES = 80
MIN_FLIP_DIFF = 2

# -- Generation retry budget --
MAX_RETRIES = 1000


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box (border color 3) at top-left and place target colors."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims(flipped: bool) -> tuple[int, int]:
    """Sample component (h, w) according to shape-mode weights."""
    while True:
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if not flipped or w >= 3:
            return h, w


def _random_patch(h: int, w: int, colors: list[int], n_colored: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    fill = [random.choice(colors) for _ in range(n_colored)]
    random.shuffle(fill)
    rs, cs = zip(*coords)
    patch[rs, cs] = fill
    return patch


def _flip_diff(patch: np.ndarray) -> int:
    return int(np.count_nonzero(patch != patch[:, ::-1]))


def _component_patch(h: int, w: int, targets: list[int], flipped: bool) -> np.ndarray | None:
    """Build an h x w rectangle with base color 0 and sprinkled colored pixels."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    non_targets = [v for v in PALETTE if v not in targets]

    for _ in range(PATCH_RETRIES):
        if flipped:
            lo_here = max(lo, NUM_TARGETS)
            n_colored = random.randint(lo_here, hi)
            patch = _random_patch(h, w, PALETTE, n_colored)

            cols = {int(v) for v in np.unique(patch)}
            if not set(targets).issubset(cols):
                missing = [c for c in targets if c not in cols]
                zero_coords = [(r, c) for r in range(h) for c in range(w) if patch[r, c] == 0]
                if len(zero_coords) < len(missing):
                    continue
                for color, (r, c) in zip(missing, random.sample(zero_coords, len(missing))):
                    patch[r, c] = color
            if _flip_diff(patch) < MIN_FLIP_DIFF:
                continue
            return patch

        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed = non_targets + random.sample(targets, used_target_count)
        n_colored = random.randint(lo, hi)
        patch = _random_patch(h, w, allowed, n_colored)
        cols = {int(v) for v in np.unique(patch)}
        if set(targets).issubset(cols):
            continue
        return patch

    return None


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output is determined by main()."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    changed_flipped = 0
    for flipped in roles:
        h, w = _component_dims(flipped)
        patch = _component_patch(h, w, targets, flipped)
        if patch is None:
            return None
        try:
            r, c = random_free_location(
                inp, np.zeros_like(patch), transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        if flipped:
            mirrored = patch[:, ::-1]
            paste(out, mirrored, r, c)
            if not np.array_equal(patch, mirrored):
                changed_flipped += 1
        else:
            paste(out, patch, r, c)

    if changed_flipped == 0:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
