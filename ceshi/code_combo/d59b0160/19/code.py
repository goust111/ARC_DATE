# Combo variant of d59b0160 (index 19)
# combo_spec: D5/2+D16/1
# Name: 图例颜色有序序列+空间模式匹配

import math
import random
from itertools import product

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis:
# fusion_variant:
#   core_rule:
#     - The legend no longer defines an unordered palette. Its three colored cells are read in
#       row-major order to form an ordered sequence.
#     - A non-legend component is deleted only if those same three colors can be located inside the
#       component in that same row-major color order and with the same relative row/column pattern
#       as the legend. This upgrades the old set test into a joint order+layout match.
#   concrete_visualization:
#     - The legend stays a 4x4 top-left frame with border color 3 and bg 7. Its 3x3 interior uses
#       one of a small family of three-pixel templates. Each template is visibly different:
#       diagonal, bent-right, bent-left, etc.
#     - Each large object is still a solid color-0 rectangle with a few colored markers. The three
#       legend colors appear at most once per object, so the intended triplet is visually clean.
#     - Deleted objects reproduce the legend template at a larger scale. Kept objects include:
#       one full-color object with the right scan order but the wrong template, one full-color
#       object with the right template but a wrong color order, and one object missing a target.
#   behavior_trace:
#     - Example legend scan order: [2, 5, 8], template positions [(0,0), (1,2), (2,1)].
#     - Original task: any component containing {2,5,8} vanished, even if 2/5/8 formed a different
#       triangle.
#     - D5/2-only task: a component still vanished if its first-seen order was [2,5,8], even when
#       those pixels sat in a different relative layout.
#     - Fusion task here: that same full-color same-order component is kept unless the positions of
#       2/5/8 also realize the same relative row/column pattern as the legend.
#   complexity_calibration:
#     - Only 4 legend template signatures are used. That is enough to make the spatial rule visible
#       without exploding the hypothesis space.
#     - Every sample contains exactly one kept "wrong_shape" component, so the new D16-style branch
#       is always exercised and the dataset cannot collapse back to D5/2 or the original set rule.
#     - Every sample also contains one kept "wrong_order" component, so the ordered-sequence branch
#       remains visually supported instead of being implied only indirectly.
#   implicit_assumptions_review:
#     - The original `_find_legend(...)` returned a set; that discards both order and geometry and
#       must become a sequence plus pixel positions.
#     - The original `issubset` test assumed color presence alone determined deletion. After the
#       fusion, solver logic must explicitly check (1) row-major target order and (2) relative
#       pattern signature.
#     - A new solver assumption is that target-color matches are inspectable as distinct points.
#       The generator guarantees this by placing each target color at most once per non-legend
#       component and using only non-target extras elsewhere.
#     - The solver still examines every non-bg 4-component, including the legend border pieces, so
#       legend templates are chosen so that no legend-side component can contain all three targets.
#   placement_legality:
#     - Objects are pasted as full rectangles of occupancy using `random_free_location(..., border=1,
#       padding=1, padding_connectivity=8)`, so they stay in-bounds, separated from each other, and
#       one cell away from the outer frame and legend.
#     - Component widths are tightened to >=3 so every template with 3 column ranks can be scaled
#       into the object bbox without collisions or degenerate columns.
#     - Forced target anchors are always chosen inside the rectangle bbox; even the smallest allowed
#       3x3 component can realize any template.
#   visual_observability:
#     - A human can compare two full-color objects and see that one disappears only when both the
#       color readout and the geometric arrangement echo the legend.
#     - The rule uses only visible row/column relationships; it never relies on hidden facts such as
#       color id ordering or tie-breaking by numeric values.
#   cleanliness_and_inference:
#     - Non-target sprinkles use only non-legend colors, so target triplets are easy to isolate.
#     - The color-0 rectangle remains the object carrier, preserving the original "single connected
#       block with markers inside" look instead of introducing cluttered multi-object interactions.
#   generator_guarantees_for_main:
#     - Legend length is always 3 with distinct colors.
#     - Deleted components have the exact legend color order and legend pattern.
#     - Each sample includes at least one full-color kept component that matches order but not
#       pattern, proving the fusion behavior differs from D5/2.
#     - Each sample includes at least one full-color kept component that matches pattern but not
#       color order, proving order remains active after adding the spatial constraint.


# --------------- solver ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# Row-major-safe legend templates inside the 3x3 interior.
LEGEND_TEMPLATES = [
    [(0, 0), (1, 2), (2, 1)],
    [(0, 1), (1, 0), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 1), (1, 2), (2, 0)],
]


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, list[int], list[tuple[int, int]]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend sequence is read from the interior (0:sz-1, 0:sz-1) in row-major order,
    keeping only non-bg and non-border cells, and we also record their interior coordinates.
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
            pos: list[tuple[int, int]] = []
            for r in range(inside.shape[0]):
                for c in range(inside.shape[1]):
                    color = int(inside[r, c])
                    if color not in (bg, x):
                        seq.append(color)
                        pos.append((int(r), int(c)))
            return int(sz), int(x), seq, pos
    assert False, "no legend border found in grid"


def _compress_ranks(values: list[int]) -> tuple[int, ...]:
    """Map sorted unique values to 0..k-1 while preserving equality patterns."""
    uniq = {v: i for i, v in enumerate(sorted(set(values)))}
    return tuple(int(uniq[v]) for v in values)


def _pattern_signature(pos: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Translation- and scale-invariant row/column rank signature of ordered points."""
    rows = [int(r) for r, _c in pos]
    cols = [int(c) for _r, c in pos]
    return _compress_ranks(rows), _compress_ranks(cols)


def _row_major_color_order(colors: list[int], pos: tuple[tuple[int, int], ...]) -> list[int]:
    """Colors ordered by the row-major order of their associated positions."""
    pairs = sorted(zip(pos, colors), key=lambda item: (item[0][0], item[0][1]))
    return [int(color) for (_p, color) in pairs]


def _component_target_positions(
    g: np.ndarray, pts: np.ndarray, targets: set[int]
) -> dict[int, list[tuple[int, int]]]:
    """Absolute positions of each target color inside a component."""
    found = {int(color): [] for color in targets}
    for r, c in pts:
        color = int(g[r, c])
        if color in found:
            found[color].append((int(r), int(c)))
    return found


def _component_matches_pattern(
    g: np.ndarray,
    pts: np.ndarray,
    legend_seq: list[int],
    legend_sig: tuple[tuple[int, ...], tuple[int, ...]],
) -> bool:
    """
    Return True iff the component contains one occurrence of each legend color whose
    row-major order and relative pattern both match the legend.
    """
    positions_by_color = _component_target_positions(g, pts, set(legend_seq))
    if any(not positions_by_color[color] for color in legend_seq):
        return False

    for choice in product(*(positions_by_color[color] for color in legend_seq)):
        if len(set(choice)) != len(choice):
            continue
        if _row_major_color_order(legend_seq, choice) != legend_seq:
            continue
        if _pattern_signature(choice) == legend_sig:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose ordered target layout matches the legend.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a small square legend region bordered on its bottom and right edges by a
      single color; the interior places three target colors, and their row-major scan order is the
      reference color sequence.
    - Elsewhere: multicolored 4-connected objects on the bg, usually color-0 rectangles carrying a
      few colored marker pixels.

    Rule:
    - Read the legend interior from top to bottom, left to right, skipping bg and border cells.
      This yields both an ordered 3-color sequence and a 3-point spatial template.
    - For each connected component, look for one occurrence of each legend color. A component is
      erased only when those three chosen pixels appear in the same row-major color order and have
      the same relative row/column arrangement as the legend template.
    - Components missing a legend color, having the colors in another order, or arranging them in a
      different pattern are kept unchanged.
    - Output grid size stays the same.

    Key insight: The legend is now a tiny ordered geometric stencil. Matching the palette is not
    enough; the component must reproduce both the legend's color sequence and its spatial pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_seq, legend_pos = _find_legend(g, bg)
    assert len(legend_seq) == NUM_TARGETS, "legend must contain exactly three target cells"
    assert len(set(legend_seq)) == NUM_TARGETS, "legend sequence must not repeat colors"
    legend_sig = _pattern_signature(legend_pos)

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if not set(legend_seq).issubset(cols):
            continue
        if _component_matches_pattern(g, pts, legend_seq, legend_sig):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (o5x-combo) ---------------

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [6, 2, 2]  # medium, tall, wide
TALL_H = (6, 10)
TALL_W = (3, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 6)
MED_W = (3, 7)

# -- Component fill density (fraction of colored non-zero pixels inside each rectangle) --
DENSITY_LO = 0.10
DENSITY_HI = 0.40

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [2, 5, 10]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 500


def _place_legend(
    inp: np.ndarray, out: np.ndarray, targets: list[int], template: list[tuple[int, int]]
) -> None:
    """Draw the 4x4 legend box and place the target sequence on the chosen template."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    for color, (r, c) in zip(targets, template):
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


def _axis_anchors(unique_count: int, span: int) -> list[int]:
    """Canonical anchor coordinates for 1/2/3 ordered ranks along one axis."""
    if unique_count == 1:
        return [span // 2]
    if unique_count == 2:
        return [0, span - 1]
    assert unique_count == 3
    return [0, span // 2, span - 1]


def _scaled_positions(template: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    """Scale a 3x3 legend template into an h x w component using only rank structure."""
    row_vals = sorted({int(r) for r, _c in template})
    col_vals = sorted({int(c) for _r, c in template})
    row_map = dict(zip(row_vals, _axis_anchors(len(row_vals), h)))
    col_map = dict(zip(col_vals, _axis_anchors(len(col_vals), w)))
    return [(int(row_map[r]), int(col_map[c])) for r, c in template]


def _wrong_template(correct_template: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pick a different legend template signature for the kept wrong-shape component."""
    correct_sig = _pattern_signature(correct_template)
    choices = [tpl for tpl in LEGEND_TEMPLATES if _pattern_signature(tpl) != correct_sig]
    return random.choice(choices)


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
    """Pick a proper ordered subsequence of the target colors."""
    k = random.randint(0, NUM_TARGETS - 1)
    if k == 0:
        return []
    return list(targets[:k])


def _component_patch(
    h: int, w: int, targets: list[int], legend_template: list[tuple[int, int]], role: str
) -> np.ndarray:
    """
    Build an h x w rectangle with base color 0 plus colored markers.

    Roles:
      - delete: full legend sequence on the matching spatial template
      - wrong_shape: same legend sequence, different spatial template
      - wrong_order: same spatial template, wrong color order
      - missing: proper subsequence only
    """
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "delete":
        anchors = _scaled_positions(legend_template, h, w)
        forced_colors = list(targets)
    elif role == "wrong_shape":
        anchors = _scaled_positions(_wrong_template(legend_template), h, w)
        forced_colors = list(targets)
    elif role == "wrong_order":
        anchors = _scaled_positions(legend_template, h, w)
        forced_colors = _wrong_permutation(targets)
    else:
        anchors = _scaled_positions(legend_template, h, w)
        forced_colors = _ordered_subset(targets)
        anchors = anchors[: len(forced_colors)]

    lo = max(lo, len(forced_colors))
    n_colored = random.randint(lo, hi)
    extra_count = n_colored - len(forced_colors)

    patch = np.zeros((h, w), dtype=int)
    used_coords: set[tuple[int, int]] = set()
    for color, (r, c) in zip(forced_colors, anchors):
        patch[r, c] = int(color)
        used_coords.add((int(r), int(c)))

    if extra_count > 0:
        free_coords = [(r, c) for r in range(h) for c in range(w) if (r, c) not in used_coords]
        extra_coords = random.sample(free_coords, extra_count)
        for r, c in extra_coords:
            patch[r, c] = random.choice(non_targets)

    return patch


def _component_roles() -> list[str]:
    """Build one role list with exactly 3 kept components covering both fusion branches."""
    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    roles = ["delete"] * deleted_count + ["wrong_shape", "wrong_order", "missing"]
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose expected output matches `main`."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    legend_template = random.choice(LEGEND_TEMPLATES)
    _place_legend(inp, out, targets, legend_template)

    for role in _component_roles():
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, legend_template, role)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
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
