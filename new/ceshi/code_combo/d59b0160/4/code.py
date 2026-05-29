# Combo variant of d59b0160 (index 4)
# combo_spec: D2/0+D13/0
# Name: 匹配分量重着色+保留分量统一着色

"""
Design analysis for the combo variant
=====================================

combo_rule:
  preserved_structure:
    - Grid stays 16x16 with outer-border background color 7.
    - The legend stays in the top-left 4x4 square; its bottom row and right column are border color 3.
    - The legend interior still shows exactly 3 distinct target colors from {1,2,4,5,6,8,9}.
    - Outside the legend are separated 4-connected rectangular components made entirely of non-bg colors.
  fused_change:
    - Matching components are no longer deleted; they are recolored to the legend border color 3.
    - Non-matching components are no longer copied verbatim; they are flattened to their own most frequent color.
    - Therefore every ordinary component changes, but the two branches use different recolor sources.

concretization_and_trace:
  example:
    - legend colors = {1,4,8}, border color = 3
    - component A colors/frequencies = {0:9, 1:2, 4:1, 8:1}; A matches because it contains 1,4,8
    - component B colors/frequencies = {6:10, 0:3, 4:1}; B does not match because it misses 1 and 8
  original_task:
    - A would be erased to background 7
    - B would be copied unchanged
  fused_variant:
    - A keeps the same footprint but every cell becomes 3
    - B keeps the same footprint but every cell becomes 6, its unique majority color
  visible_difference:
    - Matched objects become solid green blocks tied to the legend frame.
    - Unmatched objects also lose texture, but their output color comes from inside the object itself.

complexity_calibration:
  - Keep one stable selector: "contains all legend colors".
  - Use only two recolor targets:
      1. fixed border color for matching components
      2. per-component dominant color for non-matching components
  - Force 1-3 matching components and exactly 3 non-matching components so every sample exhibits both branches.
  - Force non-matching components to be multicolor with a unique majority; otherwise the D13 branch could become invisible or depend on tie-break lore.
  - Exclude color 3 from ordinary input components so the matched recolor branch remains visually explicit.

implicit_assumption_audit:
  - `_bg_boundary` still assumes the outer frame is dominated by background 7; generator preserves that.
  - `_find_legend` still assumes the first qualifying top-left square is the legend; generator keeps the same legend geometry.
  - `components_mask(g != bg)` assumes all object cells are non-bg and each object is 4-connected; generator uses solid rectangles with non-bg fills.
  - Recoloring all non-matching components would incorrectly recolor the legend unless the legend square is skipped explicitly.
  - Skipping any component fully inside the legend square assumes no ordinary object touches the legend; generator enforces a 1-cell padded gap.
  - `_mode_color` would be ambiguous on ties; generator guarantees a unique dominant color for every non-matching component.
  - The matching branch assumes border color is the intended recolor target; generator fixes the frame color at 3 and keeps 3 absent from ordinary inputs.

placement_legality:
  - Components are placed with `random_free_location(..., padding=1, padding_connectivity=8)`.
  - Because each patch is fully non-bg, placement prevents overlap and also prevents contact with the legend and with other components.
  - Dimension ranges start at area 9, which is enough to realize "all 3 target colors present" and "unique majority color" simultaneously.

visual_observability:
  - In 2-4 examples a solver can see that every ordinary object becomes monochrome.
  - The reason for the monochrome color splits cleanly:
      - objects containing the full legend palette turn into the legend frame color
      - the others turn into their own obvious dominant color
  - This avoids hidden numeric ordering: the border supplies the fixed recolor cue, and the object texture supplies the self-color cue.

cleanliness_and_inference:
  - Output grids stay clean because only whole-component recolors happen; no overlays or extra markers are added.
  - Object spacing remains at least one background cell, so the before/after comparison is easy to parse.
  - The generator makes each surviving branch structurally repeated across samples, which gives enough evidence for humans to infer the fusion instead of mistaking it for arbitrary recoloring.
"""

import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the smallest top-left square whose bottom edge and right edge share one color.
    Its interior colors, excluding bg and the border color, define the target set.
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


def _mode_color(values: np.ndarray) -> int:
    """Return the most frequent color in `values`; ties fall to the smallest color."""
    vals, cnts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every ordinary component, using one of two different monochrome targets.

    Input structure:
    - Background fills most of the board.
    - A compact top-left legend is framed on its bottom and right edges by one border color; the interior lists the target palette.
    - Elsewhere are separated 4-connected multicolor components made of non-background cells.

    Rule:
    - Read the target color set from the legend interior and the fixed replacement color from the legend border.
    - For each non-background connected component outside the legend square, test whether it contains **all** legend colors.
    - If it does, repaint the whole component to the border color; otherwise repaint the whole component to its own most frequent color.
    - Keep the grid size unchanged.

    Key insight: The legend still decides which objects are "matches", but now both branches simplify texture: matches inherit the legend-frame color, while non-matches collapse to their internal dominant color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        rows, cols = pts[:, 0], pts[:, 1]
        if bool(np.all(rows < sz) and np.all(cols < sz)):
            continue
        comp_vals = g[rows, cols]
        comp_colors = set(map(int, comp_vals))
        if legend_colors.issubset(comp_colors):
            out[rows, cols] = border_color
        else:
            out[rows, cols] = _mode_color(comp_vals)
    return out


# --------------- generator (combo: border recolor + dominant-color flatten) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
OBJECT_COLORS = [0] + PALETTE
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (5, 10)
TALL_W = (2, 3)
WIDE_H = (3, 4)
WIDE_W = (4, 7)
MED_H = (3, 6)
MED_W = (3, 6)

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [3, 6, 5]
NUM_KEPT = 3

MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend at top-left and copy it unchanged to output."""
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


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions while keeping enough area for a unique majority color."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], matched: bool) -> tuple[np.ndarray, int]:
    """
    Build a full non-bg rectangle with a unique dominant color.

    For matched components, all legend colors are guaranteed to appear.
    For unmatched components, at least one legend color is missing.
    """
    area = h * w
    non_targets = [v for v in OBJECT_COLORS if v not in targets]

    if matched:
        allowed = list(OBJECT_COLORS)
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed_targets = random.sample(targets, used_target_count)
        allowed = non_targets + allowed_targets

    dominant = random.choice(allowed)
    minority_pool = [v for v in allowed if v != dominant]
    if not minority_pool:
        raise ValueError("component needs at least one minority color")

    if matched:
        required = [t for t in targets if t != dominant]
        if not required:
            required = [random.choice(minority_pool)]
    else:
        required = [random.choice(minority_pool)]

    max_mutations = (area - 1) // 2
    min_mutations = len(required)
    if min_mutations > max_mutations:
        raise ValueError("component too small for requested color constraints")
    n_mutations = random.randint(min_mutations, max_mutations)

    patch = np.full((h, w), dominant, dtype=int)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_mutations)

    mutation_colors = list(required)
    mutation_colors.extend(random.choice(minority_pool) for _ in range(n_mutations - len(required)))
    random.shuffle(mutation_colors)

    for (r, c), color in zip(coords, mutation_colors):
        patch[r, c] = color
    return patch, int(dominant)


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid combo example and return its input grid."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for matched in roles:
        h, w = _component_dims()
        try:
            patch, dominant = _component_patch(h, w, targets, matched)
            r, c = random_free_location(
                inp,
                patch,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        paste(out, np.full_like(patch, BORDER if matched else dominant), r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
