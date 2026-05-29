# Combo variant of d59b0160 (index 12)
# combo_spec: D7/0+D8/1
# Name: 黑色背景+完整矩形边框

from pathlib import Path
import math
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVERS_DIR = ROOT / "solvers"
if str(SOLVERS_DIR) not in sys.path:
    sys.path.append(str(SOLVERS_DIR))

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the fused variant:
# - Concrete behavior shift:
#   Original samples used orange(7) as background and let object rectangles be mostly black(0),
#   so `g != bg` cleanly recovered each rectangle as a foreground component. After switching the
#   background to black(0), that construction collapses: a black-filled rectangle would visually
#   dissolve into the background and the solver would stop seeing the intended objects. To keep
#   the same deletion logic while honoring the new background, components are rebuilt as solid
#   non-black rectangles with a few recolored accent cells. The test "does this 4-connected
#   component contain all legend colors?" stays unchanged.
# - Legend concretization:
#   A 4x4 full frame would leave only a 2x2 interior, which is too cramped for three target
#   swatches and makes the legend visually muddy. The fused version therefore uses a 5x5 legend:
#   green border(3) on all four sides, 3x3 interior, three target cells inside. This makes the
#   full-rectangle change visible while preserving the original "three-color legend" role.
# - Small example trace:
#   Original: a top-left L frame on orange background marks colors {1,4,8}; a black rectangle
#   elsewhere containing 1,4,8 is erased to orange. Fused: a top-left 5x5 full green frame on
#   black background marks the same set {1,4,8}; a solid grey/yellow/etc. rectangle elsewhere
#   containing 1,4,8 is erased to black. Only the visual packaging changes.
# - Hidden assumptions audited line-by-line:
#   `_bg_boundary` still works because the 16x16 outer border has 60 cells and the 5x5 legend
#   contributes only 9 green perimeter cells, so black remains the border majority.
#   `_find_legend` can no longer search for an L; it must verify an entire square perimeter.
#   `components_mask(g != bg)` can no longer include the legend region, because inner legend
#   swatches may touch the full frame and would otherwise be mistaken for deletable objects.
#   The generator therefore keeps the legend isolated and the solver explicitly blanks the legend
#   window before connected-component analysis.
# - Legality / cleanliness:
#   Generator placement uses `random_free_location(..., padding=1)` on a canvas where the legend
#   border is already occupied, so objects never overlap, never leave the grid, and stay separated
#   from one another and from the legend. Target-cell layouts come from a short curated list whose
#   cells all touch the inner rim of the frame: this keeps the legend readable and also ensures the
#   full-frame change is behaviorally meaningful, because an L-border solver would now tend to merge
#   those swatches into the frame component and mis-handle the legend.
# - Non-degeneracy check target:
#   The produced inputs should visibly have black background + full rectangular legend, and the
#   original L-border solver should fail or disagree on many of them because the legend geometry
#   and component construction are materially different.


def _bg_boundary(g: np.ndarray) -> int:
    """Background is the most common color on the outer 1-cell border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the smallest top-left square whose full perimeter is a single non-bg color.

    Returns:
      (size, border_color, legend_colors)
    where legend_colors are the non-bg, non-border colors inside the perimeter.
    """
    h, w = g.shape
    for sz in range(3, min(h, w) + 1):
        border_vals = np.concatenate(
            [g[0, :sz], g[sz - 1, :sz], g[1 : sz - 1, 0], g[1 : sz - 1, sz - 1]]
        )
        uniq = set(map(int, np.unique(border_vals)))
        if len(uniq) != 1:
            continue
        border_color = next(iter(uniq))
        if border_color == bg:
            continue
        inside = g[1 : sz - 1, 1 : sz - 1]
        colors = set(map(int, np.unique(inside)))
        colors.discard(bg)
        colors.discard(border_color)
        if colors:
            return int(sz), int(border_color), colors
    raise AssertionError("no full rectangular legend found in top-left corner")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove objects that realize the complete legend palette under the fused visual variant.

    Input structure:
    - Black (0) is the global background.
    - The top-left corner contains a green (3) legend frame drawn as a complete square border.
      The colors strictly inside that frame define the target set.
    - Outside the legend window, the scene contains several 4-connected multicolor rectangles on
      the background, separated from each other by at least one background cell.

    Rule:
    - Read the target-color set from the interior of the top-left rectangular frame.
    - Ignore the legend area itself when looking for scene objects.
    - For every remaining 4-connected non-background component, erase it to black iff its color
      set contains every legend color. Components missing at least one legend color stay unchanged.

    Key insight:
    - The fusion changes only the presentation: black replaces orange as background and the legend
      cue is now a full rectangle instead of an L-corner. The actual reasoning stays "legend gives
      a required color set; delete exactly the components that contain the full set."
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend_sz, _legend_border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    obj_mask = (g != bg).copy()
    obj_mask[:legend_sz, :legend_sz] = False

    out = g.copy()
    for pts in components_mask(obj_mask, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (combo 12) ---------------

GRID_H, GRID_W = 16, 16
BG = 0
BORDER = 3
LEGEND_SIZE = 5
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 11)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 7)
MED_W = (2, 8)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

MAX_RETRIES = 400

LEGEND_PATTERNS = [
    [(0, 0), (0, 2), (2, 0)],
    [(0, 0), (0, 2), (2, 2)],
    [(0, 0), (2, 0), (2, 2)],
    [(0, 2), (2, 0), (2, 2)],
]


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw a 5x5 full rectangular legend at top-left and place 3 target swatches inside."""
    s = LEGEND_SIZE
    positions = random.choice(LEGEND_PATTERNS)
    shuffled_targets = targets[:]
    random.shuffle(shuffled_targets)

    for g in (inp, out):
        g[:s, :s] = BG
        g[0, :s] = BORDER
        g[s - 1, :s] = BORDER
        g[:s, 0] = BORDER
        g[:s, s - 1] = BORDER

    for color, (r, c) in zip(shuffled_targets, positions):
        inp[r + 1, c + 1] = color
        out[r + 1, c + 1] = color


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions with a bias toward medium shapes."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], deleted: bool) -> np.ndarray:
    """
    Build a solid non-black rectangle and recolor a subset of cells.

    The base color is chosen from the non-target palette so connectivity does not depend on black.
    Deleted components are forced to contain all target colors; kept components are forced to miss
    at least one target color.
    """
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if deleted:
        lo = max(lo, NUM_TARGETS)

    non_targets = [v for v in PALETTE if v not in targets]
    base = random.choice(non_targets)
    patch = np.full((h, w), base, dtype=int)

    n_marks = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_marks)

    if deleted:
        accent_pool = PALETTE[:]
        colors = list(targets) + [random.choice(accent_pool) for _ in range(n_marks - NUM_TARGETS)]
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        kept_targets = random.sample(targets, used_target_count)
        accent_pool = [v for v in non_targets if v != base] + kept_targets
        if not accent_pool:
            accent_pool = kept_targets or [base]
        colors = [random.choice(accent_pool) for _ in range(n_marks)]

    random.shuffle(colors)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input for the fused rule."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_KEPT) + [False] * NUM_KEPT
    random.shuffle(roles)

    for deleted in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted)
        try:
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
        if not deleted:
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
