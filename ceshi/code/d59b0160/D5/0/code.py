# Variant of d59b0160
# Dimension: D5 — 图例颜色集合的提取方式 (legend_lookup)
# Change: 图例区域内所有非背景(7)、非边框(3)颜色的集合作为目标颜色集... → 仅使用图例中的特定位置颜色

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D5 variant
# a) Concrete behavior and execution trace:
#    - The legend stays as the original 4x4 top-left frame: bottom row and right column are color 3,
#      interior is a 3x3 area.
#    - Only the interior main diagonal [(0,0), (1,1), (2,2)] defines the deletion palette.
#      Three extra off-diagonal cells [(0,2), (1,2), (2,0)] carry distractor colors.
#    - Small trace:
#      original lookup: unique(interior) might return {1, 2, 4, 5, 6, 8}
#      new lookup: only read cells [(0,0), (1,1), (2,2)] -> {1, 4, 8}
#      component colors {0,1,4,8,5} was kept before (missing 2 and 6) but is deleted now.
#    - The changed step is precisely "legend palette extraction"; component detection and erasure stay the same.
# b) Complexity calibration:
#    - Keep exactly 3 true targets so the task still matches the original deletion arity.
#    - Add exactly 3 distinct distractor colors so "read every legend color" is visibly wrong.
#    - Use a fixed spatial rule (main diagonal) rather than many layouts, because ARC examples are sparse.
# c) Implicit-assumption audit against the original main():
#    - _bg_boundary still works because the outer border remains bg-heavy.
#    - _find_legend can still detect the same 4x4 corner frame; that assumption is unchanged.
#    - The old `np.unique(inside)` extraction is no longer valid because off-diagonal colors are decoys.
#    - The solver now assumes the diagonal cells are non-bg and distinct; the generator guarantees this.
#    - The solver also assumes the legend itself is not one component containing all target colors.
#      Therefore the generator must avoid filling bridging cells that connect top-left target to the border cluster.
# d) Placement legality:
#    - Components are still pasted with `random_free_location(..., padding=1)` so they do not overlap
#      or touch through 8-neighborhood padding.
#    - Legend distractor cells use a safe fixed pattern: [(0,2), (1,2), (2,0)].
#      This keeps (0,0) isolated while (1,1) may join the border-connected target at (2,2),
#      so no legend component contains all three targets.
# e) Visual observability:
#    - Across examples the diagonal always carries the effective colors, while non-diagonal legend colors vary.
#    - Deleted objects are forced to miss at least one distractor color, so the new behavior diverges from
#      the original "all legend colors" interpretation on the same input.
# f) Visual cleanliness:
#    - The legend is still small and separated from the main objects.
#    - Only three distractor cells are added, avoiding a cluttered fully-filled 3x3 interior.
# g) Color-rule inferability:
#    - No color-id ordering is used. The rule is entirely positional.
#    - Kept objects sometimes contain all distractor colors but miss one diagonal color, giving extra evidence
#      that off-diagonal legend colors are irrelevant.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Find the top-left legend square whose bottom edge and right edge share one border color.

    The D5 variant keeps the same legend frame geometry as the source task, so only the
    downstream interpretation of interior cells changes.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            return int(sz), int(x)
    raise AssertionError("no legend border found in grid")


def _legend_diagonal_colors(g: np.ndarray, sz: int, border_color: int, bg: int) -> set[int]:
    """Read only the interior main diagonal of the legend; off-diagonal legend colors are decoys."""
    interior_size = sz - 1
    assert interior_size >= 1, "legend interior is empty"
    colors = {int(g[i, i]) for i in range(interior_size)}
    colors.discard(bg)
    colors.discard(border_color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove components that cover the legend's diagonal color code.

    Input structure:
    - Most cells are background.
    - The top-left corner contains the same framed legend shape as the source task.
    - Inside that legend, colored diagonal cells are meaningful and extra off-diagonal colors are distractors.
    - Elsewhere, non-background rectangles form 4-connected multicolor components.

    Rule:
    - Detect the legend frame in the top-left corner.
    - Read only the interior diagonal cells of the legend as the target color set.
    - For each 4-connected non-background component, test whether it contains every diagonal color.
    - Erase components that do; keep components that miss at least one diagonal color, even if they include distractor colors.

    Key insight: The legend is no longer interpreted as "all colors shown inside the box"; its spatial layout matters, and only the diagonal swatches define the filter.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, border_color = _find_legend(g, bg)
    legend_colors = _legend_diagonal_colors(g, sz, border_color, bg)
    assert legend_colors, "legend diagonal contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
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
NUM_DISTRACTORS = 3
LEGEND_TARGET_POSITIONS = [(0, 0), (1, 1), (2, 2)]
LEGEND_DISTRACTOR_POSITIONS = [(0, 2), (1, 2), (2, 0)]

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 300


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int], distractors: list[int]) -> None:
    """Draw the legend and encode targets only on the interior diagonal."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    for color, (r, c) in zip(targets, LEGEND_TARGET_POSITIONS):
        inp[r, c] = color
        out[r, c] = color
    for color, (r, c) in zip(distractors, LEGEND_DISTRACTOR_POSITIONS):
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


def _component_patch(
    h: int,
    w: int,
    targets: list[int],
    distractors: list[int],
    deleted: bool,
    prefer_all_distractors: bool,
) -> np.ndarray:
    """Build one rectangle component whose color set matches the desired role."""
    area = h * w
    non_special = [v for v in PALETTE if v not in targets and v not in distractors]

    if deleted:
        used_distractor_count = random.randint(0, NUM_DISTRACTORS - 1)
        used_distractors = random.sample(distractors, used_distractor_count)
        mandatory = list(targets) + used_distractors
        pool = targets + used_distractors + non_special
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        used_targets = random.sample(targets, used_target_count)
        if prefer_all_distractors:
            used_distractors = list(distractors)
        else:
            used_distractor_count = random.randint(0, NUM_DISTRACTORS)
            used_distractors = random.sample(distractors, used_distractor_count)
        mandatory = used_targets + used_distractors
        pool = used_targets + used_distractors + non_special

    if not pool:
        pool = list(non_special)

    lo = max(1, math.ceil(area * DENSITY_LO), len(mandatory))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)

    colors = list(mandatory) + [random.choice(pool) for _ in range(n_colored - len(mandatory))]
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input grid whose expected output is defined by this variant."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    chosen = random.sample(PALETTE, NUM_TARGETS + NUM_DISTRACTORS)
    targets = chosen[:NUM_TARGETS]
    distractors = chosen[NUM_TARGETS:]
    _place_legend(inp, out, targets, distractors)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    roles = [True] * deleted_count + [False] * NUM_KEPT
    random.shuffle(roles)

    kept_with_all_distractors = False
    for deleted in roles:
        h, w = _component_dims()
        prefer_all_distractors = (not deleted) and (not kept_with_all_distractors)
        patch = _component_patch(h, w, targets, distractors, deleted, prefer_all_distractors)
        try:
            r, c = random_free_location(
                inp, np.zeros_like(patch), transparent=BG, padding=1, padding_connectivity=8
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if not deleted:
            paste(out, patch, r, c)
            if prefer_all_distractors:
                kept_with_all_distractors = True

    if not kept_with_all_distractors:
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
