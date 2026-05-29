# Combo variant of d59b0160 (index 7)
# combo_spec: D1/2+D4/0
# Name: 8-连通+不含图例颜色则删除

import math
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant D1/2 + D4/0
# - Concrete behavior:
#   The legend is still a 4x4 corner frame whose 3 interior swatches define three target colors.
#   Non-legend objects are still rectangular blocks made of color 0 with scattered accent colors.
#   The fused rule changes two stages at once:
#   * connectivity is now 8-neighbor, so diagonal contact merges rectangles into one component
#   * deletion keeps only components that share at least one legend color; components with zero overlap vanish
# - Small trace:
#   If the legend colors are {1,4,6}, consider two rectangles touching only at one corner.
#   Rectangle A colors = {0,1,5}; rectangle B colors = {0,2,8}.
#   Under the old 4-connected split, A survives and B is deleted.
#   Under this combo, A and B become one 8-connected component with colors {0,1,2,5,8},
#   so the merged object survives because it contains target color 1 somewhere in the union.
#   This is the intended interaction: 8-connectivity can "rescue" a target-free rectangle.
# - Complexity calibration:
#   To keep the fusion visible from a few ARC examples, every generated instance contains one forced
#   diagonal bridge pair plus repeated isolated cases:
#   * one merged pair with mixed membership (one half has target colors, the other half has none)
#   * two isolated kept rectangles that each show partial legend overlap
#   * two isolated deleted rectangles that show complete legend avoidance
#   The repeated categories make the rule readable without introducing too many object types.
# - Implicit assumptions that changed:
#   The original solver could safely process every non-background component because the legend never
#   satisfied the old ALL-target deletion test. That is no longer true: under the NONE-target rule,
#   a legend-border fragment could be erased if inspected directly. The solver must therefore skip
#   any component fully contained inside the detected legend square.
#   A second changed assumption is connectivity itself: component extraction must use connectivity=8,
#   and the generator must maintain 8-neighbor separation between unrelated groups.
# - Placement legality:
#   All patches are placed with random_free_location(... padding=1, padding_connectivity=8), which
#   guarantees no overlap and no accidental 8-neighbor touching between different groups or with the legend.
#   The only intentional 8-touch occurs inside the composite bridge patch, where two rectangles meet
#   diagonally at exactly one corner.
# - Visual observability and cleanliness:
#   The diagonal bridge is easy to see as two blocks kissing at one corner. The passenger block in that
#   bridge uses no legend colors, so if it survives in the output the solver can infer that diagonal merging
#   matters. The isolated deleted and kept rectangles provide simpler supporting examples of the inverted rule.
# - New solver assumptions and generator guarantees:
#   The solver assumes that any component fully inside the legend square belongs to the legend, and that any
#   target-free cells surviving in the output are connected by 8-neighbor paths to target-bearing cells.
#   The generator enforces both conditions and rejects samples where the combo output matches the 4-connected
#   inverted-rule baseline, preventing degeneration into a single-dimension variant.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """Detect the top-left legend square and return (size, border_color, interior_colors)."""
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
    raise AssertionError("no legend border found in grid")


def _inside_legend_component(pts: np.ndarray, sz: int) -> bool:
    """Legend fragments live entirely inside the detected top-left legend square."""
    return bool(np.all((pts[:, 0] < sz) & (pts[:, 1] < sz)))


def _solve_with_connectivity(grid: np.ndarray, connectivity: int) -> np.ndarray:
    """Shared rule implementation used by the combo solver and the 4-connected baseline check."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=connectivity)
    for pts in comps:
        if _inside_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if cols.isdisjoint(legend_colors):
            for r, c in pts:
                out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove only those multicolor objects that never use a legend color, after merging by 8-connectivity.

    Input structure:
    - The background color is recovered from the outer border of the grid.
    - A 4x4 corner legend sits at the top-left, with a shared border color along its bottom and right edges.
    - The legend interior shows the target color set, while the rest of the canvas contains rectangular non-bg objects.

    Rule:
    - Read the target colors from inside the legend frame.
    - Ignore any connected component that belongs entirely to the legend square itself.
    - Treat all other non-background cells as 8-connected components, so diagonal contact merges rectangles.
    - Erase a component only when none of its colors appear in the legend; if it contains at least one legend color anywhere, keep the whole merged component.

    Key insight:
    - The legend now marks colors that let an object survive, and diagonal touching matters because one target-bearing
    rectangle can protect a diagonally attached target-free rectangle once both are merged into a single 8-connected object.
    """
    return _solve_with_connectivity(grid, connectivity=8)


# --------------- generator (o5x) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

DENSITY_LO = 0.12
DENSITY_HI = 0.50
MAX_RETRIES = 400

KEEP_SINGLES = 2
DELETE_SINGLES = 2


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place the three target colors."""
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


def _sample_single_dims() -> tuple[int, int]:
    """Sample a moderate rectangle so five groups can fit cleanly on 16x16."""
    mode = random.choices(["medium", "tall", "wide"], weights=[4, 2, 2])[0]
    if mode == "tall":
        return random.randint(4, 7), random.randint(2, 4)
    if mode == "wide":
        return random.randint(2, 4), random.randint(4, 7)
    return random.randint(3, 5), random.randint(3, 6)


def _sample_bridge_dims() -> tuple[int, int]:
    """Sample smaller rectangles for the forced diagonal bridge pair."""
    return random.randint(2, 4), random.randint(2, 4)


def _scatter_rect(
    h: int,
    w: int,
    guaranteed: list[int],
    extras: list[int],
) -> np.ndarray:
    """Build one full rectangular component: color 0 base plus scattered accent colors."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO), len(guaranteed))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)

    patch = np.zeros((h, w), dtype=int)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = list(guaranteed) + [random.choice(extras) for _ in range(n_colored - len(guaranteed))]
    random.shuffle(colors)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _single_patch(targets: list[int], kept: bool) -> np.ndarray:
    """Create one isolated rectangle that is either guaranteed kept or guaranteed deleted."""
    h, w = _sample_single_dims()
    non_targets = [v for v in PALETTE if v not in targets]
    if kept:
        target_count = random.randint(1, NUM_TARGETS - 1)
        guaranteed = random.sample(targets, target_count) + [random.choice(non_targets)]
        extras = non_targets + guaranteed[:target_count]
    else:
        guaranteed = random.sample(non_targets, 2)
        extras = non_targets
    return _scatter_rect(h, w, guaranteed, extras)


def _bridge_patch(targets: list[int]) -> np.ndarray:
    """
    Build the key combo witness:
    two rectangles that touch only diagonally, where only one half uses a legend color.
    """
    non_targets = [v for v in PALETTE if v not in targets]

    h1, w1 = _sample_bridge_dims()
    h2, w2 = _sample_bridge_dims()
    anchor = _scatter_rect(
        h1,
        w1,
        guaranteed=[random.choice(targets), random.choice(non_targets)],
        extras=non_targets + targets[:1],
    )
    passenger = _scatter_rect(
        h2,
        w2,
        guaranteed=random.sample(non_targets, 2),
        extras=non_targets,
    )

    patch = np.full((h1 + h2, w1 + w2), BG, dtype=int)
    paste(patch, anchor, 0, 0)
    paste(patch, passenger, h1, w1)
    return patch


def _place_patch(inp: np.ndarray, patch: np.ndarray) -> tuple[int, int]:
    """Place a patch with 8-neighbor clearance from all existing non-bg cells."""
    return random_free_location(
        inp,
        patch,
        transparent=BG,
        padding=1,
        padding_connectivity=8,
    )


def _bridge_effect_pixels(inp: np.ndarray) -> int:
    """
    Quantify the fused behavior by comparing against the 4-connected inverted-rule baseline.
    Positive pixels here are objects rescued only by the 8-connected merge.
    """
    combo_out = main(inp)
    baseline_out = _solve_with_connectivity(inp, connectivity=4)
    return int(np.count_nonzero((combo_out != BG) & (baseline_out == BG)))


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid combo sample; return the input grid or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    bridge = _bridge_patch(targets)
    try:
        r, c = _place_patch(inp, bridge)
    except ValueError:
        return None
    paste(inp, bridge, r, c)
    paste(out, bridge, r, c)

    for _ in range(KEEP_SINGLES):
        patch = _single_patch(targets, kept=True)
        try:
            r, c = _place_patch(inp, patch)
        except ValueError:
            return None
        paste(inp, patch, r, c)
        paste(out, patch, r, c)

    for _ in range(DELETE_SINGLES):
        patch = _single_patch(targets, kept=False)
        try:
            r, c = _place_patch(inp, patch)
        except ValueError:
            return None
        paste(inp, patch, r, c)

    if not np.array_equal(main(inp), out):
        return None
    if _bridge_effect_pixels(inp) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
