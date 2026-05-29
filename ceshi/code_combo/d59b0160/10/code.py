# Combo variant of d59b0160 (index 10)
# combo_spec: D2/1+D6/3
# Name: 反转保留删除+删除留幽灵框架
#
# Design analysis
# ===============
# a) Concrete behavior and execution trace:
#    - Keep the original legend geometry: a 4x4 top-left square whose bottom edge and right edge are
#      green (color 3), with exactly three target swatches inside.
#    - Keep the original object grammar: each non-legend object is one 4-connected solid rectangle of
#      non-background cells, mostly color 0 with sprinkled accent colors.
#    - Change the object action in two linked steps:
#      1. invert the selection effect, so components containing all legend colors are now preserved;
#      2. every non-matching component is removed and replaced by a green bounding-box frame.
#    - Small trace:
#      legend colors = {1,4,8}
#      component A colors = {0,1,4,8,5}
#      component B colors = {0,1,4,6}
#      original task: delete A to background, keep B intact
#      D2-only inversion: keep A, erase B to background
#      fused rule here: keep A, erase B's interior and draw a green rectangle around B's old bbox
#      The visible difference from both source behaviors is that survivors are the full matches, while
#      rejected objects do not simply vanish; they leave positional "ghost frames".
# b) Complexity calibration:
#    - Generated grids always include both categories: at least one matching component and exactly three
#      non-matching components. This prevents degeneration into a pure keep-only or pure framing task.
#    - Ghosted components are forced to be at least 3x3 so the hollow frame is visually legible rather
#      than collapsing into a 1-cell-thick strip.
#    - The frame color is the same green as the legend border, and green is excluded from the object
#      palette, so the replacement mark is salient and structurally tied to the legend.
# c) Implicit-assumption audit of the original solver:
#    - `_bg_boundary(...)` still holds because the outer border is still dominated by background 7.
#    - `_find_legend(...)` still holds because the legend shape and placement are unchanged.
#    - The original loop over `components_mask(g != bg)` can no longer process every component uniformly:
#      after inversion, the legend's own non-matching fragments would be deleted or framed unless they are
#      explicitly excluded. The fused solver therefore skips any component wholly inside the legend square.
#    - The old assumption "delete means fill points with bg" is invalid. Deletion now means clear the
#      component and redraw its inclusive bounding-box border in the legend border color.
# d) Placement legality:
#    - The generator still places each rectangle by reserving its full bbox with
#      `random_free_location(..., padding=1, padding_connectivity=8)`, so objects never overlap or touch.
#    - Because the ghost frame is drawn strictly inside the removed component's bbox, output markers stay
#      in-bounds and cannot collide with neighboring objects that were already separated in the input.
#    - The legend is drawn before object placement, so the top-left region remains reserved.
# e) Visual observability:
#    - Across a few examples, humans can compare kept full-palette objects against rejected partial-palette
#      objects and see that rejected ones consistently turn into empty green rectangles.
#    - No invisible ordering or numeric color semantics are used; the rule depends only on visible legend
#      membership and obvious object geometry.
# f) Visual cleanliness:
#    - Objects remain padded apart and the ghost mark is one simple rectangle per removed object, so the
#      output stays sparse and readable.
#    - Using the object's own bbox avoids noisy remnants or overlaps that would make it hard to infer which
#      removed object produced which marker.
# g) Inferability and anti-degeneration guard:
#    - Matching objects are preserved exactly, non-matching objects become frames, and both categories are
#      guaranteed to appear in generated samples.
#    - Therefore this fusion cannot collapse back to the source task or to the D2-only inversion: the same
#      input would yield materially different outputs under those simpler rules.

from pathlib import Path
import sys

import math
import random

import numpy as np

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    solvers_dir = parent / "solvers"
    if solvers_dir.is_dir():
        sys.path.insert(0, str(solvers_dir))
        break
else:
    raise ImportError("could not locate solvers/ for arc_utils import")

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the smallest top-left square whose bottom row and right column share one border color.
    Return (size, border_color, set_of_target_colors_inside).
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


def _is_legend_component(pts: np.ndarray, sz: int) -> bool:
    """Legend cells live entirely inside the top-left sz x sz square."""
    return bool(np.all(pts[:, 0] < sz) and np.all(pts[:, 1] < sz))


def _draw_bbox_border(out: np.ndarray, pts: np.ndarray, color: int) -> None:
    """Draw the inclusive bounding-box border of `pts` into `out`."""
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep legend-matching objects and replace every other object with a ghost frame.

    Input structure:
    - One background color fills most of the canvas.
    - A small legend square sits in the top-left corner, with a shared border color on its bottom edge
      and right edge and three target colors inside.
    - Elsewhere, each object is a 4-connected non-background component separated from the others by
      background cells.

    Rule:
    - Read the target color set from the legend interior.
    - Examine each non-background 4-connected component outside the legend.
    - If a component contains every legend color, leave it unchanged.
    - Otherwise erase that component to background and draw its bounding-rectangle border in the legend
      border color.

    Key insight: the legend still defines the palette test, but the outcome is now asymmetric in two ways:
    full matches survive intact, while non-matches are not merely deleted; they are converted into hollow
    positional markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, frame_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _is_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            continue
        for r, c in pts:
            out[r, c] = bg
        _draw_bbox_border(out, pts, frame_color)
    return out


# --------------- generator (o5x combo variant) ---------------

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

# -- Component fill density (fraction of non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & role split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_GHOSTED = 3

# -- Generation retry budget --
MAX_RETRIES = 600


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place the 3 target colors."""
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


def _component_dims(ghosted: bool) -> tuple[int, int]:
    """Sample component dimensions, forcing ghosted ones to have visible hollow frames."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        h = random.randint(*TALL_H)
        w = random.randint(*TALL_W)
    elif mode == "wide":
        h = random.randint(*WIDE_H)
        w = random.randint(*WIDE_W)
    else:
        h = random.randint(*MED_H)
        w = random.randint(*MED_W)

    if ghosted:
        h = max(h, 3)
        w = max(w, 3)
    return h, w


def _component_patch(h: int, w: int, targets: list[int], matches: bool) -> np.ndarray:
    """Build a solid non-background rectangle with sparse colored accents."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    if matches:
        lo = max(lo, NUM_TARGETS)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if matches:
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


def _frame_patch(h: int, w: int) -> np.ndarray:
    """Return a hollow rectangular frame in BORDER on BG."""
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = BORDER
    patch[-1, :] = BORDER
    patch[:, 0] = BORDER
    patch[:, -1] = BORDER
    return patch


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output shows both preserved matches and ghost frames."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = [True] * (component_count - NUM_GHOSTED) + [False] * NUM_GHOSTED
    random.shuffle(roles)

    for matches in roles:
        h, w = _component_dims(ghosted=not matches)
        patch = _component_patch(h, w, targets, matches)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if matches:
            paste(out, patch, r, c)
        else:
            paste(out, _frame_patch(h, w), r, c, transparent=BG)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
