# Combo variant of d59b0160 (index 23)
# combo_spec: D2/0+D13/1
# Name: 匹配分量重着色+保留分量高亮图例颜色

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location

"""
Design analysis for combo D2/0 + D13/1
======================================

a) Concrete fused behavior and trace:
  - The input keeps the original layout: a 16x16 field with background 7, a 4x4 legend in the
    top-left, and several separated 4-connected rectangular components elsewhere.
  - The legend still provides two signals: its interior gives the 3 target colors, and its
    border color 3 becomes the output-action color for both branches.
  - Example with legend colors {1,4,8}:
      component A colors = {0,1,4,8,9}
      component B colors = {0,1,4,6}
    Source task:
      A contains all legend colors, so A is deleted to 7.
      B is missing 8, so B is copied unchanged.
    Fused task:
      A contains all legend colors, so every pixel of A becomes border color 3.
      B is missing 8, so B stays in place, but its 1/4 pixels turn into 3 while 0/6 stay.
    Visible difference:
      matched components become solid green silhouettes, while kept components show green
      in-place highlights at the target-color pixels.

b) Complexity calibration:
  - The selection predicate stays identical to the source task, so the new difficulty comes from
    output interpretation rather than a new matching rule.
  - To avoid degenerate samples, every grid contains both branches:
      * at least one matched component, which becomes a full solid block of 3
      * exactly three kept components, each containing 1 or 2 target colors so highlighting is visible
  - At least one kept component contains 2 target colors, which gives repeated evidence that the
    highlight rule depends on legend-color membership rather than on a single special color.

c) Hidden-assumption audit against the original solver:
  - `_bg_boundary(g)` still works because the outer frame remains dominated by background 7.
  - `_find_legend(g, bg)` still works because the legend geometry is unchanged.
  - `components_mask(g != bg, connectivity=4)` also finds fragments inside the legend. That was
    harmless in the source task because most legend fragments were unchanged anyway, but in this
    combo variant both branches recolor to border color 3, so the legend must be skipped
    explicitly or it would be corrupted.
  - Component color membership must be read from the original input grid `g`, not from a partially
    edited output array, or earlier recoloring would leak into later decisions.
  - The old deletion assignment `out[r, c] = bg` no longer fits the fused rule. Matching
    components now use `border_color`, while kept components recolor only their target-color pixels.
  - The solver now assumes the legend border color is the only recolor target. The generator
    guarantees this by keeping normal components free of color 3 in the input.

d) Placement legality:
  - The legend occupies only the top-left 4x4 square.
  - External components are placed with `random_free_location(..., padding=1, padding_connectivity=8)`
    using their full rectangular footprint, so they do not overlap, touch each other, or touch the legend.
  - Bounding boxes may touch the grid edge, which is allowed by the source task.

e) Visual observability:
  - The combo must be inferable from 2-4 examples. Solid green matched rectangles and partial green
    highlights inside unmatched rectangles are visually different and both tied to the same legend frame.
  - Because color 3 is absent from ordinary input components, any new green pixels in the output are
    clearly explained by the legend border rather than by noise.

f) Visual cleanliness:
  - No extra markers are added outside the original component footprints.
  - Recoloring happens in-place, so the output preserves the same object layout and avoids clutter.
  - Components stay separated by background gaps, so the two branch effects remain easy to compare.

g) Color-change inferability:
  - The recolor target is not hidden or numeric; it is the visible legend border color.
  - The generator restricts kept components to 1 or 2 target colors, creating repeated examples
    where only legend-member pixels flip to 3 and non-member pixels do not.

9) New solver assumptions and generator guarantees:
  - Assumption: any component fully inside the top-left legend square belongs to the legend.
    Guarantee: external components are never placed there or adjacent to it.
  - Assumption: each kept component is missing at least one legend color.
    Guarantee: kept components are generated with exactly 1 or 2 distinct target colors.
  - Assumption: at least one component triggers the full-component recolor branch and at least one
    kept component triggers the partial highlight branch.
    Guarantee: component counts are 4-6 with exactly 3 kept components, and every kept component
    is forced to contain highlightable target-color pixels.
"""


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


def _is_legend_component(pts: np.ndarray, legend_size: int) -> bool:
    """Legend fragments live entirely inside the top-left legend square."""
    return bool(np.all(pts[:, 0] < legend_size) and np.all(pts[:, 1] < legend_size))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor palette-complete components and highlight legend-colored pixels in the rest.

    Input structure:
    - The background is the dominant color on the outer frame.
    - The top-left corner contains a square legend whose bottom edge and right edge share one border color, while the interior shows the target palette.
    - Away from that legend sit separated 4-connected multicolor components on the background.

    Rule:
    - Read the target color set from the legend interior and the output color from the legend border.
    - Ignore legend-local components, then inspect every other connected component.
    - If a component contains all legend colors, recolor the whole component to the border color; otherwise keep its shape and non-target pixels, but recolor each legend-colored pixel inside it to the border color.

    Key insight: The legend now controls both which components are "complete" and which color marks them, so full matches become solid border-color blocks while partial matches reveal their target-color locations by in-place highlighting.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend_size, border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _is_legend_component(pts, legend_size):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = border_color
        else:
            for r, c in pts:
                if int(g[r, c]) in legend_colors:
                    out[r, c] = border_color
    return out


# --------------- generator (combo D2/0 + D13/1) ---------------

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

# -- Component count & matched/kept split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 300


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box and place target colors without making the legend itself a full match."""
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
    matched: bool,
    kept_target_count: int,
) -> np.ndarray:
    """Build an h x w component patch with forced target-color participation."""
    area = h * w
    required = NUM_TARGETS if matched else kept_target_count
    lo = max(required, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if matched:
        colors = list(targets)
        filler_palette = PALETTE
    else:
        chosen_targets = random.sample(targets, kept_target_count)
        colors = list(chosen_targets)
        filler_palette = non_targets + chosen_targets

    colors.extend(random.choice(filler_palette) for _ in range(n_colored - len(colors)))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _recolor_patch(patch: np.ndarray, color: int) -> np.ndarray:
    """Turn a matched component into one solid color while preserving its footprint."""
    return np.full_like(patch, int(color))


def _highlight_patch(patch: np.ndarray, targets: list[int]) -> np.ndarray:
    """Replace legend-color pixels inside a kept component with the border color."""
    out = patch.copy()
    out[np.isin(out, np.array(targets, dtype=int))] = BORDER
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid fused example; return input or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    kept_target_counts = [2] + [random.choice([1, 2]) for _ in range(NUM_KEPT - 1)]
    random.shuffle(kept_target_counts)

    roles: list[tuple[bool, int]] = [(True, 0)] * (component_count - NUM_KEPT)
    roles.extend((False, k) for k in kept_target_counts)
    random.shuffle(roles)

    for matched, kept_target_count in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, matched, kept_target_count)
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
        transformed = _recolor_patch(patch, BORDER) if matched else _highlight_patch(patch, targets)
        paste(out, transformed, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
