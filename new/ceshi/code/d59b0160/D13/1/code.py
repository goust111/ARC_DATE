# Variant of d59b0160
# Dimension: D13 — 保留分量的处理方式 (transform)
# Change: 不匹配的连通分量完全保留不变（原样复制到输出）... → 保留分量中的图例颜色像素被高亮

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location

# Design analysis for the D13 variant
# a) Concrete behavior and trace:
#    - Input stays structurally the same as the source task: a 4x4 legend in the top-left,
#      plus several separated non-bg rectangular components built from base color 0 and
#      sprinkled palette colors.
#    - Output still deletes any non-legend component that contains all legend colors.
#      The new behavior only changes kept components: every pixel whose input color is one
#      of the legend colors is recolored to the legend border color 3, while all other
#      pixels in that component stay in place.
#    - Example with legend colors {1,4,8}:
#      kept component colors {0,1,9,4} -> source task kept it unchanged; variant keeps the
#      component shape but turns the 1/4 pixels into 3. Deleted component {0,1,4,8} is
#      still erased to bg exactly as before.
# b) Complexity calibration:
#    - If kept components were allowed to contain zero legend colors, many samples would
#      visually collapse back to the original task. The generator therefore forces each kept
#      component to contain 1 or 2 legend colors, and at least one kept component contains 2.
#    - The only new output color outside the legend is 3, which is a strong and repeated
#      visual cue tied directly to the legend border.
# c) Hidden assumptions from the original solver:
#    - `components_mask(g != bg)` also sees the legend itself. That was harmless before
#      because the legend components almost never contained all target colors, so they were
#      naturally preserved. In this variant, recoloring all kept components would corrupt the
#      legend unless legend-local components are skipped explicitly.
#    - Color membership for each component must be computed from the input grid `g`, not from
#      the progressively edited output, otherwise early highlighting could affect later tests.
# d) Placement legality:
#    - Components are still placed by reserving their full bounding rectangles with
#      `random_free_location(..., padding=1, padding_connectivity=8)`, so they do not overlap
#      the legend or each other and remain distinct 4-connected objects.
# e) Visual observability:
#    - Every generated grid has exactly three kept components and each of them contains
#      highlightable legend-color pixels, so the variant behavior is always visible.
# f) Visual cleanliness:
#    - Highlighting is in-place recoloring rather than extra overlay geometry, so object
#      boundaries stay unchanged and the output remains easy to parse.
# g) Color-inference constraint:
#    - No hidden ordering by color id is introduced. The recolor target is always the visible
#      border color of the legend, which humans can infer directly from the examples.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: the only components fully inside the legend square belong to the legend.
#      Guarantee: the generator never places external components inside or touching that box.
#    - Assumption: kept components are missing at least one legend color.
#      Guarantee: the generator only assigns 1 or 2 of the 3 legend colors to kept objects.


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
    Remove palette-complete components and highlight surviving legend-colored pixels.

    Input structure:
    - Most cells are background, defined by the dominant color on the outer frame.
    - The top-left corner contains a square legend whose bottom edge and right edge share one border color.
    - Away from that legend are separated 4-connected multicolor components built on the background.

    Rule:
    - Read the target color set from the legend interior.
    - Ignore the legend itself, then inspect every other connected component.
    - If a component contains all legend colors, erase it to background; otherwise keep it, but recolor every pixel whose color belongs to the legend set into the legend border color.

    Key insight: The legend still specifies which components are complete enough to delete, but for incomplete components it also supplies the highlight color, exposing where the partial legend palette appears.
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
                out[r, c] = bg
        else:
            for r, c in pts:
                if int(g[r, c]) in legend_colors:
                    out[r, c] = border_color
    return out


# --------------- generator (o5x) ---------------

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

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 300


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


def _component_dims() -> tuple[int, int]:
    """Sample component (h, w) according to shape-mode weights."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _highlight_patch(patch: np.ndarray, targets: list[int]) -> np.ndarray:
    """Replace every legend-color pixel in a kept component with the border color."""
    out = patch.copy()
    mask = np.isin(out, np.array(targets, dtype=int))
    out[mask] = BORDER
    return out


def _component_patch(h: int, w: int, targets: list[int], deleted: bool, kept_target_count: int) -> np.ndarray:
    """Build an h x w component patch with guaranteed target-color participation."""
    area = h * w
    required = NUM_TARGETS if deleted else kept_target_count
    lo = max(required, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in targets]

    if deleted:
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


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid (input, output) pair; return input or None on failure."""
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

    for deleted, kept_target_count in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, deleted, kept_target_count)
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
        if not deleted:
            paste(out, _highlight_patch(patch, targets), r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
