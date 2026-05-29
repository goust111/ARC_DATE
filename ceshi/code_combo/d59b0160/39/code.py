# Combo variant of d59b0160 (index 39)
# combo_spec: D5/0+D11/2+D15/0
# Name: 紧凑图例+特定位置颜色+颜色数量可变

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior and execution trace:
#    - The source task uses a 4x4 corner frame whose 3x3 interior contains three colored swatches and empty bg cells.
#      Solver logic = read all interior non-bg colors, then delete every non-bg 4-connected component containing all of them.
#    - This combo changes three linked stages at once:
#      1) D11/2: the legend interior becomes a fully filled k x k square with no empty cells.
#      2) D5/0: only the main diagonal cells of that square are semantically active; off-diagonal cells are distractors.
#      3) D15/0: k itself varies from 2 to 5, so the number of active colors also varies from 2 to 5.
#    - Small trace:
#      if k=4, the top-left legend is a 5x5 framed block; the 4x4 interior is fully colored.
#      diagonal cells might be {1, 4, 8, 9}; off-diagonal cells might show {2, 5, 6}.
#      old-style lookup on the same geometry would read {1, 2, 4, 5, 6, 8, 9};
#      new lookup reads only {1, 4, 8, 9}.
#      therefore a component colored {0, 1, 4, 8, 9, 2, 5} is deleted now even though it is missing distractor 6.
#    - The important interaction is that "compact" and "position-based" are not independent:
#      once every legend cell is colored, the solver must explicitly ignore non-diagonal colors rather than relying on bg gaps.
# b) Complexity calibration:
#    - The active-position rule is fixed to the main diagonal across all examples; this is strong enough to infer from few samples.
#    - k is limited to 2..5. That gives visible variation without making the legend too large for a 16x16 grid.
#    - Off-diagonal colors come from a small distractor set (usually 2-3 distinct colors) repeated across the square.
#      This keeps the compact legend visually readable instead of turning it into a 25-color noise block.
#    - Generator enforces both contrast directions:
#      at least one deleted component misses some shown distractor;
#      at least one kept component contains all shown distractors but misses a diagonal target.
#      Those witnesses prevent the task from degenerating back to "use every legend color".
# c) Implicit-assumption audit against the original main():
#    - `_bg_boundary` still needs the outer frame to stay bg-dominant. The generator keeps a border of bg around placed objects.
#    - `_find_legend` can no longer assume a fixed 4x4 size, so it must detect a variable top-left framed square.
#    - The original legend-color extraction via `np.unique(interior)` is wrong here because every interior pixel is colored.
#      The solver must read only diagonal coordinates.
#    - The original component loop included the legend itself because it labeled every non-bg cell.
#      With a dense legend, that would immediately erase the legend component, so the new solver must mask out the legend region.
#    - The new diagonal lookup assumes diagonal cells are non-bg and non-border; the generator guarantees that by construction.
# d) Placement legality:
#    - The legend occupies exactly the top-left (k+1) x (k+1) square: k x k colored interior plus bottom/right border in color 3.
#    - Object rectangles are placed with `random_free_location(..., padding=1, border=1)` on a full-rectangle occupancy mask.
#      This prevents overlap, out-of-bounds placement, and accidental touching with the legend or with each other.
#    - Component dimensions are resampled until their area can hold all mandatory colors for that role.
# e) Visual observability:
#    - Across examples, the only stable structural cue inside the compact legend is the main diagonal.
#      Deleted objects track those diagonal colors even when they omit some off-diagonal colors.
#    - Because k varies, examples show that the number of required colors is not fixed at three.
#      A 2-color example and a 5-color example still share the same positional rule.
# f) Visual cleanliness:
#    - The legend remains a single corner block with one border color, so it is easy to isolate from the object field.
#    - Off-diagonal distractors are repeated from a small palette instead of using many unrelated colors.
#    - Objects remain separated rectangles with sparse interior sprinkles, matching the source task's overall visual language.
# g) Color-rule inferability:
#    - The rule depends only on position within the legend square, never on numeric ordering of color ids.
#    - Repeated distractor colors on off-diagonal cells make it plausible for a human to discount them as filler,
#      while the diagonal offers a consistent geometric explanation across k=2..5.
# h) New solver assumptions to enforce in generation:
#    - `main()` assumes the detected legend is the intended top-left frame; generator rejects samples if detection returns the wrong size.
#    - `main()` assumes non-legend components do not intrude into the legend square; placement with padding enforces that.
#    - `main()` assumes there is at least one active diagonal color and at least one shown distractor color; generator always provides both.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Find the largest top-left framed square whose bottom edge and right edge share one border color.

    In this combo, the border stays as an L-shaped frame but its size varies with the number of active colors.
    """
    h, w = g.shape
    found: tuple[int, int] | None = None
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            found = (int(sz), int(x))
    if found is None:
        raise AssertionError("no legend border found in grid")
    return found


def _legend_diagonal_colors(g: np.ndarray, sz: int, border_color: int, bg: int) -> set[int]:
    """Read only the interior main diagonal as the active target-color set."""
    interior_size = sz - 1
    colors = {int(g[i, i]) for i in range(interior_size)}
    colors.discard(bg)
    colors.discard(border_color)
    return colors


def _all_legend_colors(g: np.ndarray, sz: int, border_color: int, bg: int) -> set[int]:
    """Original-style baseline: read every non-bg, non-border color in the legend interior."""
    inside = g[0 : sz - 1, 0 : sz - 1]
    colors = set(map(int, np.unique(inside)))
    colors.discard(bg)
    colors.discard(border_color)
    return colors


def _erase_matching_components(g: np.ndarray, bg: int, legend_size: int, target_colors: set[int]) -> np.ndarray:
    """Shared component erasure logic with the legend square excluded from labeling."""
    out = g.copy()
    fg = (g != bg).copy()
    fg[:legend_size, :legend_size] = False
    comps = components_mask(fg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if target_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def _baseline_all_legend_colors_main(grid: np.ndarray) -> np.ndarray:
    """Reference semantics used only for generator anti-degeneracy checks."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    sz, border_color = _find_legend(g, bg)
    legend_colors = _all_legend_colors(g, sz, border_color, bg)
    return _erase_matching_components(g, bg, sz, legend_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove components that contain the compact legend's active diagonal palette.

    Input structure:
    - Most cells are background.
    - The top-left corner contains a variable-size legend: a fully colored k x k square whose bottom edge and right edge
      are bordered by one color, making a (k+1) x (k+1) corner frame.
    - Every interior legend position is occupied, but only the main diagonal positions are meaningful.
    - Elsewhere, sparse multicolor rectangles form separate 4-connected non-background components.

    Rule:
    - Detect the top-left framed legend and infer its size from the shared border color.
    - Read only the interior main diagonal colors as the target set; ignore off-diagonal legend colors.
    - Excluding the legend region itself, find each 4-connected non-background component in the rest of the grid.
    - Erase every component that contains all diagonal target colors; keep components that miss at least one of them.

    Key insight: The legend is both compact and deceptive: all cells are colored, so the solver must use the diagonal position pattern, not the full set of visible legend colors, and it must adapt to a target-set size that changes from example to example.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, border_color = _find_legend(g, bg)
    legend_colors = _legend_diagonal_colors(g, sz, border_color, bg)
    assert legend_colors, "legend diagonal contains no target colors"

    return _erase_matching_components(g, bg, sz, legend_colors)


# --------------- generator (o5x) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
PALETTE = [1, 2, 4, 5, 6, 8, 9]
TARGET_COUNT_RANGE = (2, 5)
MAX_SHOWN_DISTRACTORS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (6, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (4, 8)
MED_H = (3, 7)
MED_W = (3, 8)

# -- Component fill density (fraction of colored non-zero pixels inside the black rectangle) --
DENSITY_LO = 0.12
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [3, 5, 4]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 400


def _shown_distractor_count(target_count: int, available: int) -> int:
    """Use a small repeated distractor palette so the legend stays readable."""
    desired = 2 if target_count == 2 else 3
    return max(1, min(available, desired, MAX_SHOWN_DISTRACTORS))


def _build_legend_interior(targets: list[int], shown_distractors: list[int]) -> np.ndarray:
    """Create a dense k x k legend whose diagonal is meaningful and off-diagonal cells are distractors."""
    k = len(targets)
    interior = np.zeros((k, k), dtype=int)
    for i, color in enumerate(targets):
        interior[i, i] = color

    m = len(shown_distractors)
    offset = random.randrange(m)
    for r in range(k):
        for c in range(k):
            if r == c:
                continue
            interior[r, c] = shown_distractors[(offset + 2 * r + c) % m]
    return interior


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int], shown_distractors: list[int]) -> int:
    """Draw the variable-size compact legend at the top-left corner."""
    k = len(targets)
    s = k + 1
    interior = _build_legend_interior(targets, shown_distractors)

    for g in (inp, out):
        g[:s, :s] = BG
        g[:k, :k] = interior
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER
    return s


def _component_dims(min_area: int) -> tuple[int, int]:
    """Sample component dimensions whose area can host the mandatory colors."""
    for _ in range(50):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if h * w >= min_area:
            return h, w

    h = max(3, math.ceil(min_area / 3))
    w = max(3, math.ceil(min_area / h))
    return h, w


def _component_patch(
    h: int,
    w: int,
    targets: list[int],
    shown_distractors: list[int],
    deleted: bool,
    force_baseline_difference: bool,
) -> np.ndarray:
    """Build one black-base rectangle whose color set matches the intended role."""
    area = h * w
    non_special = [v for v in PALETTE if v not in targets and v not in shown_distractors]

    if deleted:
        mandatory = list(targets)
        if force_baseline_difference and shown_distractors:
            used_distractor_count = random.randint(0, len(shown_distractors) - 1)
        else:
            used_distractor_count = random.randint(0, len(shown_distractors))
        used_distractors = random.sample(shown_distractors, used_distractor_count)
        mandatory += used_distractors
        pool = targets + used_distractors + non_special
    else:
        used_target_count = random.randint(0, len(targets) - 1)
        used_targets = random.sample(targets, used_target_count)
        if force_baseline_difference:
            used_distractors = list(shown_distractors)
        else:
            used_distractor_count = random.randint(0, len(shown_distractors))
            used_distractors = random.sample(shown_distractors, used_distractor_count)
        mandatory = used_targets + used_distractors
        pool = used_targets + used_distractors + non_special

    if not pool:
        pool = list(non_special)
    if not pool:
        pool = list(targets if deleted else shown_distractors)

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
    """Try once to build a valid input grid for this combo variant."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    target_count = random.randint(*TARGET_COUNT_RANGE)
    targets = random.sample(PALETTE, target_count)
    non_targets = [v for v in PALETTE if v not in targets]
    shown_distractors = random.sample(
        non_targets, _shown_distractor_count(target_count, len(non_targets))
    )
    legend_size = _place_legend(inp, out, targets, shown_distractors)

    detected_size, detected_border = _find_legend(inp, BG)
    if detected_size != legend_size or detected_border != BORDER:
        return None
    if _legend_diagonal_colors(inp, detected_size, detected_border, BG) != set(targets):
        return None
    if not (set(shown_distractors) & _all_legend_colors(inp, detected_size, detected_border, BG)):
        return None

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    specs = (
        [(True, True)]
        + [(True, False)] * (deleted_count - 1)
        + [(False, True)]
        + [(False, False)] * (NUM_KEPT - 1)
    )
    random.shuffle(specs)

    for deleted, force_baseline_difference in specs:
        mandatory_count = len(targets)
        if deleted:
            if force_baseline_difference:
                mandatory_count += max(0, len(shown_distractors) - 1)
            else:
                mandatory_count += len(shown_distractors)
        else:
            if force_baseline_difference:
                mandatory_count = len(shown_distractors)
            else:
                mandatory_count = max(1, len(shown_distractors))

        h, w = _component_dims(max(8, mandatory_count))
        patch = _component_patch(
            h,
            w,
            targets,
            shown_distractors,
            deleted=deleted,
            force_baseline_difference=force_baseline_difference,
        )
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
        if not deleted:
            paste(out, patch, r, c)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_baseline_all_legend_colors_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
