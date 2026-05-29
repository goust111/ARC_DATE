# Combo variant of d59b0160 (index 24)
# combo_spec: D1/2+D15/2
# Name: 图例定义排除色+不含图例颜色则删除

import math
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo D1/2 + D15/2 variant
# a) Concrete behavior and trace:
#    The corner legend keeps the same 4x4 geometry, but its 3x3 interior now shows 4 exclusion colors,
#    not the 3 colors that get deleted. The fixed semantic palette stays {1,2,4,5,6,8,9}; therefore
#    the 3 target colors are exactly the palette entries absent from the legend.
#    Example: if the legend shows exclusions {1,4,6,8}, then the hidden target set is {2,5,9}.
#    For three components:
#    * {0,2,5}:
#      - original task would keep it because it does not contain all shown legend colors;
#      - D15-only complement-ALL would also keep it because target 9 is missing;
#      - this combo deletes it because it contains no exclusion color at all.
#    * {0,2,5,9}:
#      - original task keeps it;
#      - D15-only complement-ALL deletes it;
#      - this combo also deletes it, now interpretable as "target-only objects vanish".
#    * {0,1,4,6,8}:
#      - original task deletes it because it contains every shown legend color;
#      - D15-only complement-ALL keeps it because it contains no hidden targets;
#      - this combo keeps it because touching any exclusion color is enough to survive.
#    The visible effect is therefore: deleted objects use only colors absent from the legend, while any
#    object that includes a shown legend color stays.
# b) Complexity calibration:
#    The two constituent changes compose into a double negation. Operationally, the erase test collapses
#    to "component color set has no overlap with the shown legend colors". That means the generator must
#    do extra work so the samples do not look like a trivial D1-only variant. We force two witness roles:
#    - one deleted component that uses only a proper subset of the hidden target colors; this differs from
#      the D15-only complement-ALL behavior and makes the D1 inversion visible after complementing;
#    - one kept component that contains all 4 shown legend colors; this differs from the original task and
#      makes it obvious that the shown colors are exclusions/protective colors rather than deletion colors.
#    When there is room for a second deleted object, it contains all 3 hidden targets so the full target
#    complement is also easy to spot across a few examples.
# c) Implicit-assumption audit:
#    _bg_boundary() still relies on the outer frame being mostly background; the generator does not force
#    every candidate to satisfy this, so candidates that confuse border-based background detection are
#    rejected during verification instead of being emitted.
#    _find_legend() still relies on the smallest top-left corner frame whose bottom and right edges share
#    one non-bg color; the legend geometry is unchanged, so this remains valid.
#    The original solver could safely scan every non-bg component because the legend itself never matched
#    the old ALL rule. Under the new NONE-overlap rule, the border fragment inside the legend square would
#    be erased unless we explicitly skip components contained entirely within the legend box.
#    The fixed semantic palette assumption becomes part of the input model: the legend shows 4 exclusions,
#    and the missing 3 colors are the hidden targets. The generator always uses that same 7-color palette.
# d) Placement legality:
#    Each object is placed by asking random_free_location for a full rectangular footprint against the
#    current canvas, with padding=1 and padding_connectivity=8. Because placement uses an all-occupied
#    rectangle mask, components never overlap and always keep at least one background cell of separation
#    from one another and from the legend.
# e) Visual observability:
#    The 4 legend swatches sit on non-touching slots in the 3x3 interior, so the legend reads as a clean
#    set of examples. Deleted objects are target-only, while kept objects always contain at least one shown
#    swatch color; this makes the double-negation rule visible without referring to numeric color ids.
# f) Visual cleanliness:
#    There are only three visual layers: background, one corner legend, and separated rectangular objects.
#    No object overlaps the legend or another object, so the solver-relevant color evidence remains easy to
#    inspect by eye.
# g) Change inferability:
#    The palette size is kept small and fixed. Across the legend plus at least one deleted object, all
#    seven semantic colors are visible: four appear in the legend as exclusions, and the remaining target
#    colors appear inside deleted components. This provides the structural clue needed to infer that "colors
#    missing from the legend" and "objects lacking legend colors" are jointly responsible for deletion.


GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
SEMANTIC_PALETTE = set(PALETTE)
NUM_TARGETS = 3
NUM_LEGEND_COLORS = len(PALETTE) - NUM_TARGETS

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

DENSITY_LO = 0.10
DENSITY_HI = 0.55

COMP_COUNTS = [5, 6]
COMP_WEIGHTS = [2, 5]
NUM_KEPT = 3

MAX_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the top-left legend whose bottom edge and right edge share one border color.
    Return (size, border_color, interior_colors excluding bg/border).
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
    raise AssertionError("no legend border found in grid")


def _inside_legend_component(pts: np.ndarray, sz: int) -> bool:
    """Legend fragments live entirely inside the detected top-left legend square."""
    return bool(np.all((pts[:, 0] < sz) & (pts[:, 1] < sz)))


def _target_colors_from_legend(legend_colors: set[int]) -> set[int]:
    """Interpret the shown legend swatches as exclusions from the fixed semantic palette."""
    exclusions = {int(c) for c in legend_colors if int(c) in SEMANTIC_PALETTE}
    return SEMANTIC_PALETTE - exclusions


def _apply_combo_rule(g: np.ndarray, bg: int, sz: int, exclusion_colors: set[int]) -> np.ndarray:
    """
    Erase non-legend components that contain none of the shown exclusion colors.

    Although we still compute the hidden target complement for analysis and generation, the composed
    predicate simplifies to this direct overlap test on the displayed exclusion swatches.
    """
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        if _inside_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if cols.isdisjoint(exclusion_colors):
            for r, c in pts:
                out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase only those non-legend objects that avoid every legend exclusion color.

    Input structure:
    - The background is read from the outer border.
    - The top-left corner contains the same framed legend square as the base task, but the interior now
      lists exclusion colors: the semantic colors shown inside the frame are the ones objects may share in
      order to survive.
    - Other non-background cells form separate 4-connected rectangular objects elsewhere on the canvas.

    Rule:
    - Detect the top-left legend and read its interior swatches as the exclusion subset of the fixed
      semantic palette {1,2,4,5,6,8,9}.
    - The hidden target colors are the palette entries missing from the legend.
    - Ignore components that belong entirely to the legend square itself.
    - For every other 4-connected component, erase it iff it contains no shown exclusion color at all;
      equivalently, erase components made only of hidden target colors (plus the object's base color 0).

    Key insight: The legend is doubly negated. It does not list "what gets deleted"; it lists the colors
    that prevent deletion. Objects vanish exactly when they are built only from the colors absent from the
    legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border_color, legend_colors = _find_legend(g, bg)
    exclusion_colors = {int(c) for c in legend_colors if int(c) in SEMANTIC_PALETTE}
    target_colors = _target_colors_from_legend(legend_colors)
    assert len(exclusion_colors) == NUM_LEGEND_COLORS, "legend must show exactly four exclusion colors"
    assert len(target_colors) == NUM_TARGETS, "legend complement must leave exactly three target colors"

    return _apply_combo_rule(g, bg, sz, exclusion_colors)


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions with the original shape mix."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _legend_positions() -> list[tuple[int, int]]:
    """Use non-touching slots inside the 3x3 legend interior."""
    return random.sample([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)], NUM_LEGEND_COLORS)


def _place_legend(inp: np.ndarray, out: np.ndarray, exclusions: list[int]) -> None:
    """Draw the legend border and place exclusion-color swatches inside it."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    positions = _legend_positions()
    random.shuffle(positions)
    for color, (r, c) in zip(exclusions, positions):
        inp[r, c] = color
        out[r, c] = color


def _colored_count_bounds(area: int, min_needed: int) -> tuple[int, int]:
    lo = max(1, math.ceil(area * DENSITY_LO), min_needed)
    hi = max(lo, math.floor(area * DENSITY_HI), min_needed)
    return lo, hi


def _sample_present_subset(colors: list[int], *, min_count: int, max_count: int) -> list[int]:
    """Return a random subset with bounded size, preserving uniqueness."""
    count = random.randint(min_count, max_count)
    return random.sample(colors, count)


def _component_patch(h: int, w: int, targets: list[int], exclusions: list[int], role: str) -> np.ndarray:
    """
    Build a rectangular component patch with role-specific overlap against legend exclusions.

    Roles:
    - deleted_partial_targets: contains 1-2 hidden target colors and no exclusion colors.
    - deleted_all_targets: contains all 3 hidden target colors and no exclusion colors.
    - kept_all_exclusions: contains all 4 shown exclusion colors.
    - kept_mixed: contains at least one shown exclusion color and optional target colors.
    """
    area = h * w

    if role == "deleted_partial_targets":
        mandatory = _sample_present_subset(targets, min_count=1, max_count=NUM_TARGETS - 1)
        filler_pool = list(mandatory)
    elif role == "deleted_all_targets":
        mandatory = list(targets)
        filler_pool = list(targets)
    elif role == "kept_all_exclusions":
        max_targets = min(NUM_TARGETS - 1, max(0, area - len(exclusions)))
        used_targets = _sample_present_subset(targets, min_count=0, max_count=max_targets)
        mandatory = list(exclusions) + used_targets
        filler_pool = list(exclusions) + used_targets
    elif role == "kept_mixed":
        max_exclusions = min(len(exclusions), area)
        used_exclusions = _sample_present_subset(exclusions, min_count=1, max_count=max_exclusions)
        remaining_slots = max(0, area - len(used_exclusions))
        used_targets = _sample_present_subset(targets, min_count=0, max_count=min(NUM_TARGETS, remaining_slots))
        mandatory = used_exclusions + used_targets
        filler_pool = used_exclusions + used_targets
    else:
        raise ValueError(f"unknown role: {role}")

    mandatory = list(dict.fromkeys(mandatory))
    if not filler_pool:
        filler_pool = list(exclusions if exclusions else targets)

    lo, hi = _colored_count_bounds(area, len(mandatory))
    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = mandatory + [random.choice(filler_pool) for _ in range(n_colored - len(mandatory))]
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _role_list(component_count: int) -> list[str]:
    """Guarantee witnesses against both the base task and the D15-only complement-ALL variant."""
    deleted_count = component_count - NUM_KEPT
    roles = ["deleted_partial_targets"]
    if deleted_count >= 2:
        roles.append("deleted_all_targets")
    roles += ["kept_all_exclusions"]
    roles += ["kept_mixed"] * (NUM_KEPT - 1)
    remaining = component_count - len(roles)
    roles += random.choices(["deleted_all_targets", "kept_mixed"], k=remaining)
    random.shuffle(roles)
    return roles


def _nonlegend_components(g: np.ndarray, bg: int) -> list[np.ndarray]:
    """Return components outside the dedicated top-left legend box."""
    sz, _border_color, _legend_colors = _find_legend(g, bg)
    comps = []
    for pts in components_mask(g != bg, connectivity=4):
        if not _inside_legend_component(pts, sz):
            comps.append(pts)
    return comps


def _apply_original_rule(g: np.ndarray, bg: int, sz: int, legend_colors: set[int]) -> np.ndarray:
    """Base-task behavior: erase components that contain all shown legend colors."""
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        if _inside_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def _apply_d15_only_rule(g: np.ndarray, bg: int, sz: int, target_colors: set[int]) -> np.ndarray:
    """Single-variant D15 behavior: erase components that contain all hidden target colors."""
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        if _inside_legend_component(pts, sz):
            continue
        cols = {int(g[r, c]) for r, c in pts}
        if target_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def _fusion_witness_counts(g: np.ndarray) -> dict[str, int]:
    """Measure whether a grid visibly exercises the combo semantics."""
    bg = _bg_boundary(g)
    sz, _border_color, legend_colors = _find_legend(g, bg)
    exclusions = {int(c) for c in legend_colors if int(c) in SEMANTIC_PALETTE}
    targets = _target_colors_from_legend(legend_colors)
    out = main(g)

    deleted_partial_targets = 0
    deleted_all_targets = 0
    kept_all_exclusions = 0
    kept_count = 0
    deleted_count = 0

    for pts in _nonlegend_components(g, bg):
        cols = {int(g[r, c]) for r, c in pts}
        semantic_cols = {c for c in cols if c in SEMANTIC_PALETTE}
        erased = all(int(out[r, c]) == bg for r, c in pts)
        if erased:
            deleted_count += 1
            if semantic_cols and semantic_cols.issubset(targets):
                if targets.issubset(semantic_cols):
                    deleted_all_targets += 1
                else:
                    deleted_partial_targets += 1
        else:
            kept_count += 1
            if exclusions.issubset(cols):
                kept_all_exclusions += 1

    return {
        "component_count": deleted_count + kept_count,
        "deleted_count": deleted_count,
        "kept_count": kept_count,
        "deleted_partial_targets": deleted_partial_targets,
        "deleted_all_targets": deleted_all_targets,
        "kept_all_exclusions": kept_all_exclusions,
        "legend_color_count": len(exclusions),
        "target_count": len(targets),
        "diff_vs_original": int(not np.array_equal(out, _apply_original_rule(g, bg, sz, exclusions))),
        "diff_vs_d15_only": int(not np.array_equal(out, _apply_d15_only_rule(g, bg, sz, targets))),
    }


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input for the combo variant."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    exclusions = [c for c in PALETTE if c not in targets]
    _place_legend(inp, out, exclusions)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    for role in _role_list(component_count):
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, exclusions, role)
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
        if role.startswith("kept"):
            paste(out, patch, r, c)

    try:
        predicted = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(predicted, out):
        return None

    try:
        witnesses = _fusion_witness_counts(inp)
    except AssertionError:
        return None
    if witnesses["kept_count"] != NUM_KEPT:
        return None
    if witnesses["deleted_count"] < 2:
        return None
    if witnesses["deleted_partial_targets"] < 1:
        return None
    if witnesses["kept_all_exclusions"] < 1:
        return None
    if witnesses["legend_color_count"] != NUM_LEGEND_COLORS:
        return None
    if witnesses["target_count"] != NUM_TARGETS:
        return None
    if witnesses["diff_vs_original"] != 1:
        return None
    if witnesses["diff_vs_d15_only"] != 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
