# Variant of d59b0160
# Dimension: D15 — 图例颜色数量 (meta)
# Change: 图例包含3种不同的目标颜色（在所有训练/测试示例中均为3种）... → 图例包含所有非背景颜色减去排除色

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D15 variant
# a) Concrete behavior:
#    The 4x4 top-left legend keeps the same border shape, but its 3x3 interior now shows
#    the excluded colors, not the deletion colors. We keep the original 7-color semantic
#    palette {1,2,4,5,6,8,9}. If the legend shows {1,4,6,8}, then the true targets are
#    the missing colors {2,5,9}. A component with {0,1,4,6,8,2,5} must stay because 9
#    is absent, while a component with {0,2,5,9} must be erased even though it lacks the
#    shown legend colors. The change happens only at the "interpret legend colors" stage.
# b) Complexity calibration:
#    The legend now shows 4 colors instead of 3, which is visually distinct. To make the
#    negation readable from a few examples, every generated grid includes both:
#    - a kept component containing all legend colors but missing one true target;
#    - a deleted component containing all true targets while missing at least one legend color.
#    These are direct counterexamples to the original semantics.
# c) Implicit-assumption audit:
#    _bg_boundary() still works because the outer frame remains background-dominant.
#    _find_legend() still works because the legend border geometry is unchanged.
#    The original main() treated the interior set as the required deletion set; that
#    assumption is exactly what changes here. We therefore compute targets as the palette
#    complement of the legend swatches before testing components.
# d) Placement legality:
#    Components are still full rectangular patches whose non-bg footprint includes both
#    0-cells and colored cells, so each patch is one 4-connected component. Placement uses
#    random_free_location(..., transparent=BG, padding=1), which keeps patches separated
#    from each other and from the legend.
# e) Visual observability:
#    The legend swatches are placed on checkerboard-like non-touching slots inside the 3x3
#    interior so they read as separate samples. Because kept/deleted counterexamples are
#    guaranteed, a human can see that the removed objects share the colors absent from the
#    legend, not the colors shown inside it.
# f) Visual cleanliness:
#    No overlays are introduced. The legend stays confined to the top-left 4x4 box, and
#    components remain spaced by at least one background cell.
# g) Color-rule inference:
#    The semantic palette is fixed and small (7 colors), matching the original task. The
#    generator ensures all 7 palette colors are visible across the legend plus components,
#    so the "take the complement" rule is inferable from structure rather than color ids.


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

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

MAX_RETRIES = 300


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the outer border."""
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


def _target_colors_from_legend(legend_colors: set[int]) -> set[int]:
    """Interpret legend colors as exclusions from the fixed semantic palette."""
    exclusions = {int(c) for c in legend_colors if int(c) in SEMANTIC_PALETTE}
    targets = SEMANTIC_PALETTE - exclusions
    return targets


def _apply_deletion_rule(g: np.ndarray, bg: int, required_colors: set[int]) -> np.ndarray:
    """Erase every non-bg component whose color set contains all required colors."""
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if required_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase components that contain every palette color omitted from the legend.

    Input structure:
    - Background (bg) covers most of the grid.
    - The top-left corner contains the same square legend frame as the original task, but
      the interior now shows exclusion swatches rather than the true deletion colors.
    - Elsewhere are multicolored 4-connected components built from non-bg cells.

    Rule:
    - Read the interior legend colors as the excluded subset of the fixed palette.
    - Compute the actual target set as the palette colors that do not appear in the legend.
    - For each connected component, erase it iff it contains all of those missing colors.
    - Leave the legend and all other components unchanged.

    Key insight: The legend must be interpreted by complement: the decisive colors are the
    few palette entries absent from the legend, not the ones displayed inside it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    target_colors = _target_colors_from_legend(legend_colors)
    assert target_colors, "legend excludes the entire semantic palette"

    return _apply_deletion_rule(g, bg, target_colors)


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


def _sample_missing_subset(colors: list[int]) -> tuple[list[int], list[int]]:
    """Return (present_subset, missing_subset) with at least one color missing."""
    missing_count = random.randint(1, len(colors))
    missing = random.sample(colors, missing_count)
    present = [c for c in colors if c not in missing]
    return present, missing


def _component_patch(h: int, w: int, targets: list[int], exclusions: list[int], role: str) -> np.ndarray:
    """
    Build a rectangular component patch with role-specific color requirements.

    Roles:
    - deleted_missing_legend: contains all targets, but misses >=1 legend color.
    - deleted_mixed: contains all targets and any subset of legend colors.
    - kept_all_legend: contains all legend colors, but misses >=1 target color.
    - kept_mixed: misses >=1 target color and may contain any subset of legend colors.
    """
    if role == "deleted_missing_legend":
        allowed_exclusions, _missing = _sample_missing_subset(exclusions)
        mandatory = list(targets) + allowed_exclusions
        filler_pool = list(targets) + allowed_exclusions
    elif role == "deleted_mixed":
        k = random.randint(0, len(exclusions))
        allowed_exclusions = random.sample(exclusions, k)
        mandatory = list(targets)
        filler_pool = list(targets) + allowed_exclusions
    elif role == "kept_all_legend":
        present_targets, _missing = _sample_missing_subset(targets)
        mandatory = list(exclusions) + present_targets
        filler_pool = list(exclusions) + present_targets
    elif role == "kept_mixed":
        present_targets, _missing = _sample_missing_subset(targets)
        k = random.randint(0, len(exclusions))
        used_exclusions = random.sample(exclusions, k)
        mandatory = present_targets + used_exclusions
        filler_pool = present_targets + used_exclusions
    else:
        raise ValueError(f"unknown role: {role}")

    mandatory = list(dict.fromkeys(mandatory))
    if not filler_pool:
        filler_pool = [random.choice(targets + exclusions)]

    area = h * w
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
    """Guarantee visible counterexamples to the original legend-as-target interpretation."""
    deleted_count = component_count - NUM_KEPT
    roles = ["deleted_missing_legend"] + ["deleted_mixed"] * (deleted_count - 1)
    roles += ["kept_all_legend"] + ["kept_mixed"] * (NUM_KEPT - 1)
    random.shuffle(roles)
    return roles


def _nonlegend_components(g: np.ndarray, bg: int) -> list[np.ndarray]:
    """Return components outside the dedicated top-left legend box."""
    comps = []
    for pts in components_mask(g != bg, connectivity=4):
        if any(r >= LEGEND_SIZE or c >= LEGEND_SIZE for r, c in pts):
            comps.append(pts)
    return comps


def _variant_witness_counts(g: np.ndarray) -> dict[str, int]:
    """Measure whether the grid visibly exercises the complement semantics."""
    bg = _bg_boundary(g)
    _sz, _x, legend_colors = _find_legend(g, bg)
    exclusions = {int(c) for c in legend_colors if int(c) in SEMANTIC_PALETTE}
    targets = _target_colors_from_legend(legend_colors)
    out = main(g)

    deleted_missing_legend = 0
    kept_with_all_legend = 0
    deleted_count = 0

    for pts in _nonlegend_components(g, bg):
        cols = {int(g[r, c]) for r, c in pts}
        erased = all(int(out[r, c]) == bg for r, c in pts)
        if erased:
            deleted_count += 1
            if not exclusions.issubset(cols):
                deleted_missing_legend += 1
        elif exclusions.issubset(cols):
            kept_with_all_legend += 1

    return {
        "deleted_count": deleted_count,
        "deleted_missing_legend": deleted_missing_legend,
        "kept_with_all_legend": kept_with_all_legend,
        "legend_color_count": len(exclusions),
        "target_count": len(targets),
    }


def _original_style_output(g: np.ndarray) -> np.ndarray:
    """Apply the original direct-legend semantics to the same input."""
    bg = _bg_boundary(g)
    _sz, _x, legend_colors = _find_legend(g, bg)
    legend_as_targets = {int(c) for c in legend_colors if int(c) in SEMANTIC_PALETTE}
    assert legend_as_targets, "legend contains no semantic colors"
    return _apply_deletion_rule(g, bg, legend_as_targets)


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input for the complement-legend variant."""
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

    if not np.array_equal(main(inp), out):
        return None

    witnesses = _variant_witness_counts(inp)
    if witnesses["deleted_missing_legend"] < 1:
        return None
    if witnesses["kept_with_all_legend"] < 1:
        return None
    if witnesses["legend_color_count"] != NUM_LEGEND_COLORS:
        return None
    if witnesses["target_count"] != NUM_TARGETS:
        return None
    if np.array_equal(main(inp), _original_style_output(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
