# Variant of d59b0160
# Dimension: D1 — 颜色匹配条件：全部包含 vs 其他逻辑 (match_cond)
# Change: 连通分量必须包含图例中的所有颜色(ALL)才被删除，缺少任何一种图例颜色则保留
#         → 恰好包含图例颜色集合（不多不少）

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D1 variant
# a) Concrete behavior trace:
#    - The top-left 4x4 legend still shows exactly 3 target colors.
#    - Every non-legend object is still one 4-connected component made from a 0-colored base
#      plus sprinkled non-zero colors.
#    - The decisive set is therefore "component colors after discarding the internal filler 0".
#      Example with legend {1,4,8}:
#        original rule: component colors {0,1,4,8,6} contains all targets -> delete
#        variant rule:  non-zero colors are {1,4,8,6} != {1,4,8} -> keep
#      The new visible branch is a "superset" object: it has all legend colors plus an extra color,
#      so it survives now even though the original solver would erase it.
# b) Complexity calibration:
#    - Keep the legend format, grid size, palette, and component placement from the source task.
#    - Restrict objects to three repeated categories:
#        exact   = colors exactly equal legend set -> deleted
#        subset  = missing >=1 legend color       -> kept
#        superset= all legend colors + extra      -> kept
#      This is stricter than ALL but still learnable from a few examples because the only new
#      distinction is "extra color present or not".
# c) Implicit assumption audit:
#    - _bg_boundary still assumes the outer frame reveals the real background; generator keeps bg=7.
#    - _find_legend still assumes a top-left square with bottom/right border color 3; unchanged.
#    - components_mask(g != bg) still treats each object's 0-filled rectangle as one component.
#      Under exact matching, comparing the raw component color set would be wrong because 0 is always
#      present inside objects. The solver must discard 0 before equality testing.
#    - The generator must guarantee that exact-deleted objects use only target colors, while superset
#      kept objects contain at least one non-target color in addition to all targets.
# d) Placement legality:
#    - Each component is generated as one rectangular patch and placed with random_free_location
#      using a full-bbox occupancy mask, so components never overlap and keep a 1-cell gap.
#    - The legend is written first in the top-left corner, so object placement also respects it.
# e) Visual observability:
#    - Each generated grid contains at least one exact-deleted object and at least one superset-kept
#      object. This guarantees the new rule changes what disappears in the output.
#    - Subset-kept objects remain as support examples for the "missing a legend color" branch.
# f) Visual cleanliness:
#    - Objects stay spatially separated, the legend remains isolated in one corner, and there is no
#      extra annotation layer. The only evidence comes from the color content inside each object.
# g) Color inferability:
#    - Only three target colors are shown in the legend, and object colors come from the same small
#      palette as the source task. No ordering, arithmetic, or hidden color-id convention is used.
# 9) New solver assumptions checked against the generator:
#    - main() assumes exact deletion is based on non-zero colors only; generator always uses 0 as the
#      internal filler and never places 0 inside the legend.
#    - main() assumes there is at least one target color in the legend and at least one deletable
#      component. The generator enforces 3 legend colors and 1-3 exact-deleted objects.


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


def _component_palette(g: np.ndarray, pts: np.ndarray, bg: int) -> set[int]:
    """Visible object palette for matching: discard the global bg and the internal filler 0."""
    cols = {int(g[r, c]) for r, c in pts}
    cols.discard(bg)
    cols.discard(0)
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove only those components whose visible palette is an exact legend match.

    Input structure:
    - The frame color still defines the global background.
    - A 4x4 legend remains in the top-left corner, with a single-color bottom/right border and
      three target colors placed inside the 3x3 interior.
    - Elsewhere, each object is a 4-connected component built from a 0-colored body plus
      scattered non-zero colors from the palette.

    Rule:
    - Read the target color set from the legend interior.
    - For each 4-connected component, collect its non-zero colors and ignore the 0-colored body.
    - Erase the component iff that non-zero color set is exactly equal to the legend color set.
    - If the component is missing a legend color or contains any extra non-target color, keep it.

    Key insight: The legend now specifies an exact palette signature rather than a mere inclusion
    test, so objects that are supersets of the legend colors survive instead of being removed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _component_palette(g, pts, bg) == legend_colors:
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (D1 exact-set variant) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of non-zero pixels) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & role mix --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3
SUPERSET_KEPT_CHOICES = [1, 2]

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


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """Build one component patch; role controls whether it is exact, subset, or superset."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = math.floor(area * DENSITY_HI)
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "exact":
        lo = max(lo, NUM_TARGETS)
    elif role == "superset":
        lo = max(lo, NUM_TARGETS + 1)
    hi = max(lo, hi)

    n_colored = random.randint(lo, hi)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)

    if role == "exact":
        colors = list(targets) + [random.choice(targets) for _ in range(n_colored - NUM_TARGETS)]
    elif role == "subset":
        used_target_count = random.randint(1, NUM_TARGETS - 1)
        allowed = random.sample(targets, used_target_count)
        colors = [random.choice(allowed) for _ in range(n_colored)]
    elif role == "superset":
        colors = list(targets) + [random.choice(non_targets)]
        remaining = n_colored - (NUM_TARGETS + 1)
        colors.extend(random.choice(targets + non_targets) for _ in range(remaining))
    else:
        raise ValueError(f"unknown role: {role}")

    random.shuffle(colors)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _component_roles(component_count: int) -> list[str]:
    """Choose a role mix with exact deletions plus both kept branches."""
    deleted_count = component_count - NUM_KEPT
    superset_kept = random.choice(SUPERSET_KEPT_CHOICES)
    subset_kept = NUM_KEPT - superset_kept
    roles = ["exact"] * deleted_count + ["superset"] * superset_kept + ["subset"] * subset_kept
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input; return it if main() reproduces the intended output."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = _component_roles(component_count)

    for role in roles:
        h, w = _component_dims()
        patch = _component_patch(h, w, targets, role)
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
        if role != "exact":
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
