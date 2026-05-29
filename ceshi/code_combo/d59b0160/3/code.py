# Combo variant of d59b0160 (index 3)
# combo_spec: D1/0+D17/0
# Name: ANY匹配+面积阈值

import math
import random
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent))
        break
    if (_parent / "solvers" / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent / "solvers"))
        break

from arc_utils import components_mask, paste, random_free_location


# Design analysis for combo D1/0 + D17/0
# a) Concrete behavior and trace:
#    The visual scaffold stays the same: a 4x4 legend sits in the top-left corner,
#    with green border color 3 on its bottom row and right column, and exactly three
#    target colors placed in the 3x3 interior. Elsewhere, separated rectangular
#    components are built from foreground color 0 with extra sprinkled palette colors.
#    Example with legend {1,4,8} and threshold 18:
#    - A 5x4 component (area 20) with colors {0,1,9} was kept by the base ALL rule
#      because it missed 4 and 8; here it is deleted because ANY hit + area > 18.
#    - A 3x4 component (area 12) with colors {0,1,4} is also a hit under ANY, but it
#      survives because the area gate blocks deletion.
#    - A 5x4 component with colors {0,5,6} survives despite being large, so size
#      alone is not sufficient.
# b) Complexity calibration:
#    The generator uses only two clear size bands: small components are always at
#    most 16 cells, large components are always at least 20 cells, with the solver
#    threshold fixed at 18. This avoids borderline counting ambiguity while keeping
#    the area rule visible. Match types are limited to miss / partial-hit / full-hit,
#    so the task combines two dimensions without exploding the hypothesis space.
# c) Hidden assumption audit:
#    The original solver scanned all non-background cells together. Under ANY, that
#    would incorrectly consider the legend itself a deletable component, so the
#    legend square must be masked out before component labeling. The solver also
#    assumes component area means the whole non-background rectangle; that remains
#    valid because the generator pastes solid 0-based rectangles and only sprinkles
#    extra colors inside them. Finally, component spacing must preserve distinct
#    4-connected objects; the generator enforces that with a one-cell padding gap.
# d) Placement legality:
#    Placement still uses random_free_location(..., padding=1) on full rectangular
#    occupancy masks, so components neither overlap nor touch each other or the
#    legend, and every pasted patch remains inside the 16x16 board.
# e) Visual observability:
#    Every sample contains at least one large partial-hit component that disappears
#    and at least one small hit component that remains. That pair is the minimum
#    evidence needed to infer BOTH ANY semantics and the area exemption. Each sample
#    also contains a large miss component that remains, ruling out a pure size-based
#    deletion explanation.
# f) Visual cleanliness:
#    No extra markers are added. Objects stay as sparse rectangles on a flat orange
#    background, with single-cell gaps between regions, so legend, components, and
#    deletions remain easy to parse by eye.
# g) Change inferability:
#    There is no color remapping. The only signals are legend membership and component
#    size, both of which are directly visible. The palette remains capped at the same
#    seven non-background colors as the base task, and the repeated role pattern
#    across samples gives enough structure for humans to infer the fused rule.


AREA_THRESHOLD = 18

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component geometry bands --
SMALL_DIMS = [(3, 2), (3, 3), (4, 2), (3, 4), (4, 3), (4, 4), (5, 2)]
LARGE_DIMS = [(4, 5), (5, 4), (4, 6), (6, 4), (5, 5), (5, 6), (6, 5), (7, 3), (3, 7)]

# -- Component fill density (fraction of colored pixels inside a 0-rectangle) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Generation retry budget --
MAX_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the smallest top-left square whose bottom edge and right edge share one
    non-background color, then read the interior legend colors from that square.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase only the large connected components that overlap the legend palette.

    Input structure:
    - One background color fills most of the board.
    - The top-left corner contains a small legend square whose bottom edge and right
      edge use one border color, while the 3x3 interior shows the target palette.
    - Away from the legend are separate 4-connected multicolor components built from
      foreground rectangles with sprinkled colors.

    Rule:
    - Read the legend color set from inside the corner square.
    - Exclude the legend square itself from the component search.
    - For each other connected component, test whether it contains **any** legend
      color and compute its area in cells.
    - Erase the component only if both conditions hold: it shares at least one legend
      color and its area is greater than 18.
    - Keep components with no legend colors, and also keep matching components whose
      area is 18 or less.

    Key insight: The legend is now a permissive color trigger, but that trigger is
    gated by object size, so overlap with the legend matters only for sufficiently
    large components.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    mask = (g != bg).copy()
    mask[:sz, :sz] = False
    comps = components_mask(mask, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        area = int(len(pts))
        if cols & legend_colors and area > AREA_THRESHOLD:
            for r, c in pts:
                out[r, c] = bg
    return out


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place target colors."""
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


def _sample_colored_count(area: int, min_required: int) -> int:
    lo = max(1, math.ceil(area * DENSITY_LO), min_required)
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _component_dims(role: str) -> tuple[int, int]:
    """Choose dimensions from the small or large band implied by the role."""
    if role.startswith("small_"):
        return random.choice(SMALL_DIMS)
    if role.startswith("large_"):
        return random.choice(LARGE_DIMS)
    raise ValueError(f"unknown role: {role}")


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """
    Build a rectangular component patch from a 0-filled base plus sprinkled colors.

    Roles:
    - large_partial_delete: area > threshold, contains 1-2 target colors
    - large_full_delete: area > threshold, contains all target colors
    - small_partial_keep: area <= threshold, contains 1-2 target colors
    - small_full_keep: area <= threshold, contains all target colors
    - large_miss_keep: area > threshold, contains no target colors
    """
    area = h * w
    non_targets = [v for v in PALETTE if v not in targets]

    if role in {"large_partial_delete", "small_partial_keep"}:
        hit_count = random.randint(1, NUM_TARGETS - 1)
        chosen_targets = random.sample(targets, hit_count)
        n_colored = _sample_colored_count(area, hit_count)
        forced = list(chosen_targets)
        allowed = non_targets + chosen_targets
    elif role in {"large_full_delete", "small_full_keep"}:
        n_colored = _sample_colored_count(area, NUM_TARGETS)
        forced = list(targets)
        allowed = PALETTE
    elif role == "large_miss_keep":
        n_colored = _sample_colored_count(area, 1)
        forced = []
        allowed = non_targets
    else:
        raise ValueError(f"unknown role: {role}")

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = forced + [random.choice(allowed) for _ in range(n_colored - len(forced))]
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _component_roles() -> list[str]:
    """
    Build a fixed evidence set for the fused rule.

    Each sample contains:
    - a large partial-hit deleted component to show ANY vs ALL
    - a small partial-hit kept component to show the area exemption
    - a small full-hit kept component to show area outranks even stronger matches
    - a large miss kept component to show size alone does not delete
    - one extra large deleted component for repeated support
    """
    extra_deleted = random.choices(
        ["large_partial_delete", "large_full_delete"],
        weights=[3, 1],
    )[0]
    roles = [
        "large_partial_delete",
        extra_deleted,
        "small_partial_keep",
        "small_full_keep",
        "large_miss_keep",
    ]
    random.shuffle(roles)
    return roles


def _role_is_deleted(role: str) -> bool:
    return role in {"large_partial_delete", "large_full_delete"}


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output is reproduced by main()."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    for role in _component_roles():
        h, w = _component_dims(role)
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
        if not _role_is_deleted(role):
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
