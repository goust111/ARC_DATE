# Combo variant of d59b0160 (index 16)
# combo_spec: D1/3+D5/1
# Name: 颜色数量超阈值删除+图例颜色计数匹配

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    _candidate = _parent / "solvers" / "arc_utils.py"
    if _candidate.exists():
        sys.path.insert(0, str(_candidate.parent))
        break

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant D1/3 + D5/1
# a) Concreteization and behavior trace
# - The legend remains a 4x4 top-left box with border color 3, but its 3x3 interior now encodes
#   both color identity and multiplicity. This combo uses exactly 4 target colors, with one color
#   repeated twice and the other three shown once each. That gives legend requirements like
#   {1:2, 2:1, 4:1, 5:1}.
# - The deletion threshold is no longer "all colors"; it is a quorum over matched legend kinds.
#   With 4 target colors the threshold is ceil(4/2)=2, so a component is erased iff more than
#   2 legend kinds meet their per-color quota, i.e. iff 3 or 4 kinds match.
# - Trace on one concrete input:
#   legend counts = {1:2, 2:1, 4:1, 5:1}, threshold = 2
#   component A counts = {1:2, 2:1, 4:1}
#     original base task: keep, because color 5 is absent
#     quota-only ALL rule: keep, because only 3 of 4 legend kinds match
#     combo rule: matched kinds = 3, and 3 > 2, so delete
#   component B counts = {1:1, 2:1, 4:1}
#     presence-only threshold rule: delete, because 3 legend colors appear
#     combo rule: color 1 is under quota, so matched kinds = 2, and 2 is not > 2, so keep
# - This pair is the essential fusion behavior: one example needs the D1 quorum change, and the
#   other needs the D5 quota change. Neither single change explains both outcomes.
# b) Complexity calibration
# - The generator fixes the legend to 4 target kinds. This is necessary because with only 3 kinds,
#   ceil(3/2)=2 and "matched kinds > 2" collapses back to an ALL-style gate.
# - The legend count pattern is fixed to [2,1,1,1] instead of allowing many count families.
#   That keeps the visual language simple while still making multiplicity observable.
# - Each sample is forced to include:
#   1. one deleted component with exactly 3 quota-matched legend kinds
#   2. one kept component showing 3 legend colors present but only 2 quota-matched kinds
#   3. one kept component with exactly 2 matched kinds to show the strict cutoff
# - These repeated support cases keep the rule inferable from a few ARC examples.
# c) Hidden-assumption audit against the original main()
# - _bg_boundary still works because the outer frame remains mostly background.
# - _find_legend originally returned a set; that would lose multiplicity and break D5. It must now
#   return a histogram of legend colors.
# - The original component loop only tracked distinct colors. Under the combo, that is insufficient:
#   per-color pixel counts determine whether a legend kind is matched at all.
# - components_mask(g != bg) also finds legend-side fragments as components. That means the legend
#   must not accidentally satisfy the deletion rule itself. The generator therefore uses a safe
#   legend layout where the border-connected legend fragment has only 2 quota-matched kinds.
# - The new threshold logic assumes the number of legend kinds is large enough to make the quorum
#   differ from ALL. The generator guarantees 4 distinct legend kinds in every sample.
# d) Placement legality
# - Every non-legend object is still a solid rectangle of non-background cells with scattered
#   colored pixels inside, so each placed patch is exactly one 4-connected component.
# - random_free_location is used with a zero patch of the same bbox plus 1-cell padding, so
#   components never overlap or touch, and they stay separated from the legend as well.
# - Component dimensions are sampled subject to the maximum color density, so role-specific quota
#   requirements always fit inside the chosen rectangle.
# e) Visual observability
# - The repeated legend color is visible twice in the legend, so humans can notice that one color
#   is treated differently.
# - Deleted "3-of-4 quota-matched" components visibly differ from the original ALL rule.
# - Kept "3-present-but-only-2-matched" components visibly differ from a presence-only threshold
#   rule. This makes the two counting layers independently observable from input/output pairs.
# f) Visual cleanliness
# - The legend uses only 5 colored cells inside its 3x3 area, leaving four bg separators that keep
#   the repeated color and the singletons readable rather than cluttered.
# - Non-legend components remain isolated rectangles on a plain background, preserving the visual
#   grammar of the base task.
# g) Rule inferability
# - No hidden ordering, color arithmetic, or positional lookup is introduced.
# - The only new information is visible counting: repeated swatches in the legend raise that
#   color's quota, and the number of quota-satisfied legend kinds is compared to one stable cutoff.
# 9) Assumptions newly introduced by the updated main()
# - main() now assumes a legend kind is matched only when its pixel count in a component reaches
#   the legend quota. The generator preserves this by constructing components with exact required
#   target-count profiles and by using non-target colors only as filler.
# - main() now compares the matched-kind count to a strict threshold. The generator preserves the
#   intended boundary cases by explicitly creating roles with matched-kind counts of 2, 3, and 4.
# - main() does not special-case the legend component. The generator preserves safety by placing the
#   repeated legend color once in an isolated cell and once in the border-connected fragment, so the
#   border-connected fragment stays at exactly 2 matched kinds and is never erased.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border (bg defined by the frame, not global frequency)."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, dict[int, int]]:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend requirement map is the color histogram of the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1].ravel()
            counts = Counter(int(v) for v in inside if int(v) not in (bg, x))
            return int(sz), int(x), dict(counts)
    raise AssertionError("no legend border found in grid")


def _delete_threshold(n_legend_colors: int) -> int:
    """Delete a component only when its matched legend-kind count exceeds this quorum threshold."""
    return math.ceil(int(n_legend_colors) / 2)


def _matched_legend_kinds(component_counts: Counter[int], legend_counts: dict[int, int]) -> int:
    return sum(
        1 for color, need in legend_counts.items()
        if component_counts.get(color, 0) >= need
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose quota-matched legend palette is large enough.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a small square legend region bordered on its bottom and right edges by one
      color; the interior lists target colors, and repeated swatches increase that color's quota.
    - Elsewhere: multicolored 4-connected objects sit on the bg.

    Rule:
    - Read the legend as a per-color requirement map by counting each non-bg, non-border color in
      the legend interior.
    - Let the deletion threshold be ceil(number_of_distinct_legend_colors / 2).
    - For each connected component, count its pixels of every color, then count how many legend
      colors meet or exceed their required quota inside that component.
    - Erase the component iff its number of quota-matched legend colors is strictly greater than
      the threshold; otherwise leave it unchanged.

    Key insight: The legend defines two nested counts: repeated swatches raise per-color quotas,
    and components disappear only when enough legend colors clear those quotas.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_counts = _find_legend(g, bg)
    assert legend_counts, "legend interior contains no target colors"

    threshold = _delete_threshold(len(legend_counts))
    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        component_counts = Counter(int(g[r, c]) for r, c in pts)
        if _matched_legend_kinds(component_counts, legend_counts) > threshold:
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (combo D1/3 + D5/1) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 4
LEGEND_COUNT_PATTERN = (2, 1, 1, 1)
DELETE_THRESHOLD = _delete_threshold(NUM_TARGETS)

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

MAX_RETRIES = 600

ISOLATED_REPEAT_POS = (0, 0)
ISOLATED_SINGLE_POS = (1, 1)
BORDER_LEGEND_POSITIONS = [(0, 2), (2, 0), (2, 2)]


def _sample_requirements() -> dict[int, int]:
    targets = random.sample(PALETTE, NUM_TARGETS)
    repeated = random.choice(targets)
    requirements = {color: 1 for color in targets}
    requirements[repeated] = 2
    return requirements


def _repeat_and_singles(requirements: dict[int, int]) -> tuple[int, list[int]]:
    repeated = [color for color, need in requirements.items() if need == 2]
    singles = [color for color, need in requirements.items() if need == 1]
    assert len(repeated) == 1 and len(singles) == 3
    return repeated[0], singles


def _place_legend(inp: np.ndarray, out: np.ndarray, requirements: dict[int, int]) -> None:
    """
    Draw a 4x4 legend whose border-connected fragment has only two quota-matched kinds.

    Layout:
    - repeated color: one isolated copy at (0,0), one border-connected copy on a corner
    - one singleton: isolated at (1,1)
    - two singletons: placed on the remaining border-touching corners
    This guarantees the border-connected legend fragment shows exactly two matched singletons while
    the repeated color remains under quota there.
    """
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    repeated, singles = _repeat_and_singles(requirements)
    random.shuffle(singles)
    repeat_border = random.choice(BORDER_LEGEND_POSITIONS)
    other_borders = [pos for pos in BORDER_LEGEND_POSITIONS if pos != repeat_border]
    random.shuffle(other_borders)

    placements = [
        (repeated, ISOLATED_REPEAT_POS),
        (repeated, repeat_border),
        (singles[0], ISOLATED_SINGLE_POS),
        (singles[1], other_borders[0]),
        (singles[2], other_borders[1]),
    ]
    for color, (r, c) in placements:
        inp[r, c] = color
        out[r, c] = color


def _component_dims(min_needed: int) -> tuple[int, int]:
    """Sample component dimensions whose colored capacity can accommodate min_needed pixels."""
    min_area = max(1, math.ceil(min_needed / DENSITY_HI))
    for _ in range(100):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if h * w >= min_area:
            return h, w
    h = max(3, min_area)
    w = max(2, math.ceil(min_area / h))
    return h, w


def _sample_colored_count(area: int, min_needed: int) -> int:
    lo = max(1, math.ceil(area * DENSITY_LO), min_needed)
    hi = max(lo, min(area, math.floor(area * DENSITY_HI)))
    return random.randint(lo, hi)


def _required_colors_for_role(requirements: dict[int, int], role: str) -> list[int]:
    repeated, singles = _repeat_and_singles(requirements)
    if role == "deleted_quorum":
        chosen = random.sample(singles, 2)
        return [repeated, repeated, *chosen]
    if role == "deleted_full":
        return [repeated, repeated, *singles]
    if role == "kept_under_quota":
        chosen = random.sample(singles, 2)
        return [repeated, *chosen]
    if role == "kept_two_exact":
        chosen = random.sample(singles, 2)
        return chosen
    if role == "kept_low":
        mode = random.choice(["repeat_only", "single_only", "repeat_plus_single"])
        if mode == "repeat_only":
            return [repeated]
        if mode == "single_only":
            return [random.choice(singles)]
        return [repeated, random.choice(singles)]
    raise ValueError(f"unknown role: {role}")


def _component_patch(h: int, w: int, requirements: dict[int, int], role: str) -> np.ndarray:
    area = h * w
    required = _required_colors_for_role(requirements, role)
    n_colored = _sample_colored_count(area, len(required))
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)

    non_targets = [color for color in PALETTE if color not in requirements]
    colors = required + [random.choice(non_targets) for _ in range(n_colored - len(required))]
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _build_roles(component_count: int) -> list[str]:
    deleted_count = component_count - NUM_KEPT
    delete_roles = ["deleted_quorum"]
    for _ in range(deleted_count - 1):
        delete_roles.append(random.choice(["deleted_quorum", "deleted_full"]))

    keep_roles = ["kept_under_quota", "kept_two_exact", "kept_low"]
    roles = delete_roles + keep_roles
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose objects exhibit both counting layers."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    requirements = _sample_requirements()
    _place_legend(inp, out, requirements)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = _build_roles(component_count)

    for role in roles:
        min_needed = len(_required_colors_for_role(requirements, role))
        h, w = _component_dims(min_needed)
        patch = _component_patch(h, w, requirements, role)
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
        if not role.startswith("deleted_"):
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
