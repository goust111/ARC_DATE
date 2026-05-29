# Combo variant of d59b0160 (index 9)
# combo_spec: D5/1+D6/1
# Name: 图例颜色计数匹配+部分删除仅图例颜色像素

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import components_mask, paste, random_free_location


# Design analysis
# ---------------
# fused_rule:
#   - The legend keeps the original 4x4 top-left frame with border color 3 on the bottom row
#     and right column, but the 3x3 interior is now a quota table rather than a plain set.
#   - If the legend interior shows color 5 twice, color 8 once, and color 9 once, then a
#     component matches only when it contains at least {5:2, 8:1, 9:1}.
#   - Matching components are not erased wholesale. Only pixels whose own colors belong to the
#     legend quota map are turned into bg=7; black scaffold pixels and non-legend accent colors
#     survive, creating a visible punched-hole effect.
# concrete_trace:
#   - Original task:
#       legend colors {5,8,9}
#       component counts {0:20, 5:1, 8:3, 9:1, 4:2}
#       because all three kinds appear, the whole rectangle disappears.
#   - Fused variant:
#       legend requirements {5:2, 8:1, 9:1}
#       same component counts do not meet the quota for color 5, so the component stays intact.
#       another component with counts {0:18, 5:2, 8:1, 9:2, 4:1} does match, but its output
#       becomes only the non-legend residue {0,4}; it is perforated instead of removed.
#   - The two modifications interact on the same object: counting decides whether deletion fires,
#     and selective deletion decides which pixels vanish once it does fire.
# complexity_calibration:
#   - The generator uses a small family of count patterns [(2,1,1), (2,2,1), (3,1,1)] so at
#     least one legend color always repeats, but the quota table remains easy to read.
#   - Every sample includes at least one kept component that contains all legend color kinds but
#     misses one repeated quota by exactly one pixel. This makes the counting rule visually
#     necessary; a plain set-membership solver would make the wrong prediction.
#   - Every deleted component is forced to contain visible non-legend residue after deletion,
#     typically both black scaffold cells and at least one non-target accent color. This makes
#     the partial-deletion rule visibly different from full erasure.
# hidden_assumption_audit_for_main:
#   - _find_legend in the original solver returned a set; that silently loses multiplicity and is
#     invalid here, so it must return a histogram.
#   - The component test can no longer rely on a set of colors; it must count exact pixel totals
#     inside each 4-connected component.
#   - Selective deletion must still iterate over components found in the original input grid, not
#     in a progressively edited output grid. Otherwise deleting some legend-color pixels could
#     split a matched component and corrupt later decisions.
#   - The legend itself must not satisfy its own quota. To guarantee this, one occurrence of a
#     repeated legend color is isolated at (0,0) while (0,1) and (1,0) stay background, so no
#     legend component reaches the repeated color's full quota.
# placement_legality:
#   - Each object is placed by reserving its full rectangular bbox via random_free_location with
#     padding=1, so components never overlap, never touch 4-neighbor-wise, and stay separated
#     from the legend by a moat of background.
#   - Component dimensions are sampled only when the colored-capacity under the density bounds can
#     accommodate the required quota pixels, so deleted and under-quota roles are physically
#     realizable inside the chosen rectangles.
# visual_observability_and_cleanliness:
#   - Solvers can see repeated legend swatches directly; no hidden dependence on numeric color ids
#     or ordering is used.
#   - Deleted objects keep a recognizable skeleton after the transformation, so the output does
#     not look like arbitrary object removal.
#   - Kept objects include both "missing one kind" counterexamples and "all kinds present but one
#     quota short" counterexamples, which separates the two reasoning dimensions cleanly.
# inferability_check:
#   - If the counting dimension failed to matter, the under-quota all-kinds component would be
#     transformed by a D6-only set solver; the generator forbids that degeneration.
#   - If the partial-deletion dimension failed to matter, matched components would disappear
#     completely as in D5-only counting-full-delete; the generator forbids that by preserving
#     non-legend residue in every matched component.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, dict[int, int]]:
    """
    Detect the smallest top-left legend whose bottom and right edges are one color.

    Returns (size, border_color, requirement_counts), where requirement_counts counts every
    non-bg, non-border pixel inside the legend interior.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_color = int(g[sz - 1, sz - 1])
        if border_color == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_color) and np.all(g[0:sz, sz - 1] == border_color):
            inside = g[0 : sz - 1, 0 : sz - 1].ravel()
            counts = Counter(int(v) for v in inside if int(v) not in (bg, border_color))
            return int(sz), int(border_color), dict(counts)
    assert False, "no legend border found in grid"


def _matches_requirements(component_counts: Counter[int], legend_counts: dict[int, int]) -> bool:
    return all(component_counts.get(color, 0) >= need for color, need in legend_counts.items())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Punch out legend-colored pixels inside components that satisfy legend quotas.

    Input structure:
    - The outer frame reveals the background color, and the top-left corner contains a small
      legend square whose bottom edge and right edge share one border color.
    - Inside that legend square, target colors may repeat; repeated swatches are meaningful and
      represent larger required counts rather than duplicate decoration.
    - Everywhere else, non-background cells form separated 4-connected rectangular objects built
      from black scaffold cells plus sprinkled colored pixels.

    Rule:
    - Read the legend interior as a per-color requirement table by counting how many times each
      non-bg, non-border color appears there.
    - For every 4-connected non-background component in the original input, count its own pixels
      by color and test whether each legend color appears at least as many times as the legend
      requires.
    - If a component meets the full quota table, erase only the pixels in that component whose
      colors belong to the legend table; leave all non-legend pixels of that same component
      unchanged.
    - Components that miss any quota stay exactly as they are.

    Key insight: the legend now specifies both which colors matter and how many pixels of each
    color are needed. Matching happens at the component level by counting, but the resulting
    deletion happens at the pixel level only for legend-colored cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _border_color, legend_counts = _find_legend(g, bg)
    assert legend_counts, "legend interior contains no target colors"
    legend_colors = set(legend_counts)

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        component_counts = Counter(int(g[r, c]) for r, c in pts)
        if _matches_requirements(component_counts, legend_counts):
            for r, c in pts:
                if int(g[r, c]) in legend_colors:
                    out[r, c] = bg
    return out


# --------------- generator (combo d5-counting + d6-partial-delete) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3
COUNT_PATTERNS = [(2, 1, 1), (2, 2, 1), (3, 1, 1)]

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

MAX_RETRIES = 800


def _sample_requirements() -> dict[int, int]:
    targets = random.sample(PALETTE, NUM_TARGETS)
    pattern = list(random.choice(COUNT_PATTERNS))
    random.shuffle(pattern)
    return {color: count for color, count in zip(targets, pattern)}


def _place_legend(inp: np.ndarray, out: np.ndarray, requirements: dict[int, int]) -> None:
    """Draw the legend and place repeated swatches while keeping one repeated color isolated."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    repeated_colors = [color for color, count in requirements.items() if count > 1]
    anchor_color = random.choice(repeated_colors)

    positions = [(0, 0)]
    available = [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    positions.extend(random.sample(available, sum(requirements.values()) - 1))

    cells = [anchor_color]
    remaining = requirements.copy()
    remaining[anchor_color] -= 1
    for color, count in remaining.items():
        cells.extend([color] * count)
    tail = cells[1:]
    random.shuffle(tail)
    cells = [cells[0]] + tail

    for color, (r, c) in zip(cells, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims(min_needed: int) -> tuple[int, int]:
    """Sample dimensions whose colored capacity can hold the required informative pixels."""
    min_area = max(6, math.ceil(min_needed / DENSITY_HI))
    for _ in range(200):
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
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _sprinkle_patch(h: int, w: int, colors: list[int]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], len(colors))
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _deleted_patch(h: int, w: int, requirements: dict[int, int]) -> np.ndarray:
    non_targets = [color for color in PALETTE if color not in requirements]
    min_needed = sum(requirements.values()) + 1
    n_colored = _sample_colored_count(h * w, min_needed)

    colors: list[int] = []
    for color, need in requirements.items():
        colors.extend([color] * need)
    colors.append(random.choice(non_targets))
    pool = list(requirements) * 3 + non_targets * 2
    colors.extend(random.choice(pool) for _ in range(n_colored - len(colors)))
    random.shuffle(colors)
    return _sprinkle_patch(h, w, colors)


def _under_quota_patch(h: int, w: int, requirements: dict[int, int]) -> np.ndarray:
    shortage_color = random.choice([color for color, need in requirements.items() if need > 1])
    min_needed = sum(requirements.values()) - 1
    n_colored = _sample_colored_count(h * w, min_needed)

    colors: list[int] = []
    for color, need in requirements.items():
        take = need - 1 if color == shortage_color else need
        colors.extend([color] * take)

    extras_pool = [color for color in PALETTE if color != shortage_color]
    colors.extend(random.choice(extras_pool) for _ in range(n_colored - len(colors)))
    random.shuffle(colors)
    return _sprinkle_patch(h, w, colors)


def _missing_kind_patch(h: int, w: int, requirements: dict[int, int]) -> np.ndarray:
    omitted = random.choice(list(requirements))
    remaining_targets = [color for color in requirements if color != omitted]
    included_target_count = random.randint(0, len(remaining_targets))
    included_targets = random.sample(remaining_targets, included_target_count)

    n_colored = _sample_colored_count(h * w, 1)
    allowed = [color for color in PALETTE if color != omitted] + included_targets
    colors = [random.choice(allowed) for _ in range(n_colored)]
    return _sprinkle_patch(h, w, colors)


def _component_patch(h: int, w: int, requirements: dict[int, int], role: str) -> np.ndarray:
    if role == "deleted":
        return _deleted_patch(h, w, requirements)
    if role == "under_quota":
        return _under_quota_patch(h, w, requirements)
    if role == "missing_kind":
        return _missing_kind_patch(h, w, requirements)
    raise ValueError(f"unknown role: {role}")


def _min_needed_for_role(requirements: dict[int, int], role: str) -> int:
    if role == "deleted":
        return sum(requirements.values()) + 1
    if role == "under_quota":
        return sum(requirements.values()) - 1
    if role == "missing_kind":
        return 1
    raise ValueError(f"unknown role: {role}")


def _component_roles(component_count: int) -> list[str]:
    deleted_count = component_count - NUM_KEPT
    kept_roles = ["under_quota", "missing_kind", random.choice(["under_quota", "missing_kind"])]
    roles = ["deleted"] * deleted_count + kept_roles
    random.shuffle(roles)
    return roles


def _apply_partial_deletion(patch: np.ndarray, requirements: dict[int, int]) -> np.ndarray:
    out_patch = patch.copy()
    legend_colors = np.array(list(requirements), dtype=int)
    out_patch[np.isin(out_patch, legend_colors)] = BG
    return out_patch


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    requirements = _sample_requirements()
    _place_legend(inp, out, requirements)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = _component_roles(component_count)

    for role in roles:
        h, w = _component_dims(_min_needed_for_role(requirements, role))
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
        paste(out, _apply_partial_deletion(patch, requirements) if role == "deleted" else patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
