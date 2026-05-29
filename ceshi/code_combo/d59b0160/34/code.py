# Combo variant of d59b0160 (index 34)
# combo_spec: D8/1+D18/1
# Name: 互补色匹配+黑色背景
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original task: the legend shows the exact color set that must appear inside a
#      non-background 4-connected component, and the global background is orange 7.
#    - Fused task: the outer background becomes black 0, and each legend color X now
#      stands for the component color (9-X) instead of X itself.
#    - Example: if the legend interior shows {1,4,7}, the deletion set is {8,5,2}.
#      A maroon carrier rectangle containing accent pixels {8,5,2} is erased; a similar
#      rectangle containing {1,4,7} is kept. Under the original rule that kept rectangle
#      would be deleted, so the fusion changes the actual decision, not just the palette.
# b) Complexity calibration:
#    - The generator always includes at least one "mirror decoy" kept component that
#      contains the raw legend colors but misses at least one complementary color. This
#      makes the complement relation visible and guarantees fused behavior differs from
#      exact-match behavior on the same input.
#    - Components remain large solid rectangles with a single carrier color plus a modest
#      number of accent pixels. The scene stays readable on black while still showing
#      repeated color evidence.
# c) Implicit assumptions broken by the fusion:
#    - The original generator relied on color 0 as an internal filler. That becomes
#      impossible once 0 is the global background, because such filler would disconnect
#      or erase components. Here every component uses a nonzero carrier color instead.
#    - Exact legend matching is no longer valid; `main()` must map legend colors through
#      `9 - x` before testing each component.
#    - The edge-majority background detector still works only if ordinary components do
#      not occupy the outer frame too heavily, so the generator keeps them one cell away
#      from the boundary.
#    - If a legend color were 9, its complement would be 0, which cannot belong to a
#      standard non-background component. To keep samples realizable and inferable, the
#      generator excludes 9 from the legend pool while the solver still applies the
#      general complement formula.
# d) Placement legality:
#    - The whole 4x4 legend window is reserved during placement, not just its nonzero
#      cells, so black legend interior cells cannot be overwritten accidentally.
#    - Every rectangular component is placed with `border=1` and `padding=1`, which keeps
#      components in bounds, non-overlapping, and separated by at least one black cell.
# e) Visual observability:
#    - Black background versus nonblack carrier rectangles makes the new background role
#      immediately visible.
#    - The deletion signal is observable because disappearing objects share the color set
#      complementary to the legend, while at least one surviving object repeats the raw
#      legend colors directly.
# f) Visual cleanliness:
#    - The legend remains compact in the top-left corner, and the ordinary objects occupy
#      the interior with enforced gaps. There is no layering or overlap between signals.
# g) Inferability of the color change:
#    - The color change is not a free-for-all remapping. It is a single involution
#      `X <-> 9-X`, repeated across all three legend colors in a sample and across all
#      deletable objects. The mirror decoy provides a direct contrast against exact-match
#      reasoning.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes the smallest top-left L-shaped border is the legend. The generator
#      keeps that corner exclusive to the legend and reserves the entire legend window.
#    - `main()` assumes deletable components are ordinary nonblack 4-connected objects.
#      The generator uses solid nonzero rectangles, so every pasted object is exactly one
#      component.
#    - `main()` assumes kept components are missing at least one required complementary
#      color. The generator enforces this explicitly for both mirror-decoy and partial
#      kept components.

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import components_mask, paste, random_free_location


GRID_H, GRID_W = 16, 16
BG = 0
BORDER = 3
LEGEND_SIZE = 4
NUM_TARGETS = 3

LEGEND_POOL = [1, 2, 4, 5, 6, 7, 8]
ALL_NONZERO = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CARRIER = 9

SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (6, 11)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 7)
MED_W = (2, 8)

DENSITY_LO = 0.16
DENSITY_HI = 0.42

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 18]
NUM_KEPT = 3
MAX_RETRIES = 1600


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color on the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """Read the top-left legend bordered on its bottom row and right column."""
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_color = int(g[sz - 1, sz - 1])
        if border_color == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_color) and np.all(g[0:sz, sz - 1] == border_color):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(border_color)
            return int(sz), border_color, colors
    raise AssertionError("no legend border found in grid")


def _complement(colors: set[int] | list[int]) -> set[int]:
    """Map each ARC color x to its complement 9-x."""
    return {int(9 - int(color)) for color in colors}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose colors realize the legend's complementary palette.

    Input structure:
    - Color 0 is the global background across the whole grid.
    - The top-left corner still contains a 4x4 legend window with a green color-3
      border on its bottom row and right column.
    - Elsewhere, nonblack multicolor rectangles form separate 4-connected components.

    Rule:
    - Read the legend colors from the 3x3 interior.
    - Convert each legend color X into the required component color 9-X.
    - Erase every nonblack 4-connected component whose color set contains all of those
      complementary colors, replacing the whole component with black 0.
    - Leave the legend and all incomplete components unchanged.

    Key insight:
    - The legend is no longer a literal palette copy. It is a coded specification, and
      the objects that disappear are the ones carrying the color complements of the
      legend rather than the legend colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"
    required_colors = _complement(legend_colors)

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if required_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the standard 4x4 legend at the top-left corner."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        pos for pos in border_side
        if abs(pos[0] - isolated[0]) + abs(pos[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color
        out[r, c] = color


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions with the original tall/wide/medium flavor."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _accent_count(area: int) -> int:
    lo = max(NUM_TARGETS, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _color_choices_excluding(excluded: set[int]) -> list[int]:
    return [color for color in ALL_NONZERO if color not in excluded]


def _build_deleted_patch(h: int, w: int, legend_targets: list[int], required_targets: list[int]) -> np.ndarray:
    """
    Build a deletable component.

    It always contains the full complementary target set and avoids raw legend colors in
    its extra accents so the complement relation stays visually cleaner.
    """
    patch = np.full((h, w), CARRIER, dtype=int)
    n_accents = _accent_count(h * w)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_accents)

    excluded = set(legend_targets) | {CARRIER}
    extras_pool = _color_choices_excluding(excluded)
    preferred_pool = required_targets + extras_pool

    colors = list(required_targets)
    for _ in range(n_accents - len(required_targets)):
        colors.append(random.choice(preferred_pool))
    random.shuffle(colors)

    for (r, c), color in zip(coords, colors):
        patch[r, c] = int(color)
    return patch


def _build_mirror_kept_patch(h: int, w: int, legend_targets: list[int], required_targets: list[int]) -> np.ndarray:
    """
    Build a kept component that mirrors the raw legend colors.

    It includes all legend colors but intentionally misses at least one complementary
    color that is absent from the legend itself, making it the canonical anti-baseline
    decoy for this combo.
    """
    patch = np.full((h, w), CARRIER, dtype=int)
    n_accents = _accent_count(h * w)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_accents)

    legend_set = set(legend_targets)
    required_set = set(required_targets)
    missable = sorted(required_set - legend_set)
    assert missable, "legend set unexpectedly closed under complement"
    missing_color = random.choice(missable)

    extras_pool = [
        color for color in ALL_NONZERO
        if color != CARRIER and color != missing_color and color not in required_set
    ]
    colors = list(legend_targets)
    for _ in range(n_accents - len(legend_targets)):
        colors.append(random.choice(legend_targets + extras_pool))
    random.shuffle(colors)

    for (r, c), color in zip(coords, colors):
        patch[r, c] = int(color)
    return patch


def _build_partial_kept_patch(h: int, w: int, legend_targets: list[int], required_targets: list[int]) -> np.ndarray:
    """Build a kept component that contains only a strict subset of the required colors."""
    patch = np.full((h, w), CARRIER, dtype=int)
    n_accents = _accent_count(h * w)
    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_accents)

    subset_size = random.randint(0, NUM_TARGETS - 1)
    subset = random.sample(required_targets, subset_size)
    extras_pool = [
        color for color in ALL_NONZERO
        if color != CARRIER and color not in set(required_targets)
    ]
    colors = list(subset)
    while len(colors) < n_accents:
        pool = extras_pool + legend_targets
        colors.append(random.choice(pool))
    random.shuffle(colors)

    for (r, c), color in zip(coords, colors):
        patch[r, c] = int(color)
    return patch


def _role_plan(component_count: int) -> list[str]:
    """
    Return component roles.

    `delete` components are erased by the fused rule.
    `mirror` is a kept exact-match decoy against the original rule.
    `partial` is a kept near-miss under the fused rule.
    """
    deleted_count = component_count - NUM_KEPT
    roles = ["delete"] * deleted_count + ["mirror"] + ["partial"] * (NUM_KEPT - 1)
    random.shuffle(roles)
    return roles


def _build_patch(role: str, legend_targets: list[int], required_targets: list[int]) -> np.ndarray:
    h, w = _component_dims()
    if role == "delete":
        return _build_deleted_patch(h, w, legend_targets, required_targets)
    if role == "mirror":
        return _build_mirror_kept_patch(h, w, legend_targets, required_targets)
    if role == "partial":
        return _build_partial_kept_patch(h, w, legend_targets, required_targets)
    raise ValueError(f"unknown role: {role}")


def _legend_reserved_canvas(canvas: np.ndarray) -> np.ndarray:
    """Reserve the full legend window during placement, not only its nonzero cells."""
    reserved = canvas.copy()
    reserved[:LEGEND_SIZE, :LEGEND_SIZE] = BORDER
    return reserved


def _legacy_exact_main(grid: np.ndarray) -> np.ndarray:
    """Base behavior with black background but without the complement remapping."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    _sz, _border_color, legend_colors = _find_legend(g, bg)
    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


def _count_mirror_decoys(grid: np.ndarray) -> int:
    """Count kept components that satisfy the old exact-match rule but not the fused rule."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    _sz, _border_color, legend_colors = _find_legend(g, bg)
    required_colors = _complement(legend_colors)

    count = 0
    for pts in components_mask(g != bg, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols) and not required_colors.issubset(cols):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    legend_targets = random.sample(LEGEND_POOL, NUM_TARGETS)
    required_targets = sorted(9 - color for color in legend_targets)
    _place_legend(inp, out, legend_targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    roles = _role_plan(component_count)

    placement_canvas = _legend_reserved_canvas(inp)
    for role in roles:
        patch = _build_patch(role, legend_targets, required_targets)
        try:
            r, c = random_free_location(
                placement_canvas,
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(inp, patch, r, c)
        paste(placement_canvas, patch, r, c)
        if role != "delete":
            paste(out, patch, r, c)

    fused = main(inp)
    if not np.array_equal(fused, out):
        return None

    legacy = _legacy_exact_main(inp)
    if np.array_equal(fused, legacy):
        return None

    if _count_mirror_decoys(inp) < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
