# Combo variant of d59b0160 (index 35)
# combo_spec: D2/2+D16/0
# Name: 匹配分量过滤保留图例色+图例颜色空间相邻

"""
design_analysis:
  source_task: d59b0160
  preserved_structure:
    grid: 16x16 with frame-defined bg=7
    legend:
      position: fixed top-left 4x4
      border: color 3 on the legend's bottom row and right column
      payload: exactly 3 distinct target colors from {1,2,4,5,6,8,9}
    objects:
      type: separate 4-connected rectangles whose base fill is color 0, with extra colored pixels embedded inside
      spacing: each placed rectangle stays at least one bg cell away from other non-legend objects
  fused_rule:
    old_rule: any component whose color set contains all legend colors is deleted
    new_rule:
      - a component matches only if all legend colors are present and every pair of legend colors has a direct 4-neighbor contact edge somewhere inside that same component
      - when a component matches, it is not erased wholesale; only its legend-colored pixels survive and every other pixel in that component becomes bg
  behavior_trace:
    small_example:
      legend_colors: [1, 4, 8]
      component_input:
        - [1, 4, 0]
        - [8, 4, 0]
      original_output:
        - [7, 7, 7]
        - [7, 7, 7]
      fused_output:
        - [1, 4, 7]
        - [8, 4, 7]
      contrasting_nonmatch:
        input:
          - [1, 0, 4]
          - [0, 8, 0]
        original_output:
          - [7, 7, 7]
          - [7, 7, 7]
        fused_output:
          - [1, 0, 4]
          - [0, 8, 0]
    difference:
      - Compared with the base task, "contains all colors" is no longer enough; the legend colors must physically meet.
      - Compared with D2 alone, a full-palette component can now remain untouched if its legend colors are spatially separated.
      - Compared with D16 alone, matched components do not disappear; they collapse to the legend-colored contact cluster.
  complexity_calibration:
    - Keep the target-color count fixed at 3 so the pairwise-touch condition is learnable from a few examples.
    - Force each generated grid to contain at least one `kept_full_separated` component, so humans can see that "having all legend colors" is insufficient.
    - Keep 1-3 matched components so the filtering branch repeats, making the surviving legend-colored clusters easy to compare across examples.
  implicit_assumption_review:
    - `_bg_boundary` assumes the outer border is still mostly background; the generator keeps the frame mostly clear, so bg=7 dominates there.
    - `_find_legend` assumes the first top-left bottom/right bordered square is the legend; the generator preserves that exact geometry.
    - `components_mask(g != bg, connectivity=4)` still groups each object rectangle as one component before transformation; this remains valid because both matching predicates are component-level.
    - The old `legend_colors.issubset(cols)` test is no longer sufficient; matching now depends on an explicit adjacency graph among target-colored cells.
    - The old "overwrite matched component with bg" loop is no longer valid; matched components must be filtered pixel-by-pixel.
    - The legend itself must not satisfy the new pairwise-touch predicate. The legend placement keeps one target isolated from the border-connected cells, preventing a complete touch graph among the three target colors.
  placement_legality:
    - Rectangles are still sampled first, then placed with `random_free_location(..., padding=1, padding_connectivity=8)` so they never overlap or touch other non-bg objects.
    - The matched witness motif is a 2x2 target-color block, which fits every allowed rectangle size.
    - The separated-full witness uses three target cells chosen with pairwise Manhattan distance > 1, preventing accidental direct contact.
  visual_observability:
    - Matched components visibly shrink from solid 0-filled rectangles to sparse legend-colored clusters, so the D2 filtering effect is obvious.
    - At least one untouched component still contains all legend colors but not in a touching cluster, making the D16 constraint directly contrast with the matched cases.
    - No hidden ordering or color-id arithmetic is used; the only cues are visible palette membership and visible adjacency.
  cleanliness:
    - The variant keeps the original scene grammar: one legend plus separated rectangles.
    - Extra non-target colors are allowed only as filler, avoiding clutter from many target-color duplicates that would blur the touching predicate.
  new_solver_assumptions_to_guarantee:
    - Every matched component must contain all 3 target colors in a complete 4-neighbor contact graph; the generator hard-codes a 2x2 witness block.
    - Every matched component must also contain non-target pixels so filtering is visible; the solid 0 base guarantees that.
    - Every `kept_full_separated` component must contain all 3 target colors without any target-target contact edge; the generator samples those three coordinates with pairwise Manhattan distance > 1 and forbids additional target-colored filler.
"""

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the top-left legend whose bottom edge and right edge share one border color.
    Return (size, border_color, interior_target_colors).
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


def _legend_colors_pairwise_touch(g: np.ndarray, pts: np.ndarray, legend_colors: set[int]) -> bool:
    """
    Return True iff all legend colors are present in this component and every pair of
    legend colors has at least one direct 4-neighbor contact edge inside the component.
    """
    present = {int(g[r, c]) for r, c in pts if int(g[r, c]) in legend_colors}
    if present != legend_colors:
        return False

    pt_set = {tuple(map(int, rc)) for rc in pts}
    touches = {color: set() for color in legend_colors}
    for r, c in pt_set:
        color = int(g[r, c])
        if color not in legend_colors:
            continue
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) not in pt_set:
                continue
            other = int(g[nr, nc])
            if other in legend_colors and other != color:
                touches[color].add(other)
                touches[other].add(color)

    return all((legend_colors - {color}).issubset(touches[color]) for color in legend_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter only those connected components whose legend colors form a touching cluster.

    Input structure:
    - The outer frame is mostly background, and the top-left corner contains a small legend square.
    - That legend uses one border color along its bottom and right edges, while its interior displays the target color set.
    - Outside the legend, the non-background cells form separate multicolor 4-connected components on the background.

    Rule:
    - Read the target color set from the legend interior.
    - Examine each 4-connected non-background component independently.
    - A component matches only if it contains every target color and every pair of target colors has a direct 4-neighbor contact somewhere inside the component.
    - For a matched component, keep only the pixels whose own colors belong to the legend set and turn the rest of that component into background; leave all non-matching components unchanged.

    Key insight: The legend now specifies both a palette and a local topology, and matched objects are reduced to just the legend-colored touching witness rather than being deleted entirely.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(g != bg, connectivity=4):
        if not _legend_colors_pairwise_touch(g, pts, legend_colors):
            continue
        for r, c in pts:
            if int(g[r, c]) not in legend_colors:
                out[r, c] = bg
    return out


# --------------- generator (combo variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

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


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the legend while keeping one target disconnected from the border-connected target cells."""
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
    """Sample a component bounding box according to the original aspect mix."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _color_budget(h: int, w: int, min_count: int) -> int:
    area = h * w
    lo = max(min_count, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    return random.randint(lo, hi)


def _all_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _sample_pairwise_nonadjacent_coords(h: int, w: int, k: int) -> list[tuple[int, int]]:
    """Sample k coords with pairwise Manhattan distance > 1."""
    coords = _all_coords(h, w)
    for _ in range(80):
        random.shuffle(coords)
        picked: list[tuple[int, int]] = []
        for coord in coords:
            if all(abs(coord[0] - rr) + abs(coord[1] - cc) > 1 for rr, cc in picked):
                picked.append(coord)
                if len(picked) == k:
                    return picked
    raise ValueError("failed to sample separated coords")


def _filtered_patch(patch: np.ndarray, targets: list[int]) -> np.ndarray:
    """Keep only legend colors inside a matched component; everything else becomes bg."""
    keep = np.isin(patch, np.array(targets, dtype=int))
    return np.where(keep, patch, BG).astype(int)


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """
    Build an h x w solid-0 component.

    Roles:
    - matched: contains a 2x2 target-color witness with complete pairwise contact.
    - kept_full_separated: contains all target colors, but they are pairwise non-adjacent.
    - kept_partial: omits at least one target color.
    """
    patch = np.zeros((h, w), dtype=int)
    non_targets = [v for v in PALETTE if v not in targets]
    free = set(_all_coords(h, w))

    if role == "matched":
        n_colored = _color_budget(h, w, 4)
        top = random.randint(0, h - 2)
        left = random.randint(0, w - 2)
        trio = targets[:]
        random.shuffle(trio)
        motif = [
            ((top, left), trio[0]),
            ((top, left + 1), trio[1]),
            ((top + 1, left), trio[2]),
            ((top + 1, left + 1), trio[1]),
        ]
        for coord, color in motif:
            patch[coord] = color
            free.discard(coord)
    elif role == "kept_full_separated":
        n_colored = _color_budget(h, w, NUM_TARGETS)
        coords = _sample_pairwise_nonadjacent_coords(h, w, NUM_TARGETS)
        shuffled_targets = targets[:]
        random.shuffle(shuffled_targets)
        for color, coord in zip(shuffled_targets, coords):
            patch[coord] = color
            free.discard(coord)
    elif role == "kept_partial":
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        n_colored = _color_budget(h, w, used_target_count)
        if used_target_count > 0:
            coords = _sample_pairwise_nonadjacent_coords(h, w, used_target_count)
            subset = random.sample(targets, used_target_count)
            for color, coord in zip(subset, coords):
                patch[coord] = color
                free.discard(coord)
    else:
        raise ValueError(f"unknown role: {role}")

    remaining = n_colored - int(np.count_nonzero(patch))
    if remaining > 0:
        filler_coords = random.sample(list(free), remaining)
        filler_colors = [random.choice(non_targets) for _ in range(remaining)]
        for coord, color in zip(filler_coords, filler_colors):
            patch[coord] = color

    return patch


def _component_roles(component_count: int) -> list[str]:
    deleted_count = component_count - NUM_KEPT
    kept_full_count = random.randint(1, 2)
    kept_partial_count = NUM_KEPT - kept_full_count
    roles = (
        ["matched"] * deleted_count
        + ["kept_full_separated"] * kept_full_count
        + ["kept_partial"] * kept_partial_count
    )
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose output follows the fused rule."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    for role in _component_roles(component_count):
        h, w = _component_dims()
        try:
            patch = _component_patch(h, w, targets, role)
        except ValueError:
            return None
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
        if role == "matched":
            paste(out, _filtered_patch(patch, targets), r, c)
        else:
            paste(out, patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    """Sample a valid input grid whose output exhibits both the D2 and D16 changes."""
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
