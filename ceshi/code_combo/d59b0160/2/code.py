# Combo variant of d59b0160 (index 2)
# combo_spec: D1/1+D16/0
# Name: 精确颜色匹配+图例颜色空间相邻约束

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior trace:
#    - The legend is still a 4x4 top-left box with border color 3 and exactly 3 target colors.
#    - Every non-legend object is still one 4-connected rectangle because the pasted patch is filled
#      with 0, so the object component is the whole rectangle rather than only the colored pixels.
#    - The fused deletion test now has two gates:
#        1) visible non-zero palette == legend palette exactly
#        2) inside that same component, every pair of legend colors has at least one 4-neighbor
#           contact edge somewhere
#    - Example with legend {1,4,8}:
#        original:  {0,1,4,8,6} -> delete because it contains all targets
#        D1 only:   {1,4,8,6}    -> keep because of extra color 6
#        D16 only:  {1,4,8} with isolated pixels -> keep because targets do not all touch
#        combo:     delete only an object like a 2x2 motif [1 4; 8 4], whose non-zero palette is
#                   exactly {1,4,8} and whose three target colors pairwise touch
#      Therefore the generator must visibly include:
#        - `exact_touch`: exact palette + touching -> deleted
#        - `exact_separated`: exact palette + non-touching -> kept
#        - `superset_touch`: all targets touching + one extra color -> kept
# b) Complexity calibration:
#    - Keep the original legend format, grid size, palette, and separated-rectangle layout so the
#      task stays recognizable as d59b0160.
#    - Limit the kept branches to three repeated roles (`exact_separated`, `superset_touch`,
#      `subset`) so the rule difference is learnable from 2-4 examples instead of exploding into
#      many edge cases.
#    - Force every sample to contain at least one `exact_separated` object and one `superset_touch`
#      object; this makes both halves of the combo observable rather than relying on chance.
# c) Implicit assumption audit:
#    - `_bg_boundary` assumes the outer border reveals the real background. Generator keeps bg=7.
#    - `_find_legend` assumes a top-left box whose bottom and right edges share one border color.
#      Generator preserves that exact geometry.
#    - `components_mask(g != bg)` includes the legend component too. Under the combo rule that is
#      safe because the legend component also contains border color 3, so its visible palette is not
#      equal to the target set. Generator also keeps target colors away from value 3.
#    - Raw component colors include the internal filler 0, so exact matching must ignore both bg and
#      filler 0. Otherwise no non-legend rectangle could ever match exactly.
#    - Pairwise-touch checking assumes the legend has exactly 3 colors; generator enforces that and
#      never emits an empty legend.
# d) Placement legality:
#    - Components are still pasted as full rectangles and positioned with
#      `random_free_location(..., padding=1, padding_connectivity=8)`, so rectangles never overlap,
#      never go out of bounds, and keep a visible gap from each other and from the legend.
#    - Local motifs are built entirely inside the patch before placement, so the generator does not
#      depend on absolute grid position for correctness.
# e) Visual observability:
#    - `exact_separated` creates the key "same colors, different fate" contrast against deleted
#      `exact_touch`: both have exactly the legend colors, but only the tight-touch version vanishes.
#    - `superset_touch` creates the complementary contrast: it visibly has the same touching target
#      cluster as a deleted object, yet survives because an extra non-target color is present.
# f) Visual cleanliness:
#    - The grid still contains only one legend and several separated rectangles; no extra markers,
#      overlays, or hidden bookkeeping are introduced.
#    - The touching witness is a compact 2x2 target block, while the non-touch witness spreads the
#      three target colors apart; the two branches are easy to distinguish by eye.
# g) Color inferability:
#    - The rule depends only on visibly repeated palette membership and 4-neighbor contact.
#      No ordering, arithmetic, or color-id semantics are used.
# 9) New solver assumptions checked against the generator:
#    - `main()` assumes deleted objects can be recognized from their visible non-zero palette; every
#      generated object still uses 0 only as internal filler.
#    - `main()` assumes `exact_separated` objects really fail the touch test; generator constructs
#      them from two isolated singleton target colors plus multiple copies of a third target color
#      placed only in cells that are not 4-adjacent to those singletons.
#    - `main()` assumes `superset_touch` objects still satisfy the touch test; generator embeds the
#      same 2x2 touching witness as deleted objects and adds the extra color elsewhere.


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
    """Visible object palette for matching: ignore both the global bg and the internal filler 0."""
    cols = {int(g[r, c]) for r, c in pts}
    cols.discard(bg)
    cols.discard(0)
    return cols


def _legend_colors_pairwise_touch(g: np.ndarray, pts: np.ndarray, legend_colors: set[int]) -> bool:
    """
    Return True iff all legend colors are present in this component and every pair of
    legend colors has at least one 4-neighbor contact edge inside the component.
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


def _is_combo_match(g: np.ndarray, pts: np.ndarray, bg: int, legend_colors: set[int]) -> bool:
    """Deletion predicate for the fused rule."""
    return _component_palette(g, pts, bg) == legend_colors and _legend_colors_pairwise_touch(g, pts, legend_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove only those components that are both exact legend-palette matches and target-touch matches.

    Input structure:
    - The outer frame still reveals the global background color.
    - The top-left corner still contains a small legend square whose bottom and right edges share one
      border color, while the 3x3 interior shows exactly three target colors.
    - Elsewhere, each object is a separate 4-connected rectangle made from a 0-colored body plus
      scattered non-zero pixels from the same palette.

    Rule:
    - Read the target color set from the legend interior.
    - For each 4-connected component, collect its visible non-zero palette, ignoring the 0 body.
    - Delete the component only if that visible palette is exactly equal to the legend palette and,
      inside the component, every pair of legend colors has at least one direct 4-neighbor contact.
    - Keep components that miss a target color, contain any extra non-target color, or have the
      right palette but with the target colors spatially separated.

    Key insight: The legend now specifies a joint signature of palette and topology; matching the
    colors is not enough, and matching the touch pattern is not enough either.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        if _is_combo_match(g, pts, bg, legend_colors):
            for r, c in pts:
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

MAX_RETRIES = 1200


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


def _color_budget(h: int, w: int, min_count: int) -> tuple[int, int]:
    """Return a feasible [lo, hi] count range under the observed density schedule."""
    area = h * w
    lo = max(min_count, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    return lo, hi


def _all_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _safe_anchor_cells(
    h: int, w: int, blocker_a: tuple[int, int], blocker_b: tuple[int, int]
) -> list[tuple[int, int]]:
    """Cells not equal to or 4-adjacent to either blocker cell."""
    out: list[tuple[int, int]] = []
    br0, bc0 = blocker_a
    br1, bc1 = blocker_b
    for r, c in _all_coords(h, w):
        if (r, c) == blocker_a or (r, c) == blocker_b:
            continue
        if abs(r - br0) + abs(c - bc0) == 1:
            continue
        if abs(r - br1) + abs(c - bc1) == 1:
            continue
        out.append((r, c))
    return out


def _best_exact_separated_layout(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    """
    Choose two singleton target positions that are not 4-adjacent, maximizing the remaining
    cells available for the third target color while preserving non-touch.
    """
    coords = _all_coords(h, w)
    best: tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None = None
    for i, coord_a in enumerate(coords):
        for coord_b in coords[i + 1 :]:
            if abs(coord_a[0] - coord_b[0]) + abs(coord_a[1] - coord_b[1]) <= 1:
                continue
            safe = _safe_anchor_cells(h, w, coord_a, coord_b)
            if best is None or len(safe) > len(best[2]):
                best = (coord_a, coord_b, safe)
    if best is None:
        raise ValueError("no separated layout for exact-separate role")
    return best


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """
    Build one component patch under a role-specific witness pattern.

    Roles:
    - exact_touch: exact target palette and complete pairwise target contact -> deleted
    - exact_separated: exact target palette but incomplete target contact -> kept
    - superset_touch: touching target witness plus an extra non-target color -> kept
    - subset: missing at least one target color -> kept
    """
    patch = np.zeros((h, w), dtype=int)
    coords = _all_coords(h, w)
    non_targets = [v for v in PALETTE if v not in targets]

    if role == "exact_touch":
        lo, hi = _color_budget(h, w, 4)
        n_colored = random.randint(lo, hi)
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
        used = set()
        for coord, color in motif:
            patch[coord] = color
            used.add(coord)
        remaining = n_colored - len(motif)
        if remaining > 0:
            filler_coords = random.sample([c for c in coords if c not in used], remaining)
            for coord in filler_coords:
                patch[coord] = random.choice(targets)
        return patch

    if role == "superset_touch":
        lo, hi = _color_budget(h, w, 5)
        n_colored = random.randint(lo, hi)
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
        used = set()
        for coord, color in motif:
            patch[coord] = color
            used.add(coord)

        extras = [c for c in coords if c not in used]
        extra_coord = random.choice(extras)
        patch[extra_coord] = random.choice(non_targets)
        used.add(extra_coord)

        remaining = n_colored - len(used)
        if remaining > 0:
            filler_coords = random.sample([c for c in coords if c not in used], remaining)
            pool = targets + non_targets
            for coord in filler_coords:
                patch[coord] = random.choice(pool)
        return patch

    if role == "exact_separated":
        blocker_a, blocker_b, safe = _best_exact_separated_layout(h, w)
        lo, hi = _color_budget(h, w, 3)
        max_count = len(safe) + 2
        if max_count < 3:
            raise ValueError("exact_separated has too few safe cells")
        hi = min(hi, max_count)
        lo = min(max(lo, 3), hi)
        n_colored = random.randint(lo, hi)

        anchor_color, single_a, single_b = random.sample(targets, 3)
        anchor_coords = random.sample(safe, n_colored - 2)
        for coord in anchor_coords:
            patch[coord] = anchor_color
        patch[blocker_a] = single_a
        patch[blocker_b] = single_b
        return patch

    if role == "subset":
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        lo, hi = _color_budget(h, w, 1)
        n_colored = random.randint(lo, hi)
        present_targets = random.sample(targets, used_target_count)
        allowed = non_targets + present_targets
        chosen_coords = random.sample(coords, n_colored)
        if present_targets:
            seeded_colors = present_targets[:]
            seeded_coords = chosen_coords[:used_target_count]
            for color, coord in zip(seeded_colors, seeded_coords):
                patch[coord] = color
            for coord in chosen_coords[used_target_count:]:
                patch[coord] = random.choice(allowed)
        else:
            for coord in chosen_coords:
                patch[coord] = random.choice(non_targets)
        return patch

    raise ValueError(f"unknown role: {role}")


def _component_roles(component_count: int) -> list[str]:
    """Guarantee visibility of both combo constraints in every sample."""
    deleted_count = component_count - NUM_KEPT
    roles = ["exact_touch"] * deleted_count + ["exact_separated", "superset_touch", "subset"]
    random.shuffle(roles)
    return roles


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input whose intended output is exactly `main(inp)`."""
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
        if role != "exact_touch":
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
