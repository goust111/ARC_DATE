# Variant of d59b0160
# Dimension: D16 — 分量中颜色的空间分布约束 (meta)
# Change: 仅检查分量是否包含图例颜色，不关注颜色在分量内的空间分布... → 图例颜色必须出现在分量的外接矩形边缘

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location

# Design analysis for the D16 variant
# a) Concrete behavior:
#    The original task deletes any non-bg 4-connected component whose color set contains all legend colors.
#    Here the object layout stays the same: a top-left 4x4 legend and several rectangular 0-based components.
#    The changed event is inside the component test:
#    - Old: a target color counts if it appears anywhere in the component.
#    - New: a target color counts only if some pixel of that color lies on the bbox perimeter
#      (top row, bottom row, left column, or right column) of that connected component.
#    Example trace: if targets are {1, 4, 8} and a 5x5 rectangle contains 1 on the top edge, 4 on the right edge,
#    and 8 only at the center, the old solver erases it because all three colors are present, while the new solver
#    keeps it because 8 never touches the bbox edge.
# b) Complexity calibration:
#    The mechanism is still a single legend-driven filter over connected components. The only extra reasoning is
#    computing a component bbox and restricting the matching colors to its perimeter. This matches the requested
#    "boundary detection" flavor without adding new object classes or transformation stages.
# c) Hidden assumptions review:
#    - _find_legend still works unchanged because the legend encoding itself is unchanged.
#    - The original `cols = {g[r, c] for ...}` relied on "presence anywhere is enough"; that assumption is now false.
#      It must be replaced by a bbox-edge color set.
#    - The original generator split objects into "contains all targets" vs "missing some target". That is no longer
#      sufficient, because a kept object may contain all targets yet still fail the boundary condition. The generator
#      must explicitly construct such decoys.
# d) Placement legality:
#    Components remain full rectangular patches of non-bg cells pasted with one-cell padding, so bbox computation is
#    stable and components never merge. Decoy components are generated with h,w >= 3 so they have an interior cell.
# e) Visual observability:
#    Each generated grid contains at least one deleted component and at least one kept decoy component that contains
#    all legend colors but places one required target only in the interior. This guarantees that the D16 behavior
#    changes the output relative to the original rule on the same input.
# f) Visual cleanliness:
#    The legend stays isolated at the top-left, and each rectangular component occupies its own padded region.
#    The only distinction viewers need to track is whether the legend colors sit on the rim or inside the rectangle.
# g) Color inference:
#    No hidden color-id ordering is used. Colors keep their original role as legend swatches; only their placement
#    relative to the rectangle boundary matters.
# h) Generator validity:
#    Deleted components receive all target colors on boundary cells. Decoy-kept components receive all target colors
#    overall, but one chosen target is forbidden from all boundary cells and appears only in the interior.
# i) New solver assumptions:
#    The solver now assumes every foreground component has a well-defined bbox edge and that the legend colors are
#    non-empty. The generator guarantees this by keeping rectangular foreground components and a 3-color legend.


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


def _bbox_edge_colors(g: np.ndarray, pts: np.ndarray) -> set[int]:
    """Colors from component pixels that lie on the component's axis-aligned bbox perimeter."""
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return {
        int(g[r, c])
        for r, c in pts
        if int(r) in (r0, r1) or int(c) in (c0, c1)
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase connected components only when the legend palette reaches their bounding-frame.

    Input structure:
    - The background color fills most cells.
    - In the top-left corner, a small square legend uses a shared border color on its bottom edge and right edge; the interior lists the target colors.
    - Elsewhere, rectangular 4-connected foreground components contain color-0 filler plus scattered colored pixels.

    Rule:
    - Read the target color set from the legend interior.
    - For each non-background connected component, compute its axis-aligned bounding rectangle.
    - A legend color counts as matched only if that color appears on the rectangle's perimeter, not merely somewhere inside it.
    - Erase the component iff every legend color is represented on that perimeter; otherwise leave it unchanged.

    Key insight: The legend specifies a boundary signature rather than a plain palette-membership test, so interior-only target colors are insufficient.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _sz, _x, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask(g != bg, connectivity=4)
    for pts in comps:
        edge_cols = _bbox_edge_colors(g, pts)
        if legend_colors.issubset(edge_cols):
            for r, c in pts:
                out[int(r), int(c)] = bg
    return out


# --------------- generator (D16 variant) ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights: (mode, weight) --
SHAPE_WEIGHTS = [5, 3, 2]
TALL_H = (8, 16)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 9)

# -- Component fill density (fraction of colored pixels among the 0-filled rectangle) --
DENSITY_LO = 0.10
DENSITY_HI = 0.55

# -- Component count & role split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Generation retry budget --
MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place target colors in its 3x3 interior."""
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


def _component_dims(*, require_interior: bool) -> tuple[int, int]:
    """Sample component dimensions, optionally enforcing a non-empty interior."""
    for _ in range(100):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if not require_interior or (h >= 3 and w >= 3):
            return h, w
    raise RuntimeError("failed to sample component dimensions")


def _boundary_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    ]


def _interior_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
    ]


def _random_color(choices: list[int]) -> int:
    return int(random.choice(choices))


def _component_patch(h: int, w: int, targets: list[int], role: str) -> np.ndarray:
    """Build one rectangular component with a role-specific boundary/interior target layout."""
    area = h * w
    mandatory = NUM_TARGETS if role in {"deleted", "decoy"} else 1
    lo = max(mandatory, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)

    boundary = _boundary_coords(h, w)
    interior = _interior_coords(h, w)
    non_targets = [v for v in PALETTE if v not in targets]

    placements: dict[tuple[int, int], int] = {}
    used: set[tuple[int, int]] = set()

    if role == "deleted":
        for color, pos in zip(targets, random.sample(boundary, NUM_TARGETS)):
            placements[pos] = int(color)
            used.add(pos)
        extra_palette = PALETTE
        edge_forbidden: set[int] = set()
        interior_only_color = None

    elif role == "decoy":
        assert interior, "decoy components require an interior"
        interior_only_color = int(random.choice(targets))
        other_targets = [t for t in targets if t != interior_only_color]
        inner_pos = random.choice(interior)
        placements[inner_pos] = interior_only_color
        used.add(inner_pos)
        for color, pos in zip(other_targets, random.sample(boundary, len(other_targets))):
            placements[pos] = int(color)
            used.add(pos)
        extra_palette = PALETTE
        edge_forbidden = {interior_only_color}

    elif role == "ordinary":
        keep_count = random.randint(0, NUM_TARGETS - 1)
        present_targets = random.sample(targets, keep_count)
        allowed = non_targets + present_targets
        seed_pos = random.choice(boundary)
        placements[seed_pos] = _random_color(allowed)
        used.add(seed_pos)
        extra_palette = allowed
        edge_forbidden = set()
        interior_only_color = None

    else:
        raise ValueError(f"unknown role: {role}")

    remaining_slots = [(r, c) for r in range(h) for c in range(w) if (r, c) not in used]
    extra_count = n_colored - len(placements)
    if extra_count > len(remaining_slots):
        raise ValueError("not enough free slots for colored pixels")

    for pos in random.sample(remaining_slots, extra_count):
        r, c = pos
        allowed = list(extra_palette)
        if interior_only_color is not None and (r in (0, h - 1) or c in (0, w - 1)):
            allowed = [col for col in allowed if col not in edge_forbidden]
        if interior_only_color is not None and 0 < r < h - 1 and 0 < c < w - 1:
            allowed = list(extra_palette)
        placements[pos] = _random_color(allowed)

    patch = np.zeros((h, w), dtype=int)
    for (r, c), color in placements.items():
        patch[r, c] = int(color)

    return patch


def _component_target_stats(patch: np.ndarray, targets: set[int]) -> tuple[set[int], set[int]]:
    """Return (all_target_colors_present, boundary_target_colors_present) for one patch."""
    h, w = patch.shape
    all_targets = {int(v) for v in np.unique(patch) if int(v) in targets}
    boundary_targets = {
        int(patch[r, c])
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1)) and int(patch[r, c]) in targets
    }
    return all_targets, boundary_targets


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid input; return it or None on failure."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    target_set = set(targets)
    _place_legend(inp, out, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    decoy_count = random.randint(1, 2)
    ordinary_count = NUM_KEPT - decoy_count
    roles = (
        ["deleted"] * deleted_count
        + ["decoy"] * decoy_count
        + ["ordinary"] * ordinary_count
    )
    random.shuffle(roles)

    saw_decoy = False
    saw_deleted = False

    for role in roles:
        h, w = _component_dims(require_interior=(role == "decoy"))
        patch = _component_patch(h, w, targets, role)
        all_targets, boundary_targets = _component_target_stats(patch, target_set)

        if role == "deleted" and boundary_targets != target_set:
            return None
        if role == "decoy" and not (all_targets == target_set and boundary_targets != target_set):
            return None
        if role == "ordinary" and all_targets == target_set:
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
        if role != "deleted":
            paste(out, patch, r, c)
        saw_decoy |= role == "decoy"
        saw_deleted |= role == "deleted"

    if not (saw_decoy and saw_deleted):
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
