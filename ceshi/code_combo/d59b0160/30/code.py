# Combo variant of d59b0160 (index 30)
# combo_spec: D4/3+D8/2
# Name: 双层背景+排除颜色0连通
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Original rule labels every non-7 pixel, so a black-0 carrier rectangle with two
#    colored islands is one large component. Fused rule treats 0 as internal blank
#    space and excludes it from labeling, so the same carrier splits into separate
#    4-connected colored islands. Example with legend {1,2,4}: if one island contains
#    {1,2,4} and another contains {5,6}, the original task would erase the whole
#    carrier, while the fused task erases only the first island and leaves the 0 cells
#    plus the {5,6} island intact.
# b) Complexity calibration:
#    The generator uses a small number of large 0-rectangles ("carriers"), each holding
#    2-3 colored islands. This makes the effect of color-0-as-background visible without
#    introducing too many independent degrees of freedom or too much clutter.
# c) Implicit assumptions broken by the fusion:
#    - `g != bg` is no longer the right component mask; color 0 must also be excluded.
#    - Deleting a component no longer removes the surrounding 0 region, because that 0
#      region is not part of the component anymore.
#    - A rectangle full of 0 with sprinkled colors can no longer be treated as one
#      object by construction; 0 now actively splits it into subcomponents.
# d) Placement legality:
#    Every carrier is pasted as a solid rectangle of non-7 pixels, so
#    `random_free_location(..., transparent=7, padding=1)` prevents overlap, touching,
#    and out-of-bounds placement. Inside each carrier, islands are kept 8-neighbor
#    separated so they remain distinct under 4-connectivity and visually readable.
# e) Visual observability:
#    The contrast between outer orange 7 and inner black 0 is human-visible. At least
#    one carrier is always mixed: it contains both a deletable island and a kept island,
#    so the fused output visibly differs from the legacy behavior on every accepted grid.
# f) Visual cleanliness:
#    Islands stay one cell away from carrier borders and one cell away from each other.
#    This preserves clear black gutters and avoids ambiguous overlaps.
# g) Inferability:
#    Colors are not remapped; only membership in the legend set matters. The repeated
#    pattern "black carrier + several colored islands + only some islands disappear"
#    provides a structural cue that 0 is acting as a visible empty separator.
# 9) New solver assumptions and generator guarantees:
#    - Border majority still identifies the outer background as 7: guaranteed because
#      carriers and legend do not dominate the outer frame.
#    - The smallest top-left L-border is still the legend: guaranteed by reserving that
#      corner and keeping carriers separated from it.
#    - Kept islands are missing at least one target color: guaranteed by construction.
#    - Mixed carriers exist and make fused != legacy output: checked before accepting.

import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color on the 1-cell outer frame."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete only the colored islands that realize the full legend palette.

    Input structure:
    - Color 7 is the outer background around the whole scene.
    - The top-left corner contains the same bordered legend format as the base task.
    - Elsewhere, many colored pixels sit inside larger black color-0 carrier regions; the
      0 cells are visible blanks inside those carriers rather than part of the components.

    Rule:
    - Read the target color set from the legend interior.
    - Build 4-connected components from pixels whose color is neither 7 nor 0.
    - Erase every such component whose color set contains all legend colors, replacing
      only those colored pixels with 7.
    - Leave color-0 blanks, the legend, and all incomplete components unchanged.

    Key insight:
    - Color 0 plays two roles at once: it is a visible inner background and it also
      breaks connectivity, so one black carrier can contain several separately judged
      colored islands.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    inner_bg = 0

    _sz, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    comps = components_mask((g != bg) & (g != inner_bg), connectivity=4)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out


GRID_H, GRID_W = 16, 16
BG = 7
INNER_BG = 0
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, targets: list[int]) -> None:
    """Draw the top-left legend with exactly the target colors in its interior."""
    s = LEGEND_SIZE
    inp[:s, :s] = BG
    inp[s - 1, :s] = BORDER
    inp[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    side_cells = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        pos for pos in side_cells
        if abs(pos[0] - isolated[0]) + abs(pos[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        inp[r, c] = color


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Base-task behavior used only to reject degenerate generated samples."""
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


def _role_groups(total_components: int) -> list[list[bool]]:
    """
    Partition roles into carrier-local groups.

    Each bool is True for a deletable component and False for a kept component.
    The first carrier is always mixed so the fusion has visible effect.
    """
    deleted = total_components - NUM_KEPT
    remaining = [True] * (deleted - 1) + [False] * (NUM_KEPT - 1)
    random.shuffle(remaining)

    if total_components == 4:
        sizes = [2, 2]
    elif total_components == 5:
        sizes = [2, 3]
    else:
        sizes = [2, 2, 2]

    groups = [[True, False]]
    idx = 0
    for size in sizes[1:]:
        group = remaining[idx : idx + size]
        random.shuffle(group)
        groups.append(group)
        idx += size
    return groups


def _carrier_dims(n_components: int) -> tuple[int, int]:
    """Sample a carrier rectangle large enough to host 2-3 islands."""
    if n_components == 2:
        return random.randint(5, 7), random.randint(5, 7)
    return random.randint(6, 8), random.randint(6, 8)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Small 8-neighbor dilation for carrier-internal spacing."""
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _grow_shape(h: int, w: int, n_cells: int, blocked: np.ndarray) -> np.ndarray | None:
    """Grow a 4-connected colored island inside currently free carrier cells."""
    starts = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    if not starts:
        return None

    for _ in range(40):
        start = random.choice(starts)
        cells = {start}
        while len(cells) < n_cells:
            frontier = set()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) == n_cells:
            return np.array(sorted(cells), dtype=int)
    return None


def _component_colors(n_cells: int, targets: list[int], deleted: bool) -> list[int]:
    """Assign colors while controlling whether the legend set is complete."""
    non_targets = [color for color in PALETTE if color not in targets]
    if deleted:
        colors = list(targets)
        for _ in range(n_cells - NUM_TARGETS):
            colors.append(random.choice(PALETTE))
    else:
        used_target_count = random.randint(0, NUM_TARGETS - 1)
        allowed_targets = random.sample(targets, used_target_count)
        allowed = non_targets + allowed_targets
        colors = [random.choice(allowed) for _ in range(n_cells)]
    random.shuffle(colors)
    return colors


def _build_carrier(group_roles: list[bool], targets: list[int]) -> np.ndarray | None:
    """
    Build one black carrier rectangle with several separated colored islands.

    The carrier is all 0 by default, so legacy behavior would join everything inside
    it into one component, while fused behavior judges the islands independently.
    """
    n_components = len(group_roles)
    for _ in range(80):
        h, w = _carrier_dims(n_components)
        patch = np.full((h, w), INNER_BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        border_block = np.zeros((h, w), dtype=bool)
        border_block[0, :] = True
        border_block[-1, :] = True
        border_block[:, 0] = True
        border_block[:, -1] = True

        roles = list(group_roles)
        random.shuffle(roles)
        ok = True
        for deleted in roles:
            blocked = border_block | _dilate8(occupied)
            lo = NUM_TARGETS if deleted else 3
            hi = 6 if n_components == 3 else 7
            n_cells = random.randint(lo, hi)
            shape = _grow_shape(h, w, n_cells, blocked)
            if shape is None:
                ok = False
                break
            colors = _component_colors(len(shape), targets, deleted)
            for (r, c), color in zip(shape, colors):
                patch[r, c] = color
            occupied[shape[:, 0], shape[:, 1]] = True

        if ok:
            return patch
    return None


def _gen_attempt() -> np.ndarray | None:
    """Construct one candidate input grid and reject it if the fusion is not observable."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, targets)

    total_components = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    for group_roles in _role_groups(total_components):
        patch = _build_carrier(group_roles, targets)
        if patch is None:
            return None
        try:
            r, c = random_free_location(
                inp,
                patch,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)

    fused = main(inp)
    legacy = _legacy_main(inp)
    if np.array_equal(fused, legacy):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
