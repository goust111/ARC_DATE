# Variant of d59b0160
# Dimension: D14 — 网格输出形状 (grid_shape)
# Change: 输出与输入尺寸完全相同（same），删除的分量区域变为背景色... → 多个保留分量分别输出为独立网格

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location

# Design analysis for the D14 variant
# a) Concrete behavior:
#    The input still has a 4x4 legend in the top-left corner and several other
#    rectangular 4-connected components on the orange background. In the
#    original task, components containing all legend colors were erased inside
#    the same 16x16 canvas. In this variant, those components are still the
#    ones removed logically, but the output no longer preserves the original
#    canvas: each surviving component is cropped to its tight bounding box and
#    emitted as its own small frame. The frames are then concatenated
#    left-to-right with a 1-cell background gap.
#    Example trace:
#    - Original: keep components A,B,C in place and paint deleted component D
#      back to bg.
#    - Variant: keep A,B,C, crop them to their own rectangles, sort them by the
#      input reading order (top row, then left column), and place them as
#      [A][bg gap][B][bg gap][C].
# b) Complexity calibration:
#    Only the output organization changes. The filter condition is unchanged, so
#    the task stays close to the source. The layout rule is fixed horizontal
#    concatenation; adding both horizontal and vertical branches would create an
#    unnecessary hidden subtask for a D14-only change.
# c) Implicit assumptions from the old solver:
#    - Labeling every non-bg cell as a candidate component accidentally includes
#      the legend itself. That was only harmless if legend pixels were somehow
#      disconnected, which is not generally true. The new solver explicitly
#      masks out the legend square before component extraction.
#    - Copying the whole grid as output relied on the old same-size invariant.
#      That is invalid now; output size is derived from the surviving crops.
# d) Placement legality:
#    The generator keeps a 1-cell gap between the legend and every component,
#    between all components, and also away from the outer border. That preserves
#    the original _bg_boundary assumption that the frame mostly shows bg.
# e) Visual observability:
#    The main visible change is structural, not color-based: the output is a
#    row of cropped subgrids rather than a cleaned version of the original
#    scene. Reading-order sorting is visible from the input positions.
# f) Visual cleanliness:
#    A 1-cell background separator is inserted between output crops so adjacent
#    frames never fuse visually.
# g) Color inference:
#    No new color mapping is introduced. Colors keep their original meaning; the
#    legend still specifies the deletion palette.
# New solver assumptions checked against the generator:
#    - The legend is the top-left square found by _find_legend; generator always
#      creates exactly that structure.
#    - The outer border is still bg-dominated, because non-legend components are
#      placed with a 1-cell margin from the frame.
#    - There is at least one surviving component; generator always creates three.
#    - Surviving components are ordered by bbox top-left position; generator
#      places disjoint objects so this ordering is well-defined and visible.

# --------------- solver ---------------

OUTPUT_GAP = 1


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Find the smallest top-left square whose bottom edge and right edge share one
    border color, then read the legend colors from its interior.
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


def _collect_kept_patches(g: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """Return bg and surviving component crops in reading order."""
    bg = _bg_boundary(g)
    sz, _border, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    fg_mask = (g != bg)
    fg_mask[:sz, :sz] = False

    kept: list[tuple[tuple[int, int], np.ndarray]] = []
    for pts in components_mask(fg_mask, connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        kept.append(((int(r0), int(c0)), patch))

    kept.sort(key=lambda item: item[0])
    return bg, [patch for _, patch in kept]


def _assemble_patches(patches: list[np.ndarray], bg: int) -> np.ndarray:
    """Concatenate crops left-to-right with a single background gap."""
    assert patches, "no surviving components to assemble"
    height = max(int(p.shape[0]) for p in patches)
    width = sum(int(p.shape[1]) for p in patches) + OUTPUT_GAP * (len(patches) - 1)
    out = np.full((height, width), int(bg), dtype=int)

    c = 0
    for patch in patches:
        paste(out, patch, 0, c)
        c += int(patch.shape[1]) + OUTPUT_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the components that survive the legend filter and line them up as separate crops.

    Input structure:
    - The background fills most of the 16x16 canvas.
    - A top-left legend square has a single-color bottom edge and right edge; its interior shows the target colors.
    - Away from that legend, the foreground consists of several 4-connected multicolor rectangular components.

    Rule:
    - Read the target color set from the legend interior.
    - Ignore the legend square itself, then inspect every other non-background 4-connected component.
    - Discard each component that contains all target colors; keep every component that is missing at least one target color.
    - Crop each kept component to its tight bounding box and concatenate those crops left-to-right with a 1-cell background gap.

    Key insight:
    The legend still tells you which components fail the color test, but the answer is no longer a cleaned copy of the input scene; it is a new strip made only from the surviving component frames.
    """
    g = np.asarray(grid, dtype=int)
    bg, patches = _collect_kept_patches(g)
    return _assemble_patches(patches, bg)


# --------------- generator ---------------

# -- Grid & legend constants --
GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

# -- Component shape weights --
SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (6, 10)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 8)
MED_H = (3, 6)
MED_W = (3, 6)

# -- Component fill density --
DENSITY_LO = 0.12
DENSITY_HI = 0.55

# -- Component count & kept/deleted split --
COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [1, 4, 25]
NUM_KEPT = 3

# -- Output-shape constraints --
MAX_OUTPUT_W = 24
MAX_OUTPUT_H = 12

# -- Generation retry budget --
MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend box at top-left and place the target colors."""
    s = LEGEND_SIZE
    inp[:s, :s] = BG
    inp[s - 1, :s] = BORDER
    inp[:s, s - 1] = BORDER

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


def _component_dims() -> tuple[int, int]:
    """Sample a component size from the original tall / wide / medium families."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _component_patch(h: int, w: int, targets: list[int], required_targets: list[int]) -> np.ndarray:
    """Build a solid rectangular component with black base and colored sprinkles."""
    area = h * w
    lo = max(1, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    non_targets = [v for v in PALETTE if v not in targets]

    guaranteed = list(required_targets)
    guaranteed.append(random.choice(non_targets))
    lo = max(lo, len(guaranteed))
    n_colored = random.randint(lo, hi)

    allowed = non_targets + list(required_targets)
    if len(required_targets) == NUM_TARGETS:
        allowed = list(PALETTE)

    colors = guaranteed + [random.choice(allowed) for _ in range(n_colored - len(guaranteed))]
    random.shuffle(colors)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    patch = np.zeros((h, w), dtype=int)
    rs, cs = zip(*coords)
    patch[rs, cs] = colors
    return patch


def _expected_output(inp: np.ndarray) -> np.ndarray:
    """Build the expected variant output from the actual placed input."""
    bg, patches = _collect_kept_patches(inp)
    return _assemble_patches(patches, bg)


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, targets)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    roles = [("kept", k) for k in random.sample([0, 1, 2], NUM_KEPT)]
    roles += [("deleted", NUM_TARGETS)] * deleted_count
    random.shuffle(roles)

    for role, target_count in roles:
        h, w = _component_dims()
        required_targets = random.sample(targets, target_count)
        patch = _component_patch(h, w, targets, required_targets)
        try:
            r, c = random_free_location(
                inp,
                np.zeros_like(patch),
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)

    out = _expected_output(inp)
    if out.shape == inp.shape:
        return None
    if out.shape[0] > MAX_OUTPUT_H or out.shape[1] > MAX_OUTPUT_W:
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
