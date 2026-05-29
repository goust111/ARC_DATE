# Variant of d59b0160
# Dimension: D9 — 连通分量的识别范围 (src)
# Change: 图例区域外的所有非背景像素参与连通分量分析，包括各种颜色（0-9中除了7和3）... → 按颜色3分隔线划分区域

import math
import os
import random
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "solvers"))
from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D9 variant
# ----------------------------------
# variant_spec:
#   source_task: none
#   core_change:
#     before:
#       - main() treated every non-background pixel outside the legend as part of a
#         single global search space for 4-connected components.
#       - Deletion was decided component by component.
#     after:
#       - Extra color-3 lines may appear in the main grid and act as separators.
#       - First split the non-3 area outside the legend into separator-bounded regions.
#       - Inside each region, gather all non-background, non-3 object pixels together.
#       - If the union of colors inside that region contains the whole legend palette,
#         erase every object pixel in that region.
#   concrete_behavior:
#     - The 4x4 top-left legend stays unchanged: bg=7, border=3, 3 target colors inside.
#     - The main board contains one vertical and one horizontal separator in color 3.
#       They connect to borders / each other, so the board is visibly partitioned.
#     - Each region contains one or more disconnected rectangular objects made from a
#       color-0 body plus sprinkled colors from {1,2,4,5,6,8,9}.
#     - Deleted regions are intentionally built from multiple disconnected objects whose
#       color union covers all 3 target colors, while no single object necessarily does.
#       This makes the region-level scope visually observable.
#   trace_example:
#     - Suppose the legend colors are {1,4,8}.
#     - In one region there are two separate objects:
#         object A has colors {0,1,4}
#         object B has colors {0,8,5}
#     - Original logic: neither connected component has all 3 target colors, so both stay.
#     - Variant logic: the separator-defined region contains {0,1,4,8,5}, so the whole
#       region is erased.
#   complexity_calibration:
#     - The separator layout uses only one horizontal + one vertical line. That yields
#       four easy-to-read regions without introducing too many cases.
#     - Deleted regions always need at least two objects; kept regions may have one or two.
#       This is enough to expose the new scope rule without making samples noisy.
#   old_main_assumptions_that_break:
#     - _find_legend(...): still valid; the legend format is unchanged.
#     - components_mask(g != bg, connectivity=4): no longer valid because color 3 can
#       appear in the main board as walls rather than payload pixels.
#     - cols = {g[r,c] for r,c in pts}: wrong scope for the variant, because matching must
#       be done on all object pixels inside one separator-bounded region, not one global
#       connected component.
#     - filling all pts with bg: still valid after the matching scope is changed, but the
#       points to erase must be "all object pixels in the region" instead of one component.
#   generator_legality:
#     - Separator lines are drawn before object placement and never overlap the legend.
#     - Objects are placed inside region-local canvases, so they cannot cross walls,
#       overlap one another, or leave the grid.
#     - The solver's region detector excludes the whole legend box and all color-3 cells;
#       the generator keeps that invariant exact.
#   visibility_checks:
#     - Every sample has multiple separator-defined regions.
#     - At least one deleted region is solved differently from the original component-wise
#       rule because its target colors are split across disconnected objects.
#     - Colors are never interpreted by numeric order; only explicit membership in the
#       legend set matters.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the top-left legend whose bottom edge and right edge share one border color.
    Return (size, border_color, interior_color_set).
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


def _separator_regions(g: np.ndarray, legend_sz: int, border_color: int) -> list[np.ndarray]:
    """Connected regions outside the legend after treating color-3 cells as walls."""
    region_space = np.ones_like(g, dtype=bool)
    region_space[:legend_sz, :legend_sz] = False
    region_space &= (g != border_color)
    return components_mask(region_space, connectivity=4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every separator-bounded region whose combined object colors realize the legend palette.

    Input structure:
    - Background (bg) still dominates the board, and the top-left 4x4 legend keeps the same format: color 3 on its bottom and right edges, target colors inside.
    - Outside the legend, extra color-3 lines may appear and split the main board into several regions.
    - Each such region can contain one or more disconnected non-background objects built from color 0 plus accent colors.

    Rule:
    - Read the target color set from inside the legend border.
    - Treat every non-3 cell outside the legend as belonging to one separator-defined region.
    - For each region, collect all non-background, non-3 pixels in that region and inspect their combined colors.
    - If that region contains all legend colors, erase every object pixel in the region; otherwise leave the region unchanged.

    Key insight: color 3 is no longer only the legend frame; it becomes a wall that changes the scope of palette matching from global connected components to local regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend_sz, border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for region in _separator_regions(g, legend_sz, border_color):
        object_pts = [(int(r), int(c)) for r, c in region if int(g[r, c]) != bg]
        if not object_pts:
            continue
        cols = {int(g[r, c]) for r, c in object_pts}
        if legend_colors.issubset(cols):
            for r, c in object_pts:
                out[r, c] = bg
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SPLIT_ROWS = [9, 10, 11]
SPLIT_COLS = [7, 8, 9]

DELETED_REGIONS = 2
MAX_RETRIES = 400
DENSITY_LO = 0.18
DENSITY_HI = 0.55


def _place_legend(inp: np.ndarray, out: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend at top-left and place the 3 target colors once each."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER

    cells = [(r, c) for r in range(s - 1) for c in range(s - 1)]
    for color, (r, c) in zip(targets, random.sample(cells, len(targets))):
        inp[r, c] = color
        out[r, c] = color


def _draw_separators(inp: np.ndarray, out: np.ndarray, split_r: int, split_c: int) -> None:
    """Add one horizontal and one vertical color-3 separator in the main board."""
    for g in (inp, out):
        g[split_r, :] = BORDER
        g[:, split_c] = BORDER


def _region_boxes(split_r: int, split_c: int) -> list[tuple[int, int, int, int]]:
    """Placement boxes fully contained in the four separator-defined regions."""
    return [
        (LEGEND_SIZE, 0, split_r - 1, split_c - 1),
        (LEGEND_SIZE, split_c + 1, split_r - 1, GRID_W - 1),
        (split_r + 1, 0, GRID_H - 1, split_c - 1),
        (split_r + 1, split_c + 1, GRID_H - 1, GRID_W - 1),
    ]


def _component_dims(box_h: int, box_w: int, region_canvas: np.ndarray) -> tuple[int, int]:
    """Choose a small rectangle that fits the local region comfortably."""
    existing = int(np.count_nonzero(region_canvas != BG))
    if existing == 0:
        max_h = min(4, box_h)
        max_w = min(4, box_w)
    else:
        max_h = min(3, box_h)
        max_w = min(3, box_w)
    min_h = 2 if box_h >= 2 else 1
    min_w = 2 if box_w >= 2 else 1
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _component_patch(
    h: int,
    w: int,
    required: list[int],
    allowed_targets: list[int],
    legend_targets: list[int],
) -> np.ndarray:
    """Build a color-0 rectangle with sprinkled non-zero colors."""
    area = h * w
    lo = max(len(required), math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in legend_targets]
    color_pool = non_targets + allowed_targets
    if not color_pool:
        color_pool = PALETTE[:]

    colors = list(required)
    colors.extend(random.choice(color_pool) for _ in range(n_colored - len(required)))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = color
    return patch


def _deleted_target_split(targets: list[int]) -> list[list[int]]:
    """Split the 3 target colors across at least two components."""
    shuffled = random.sample(targets, len(targets))
    if random.random() < 0.5:
        return [shuffled[:2], shuffled[2:]]
    return [[shuffled[0]], shuffled[1:]]


def _kept_target_split(targets: list[int], n_components: int) -> list[list[int]]:
    """Choose a proper subset of legend colors for a kept region and distribute it."""
    present_count = random.randint(0, NUM_TARGETS - 1)
    present = random.sample(targets, present_count)
    splits = [[] for _ in range(n_components)]
    if not present:
        return splits
    for idx, color in enumerate(present):
        splits[idx % n_components].append(color)
    random.shuffle(splits)
    return splits


def _populate_region(
    inp: np.ndarray,
    out: np.ndarray,
    box: tuple[int, int, int, int],
    targets: list[int],
    deleted: bool,
) -> bool:
    """Place one region worth of disconnected objects; return False on placement failure."""
    r0, c0, r1, c1 = box
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    if box_h < 2 or box_w < 2:
        return False

    region_canvas = np.full((box_h, box_w), BG, dtype=int)
    if deleted:
        target_splits = _deleted_target_split(targets)
    else:
        n_components = random.choice([1, 2])
        target_splits = _kept_target_split(targets, n_components)

    for required in target_splits:
        placed = False
        for _ in range(20):
            h, w = _component_dims(box_h, box_w, region_canvas)
            patch = _component_patch(h, w, required, required, targets)
            try:
                rr, cc = random_free_location(
                    region_canvas,
                    patch,
                    transparent=BG,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            paste(region_canvas, patch, rr, cc)
            placed = True
            break
        if not placed:
            return False

    paste(inp, region_canvas, r0, c0, transparent=BG)
    if not deleted:
        paste(out, region_canvas, r0, c0, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Construct one candidate input grid and validate it against main()."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    _place_legend(inp, out, targets)

    split_r = random.choice(SPLIT_ROWS)
    split_c = random.choice(SPLIT_COLS)
    _draw_separators(inp, out, split_r, split_c)

    boxes = _region_boxes(split_r, split_c)
    deleted_idx = set(random.sample(range(len(boxes)), DELETED_REGIONS))

    for idx, box in enumerate(boxes):
        if not _populate_region(inp, out, box, targets, idx in deleted_idx):
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
