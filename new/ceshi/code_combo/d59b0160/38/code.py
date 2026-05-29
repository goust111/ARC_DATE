# Combo variant of d59b0160 (index 38)
# combo_spec: D7/1+D9/2
# Name: 单线分隔符+颜色3分隔线划分区域

import math
import os
import random
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "solvers"))
from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# combo_summary:
#   - D7/1 removes the original top-left L-corner legend frame.
#   - D9/2 adds extra color-3 separator lines inside the main board and changes the
#     matching scope from "one connected component" to "one separator-bounded region".
#   - The fused task therefore needs two different uses of the same color-3 geometry:
#       1. one straight line is the legend/main divider
#       2. the other straight lines are walls inside the main area
#   - The solver must identify which line is the legend divider before it can even decide
#     which cells belong to the search space for region deletion.
#
# concrete_layout_used_here:
#   - The legend divider is a full-height vertical color-3 line near the left side.
#   - The actual legend is the top-left patch on the divider's left side:
#       exactly 3 non-bg, non-3 swatches appear there once each; the rest is bg.
#   - A second full-height vertical color-3 line and one horizontal color-3 line lie on
#     the divider's right side, splitting the main board into four rectangular regions.
#   - Each region contains one or more disconnected black-based rectangles.
#
# behavior_trace:
#   - Suppose the legend colors are {1, 4, 8}.
#   - Region A has two disconnected objects:
#       object 1 contains {0, 1}
#       object 2 contains {0, 4, 8, 5}
#   - Original task logic:
#       object 1 does not have all targets -> keep
#       object 2 has all 3 targets -> delete only object 2
#   - Fused logic:
#       first detect that the leftmost full-height color-3 line with a tiny 3-swatch
#       strip on its left is the legend divider
#       then ignore every color-3 wall and treat Region A as one bounded area
#       union of Region A colors is {0, 1, 4, 5, 8} -> delete all object pixels in A
#   - The key difference is visible only after both changes are combined:
#       legend detection is no longer frame-based, and deletion scope is no longer
#       component-based.
#
# complexity_calibration:
#   - The legend divider orientation is fixed to vertical in generation even though the
#     solver accepts horizontal separators too. This keeps examples consistent enough for
#     humans while still realizing the single-line-separator idea.
#   - The main board always uses one extra vertical wall plus one horizontal wall, giving
#     exactly four easy-to-read regions. This is enough to make region scope observable
#     without turning the board into a maze.
#   - Deleted regions are built from multiple disconnected objects whose target colors are
#     intentionally split, so at least one sample-level decision differs from any naive
#     componentwise reading.
#
# old_main_assumptions_that_break:
#   - _find_legend(...) from the source task assumed an L-shaped frame; that is invalid.
#   - components_mask(g != bg, connectivity=4) is invalid because color 3 now appears in
#     the main board as walls rather than payload.
#   - Checking target coverage per connected component is invalid because the new matching
#     scope is a whole separator-bounded region.
#   - Treating every non-bg color as object data is invalid because color 3 is structural.
#
# generator_legality:
#   - All walls are drawn before objects are placed.
#   - Every object is placed inside one region-local canvas, so it cannot cross a wall,
#     overlap another object, or leave the grid.
#   - The legend side contains only bg plus exactly 3 target swatches, so the legend
#     divider is uniquely detectable despite sharing color 3 with normal walls.
#
# visibility_and_cleanliness_checks:
#   - There are always multiple color-3 lines, so the solver must distinguish roles.
#   - Every region is occupied, but only deleted regions have full target coverage.
#   - Deleted regions always include a region-level-only witness: their union has all
#     target colors while no single object in that region has all of them.
#   - No rule depends on color ordering or numeric magnitude; only explicit membership in
#     the legend palette matters.
#
# new_solver_assumptions_and_generator_support:
#   - Assumption: the legend divider is a full straight border-color line touching both
#     outer borders in one axis, and the top-left side of that line contains exactly 3
#     non-bg, non-border cells with no black object cells.
#     Generator guarantee: the left strip below the swatches is pure bg.
#   - Assumption: main objects never use color 3.
#     Generator guarantee: object palettes exclude BORDER.
#   - Assumption: region walls fully block 4-connectivity between neighboring regions.
#     Generator guarantee: the wall network is continuous and objects are placed strictly
#     inside region boxes.


GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

LEGEND_ROWS = 4
LEGEND_SEP_COLS = [3, 4]
SPLIT_ROWS = [6, 7, 8, 9, 10]
SPLIT_COLS = [9, 10, 11, 12]

DELETED_REGIONS = 2
MAX_RETRIES = 500
DENSITY_LO = 0.18
DENSITY_HI = 0.50


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_candidates(g: np.ndarray, bg: int) -> list[tuple[str, int, int, set[int], np.ndarray]]:
    """Find top-left single-line legend separators and return candidate descriptors."""
    h, w = g.shape
    candidates: list[tuple[str, int, int, set[int], np.ndarray]] = []

    for c in range(1, w - 1):
        x = int(g[0, c])
        if x == bg:
            continue
        if np.all(g[:, c] == x):
            legend_side = g[:, :c]
            payload_mask = (legend_side != bg) & (legend_side != x)
            colors = set(map(int, np.unique(legend_side[payload_mask])))
            if payload_mask.sum() == NUM_TARGETS and len(colors) == NUM_TARGETS and not np.any(legend_side == 0):
                main_mask = np.zeros_like(g, dtype=bool)
                main_mask[:, c + 1 :] = True
                candidates.append(("vertical", c, x, colors, main_mask))

    for r in range(1, h - 1):
        x = int(g[r, 0])
        if x == bg:
            continue
        if np.all(g[r, :] == x):
            legend_side = g[:r, :]
            payload_mask = (legend_side != bg) & (legend_side != x)
            colors = set(map(int, np.unique(legend_side[payload_mask])))
            if payload_mask.sum() == NUM_TARGETS and len(colors) == NUM_TARGETS and not np.any(legend_side == 0):
                main_mask = np.zeros_like(g, dtype=bool)
                main_mask[r + 1 :, :] = True
                candidates.append(("horizontal", r, x, colors, main_mask))

    return candidates


def _find_single_line_legend(g: np.ndarray, bg: int) -> tuple[str, int, int, set[int], np.ndarray]:
    """Return the unique top-left legend separator: (orientation, index, border_color, colors, main_mask)."""
    candidates = _legend_candidates(g, bg)
    assert candidates, "no single-line legend separator found"
    assert len(candidates) == 1, "ambiguous single-line legend separator"
    return candidates[0]


def _separator_regions(g: np.ndarray, main_mask: np.ndarray, border_color: int) -> list[np.ndarray]:
    """Connected regions inside the main area after treating color-3 lines as walls."""
    region_space = main_mask & (g != border_color)
    return components_mask(region_space, connectivity=4)


def _region_object_points(g: np.ndarray, region: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """All non-background pixels inside one region."""
    return [(int(r), int(c)) for r, c in region if int(g[r, c]) != bg]


def _region_metrics(g: np.ndarray) -> dict[str, int]:
    """Quantify whether region-level deletion really differs from componentwise deletion."""
    bg = _bg_boundary(g)
    orient, idx, border_color, legend_colors, main_mask = _find_single_line_legend(g, bg)
    deleted_regions = 0
    region_only_deleted = 0
    full_height_vertical_walls = sum(int(np.all(g[:, c] == border_color)) for c in range(g.shape[1]))

    for region in _separator_regions(g, main_mask, border_color):
        object_pts = _region_object_points(g, region, bg)
        if not object_pts:
            continue
        cols = {int(g[r, c]) for r, c in object_pts}
        if not legend_colors.issubset(cols):
            continue
        deleted_regions += 1

        object_mask = np.zeros_like(g, dtype=bool)
        for r, c in object_pts:
            object_mask[r, c] = True
        comps = components_mask(object_mask, connectivity=4)
        if all(not legend_colors.issubset({int(g[r, c]) for r, c in comp}) for comp in comps):
            region_only_deleted += 1

    ordinary_vertical_walls = full_height_vertical_walls - (1 if orient == "vertical" else 0)
    return {
        "legend_divider_index": int(idx),
        "deleted_regions": int(deleted_regions),
        "region_only_deleted": int(region_only_deleted),
        "ordinary_vertical_walls": int(ordinary_vertical_walls),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every separator-bounded main-board region whose combined object colors match the legend palette.

    Input structure:
    - A straight color-3 separator line near the top-left acts as the legend divider instead of the original L-shaped frame.
    - The top-left side of that divider contains exactly three target-color swatches on background.
    - On the main side of the divider, extra color-3 lines further split the board into multiple regions.
    - Each region may contain one or more disconnected non-background objects made from black rectangles with colored accents.

    Rule:
    - Identify the unique single color-3 line whose top-left side is the tiny legend strip, then read the three legend colors from that side.
    - Restrict attention to the cells on the opposite side of that divider: this is the main board.
    - Treat every color-3 line in the main board as a wall and split the main board into 4-connected regions.
    - For each region, collect all non-background, non-3 object pixels in the region.
    - If the union of those colors contains all legend colors, erase every object pixel in that region; otherwise keep the region unchanged.

    Key insight: color 3 has two structural jobs at once, so the solver must first separate the legend-defining line from the ordinary walls, then evaluate target coverage region by region rather than object by object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    _orient, _idx, border_color, legend_colors, main_mask = _find_single_line_legend(g, bg)
    assert legend_colors, "legend contains no target colors"

    out = g.copy()
    for region in _separator_regions(g, main_mask, border_color):
        object_pts = _region_object_points(g, region, bg)
        if not object_pts:
            continue
        cols = {int(g[r, c]) for r, c in object_pts}
        if legend_colors.issubset(cols):
            for r, c in object_pts:
                out[r, c] = bg
    return out


def _place_legend(inp: np.ndarray, out: np.ndarray, sep_col: int, targets: list[int]) -> None:
    """Draw the full-height legend divider and place 3 swatches in the top-left strip."""
    for g in (inp, out):
        g[:, sep_col] = BORDER

    cells = [(r, c) for r in range(LEGEND_ROWS) for c in range(sep_col)]
    for color, (r, c) in zip(targets, random.sample(cells, len(targets))):
        inp[r, c] = color
        out[r, c] = color


def _draw_main_separators(inp: np.ndarray, out: np.ndarray, sep_col: int, split_r: int, split_c: int) -> None:
    """Add one horizontal and one vertical wall inside the main board."""
    for g in (inp, out):
        g[split_r, sep_col + 1 :] = BORDER
        g[:, split_c] = BORDER


def _region_boxes(sep_col: int, split_r: int, split_c: int) -> list[tuple[int, int, int, int]]:
    """Four boxes induced by the wall layout on the main side of the legend divider."""
    return [
        (0, sep_col + 1, split_r - 1, split_c - 1),
        (0, split_c + 1, split_r - 1, GRID_W - 1),
        (split_r + 1, sep_col + 1, GRID_H - 1, split_c - 1),
        (split_r + 1, split_c + 1, GRID_H - 1, GRID_W - 1),
    ]


def _component_dims(box_h: int, box_w: int, region_canvas: np.ndarray) -> tuple[int, int]:
    """Choose a compact rectangle that fits the local region and leaves room for peers."""
    existing = int(np.count_nonzero(region_canvas != BG))
    max_h = min(4 if existing == 0 else 3, box_h)
    max_w = min(4 if existing == 0 else 3, box_w)
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
    """Build one black-based rectangular object and force selected target colors to appear."""
    area = h * w
    lo = max(len(required), math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    non_targets = [v for v in PALETTE if v not in legend_targets]
    color_pool = non_targets + list(allowed_targets)
    if not color_pool:
        color_pool = non_targets[:]

    colors = list(required)
    colors.extend(random.choice(color_pool) for _ in range(n_colored - len(required)))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = color
    return patch


def _deleted_region_plan(targets: list[int]) -> list[tuple[list[int], list[int]]]:
    """Split all 3 target colors across multiple objects so only the whole region qualifies."""
    shuffled = random.sample(targets, len(targets))
    pattern = random.choice(["2+1", "1+1+1"])
    if pattern == "2+1":
        groups = [shuffled[:2], shuffled[2:]]
    else:
        groups = [[color] for color in shuffled]
    return [(group, group) for group in groups]


def _kept_region_plan(targets: list[int]) -> list[tuple[list[int], list[int]]]:
    """Choose a strict subset of legend colors for a kept region and distribute it."""
    n_components = random.choice([1, 2])
    present_count = random.randint(0, NUM_TARGETS - 1)
    present = random.sample(targets, present_count)
    required_lists = [[] for _ in range(n_components)]
    for idx, color in enumerate(present):
        required_lists[idx % n_components].append(color)
    random.shuffle(required_lists)
    return [(required, present) for required in required_lists]


def _populate_region(
    inp: np.ndarray,
    out: np.ndarray,
    box: tuple[int, int, int, int],
    targets: list[int],
    deleted: bool,
) -> bool:
    """Place one region's disconnected objects without crossing walls."""
    r0, c0, r1, c1 = box
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    if box_h < 2 or box_w < 2:
        return False

    plans = _deleted_region_plan(targets) if deleted else _kept_region_plan(targets)
    region_canvas = np.full((box_h, box_w), BG, dtype=int)

    for required, allowed_targets in plans:
        placed = False
        for _ in range(25):
            h, w = _component_dims(box_h, box_w, region_canvas)
            patch = _component_patch(h, w, required, allowed_targets, targets)
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
    """Construct one candidate input and keep only layouts that exhibit the fused behavior."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    targets = random.sample(PALETTE, NUM_TARGETS)
    sep_col = random.choice(LEGEND_SEP_COLS)
    split_r = random.choice(SPLIT_ROWS)
    split_c = random.choice([c for c in SPLIT_COLS if c > sep_col + 3])

    _place_legend(inp, out, sep_col, targets)
    _draw_main_separators(inp, out, sep_col, split_r, split_c)

    boxes = _region_boxes(sep_col, split_r, split_c)
    deleted_idx = set(random.sample(range(len(boxes)), DELETED_REGIONS))
    for idx, box in enumerate(boxes):
        if not _populate_region(inp, out, box, targets, idx in deleted_idx):
            return None

    metrics = _region_metrics(inp)
    if metrics["deleted_regions"] != DELETED_REGIONS:
        return None
    if metrics["region_only_deleted"] < 1:
        return None
    if metrics["ordinary_vertical_walls"] < 1:
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
