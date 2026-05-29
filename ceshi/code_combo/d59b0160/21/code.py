# Combo variant of d59b0160 (index 21)
# combo_spec: D9/2+D10/3
# Name: 颜色3分隔线划分区域+多图例多轮筛选

import math
import os
import random
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "solvers"))
from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Keep two readable legends with the same color-3 frame language:
#      top-left legend = original orientation, bottom-right legend = mirrored orientation.
#    - Add one horizontal and one vertical color-3 separator to partition the non-legend board
#      into four wall-bounded regions.
#    - Each round reads one legend, then checks each separator-defined region as a whole.
#      If the union of colors on all non-background object pixels inside that region contains
#      the full legend set, erase every object pixel in that region.
#    - Small trace:
#        round-1 legend = {1,4,8}
#        one region has two disconnected objects with colors {0,1,4} and {0,8,5}
#        original task: neither component has all {1,4,8}, so both survive
#        combo task, round 1: the region union has {0,1,4,8,5}, so the whole region is erased
#        another region may realize only the second legend colors, so it survives round 1 and
#        disappears only after round 2. This makes both "region scope" and "ordered rounds"
#        visible in the same sample.
# b) Complexity calibration:
#    - Use exactly two legends and exactly one cross of separators. Four regions are enough to
#      show repeated evidence without clutter.
#    - Legend color sets are disjoint 3-color triplets chosen from the 7-color palette. This
#      keeps the two rounds visually distinguishable.
#    - Region roles come from a short blueprint family containing at least one first-round
#      deletion, one second-round deletion, and one kept region. Deleted regions usually split
#      required colors across multiple disconnected objects so the D9 behavior is observable.
# c) Hidden assumptions that break from the original main():
#    - A single `_find_legend(...)` is no longer enough; the combo needs two corner-specific
#      legend detectors and a fixed application order.
#    - `components_mask(g != bg, ...)` is wrong because color 3 is now both legend border and
#      separator wall. The search space must exclude walls and both legend boxes first.
#    - Checking each connected component independently is wrong because matching happens on the
#      color union of an entire separator-bounded region.
#    - A single deletion pass is wrong because the second legend is applied after round 1 on the
#      updated grid state.
# d) Placement legality:
#    - Separators are drawn before any objects and never overlap either legend.
#    - Each region is populated through a local blocker canvas, so objects cannot cross walls,
#      overlap legends, overlap one another, or leave the grid.
#    - Objects are full color-0 rectangles with sparse accent colors, preserving the source
#      task's visual language while keeping 4-connectivity obvious.
# e) Visual observability:
#    - Human-visible cues are explicit: two corner legends, one separator cross, and region-level
#      deletions. The ordering signal is the fixed corner order: top-left first, bottom-right
#      second.
#    - At least one generated region is deleted only because target colors are split across
#      disconnected objects, so examples do not collapse back to the original component rule.
# f) Visual cleanliness:
#    - Objects keep a one-cell gap from one another and from legend cutouts inside their region
#      canvas, which avoids tangles around the walls.
#    - The separator structure is minimal and consistent, so the board still reads cleanly.
# g) New solver assumptions and generator guarantees:
#    - The solver assumes legends stay in the two corners and use the two known border
#      orientations. The generator enforces exactly that.
#    - The solver assumes wall color 3 does not appear inside ordinary objects. The generator
#      excludes color 3 from object palettes.
#    - The solver assumes every region can be treated independently once walls and legends are
#      masked out. The generator ensures all objects lie wholly inside one region.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_top_left_legend(g: np.ndarray, bg: int) -> dict:
    """Find the top-left legend: shared border color on bottom edge and right edge."""
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
            return {
                "size": int(sz),
                "border": int(x),
                "colors": colors,
                "bbox": (0, 0, sz - 1, sz - 1),
            }
    raise AssertionError("no top-left legend border found")


def _find_bottom_right_legend(g: np.ndarray, bg: int) -> dict:
    """Find the mirrored bottom-right legend: shared border color on top edge and left edge."""
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        r0 = h - sz
        c0 = w - sz
        x = int(g[r0, c0])
        if x == bg:
            continue
        if np.all(g[r0, c0 : c0 + sz] == x) and np.all(g[r0 : r0 + sz, c0] == x):
            inside = g[r0 + 1 : r0 + sz, c0 + 1 : c0 + sz]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return {
                "size": int(sz),
                "border": int(x),
                "colors": colors,
                "bbox": (r0, c0, h - 1, w - 1),
            }
    raise AssertionError("no bottom-right legend border found")


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _legend_data(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    legends = [_find_top_left_legend(g, bg), _find_bottom_right_legend(g, bg)]
    for legend in legends:
        assert legend["colors"], "legend interior contains no target colors"
    legend_mask = np.zeros_like(g, dtype=bool)
    for legend in legends:
        legend_mask |= _bbox_mask(g.shape, legend["bbox"])
    return legends, legend_mask


def _separator_regions(g: np.ndarray, legend_mask: np.ndarray, border_color: int) -> list[np.ndarray]:
    """Connected open areas after removing legends and treating color-3 cells as walls."""
    region_space = (~legend_mask) & (g != border_color)
    return components_mask(region_space, connectivity=4)


def _apply_round(
    out: np.ndarray,
    legend: dict,
    legend_mask: np.ndarray,
    bg: int,
    *,
    collect_metrics: bool = False,
) -> dict:
    metrics = {"deleted_regions": 0, "split_deleted_regions": 0}
    for region in _separator_regions(out, legend_mask, legend["border"]):
        region_mask = np.zeros_like(out, dtype=bool)
        region_mask[region[:, 0], region[:, 1]] = True
        object_mask = region_mask & (out != bg)
        if not object_mask.any():
            continue

        colors = {int(v) for v in out[object_mask]}
        if not legend["colors"].issubset(colors):
            continue

        metrics["deleted_regions"] += 1
        if collect_metrics:
            comp_match = False
            for comp in components_mask(object_mask, connectivity=4):
                comp_colors = {int(out[r, c]) for r, c in comp}
                if legend["colors"].issubset(comp_colors):
                    comp_match = True
                    break
            if not comp_match:
                metrics["split_deleted_regions"] += 1

        out[object_mask] = bg
    return metrics


def _solve_with_metrics(grid: np.ndarray, *, collect_metrics: bool) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legends, legend_mask = _legend_data(g, bg)

    out = g.copy()
    metrics = {
        "round1_deleted_regions": 0,
        "round2_deleted_regions": 0,
        "split_deleted_regions": 0,
        "late_deleted_regions": 0,
    }
    for idx, legend in enumerate(legends):
        round_metrics = _apply_round(out, legend, legend_mask, bg, collect_metrics=collect_metrics)
        if idx == 0:
            metrics["round1_deleted_regions"] = round_metrics["deleted_regions"]
        else:
            metrics["round2_deleted_regions"] = round_metrics["deleted_regions"]
            metrics["late_deleted_regions"] = round_metrics["deleted_regions"]
        metrics["split_deleted_regions"] += round_metrics["split_deleted_regions"]
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply two ordered legend filters, but evaluate matches on separator-bounded regions rather than on single objects.

    Input structure:
    - Background (bg) still dominates the board.
    - A top-left 4x4 legend uses color 3 on its bottom edge and right edge; a bottom-right 4x4 legend uses color 3 on its top edge and left edge.
    - Outside the legends, extra color-3 lines split the board into several regions.
    - Each region may contain one or more disconnected multicolor rectangles built from color 0 plus accent colors.

    Rule:
    - Read the first target set from the top-left legend, then the second target set from the bottom-right legend.
    - For each round, ignore both legend boxes and treat every color-3 cell as a wall.
    - Within each wall-bounded region, collect all non-background object pixels in that region.
    - If that regional color union contains the current legend's full color set, erase every object pixel in that region.
    - Recompute on the updated grid and repeat for the second legend.

    Key insight: color 3 serves two jobs at once: it marks the legends and it partitions the board, so the solver must combine fixed legend order with region-level palette matching.
    """
    out, _metrics = _solve_with_metrics(grid, collect_metrics=False)
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_TARGETS = 3

SPLIT_ROWS = [8, 9]
SPLIT_COLS = [7, 8]

DENSITY_LO = 0.14
DENSITY_HI = 0.45
MAX_RETRIES = 500

ROLE_BLUEPRINTS = [
    ["first_only", "second_only", "keep", "keep"],
    ["first_only", "second_only", "both", "keep"],
    ["first_only", "second_only", "keep", "both"],
    ["both", "second_only", "keep", "first_only"],
]


def _legend_positions_top_left() -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(LEGEND_SIZE - 1) for c in range(LEGEND_SIZE - 1)]
    return random.sample(cells, NUM_TARGETS)


def _legend_positions_bottom_right() -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(LEGEND_SIZE - 1) for c in range(LEGEND_SIZE - 1)]
    return random.sample(cells, NUM_TARGETS)


def _choose_legend_sets() -> tuple[list[int], list[int], list[int]]:
    picks = random.sample(PALETTE, 6)
    first = picks[:NUM_TARGETS]
    second = picks[NUM_TARGETS : 2 * NUM_TARGETS]
    extras = [c for c in PALETTE if c not in first and c not in second]
    return first, second, extras


def _place_legends(inp: np.ndarray, first: list[int], second: list[int]) -> None:
    s = LEGEND_SIZE
    br_r0 = GRID_H - s
    br_c0 = GRID_W - s

    inp[:s, :s] = BG
    inp[s - 1, :s] = BORDER
    inp[:s, s - 1] = BORDER

    inp[br_r0:, br_c0:] = BG
    inp[br_r0, br_c0:] = BORDER
    inp[br_r0:, br_c0] = BORDER

    for color, (r, c) in zip(first, _legend_positions_top_left()):
        inp[r, c] = color
    for color, (lr, lc) in zip(second, _legend_positions_bottom_right()):
        inp[br_r0 + 1 + lr, br_c0 + 1 + lc] = color


def _draw_separators(inp: np.ndarray, split_r: int, split_c: int) -> None:
    inp[split_r, :] = BORDER
    inp[:, split_c] = BORDER


def _global_to_local(
    global_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    gr0, gc0, gr1, gc1 = global_bbox
    r0, c0, _r1, _c1 = outer_bbox
    return gr0 - r0, gc0 - c0, gr1 - r0, gc1 - c0


def _region_specs(split_r: int, split_c: int) -> list[dict]:
    br_bbox = (GRID_H - LEGEND_SIZE, GRID_W - LEGEND_SIZE, GRID_H - 1, GRID_W - 1)
    specs = [
        {"bbox": (0, 0, split_r - 1, split_c - 1), "forbidden": [(0, 0, LEGEND_SIZE - 1, LEGEND_SIZE - 1)]},
        {"bbox": (0, split_c + 1, split_r - 1, GRID_W - 1), "forbidden": []},
        {"bbox": (split_r + 1, 0, GRID_H - 1, split_c - 1), "forbidden": []},
        {"bbox": (split_r + 1, split_c + 1, GRID_H - 1, GRID_W - 1), "forbidden": []},
    ]
    specs[3]["forbidden"].append(_global_to_local(br_bbox, specs[3]["bbox"]))
    return specs


def _paint_forbidden(canvas: np.ndarray, bbox: tuple[int, int, int, int], value: int) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(canvas.shape[0] - 1, r1)
    cc1 = min(canvas.shape[1] - 1, c1)
    if rr0 <= rr1 and cc0 <= cc1:
        canvas[rr0 : rr1 + 1, cc0 : cc1 + 1] = value


def _component_dims(box_h: int, box_w: int, blocker: np.ndarray) -> tuple[int, int]:
    existing = int(np.count_nonzero(blocker != BG))
    max_h = min(4 if existing == 0 else 3, box_h)
    max_w = min(4 if existing == 0 else 3, box_w)
    min_h = 2 if box_h >= 2 else 1
    min_w = 2 if box_w >= 2 else 1
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _component_patch(h: int, w: int, must_have: list[int], filler_pool: list[int]) -> np.ndarray:
    area = h * w
    min_colored = len(must_have)
    if filler_pool:
        min_colored = max(min_colored, 1)
    lo = max(min_colored, math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = list(must_have)
    pool = filler_pool if filler_pool else [0]
    while len(colors) < n_colored:
        colors.append(random.choice(pool))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = color
    return patch


def _split_colors(colors: list[int]) -> list[list[int]]:
    shuffled = random.sample(colors, len(colors))
    return [shuffled[:2], shuffled[2:]]


def _component_plans_for_role(first: list[int], second: list[int], extras: list[int], role: str) -> list[dict]:
    if role == "first_only":
        second_subset = random.sample(second, random.randint(0, NUM_TARGETS - 1))
        first_splits = _split_colors(first)
        return [
            {"must": first_splits[0], "pool": extras + second_subset + first_splits[0]},
            {"must": first_splits[1], "pool": extras + second_subset + first_splits[1]},
        ]
    if role == "second_only":
        first_subset = random.sample(first, random.randint(0, NUM_TARGETS - 1))
        second_splits = _split_colors(second)
        return [
            {"must": second_splits[0], "pool": extras + first_subset + second_splits[0]},
            {"must": second_splits[1], "pool": extras + first_subset + second_splits[1]},
        ]
    if role == "both":
        first_splits = _split_colors(first)
        second_splits = _split_colors(second)
        return [
            {"must": first_splits[0] + second_splits[0], "pool": extras + first_splits[0] + second_splits[0]},
            {"must": first_splits[1] + second_splits[1], "pool": extras + first_splits[1] + second_splits[1]},
        ]

    present_first = random.sample(first, random.randint(0, NUM_TARGETS - 1))
    present_second = random.sample(second, random.randint(0, NUM_TARGETS - 1))
    n_components = random.choice([1, 2])
    distributed = [[] for _ in range(n_components)]
    combined = present_first + present_second
    random.shuffle(combined)
    for idx, color in enumerate(combined):
        distributed[idx % n_components].append(color)
    if not combined:
        distributed[0].append(random.choice(extras))
    return [{"must": part, "pool": extras + part} for part in distributed]


def _populate_region(
    inp: np.ndarray,
    spec: dict,
    plans: list[dict],
) -> bool:
    r0, c0, r1, c1 = spec["bbox"]
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    if box_h < 2 or box_w < 2:
        return False

    objects = np.full((box_h, box_w), BG, dtype=int)
    blocker = np.full((box_h, box_w), BG, dtype=int)
    for forb in spec["forbidden"]:
        _paint_forbidden(blocker, forb, 1)

    for plan in plans:
        placed = False
        for _ in range(30):
            h, w = _component_dims(box_h, box_w, blocker)
            patch = _component_patch(h, w, plan["must"], plan["pool"])
            try:
                rr, cc = random_free_location(
                    blocker,
                    patch,
                    transparent=BG,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            paste(blocker, patch, rr, cc)
            paste(objects, patch, rr, cc)
            placed = True
            break
        if not placed:
            return False

    paste(inp, objects, r0, c0, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    first, second, extras = _choose_legend_sets()
    _place_legends(inp, first, second)

    split_r = random.choice(SPLIT_ROWS)
    split_c = random.choice(SPLIT_COLS)
    _draw_separators(inp, split_r, split_c)

    specs = _region_specs(split_r, split_c)
    roles = random.choice(ROLE_BLUEPRINTS).copy()
    random.shuffle(roles)
    for spec, role in zip(specs, roles):
        plans = _component_plans_for_role(first, second, extras, role)
        if not _populate_region(inp, spec, plans):
            return None

    out, metrics = _solve_with_metrics(inp, collect_metrics=True)
    if metrics["round1_deleted_regions"] < 1:
        return None
    if metrics["late_deleted_regions"] < 1:
        return None
    if metrics["split_deleted_regions"] < 1:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo input")


def fusion_metrics(grid: np.ndarray) -> dict:
    """Expose the combo-specific observability metrics for external checks."""
    _out, metrics = _solve_with_metrics(grid, collect_metrics=True)
    return metrics
