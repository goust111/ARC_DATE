# Combo variant of 88e364bc (index 20)
# combo_spec: D1/0+D19/2
# Name: 向量差解码+箭头图例

"""
Design analysis for the combo variant
=====================================

a_concretization_and_behavior_trace:
  preserved_core:
    - 20x20 grid, black background, colored wall loops, yellow movers, and small border legends.
    - Wall color still identifies which legend controls which enclosed region.
    - Yellow pixels still slide and stack against the far wall inside each enclosed room.
  fused_change:
    - There are exactly two legends, fixed at the top-left and top-right corners.
    - Each legend is now a two-color rectangle: background plate color 1 plus wall color C
      drawn as a coarse arrow silhouette.
    - The visible arrow direction is the legend's displayed vector.
    - Regions of the left legend's color use that displayed vector directly.
    - Regions of the right legend's color use sign(display_left - display_right), componentwise.
  concrete_example:
    - Suppose the left legend uses color 7 and shows a right arrow, so display_left = (0, 1).
    - Suppose the right legend uses color 5 and shows a down arrow, so display_right = (1, 0).
    - In the source task or in the pure arrow-shape variant, color-5 rooms would slide down.
    - In this combo, color-5 rooms instead slide in sign((0,1) - (1,0)) = (-1, 1), i.e. up-right.
    - So the solver pipeline becomes:
        1. detect two border rectangles;
        2. decode each rectangle's arrow shape into a displayed vector;
        3. order them left-to-right across the top border;
        4. derive the second wall color's actual vector by vector subtraction;
        5. slide yellows with the derived mapping.

b_complexity_calibration:
  - The direction vocabulary stays at the same eight legal ARC step vectors.
  - Only one arithmetic dependency is introduced: the second legend depends on the first.
    This keeps the rule learnable from a few examples.
  - The arrow silhouettes are intentionally chunky and asymmetric, so humans can read them
    without needing centroid math, while the subtraction still creates a visible second-stage rule.
  - The generator rejects pairs where the derived vector collapses back to either displayed vector,
    because that would hide the combo effect.

c_implicit_assumptions_review_of_main:
  - Original assumption broken by D1/0:
    `_legend_directions` can no longer expect three legend colors or center/indicator semantics.
    Direction must come from template-matching the dominant wall-color mask.
  - Original assumption broken by D19/2:
    one detected legend no longer implies one final movement vector.
    The second legend's final vector depends on the first legend.
  - Assumptions retained:
    enclosed passable-region detection, border-color majority voting, slide order,
    and diagonal corner-cut prevention remain valid.
  - New assumptions that main() now relies on:
    1. exactly two detectable legend rectangles are present;
    2. their order is visually stable from top-left to top-right;
    3. the dominant legend-color mask matches one of the 5x5/6x7 arrow templates;
    4. the vector difference is non-zero.

d_placement_legality:
  - Legends are pinned to TL and TR and reserve a one-cell buffer, so walls cannot merge into them.
  - Wall loops are still grown only on interior free cells and may not touch the outer border.
  - Each legend color is guaranteed to own at least one enclosed region.
  - Reverse-sliding uses the actual combo direction, so every sampled input is checked by the solver.

e_visual_observability:
  - Humans can first read the left arrow in the usual way, then compare it to the right arrow.
  - Right-colored rooms visibly disobey their own displayed arrow and instead follow the vector
    difference, which makes the fusion observable rather than hidden.
  - Fixing both legends to the top row removes ordering ambiguity; the subtraction cue is spatial.

f_visual_cleanliness:
  - Only two legends appear, both separated from rooms by reserved space.
  - Rooms remain capped at four total, avoiding clutter under the legends.
  - The legend itself is only two colors, so the directional cue is visually cleaner than the
    original three-color centroid code.

g_color_change_inference:
  - No new color remapping rule is introduced.
  - Color C still means "this legend controls walls of color C".
  - Color 1 is just a legend plate color; all nontrivial inference comes from arrow orientation
    plus the ordered vector-difference relation between the two legends.

h_generator_non_degeneracy_metric:
  - Metric: compare the combo output against a legacy-arrow output that incorrectly lets the
    second wall color follow its own displayed arrow directly.
  - A sample is accepted only if at least one cell inside a second-color room differs.

i_post_edit_assumption_audit:
  - main() assumes reading order identifies the arithmetic roles; generate_input() enforces TL/TR.
  - main() assumes 5x5 templates mean cardinal and 6x7 templates mean diagonal; generator preserves that.
  - main() assumes the dominant legend color is the wall color; generator uses masks where the arrow
    color is strictly more frequent than the plate color.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]

LEGEND_BG = 1
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    2: (18, 60),
    3: (14, 38),
    4: (12, 26),
}
BLOB_GROW_RETRIES = 30

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 450

CARDINAL_ARROW_RIGHT = np.array(
    [
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0],
    ],
    dtype=bool,
)

DIAGONAL_ARROW_DR = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=bool,
)


def _sign(v: float) -> int:
    if abs(v) < 1e-6:
        return 0
    return 1 if v > 0 else -1


def _vector_diff_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    dr = _sign(a[0] - b[0])
    dc = _sign(a[1] - b[1])
    if dr == 0 and dc == 0:
        return None
    return dr, dc


def _cardinal_arrow_mask(dr: int, dc: int) -> np.ndarray:
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(CARDINAL_ARROW_RIGHT, rot[(dr, dc)])


def _diagonal_arrow_mask(dr: int, dc: int) -> np.ndarray:
    if (dr, dc) == (1, 1):
        return DIAGONAL_ARROW_DR.copy()
    if (dr, dc) == (-1, 1):
        return np.flipud(DIAGONAL_ARROW_DR)
    if (dr, dc) == (1, -1):
        return np.fliplr(DIAGONAL_ARROW_DR)
    if (dr, dc) == (-1, -1):
        return np.rot90(DIAGONAL_ARROW_DR, 2)
    raise ValueError((dr, dc))


CARDINAL_TEMPLATE_MASKS = {(dr, dc): _cardinal_arrow_mask(dr, dc) for dr, dc in CARDINAL_DIRS}
DIAGONAL_TEMPLATE_MASKS = {(dr, dc): _diagonal_arrow_mask(dr, dc) for dr, dc in DIAGONAL_DIRS}


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((5, 5), LEGEND_BG, dtype=int)
    patch[_cardinal_arrow_mask(dr, dc)] = fill
    return patch


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((6, 7), LEGEND_BG, dtype=int)
    patch[_diagonal_arrow_mask(dr, dc)] = fill
    return patch


def _legend_patch(fill: int, dr: int, dc: int) -> np.ndarray:
    if (dr, dc) in CARDINAL_DIRS:
        return _legend_5x5(fill, dr, dc)
    return _legend_6x7(fill, dr, dc)


def _match_arrow_direction(fill_mask: np.ndarray) -> tuple[int, int] | None:
    if fill_mask.shape == (5, 5):
        for direction, template in CARDINAL_TEMPLATE_MASKS.items():
            if np.array_equal(fill_mask, template):
                return direction
    if fill_mask.shape == (6, 7):
        for direction, template in DIAGONAL_TEMPLATE_MASKS.items():
            if np.array_equal(fill_mask, template):
                return direction
    return None


def _decode_legend_component(g: np.ndarray, pts: np.ndarray) -> dict | None:
    h, w = g.shape
    if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
        return None

    r0, c0, r1, c1 = bbox_coords(pts)
    shape = (r1 - r0 + 1, c1 - c0 + 1)
    area = shape[0] * shape[1]
    if area != len(pts):
        return None
    if shape not in ((5, 5), (6, 7)):
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    vals, counts = np.unique(patch, return_counts=True)
    if len(vals) != 2:
        return None
    if LEGEND_BG not in vals:
        return None

    fill_candidates = [int(v) for v in vals if int(v) != LEGEND_BG]
    if len(fill_candidates) != 1:
        return None
    fill_col = fill_candidates[0]
    if int(np.count_nonzero(patch == fill_col)) <= int(np.count_nonzero(patch == LEGEND_BG)):
        return None

    direction = _match_arrow_direction(patch == fill_col)
    if direction is None:
        return None

    return {
        "color": fill_col,
        "display": direction,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(len(pts)),
    }


def _detect_legends(g: np.ndarray, bg: int) -> list[dict]:
    best_by_color: dict[int, dict] = {}
    for pts in components_mask(g != bg, connectivity=4):
        info = _decode_legend_component(g, pts)
        if info is None:
            continue
        prev = best_by_color.get(info["color"])
        if prev is None or info["area"] > prev["area"]:
            best_by_color[info["color"]] = info
    legends = list(best_by_color.values())
    legends.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return legends


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    legends = _detect_legends(g, bg)
    assert len(legends) == 2, "combo variant requires exactly two detected legends"

    first, second = legends
    derived_second = _vector_diff_dir(first["display"], second["display"])
    assert derived_second is not None, "legend vector subtraction must produce a non-zero direction"

    return {
        int(first["color"]): tuple(first["display"]),
        int(second["color"]): tuple(derived_second),
    }


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _solve_with_dirs(grid: np.ndarray, dirs: dict[int, tuple[int, int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == MOVER:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = MOVER

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside enclosed rooms using arrow legends plus a derived second instruction.

    Input structure:
    - The background is the most common color. Colored wall loops enclose passable rooms containing yellow pixels.
    - Two small legend rectangles sit on the top border. Each uses wall color C as a chunky arrow silhouette on a color-1 plate.
    - The left and right legends correspond to two distinct wall colors appearing on the rooms below.

    Rule:
    - Decode each legend's displayed direction from the orientation of its wall-colored arrow shape.
    - Rooms bordered by the left legend's wall color move in that displayed direction.
    - Rooms bordered by the right legend's wall color move in sign(left_display - right_display), then all yellows slide and stack along that actual direction.

    Key insight: Color still links each legend to its rooms, but the motion is a two-stage inference:
    first read arrow shapes as vectors, then derive the second color's true motion by vector subtraction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    return _solve_with_dirs(g, dirs)


def _grow_blob(
    canvas_mask: np.ndarray,
    seed: tuple[int, int],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _sample_instruction_pair(
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    candidates = []
    for left_display in ALL_DIRS:
        for right_display in ALL_DIRS:
            if left_display == right_display:
                continue
            derived = _vector_diff_dir(left_display, right_display)
            if derived is None:
                continue
            if derived == left_display or derived == right_display:
                continue
            candidates.append((left_display, right_display, derived))
    return rng.choice(candidates)


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    wall_a, wall_b = rng.sample(WALL_COLOR_CHOICES, 2)
    disp_a, disp_b, actual_b = _sample_instruction_pair(rng)

    legends = [
        {"role": "A", "color": wall_a, "display": disp_a, "actual": disp_a, "corner": "TL"},
        {"role": "B", "color": wall_b, "display": disp_b, "actual": actual_b, "corner": "TR"},
    ]

    legend_occupied = np.zeros((h, w), dtype=bool)
    for leg in legends:
        patch = _legend_patch(int(leg["color"]), *tuple(leg["display"]))
        ph, pw = patch.shape
        r0 = 0
        c0 = 0 if leg["corner"] == "TL" else w - pw
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True

    return legend_occupied, legends


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        return cand
    return None


def _grow_regions(
    rng: random.Random,
    legends: list[dict],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_actual = {int(leg["color"]): tuple(leg["actual"]) for leg in legends}
    color_legacy = {int(leg["color"]): tuple(leg["display"]) for leg in legends}
    colors = [int(leg["color"]) for leg in legends]

    region_colors = []
    for color in colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "color": rc,
                "dir": color_actual[rc],
                "legacy_dir": color_legacy[rc],
                "interior": blob,
                "walls": walls,
            }
        )

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return r, c
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        yellows_input = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == MOVER).sum()) != int((input_grid == MOVER).sum()):
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == MOVER]))


def _variant_effective(input_grid: np.ndarray, regions: list[dict]) -> bool:
    legacy_dirs = {reg["color"]: reg["legacy_dir"] for reg in regions}
    affected_colors = {reg["color"] for reg in regions if reg["dir"] != reg["legacy_dir"]}
    if not affected_colors:
        return False

    combo_out = main(input_grid)
    legacy_out = _solve_with_dirs(input_grid, legacy_dirs)

    mask = np.zeros(input_grid.shape, dtype=bool)
    for reg in regions:
        if reg["color"] not in affected_colors:
            continue
        for r, c in reg["interior"]:
            mask[r, c] = True
    return bool(mask.any() and np.any(combo_out[mask] != legacy_out[mask]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    if not _variant_effective(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
