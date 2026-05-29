# Combo variant of 88e364bc (index 39)
# combo_spec: D2/2+D19/2
# Name: 颜色向量叠加匹配+向量差解码

"""
Design analysis for combo D2/2 + D19/2
======================================

source_task:
  reference_task: 88e364bc
  preserved_core:
    - 20x20 black background.
    - Border legends are filled rectangles whose dominant fill color matches wall colors.
    - Yellow pixels slide inside enclosed rooms and stack against the far wall.
    - Diagonal motion still obeys the no-corner-cutting rule from the source solver.

a_concretization_and_behavior_trace:
  concrete_fused_rule:
    - There are exactly two legends, fixed at the top-left and top-right corners.
    - Let legend A display vector be disp_A and legend B display vector be disp_B.
    - A-colored rooms use actual_A = disp_A.
    - B-colored rooms use actual_B = sign(disp_A - disp_B).
    - A mixed room has a single enclosed interior but its wall ring visibly contains both
      legend fill colors. That room uses mixed_dir = sign(actual_A + actual_B).
  small_example:
    - Suppose A displays right  = (0, 1).
    - Suppose B displays down   = (1, 0).
    - Then actual_B = sign((0,1) - (1,0)) = (-1, 1) = up-right.
    - The mixed room uses sign((0,1) + (-1,1)) = (-1, 1) = up-right.
  source_vs_fused_trace:
    - Source task:
        1. Count adjacent wall colors for a room.
        2. Pick the majority wall color.
        3. Use that color's displayed legend direction directly.
    - Fused task:
        1. Detect the two legends in reading order.
        2. Decode B's true direction via vector difference against A.
        3. Inspect which legend wall colors appear around each room.
        4. If one legend color appears, use that legend's true direction.
        5. If both legend colors appear, add the two true directions and slide by that sum.
  visible_difference:
    - A mixed-color wall no longer behaves like a noisy majority-color wall.
    - B-only rooms no longer follow B's shown arrow.
    - The mixed room depends on both modifications at once because one summand is itself decoded
      through the A-B vector difference.

b_complexity_calibration:
  - The generator always provides three evidence room types in the same grid:
      1. one A-only room,
      2. one B-only room,
      3. one mixed A+B room.
  - This keeps the number of instructional cases small and repeated enough for humans to compare.
  - The sampled direction pair is restricted so:
      - actual_B is non-zero and differs from B's displayed direction;
      - mixed_dir is non-zero and differs from both pure-room directions.
    These constraints prevent the fused rule from collapsing into the source task or the D19-only task.

c_implicit_assumptions_review_of_main:
  old_assumptions_broken:
    - "one wall color per room" is false for the mixed room.
    - "one legend directly determines one room direction" is false for B and also false for the
      mixed room, whose motion depends on both legends.
  line_level_consequences:
    - Legend component detection remains valid because legends are still filled border rectangles
      with three colors and a centroid-based arrow cue.
    - Region extraction via enclosed passable components remains valid.
    - Border analysis must now collect the set of legend wall colors around a room instead of
      taking only the single most frequent color.
    - Direction resolution becomes:
        A color only -> actual_A
        B color only -> actual_B = sign(disp_A - disp_B)
        both colors  -> sign(actual_A + actual_B)
  new_solver_assumptions:
    - Exactly two detectable legends exist and are ordered left-to-right on the top edge.
    - The only non-background wall colors adjacent to room interiors are the two legend fill colors.
    - If a room sees both legend colors, their vector sum is non-zero.

d_placement_legality:
  - Legends occupy the two top corners with a one-cell safety band below and beside them.
  - Rooms are placed in three disjoint slots below the legends, so no wall overlaps another wall
    or a legend.
  - Every room is a closed rectangle, so its interior is guaranteed to be one bounded component.
  - The mixed room wall ring is colored after the room geometry is fixed; top/left segments use
    A's fill color and the remaining segments use B's fill color, guaranteeing both colors appear.

e_visual_observability:
  - The A-only room anchors the normal meaning of a legend arrow.
  - The B-only room shows that B's displayed arrow is not the real motion cue.
  - The mixed room shows that a two-color wall is a separate instruction class rather than a
    majority-color tie-break.
  - Because these three room types appear together, a solver can compare them within one example
    instead of needing hidden metadata or color-id arithmetic.

f_visual_cleanliness:
  - The scene uses only two wall colors plus yellow movers, matching the original palette habits.
  - The mixed room uses a clean top-left versus bottom-right color split, so the two wall colors
    are easy to see without clutter.
  - Fixed room slots avoid accidental tangles that would make boundary ownership hard to parse.

g_color_change_inference:
  - No color remapping is introduced; wall colors still point back to the legends that share them.
  - The new burden is structural: a room may be linked to one legend color or to both.
  - The generator avoids tiny minority fragments, so the mixed-color cue is a visible wall pattern
    rather than an imperceptible count trick.

h_generator_non_degeneracy_metric:
  - decode_diff_cells:
      Compare the true output against a wrong interpreter that keeps the mixed-room addition but
      uses B's displayed arrow directly instead of sign(disp_A - disp_B).
  - mix_diff_cells:
      Compare the true output against a wrong interpreter that decodes B correctly but collapses
      the mixed room back to its majority wall color.
  - Generation accepts a sample only if both metrics are positive.

i_post_edit_assumption_audit:
  - main() now assumes top-row reading order is the intended legend order; generate_input()
    enforces this by always placing A at TL and B at TR.
  - main() assumes room borders expose only legend fill colors, not stray structural colors;
    generate_input() enforces this by using exactly those colors for room walls.
  - main() assumes the mixed room really contains both wall colors; generate_input() enforces this
    by painting an A-majority but visibly bi-colored wall ring.
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
LEGEND_BUFFER = 1

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 4
GEN_RETRIES = 500


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


def _vector_sum_dir(vectors: list[tuple[int, int]]) -> tuple[int, int] | None:
    dr = _sign(sum(v[0] for v in vectors))
    dc = _sign(sum(v[1] for v in vectors))
    if dr == 0 and dc == 0:
        return None
    return dr, dc


def _decode_legend_component(g: np.ndarray, pts: np.ndarray) -> dict | None:
    h, w = g.shape
    if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
        return None

    r0, c0, r1, c1 = bbox_coords(pts)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if area != len(pts):
        return None

    counts: dict[int, int] = {}
    for r, c in pts:
        v = int(g[r, c])
        counts[v] = counts.get(v, 0) + 1
    if len(counts) != 3:
        return None

    fill_col = max(counts, key=counts.get)
    others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
    if len(others) != 2:
        return None
    others.sort()

    if others[0][0] == others[1][0]:
        center_rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
        scored = []
        for _cnt, col in others:
            pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
            scored.append((np.sum((pts_col.mean(axis=0) - center_rc) ** 2), col))
        scored.sort()
        center_col = scored[0][1]
        indicator_col = scored[1][1]
    else:
        indicator_col = others[0][1]
        center_col = others[1][1]

    center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
    indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
    if len(center) == 0 or len(indicator) == 0:
        return None

    cc = center.mean(axis=0)
    ci = indicator.mean(axis=0)
    display = (_sign(ci[0] - cc[0]), _sign(ci[1] - cc[1]))
    if display == (0, 0):
        return None

    return {
        "color": int(fill_col),
        "display": tuple(display),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(len(pts)),
    }


def _detect_legends(g: np.ndarray, bg: int) -> list[dict]:
    non_bg = g != bg
    best_by_color: dict[int, dict] = {}
    for pts in components_mask(non_bg, connectivity=4):
        info = _decode_legend_component(g, pts)
        if info is None:
            continue
        prev = best_by_color.get(info["color"])
        if prev is None or info["area"] > prev["area"]:
            best_by_color[info["color"]] = info
    legends = list(best_by_color.values())
    legends.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return legends


def _resolve_legends(g: np.ndarray, bg: int) -> list[dict]:
    legends = _detect_legends(g, bg)
    assert len(legends) >= 2, "fused variant requires two legends"
    first, second = legends[:2]
    actual_b = _vector_diff_dir(first["display"], second["display"])
    assert actual_b is not None, "legend vector difference must be non-zero"
    return [
        {"role": "A", "color": int(first["color"]), "display": tuple(first["display"]), "actual": tuple(first["display"])},
        {"role": "B", "color": int(second["color"]), "display": tuple(second["display"]), "actual": tuple(actual_b)},
    ]


def _region_direction(
    border_counts: dict[int, int],
    actual_dirs: dict[int, tuple[int, int]],
) -> tuple[int, int] | None:
    present = [color for color in actual_dirs if border_counts.get(color, 0) > 0]
    if not present:
        return None
    if len(present) == 1:
        return actual_dirs[present[0]]
    vectors = [actual_dirs[color] for color in present]
    return _vector_sum_dir(vectors)


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
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


def _solve_with_actual_dirs(grid: np.ndarray, actual_dirs: dict[int, tuple[int, int]]) -> np.ndarray:
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

        direction = _region_direction(border_counts, actual_dirs)
        if direction is None:
            continue
        dr, dc = direction

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = MOVER

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside enclosed rooms, but decode the room direction through a fused
    two-legend rule instead of a one-color one-arrow lookup.

    Input structure:
    - The background is black and yellow pixels sit in closed rooms bounded by colored wall rings.
    - Two legend blocks sit on the top border. Their dominant fill colors are the same two colors
      used for the room walls.
    - Some rooms are bordered by a single legend color, while one room is visibly bordered by both
      legend colors at once.

    Rule:
    - Read the legends from left to right as A then B.
    - A's true direction is the arrow shown inside A.
    - B's true direction is not its own shown arrow; it is sign(A_display - B_display).
    - A single-color room uses that legend's true direction, while a two-color room uses the
      vector sum of both true directions.
    - All yellow pixels in a room slide as far as possible in that resolved direction and stack.

    Key insight:
    The wall colors tell you whether a room listens to one legend or to both, and one of those
    legend directions must itself be decoded from the relationship between the two visible arrows.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legends = _resolve_legends(g, bg)
    actual_dirs = {int(leg["color"]): tuple(leg["actual"]) for leg in legends}
    return _solve_with_actual_dirs(g, actual_dirs)


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _legend_patch(fill: int, dr: int, dc: int) -> np.ndarray:
    if (dr, dc) in CARDINAL_DIRS:
        return _legend_5x5(fill, dr, dc)
    return _legend_6x7(fill, dr, dc)


def _sample_instruction_pair(
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int] | None]:
    candidates = []
    for disp_a in ALL_DIRS:
        for disp_b in ALL_DIRS:
            actual_b = _vector_diff_dir(disp_a, disp_b)
            if actual_b is None or actual_b == disp_b:
                continue
            mixed_actual = _vector_sum_dir([disp_a, actual_b])
            if mixed_actual is None or mixed_actual == disp_a or mixed_actual == actual_b:
                continue
            mixed_display_wrong = _vector_sum_dir([disp_a, disp_b])
            if mixed_display_wrong == mixed_actual:
                continue
            candidates.append((disp_a, disp_b, actual_b, mixed_actual, mixed_display_wrong))
    return rng.choice(candidates)


def _place_legends(rng: random.Random, grid: np.ndarray) -> list[dict]:
    _h, w = grid.shape
    color_a, color_b = rng.sample(WALL_COLOR_CHOICES, 2)
    disp_a, disp_b, actual_b, _mixed_actual, _mixed_display_wrong = _sample_instruction_pair(rng)

    legends = [
        {
            "role": "A",
            "color": int(color_a),
            "display": tuple(disp_a),
            "actual": tuple(disp_a),
            "corner": "TL",
        },
        {
            "role": "B",
            "color": int(color_b),
            "display": tuple(disp_b),
            "actual": tuple(actual_b),
            "corner": "TR",
        },
    ]

    for leg in legends:
        patch = _legend_patch(leg["color"], *leg["display"])
        ph, pw = patch.shape
        if leg["corner"] == "TL":
            r0, c0 = 0, 0
        else:
            r0, c0 = 0, w - pw
        paste(grid, patch, r0, c0)
        leg["bbox"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)

    return legends


def _rect_region(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _paint_room_walls(grid: np.ndarray, region: dict, legends_by_role: dict[str, dict]) -> None:
    if region["kind"] != "mixed":
        color = legends_by_role[region["role"]]["color"]
        for r, c in region["walls"]:
            grid[r, c] = color
        return

    rows = [r for r, _ in region["interior"]]
    cols = [c for _, c in region["interior"]]
    top = min(rows) - 1
    left = min(cols) - 1
    color_a = legends_by_role["A"]["color"]
    color_b = legends_by_role["B"]["color"]
    for r, c in region["walls"]:
        grid[r, c] = color_a if (r == top or c == left) else color_b


def _build_regions(legends: list[dict], grid_shape: tuple[int, int]) -> list[dict]:
    del grid_shape
    legends_by_role = {leg["role"]: leg for leg in legends}
    mixed_actual = _vector_sum_dir([legends_by_role["A"]["actual"], legends_by_role["B"]["actual"]])
    assert mixed_actual is not None

    regions = [
        {
            "kind": "pure",
            "role": "A",
            "interior": _rect_region(8, 2, 4, 5),
            "actual_dir": tuple(legends_by_role["A"]["actual"]),
            "decode_wrong_dir": tuple(legends_by_role["A"]["actual"]),
            "mix_wrong_dir": tuple(legends_by_role["A"]["actual"]),
        },
        {
            "kind": "pure",
            "role": "B",
            "interior": _rect_region(8, 13, 4, 5),
            "actual_dir": tuple(legends_by_role["B"]["actual"]),
            "decode_wrong_dir": tuple(legends_by_role["B"]["display"]),
            "mix_wrong_dir": tuple(legends_by_role["B"]["actual"]),
        },
        {
            "kind": "mixed",
            "role": "AB",
            "interior": _rect_region(14, 5, 4, 10),
            "actual_dir": tuple(mixed_actual),
            "decode_wrong_dir": tuple(_vector_sum_dir([legends_by_role["A"]["actual"], legends_by_role["B"]["display"]])),
            "mix_wrong_dir": tuple(legends_by_role["A"]["actual"]),
        },
    ]

    for region in regions:
        region["walls"] = _blob_wall_cells(region["interior"], GRID_H, GRID_W)

    return regions


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
    for region in regions:
        blob = region["interior"]
        dr, dc = region["actual_dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            candidates = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if not candidates:
                return False
            second = candidates[rng.randrange(len(candidates))]
            if second != first:
                output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
        already_stopped: set[tuple[int, int]] = set()
        yellows_input: list[tuple[int, int]] = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = MOVER
    return True


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [region["interior"] for region in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _simulate_region_dirs(input_grid: np.ndarray, regions: list[dict], dir_field: str) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    for region in regions:
        movers = [p for p in region["interior"] if int(input_grid[p]) == MOVER]
        for r, c in movers:
            out[r, c] = 0
        dr, dc = region[dir_field]
        for r, c in _slide_points(region["interior"], movers, dr, dc):
            out[r, c] = MOVER
    return out


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    simulated = _simulate_region_dirs(input_grid, regions, "actual_dir")
    if not np.array_equal(solver_out, simulated):
        return False
    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    return int((solver_out == MOVER).sum()) == int((input_grid == MOVER).sum())


def _fusion_metrics(input_grid: np.ndarray, regions: list[dict]) -> dict[str, int]:
    actual = main(input_grid)
    decode_wrong = _simulate_region_dirs(input_grid, regions, "decode_wrong_dir")
    mix_wrong = _simulate_region_dirs(input_grid, regions, "mix_wrong_dir")

    decode_mask = np.zeros(input_grid.shape, dtype=bool)
    mixed_mask = np.zeros(input_grid.shape, dtype=bool)
    for region in regions:
        target_mask = mixed_mask if region["kind"] == "mixed" else decode_mask if region["role"] == "B" else None
        if target_mask is None:
            continue
        for r, c in region["interior"]:
            target_mask[r, c] = True
    # Mixed room also depends on B's decoded direction, so count it for decode impact too.
    decode_mask |= mixed_mask

    return {
        "decode_diff_cells": int(np.count_nonzero(actual[decode_mask] != decode_wrong[decode_mask])),
        "mix_diff_cells": int(np.count_nonzero(actual[mixed_mask] != mix_wrong[mixed_mask])),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    legends = _place_legends(rng, grid)
    legends_by_role = {leg["role"]: leg for leg in legends}
    regions = _build_regions(legends, grid.shape)

    for region in regions:
        _paint_room_walls(grid, region, legends_by_role)

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None

    metrics = _fusion_metrics(input_grid, regions)
    if metrics["decode_diff_cells"] <= 0 or metrics["mix_diff_cells"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
