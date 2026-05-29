# Variant of 88e364bc
# Dimension: D19 — 图例数量与方向多样性 (symbol_instructions)
# Change: 1-2个图例，每个图例对应一种墙壁颜色和一个方向... → 图例方向需要解码计算

"""
Design analysis for the D19 variant
===================================

source_task:
  reference_task: none
  action: skipped

a_concretization_and_behavior_trace:
  preserved_visual_language:
    - 20x20 black background.
    - Colored wall loops enclose irregular rooms.
    - Yellow pixels slide inside rooms.
    - Border legends are still filled rectangles using the room-wall color as fill,
      blue(1) as center mark, and red(2) as indicator mark.
  concrete_variant:
    - There are exactly two legends so the "difference of two legend vectors" is always visible.
    - Legend A is fixed at the top-left corner.
    - Legend B is fixed at the top-right corner.
    - Rooms with wall color A still use A's displayed direction directly.
    - Rooms with wall color B do not use B's displayed direction directly.
      Their true slide direction is sign(display_A - display_B), computed componentwise.
    - Example:
        display_A = right  = (0, 1)
        display_B = down   = (1, 0)
        true_B    = sign((0,1) - (1,0)) = (-1, 1) = up-right
      In the original task, B-rooms would slide down because B directly encoded down.
      In this variant, those same yellow pixels slide up-right instead.
  pipeline_impact:
    solver:
      - Legend parsing can no longer map each legend independently to its wall color.
      - The two legends must be ordered first, then B's true direction must be derived.
    generator:
      - It must always place two legends.
      - It must generate at least one sample region for the second wall color whose output
        differs from the old direct-legend interpretation, otherwise the variant would visually
        collapse back to the source task.

b_complexity_calibration:
  - The arithmetic rule is new, but the rest of the scene stays close to the source task.
  - Only one derived direction is introduced: the second legend's color uses sign(A-B).
    This keeps the number of instruction types small enough for humans to infer.
  - Two wall colors remain enough to show repeated evidence without overcrowding 20x20.
  - The generator rejects direction pairs where B's derived direction equals A or B directly,
    because that would make the arithmetic less visible.

c_implicit_assumptions_review_of_main:
  old_assumptions_broken_by_variant:
    - _legend_directions previously assumed "one legend component => one final movement vector".
      That is no longer true for the second legend.
    - The old dict[color] = decoded_arrow logic was safe only because each region direction was
      local to one legend. Now direction for the second wall color depends on another legend.
  line_level_consequences:
    - Legend detection itself is still valid: legends are border-touching filled rectangles with
      exactly three colors and a center/indicator centroid offset.
    - Region extraction, border-color counting, slide order, and diagonal corner-cut checks remain valid.
    - The only solver change needed is to replace direct per-legend direction assignment with:
        1. detect legend A and legend B in reading order;
        2. decode their displayed vectors;
        3. assign color(A) -> display_A;
        4. assign color(B) -> sign(display_A - display_B).
  new_solver_assumptions:
    - Exactly two legends are present and both are detectable as filled border rectangles.
    - Their reading order is visually stable because the generator always uses TL for A and TR for B.
    - sign(display_A - display_B) is non-zero; the generator guarantees this.

d_placement_legality:
  - Legends sit in opposite top corners with a one-cell reservation buffer, so they never overlap walls.
  - Wall loops are still grown only on free interior cells and their wall rings may not touch the border.
  - Each region stores both its true direction and its legacy direct direction so the generator can
    verify the variant is active.
  - Reverse-sliding still uses the true direction, so every generated input is oracle-checked by main().

e_visual_observability:
  - Humans can compare the two visible arrows directly because both legends are always present.
  - The second wall color repeatedly violates the naive "follow your own arrow" rule.
  - The first wall color still provides the anchor meaning of a normal legend arrow,
    which helps the solver notice that only the second color behaves differently.

f_visual_cleanliness:
  - Fixing legends to the top corners removes ordering ambiguity and keeps the arithmetic cue tidy.
  - Region counts are capped to avoid clutter under the two top legends.
  - Walls, legends, and yellow movers still occupy separate layers with clear colors.

g_color_change_inference:
  - No new color remapping is introduced.
  - Colors only serve the same structural role as before: legend fill color matches wall color.
  - The new inference burden is directional arithmetic, not a hidden color code.

h_generator_non_degeneracy_metric:
  - Metric: compare variant output against a "legacy" output that incorrectly uses B's displayed
    direction directly. At least one cell inside a B-colored room must differ.
  - This rejects samples where the arithmetic exists in principle but does not affect the visible output.

i_post_edit_assumption_audit:
  - main() now assumes the first two detected legends are the intended ordered pair.
    generate_input() enforces this by always placing exactly two legends at TL and TR.
  - main() assumes the derived vector is one of the 8 legal ARC movement directions.
    generate_input() enforces this by sampling only pairs with a non-zero componentwise sign difference.
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
GEN_RETRIES = 400


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
        rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
        candidates = []
        for _cnt, col in others:
            pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
            candidates.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
        candidates.sort()
        center_col = candidates[0][1]
        indicator_col = candidates[1][1]
    else:
        indicator_col = others[0][1]
        center_col = others[1][1]

    center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
    indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
    if not center or not indic:
        return None

    cc = np.mean(np.array(center, dtype=float), axis=0)
    ci = np.mean(np.array(indic, dtype=float), axis=0)
    display = (_sign(ci[0] - cc[0]), _sign(ci[1] - cc[1]))
    if display == (0, 0):
        return None

    return {
        "color": fill_col,
        "display": display,
        "bbox": (r0, c0, r1, c1),
        "area": len(pts),
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


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    legends = _detect_legends(g, bg)
    assert len(legends) >= 2, "variant requires two detected legends"

    first, second = legends[:2]
    derived_second = _vector_diff_dir(first["display"], second["display"])
    assert derived_second is not None, "legend vector subtraction must yield a non-zero direction"

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
    Variant rule:
    - Decode two border legends in reading order: A at top-left, B at top-right.
    - A-colored regions still slide in A's displayed direction.
    - B-colored regions slide in sign(display_A - display_B), not in B's displayed direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    return _solve_with_dirs(g, dirs)


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
    for d1 in ALL_DIRS:
        for d2 in ALL_DIRS:
            if d1 == d2:
                continue
            derived = _vector_diff_dir(d1, d2)
            if derived is None:
                continue
            if derived == d1 or derived == d2:
                continue
            candidates.append((d1, d2, derived))
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
        patch = _legend_patch(leg["color"], *leg["display"])
        ph, pw = patch.shape
        if leg["corner"] == "TL":
            r0, c0 = 0, 0
        else:
            r0, c0 = 0, w - pw
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
    actual_dirs: dict[int, tuple[int, int]] = {}
    legacy_dirs: dict[int, tuple[int, int]] = {}
    affected_colors = set()
    for reg in regions:
        actual_dirs[reg["color"]] = reg["dir"]
        legacy_dirs[reg["color"]] = reg["legacy_dir"]
        if reg["dir"] != reg["legacy_dir"]:
            affected_colors.add(reg["color"])
    if not affected_colors:
        return False

    variant_out = main(input_grid)
    legacy_out = _solve_with_dirs(input_grid, legacy_dirs)
    mask = np.zeros(input_grid.shape, dtype=bool)
    for reg in regions:
        if reg["color"] not in affected_colors:
            continue
        for r, c in reg["interior"]:
            mask[r, c] = True
    return bool(mask.any() and np.any(variant_out[mask] != legacy_out[mask]))


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
