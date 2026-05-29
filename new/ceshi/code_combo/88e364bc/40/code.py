# Combo variant of 88e364bc (index 40)
# combo_spec: D8/1+D14/3
# Name: 距离编码色块+密集像素

"""
Design analysis for the combo variant.

source_task:
  - 88e364bc

combo_summary:
  - Keep the original legend-decoded slide direction and enclosed-wall regions.
  - Increase every active region to 3-5 yellow movers.
  - After sliding, remove yellow entirely and replace each mover with a red distance bar
    of exactly the traveled length, attached to the far wall side of its lane.

concrete_visualization:
  - Use a rectangular room whose direction is, for example, right.
  - Suppose one lane ends at the wall-side cell p0 and the lane behind it is
    [p4, p3, p2, p1, p0] from input-side to wall-side.
  - If a yellow starts at p3, the original task would output a yellow at p0.
  - In this combo task, the yellow vanishes and cells [p0, p1, p2] become red,
    so the visible output is a length-3 bar touching the right wall.
  - With 4 different lanes and distances [1, 4, 2, 5], the room output becomes a
    compact wall-adjacent histogram: four nearby red bars of different lengths.

behavior_trace:
  - Original behavior:
    input yellows stay yellow and only their final resting coordinates matter.
  - Fused behavior:
    1. read direction from the matching border legend,
    2. slide each yellow exactly as before,
    3. measure the number of successful slide steps,
    4. erase the yellow,
    5. paint a red segment of that length backward from the terminal cell.
  - Main difference:
    output now depends on both destination and travel distance, and dense regions
    expose several bars at once instead of a tiny number of terminal yellow dots.

complexity_calibration:
  - The generator uses 1-2 legend colors and one room per color, so there is repeated
    support for the color->direction linkage without overfilling 20x20.
  - Directions are restricted to cardinal directions in generation. This is deliberate:
    diagonal distance bars on irregular walls are much harder for humans to parse from
    2-4 examples, while cardinal histogram bars are immediately legible.
  - Room interiors are narrow in the stacking axis and longer in the motion axis, which
    makes the distance signal dominant and avoids accidental overlap between bars.

implicit_assumption_audit_main:
  - _legend_directions still assumes border-touching filled rectangles with exactly three
    colors. The generator preserves that structure unchanged.
  - components_mask(passable, 4) still assumes each active room is an enclosed bg/yellow
    component. Rectangular one-color walls guarantee this.
  - Border color detection by majority adjacent wall color still assumes one dominant wall
    color per region. Each room wall is monochrome, so the assumption is exact.
  - _slide_with_distances assumes the solver order is far-to-near along the motion vector.
    That remains true; the generator mostly uses independent parallel lanes so the bars stay
    visually separable, but the solver itself still supports blocking.
  - The new bar-painting step assumes the traveled path cells stay inside the region when
    traced backward from the stop. The generator guarantees this by using straight lanes and
    distances no larger than the available lane length.
  - The new rule also assumes slide distance is at least 1, otherwise no bar would appear.
    The generator enforces distances in [1, max_lane].

placement_legality:
  - Legends occupy distinct corners and reserve a padding halo.
  - Each room is placed as a non-overlapping rectangle with an extra buffer around it.
  - Yellow starts are chosen on separate lanes, never on walls, never outside bounds, and
    never in the reserved legend zones.
  - Output bars are planned first, then inputs are derived by shifting each yellow backward,
    so solver correctness can be checked against an explicit expected output.

visual_observability:
  - The input shows several yellow seeds inside the same walled room, already suggesting that
    the region is more crowded than the source task.
  - The output removes yellow entirely and replaces it with a cluster of red bars hugging one
    wall; across examples, the wall side changes with the legend direction.
  - Because bar length varies within a room, a human can infer that distance matters rather
    than only final placement.

tidiness_and_inference:
  - Bars use a single dedicated marker color (red=2), distinct from wall colors and from the
    yellow movers, so object roles stay easy to separate.
  - The generator forces at least two distinct bar lengths per room, preventing degenerate
    examples where dense movers would collapse into visually identical marks.
  - This means the D8 density and D14 distance encoding are coupled: more yellows create more
    simultaneously visible distance bars, which is exactly the fused visual pattern.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste

BG = 0
CENTER_COLOR = 1
MARKER_COLOR = 2
MOVER_COLOR = 4

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]

LEGEND_BUFFER = 1
ROOM_BUFFER = 2
ROOM_LONG_RANGE = (6, 9)  # inclusive interior length along motion axis
ROOM_SLACK_RANGE = (1, 3)  # extra empty lanes help keep dense bars visually separable
GEN_RETRIES = 400


def _projection_key(p: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int]:
    return dr * p[0] + dc * p[1], p[0], p[1]


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend rectangles on the border and decode their directions.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) != 2:
            continue
        others.sort()

        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            choices = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                choices.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            choices.sort()
            center_col = choices[0][1]
            indicator_col = choices[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indic = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indic) == 0:
            continue

        delta = indic.mean(axis=0) - center.mean(axis=0)
        dr = 0 if abs(float(delta[0])) < 1e-6 else (1 if delta[0] > 0 else -1)
        dc = 0 if abs(float(delta[1])) < 1e-6 else (1 if delta[1] > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _slide_with_distances(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: _projection_key(p, dr, dc), reverse=True)
    occupied = set()
    out_pts: list[tuple[int, int, int]] = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        steps = 0
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
            steps += 1
        occupied.add((r, c))
        out_pts.append((r, c, steps))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each yellow pixel by the legend direction, then convert its travel distance into a red bar.

    Input structure:
    - Background is the most common color.
    - Border legends are filled rectangles whose dominant fill color matches a room's wall color.
    - Each enclosed room contains 3-5 yellow pixels on clear lanes inside a monochrome wall boundary.

    Rule:
    - For each enclosed room, infer its wall color and read the corresponding legend direction.
    - Slide all yellows exactly as in the source task, using far-to-near processing along that direction.
    - Remove the yellows; for each moved pixel, paint a red segment of length equal to the number of
      traveled steps, ending at the terminal cell and extending backward opposite the motion.

    Key insight:
    - The legend still tells you where motion goes, but the output no longer keeps the yellow endpoints.
      You must recover each slide distance and reinterpret the traveled path as a wall-adjacent bar chart.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = (g == bg) | (g == MOVER_COLOR)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER_COLOR]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == MOVER_COLOR:
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

        for r, c, steps in _slide_with_distances(pset, movers, dr, dc):
            for k in range(steps):
                rr = r - k * dr
                cc = c - k * dc
                out[rr, cc] = MARKER_COLOR

    return out


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((5, 5), fill, dtype=int)
    patch[1, 1] = CENTER_COLOR
    patch[1, 2] = CENTER_COLOR
    patch[1, 3] = MARKER_COLOR
    patch[3, 1] = CENTER_COLOR
    patch[3, 2] = CENTER_COLOR
    patch[3, 3] = MARKER_COLOR
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(patch, rot[(dr, dc)])


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wall_color in zip(corners, wall_colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))
    return occupied, legends


def _choose_region_specs(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
) -> list[dict]:
    specs = []
    for wall_color, direction in legends:
        n_yellows = rng.randint(3, 5)
        dr, dc = direction
        motion_len = rng.randint(*ROOM_LONG_RANGE)
        slack = rng.randint(*ROOM_SLACK_RANGE)
        if dr == 0:
            interior_h = n_yellows + slack
            interior_w = motion_len
        else:
            interior_h = motion_len
            interior_w = n_yellows + slack

        max_dist = interior_w - 1 if dr == 0 else interior_h - 1
        if max_dist < n_yellows:
            motion_len = max(motion_len, n_yellows + 1)
            if dr == 0:
                interior_w = motion_len
            else:
                interior_h = motion_len
            max_dist = interior_w - 1 if dr == 0 else interior_h - 1

        room_h = interior_h + 2
        room_w = interior_w + 2
        specs.append(
            {
                "color": wall_color,
                "dir": direction,
                "n_yellows": n_yellows,
                "interior_h": interior_h,
                "interior_w": interior_w,
                "room_h": room_h,
                "room_w": room_w,
                "area": room_h * room_w,
            }
        )
    specs.sort(key=lambda spec: spec["area"], reverse=True)
    return specs


def _candidate_origins(room_h: int, room_w: int) -> list[tuple[int, int]]:
    origins = []
    for r0 in range(1, GRID_H - room_h):
        for c0 in range(1, GRID_W - room_w):
            origins.append((r0, c0))
    return origins


def _fits_reserved(occupied: np.ndarray, r0: int, c0: int, room_h: int, room_w: int) -> bool:
    r_lo = max(0, r0 - ROOM_BUFFER)
    r_hi = min(GRID_H, r0 + room_h + ROOM_BUFFER)
    c_lo = max(0, c0 - ROOM_BUFFER)
    c_hi = min(GRID_W, c0 + room_w + ROOM_BUFFER)
    return not occupied[r_lo:r_hi, c_lo:c_hi].any()


def _reserve_room(occupied: np.ndarray, r0: int, c0: int, room_h: int, room_w: int) -> None:
    r_lo = max(0, r0 - ROOM_BUFFER)
    r_hi = min(GRID_H, r0 + room_h + ROOM_BUFFER)
    c_lo = max(0, c0 - ROOM_BUFFER)
    c_hi = min(GRID_W, c0 + room_w + ROOM_BUFFER)
    occupied[r_lo:r_hi, c_lo:c_hi] = True


def _paint_room(grid: np.ndarray, plan: dict) -> None:
    r0, c0 = plan["top_left"]
    room_h = plan["room_h"]
    room_w = plan["room_w"]
    color = plan["color"]
    grid[r0, c0 : c0 + room_w] = color
    grid[r0 + room_h - 1, c0 : c0 + room_w] = color
    grid[r0 : r0 + room_h, c0] = color
    grid[r0 : r0 + room_h, c0 + room_w - 1] = color


def _lane_indices(rng: random.Random, interior_size: int, n_yellows: int) -> list[int]:
    # Prefer a one-cell gap between active lanes when the room is wide/tall enough so
    # the resulting distance bars read as a compact histogram instead of one merged blob.
    if interior_size >= 2 * n_yellows - 1:
        max_offset = interior_size - (2 * n_yellows - 1)
        offset = rng.randint(0, max_offset)
        lanes = [offset + 2 * k for k in range(n_yellows)]
        return lanes

    lanes = list(range(interior_size))
    rng.shuffle(lanes)
    return sorted(lanes[:n_yellows])


def _sample_distances(rng: random.Random, max_dist: int, n_yellows: int) -> list[int]:
    distances = rng.sample(list(range(1, max_dist + 1)), n_yellows)
    rng.shuffle(distances)
    return distances


def _realize_plan(rng: random.Random, plan: dict) -> dict:
    r0, c0 = plan["top_left"]
    dr, dc = plan["dir"]
    interior_h = plan["interior_h"]
    interior_w = plan["interior_w"]
    n_yellows = plan["n_yellows"]

    if dr == 0:
        lanes = _lane_indices(rng, interior_h, n_yellows)
        distances = _sample_distances(rng, interior_w - 1, n_yellows)
    else:
        lanes = _lane_indices(rng, interior_w, n_yellows)
        distances = _sample_distances(rng, interior_h - 1, n_yellows)

    input_yellows: list[tuple[int, int]] = []
    bar_cells: list[tuple[int, int]] = []

    for lane, dist in zip(lanes, distances):
        if (dr, dc) == (0, 1):
            rr = r0 + 1 + lane
            term_r, term_c = rr, c0 + interior_w
            start_r, start_c = term_r, term_c - dist
        elif (dr, dc) == (0, -1):
            rr = r0 + 1 + lane
            term_r, term_c = rr, c0 + 1
            start_r, start_c = term_r, term_c + dist
        elif (dr, dc) == (1, 0):
            cc = c0 + 1 + lane
            term_r, term_c = r0 + interior_h, cc
            start_r, start_c = term_r - dist, term_c
        elif (dr, dc) == (-1, 0):
            cc = c0 + 1 + lane
            term_r, term_c = r0 + 1, cc
            start_r, start_c = term_r + dist, term_c
        else:
            raise ValueError((dr, dc))

        input_yellows.append((start_r, start_c))
        for k in range(dist):
            bar_cells.append((term_r - k * dr, term_c - k * dc))

    if len(set(input_yellows)) != len(input_yellows):
        raise ValueError("input yellows must be distinct")
    if len(set(distances)) < 2:
        raise ValueError("need at least two distinct distances per room")

    realized = dict(plan)
    realized["input_yellows"] = input_yellows
    realized["bar_cells"] = bar_cells
    realized["distances"] = distances
    return realized


def _build_layout(rng: random.Random, legends: list[tuple[int, tuple[int, int]]], occupied: np.ndarray) -> list[dict] | None:
    specs = _choose_region_specs(rng, legends)
    plans = []
    for spec in specs:
        candidates = _candidate_origins(spec["room_h"], spec["room_w"])
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if _fits_reserved(occupied, r0, c0, spec["room_h"], spec["room_w"]):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        spec = dict(spec)
        spec["top_left"] = chosen
        _reserve_room(occupied, chosen[0], chosen[1], spec["room_h"], spec["room_w"])
        plans.append(spec)

    realized = []
    for plan in plans:
        try:
            realized.append(_realize_plan(rng, plan))
        except ValueError:
            return None
    return realized


def _expected_output(base_grid: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = base_grid.copy()
    for plan in plans:
        for r, c in plan["input_yellows"]:
            out[r, c] = BG
        for r, c in plan["bar_cells"]:
            out[r, c] = MARKER_COLOR
    return out


def _oracle_check(input_grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    return np.array_equal(solver_out, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    occupied, legends = _place_legends(rng, grid)
    plans = _build_layout(rng, legends, occupied.copy())
    if plans is None:
        return None

    for plan in plans:
        _paint_room(grid, plan)

    input_grid = grid.copy()
    for plan in plans:
        for r, c in plan["input_yellows"]:
            input_grid[r, c] = MOVER_COLOR

    expected = _expected_output(grid, plans)
    if not _oracle_check(input_grid, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
