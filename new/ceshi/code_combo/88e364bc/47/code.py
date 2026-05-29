# Combo variant of 88e364bc (index 47)
# combo_spec: D3/3+D9/1
# Name: 反射路径+仅对角线方向

"""
design_analysis:
  fused_goal:
    keep:
      - wall color still selects a border legend, and the legend still encodes a direction by the
        center-to-indicator offset
      - yellow cells still move independently per enclosed room and later yellows still stack
        against earlier settled yellows
      - bounded passable components remain the solver's unit of work
    change:
      - directions are diagonal only, so every legend is the 6x7 diagonal template
      - motion is no longer monotone-to-wall; each yellow spends a fixed reflection budget and may
        bounce several times before stopping
      - rooms are generated as plain rectangles so the diagonal rebound pattern is visually clean:
        humans can see the Z-like shuttle between opposite walls

  a_concretization_and_trace:
    appearance:
      - the board is still 20x20 with bg=0, yellow movers=4, and wall colors in {3,5,6,7,8,9}
      - each legend is a corner-touching 6x7 filled rectangle with fill color equal to one wall
        color; there are no 5x5 cardinal legends anymore
      - each room is a sealed colored rectangle with empty interior
    trace_example:
      - take a room whose interior is rows 6..11 and cols 5..10, and a legend direction (+1,+1)
      - original task:
        - a yellow at (8,7) would slide monotonically to the southeast-most reachable cell and
          stay there
      - fused task:
        - it moves step by step along (+1,+1)
        - on reaching the east wall first, the column axis flips and the yellow continues along
          (+1,-1)
        - on touching the south wall later, the row axis flips and it continues along (-1,-1)
        - after REFLECT_STEPS total moves, it can stop mid-room rather than on the far wall
      - visible difference:
        - output yellows no longer merely pack one corner; they often land along an interior
          zigzag endpoint produced by one or more diagonal bounces

  b_complexity_calibration:
    controls:
      - legends: 1-2
      - directions: exactly 4 diagonals
      - rooms: always 2 rectangular rooms total, so every sample has repeated evidence
      - yellows: 1-2 per room
      - reflection budget: global fixed constant REFLECT_STEPS
    anti_simplification_checks:
      - one room per sample is forced to contain a multi-bounce trace
      - every placed yellow must end somewhere different from the original straight-slide solver
      - when there is one legend, two rooms share that wall color and direction, reinforcing the
        legend-to-wall mapping instead of leaving a singleton clue

  c_implicit_assumption_audit_of_main:
    - _legend_directions assumes border-touching solid rectangles with exactly three colors.
      Generator preserves that exactly with the original 6x7 diagonal legend template.
    - passable-region extraction assumes rooms are enclosed and not connected to the outside.
      Generator builds full rectangular wall perimeters with corners included, so bounded
      components equal the intended interiors.
    - border color lookup assumes one dominant wall color surrounds a room.
      Generator assigns exactly one wall color per room.
    - original _slide_points assumed monotone progress in a fixed direction.
      This assumption is invalid after D3/3, so the fused solver replaces it with a finite-step
      reflection simulator.
    - diagonal movement assumes corner-cut prevention via both orthogonal support cells.
      This remains valid and is now also used to decide which axis reflects on a wall hit.
    - stacking order still depends on the initial legend vector.
      Generator reverse-validates yellow placement under that same far-to-near ordering.

  d_placement_legality:
    - legends reserve a one-cell moat
    - each room reserves its walls, interior, and one extra exterior buffer cell before another
      object may be placed
    - room walls are kept strictly inside the 20x20 board
    - topology is revalidated by relabeling bounded background components after painting walls

  e_visual_observability:
    - the diagonal-only restriction is visible immediately because all legends share the 6x7
      diagonal shape and no cardinal legend appears
    - reflection is visible because the generator rejects any yellow whose fused endpoint matches
      its old straight-slide endpoint
    - a multi-bounce room is forced in every sample, so examples show more than a trivial single
      rebound

  f_cleanliness:
    - rectangles keep the room silhouettes simple and readable even when the final yellow lands
      away from a wall
    - walls, legends, and rooms are buffered so same-color structures do not touch accidentally
    - no extra markers or colors are added; only the yellow endpoints change

  g_change_inferability:
    - color still carries just one interpretable relation: legend fill color matches room walls
    - the new hidden variable is not a per-instance mystery; the reflection budget is global
    - because samples all show diagonal legends plus reflected endpoints, a human can infer that
      the path geometry, not a color remap or local exception, changed

  solver_generator_contract_after_change:
    - solver expects legends to decode to diagonal vectors in generated data
    - solver expects every movable room to be a bounded passable component with one wall color
    - generator guarantees at least one multi-bounce trace and that all chosen yellow starts are
      reflection-sensitive under the same occupancy order used by main
"""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REFLECT_STEPS = 10
YELLOWS_PER_REGION = [1, 2]
MIN_MOVES_FOR_SAMPLE = 4
PLACEMENT_ATTEMPTS = 160
ROOM_RETRIES = 220
GEN_RETRIES = 600


class TraceResult(NamedTuple):
    final: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    bounces: int
    moves: int


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and decode their center-to-indicator direction."""
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

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        if len(others) < 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            ranked = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                ranked.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            ranked.sort()
            center_col = ranked[0][1]
            indicator_col = ranked[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 or dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _linear_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    return (
        (nr, nc) in region_set
        and (r + dr, c) in region_set
        and (r, c + dc) in region_set
    )


def _forward_slide_linear(
    region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int, blocked: set[tuple[int, int]]
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _linear_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _move_if_possible(
    region_set: set[tuple[int, int]], occupied: set[tuple[int, int]], r: int, c: int, dr: int, dc: int
) -> tuple[tuple[int, int] | None, str]:
    nr, nc = r + dr, c + dc
    row_ok = (r + dr, c) in region_set
    col_ok = (r, c + dc) in region_set
    diag_ok = (nr, nc) in region_set
    if not (row_ok and col_ok and diag_ok):
        return None, "wall"
    if (nr, nc) in occupied:
        return None, "occupied"
    return (nr, nc), "move"


def _reflected_direction(
    region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int
) -> tuple[int, int] | None:
    row_block = (r + dr, c) not in region_set
    col_block = (r, c + dc) not in region_set
    diag_block = (r + dr, c + dc) not in region_set
    if not (row_block or col_block or diag_block):
        return None

    new_dr, new_dc = dr, dc
    flipped = False
    if row_block:
        new_dr = -new_dr
        flipped = True
    if col_block:
        new_dc = -new_dc
        flipped = True
    if diag_block and not row_block and not col_block:
        new_dr = -new_dr
        new_dc = -new_dc
        flipped = True
    if not flipped:
        return None
    return int(new_dr), int(new_dc)


def _trace_reflection(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    occupied: set[tuple[int, int]],
    *,
    energy: int = REFLECT_STEPS,
) -> TraceResult:
    r, c = start
    cur_dr, cur_dc = int(dr), int(dc)
    path: list[tuple[int, int]] = []
    bounces = 0

    for _ in range(int(energy)):
        moved, reason = _move_if_possible(region_set, occupied, r, c, cur_dr, cur_dc)
        if moved is not None:
            r, c = moved
            path.append((r, c))
            continue
        if reason == "occupied":
            break

        new_dir = _reflected_direction(region_set, r, c, cur_dr, cur_dc)
        if new_dir is None or new_dir == (cur_dr, cur_dc):
            break
        cur_dr, cur_dc = new_dir
        bounces += 1

        moved, _reason = _move_if_possible(region_set, occupied, r, c, cur_dr, cur_dc)
        if moved is None:
            break
        r, c = moved
        path.append((r, c))

    return TraceResult(final=(r, c), path=tuple(path), bounces=bounces, moves=len(path))


def _reflect_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> tuple[list[tuple[int, int]], list[TraceResult]]:
    if not points:
        return [], []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts: list[tuple[int, int]] = []
    traces: list[TraceResult] = []
    for start in points_sorted:
        trace = _trace_reflection(region_set, start, dr, dc, occupied)
        occupied.add(trace.final)
        out_pts.append(trace.final)
        traces.append(trace)
    return out_pts, traces


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        if dr == 0 or dc == 0:
            continue

        for r, c in movers:
            out[r, c] = bg
        new_pts, _traces = _reflect_points(pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover

    return out


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs

    passable = (g == bg) | (g == mover)
    out = g.copy()
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        if dr == 0 or dc == 0:
            continue

        ordered = sorted(movers, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
        blocked: set[tuple[int, int]] = set()
        for r, c in movers:
            out[r, c] = bg
        for start in ordered:
            final = _forward_slide_linear(pset, start, dr, dc, blocked)
            blocked.add(final)
            out[final] = mover
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move yellow pixels inside sealed rooms according to corner legends, but only along diagonal
    paths that can rebound off room walls.

    Input structure:
    - Background is the most common color; yellow movers (4) occupy enclosed empty rooms bounded by
      colored walls.
    - Border legends are solid 6x7 rectangles at grid corners whose fill color matches a room's
      wall color.
    - Each legend contains center and indicator marks whose offset encodes one of the four diagonal
      directions.

    Rule:
    - For each bounded passable room, identify its surrounding wall color and use the matching
      legend to get the initial diagonal direction.
    - Process yellows from farthest to nearest along that initial direction so already-settled
      yellows act as blockers.
    - Simulate up to a fixed number of diagonal steps. When the next step hits a wall, flip the
      row axis, column axis, or both depending on which supports are blocked, then continue if the
      reflected step is legal.

    Key insight:
    - The legend still tells you which way the yellow starts, but the final endpoint is no longer
      "the far wall." You must track a finite diagonal reflection path inside the room.
    """
    return _solve(grid)


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


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[dict] = []
    for corner, color in zip(corners, colors):
        dr, dc = rng.choice(DIAGONAL_DIRS)
        patch = _legend_6x7(color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"color": color, "dir": (dr, dc)})

    return legend_occupied, legends


def _rect_room(r0: int, c0: int, h_int: int, w_int: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    interior = {(r, c) for r in range(r0, r0 + h_int) for c in range(c0, c0 + w_int)}
    walls = set()
    top, bottom = r0 - 1, r0 + h_int
    left, right = c0 - 1, c0 + w_int
    for r in range(top, bottom + 1):
        walls.add((r, left))
        walls.add((r, right))
    for c in range(left, right + 1):
        walls.add((top, c))
        walls.add((bottom, c))
    return interior, walls


def _expanded_bbox_cells(r0: int, c0: int, h_int: int, w_int: int, margin: int) -> list[tuple[int, int]]:
    top = r0 - 1 - margin
    bottom = r0 + h_int + margin
    left = c0 - 1 - margin
    right = c0 + w_int + margin
    return [(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)]


def _candidate_starts(blob: set[tuple[int, int]], dr: int, dc: int) -> tuple[list[tuple[int, int]], bool]:
    good: list[tuple[int, int]] = []
    has_multi = False
    for start in blob:
        trace = _trace_reflection(blob, start, dr, dc, set())
        straight = _forward_slide_linear(blob, start, dr, dc, set())
        if trace.bounces >= 1 and trace.moves >= MIN_MOVES_FOR_SAMPLE and trace.final != straight:
            good.append(start)
            has_multi |= trace.bounces >= 2
    return good, has_multi


def _try_place_room(rng: random.Random, occupied: np.ndarray, spec: dict) -> dict | None:
    h, w = occupied.shape
    dr, dc = spec["dir"]

    for _ in range(ROOM_RETRIES):
        h_int = rng.randint(5, 8)
        w_int = rng.randint(5, 8)
        row_min = 2
        row_max = h - h_int - 2
        col_min = 2
        col_max = w - w_int - 2
        if row_min > row_max or col_min > col_max:
            return None

        r0 = rng.randint(row_min, row_max)
        c0 = rng.randint(col_min, col_max)
        reserve_cells = _expanded_bbox_cells(r0, c0, h_int, w_int, margin=1)
        if any(not (0 <= r < h and 0 <= c < w) or occupied[r, c] for r, c in reserve_cells):
            continue

        interior, walls = _rect_room(r0, c0, h_int, w_int)
        starts, has_multi = _candidate_starts(interior, dr, dc)
        if len(starts) < 2:
            continue
        if spec["require_multi_bounce"] and not has_multi:
            continue

        return {
            "color": spec["color"],
            "dir": (dr, dc),
            "interior": interior,
            "walls": walls,
            "starts_pool": starts,
            "bbox": (r0, c0, h_int, w_int),
            "require_multi_bounce": spec["require_multi_bounce"],
        }
    return None


def _place_rooms(rng: random.Random, legends: list[dict], legend_occupied: np.ndarray) -> list[dict] | None:
    occupied = legend_occupied.copy()
    if len(legends) == 1:
        specs = [dict(legends[0]), dict(legends[0])]
    else:
        specs = [dict(legends[0]), dict(legends[1])]
    multi_idx = rng.randrange(len(specs))
    for idx, spec in enumerate(specs):
        spec["require_multi_bounce"] = idx == multi_idx

    rooms: list[dict] = []
    for spec in specs:
        room = _try_place_room(rng, occupied, spec)
        if room is None:
            return None
        rooms.append(room)
        r0, c0, h_int, w_int = room["bbox"]
        for r, c in _expanded_bbox_cells(r0, c0, h_int, w_int, margin=1):
            occupied[r, c] = True
    return rooms


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


def _simulate_region_starts(
    blob: set[tuple[int, int]], starts: list[tuple[int, int]], dr: int, dc: int
) -> list[TraceResult]:
    ordered = sorted(starts, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    traces: list[TraceResult] = []
    blocked: set[tuple[int, int]] = set()
    for start in ordered:
        trace = _trace_reflection(blob, start, dr, dc, blocked)
        traces.append(trace)
        blocked.add(trace.final)
    return traces


def _place_region_yellows(
    rng: random.Random, reg: dict, grid: np.ndarray
) -> tuple[list[tuple[int, int]], list[TraceResult]] | None:
    blob = reg["interior"]
    dr, dc = reg["dir"]
    pool = reg["starts_pool"]

    desired_counts = [rng.choice(YELLOWS_PER_REGION), 1]
    for n_yellows in desired_counts:
        if len(pool) < n_yellows:
            continue
        for _ in range(PLACEMENT_ATTEMPTS):
            starts = rng.sample(pool, n_yellows)
            traces = _simulate_region_starts(blob, starts, dr, dc)
            ordered = sorted(starts, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
            finals = [trace.final for trace in traces]
            if len(set(finals)) != len(finals):
                continue

            blocked: set[tuple[int, int]] = set()
            ok = True
            for start, trace in zip(ordered, traces):
                straight_final = _forward_slide_linear(blob, start, dr, dc, blocked)
                if trace.bounces < 1 or trace.moves < MIN_MOVES_FOR_SAMPLE or trace.final == straight_final:
                    ok = False
                    break
                blocked.add(trace.final)
            if not ok:
                continue
            if reg["require_multi_bounce"] and not any(trace.bounces >= 2 for trace in traces):
                continue

            for r, c in ordered:
                grid[r, c] = 4
            return ordered, traces
    return None


def _place_yellows(
    rng: random.Random, regions: list[dict], input_grid: np.ndarray
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[TraceResult]]] | None:
    region_starts: dict[int, list[tuple[int, int]]] = {}
    region_traces: dict[int, list[TraceResult]] = {}
    any_multi = False

    for idx, reg in enumerate(regions):
        placed = _place_region_yellows(rng, reg, input_grid)
        if placed is None:
            return None
        starts, traces = placed
        region_starts[idx] = starts
        region_traces[idx] = traces
        any_multi |= any(trace.bounces >= 2 for trace in traces)

    if not any_multi:
        return None
    return region_starts, region_traces


def _expected_output(
    input_grid: np.ndarray, regions: list[dict], region_starts: dict[int, list[tuple[int, int]]]
) -> np.ndarray:
    out = input_grid.copy()
    for idx, reg in enumerate(regions):
        starts = region_starts[idx]
        for r, c in starts:
            out[r, c] = 0
        finals, _traces = _reflect_points(reg["interior"], starts, reg["dir"][0], reg["dir"][1])
        for r, c in finals:
            out[r, c] = 4
    return out


def _oracle_check(
    input_grid: np.ndarray, expected_out: np.ndarray, regions: list[dict], region_traces: dict[int, list[TraceResult]]
) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    if not np.array_equal(solver_out, expected_out):
        return False
    if np.array_equal(solver_out, _baseline_main(input_grid)):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False

    h, w = input_grid.shape
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == 4])):
        return False
    return any(trace.bounces >= 1 for traces in region_traces.values() for trace in traces)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _place_rooms(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]
    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    placed = _place_yellows(rng, regions, input_grid)
    if placed is None:
        return None
    region_starts, region_traces = placed

    expected_out = _expected_output(input_grid, regions, region_starts)
    if not _oracle_check(input_grid, expected_out, regions, region_traces):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
