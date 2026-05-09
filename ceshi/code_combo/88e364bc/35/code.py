# Combo variant of 88e364bc (index 35)
# combo_spec: D4/1+D21/1
# Name: 内部障碍物+颜色标记停止

from collections import deque
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

"""
Design analysis for the combo variant
=====================================

combo_intent:
  preserved:
    - border legends still map wall color -> movement direction
    - each movable object is a yellow pixel inside an enclosed colored room
    - output keeps legends / walls unchanged and only relocates yellows
  fused_change:
    - room interiors now contain two obstacle roles:
      - green obstacle cells (3): ordinary internal blockers
      - blue stop-obstacle cells (1): also blockers, but specifically mark where motion must stop
    - a yellow does not slide blindly to the far wall anymore
    - instead, it navigates through the room toward the legend direction, detouring around green
      blockers, and stops on the last passable cell immediately before a blue stop obstacle

a) concrete behavior trace
--------------------------
- Right-moving room, simplified row-level picture:
    wall | yellow .... green-bar-with-gap .... blue-stop . wall
  but the yellow and blue are on the same row while the green bar blocks that row, so the only
  route goes up or down through the gap and then back to the blue's row.
- Original task:
  it considered only straight repeated (0,+1) steps, so it would stop at the first blocker or far
  wall and had no notion of internal routing or stop markers.
- D21-only behavior:
  it would route around the green bar and then continue past the blue cell by going around it too,
  finally resting on the true far-right side.
- Fused behavior here:
  it still routes around the green bar, but once it reaches a passable cell whose next forward cell
  is blue, that cell becomes the destination. The blue cell is not traversed; it is the stopping
  signal carried by an obstacle.
- Visible difference:
  output yellows end one step before a blue obstacle, even though more space remains beyond that
  obstacle and is reachable by detouring.

b) complexity calibration
-------------------------
- Only two new interior semantics are introduced and they are stable across all samples:
  green = ordinary obstacle, blue = stop obstacle.
- Legends are restricted to cardinal directions in generated samples. This keeps the fusion cue
  legible: humans can focus on "detour around green, then stop before blue" instead of also having
  to disentangle diagonal routing.
- Each room contains exactly one yellow and one blue stop obstacle. This avoids ambiguous multi-
  target matching while still making the fused rule obvious in 2-4 examples.

c) hidden assumption audit of the original main()
-------------------------------------------------
1. `passable = (g == bg) | (g == mover)`:
   broken. The bounded room is now the meta-component `{bg, yellow, green obstacle, blue stop}`.
   Using only passable cells would split the room at internal obstacles and lose the border color.
2. `components_mask(passable, connectivity=4)`:
   broken for the same reason. Region identity and motion state space diverge; room detection must
   include obstacle cells, while movement excludes them.
3. `_slide_points(... keep stepping dr,dc ...)`:
   broken twice. The fused rule needs path planning around blockers and a new terminal predicate:
   "current cell is directly before a blue stop obstacle in the motion direction".
4. Border-color counting ignored only bg and yellow:
   broken. It must also ignore both internal obstacle colors, otherwise the solver could mistake an
   interior obstacle for the wall color controlling the room.
5. Original generator assumed any non-wall interior cell was floor:
   broken. Placement must guarantee green blockers stay internal, blue stop cells are reachable from
   the correct side, and the room still has a valid detouring path.

Potential fusion conflict and resolution:
- D4/1 alone used a passable colored marker that a yellow lands on.
- D21/1 makes obstacles non-passable.
- The fusion resolves this by turning the stop cue into a non-passable blue obstacle. The yellow
  therefore stops immediately before the cue instead of landing on it. That is the key combined
  behavior, not a simple concatenation of the two single variants.

d) placement legality
---------------------
- Rooms are axis-aligned rectangles with a one-cell exterior buffer, so legends and different rooms
  never overlap or touch ambiguously.
- Green bars are placed strictly inside the room, leaving at least one gap so passable floor stays
  connected.
- Blue stop obstacles are placed strictly inside the room and away from the perimeter so there is
  visible space beyond them. This prevents degeneration into "blue acts like a wall".
- The generator verifies:
  - the yellow's chosen destination is the unique best reachable stop-front cell
  - the path contains an off-axis detour
  - an obstacle-only baseline would end farther in the legend direction, so the stop rule matters

e) visual observability
-----------------------
- From examples, a human can see the repeated pattern:
  legend points right/left/up/down; green bar blocks a direct route; blue obstacle sits before the
  far side; the yellow detours around green but halts one cell before blue instead of reaching the
  actual far wall.
- No rule depends on invisible numeric ordering. Destination ranking uses geometric projection onto
  the legend direction, which corresponds to an observable "more to the right/left/up/down".

f) visual cleanliness
---------------------
- Rectangular rooms are intentional. The original irregular fences would combine poorly with two
  interior obstacle types and make the new cue harder to spot.
- There is exactly one blue stop obstacle per room and sparse green structure, so the room interior
  stays readable instead of becoming a dense maze.
- Blue does appear in legends as center dots, but only inside border rectangles; inside rooms, a
  lone blue cell consistently means stop obstacle. Location disambiguates the role cleanly.

g) inferability of the change
-----------------------------
- No free-form color mapping is introduced. Blue always means "stop obstacle" inside rooms and
  green always means "ordinary obstacle".
- The fused rule has repeated structural support across samples: same yellow, same blue cue, same
  detour requirement, same legend-to-wall-color relation.

new solver assumptions introduced here
--------------------------------------
- Generated samples use only cardinal legend directions.
- Each yellow-bearing room contains at least one reachable blue stop obstacle and exactly one yellow.
- The best destination is the farthest reachable passable cell immediately before a blue stop
  obstacle, with ties broken by shortest route and smallest sideways deviation.
- The generator enforces each assumption explicitly.
"""


STOP_OBSTACLE = 1
LEGEND_INDICATOR = 2
OBSTACLE = 3
MOVER = 4

GRID_H, GRID_W = 20, 20
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
ROOM_BUFFER = 1
MAX_TOTAL_REGIONS = 2
ROOM_RETRIES = 120
GEN_RETRIES = 300


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legend rectangles and decode their directions."""
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
        if len(others) < 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
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
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _perp_offset(start: tuple[int, int], end: tuple[int, int], dr: int, dc: int) -> int:
    sr, sc = start
    er, ec = end
    return abs((er - sr) * dc - (ec - sc) * dr)


def _step_order(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == 0 and dc == 1:
        return [(0, 1), (-1, 0), (1, 0), (0, -1)]
    if dr == 0 and dc == -1:
        return [(0, -1), (-1, 0), (1, 0), (0, 1)]
    if dr == 1 and dc == 0:
        return [(1, 0), (0, -1), (0, 1), (-1, 0)]
    if dr == -1 and dc == 0:
        return [(-1, 0), (0, -1), (0, 1), (1, 0)]

    steps = [
        (dr, dc),
        (dr, 0),
        (0, dc),
        (-dr, dc),
        (dr, -dc),
        (-dr, 0),
        (0, -dc),
        (-dr, -dc),
    ]
    out = []
    seen = set()
    for step in steps:
        if step != (0, 0) and step not in seen:
            out.append(step)
            seen.add(step)
    return out


def _neighbors_for_motion(
    passable: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    pos: tuple[int, int],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    r, c = pos
    out = []
    for sdr, sdc in _step_order(dr, dc):
        nr, nc = r + sdr, c + sdc
        nxt = (nr, nc)
        if nxt not in passable or nxt in blocked:
            continue
        if abs(sdr) == 1 and abs(sdc) == 1:
            side1 = (r + sdr, c)
            side2 = (r, c + sdc)
            if side1 not in passable or side2 not in passable:
                continue
            if side1 in blocked or side2 in blocked:
                continue
        out.append(nxt)
    return out


def _reachable_info(
    passable: set[tuple[int, int]],
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    dist = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _neighbors_for_motion(passable, blocked, cur, dr, dc):
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            parent[nxt] = cur
            q.append(nxt)
    return dist, parent


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None],
    start: tuple[int, int],
    dest: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [dest]
    while path[-1] != start:
        prev = parent[path[-1]]
        assert prev is not None
        path.append(prev)
    path.reverse()
    return path


def _best_reachable_destination(
    passable: set[tuple[int, int]],
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    dist, parent = _reachable_info(passable, start, blocked, dr, dc)
    dest = min(
        dist,
        key=lambda p: (
            -_projection(p, dr, dc),
            dist[p],
            _perp_offset(start, p, dr, dc),
            p[0],
            p[1],
        ),
    )
    return dest, _reconstruct_path(parent, start, dest)


def _best_stop_destination(
    passable: set[tuple[int, int]],
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
    stop_cells: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if not stop_cells:
        raise ValueError("combo rooms must contain at least one stop obstacle")
    dist, parent = _reachable_info(passable, start, blocked, dr, dc)
    candidates = [p for p in dist if (p[0] + dr, p[1] + dc) in stop_cells]
    if not candidates:
        raise ValueError("no reachable stop-front destination for this mover")
    dest = min(
        candidates,
        key=lambda p: (
            -_projection(p, dr, dc),
            dist[p],
            _perp_offset(start, p, dr, dc),
            p[0],
            p[1],
        ),
    )
    return dest, _reconstruct_path(parent, start, dest)


def _path_has_detour(path: list[tuple[int, int]], dr: int, dc: int) -> bool:
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        if (r1 - r0, c1 - c0) != (dr, dc):
            return True
    return False


def _room_data_from_grid(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    room_mask = (g == bg) | (g == MOVER) | (g == OBSTACLE) | (g == STOP_OBSTACLE)
    rooms = []
    for pts in components_mask(room_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        room = {(int(r), int(c)) for r, c in pts}
        movers = [p for p in room if int(g[p]) == MOVER]
        if not movers:
            continue
        passable = {p for p in room if int(g[p]) in (bg, MOVER)}
        stop_cells = {p for p in room if int(g[p]) == STOP_OBSTACLE}
        border_counts: dict[int, int] = {}
        for r, c in room:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in room:
                    continue
                v = int(g[nr, nc])
                if v in (bg, MOVER, OBSTACLE, STOP_OBSTACLE):
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        rooms.append(
            {
                "room": room,
                "passable": passable,
                "movers": movers,
                "stop_cells": stop_cells,
                "dir": dirs[border_col],
                "border_color": border_col,
            }
        )
    return rooms


def _path_planned_stop_slide(
    passable: set[tuple[int, int]],
    stop_cells: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for start in points_sorted:
        dest, _ = _best_stop_destination(passable, start, occupied, stop_cells, dr, dc)
        occupied.add(dest)
        out_pts.append(dest)
    return out_pts


def _obstacle_only_main(grid: np.ndarray) -> np.ndarray:
    """Baseline used only for generator checks and post-hoc metrics."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    for room in _room_data_from_grid(g):
        passable = room["passable"]
        movers = room["movers"]
        dr, dc = room["dir"]
        for r, c in movers:
            out[r, c] = bg
        occupied: set[tuple[int, int]] = set()
        points_sorted = sorted(movers, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
        for start in points_sorted:
            dest, _ = _best_reachable_destination(passable, start, occupied, dr, dc)
            occupied.add(dest)
            out[dest] = MOVER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route yellow pixels through enclosed rooms using border legends and stop-obstacle cues.

    Input structure:
    - The background is black, border legends sit in corners, and each legend fill color matches
      the wall color of one or more enclosed rooms.
    - Inside a room, yellow pixels move on black floor; green cells are ordinary internal
      obstacles; blue cells are also internal obstacles but additionally mark where motion should
      terminate when approached from the legend direction.

    Rule:
    - For each bounded room, recover the legend direction from the wall color exactly as in the
      source task.
    - Treat the whole room `{floor, yellow, green, blue}` as one object for region detection, but
      allow movement only on `{floor, yellow}` cells.
    - A yellow searches the reachable passable cells in that room and ends at the farthest one
      whose next step in the legend direction would hit a blue stop obstacle. Later yellows in the
      same room cannot reuse already occupied destinations.

    Key insight:
    - The new reasoning chain combines two interior semantics: green cells change the route, while
      blue obstacle cells change the stopping condition. The destination is therefore not "the far
      wall" and not "the blue cell itself", but the last reachable floor cell directly before blue.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    rooms = _room_data_from_grid(g)
    assert rooms, "at least one bounded legend-controlled room is required"

    for room in rooms:
        movers = room["movers"]
        if not movers:
            continue
        dr, dc = room["dir"]
        for r, c in movers:
            out[r, c] = bg
        new_pts = _path_planned_stop_slide(room["passable"], room["stop_cells"], movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = MOVER
    return out


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = STOP_OBSTACLE
    g[1, 2] = STOP_OBSTACLE
    g[1, 3] = LEGEND_INDICATOR
    g[3, 1] = STOP_OBSTACLE
    g[3, 2] = STOP_OBSTACLE
    g[3, 3] = LEGEND_INDICATOR
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


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


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wc in zip(corners, wall_colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))
    return legend_occupied, legends


def _room_wall_cells(r0: int, c0: int, h_out: int, w_out: int) -> set[tuple[int, int]]:
    r1 = r0 + h_out - 1
    c1 = c0 + w_out - 1
    walls = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                walls.add((r, c))
    return walls


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _room_ranges(r0: int, c0: int, h_out: int, w_out: int) -> tuple[int, int, int, int]:
    r1 = r0 + h_out - 1
    c1 = c0 + w_out - 1
    return r1, c1, r0 + 1, c0 + 1


def _build_room_objects(
    r0: int,
    c0: int,
    h_out: int,
    w_out: int,
    dr: int,
    dc: int,
    rng: random.Random,
) -> dict | None:
    r1, c1, ri0, ci0 = _room_ranges(r0, c0, h_out, w_out)
    interior = {(r, c) for r in range(ri0, r1) for c in range(ci0, c1)}
    rows = list(range(ri0, r1))
    cols = list(range(ci0, c1))

    if dr == 0 and dc == 1:
        if len(rows) < 5 or len(cols) < 6:
            return None
        bar_c = rng.randint(ci0 + 2, c1 - 4)
        gap_r = rng.choice(rows[1:-1])
        stop_r_choices = [r for r in rows[1:-1] if r != gap_r]
        if not stop_r_choices:
            return None
        stop_r = rng.choice(stop_r_choices)
        stop_cell = (stop_r, c1 - 2)
        start = (stop_r, ci0)
        output = (stop_r, c1 - 3)
        obstacles = {(r, bar_c) for r in rows if r != gap_r}
    elif dr == 0 and dc == -1:
        if len(rows) < 5 or len(cols) < 6:
            return None
        bar_c = rng.randint(ci0 + 3, c1 - 3)
        gap_r = rng.choice(rows[1:-1])
        stop_r_choices = [r for r in rows[1:-1] if r != gap_r]
        if not stop_r_choices:
            return None
        stop_r = rng.choice(stop_r_choices)
        stop_cell = (stop_r, ci0 + 1)
        start = (stop_r, c1 - 1)
        output = (stop_r, ci0 + 2)
        obstacles = {(r, bar_c) for r in rows if r != gap_r}
    elif dr == 1 and dc == 0:
        if len(rows) < 6 or len(cols) < 5:
            return None
        bar_r = rng.randint(ri0 + 2, r1 - 4)
        gap_c = rng.choice(cols[1:-1])
        stop_c_choices = [c for c in cols[1:-1] if c != gap_c]
        if not stop_c_choices:
            return None
        stop_c = rng.choice(stop_c_choices)
        stop_cell = (r1 - 2, stop_c)
        start = (ri0, stop_c)
        output = (r1 - 3, stop_c)
        obstacles = {(bar_r, c) for c in cols if c != gap_c}
    elif dr == -1 and dc == 0:
        if len(rows) < 6 or len(cols) < 5:
            return None
        bar_r = rng.randint(ri0 + 3, r1 - 3)
        gap_c = rng.choice(cols[1:-1])
        stop_c_choices = [c for c in cols[1:-1] if c != gap_c]
        if not stop_c_choices:
            return None
        stop_c = rng.choice(stop_c_choices)
        stop_cell = (ri0 + 1, stop_c)
        start = (r1 - 1, stop_c)
        output = (ri0 + 2, stop_c)
        obstacles = {(bar_r, c) for c in cols if c != gap_c}
    else:
        return None

    if start == output or start == stop_cell or output == stop_cell:
        return None

    passable = interior - obstacles - {stop_cell}
    if start not in passable or output not in passable:
        return None
    if not _connected(passable):
        return None

    try:
        combo_dest, combo_path = _best_stop_destination(passable, start, set(), {stop_cell}, dr, dc)
    except ValueError:
        return None
    obstacle_dest, obstacle_path = _best_reachable_destination(passable, start, set(), dr, dc)
    if combo_dest != output:
        return None
    if obstacle_dest == combo_dest:
        return None
    if _projection(obstacle_dest, dr, dc) <= _projection(combo_dest, dr, dc):
        return None
    if not _path_has_detour(combo_path, dr, dc):
        return None
    if not _path_has_detour(obstacle_path, dr, dc):
        return None
    if (combo_dest[0] + dr, combo_dest[1] + dc) != stop_cell:
        return None

    return {
        "passable": passable,
        "obstacles": obstacles,
        "stop_cell": stop_cell,
        "input_yellow": start,
        "output_yellow": output,
        "combo_path": combo_path,
        "obstacle_path": obstacle_path,
    }


def _sample_room(
    rng: random.Random,
    occupied: np.ndarray,
    wall_color: int,
    direction: tuple[int, int],
    total_regions: int,
) -> dict | None:
    h, w = occupied.shape
    dr, dc = direction

    for _ in range(ROOM_RETRIES):
        if dr == 0:
            if total_regions == 1:
                h_in = rng.randint(6, 8)
                w_in = rng.randint(8, 10)
            elif total_regions == 2:
                h_in = rng.randint(6, 7)
                w_in = rng.randint(8, 9)
            else:
                h_in = rng.randint(5, 6)
                w_in = rng.randint(7, 8)
        else:
            if total_regions == 1:
                h_in = rng.randint(8, 10)
                w_in = rng.randint(6, 8)
            elif total_regions == 2:
                h_in = rng.randint(8, 9)
                w_in = rng.randint(6, 7)
            else:
                h_in = rng.randint(7, 8)
                w_in = rng.randint(5, 6)

        h_out, w_out = h_in + 2, w_in + 2
        if h_out + 2 >= h or w_out + 2 >= w:
            continue

        r0 = rng.randint(1, h - h_out - 1)
        c0 = rng.randint(1, w - w_out - 1)
        rr0 = max(0, r0 - ROOM_BUFFER)
        rr1 = min(h, r0 + h_out + ROOM_BUFFER)
        cc0 = max(0, c0 - ROOM_BUFFER)
        cc1 = min(w, c0 + w_out + ROOM_BUFFER)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue

        built = _build_room_objects(r0, c0, h_out, w_out, dr, dc, rng)
        if built is None:
            continue

        walls = _room_wall_cells(r0, c0, h_out, w_out)
        room = built["passable"] | built["obstacles"] | {built["stop_cell"]}
        return {
            "color": wall_color,
            "dir": (dr, dc),
            "walls": walls,
            "passable": built["passable"],
            "obstacles": built["obstacles"],
            "stop_cell": built["stop_cell"],
            "room": room,
            "input_yellow": built["input_yellow"],
            "output_yellow": built["output_yellow"],
            "combo_path": built["combo_path"],
            "obstacle_path": built["obstacle_path"],
            "reserve": (rr0, rr1, cc0, cc1),
        }
    return None


def _build_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    color_dir = {wc: d for wc, d in legends}
    region_colors = [wc for wc, _ in legends]
    if len(region_colors) < MAX_TOTAL_REGIONS and rng.random() < 0.6:
        region_colors.append(rng.choice(region_colors))
    rng.shuffle(region_colors)

    occupied = legend_occupied.copy()
    regions = []
    total_regions = len(region_colors)
    for color in region_colors:
        reg = _sample_room(rng, occupied, color, color_dir[color], total_regions)
        if reg is None:
            return None
        rr0, rr1, cc0, cc1 = reg["reserve"]
        occupied[rr0:rr1, cc0:cc1] = True
        regions.append(reg)

    if {reg["color"] for reg in regions} != set(color_dir):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    meta = (grid == 0) | (grid == OBSTACLE) | (grid == STOP_OBSTACLE) | (grid == MOVER)
    labeled, n_comp = label_mask(meta, connectivity=4)
    h, w = grid.shape
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        if not any(int(grid[r, c]) in (0, MOVER) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    expected = [reg["room"] for reg in regions]
    if len(bounded) != len(expected):
        return False
    return all(comp in expected for comp in bounded)


def _paint_regions(grid: np.ndarray, regions: list[dict], *, with_yellow: bool) -> None:
    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]
        for r, c in reg["obstacles"]:
            grid[r, c] = OBSTACLE
        sr, sc = reg["stop_cell"]
        grid[sr, sc] = STOP_OBSTACLE
        if with_yellow:
            yr, yc = reg["input_yellow"]
            grid[yr, yc] = MOVER


def _expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    expected = input_grid.copy()
    for reg in regions:
        yr, yc = reg["input_yellow"]
        expected[yr, yc] = 0
        rr, cc = reg["output_yellow"]
        expected[rr, cc] = MOVER
    return expected


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    expected = _expected_output(input_grid, regions)
    return np.array_equal(solver_out, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _build_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    structural = grid.copy()
    _paint_regions(structural, regions, with_yellow=False)
    if not _validate_topology(structural, regions):
        return None

    input_grid = structural.copy()
    _paint_regions(input_grid, regions, with_yellow=True)
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
