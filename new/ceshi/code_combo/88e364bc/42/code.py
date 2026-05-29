# Combo variant of 88e364bc (index 42)
# combo_spec: D16/0+D21/1
# Name: 允许角切割+内部障碍物

"""
Design analysis for the combo variant
====================================

combo_spec:
  preserved:
    - grid stays 20x20 with background 0
    - border legends still sit on corners and encode a direction by center->indicator centroid
    - legend fill color still matches the wall color of the region it controls
    - yellow pixels still move toward the extreme side of their room and stack there
  changed_core:
    - rooms contain internal obstacle color 3, which is part of the room identity but not passable
    - movement is now path-based inside the room rather than repeated one-vector stepping
    - diagonal steps explicitly allow corner cutting: if the diagonal destination is passable, the
      two orthogonal side cells are not checked, so a yellow may slip through a diagonal pinch
      formed by obstacle corners

analysis:
  a_specific_behavior:
    - Concrete look: a rectangular room with colored outer wall contains black floor, green obstacle
      bars, and one yellow. The obstacle layout creates two chambers separated by a zig-zag barrier.
      The chambers touch only at one diagonal pinch, e.g. cells (r,c) and (r+1,c+1) are floor but
      (r,c+1) and (r+1,c) are obstacles.
    - Original logic at the changed step:
      original_main() treated the room as one 0/4 component and only repeated the legend vector.
      For a diagonal legend it also forbade corner cutting, so that diagonal pinch was impassable.
      With obstacles present, it would either split the room incorrectly or strand the yellow on
      the near side of the barrier.
    - Combo logic at the changed step:
      new main() first finds the whole room as one bounded component of {background, yellow,
      obstacle}. Inside that room it runs a BFS over passable cells with 8-neighbor motion biased
      toward the legend direction. For diagonal steps it allows moving straight into the diagonal
      target even when the orthogonal side cells are obstacles. The yellow therefore reaches the
      far chamber and then continues to the wall-most reachable destination.
    - Small example:
      if the legend points down-right, a yellow in the north-west chamber can only reach the
      south-east chamber by crossing one obstacle-corner pinch. Without obstacles there would be no
      detour problem; without corner cutting that pinch would remain sealed. The fused rule is what
      makes the path possible.

  b_complexity_calibration:
    - Only one extra semantic color is introduced: green 3 always means internal obstacle.
    - Generated legends are diagonal-only. This is deliberate: the fused behavior depends on a
      diagonal slip through obstacle corners, so cardinal cases would hide the D16 contribution.
    - Rooms are simple rectangles while obstacle barriers vary by orientation and gate position.
      That keeps the scene readable while still giving repeated evidence that the diagonal pinch is
      the critical structure.

  c_hidden_assumption_audit_of_original_main:
    - `passable = (g == bg) | (g == mover)` no longer identifies the room, because the room may be
      split into several passable islands by obstacle color 3.
    - `components_mask(passable, connectivity=4)` assumed region identity and motion state space
      were the same object. Here region identity is the bounded component of {0,3,4}, while motion
      state space is only {0,4}.
    - `_slide_points()` assumed every legal move is the repeated legend vector. That fails twice:
      obstacle layouts require routing, and the diagonal gate is legal only if corner cutting is
      allowed.
    - The border-color scan assumed every non-bg/non-yellow neighbor was an outer wall. In the
      combo variant it must ignore obstacle color 3.
    - The original diagonal check assumed both orthogonal intermediates must be inside the room.
      The D16 change explicitly invalidates that assumption.

  d_placement_legality:
    - Rooms are buffered away from each other and from legends.
    - Obstacles are placed strictly inside rooms and never overwrite walls or legends.
    - Each obstacle barrier spans the room interior top-to-bottom and leaves exactly one diagonal
      gate, so chambers do not reconnect by accidental side gaps.
    - Yellow starts are selected only from cells whose combo-path improves the reachable projection
      compared with the same motion graph under no-corner-cut rules.

  e_visual_observability:
    - A human can see the legend direction, see the obstacle barrier, and see that the input yellow
      begins on the wrong side of that barrier yet the output yellow appears on the far diagonal
      side. The diagonal pinch is visible in the input itself.
    - The generator rejects starts unless the chosen path contains at least one real corner-cut
      diagonal step and the no-corner-cut solver would stop closer to the start side.

  f_visual_cleanliness:
    - Walls, obstacles, legends, and yellow do not overlap and use distinct roles.
    - Rectangular rooms reduce clutter so the informative pattern is the obstacle gate, not noisy
      room geometry.
    - At most two rooms are generated, which avoids overcrowding around the 6x7 diagonal legends.

  g_color_inference:
    - Color 3 has one stable meaning across all samples: internal obstacle.
    - Wall colors still only provide legend-to-room matching; they do not encode movement by value.

  new_solver_assumptions_and_generator_guarantees:
    - The solver assumes obstacle color 3 never doubles as a wall color; generator uses wall colors
      only from {5,6,7,8,9}.
    - The solver assumes each room is a bounded 4-connected component in the meta-mask {0,3,4};
      generator builds rooms exactly that way.
    - The solver assumes legends exist and point diagonally; generator always places 1-2 diagonal
      legends.
    - The solver assumes enough free space exists to route to a deterministic best destination;
      generator only keeps rooms with validated candidate starts.
"""

from collections import deque
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


OBSTACLE = 3
MOVER = 4

GRID_H, GRID_W = 20, 20
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
ROOM_BUFFER = 1
MAX_ROOMS = 2
ROOM_RETRIES = 120
GEN_RETRIES = 400


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
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

        fill = max(counts, key=counts.get)
        others = sorted((cnt, col) for col, cnt in counts.items() if col != fill)
        if len(others) != 2:
            continue
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tmp = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tmp.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            tmp.sort()
            center_col = tmp[0][1]
            indicator_col = tmp[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue
        vr, vc = (indicator.mean(axis=0) - center.mean(axis=0)).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if (dr, dc) not in DIAGONAL_DIRS:
            continue

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _perp_offset(start: tuple[int, int], end: tuple[int, int], dr: int, dc: int) -> int:
    sr, sc = start
    er, ec = end
    return abs((er - sr) * dc - (ec - sc) * dr)


def _step_order(dr: int, dc: int) -> list[tuple[int, int]]:
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


def _is_corner_cut_step(
    passable: set[tuple[int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if abs(dr) != 1 or abs(dc) != 1:
        return False
    return (a[0] + dr, a[1]) not in passable or (a[0], a[1] + dc) not in passable


def _neighbors_for_motion(
    passable: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    pos: tuple[int, int],
    dr: int,
    dc: int,
    allow_corner_cut: bool,
) -> list[tuple[int, int]]:
    r, c = pos
    out = []
    for sdr, sdc in _step_order(dr, dc):
        nxt = (r + sdr, c + sdc)
        if nxt not in passable or nxt in blocked:
            continue
        if abs(sdr) == 1 and abs(sdc) == 1 and not allow_corner_cut:
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
    allow_corner_cut: bool,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    dist = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _neighbors_for_motion(passable, blocked, cur, dr, dc, allow_corner_cut):
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            parent[nxt] = cur
            q.append(nxt)
    return dist, parent


def _best_destination_and_path(
    passable: set[tuple[int, int]],
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
    dr: int,
    dc: int,
    allow_corner_cut: bool,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    dist, parent = _reachable_info(passable, start, blocked, dr, dc, allow_corner_cut)
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
    path = [dest]
    while path[-1] != start:
        prev = parent[path[-1]]
        assert prev is not None
        path.append(prev)
    path.reverse()
    return dest, path


def _path_planned_slide(
    passable: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
    allow_corner_cut: bool,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for start in points_sorted:
        dest, _ = _best_destination_and_path(passable, start, occupied, dr, dc, allow_corner_cut)
        occupied.add(dest)
        out_pts.append(dest)
    return out_pts


def _path_has_corner_cut(passable: set[tuple[int, int]], path: list[tuple[int, int]]) -> bool:
    return any(_is_corner_cut_step(passable, a, b) for a, b in zip(path, path[1:]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside colored rooms by the direction encoded in corner legends.

    Input structure:
    - Black background surrounds one or two enclosed rooms bounded by colored walls.
    - Each room interior may contain black floor, yellow movers, and green internal obstacles.
    - Border legends are diagonal 6x7 rectangles whose fill color matches the wall color of the
      room they control.

    Rule:
    - For each bounded room, recover the full room from the meta-mask {background, yellow,
      obstacle}, then recover the passable floor by removing obstacle cells.
    - Use the legend matching that room's wall color to get a diagonal direction.
    - Each yellow moves to the reachable cell with maximal projection in that direction, routing
      through the room and stacking with previously settled yellows. Diagonal steps may cut corners:
      only the destination cell must be passable.

    Key insight:
    - The obstacle barrier can separate the room into chambers that touch only at a diagonal pinch.
      Matching wall color to legend gives the intended diagonal, and allowing diagonal corner cuts
      is exactly what lets the yellow escape through that pinch to the far side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one diagonal border legend"

    room_mask = (g == bg) | (g == MOVER) | (g == OBSTACLE)
    out = g.copy()

    for pts in components_mask(room_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        room = {(int(r), int(c)) for r, c in pts}
        passable = {p for p in room if int(g[p]) in (bg, MOVER)}
        movers = [p for p in passable if int(g[p]) == MOVER]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        for r, c in room:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in room:
                    continue
                v = int(g[nr, nc])
                if v in (bg, MOVER, OBSTACLE):
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

        new_pts = _path_planned_slide(passable, movers, dr, dc, allow_corner_cut=True)
        for r, c in new_pts:
            out[r, c] = MOVER

    return out


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


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, color in zip(corners, wall_colors):
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
        legends.append((color, (dr, dc)))
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


def _build_corner_gate_obstacles(
    r0: int,
    c0: int,
    h_out: int,
    w_out: int,
    dr: int,
    dc: int,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    r1 = r0 + h_out - 1
    c1 = c0 + w_out - 1
    interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}

    split_cols = list(range(c0 + 2, c1 - 2))
    gate_rows = list(range(r0 + 2, r1 - 2))
    if not split_cols or not gate_rows:
        return None
    split_c = rng.choice(split_cols)
    gate_r = rng.choice(gate_rows)

    obstacles: set[tuple[int, int]] = set()
    if dr == dc:
        for r in range(r0 + 1, gate_r + 1):
            obstacles.add((r, split_c + 1))
        for r in range(gate_r + 1, r1):
            obstacles.add((r, split_c))
    else:
        for r in range(r0 + 1, gate_r + 1):
            obstacles.add((r, split_c))
        for r in range(gate_r + 1, r1):
            obstacles.add((r, split_c + 1))

    passable = interior - obstacles
    if not passable:
        return None
    return passable, obstacles


def _sample_room(
    rng: random.Random,
    occupied: np.ndarray,
    wall_color: int,
    direction: tuple[int, int],
    n_rooms: int,
) -> dict | None:
    h, w = occupied.shape
    dr, dc = direction

    for _ in range(ROOM_RETRIES):
        if n_rooms == 1:
            h_in = rng.randint(8, 10)
            w_in = rng.randint(8, 10)
        else:
            h_in = rng.randint(7, 8)
            w_in = rng.randint(7, 9)
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

        built = _build_corner_gate_obstacles(r0, c0, h_out, w_out, dr, dc, rng)
        if built is None:
            continue
        passable, obstacles = built
        walls = _room_wall_cells(r0, c0, h_out, w_out)
        room = passable | obstacles
        return {
            "color": wall_color,
            "dir": (dr, dc),
            "walls": walls,
            "passable": passable,
            "obstacles": obstacles,
            "room": room,
            "reserve": (rr0, rr1, cc0, cc1),
        }
    return None


def _build_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    occupied = legend_occupied.copy()
    regions = []
    n_rooms = min(MAX_ROOMS, len(legends))
    chosen_legends = legends[:]
    rng.shuffle(chosen_legends)
    chosen_legends = chosen_legends[:n_rooms]

    for color, direction in chosen_legends:
        reg = _sample_room(rng, occupied, color, direction, n_rooms)
        if reg is None:
            return None
        rr0, rr1, cc0, cc1 = reg["reserve"]
        occupied[rr0:rr1, cc0:cc1] = True
        regions.append(reg)
    return regions


def _region_metrics(reg: dict, blocked: set[tuple[int, int]] | None = None) -> dict:
    passable = reg["passable"]
    dr, dc = reg["dir"]
    blocked = set() if blocked is None else set(blocked)
    improvements = []
    candidate_info = []

    for start in passable:
        if start in blocked:
            continue
        dest_combo, path_combo = _best_destination_and_path(passable, start, blocked, dr, dc, True)
        dest_plain, _ = _best_destination_and_path(passable, start, blocked, dr, dc, False)
        gain = _projection(dest_combo, dr, dc) - _projection(dest_plain, dr, dc)
        if (
            dest_combo != start
            and _projection(dest_combo, dr, dc) > _projection(start, dr, dc)
            and gain > 0
            and _path_has_corner_cut(passable, path_combo)
        ):
            improvements.append(gain)
            candidate_info.append(
                {
                    "start": start,
                    "dest_combo": dest_combo,
                    "dest_plain": dest_plain,
                    "path_combo": path_combo,
                    "gain": gain,
                    "path_len": len(path_combo),
                }
            )

    candidate_info.sort(
        key=lambda item: (
            item["gain"],
            item["path_len"],
            -_projection(item["start"], dr, dc),
            -item["start"][0],
            -item["start"][1],
        ),
        reverse=True,
    )
    return {
        "candidate_count": len(candidate_info),
        "max_gain": max(improvements) if improvements else 0,
        "candidates": candidate_info,
    }


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        metrics = _region_metrics(reg)
        if not metrics["candidates"]:
            return False
        top = metrics["candidates"][: min(6, len(metrics["candidates"]))]
        chosen = rng.choice(top)
        r, c = chosen["start"]
        grid[r, c] = MOVER
        reg["chosen"] = chosen
    return True


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    meta = (grid == 0) | (grid == OBSTACLE)
    bounded = []
    h, w = grid.shape
    for pts in components_mask(meta, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        bounded.append({(int(r), int(c)) for r, c in pts})
    expected = [reg["room"] for reg in regions]
    if len(bounded) != len(expected):
        return False
    return all(comp in expected for comp in bounded)


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    if (solver_out == MOVER).sum() != (input_grid == MOVER).sum():
        return False
    fixed_mask = (input_grid != 0) & (input_grid != MOVER)
    if not np.array_equal(solver_out[fixed_mask], input_grid[fixed_mask]):
        return False
    for reg in regions:
        chosen = reg.get("chosen")
        if chosen is None:
            return False
        if solver_out[chosen["dest_combo"]] != MOVER:
            return False
        if chosen["dest_combo"] == chosen["dest_plain"]:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _build_regions(rng, legends, legend_occupied)
    if regions is None or not regions:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]
        for r, c in reg["obstacles"]:
            grid[r, c] = OBSTACLE

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
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
