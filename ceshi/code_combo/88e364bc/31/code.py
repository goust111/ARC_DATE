# Combo variant of 88e364bc (index 31)
# combo_spec: D14/1+D15/0
# Name: 墙壁碰撞擦除+迭代稳定

# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and trace
#    - The source task has enclosed background rooms bounded by colored walls. A border legend with
#      the same fill color as the wall encodes one movement direction for that wall color.
#    - D14/1 adds: when a yellow tries to move into a wall, the contacted wall pixel is erased.
#    - D15/0 adds: movement is no longer "slide to completion once"; instead, each round a yellow
#      can advance by at most one cell, and rounds repeat until nothing changes.
#    - Fused rule used here:
#      round start -> every yellow checks one step in its own legend direction
#      if the step is open inside the currently carved domain, it moves one cell
#      else if the step is blocked by a wall, that single wall pixel is erased
#      else if the step is blocked by another yellow, nothing happens
#      then all moves and all erasures are applied simultaneously and the next round starts
#    - Small horizontal example:
#      input room A | room B, yellow in room A moving right
#      round 1: yellow reaches the separator wall and cannot enter, so that wall cell is erased
#      round 2: yellow steps into the new gap
#      round 3+: yellow continues through room B, possibly drilling another separator later
#      difference from D14-only:
#        D14-only would stop after the first collision, leaving just one hole.
#        The fused rule keeps iterating, so the same yellow can cross into later rooms.
#
# b) Complexity calibration
#    - The generator deliberately uses only cardinal legends. That keeps the "which wall pixel was
#      erased?" cue obvious, while the iterative behavior is already visible through repeated
#      drilling; diagonal disambiguation is unnecessary noise here.
#    - All generated rooms in the drilling structure share one wall color and one direction. This
#      avoids hidden tie-breaking when a tunnel opens between formerly separate regions.
#    - Samples still contain 2-3 rooms and 1-2 yellows, so the rule has repeated visual support
#      rather than a single isolated accident.
#
# c) Hidden assumptions audit for main()
#    - Original assumption: each yellow can be solved from its initial room in one compression pass.
#      Broken by D15/0, because intermediate erasures create new future paths.
#    - Original assumption: the passable area is exactly the initial background components.
#      Broken by D14/1, because erased wall cells become newly passable.
#    - New solver assumption introduced here:
#      a yellow keeps the direction of its initial room even after drilling into another room.
#      The generator guarantees compatible layouts by making all drill-connected rooms share one
#      legend color and direction.
#    - New passability interpretation:
#      movement is confined to the union of initially enclosed room interiors plus wall cells erased
#      later. This preserves the "inside rooms" semantics and prevents leaking into the unrelated
#      outer background after opening an exterior wall.
#
# d) Placement legality
#    - The generator places one border legend with a one-cell buffer.
#    - The room structure is a single chain of axis-aligned rectangular chambers that share separator
#      walls, so every chamber is enclosed at input time and chambers do not overlap the legend.
#    - Yellow start cells are always inside the upstream chamber on distinct lanes and never overlap.
#
# e) Visual observability
#    - Every generated sample forces at least one yellow to leave its starting room by drilling
#      through a separator. This makes the fused behavior visibly different from both the original
#      task and the D14-only variant.
#    - Metrics in `_oracle_check` reject samples unless the fused output differs from a one-pass
#      "slide once and erase once" baseline.
#
# f) Visual cleanliness
#    - The chain layout keeps walls, gaps, and movers on clean rectangular lanes. Multiple erasures
#      appear as a short aligned row or column of holes instead of scattered clutter.
#
# g) Change inferability
#    - No new color mapping is introduced. The only color change is wall -> background at collision
#      sites, and those sites line up with the eventual drilled path of the yellow movers.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and map their fill color to a direction vector.
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
        if len(others) < 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            centered = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                centered.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            centered.sort()
            center_col = centered[0][1]
            indicator_col = centered[1][1]
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


def _border_color_for_cells(
    g: np.ndarray,
    bg: int,
    mover: int,
    cells: set[tuple[int, int]],
) -> int | None:
    h, w = g.shape
    border_counts: dict[int, int] = {}
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cells:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None
    return max(border_counts, key=border_counts.get)


def _collect_initial_layout(
    g: np.ndarray,
    bg: int,
    mover: int,
    dirs: dict[int, tuple[int, int]],
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    regions: list[dict] = []
    movers_out: list[dict] = []
    domain = np.zeros((h, w), dtype=bool)
    region_ids = np.full((h, w), -1, dtype=int)

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        cells = {(int(r), int(c)) for r, c in pts}
        rid = len(regions)
        for r, c in cells:
            domain[r, c] = True
            region_ids[r, c] = rid

        border_col = _border_color_for_cells(g, bg, mover, cells)
        direction = None if border_col is None else dirs.get(border_col)
        region = {
            "id": rid,
            "cells": cells,
            "border_color": border_col,
            "dir": direction,
        }
        regions.append(region)

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if movers:
            assert direction is not None, "every mover-containing region must match a legend direction"
        for r, c in movers:
            movers_out.append(
                {
                    "pos": (r, c),
                    "dir": direction,
                    "start_region": rid,
                }
            )

    return regions, movers_out, domain, region_ids


def _cell_in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _can_step(
    passable: np.ndarray,
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> bool:
    nr, nc = r + dr, c + dc
    if not _cell_in_bounds(passable.shape, nr, nc) or not passable[nr, nc]:
        return False
    if (nr, nc) in occupied:
        return False
    if dr != 0 and dc != 0:
        side1 = (r + dr, c)
        side2 = (r, c + dc)
        if not _cell_in_bounds(passable.shape, side1[0], side1[1]) or not passable[side1]:
            return False
        if not _cell_in_bounds(passable.shape, side2[0], side2[1]) or not passable[side2]:
            return False
    return True


def _impact_wall_cell(
    state: np.ndarray,
    passable: np.ndarray,
    bg: int,
    mover: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    candidates = []
    nr, nc = r + dr, c + dc
    if not _cell_in_bounds(passable.shape, nr, nc) or not passable[nr, nc]:
        candidates.append((nr, nc))
    if dr != 0 and dc != 0:
        side1 = (r + dr, c)
        side2 = (r, c + dc)
        if not _cell_in_bounds(passable.shape, side1[0], side1[1]) or not passable[side1]:
            candidates.append(side1)
        if not _cell_in_bounds(passable.shape, side2[0], side2[1]) or not passable[side2]:
            candidates.append(side2)
    for rr, cc in candidates:
        if not _cell_in_bounds(passable.shape, rr, cc):
            continue
        v = int(state[rr, cc])
        if v != bg and v != mover:
            return (rr, cc)
    return None


def _simulate_iterative_drill(
    grid: np.ndarray,
    *,
    return_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    regions, movers, passable, region_ids = _collect_initial_layout(g, bg, mover, dirs)
    state = g.copy()
    rounds = 0
    total_moves = 0
    erased_cells: set[tuple[int, int]] = set()

    while True:
        occupied = {tuple(m["pos"]) for m in movers}
        move_targets: dict[tuple[int, int], list[int]] = {}
        erase_targets: set[tuple[int, int]] = set()

        for idx, mover_info in enumerate(movers):
            r, c = mover_info["pos"]
            dr, dc = mover_info["dir"]
            if _can_step(passable, occupied, r, c, dr, dc):
                tgt = (r + dr, c + dc)
                move_targets.setdefault(tgt, []).append(idx)
                continue
            impact = _impact_wall_cell(state, passable, bg, mover, r, c, dr, dc)
            if impact is not None:
                erase_targets.add(impact)

        approved_moves = {
            idx: tgt
            for tgt, idxs in move_targets.items()
            if len(idxs) == 1
            for idx in idxs
        }
        if not approved_moves and not erase_targets:
            break

        for r, c in erase_targets:
            state[r, c] = bg
            passable[r, c] = True
            erased_cells.add((r, c))

        for idx, (nr, nc) in approved_moves.items():
            r, c = movers[idx]["pos"]
            state[r, c] = bg
            state[nr, nc] = mover
            movers[idx]["pos"] = (nr, nc)
        total_moves += len(approved_moves)
        rounds += 1

    if not return_metrics:
        return state

    movers_outside_start = 0
    for mover_info in movers:
        r, c = mover_info["pos"]
        if region_ids[r, c] != mover_info["start_region"]:
            movers_outside_start += 1
    metrics = {
        "rounds": rounds,
        "moves": total_moves,
        "erased_cells": len(erased_cells),
        "movers_outside_start": movers_outside_start,
        "region_count": len(regions),
    }
    return state, metrics


def _slide_points_with_impacts(
    g: np.ndarray,
    bg: int,
    mover: int,
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if not points:
        return [], []
    region_mask = _set_to_mask(region_set, g.shape)
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    impact_cells = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        impact = None
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                impact = _impact_wall_cell(g, region_mask, bg, mover, r, c, dr, dc)
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    impact = _impact_wall_cell(g, region_mask, bg, mover, r, c, dr, dc)
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
        if impact is not None:
            impact_cells.append(impact)
    return out_pts, impact_cells


def _set_to_mask(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _single_pass_drill_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    out = g.copy()
    regions, _, _, _ = _collect_initial_layout(g, bg, mover, dirs)
    for region in regions:
        region_set = region["cells"]
        movers = [(r, c) for r, c in region_set if int(g[r, c]) == mover]
        if not movers or region["dir"] is None:
            continue
        dr, dc = region["dir"]
        for r, c in movers:
            out[r, c] = bg
        new_pts, impacts = _slide_points_with_impacts(g, bg, mover, region_set, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover
        for r, c in impacts:
            out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance yellow pixels through enclosed rooms according to border-legend directions, while walls can be drilled away over repeated rounds.

    Input structure:
    - Background cells form one or more enclosed rooms bounded by colored walls; yellow pixels begin inside those rooms.
    - Border-touching rectangular legends on the frame use the same fill color as the controlled wall color.
    - Within each legend, the center-mark cluster and the red indicator cluster define a direction vector.

    Rule:
    - Each yellow inherits the direction of the legend matching the wall color of its starting room.
    - Simulation proceeds in synchronous rounds. In one round, a yellow may move by only one cell in its direction.
    - If that one-step move is blocked by a wall, the contacted wall pixel is erased to background instead of moving; if blocked by another yellow, nothing changes for that yellow this round.
    - After applying all moves and erasures simultaneously, repeat until no yellow moves and no new wall cell is erased.

    Key insight: the output is not a one-shot slide; it is a drilling process. Initial rooms define each yellow's direction, while erased wall cells enlarge the future travel corridor and can connect rooms that were separate in the input.
    """
    return _simulate_iterative_drill(grid)


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

ROOM_COUNT_CHOICES = [2, 3, 3]
ROOM_LENGTH_CHOICES = [3, 4, 5]
ROOM_CROSS_CHOICES = [4, 5, 6]
PLACEMENT_RETRIES = 120
GEN_RETRIES = 300


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


def _place_legend(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, int, tuple[int, int]]:
    h, w = grid.shape
    wall_color = rng.choice(WALL_COLOR_CHOICES)
    direction = rng.choice(CARDINAL_DIRS)
    corner = rng.choice(CORNERS)
    patch = _legend_5x5(wall_color, *direction)
    ph, pw = patch.shape
    r0, c0 = _corner_origin(corner, h, w, ph, pw)
    paste(grid, patch, r0, c0)

    reserved = np.zeros((h, w), dtype=bool)
    r_lo = max(0, r0 - LEGEND_BUFFER)
    r_hi = min(h, r0 + ph + LEGEND_BUFFER)
    c_lo = max(0, c0 - LEGEND_BUFFER)
    c_hi = min(w, c0 + pw + LEGEND_BUFFER)
    reserved[r_lo:r_hi, c_lo:c_hi] = True
    return reserved, wall_color, direction


def _chain_dims(direction: tuple[int, int], lengths: list[int], cross: int) -> tuple[int, int]:
    if direction[0] == 0:
        return cross + 2, sum(lengths) + len(lengths) + 1
    return sum(lengths) + len(lengths) + 1, cross + 2


def _can_place_box(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    gh, gw = reserved.shape
    if r0 <= 0 or c0 <= 0:
        return False
    if r0 + h >= gh or c0 + w >= gw:
        return False
    return not reserved[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any()


def _paint_horizontal_chain(
    grid: np.ndarray,
    wall_color: int,
    r0: int,
    c0: int,
    cross: int,
    lengths: list[int],
) -> tuple[list[dict], set[tuple[int, int]]]:
    total_h = cross + 2
    total_w = sum(lengths) + len(lengths) + 1
    walls: set[tuple[int, int]] = set()
    rooms: list[dict] = []

    for c in range(c0, c0 + total_w):
        walls.add((r0, c))
        walls.add((r0 + total_h - 1, c))

    cursor = c0
    for width in lengths:
        left_wall = cursor
        right_wall = left_wall + width + 1
        for r in range(r0, r0 + total_h):
            walls.add((r, left_wall))
            walls.add((r, right_wall))
        interior = {
            (r, c)
            for r in range(r0 + 1, r0 + 1 + cross)
            for c in range(left_wall + 1, right_wall)
        }
        rooms.append(
            {
                "interior": interior,
                "r0": r0 + 1,
                "c0": left_wall + 1,
                "h": cross,
                "w": width,
            }
        )
        cursor = right_wall

    for r, c in walls:
        grid[r, c] = wall_color
    return rooms, walls


def _paint_vertical_chain(
    grid: np.ndarray,
    wall_color: int,
    r0: int,
    c0: int,
    cross: int,
    lengths: list[int],
) -> tuple[list[dict], set[tuple[int, int]]]:
    total_h = sum(lengths) + len(lengths) + 1
    total_w = cross + 2
    walls: set[tuple[int, int]] = set()
    rooms: list[dict] = []

    for r in range(r0, r0 + total_h):
        walls.add((r, c0))
        walls.add((r, c0 + total_w - 1))

    cursor = r0
    for length in lengths:
        top_wall = cursor
        bottom_wall = top_wall + length + 1
        for c in range(c0, c0 + total_w):
            walls.add((top_wall, c))
            walls.add((bottom_wall, c))
        interior = {
            (r, c)
            for r in range(top_wall + 1, bottom_wall)
            for c in range(c0 + 1, c0 + 1 + cross)
        }
        rooms.append(
            {
                "interior": interior,
                "r0": top_wall + 1,
                "c0": c0 + 1,
                "h": length,
                "w": cross,
            }
        )
        cursor = bottom_wall

    for r, c in walls:
        grid[r, c] = wall_color
    return rooms, walls


def _build_chain(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    wall_color: int,
    direction: tuple[int, int],
) -> list[dict] | None:
    cross = rng.choice(ROOM_CROSS_CHOICES)
    room_count = rng.choice(ROOM_COUNT_CHOICES)
    lengths = [rng.choice(ROOM_LENGTH_CHOICES) for _ in range(room_count)]
    total_h, total_w = _chain_dims(direction, lengths, cross)
    h, w = grid.shape
    if total_h > h - 2 or total_w > w - 2:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(1, h - total_h - 1)
        c0 = rng.randint(1, w - total_w - 1)
        if not _can_place_box(reserved, r0, c0, total_h, total_w):
            continue
        if direction[0] == 0:
            rooms, walls = _paint_horizontal_chain(grid, wall_color, r0, c0, cross, lengths)
        else:
            rooms, walls = _paint_vertical_chain(grid, wall_color, r0, c0, cross, lengths)
        reserved[r0 - 1 : r0 + total_h + 1, c0 - 1 : c0 + total_w + 1] = True
        for room in rooms:
            room["color"] = wall_color
            room["dir"] = direction
            room["walls"] = walls
        return rooms
    return None


def _validate_topology(grid: np.ndarray, rooms: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    expected = [room["interior"] for room in rooms]
    if len(bounded) != len(expected):
        return False
    return all(comp in expected for comp in bounded)


def _upstream_room_index(direction: tuple[int, int], n_rooms: int) -> int:
    dr, dc = direction
    return 0 if dr + dc > 0 else n_rooms - 1


def _lane_points_for_room(
    room: dict,
    direction: tuple[int, int],
    lane_offsets: list[int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    points = []
    if dr == 0:
        row = room["r0"]
        back_col = room["c0"] if dc > 0 else room["c0"] + room["w"] - 1
        depth = 1 if room["w"] >= 4 else 0
        col = back_col + depth if dc > 0 else back_col - depth
        for off in lane_offsets:
            points.append((row + off, col))
    else:
        col = room["c0"]
        back_row = room["r0"] if dr > 0 else room["r0"] + room["h"] - 1
        depth = 1 if room["h"] >= 4 else 0
        row = back_row + depth if dr > 0 else back_row - depth
        for off in lane_offsets:
            points.append((row, col + off))
    return points


def _place_yellows(
    rng: random.Random,
    rooms: list[dict],
    direction: tuple[int, int],
    grid: np.ndarray,
) -> bool:
    upstream = rooms[_upstream_room_index(direction, len(rooms))]
    lane_count = upstream["h"] if direction[0] == 0 else upstream["w"]
    if lane_count <= 0:
        return False
    n_yellows = 2 if lane_count >= 2 and rng.random() < 0.75 else 1
    n_yellows = min(n_yellows, lane_count)
    lane_offsets = sorted(rng.sample(range(lane_count), n_yellows))
    points = _lane_points_for_room(upstream, direction, lane_offsets)
    if any(grid[r, c] != 0 for r, c in points):
        return False
    for r, c in points:
        grid[r, c] = 4
    return True


def analyze_fusion_behavior(input_grid: np.ndarray) -> dict:
    fused_out, fused_metrics = _simulate_iterative_drill(input_grid, return_metrics=True)
    single_pass_out = _single_pass_drill_output(input_grid)
    return {
        "rounds": fused_metrics["rounds"],
        "moves": fused_metrics["moves"],
        "erased_cells": fused_metrics["erased_cells"],
        "movers_outside_start": fused_metrics["movers_outside_start"],
        "differs_from_single_pass_drill": not np.array_equal(fused_out, single_pass_out),
    }


def _oracle_check(input_grid: np.ndarray, rooms: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    metrics = analyze_fusion_behavior(input_grid)
    if metrics["rounds"] < 2:
        return False
    if metrics["erased_cells"] < 2:
        return False
    if metrics["movers_outside_start"] < 1:
        return False
    if not metrics["differs_from_single_pass_drill"]:
        return False

    h, w = input_grid.shape
    room_mask = np.zeros((h, w), dtype=bool)
    for room in rooms:
        for r, c in room["interior"]:
            room_mask[r, c] = True
    drilled_or_room = room_mask | ((input_grid != 0) & (input_grid != 4))
    yellow_cells = solver_out == 4
    if not bool(np.all(drilled_or_room[yellow_cells])):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    reserved, wall_color, direction = _place_legend(rng, grid)
    rooms = _build_chain(rng, grid, reserved, wall_color, direction)
    if rooms is None:
        return None
    if not _validate_topology(grid, rooms):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, rooms, direction, input_grid):
        return None
    if not _oracle_check(input_grid, rooms):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
