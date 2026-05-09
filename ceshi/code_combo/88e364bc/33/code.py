# Combo variant of 88e364bc (index 33)
# combo_spec: D6/0+D18/0
# Name: 小网格+规则矩形区域

"""
Design analysis for the fused variant.

a) Concrete behavior trace
   Original task: a 20x20 board contains 1-2 border legends and several irregular fenced rooms.
   Yellow pixels move inside each enclosed room according to the direction encoded by the legend
   whose fill color matches that room's wall color.

   Fused variant: the same decode -> match -> slide chain is kept, but the canvas is reduced to
   10x10-12x12 and every enclosed room is now a strict axis-aligned rectangle with 1-cell-thick
   walls. Example: in an 11x11 grid, a compact 4x4 corner legend with fill color 7 can point
   right, and
   two separate orange rectangular rooms elsewhere on the board each contain 1-2 yellow pixels.
   The output is still obtained by sliding yellows to the far wall in the decoded direction, but
   the room-recognition step is visually simpler because each room is an obvious rectangle.

b) Complexity calibration
   The task must stay recognizable as a nontrivial legend-controlled slide puzzle. To avoid
   over-simplifying into a single isolated yellow, the generator uses 1-2 rooms per sample and
   often places two rooms that share the same controlling legend color/direction. This repeated
   support helps a human infer the rule from few examples.

c) Implicit assumptions in main()
   - `get_bg()` must still return 0: the generator leaves enough black background even on 10x10.
   - Border legends must remain isolated filled rectangles with exactly three colors: enforced.
   - Enclosed passable components must not touch the outer border: rectangular rooms are placed
     strictly inside the grid.
   - Each room must have one dominant border color that maps to a legend color: each room uses a
     single wall color identical to the legend fill.
   - If multiple yellows share a room, solver order matters: reverse construction respects the
     far-to-near processing used by `_slide_points()`.

d) Placement legality
   Rooms are placed by rectangle enumeration instead of random blob growth. Each room reserves its
   wall ring plus a one-cell moat, so rooms never overlap, never touch the legend, and never leak
   into each other. This is more reliable than irregular growth in a compact grid.

e) Visual observability
   The fused behavior should be obvious from input/output pairs: small board, compact corner
   legend, rectangular rooms, and yellows sliding toward one side of each rectangle. No hidden
   ordering uses color ids or any non-visual attribute.

f) Cleanliness and inference
   Only one legend is generated because diagonal 6x7 legends would crowd a 10x10-12x12 grid.
   A compact 4x4 cardinal legend leaves enough space for the rooms while preserving the same
   color-match and direction-decoding logic.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and decode their directions.

    The dominant color of each legend rectangle is both the legend fill and the matching room-wall
    color. The direction is `sign(centroid(indicator) - centroid(center))`.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        others.sort()
        if others[0][0] == others[1][0]:
            center_rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            options = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                dist2 = float(np.sum((pts_col.mean(axis=0) - center_rc) ** 2))
                options.append((dist2, col))
            options.sort()
            center_col = options[0][1]
            indicator_col = options[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indicator = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indicator:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indicator, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _slide_points(region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside compact rectangular rooms according to border legends.

    Input structure:
    - The board is a small 10x10-12x12 grid with black background.
    - One filled legend rectangle sits on a corner of the outer border; its dominant fill color
      matches the wall color of 1-2 interior rectangular rooms.
    - Each room is a clean axis-aligned rectangle made from a 1-cell-thick colored wall, and the
      yellow pixels lie strictly inside those rooms.

    Rule:
    - Decode the movement direction from the legend by comparing the center-colored pixels with the
      red indicator pixels.
    - For every enclosed room, find the legend whose fill color matches that room's wall color.
    - Move all yellow pixels inside that room as far as possible in the decoded direction so they
      stack against the far side of the rectangle or against earlier-stopped yellows.

    Key insight:
    - The fusion changes the geometry, not the reasoning chain: the room is now an obvious
      rectangle, so the nontrivial step is still matching wall color to legend color and reading
      the arrow-like offset inside the legend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "grid must contain at least one valid border legend"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in region_set:
                    continue
                v = int(g[nr, nc])
                if v in (bg, mover):
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1

        if not border_counts:
            continue
        wall_col = max(border_counts, key=border_counts.get)
        if wall_col not in dirs:
            continue
        dr, dc = dirs[wall_col]

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(region_set, movers, dr, dc):
            out[r, c] = mover

    return out


GRID_SIZES = [10, 11, 12]
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
ROOM_INTERIORS_BY_DIR = {
    (0, 1): [(1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4)],
    (0, -1): [(1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4)],
    (1, 0): [(5, 1), (6, 1), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3)],
    (-1, 0): [(5, 1), (6, 1), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3)],
}
YELLOWS_PER_ROOM = [1, 2]
LEGEND_BUFFER = 1
MAX_SLIDE_STEPS = 4
GEN_RETRIES = 400


def _legend_4x4(fill: int, dr: int, dc: int) -> np.ndarray:
    """4x4 legend for the four cardinal directions."""
    g = np.full((4, 4), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[2, 1] = 1
    g[2, 2] = 1
    g[1, 3] = 2
    g[2, 3] = 2
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


def _room_from_bbox(r0: int, c0: int, ih: int, iw: int) -> dict:
    """Return wall/interior cell sets for a rectangular room."""
    oh, ow = ih + 2, iw + 2
    r1, c1 = r0 + oh - 1, c0 + ow - 1
    wall: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                wall.add((r, c))
    interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    return {"bbox": (r0, c0, r1, c1), "walls": wall, "interior": interior}


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_legend(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int]]:
    h, w = grid.shape
    wall_color = rng.choice(WALL_COLOR_CHOICES)
    direction = rng.choice(CARDINAL_DIRS)
    corner = rng.choice(CORNERS)
    patch = _legend_4x4(wall_color, *direction)
    r0, c0 = _corner_origin(corner, h, w, *patch.shape)
    paste(grid, patch, r0, c0)
    occupied = np.zeros((h, w), dtype=bool)
    _reserve_bbox(occupied, (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1), LEGEND_BUFFER)
    return occupied, wall_color, direction


def _room_candidates(occupied: np.ndarray, shapes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    h, w = occupied.shape
    candidates = []
    for ih, iw in shapes:
        oh, ow = ih + 2, iw + 2
        for r0 in range(1, h - oh):
            for c0 in range(1, w - ow):
                r1, c1 = r0 + oh - 1, c0 + ow - 1
                rr0 = r0 - 1
                cc0 = c0 - 1
                rr1 = r1 + 1
                cc1 = c1 + 1
                if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append((r0, c0, ih, iw))
    return candidates


def _place_rooms(rng: random.Random, occupied: np.ndarray, wall_color: int, direction: tuple[int, int]) -> list[dict] | None:
    h, _ = occupied.shape
    count_options = [2, 1] if h >= 11 else [1]
    shapes_all = ROOM_INTERIORS_BY_DIR[direction]
    for target_count in count_options:
        attempts = 60 if target_count == 2 else 40
        for _ in range(attempts):
            occ = occupied.copy()
            rooms = []
            shape_pool = [s for s in shapes_all if s[0] * s[1] <= 9] if target_count == 2 else shapes_all[:]
            for room_idx in range(target_count):
                shapes = shape_pool[:]
                rng.shuffle(shapes)
                if target_count == 1 and room_idx == 0:
                    shapes.sort(key=lambda s: s[0] * s[1], reverse=True)
                candidates = _room_candidates(occ, shapes)
                if not candidates:
                    break
                r0, c0, ih, iw = candidates[rng.randrange(len(candidates))]
                room = _room_from_bbox(r0, c0, ih, iw)
                room["color"] = wall_color
                room["dir"] = direction
                rooms.append(room)
                _reserve_bbox(occ, room["bbox"], 1)
            if len(rooms) == target_count:
                return rooms
    return None


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    return (nr, nc) in region_set


def _forward_slide(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int,
                   blocked: set[tuple[int, int]]) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _reverse_starts(region_set: set[tuple[int, int]], out_pos: tuple[int, int], dr: int, dc: int,
                    blocked: set[tuple[int, int]], used_inputs: set[tuple[int, int]]) -> list[tuple[int, int]]:
    starts = []
    for steps in range(1, MAX_SLIDE_STEPS + 1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in region_set or (pr, pc) in blocked or (pr, pc) in used_inputs:
                ok = False
                break
            r, c = pr, pc
        if ok and _forward_slide(region_set, (r, c), dr, dc, blocked) == out_pos:
            starts.append((r, c))
    starts.sort(key=lambda p: dr * p[0] + dc * p[1])
    return starts


def _pick_output_positions(rng: random.Random, region_set: set[tuple[int, int]], dr: int, dc: int,
                           n_yellows: int) -> list[tuple[int, int]] | None:
    ordered = sorted(region_set, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    if not ordered:
        return None

    outputs = []
    blocked: set[tuple[int, int]] = set()
    used_inputs: set[tuple[int, int]] = set()
    far_key = dr * ordered[0][0] + dc * ordered[0][1]

    for k in range(n_yellows):
        if k == 0:
            pool = ordered
        else:
            near_far = [p for p in ordered if dr * p[0] + dc * p[1] >= far_key - 1 and p not in outputs]
            pool = near_far if near_far else [p for p in ordered if p not in outputs]
        chosen = None
        for out_pos in pool:
            starts = _reverse_starts(region_set, out_pos, dr, dc, blocked, used_inputs)
            if starts:
                chosen = (out_pos, starts)
                break
        if chosen is None:
            return None
        out_pos, starts = chosen
        start = rng.choice(starts)
        outputs.append(out_pos)
        blocked.add(out_pos)
        used_inputs.add(start)
    return outputs


def _place_yellows(rng: random.Random, rooms: list[dict], input_grid: np.ndarray) -> np.ndarray | None:
    expected = input_grid.copy()
    for room in rooms:
        region_set = room["interior"]
        dr, dc = room["dir"]
        n_yellows = rng.choice(YELLOWS_PER_ROOM)
        outputs = _pick_output_positions(rng, region_set, dr, dc, n_yellows)
        if not outputs:
            return None

        blocked: set[tuple[int, int]] = set()
        used_inputs: set[tuple[int, int]] = set()
        for out_pos in sorted(outputs, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True):
            starts = _reverse_starts(region_set, out_pos, dr, dc, blocked, used_inputs)
            if not starts:
                return None
            start = rng.choice(starts)
            input_grid[start] = 4
            expected[out_pos] = 4
            blocked.add(out_pos)
            used_inputs.add(start)
        room["outputs"] = set(outputs)
    return expected


def _validate_topology(struct_grid: np.ndarray, rooms: list[dict]) -> bool:
    h, w = struct_grid.shape
    labeled, n_comp = label_mask(struct_grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    expected = [room["interior"] for room in rooms]
    return len(bounded) == len(expected) and all(comp in expected for comp in bounded)


def _oracle_check(input_grid: np.ndarray, expected_output: np.ndarray) -> bool:
    try:
        solver_output = main(input_grid)
    except Exception:
        return False
    return np.array_equal(solver_output, expected_output)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = w = rng.choice(GRID_SIZES)
    struct_grid = np.zeros((h, w), dtype=int)

    occupied, wall_color, direction = _place_legend(rng, struct_grid)
    rooms = _place_rooms(rng, occupied, wall_color, direction)
    if not rooms:
        return None

    for room in rooms:
        for r, c in room["walls"]:
            struct_grid[r, c] = wall_color

    if get_bg(struct_grid) != 0:
        return None
    if not _validate_topology(struct_grid, rooms):
        return None

    input_grid = struct_grid.copy()
    expected_output = _place_yellows(rng, rooms, input_grid)
    if expected_output is None:
        return None
    if np.array_equal(input_grid, expected_output):
        return None
    if not _oracle_check(input_grid, expected_output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries for compact rectangular variant")
