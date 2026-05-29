# Variant of 88e364bc
# Dimension: D21 — 背景色与可通行区域 (meta)
# Change: 背景色为0（黑色），区域内的可通行格子为背景色或黄色；墙壁像素不可通行
# ... → 区域内有不可通行的内部障碍物

"""
Design analysis for the D21 variant
===================================

variant_spec:
  preserved:
    - grid is 20x20 with black background 0
    - legends stay on grid corners and still encode a direction by center->indicator
    - legend fill color still matches the wall color of the controlled region
    - yellow pixels still end up stacked on the far side of their region in the legend direction
  changed_core:
    - each enclosed region may contain a small amount of internal obstacle color 3
    - obstacle cells are inside the region, are not passable, and are not part of the outer wall
    - a yellow no longer uses only repeated straight (dr, dc) steps; it may route through the
      passable subgraph to reach the farthest reachable resting cell in the encoded direction

analysis:
  a_specific_behavior:
    - Concrete look: a colored outer wall encloses a room; inside the room there are black floor
      cells, one yellow pixel, and a few green obstacle pixels laid out as thin bars with gaps.
      The legend on the border still tells the intended overall direction.
    - Original execution at the critical step:
      original_main() treated every enclosed 0/4 component as one region and moved each yellow by
      repeatedly stepping exactly (dr, dc) until blocked by the wall or by another settled yellow.
    - Variant execution at the critical step:
      new main() treats the enclosed room as the union of passable cells plus obstacle cells,
      recovers the passable floor by removing obstacle color 3, then for each yellow runs a BFS on
      the passable cells and chooses the reachable cell with maximal projection onto (dr, dc).
      Ties are broken toward the shortest route and the smallest sideways deviation, so obstacle-
      free cases still behave like the original straight slide.
    - Small example:
      if the legend says "right", a vertical green bar inside the room blocks the row of the yellow.
      Original code would stop at the left side of that bar. The variant instead routes upward or
      downward through the visible gap / around the bar and still finishes against the right wall.

  b_complexity_calibration:
    - Only one new semantic color is introduced: obstacle color 3. Wall colors explicitly avoid 3,
      so the added color has a single stable meaning across all samples.
    - Legends remain cardinal in generated samples. This is a deliberate calibration choice: the
      path-planning behavior is the new difficulty, and keeping directions cardinal makes the
      obstacle effect visually legible in 2-3 examples instead of mixing in diagonal-routing cases.
    - Obstacles are thin bars with 1-2 gaps / staggered openings. They are simple enough to read,
      but still force nontrivial turns and prevent the variant from degenerating into straight slide.

  c_hidden_assumption_audit_of_original_main:
    - `passable = (g == bg) | (g == mover)` assumed the entire enclosed room was exactly the
      passable component. This fails once obstacle cells sit inside the room, because the room would
      be split into several passable components and the solver would misidentify the region.
    - `components_mask(passable, connectivity=4)` assumed region identity and movement state space
      were the same object. In the variant they diverge: region identity includes obstacle cells,
      movement state space excludes them.
    - `_slide_points()` assumed the legal move relation was "keep stepping in exactly one vector".
      That is harmless in the original task, but wrong here because a yellow may need lateral steps
      to bypass an obstacle before continuing toward the far side.
    - The border-color counting loop assumed that every non-bg / non-yellow cell touching the
      component was part of the outer wall. In the variant we must exclude obstacle color 3.

  d_placement_legality:
    - Regions are placed as closed rectangular rooms with a 1-cell exterior buffer so rooms and
      legends do not overlap or touch ambiguously.
    - Obstacle bars are placed strictly inside the room, never on the outer wall, and the generator
      verifies that the remaining passable floor stays 4-connected.
    - Each room is guaranteed to have at least one yellow start whose chosen shortest route contains
      an off-axis step, so the new rule is actually exercised.

  e_visual_observability:
    - A human can see: legend points right/up/left/down; the yellow is on the near side of a green
      blocker; the output relocates it to the far wall despite the blocker. That visual evidence is
      enough to infer "move through the room toward that side while avoiding green obstacles".
    - The generator rejects starts whose route is still a straight line, so samples do not silently
      regress to the original behavior.

  f_visual_cleanliness:
    - Obstacle color is unique and sparse.
    - Rooms do not overlap and keep a buffer, so legends, walls, obstacles, and yellows remain easy
      to separate by eye.
    - Using rectangular rooms here is intentional: it reduces clutter so the D21 change is the
      salient cue rather than irregular wall geometry.

  g_color_inference:
    - Color 3 always means "internal obstacle" and never appears as a wall color.
    - Wall colors still only encode legend-to-region matching and do not change meaning.

  new_solver_assumptions_and_generator_guarantees:
    - The solver assumes obstacle color 3 is distinct from all wall colors; generator enforces wall
      colors in {5, 6, 7, 8, 9}.
    - The solver assumes each enclosed room is one bounded component in the meta-mask
      {background, yellow, obstacle}; generator builds exactly that.
    - The solver assumes there is at least one legend; generator always places 1-2 corner legends.
    - The solver uses cardinal routing in generated samples; generator only emits cardinal legends.
"""

from collections import deque
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


OBSTACLE = 3
MOVER = 4


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend blocks on the border.

    A legend is a border-touching non-background component whose most common color is the legend
    background `C` (and also the region border color it controls). The legend contains at least
    two additional colors: a "center" color (more frequent) and an "indicator" color (less frequent).

    The movement direction for regions with border color `C` is:
      sign(centroid(indicator) - centroid(center))
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

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
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

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

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


def _best_destination_and_path(
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
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for start in points_sorted:
        dest, _ = _best_destination_and_path(passable, start, occupied, dr, dc)
        occupied.add(dest)
        out_pts.append(dest)
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions in the direction dictated by border legends.

    Variant rule:
    - A room interior may contain obstacle color 3 in addition to black floor and yellow movers.
    - Obstacles are not passable.
    - A yellow seeks the farthest reachable cell in the legend direction within its room, routing
      around obstacles if needed. Multiple yellows still stack because already-settled destinations
      are treated as blockers for later yellows in the same room.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

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

        new_pts = _path_planned_slide(passable, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = MOVER

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

ROOM_BUFFER = 1
MAX_TOTAL_REGIONS = 3
ROOM_RETRIES = 80
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


def _build_obstacles(
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
    obstacles: set[tuple[int, int]] = set()

    if dr == 0:
        candidate_cols = list(range(c0 + 2, c1 - 1))
        if not candidate_cols:
            return None
        n_bars = 1 if len(candidate_cols) < 4 else rng.choice([1, 2])
        if len(candidate_cols) < n_bars:
            return None
        bar_cols = sorted(rng.sample(candidate_cols, n_bars))
        for idx, bar_c in enumerate(bar_cols):
            bar_rows = list(range(r0 + 2, r1 - 1))
            if len(bar_rows) < 2:
                return None
            if idx % 2 == 0:
                gap_choices = bar_rows[: max(1, len(bar_rows) // 2)]
            else:
                gap_choices = bar_rows[len(bar_rows) // 2 :]
            gap_r = rng.choice(gap_choices)
            for r in bar_rows:
                if r != gap_r:
                    obstacles.add((r, bar_c))
    else:
        candidate_rows = list(range(r0 + 2, r1 - 1))
        if not candidate_rows:
            return None
        n_bars = 1 if len(candidate_rows) < 4 else rng.choice([1, 2])
        if len(candidate_rows) < n_bars:
            return None
        bar_rows = sorted(rng.sample(candidate_rows, n_bars))
        for idx, bar_r in enumerate(bar_rows):
            bar_cols = list(range(c0 + 2, c1 - 1))
            if len(bar_cols) < 2:
                return None
            if idx % 2 == 0:
                gap_choices = bar_cols[: max(1, len(bar_cols) // 2)]
            else:
                gap_choices = bar_cols[len(bar_cols) // 2 :]
            gap_c = rng.choice(gap_choices)
            for c in bar_cols:
                if c != gap_c:
                    obstacles.add((bar_r, c))

    passable = interior - obstacles
    if not passable:
        return None
    return passable, obstacles


def _connected(mask_cells: set[tuple[int, int]]) -> bool:
    if not mask_cells:
        return False
    start = next(iter(mask_cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in mask_cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(mask_cells)


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
        if total_regions == 1:
            h_in = rng.randint(6, 8)
            w_in = rng.randint(7, 9) if dr == 0 else rng.randint(6, 8)
        elif total_regions == 2:
            h_in = rng.randint(6, 7)
            w_in = rng.randint(6, 8) if dr == 0 else rng.randint(6, 7)
        else:
            h_in = rng.randint(5, 6)
            w_in = rng.randint(6, 7) if dr == 0 else rng.randint(5, 6)

        if dc == 0:
            h_in, w_in = w_in, h_in

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

        built = _build_obstacles(r0, c0, h_out, w_out, dr, dc, rng)
        if built is None:
            continue
        passable, obstacles = built
        if not _connected(passable):
            continue

        walls = _room_wall_cells(r0, c0, h_out, w_out)
        return {
            "color": wall_color,
            "dir": (dr, dc),
            "walls": walls,
            "passable": passable,
            "obstacles": obstacles,
            "room": passable | obstacles,
            "bbox": (r0, c0, r0 + h_out - 1, c0 + w_out - 1),
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
    if len(region_colors) < MAX_TOTAL_REGIONS and rng.random() < 0.55:
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


def _path_has_detour(path: list[tuple[int, int]], dr: int, dc: int) -> bool:
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        if (r1 - r0, c1 - c0) != (dr, dc):
            return True
    return False


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        passable = reg["passable"]
        dr, dc = reg["dir"]
        candidates = []
        for start in passable:
            dest, path = _best_destination_and_path(passable, start, set(), dr, dc)
            if dest == start:
                continue
            if _projection(dest, dr, dc) <= _projection(start, dr, dc):
                continue
            if not _path_has_detour(path, dr, dc):
                continue
            candidates.append((len(path), -_projection(start, dr, dc), start))
        if not candidates:
            return False
        candidates.sort(reverse=True)
        top = candidates[: min(6, len(candidates))]
        start = rng.choice(top)[2]
        grid[start] = MOVER
    return True


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    meta = (grid == 0) | (grid == OBSTACLE)
    labeled, n_comp = label_mask(meta, connectivity=4)
    bounded = []
    h, w = grid.shape
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
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
    immovable = (input_grid != 0) & (input_grid != MOVER)
    if not np.array_equal(solver_out[immovable], input_grid[immovable]):
        return False
    region_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["passable"]:
            region_mask[r, c] = True
    return bool(np.all(region_mask[solver_out == MOVER]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _build_regions(rng, legends, legend_occupied)
    if regions is None:
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
