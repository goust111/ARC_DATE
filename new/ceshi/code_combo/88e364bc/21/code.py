# Combo variant of 88e364bc (index 21)
# combo_spec: D3/1+D21/1
# Name: 内部障碍物+沿墙滑动

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
The original task had one enclosed room per region, a direction legend, and straight sliding
through open floor. This combo changes both the motion model and the room interior:
  - the legend still matches a wall color, but now it encodes clockwise vs counterclockwise
  - each room contains an internal obstacle core
  - yellow pixels do not travel through the whole floor; they stay on the 1-cell boundary shell
    between the outer wall and the obstacle core

Concrete small trace:
  - Original: a yellow inside a room with a right-pointing legend keeps stepping east until it
    hits the far wall or another settled yellow.
  - Combo: the same wall color now matches a 5x5 legend whose red dots sit either on the right
    side of the blue center block (clockwise) or the left side (counterclockwise). Inside the room
    there is a green obstacle mass. The black passable cells form only a thin shell around that
    obstacle. The yellow moves around corners on that shell and stacks near the terminal end of the
    shell after the cycle is cut at the shell cell closest to the legend corner.
  - Difference: the destination depends on contour order around a room-with-obstacle, not on a
    global direction vector. The obstacle is not just decoration; it is the shape source that makes
    the shell long and winding.

b) Complexity calibration
The design deliberately keeps several nontrivial freedoms from the base task:
  - 1-2 legend colors, each controlling 1-3 regions, with at most 4 regions total
  - irregular obstacle cores, not repeated templated boxes
  - 1-2 yellows per region, so stacking is still visible
At the same time, the state count of the legend is reduced to two rotation choices because the
hard part is now boundary tracing through obstacle-shaped contours. This keeps the rule inferable
from 2-4 examples instead of mixing obstacle reasoning with 8-way vector decoding.

c) Hidden-assumption audit of the original solver
The original `main()` relied on several assumptions that break here:
  - `_legend_directions(...)` assumed centroid offset should become a literal move vector. In the
    combo only the horizontal placement of the red markers matters; the output is a rotation sense.
  - `passable = (g == bg) | (g == mover)` assumed region identity and movement state space were the
    same object. Here region identity is the bounded meta-component `{bg, yellow, obstacle}`, while
    movers are restricted to its passable shell.
  - `components_mask(passable, connectivity=4)` assumed removing non-passable cells would still
    preserve the room object. That fails once obstacles occupy the interior.
  - `_slide_points(...)` assumed legal motion is repeated addition of one direction vector. The
    combo replaces that with movement on an ordered boundary track.
  - ordering movers by linear projection assumed the region had a single meaningful “far side” in a
    direction. The combo instead needs a linearized cycle derived from the room contour.

The two changes do not conflict if the generator enforces one stronger invariant: every bounded
room is exactly `obstacle_core ∪ 1-cell shell`, so the room contour equals the yellow movement
track and the obstacle changes that contour directly.

d) Placement legality
The generator constructs each region from the inside out:
  - grow a connected obstacle core away from the grid border and away from legends / other regions
  - define the passable shell as the 4-neighbor exterior of that core
  - define the room as `core ∪ shell`
  - define the colored wall as the 4-neighbor exterior of the room
This guarantees no overlap between legends, walls, obstacles, or yellows. A reserve band around
each completed room prevents ambiguous contacts between different regions. The generator rejects
cores whose shell is not a single simple cycle.

e) Visual observability
A human can read the combo from examples because all relevant cues are visible:
  - green cells are clearly internal blockers, distinct from the wall color
  - yellow starts directly on a thin black lane hugging both the wall and the obstacle
  - the output shows it wrapping around corners rather than crossing open floor
  - the legend repeatedly shows a left-vs-right red placement, which maps naturally to
    counterclockwise vs clockwise travel
The generator rejects simple rectangle-like shells, so examples visibly depend on the obstacle
shape instead of collapsing into “walk around a plain box”.

f) Visual cleanliness
There are only five semantic layers: background, legend fills, blue/red legend marks, green
obstacles, yellow movers. The passable shell is one cell thick, so the motion path is readable and
does not visually merge with the obstacle core. Regions are separated by a buffer, reducing clutter
when multiple wall colors appear in one 20x20 grid.

g) Change inferability
Color meaning remains stable:
  - legend fill color matches the wall color of the controlled region
  - green (3) always means internal obstacle and never a wall
  - yellow (4) always means mover
No hidden numeric ordering of colors is used in the transformation. The critical mapping is
structural: wall-color match plus red-left/red-right legend geometry.

h) New solver assumptions and generator guarantees
The combo `main()` introduces new assumptions, and the generator explicitly enforces them:
  - each bounded meta-component contains a simple obstacle-free contour track for movers
  - all movers lie on that contour track
  - the contour cell closest to the legend corner is a stable anchor for cutting the cycle
  - obstacle cells never touch the outer wall, so they remain visually internal
  - wall colors exclude obstacle color 3
  - every accepted sample contains at least one obstacle-shaped shell that is non-rectangular, so
    the obstacle genuinely changes the path rather than merely existing inside the room
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


OBSTACLE = 3
MOVER = 4

GRID_H, GRID_W = 20, 20

# Legend
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

# Regions
REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
ROOM_SIZE_BOUNDS = {
    1: (22, 70),
    2: (18, 42),
    3: (15, 28),
    4: (12, 20),
}
BLOB_GROW_RETRIES = 40
MIN_BOUNDARY_TRACK = 12
MIN_TURN_COUNT = 6
MIN_OBSTACLE_CELLS = 4
MIN_OBSTACLE_CONTACT_RATIO = 0.5

# Yellows
YELLOWS_PER_REGION = [1, 2]
MAX_BOUNDARY_STEPS = 6

# Outer loop
GEN_RETRIES = 500


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


def _corner_point(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - 1
    if corner == "BL":
        return h - 1, 0
    if corner == "BR":
        return h - 1, w - 1
    raise ValueError(corner)


def _legend_corner(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> str:
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == w - 1:
        return "TR"
    if r1 == h - 1 and c0 == 0:
        return "BL"
    if r1 == h - 1 and c1 == w - 1:
        return "BR"
    raise ValueError("legend must sit in a corner")


def _legend_5x5(fill: int, clockwise: bool) -> np.ndarray:
    """5x5 legend: red on right => clockwise, red on left => counterclockwise."""
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[3, 1] = 1
    g[3, 2] = 1
    if clockwise:
        g[1, 3] = 2
        g[3, 3] = 2
    else:
        g[1, 0] = 2
        g[3, 0] = 2
    return g


def _legend_rotation_info(g: np.ndarray, bg: int) -> dict[int, tuple[bool, str]]:
    """
    Detect border legends and return {wall_color: (clockwise, corner)}.

    The most frequent color in the legend rectangle is the fill / wall color.
    The 4-cell marker is the center color, and the 2-cell marker gives the side:
    right of center => clockwise, left of center => counterclockwise.
    """
    h, w = g.shape
    best: dict[int, tuple[int, bool, str]] = {}
    for pts in components_mask(g != bg, connectivity=4):
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
        others = [(cnt, col) for col, cnt in counts.items() if col != fill]
        if sorted(cnt for cnt, _ in others) != [2, 4]:
            continue
        others.sort()
        indicator_col = others[0][1]
        center_col = others[1][1]

        center_pts = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic_pts = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center_pts or not indic_pts:
            continue

        center = np.mean(np.array(center_pts, dtype=float), axis=0)
        indic = np.mean(np.array(indic_pts, dtype=float), axis=0)
        dc = float(indic[1] - center[1])
        if abs(dc) < 1e-6:
            continue
        clockwise = dc > 0
        corner = _legend_corner(r0, c0, r1, c1, h, w)

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), clockwise, corner)

    return {col: (clockwise, corner) for col, (_, clockwise, corner) in best.items()}


def _boundary_cells(region_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c in region_set
        if any((nr, nc) not in region_set for nr, nc in neighbors4(r, c))
    }


def _boundary_cycle_clockwise(region_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Trace the clockwise outer contour and map it back to an ordered list of boundary cells.
    """
    if not region_set:
        return []

    next_edge: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}

    def add_edge(start: tuple[int, int], end: tuple[int, int], cell: tuple[int, int]) -> None:
        if start in next_edge:
            raise ValueError("non-simple contour")
        next_edge[start] = (end, cell)

    for r, c in region_set:
        if (r - 1, c) not in region_set:
            add_edge((r, c), (r, c + 1), (r, c))
        if (r, c + 1) not in region_set:
            add_edge((r, c + 1), (r + 1, c + 1), (r, c))
        if (r + 1, c) not in region_set:
            add_edge((r + 1, c + 1), (r + 1, c), (r, c))
        if (r, c - 1) not in region_set:
            add_edge((r + 1, c), (r, c), (r, c))

    start = min(next_edge)
    cur = start
    raw_cells: list[tuple[int, int]] = []
    seen_vertices = set()
    while True:
        if cur in seen_vertices and cur != start:
            raise ValueError("contour should be a single cycle")
        seen_vertices.add(cur)
        nxt, cell = next_edge[cur]
        raw_cells.append(cell)
        cur = nxt
        if cur == start:
            break

    collapsed: list[tuple[int, int]] = []
    for cell in raw_cells:
        if not collapsed or cell != collapsed[-1]:
            collapsed.append(cell)
    if collapsed and collapsed[0] == collapsed[-1]:
        collapsed.pop()

    ordered_unique: list[tuple[int, int]] = []
    seen_cells: set[tuple[int, int]] = set()
    for cell in collapsed:
        if cell not in seen_cells:
            ordered_unique.append(cell)
            seen_cells.add(cell)
    return ordered_unique


def _ordered_boundary_track(
    region_set: set[tuple[int, int]], corner: str, clockwise: bool, h: int, w: int
) -> list[tuple[int, int]]:
    cycle = _boundary_cycle_clockwise(region_set)
    if not cycle:
        return []
    corner_pt = _corner_point(corner, h, w)
    anchor = min(
        cycle,
        key=lambda p: (abs(p[0] - corner_pt[0]) + abs(p[1] - corner_pt[1]), p[0], p[1]),
    )
    k = cycle.index(anchor)
    rotated = cycle[k:] + cycle[:k]
    if clockwise:
        return rotated
    return [rotated[0]] + list(reversed(rotated[1:]))


def _slide_along_track(track: list[tuple[int, int]], points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not track or not points:
        return list(points)
    index = {p: i for i, p in enumerate(track)}
    moving = [p for p in points if p in index]
    fixed = [p for p in points if p not in index]
    occupied: set[int] = set()
    out_pts: list[tuple[int, int]] = []
    for p in sorted(moving, key=lambda q: index[q], reverse=True):
        j = index[p]
        while j + 1 < len(track) and (j + 1) not in occupied:
            j += 1
        occupied.add(j)
        out_pts.append(track[j])
    return out_pts + fixed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside obstacle-filled rooms along the ordered boundary shell.

    Input structure:
    - The grid has 1-2 corner legends; each legend's fill color matches one wall color.
    - Every movable region is a bounded meta-component made of black shell cells, yellow movers,
      and green internal obstacle cells.
    - The passable shell is a one-cell lane wrapped between the colored outer wall and the green
      obstacle core.

    Rule:
    - For each bounded room, find its wall color and the matching legend.
    - Read the legend as clockwise or counterclockwise from the red markers' side relative to the
      blue center markers.
    - Trace the room's outer contour, cut it at the contour cell closest to the legend corner, and
      orient that contour by the legend rotation.
    - Yellow pixels slide forward on that contour shell and stack at the far end; obstacle and wall
      cells never move.

    Key insight:
    - Obstacles change the room contour itself, so the true movement path is the shell created by
      the wall/obstacle pair. The legend still matches by color, but it now chooses which way to
      traverse that contour rather than giving a Cartesian direction vector.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_info = _legend_rotation_info(g, bg)
    assert legend_info, "grid must contain at least one border legend"

    room_mask = (g == bg) | (g == MOVER) | (g == OBSTACLE)
    out = g.copy()

    for pts in components_mask(room_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        room = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
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
        if border_col not in legend_info:
            continue
        clockwise, corner = legend_info[border_col]
        track = [
            p
            for p in _ordered_boundary_track(room, corner, clockwise, h, w)
            if int(g[p]) in (bg, MOVER)
        ]
        if not track:
            continue

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_along_track(track, movers):
            out[r, c] = MOVER

    return out


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


def _room_wall_cells(room: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in room:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in room:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _room_wall_cells(blob, h, w)
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


def _track_turn_count(track: list[tuple[int, int]]) -> int:
    if len(track) < 3:
        return 0
    dirs = []
    for a, b in zip(track, track[1:] + track[:1]):
        dirs.append((b[0] - a[0], b[1] - a[1]))
    turns = 0
    for d0, d1 in zip(dirs, dirs[1:] + dirs[:1]):
        if d0 != d1:
            turns += 1
    return turns


def _shape_has_simple_shell(room: set[tuple[int, int]], corner: str, clockwise: bool, h: int, w: int) -> bool:
    boundary = _boundary_cells(room)
    core = room - boundary
    if len(boundary) < MIN_BOUNDARY_TRACK or len(core) < MIN_OBSTACLE_CELLS:
        return False

    obstacle_contact = sum(
        any((nr, nc) in core for nr, nc in neighbors4(r, c)) for r, c in boundary
    )
    if obstacle_contact < max(MIN_OBSTACLE_CELLS, int(np.ceil(MIN_OBSTACLE_CONTACT_RATIO * len(boundary)))):
        return False

    try:
        track = _ordered_boundary_track(room, corner, clockwise, h, w)
    except ValueError:
        return False
    if len(track) != len(boundary) or set(track) != boundary:
        return False

    if _track_turn_count(track) < MIN_TURN_COUNT:
        return False
    return True


def _try_grow_valid_core(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
    corner: str,
    clockwise: bool,
) -> dict | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        room = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if room is None:
            continue
        while True:
            filled = _fill_holes(room, h, w)
            if filled == room:
                break
            room = filled
        if any(not canvas[r, c] for r, c in room):
            continue

        walls = _room_wall_cells(room, h, w)
        boundary = _boundary_cells(room)
        core = room - boundary
        if not core or not walls:
            continue
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in room | walls):
            continue
        if any(occupied[r, c] for r, c in room | walls):
            continue
        if not _shape_has_simple_shell(room, corner, clockwise, h, w):
            continue

        track = _ordered_boundary_track(room, corner, clockwise, h, w)
        return {
            "core": core,
            "room": room,
            "walls": walls,
            "track": track,
        }
    return None


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, bool, str]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, bool, str]] = []

    for corner, wall_color in zip(corners, wall_colors):
        clockwise = bool(rng.randrange(2))
        patch = _legend_5x5(wall_color, clockwise)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, clockwise, corner))

    return legend_occupied, legends


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, bool, str]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_info = {wall_color: (clockwise, corner) for wall_color, clockwise, corner in legends}
    all_colors = [wall_color for wall_color, _, _ in legends]

    region_colors = []
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = ROOM_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for color in region_colors:
        clockwise, corner = color_info[color]
        target_size = rng.randint(size_min, size_max)

        canvas = np.ones((h, w), dtype=bool)
        canvas[:2, :] = False
        canvas[-2:, :] = False
        canvas[:, :2] = False
        canvas[:, -2:] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        grown = _try_grow_valid_core(rng, canvas, occupied, target_size, h, w, corner, clockwise)
        if grown is None:
            continue

        core = grown["core"]
        room = grown["room"]
        walls = grown["walls"]
        track = grown["track"]
        regions.append(
            {
                "color": color,
                "clockwise": clockwise,
                "corner": corner,
                "core": core,
                "room": room,
                "walls": walls,
                "track": track,
            }
        )

        reserve = room | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(all_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    meta = (grid == 0) | (grid == OBSTACLE)
    labeled, n_comp = label_mask(meta, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    expected = [reg["room"] for reg in regions]
    if len(bounded) != len(expected):
        return False
    return all(comp in expected for comp in bounded)


def _forward_slide_index(start_idx: int, blocked: set[int], track_len: int) -> int:
    j = start_idx
    while j + 1 < track_len and (j + 1) not in blocked:
        j += 1
    return j


def _find_reverse_track_index(
    out_idx: int,
    track_len: int,
    blocked_outputs: set[int],
    existing_inputs: set[int],
    rng: random.Random,
) -> int | None:
    max_steps = min(MAX_BOUNDARY_STEPS, out_idx)
    if max_steps <= 0:
        return None
    target_steps = rng.randint(1, max_steps)
    for steps in range(target_steps, 0, -1):
        cand = out_idx - steps
        if cand in blocked_outputs or cand in existing_inputs:
            continue
        if _forward_slide_index(cand, blocked_outputs, track_len) == out_idx:
            return cand
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    moved_any = False
    for reg in regions:
        track = reg["track"]
        track_len = len(track)
        n_yellows = rng.choice(YELLOWS_PER_REGION)
        if track_len < max(MIN_BOUNDARY_TRACK, n_yellows + 4):
            return False

        output_indices = [track_len - 1 - i for i in range(n_yellows)]
        blocked_outputs: set[int] = set()
        input_indices: list[int] = []

        for out_idx in output_indices:
            cand = _find_reverse_track_index(
                out_idx, track_len, blocked_outputs, set(input_indices), rng
            )
            if cand is None:
                return False
            input_indices.append(cand)
            blocked_outputs.add(out_idx)
            if cand != out_idx:
                moved_any = True

        for idx in input_indices:
            r, c = track[idx]
            grid[r, c] = MOVER

    return moved_any


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    immovable = (input_grid != 0) & (input_grid != MOVER)
    if not np.array_equal(solver_out[immovable], input_grid[immovable]):
        return False
    if int((solver_out == MOVER).sum()) != int((input_grid == MOVER).sum()):
        return False

    region_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["track"]:
            region_mask[r, c] = True
    return bool(np.all(region_mask[solver_out == MOVER]))


def _degenerate_without_obstacles_output(grid: np.ndarray) -> np.ndarray:
    """
    Ablation used for post-hoc analysis: remove obstacles and apply pure D3-style wall tracing.
    """
    g = np.asarray(grid, dtype=int).copy()
    bg = get_bg(g)
    g[g == OBSTACLE] = bg
    h, w = g.shape
    legend_info = _legend_rotation_info(g, bg)
    out = g.copy()
    passable = (g == bg) | (g == MOVER)

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        region = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in region:
                    continue
                v = int(g[nr, nc])
                if v in (bg, MOVER):
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in legend_info:
            continue
        clockwise, corner = legend_info[border_col]
        track = _ordered_boundary_track(region, corner, clockwise, h, w)
        if not track:
            continue

        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_along_track(track, movers):
            out[r, c] = MOVER

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]
        for r, c in reg["core"]:
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
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
