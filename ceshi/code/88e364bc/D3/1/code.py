# Variant of 88e364bc
# Dimension: D3 — 滑动路径类型 (path)
# Change: directional_movement——黄色像素沿图例指定的固定方向（基数或对角线）直线滑动，遇到区域边界或已占据位... → 沿墙壁边界滑动

"""
Design analysis for the D3 variant
==================================

a) Concrete behavior
The original task links each wall color to a border legend and interprets the red-vs-blue offset
as a straight movement vector. Here that vector is replaced by a rotation sense. The legend still
matches a wall color, but now only encodes two states:
  - red markers on the right side of the blue block => clockwise
  - red markers on the left side of the blue block => counterclockwise

To make "slide along the inner wall" visually unambiguous, yellow pixels are generated directly on
the region's inner boundary track. The region is still an enclosed irregular room, but the yellow
does not cut across open floor anymore; it travels on the ordered ring of boundary cells. The ring
is "cut" at the boundary cell closest to the matching legend's corner, turning the cycle into a
directed linear track. Yellows then slide forward on that track and stack at the far end. This
keeps the original "stack against a terminal wall" feel while changing the path from straight to
boundary-following.

Small trace example:
  - Original: a yellow at the top edge of a room with a right-pointing legend moves horizontally
    through open interior cells until the east wall.
  - Variant: the same room with a clockwise legend makes the yellow stay on the wall-adjacent
    cells, go around corners on the boundary ring, and stack near the cut point's clockwise side.
  - Difference: the destination is determined by contour order, not by linear projection.

b) Complexity calibration
This is not reduced to a trivial binary toggle on isolated objects:
  - there are still 1-2 wall colors and 1-4 regions total
  - one legend color can govern multiple regions
  - regions remain irregular closed blobs rather than identical boxes
  - each region has 1-2 yellows, so stacking remains observable
At the same time, the legend state count is intentionally reduced from 8 directions to 2 rotation
states because the new difficulty is boundary tracing, not vector decoding.

c) Hidden assumptions in the original solver that no longer hold
Relevant original lines and the broken assumptions:
  - `_legend_directions(...)` assumed centroid offset should become a literal `(dr, dc)` move.
    In the variant, only the horizontal side of the red marker matters; there is no straight-line
    displacement vector anymore.
  - `_slide_points(region_set, points, dr, dc)` assumed the legal path is "keep adding (dr, dc)
    while still in the region". That is invalid now because legal motion is constrained to the
    ordered inner-boundary track, not all region cells.
  - Sorting movers by `dr * r + dc * c` assumed there is a global projection order. The variant
    requires ordering by contour index along the boundary track.
  - Reverse generation via `_find_reverse_slide(..., dr, dc, ...)` assumed trajectories are simple
    linear rays. It must become reverse movement on a 1D boundary ordering instead.

d) Placement legality
The generator keeps the original closed-room construction and legend/wall non-overlap rules, then
adds one new legality filter: the extracted boundary track must be a simple ring of unique boundary
cells. Shapes whose contour revisits the same boundary cell from opposite sides (common in thin
1-cell corridors) are rejected. This guarantees:
  - the solver can assign every boundary cell a unique contour index
  - reverse-sliding on the track is well-defined
  - yellows never need to leave the wall-adjacent ring

e) Visual observability
A human can now see that yellows:
  - start on the room perimeter
  - move around corners
  - accumulate near one side of the region point nearest the legend corner
The legend shows a left-vs-right red placement, and across examples this repeated binary cue is
enough to infer clockwise vs counterclockwise. The visible wrapped trajectories distinguish the
variant from the original straight-line slide.

f) Visual cleanliness
Only four visual layers remain: background, wall rectangles, legends, yellows. Yellows stay on the
interior wall ring, so they do not overlap legends or wall pixels. A one-cell reserve around walls
is preserved between regions, reducing clutter.

g) Color inference
No new hidden color mapping is introduced. The fill color of each legend still matches the region's
wall color, exactly as in the original task. Blue and red retain their role as internal legend
markers; only the interpretation changes from direction-vector to rotation sense.

h) New solver assumptions introduced by this variant
The new `main()` assumes:
  - every moving yellow lies on the simple inner-boundary track
  - the matching region is simply connected and hole-free
  - the closest-to-legend corner boundary cell gives a stable cut point for linearizing the cycle
  - the boundary track contains unique cells in contour order
The generator explicitly enforces all four conditions before accepting a sample.
"""

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4

# Legend
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

# Regions
REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (16, 55),
    3: (14, 36),
    4: (12, 26),
}
BLOB_GROW_RETRIES = 35
MIN_BOUNDARY_TRACK = 10

# Yellows
YELLOWS_PER_REGION = [1, 2]
MAX_BOUNDARY_STEPS = 6

# Outer loop
GEN_RETRIES = 400


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

    The patch format intentionally mirrors the original legend detector:
    - most frequent color = fill color C = wall color
    - 4 blue cells = center
    - 2 red cells = indicator
    - indicator centroid right of center => clockwise, left => counterclockwise
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

        cc = np.mean(np.array(center_pts, dtype=float), axis=0)
        ci = np.mean(np.array(indic_pts, dtype=float), axis=0)
        dc = float(ci[1] - cc[1])
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
    Trace the outer contour clockwise and convert it to an ordered list of boundary cells.

    Each exposed cell side contributes one directed edge with the region on its right:
      north -> east, east -> south, south -> west, west -> north
    The resulting edge cycle is the clockwise contour of the polyomino.

    We then map contour edges back to the adjacent interior cells and collapse consecutive
    duplicates. The generator rejects shapes where a boundary cell would need to appear twice
    non-consecutively; here we still return first-occurrence order so the solver stays total.
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


def _slide_along_track(
    track: list[tuple[int, int]], points: list[tuple[int, int]]
) -> list[tuple[int, int]]:
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
    Slide yellow pixels inside walled regions along the inner boundary track.

    Input structure:
    - bg = most common color
    - yellow pixels lie on the inner boundary of enclosed passable regions
    - border legends are 5x5 rectangles; fill color matches wall color
    - red markers on the right/left side of the blue center encode clockwise/counterclockwise

    Rule:
    - For each enclosed region, identify its border color and matching legend.
    - Extract the ordered boundary ring of inner-wall cells.
    - Cut that ring at the boundary cell closest to the legend's corner.
    - Orient the track clockwise or counterclockwise from that cut and slide yellows forward,
      stacking at the far end of the track.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_info = _legend_rotation_info(g, bg)
    assert legend_info, "grid must contain at least one border legend"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in region_set:
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
        track = _ordered_boundary_track(region_set, corner, clockwise, h, w)
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


def _shape_has_simple_track(blob: set[tuple[int, int]], corner: str, clockwise: bool, h: int, w: int) -> bool:
    track = _ordered_boundary_track(blob, corner, clockwise, h, w)
    return len(track) >= MIN_BOUNDARY_TRACK and len(track) == len(set(track)) == len(_boundary_cells(blob))


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
    corner: str,
    clockwise: bool,
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
        if not _shape_has_simple_track(cand, corner, clockwise, h, w):
            continue
        return cand
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

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for color in region_colors:
        clockwise, corner = color_info[color]
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w, corner, clockwise)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        track = _ordered_boundary_track(blob, corner, clockwise, h, w)
        regions.append(
            {
                "color": color,
                "clockwise": clockwise,
                "corner": corner,
                "interior": blob,
                "walls": walls,
                "track": track,
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
    if {reg["color"] for reg in regions} != set(all_colors):
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
    target_steps = rng.randint(1, min(MAX_BOUNDARY_STEPS, out_idx + 1))
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
        if track_len < MIN_BOUNDARY_TRACK:
            return False
        n_yellows = rng.choice(YELLOWS_PER_REGION)
        if track_len < n_yellows + 2:
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
    if not bool(np.all(regions_mask[solver_out == MOVER])):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

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
