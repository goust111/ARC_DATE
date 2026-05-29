# Combo variant of 88e364bc (index 41)
# combo_spec: D1/0+D3/1
# Name: 箭头图例+沿墙滑动

# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Original task: a border legend encodes a literal slide vector, and each yellow moves on a
#      straight ray through the room interior until it hits the far side.
#    - D1/0 changes how direction is read: the legend is no longer a centroid puzzle; the wall
#      color itself forms a small arrow silhouette.
#    - D3/1 changes the path: yellows no longer cross open floor, they move only on the inner-wall
#      boundary track of the enclosed region.
#    - Fusion rule used here: the arrow still has an orientation, but that orientation is not a
#      free-space travel vector. After cutting the region's boundary ring at the cell closest to
#      the matching legend corner, the arrow tells us which of the two tangent directions to take
#      from that anchor. Following that tangent around the wall induces clockwise or
#      counterclockwise motion and the yellows stack at the far end of that oriented track.
#    - Small trace: if a TL-corner legend contains a wall-colored arrow pointing east, then for
#      each matching region we look at the boundary cell nearest TL. If the clockwise first step
#      from that anchor is east, the region uses the clockwise track; if the counterclockwise first
#      step is east, it uses the counterclockwise track. The same arrow would have meant "slide
#      east through the room" in the original task, but here it only fixes the starting tangent on
#      the wall ring.
# b) Complexity calibration
#    - The task still contains 1-2 legend colors, irregular non-rectangular closed rooms, and 1-2
#      yellows per active region, so the output remains structurally rich.
#    - Arrow directions still come from a small discrete family (4 cardinals via 5x5 arrows, 4
#      diagonals via 6x7 arrows), avoiding excessive freedom while keeping the visual cue obvious.
#    - Some grids contain multiple regions for the same wall color; the generator only accepts such
#      samples when those regions share the same "start tangent + rotation sense" so the legend is
#      globally consistent within the grid.
# c) Hidden-assumption audit against the original solver
#    - Original `_legend_directions(...)` assumed three-color legends and centroid arithmetic.
#      That is invalid now because legends are two-color arrow templates. The new solver matches
#      the dominant-color silhouette against 5x5 / 6x7 arrow masks.
#    - Original `_slide_points(region_set, points, dr, dc)` assumed every passable cell belongs to
#      the legal path. That fails under boundary tracing. Motion must instead be restricted to the
#      ordered boundary track extracted from the enclosed region.
#    - Original sorting by a global projection `dr*r + dc*c` also breaks. The new order is the
#      1D index on the oriented wall track.
#    - Fusion-specific new assumption: the arrow direction must equal one of the two possible first
#      steps from the anchor boundary cell. The generator guarantees this by rejecting regions whose
#      anchor tangents cannot support a consistent per-color legend.
# d) Placement legality
#    - Legend slots are reserved at corners using the maximum 6x7 footprint plus a one-cell buffer,
#      so later-chosen 5x5 or 6x7 arrows never collide with walls.
#    - Regions are grown only on interior free cells; their wall rings may not touch the grid
#      border, legends, or each other after buffering.
#    - Reverse yellow placement is done on the final oriented boundary track, so every generated
#      input is guaranteed to be solvable by the same wall-following logic used in `main()`.
# e) Visual observability
#    - A human can see three linked cues directly in the pixels: wall color matches legend color,
#      the legend visibly looks like an arrow rather than a dot-offset code, and the yellows stay
#      glued to the room perimeter while turning corners.
#    - Because the arrow matches the initial tangent at the anchor, samples make clear that the
#      arrow is not a free-space motion vector: the yellows do not cut across the room.
# f) Visual cleanliness
#    - The grid contains only background, wall rings, legend patches, and yellows. Yellows are
#      restricted to boundary cells and legends are isolated at corners with padding, so layers do
#      not visually occlude each other.
# g) Color inference
#    - No hidden remapping is introduced: the dominant legend color is still exactly the wall color
#      of the region(s) it controls. The auxiliary legend color is a neutral board color only.
# h) New solver assumptions enforced by the generator
#    - Every active region has a simple boundary cycle with a unique anchor cell under the solver's
#      tie-break.
#    - The chosen arrow direction equals the clockwise first step or the counterclockwise first
#      step for every region of that legend color.
#    - All generated yellows begin on the selected oriented boundary track and slide at least one
#      step before stacking.

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4

# Legend
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
LEGEND_BG = 1
MAX_LEGEND_SHAPE = (6, 7)

# Regions
REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (16, 55),
    3: (14, 36),
    4: (12, 26),
}
BLOB_GROW_RETRIES = 40
MIN_BOUNDARY_TRACK = 10

# Yellows
YELLOWS_PER_REGION = [1, 2]
MAX_BOUNDARY_STEPS = 6

# Outer loop
GEN_RETRIES = 500

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

CARDINAL_ARROW_RIGHT = np.array(
    [
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0],
    ],
    dtype=bool,
)

DIAGONAL_ARROW_DR = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=bool,
)


def _cardinal_arrow_mask(dr: int, dc: int) -> np.ndarray:
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(CARDINAL_ARROW_RIGHT, rot[(dr, dc)])


def _diagonal_arrow_mask(dr: int, dc: int) -> np.ndarray:
    if (dr, dc) == (1, 1):
        return DIAGONAL_ARROW_DR.copy()
    if (dr, dc) == (-1, 1):
        return np.flipud(DIAGONAL_ARROW_DR)
    if (dr, dc) == (1, -1):
        return np.fliplr(DIAGONAL_ARROW_DR)
    if (dr, dc) == (-1, -1):
        return np.rot90(DIAGONAL_ARROW_DR, 2)
    raise ValueError((dr, dc))


CARDINAL_TEMPLATE_MASKS = {(dr, dc): _cardinal_arrow_mask(dr, dc) for dr, dc in CARDINAL_DIRS}
DIAGONAL_TEMPLATE_MASKS = {(dr, dc): _diagonal_arrow_mask(dr, dc) for dr, dc in DIAGONAL_DIRS}


def _legend_base(h: int, w: int) -> np.ndarray:
    return np.full((h, w), LEGEND_BG, dtype=int)


def _legend_patch(fill: int, direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    if dr == 0 or dc == 0:
        patch = _legend_base(5, 5)
        patch[_cardinal_arrow_mask(dr, dc)] = fill
        return patch
    patch = _legend_base(6, 7)
    patch[_diagonal_arrow_mask(dr, dc)] = fill
    return patch


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


def _match_arrow_direction(fill_mask: np.ndarray) -> tuple[int, int] | None:
    if fill_mask.shape == (5, 5):
        for direction, templ in CARDINAL_TEMPLATE_MASKS.items():
            if np.array_equal(fill_mask, templ):
                return direction
    elif fill_mask.shape == (6, 7):
        for direction, templ in DIAGONAL_TEMPLATE_MASKS.items():
            if np.array_equal(fill_mask, templ):
                return direction
    return None


def _legend_arrow_info(g: np.ndarray, bg: int) -> dict[int, tuple[tuple[int, int], str]]:
    """
    Detect border legends and decode wall_color -> (arrow_direction, corner).

    The dominant legend color is still the wall color. Direction comes from matching that color's
    silhouette against arrow templates instead of comparing minority-color centroids.
    """
    h, w = g.shape
    best: dict[int, tuple[int, tuple[int, int], str]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) not in ((5, 5), (6, 7)):
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        vals, counts = np.unique(patch, return_counts=True)
        if len(vals) < 2:
            continue
        fill = int(vals[np.argmax(counts)])
        direction = _match_arrow_direction(patch == fill)
        if direction is None:
            continue

        corner = _legend_corner(r0, c0, r1, c1, h, w)
        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), direction, corner)

    return {color: (direction, corner) for color, (_, direction, corner) in best.items()}


def _boundary_cells(region_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c in region_set
        if any((nr, nc) not in region_set for nr, nc in neighbors4(r, c))
    }


def _boundary_cycle_clockwise(region_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Trace the outer contour clockwise and convert it to an ordered list of unique boundary cells.

    The contour is rejected if a boundary cell would need to appear twice non-consecutively.
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
    if len(collapsed) != len(set(collapsed)):
        return []
    return collapsed


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
    if sdr == 0 and sdc == 0:
        raise ValueError("step direction must be non-zero")
    return sdr, sdc


def _region_track_profile(
    region_set: set[tuple[int, int]], corner: str, h: int, w: int
) -> dict[str, object] | None:
    cycle = _boundary_cycle_clockwise(region_set)
    if len(cycle) < MIN_BOUNDARY_TRACK:
        return None
    if len(cycle) != len(_boundary_cells(region_set)):
        return None

    corner_pt = _corner_point(corner, h, w)
    anchor = min(
        cycle,
        key=lambda p: (abs(p[0] - corner_pt[0]) + abs(p[1] - corner_pt[1]), p[0], p[1]),
    )
    k = cycle.index(anchor)
    track_cw = cycle[k:] + cycle[:k]
    if len(track_cw) < 2:
        return None
    cw_dir = _step_dir(track_cw[0], track_cw[1])
    ccw_dir = _step_dir(track_cw[0], track_cw[-1])
    if cw_dir == ccw_dir:
        return None
    track_ccw = [track_cw[0]] + list(reversed(track_cw[1:]))
    return {
        "anchor": anchor,
        "track_cw": track_cw,
        "track_ccw": track_ccw,
        "cw_dir": cw_dir,
        "ccw_dir": ccw_dir,
    }


def _dir_score(actual: tuple[int, int], target: tuple[int, int]) -> int:
    return actual[0] * target[0] + actual[1] * target[1]


def _select_track_from_arrow(profile: dict[str, object], arrow_dir: tuple[int, int]) -> list[tuple[int, int]] | None:
    cw_dir = profile["cw_dir"]
    ccw_dir = profile["ccw_dir"]
    if arrow_dir == cw_dir and arrow_dir != ccw_dir:
        return list(profile["track_cw"])
    if arrow_dir == ccw_dir and arrow_dir != cw_dir:
        return list(profile["track_ccw"])

    cw_score = _dir_score(cw_dir, arrow_dir)
    ccw_score = _dir_score(ccw_dir, arrow_dir)
    if max(cw_score, ccw_score) <= 0 or cw_score == ccw_score:
        return None
    return list(profile["track_cw"] if cw_score > ccw_score else profile["track_ccw"])


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
    Slide yellow pixels inside walled regions along the inner wall, using arrow legends as entry tangents.

    Input structure:
    - bg is the most common color, colored walls enclose irregular rooms, and yellow pixels lie on
      the passable cells inside those rooms.
    - Border legends sit in grid corners as 5x5 or 6x7 rectangles. Their dominant color matches a
      region wall color, and that dominant color is arranged as a visible arrow silhouette.

    Rule:
    - For each enclosed region, find the matching legend by wall color and locate the boundary cell
      of the region that is closest to that legend's corner.
    - The legend arrow does not mean "move straight through the room". Instead it selects which
      tangent direction to follow from that anchor on the region's inner boundary ring.
    - Orient the wall track accordingly and slide all yellows forward on that track so they stack
      at the far end.

    Key insight: The legend color still links the correct room, but the arrow now chooses the
    starting tangent of a wall-following motion; the global effect is clockwise or
    counterclockwise boundary tracing rather than straight-line translation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_info = _legend_arrow_info(g, bg)
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

        arrow_dir, corner = legend_info[border_col]
        profile = _region_track_profile(region_set, corner, h, w)
        if profile is None:
            continue
        track = _select_track_from_arrow(profile, arrow_dir)
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


def _reserve_legend_slots(
    corners: list[str], h: int, w: int
) -> np.ndarray:
    occupied = np.zeros((h, w), dtype=bool)
    ph, pw = MAX_LEGEND_SHAPE
    for corner in corners:
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        occupied[r_lo:r_hi, c_lo:c_hi] = True
    return occupied


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
    corner: str,
) -> tuple[set[tuple[int, int]], dict[str, object]] | None:
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
        profile = _region_track_profile(cand, corner, h, w)
        if profile is None:
            continue
        return cand, profile
    return None


def _plan_legends(rng: random.Random) -> list[dict[str, object]]:
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    return [{"color": color, "corner": corner} for corner, color in zip(corners, colors)]


def _grow_regions(
    rng: random.Random,
    legend_slots: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    h, w = GRID_H, GRID_W
    color_corner = {slot["color"]: slot["corner"] for slot in legend_slots}
    all_colors = [int(slot["color"]) for slot in legend_slots]

    region_colors: list[int] = []
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = _reserve_legend_slots([str(slot["corner"]) for slot in legend_slots], h, w)
    regions: list[dict[str, object]] = []

    for color in region_colors:
        corner = str(color_corner[color])
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        grown = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w, corner)
        if grown is None:
            continue
        blob, profile = grown
        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "color": color,
                "corner": corner,
                "interior": blob,
                "walls": walls,
                "profile": profile,
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
    if {int(reg["color"]) for reg in regions} != set(all_colors):
        return None
    return regions


def _choose_legend_specs(
    rng: random.Random,
    legend_slots: list[dict[str, object]],
    regions: list[dict[str, object]],
) -> dict[int, dict[str, object]] | None:
    specs: dict[int, dict[str, object]] = {}
    for slot in legend_slots:
        color = int(slot["color"])
        corner = str(slot["corner"])
        regs = [reg for reg in regions if int(reg["color"]) == color]
        candidates: list[tuple[tuple[int, int], bool]] = []

        cw_dirs = {tuple(reg["profile"]["cw_dir"]) for reg in regs}
        if len(cw_dirs) == 1:
            candidates.append((next(iter(cw_dirs)), True))

        ccw_dirs = {tuple(reg["profile"]["ccw_dir"]) for reg in regs}
        if len(ccw_dirs) == 1:
            candidates.append((next(iter(ccw_dirs)), False))

        if not candidates:
            return None

        direction, clockwise = candidates[rng.randrange(len(candidates))]
        specs[color] = {"corner": corner, "direction": direction, "clockwise": clockwise}
        for reg in regs:
            reg["direction"] = direction
            reg["clockwise"] = clockwise
            reg["track"] = (
                list(reg["profile"]["track_cw"])
                if clockwise
                else list(reg["profile"]["track_ccw"])
            )
    return specs


def _paint_legends(grid: np.ndarray, legend_specs: dict[int, dict[str, object]]) -> None:
    h, w = grid.shape
    for color, spec in legend_specs.items():
        patch = _legend_patch(color, tuple(spec["direction"]))
        ph, pw = patch.shape
        r0, c0 = _corner_origin(str(spec["corner"]), h, w, ph, pw)
        paste(grid, patch, r0, c0)


def _validate_topology(grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
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


def _place_yellows(rng: random.Random, regions: list[dict[str, object]], grid: np.ndarray) -> bool:
    moved_any = False
    for reg in regions:
        track = list(reg["track"])
        track_len = len(track)
        n_yellows = rng.choice(YELLOWS_PER_REGION)
        if track_len < max(MIN_BOUNDARY_TRACK, n_yellows + 2):
            return False

        output_indices = [track_len - 1 - i for i in range(n_yellows)]
        blocked_outputs: set[int] = set()
        input_indices: list[int] = []

        for out_idx in output_indices:
            cand = _find_reverse_track_index(out_idx, track_len, blocked_outputs, set(input_indices), rng)
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


def _oracle_check(input_grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
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


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    legend_slots = _plan_legends(rng)
    regions = _grow_regions(rng, legend_slots)
    if regions is None:
        return None

    legend_specs = _choose_legend_specs(rng, legend_slots, regions)
    if legend_specs is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = int(reg["color"])
    _paint_legends(grid, legend_specs)

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
