# Combo variant of 88e364bc (index 23)
# combo_spec: D1/1+D12/0
# Name: 单像素偏移编码+边缘中部图例

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for combo D1/1 + D12/0
# - Original legend detection assumes corner placement, large rectangular patches, and 3 colors
#   where direction comes from centroid(indicator) - centroid(center).
# - This combo removes the center-color channel entirely. A valid legend is now exactly one
#   border-touching 3x3 solid block whose dominant color C occupies 8 cells and whose single
#   odd-color pixel sits at one of the eight non-central offsets around the block center.
# - Therefore the solver must replace "scan corners for 3-color rectangles" with
#   "scan border-touching full 3x3 components anywhere on the border, then decode the singleton
#   offset relative to the 3x3 center".
# - Edge-middle placement breaks an implicit generator/solver assumption that legends live in
#   corner-exclusive space. To keep legends visually distinct from region walls, the generator
#   reserves a 1-cell buffer around each 3x3 patch and still forbids walls from touching the outer
#   border.
# - The rule remains human-visible: wall color C still links legend to region, while the only
#   non-wall pixel inside the compact legend visibly points in one of 8 directions.
# - Degeneration checks used during generation/testing:
#   (1) every legend bbox is 3x3 and touches the border,
#   (2) its anchor is on an edge midpoint rather than a corner,
#   (3) old-style 3-color corner legends are absent,
#   (4) diagonal directions are allowed and sampled.


# ── Shared configuration ──────────────────────────────────────────

GRID_H, GRID_W = 20, 20
MOVER = 4
INDICATOR = 2
ALL_DIRS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
]
EDGES = ["T", "B", "L", "R"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (15, 85),
    2: (15, 55),
    3: (12, 35),
    4: (12, 25),
}
BLOB_GROW_RETRIES = 25
YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 300


# ── Solver helpers ────────────────────────────────────────────────

def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect compact legends on the border.

    A valid legend is a border-touching 3x3 filled block with exactly two colors:
    - fill color C appears 8 times and matches the wall color it controls
    - one singleton indicator pixel marks the direction by its offset from the 3x3 center
    """
    h, w = g.shape
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if not _is_mid_edge_bbox(r0, c0, h, w):
            continue
        if len(pts) != 9:
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if sorted(counts.values()) != [1, 8]:
            continue

        fill = max(counts, key=counts.get)
        indicator = min(counts, key=counts.get)
        indic_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == indicator]
        if len(indic_pts) != 1:
            continue

        ir, ic = indic_pts[0]
        dr, dc = ir - (r0 + 1), ic - (c0 + 1)
        if (dr, dc) == (0, 0):
            continue

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _is_mid_edge_bbox(r0: int, c0: int, h: int, w: int) -> bool:
    return (r0, c0) in {
        _edge_origin("T", h, w),
        _edge_origin("B", h, w),
        _edge_origin("L", h, w),
        _edge_origin("R", h, w),
    }


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
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
    Slide yellow pixels inside closed wall regions using compact edge legends.

    Input structure:
    - Black is background; yellow pixels appear inside enclosed passable regions bounded by colored walls.
    - Each controlling legend is a 3x3 border-touching block placed at the middle of one grid edge.
      Eight cells use wall color C, and one odd-colored pixel inside the 3x3 marks a direction.

    Rule:
    - For each enclosed region, determine its surrounding wall color C from adjacent non-background cells.
    - Find the border legend whose dominant color is the same C; the singleton pixel's offset from the
      3x3 center gives the slide vector, including diagonals.
    - All yellow pixels in that region slide in that direction until blocked by the far wall or by earlier
      stacked yellows in the same region.

    Key insight:
    - The legend-to-region link is still the shared wall color, but the direction is no longer read from a
      two-color centroid comparison. It is encoded directly by the unique off-center pixel inside a compact
      3x3 legend on a border midpoint.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "expected at least one 3x3 edge legend"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == MOVER:
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
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = MOVER
    return out


# ── Generator helpers ─────────────────────────────────────────────

def _legend_3x3(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((3, 3), fill, dtype=int)
    g[1 + dr, 1 + dc] = INDICATOR
    return g


def _edge_origin(edge: str, h: int, w: int) -> tuple[int, int]:
    mid_r = (h - 3) // 2
    mid_c = (w - 3) // 2
    if edge == "T":
        return 0, mid_c
    if edge == "B":
        return h - 3, mid_c
    if edge == "L":
        return mid_r, 0
    if edge == "R":
        return mid_r, w - 3
    raise ValueError(edge)


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int], str]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    edges = rng.sample(EDGES, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []

    for edge, wall_color in zip(edges, wall_colors):
        dr, dc = rng.choice(ALL_DIRS)
        patch = _legend_3x3(wall_color, dr, dc)
        r0, c0 = _edge_origin(edge, h, w)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + 3 + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + 3 + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc), edge))
    return legend_occupied, legends


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
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
        return cand
    return None


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int], str]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wall_color: direction for wall_color, direction, _edge in legends}
    all_colors = [wall_color for wall_color, _direction, _edge in legends]

    region_colors = []
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for region_color in region_colors:
        dr, dc = color_dir[region_color]
        target_size = rng.randint(size_min, size_max)

        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        regions.append({"color": region_color, "dir": (dr, dc), "interior": blob, "walls": walls})

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


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        yellows_input = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = MOVER
    return True


def _legend_is_mid_edge_component(grid: np.ndarray, fill: int, direction: tuple[int, int], edge: str) -> bool:
    r0, c0 = _edge_origin(edge, *grid.shape)
    if not _is_mid_edge_bbox(r0, c0, *grid.shape):
        return False
    patch = grid[r0 : r0 + 3, c0 : c0 + 3]
    if patch.shape != (3, 3):
        return False
    if (patch == fill).sum() != 8 or (patch == INDICATOR).sum() != 1:
        return False
    dr, dc = direction
    return int(patch[1 + dr, 1 + dc]) == INDICATOR


def _old_corner_legend_count(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    count = 0
    for pts in components_mask(grid != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r0, c0) not in ((0, 0), (0, w - (c1 - c0 + 1)), (h - (r1 - r0 + 1), 0), (h - (r1 - r0 + 1), w - (c1 - c0 + 1))):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue
        counts: dict[int, int] = {}
        for r, c in pts:
            counts[int(grid[r, c])] = counts.get(int(grid[r, c]), 0) + 1
        if len(counts) == 3:
            count += 1
    return count


def _oracle_check(input_grid: np.ndarray, regions: list[dict], legends: list[tuple[int, tuple[int, int], str]]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == MOVER).sum() != (input_grid == MOVER).sum():
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == MOVER])):
        return False

    dirs = _legend_directions(input_grid, 0)
    if {color for color, _direction, _edge in legends} != set(dirs):
        return False
    for color, direction, edge in legends:
        if dirs.get(color) != direction:
            return False
        if not _legend_is_mid_edge_component(input_grid, color, direction, edge):
            return False
    if _old_corner_legend_count(input_grid, 0) != 0:
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
    if not _oracle_check(input_grid, regions, legends):
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
