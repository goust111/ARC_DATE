# Variant of 88e364bc
# Dimension: D9 — 滑动方向类型 (path)
# Change: 支持基数方向（上下左右）和对角线方向（4个45度方向），共8个可能方向；基数方向用5x5图例，对角线用6x7图例... → 骑士步方向（L形移动）

"""
Design analysis for the D9 knight-move variant.

a) Concrete behavior and trace
   - Visual form:
     - Grid stays 20x20 with bg=0.
     - Each legend remains a border-touching solid rectangle whose fill color C matches one wall color.
     - The variant uses a single 7x7 legend for all 8 directions. Three blue pixels form the center mark
       at row 3, cols 2..4. One red pixel sits at a knight offset from the center cell (3,3).
       The 8 allowed offsets are:
       (-2,-1), (-2,+1), (-1,-2), (-1,+2), (+1,-2), (+1,+2), (+2,-1), (+2,+1).
     - Yellow pixels still live inside enclosed background regions bounded by colored walls.
   - Trace:
     - Original code with direction (+1,+1):
       from (6,6) a yellow visits (7,7), (8,8), ... until the next diagonal cell leaves the region or is occupied.
     - Variant with knight direction (+2,+1):
       from (6,6) the yellow visits (8,7), (10,8), (12,9), ... until the next landing square is outside
       the region or already occupied by an earlier-stopped yellow.
     - Difference:
       movement is no longer a dense straight ray through adjacent cells. It is a sparse repeated jump on a
       fixed knight lattice, so the solver must preserve the exact 2:1 displacement instead of collapsing it
       to a sign pattern.

b) Complexity calibration
   - Keep exactly 8 direction categories, mirroring the old branch count.
   - Keep the legend-to-wall-color linkage unchanged so the only conceptual novelty is the motion rule.
   - Use 1-2 legends and 1-4 regions as before; this gives repeated support for a direction without flooding
     the grid with too many independent codes.
   - Require generated cases to contain at least one yellow that travels at least two knight steps so the
     repeated-jump nature is observable, not just a one-off teleport.

c) Implicit assumption review against the original main()
   - `_legend_directions`: the old `sign(centroid(indicator)-centroid(center))` was sufficient only because
     all directions were unit/cardinal/diagonal. In the variant that would destroy the 2-vs-1 distinction and
     alias a knight move to a diagonal or cardinal move. The new code keeps the full centroid delta and snaps
     it to the nearest knight vector.
   - `_slide_points`: the old loop assumed a one-cell step and had special diagonal "corner cutting" checks.
     Those assumptions fail for knight jumps. The new step only checks the next landing square, because the
     L-shaped move is modeled as a jump.
   - Sort order: "far end first" is still correct, but now the ordering key must use the full direction vector
     as a dot product so (+2,+1) and (+1,+2) are distinguishable.
   - Region detection: 4-connected passable components still work because the walls/interiors are generated in
     the same topological style as the original task.
   - Border-color inference from 4-neighbors remains valid because walls still touch region interiors
     orthogonally and are still one cell thick.

d) Placement legality
   - Legends stay buffered away from regions, so non-background border components do not merge with walls.
   - Regions are still grown first, then ring walls are painted around them, so enclosed areas remain valid
     passable components for the solver.
   - Reverse placement for yellows checks that every chosen input really slides back to the intended output
     under the new knight dynamics, and the generator rejects any case that does not contain a real slide.

e) Visual observability
   - The blue center trio plus red offset pixel visibly encodes an L shape; across a few examples the user can
     see that the displacement is not along a straight line.
   - Multi-step jumps are enforced at least once per generated grid, which exposes that the motion repeats the
     same knight offset rather than applying a one-time relocation.

f) Visual cleanliness
   - No extra markers are added inside regions, so the picture remains: legends on the outer border, colored
     walls, and yellow movers inside clean cavities.
   - Using one legend size for all knight directions reduces incidental variation unrelated to the new rule.

g) Color inferability
   - Colors keep their old roles:
       bg=0, center=1, indicator=2, mover=4, wall colors from {3,5,6,7,8,9}.
     There is no new color mapping burden; only the spatial code changes.

h) New solver assumptions that the generator must guarantee
   - Every legend's center/indicator centroid delta must be closest to exactly one knight vector.
   - Each region containing yellow must admit at least one valid reverse knight path of length >= 1.
   - At least one yellow per grid must admit a reverse path of length >= 2, otherwise the sample is rejected.
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

KNIGHT_DIRS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (20, 85),
    2: (18, 55),
    3: (15, 36),
    4: (14, 28),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 4

GEN_RETRIES = 300


def _projection(pt: tuple[int, int], dr: int, dc: int) -> int:
    return dr * int(pt[0]) + dc * int(pt[1])


def _nearest_knight_dir(vr: float, vc: float) -> tuple[int, int] | None:
    best_dir = None
    best_dist = float("inf")
    for dr, dc in KNIGHT_DIRS:
        dist = (vr - dr) ** 2 + (vc - dc) ** 2
        if dist < best_dist:
            best_dist = dist
            best_dir = (dr, dc)
    if best_dist > 0.75:
        return None
    return best_dir


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and decode them as knight directions.

    Each legend is a border-touching filled rectangle with exactly three colors:
    - dominant fill color C, reused as the wall color it controls
    - a more frequent center color
    - a less frequent indicator color

    The full centroid delta indicator-center is snapped to one of the 8 knight vectors.
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
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rect_center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tie_break = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tie_break.append((np.sum((pts_col.mean(axis=0) - rect_center) ** 2), col))
            tie_break.sort()
            center_col = tie_break[0][1]
            indicator_col = tie_break[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue

        delta = indicator.mean(axis=0) - center.mean(axis=0)
        knight_dir = _nearest_knight_dir(float(delta[0]), float(delta[1]))
        if knight_dir is None:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), knight_dir)

    return {k: v for k, (_, v) in best.items()}


def _slide_points(region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set or (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions according to knight-direction border legends.

    The region/wall parsing is unchanged from the original task. The only rule change is the motion model:
    once a region's wall color is matched to a legend, every yellow in that region repeatedly applies the
    same knight offset until the next landing cell leaves the region or is already occupied by a previously
    stopped yellow from the same region.
    """
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
            for nr, nc in neighbors4(r, c):
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

        for r, c in movers:
            out[r, c] = bg

        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover

    return out


def _legend_7x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((7, 7), fill, dtype=int)
    g[3, 2:5] = 1
    g[3 + dr, 3 + dc] = 2
    return g


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


def _grow_blob(canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random) -> set[tuple[int, int]] | None:
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


def _jump_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    return (r + dr, c + dc) in region_set


def _forward_slide(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int, blocked: set[tuple[int, int]]) -> tuple[int, int]:
    r, c = start
    while True:
        if not _jump_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    directions = rng.sample(KNIGHT_DIRS, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []

    for corner, wall_color, (dr, dc) in zip(corners, wall_colors, directions):
        patch = _legend_7x7(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))

    return legend_occupied, legends


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
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wall_color: direction for wall_color, direction in legends}
    all_colors = [wall_color for wall_color, _ in legends]

    region_colors = []
    for wall_color in all_colors:
        region_colors.extend([wall_color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for wall_color in region_colors:
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
        regions.append(
            {
                "color": wall_color,
                "dir": color_dir[wall_color],
                "interior": blob,
                "walls": walls,
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


def _candidate_output_groups(blob: set[tuple[int, int]], dr: int, dc: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    frontier = [pt for pt in blob if (pt[0] + dr, pt[1] + dc) not in blob]
    if not frontier:
        return [], []
    frontier_sorted = sorted(frontier, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    best_key = _projection(frontier_sorted[0], dr, dc)
    top_tier = [pt for pt in frontier_sorted if _projection(pt, dr, dc) == best_key]
    return frontier_sorted, top_tier


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    min_steps: int,
    rng: random.Random,
) -> tuple[tuple[int, int], int] | None:
    max_steps = MAX_SLIDE_STEPS
    preferred = list(range(min_steps, max_steps + 1))
    rng.shuffle(preferred)
    preferred.sort(reverse=True)
    for steps in preferred:
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c), steps
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> tuple[bool, dict[str, object]]:
    step_counts: list[int] = []
    per_region_counts: list[int] = []
    has_long_jump = False

    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)
        frontier_sorted, top_tier = _candidate_output_groups(blob, dr, dc)
        if not frontier_sorted:
            return False, {}

        tried = False
        placed = False
        first_pool = top_tier + frontier_sorted[: min(6, len(frontier_sorted))]
        first_pool = list(dict.fromkeys(first_pool))

        for _ in range(20):
            first = first_pool[rng.randrange(len(first_pool))]
            output_positions = [first]
            if n_yellows == 2:
                second_candidates = []
                prev = (first[0] - dr, first[1] - dc)
                if prev in blob:
                    second_candidates.append(prev)
                second_candidates.extend(pt for pt in top_tier if pt != first)
                second_candidates = list(dict.fromkeys(second_candidates))
                if not second_candidates:
                    continue
                output_positions.append(second_candidates[rng.randrange(len(second_candidates))])

            out_sorted = sorted(output_positions, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
            already_stopped: set[tuple[int, int]] = set()
            input_positions = []
            local_steps = []
            tried = True
            local_has_long = has_long_jump

            for out_pos in out_sorted:
                preferred_min = 2 if not local_has_long else 1
                candidate = _find_reverse_slide(
                    blob,
                    out_pos,
                    dr,
                    dc,
                    already_stopped,
                    input_positions,
                    preferred_min,
                    rng,
                )
                if candidate is None and preferred_min > 1:
                    candidate = _find_reverse_slide(
                        blob,
                        out_pos,
                        dr,
                        dc,
                        already_stopped,
                        input_positions,
                        1,
                        rng,
                    )
                if candidate is None:
                    break
                start, steps = candidate
                input_positions.append(start)
                local_steps.append(steps)
                already_stopped.add(out_pos)
                if steps >= 2:
                    local_has_long = True
            else:
                for r, c in input_positions:
                    grid[r, c] = 4
                step_counts.extend(local_steps)
                per_region_counts.append(len(input_positions))
                has_long_jump = local_has_long
                placed = True
                break

        if tried and not placed:
            return False, {}
        if not tried:
            return False, {}

    return has_long_jump, {"step_counts": step_counts, "per_region_counts": per_region_counts}


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == 4).sum()) != int((input_grid == 4).sum()):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


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
    has_long_jump, placement_stats = _place_yellows(rng, regions, input_grid)
    if not placement_stats or not has_long_jump:
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
