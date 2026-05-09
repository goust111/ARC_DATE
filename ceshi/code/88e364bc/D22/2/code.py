# Variant of 88e364bc
# Dimension: D22 — 移动像素颜色与角色 (src)
# Change: 移动像素统一为黄色（颜色4），所有黄色像素行为相同... → 移动像素为红色（与图例指示色相同）

"""
design_analysis:
  variant_goal:
    core_change: movable pixels use red(2), the same color as legend indicator pixels.
    preserved_structure:
      - legends stay border-touching filled rectangles
      - legend fill color still matches the wall color it controls
      - center->indicator centroid offset still encodes the slide direction
      - enclosed-region slide physics and stacking remain unchanged
  concretization_and_trace:
    input_view:
      - corner legends still contain blue center dots and red indicator dots
      - enclosed regions now contain red movable pixels instead of yellow ones
      - therefore the grid contains two visually different roles for color 2:
        border-legend indicator pixels and interior movable pixels
    flow_example:
      original_logic:
        - passable = bg or yellow
        - legend reds are never considered movable because mover color is 4
      variant_logic:
        - passable would naively become bg or red
        - this would mix legend indicator reds into the mover search space
        - fix: detect legend rectangles first, build a legend mask, and exclude legend-mask red cells from movable/passable cells
      concrete_example:
        - a top-left legend with fill 5 contains two red indicator pixels
        - a closed wall-5 region contains two red movers
        - solver must read both kinds of red for different reasons:
          legend red => decode direction
          enclosed red => slide inside the matching region
  complexity_calibration:
    - keep only one source of ambiguity: red appears in two roles
    - do not add extra remapping, extra legend colors, or asymmetric wall rules
    - keep 1-2 legends and 1-2 movers per region so the ambiguity is repeated within each sample
    - remove yellow entirely from generated grids so the D22 change is visually obvious
  implicit_assumptions_review:
    - `_legend_directions` in the original code assumed mover color differs from indicator color.
      This is no longer true, so legend cells must be tracked explicitly.
    - `passable = (g == bg) | (g == mover)` was harmless before because no structural cell had color 4.
      Under D22 that would include legend indicator cells, so passable must exclude legend-mask reds.
    - `movers = ... if g[r,c] == mover` also needs the same exclusion.
    - generator oracle logic originally ignored all cells of mover color when checking structure.
      That was safe for yellow, but would wrongly ignore legend indicator pixels after D22.
      The new oracle instead requires every cell outside known region interiors to stay unchanged.
  placement_legality:
    - legends are still pasted first, with a 1-cell reserved buffer
    - enclosed regions are still grown only in free interior canvas cells
    - movers are still reverse-slid from valid output positions, so every mover travels at least one step
    - because wall colors exclude red, any red cell outside a legend must be a mover candidate by construction
  visual_observability:
    - humans can now see red both in legends and inside regions
    - the consistent positional split makes the rule inferable:
      border red inside a filled corner block acts as a directional cue,
      enclosed red inside a walled cavity is what moves
    - multiple samples reinforce that only interior red changes between input and output
  cleanliness:
    - no cell carries more than one role at once
    - legends remain isolated at corners, while movers remain inside closed cavities
    - the generator keeps buffers so border clues and moving objects do not visually collide
  solver_generator_contract:
    solver_assumptions:
      - legend components are border-touching filled rectangles with exactly 3 colors
      - only enclosed non-legend red cells are movable
      - wall colors are never red
    generator_guarantees:
      - every sample has at least one legend
      - all movable red cells lie inside enclosed regions
      - no non-region red cells exist except legend indicators
      - reverse-slide distance is at least 1, so red movers actually move
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _detect_legends(g: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], np.ndarray]:
    """
    Detect border legends and return both direction mapping and the occupied legend mask.

    A legend is a border-touching filled rectangle whose majority color is its fill color `C`.
    The other two colors are the center color and the indicator color. Their centroid offset
    encodes the slide direction for regions bounded by color `C`.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int], list[tuple[int, int]]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        coords = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(coords):
            continue

        counts: dict[int, int] = {}
        for r, c in coords:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        others.sort()
        if len(others) != 2:
            continue

        if others[0][0] == others[1][0]:
            rect_center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            centerish: list[tuple[float, int]] = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in coords if int(g[r, c]) == col], dtype=float)
                centerish.append((float(np.sum((pts_col.mean(axis=0) - rect_center) ** 2)), col))
            centerish.sort()
            center_col = centerish[0][1]
            indicator_col = centerish[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(r, c) for r, c in coords if int(g[r, c]) == center_col]
        indicator = [(r, c) for r, c in coords if int(g[r, c]) == indicator_col]
        if not center or not indicator:
            continue

        center_centroid = np.mean(np.array(center, dtype=float), axis=0)
        indicator_centroid = np.mean(np.array(indicator, dtype=float), axis=0)
        vr, vc = (indicator_centroid - center_centroid).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(coords) > prev[0]:
            best[fill_col] = (len(coords), (dr, dc), coords)

    legend_mask = np.zeros_like(g, dtype=bool)
    for _, (_, _, coords) in best.items():
        for r, c in coords:
            legend_mask[r, c] = True

    dirs = {fill_col: payload[1] for fill_col, payload in best.items()}
    return dirs, legend_mask


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
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
    Slide red pixels inside walled regions in the direction dictated by border-legend arrows.

    The same color 2 now plays two roles:
    - border-legend indicator pixels, which define direction
    - enclosed red movers, which slide inside matching wall-bounded regions

    The separation is spatial rather than chromatic: only non-legend red cells inside enclosed
    passable regions are movable.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 2
    dirs, legend_mask = _detect_legends(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | ((g == mover) & (~legend_mask))
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [
            (int(r), int(c))
            for r, c in pts
            if int(g[r, c]) == mover and not bool(legend_mask[int(r), int(c)])
        ]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
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


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
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

MOVERS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5

GEN_RETRIES = 300


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    """5x5 cardinal legend. Canonical direction: right."""
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    """6x7 diagonal legend. Canonical direction: down-right."""
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


def _grow_blob(
    canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random
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


def _slide_step_valid(region_set: set, r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set, start: tuple[int, int], dr: int, dc: int, blocked: set
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


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []

    for corner, wc in zip(corners, wall_colors):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wc, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))

    return legend_occupied, legends


def _try_grow_valid_blob(
    rng: random.Random, canvas: np.ndarray, occupied: np.ndarray, target_size: int, h: int, w: int
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
    rng: random.Random, legends: list[tuple[int, tuple[int, int]]], legend_occupied: np.ndarray
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

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
    blob: set,
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set,
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


def _place_movers(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    """Choose red output positions and reverse-slide to derive input positions."""
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_movers = rng.choice(MOVERS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_movers == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        mover_inputs = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, mover_inputs, rng)
            if candidate is None:
                return False
            mover_inputs.append(candidate)
            already_stopped.add(out_pos)

        for r, c in mover_inputs:
            grid[r, c] = 2
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True

    if not np.array_equal(solver_out[~regions_mask], input_grid[~regions_mask]):
        return False
    if int((solver_out[regions_mask] == 2).sum()) != int((input_grid[regions_mask] == 2).sum()):
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
    if not _place_movers(rng, regions, input_grid):
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
