# Variant of 88e364bc
# Dimension: D2 — 图例与区域的匹配机制 (symbol_instructions)
# Change: legend_lookup——图例填充色C与区域边界墙壁的多数颜色匹配，每种墙壁颜色恰好对应一个图例... → 图例位置编码匹配象限

"""
Design analysis for the D2 variant
==================================

spec:
  preserved_structure:
    grid: 20x20, bg=0
    legends: border-touching 5x5 or 6x7 filled rectangles with a 3-color arrow code
    regions: enclosed passable cavities bounded by colored walls
    movers: yellow(4) pixels slide and stack against the far wall
    direction_encoding: still read from center->indicator centroid offset inside each legend
  changed_dimension:
    from: region border majority wall color -> legend fill color lookup
    to: region geometric quadrant -> corner legend lookup
  concrete_behavior:
    - A legend in TL/TR/BL/BR controls every mover-containing region whose interior lies in that same
      geometric quadrant of the 20x20 grid.
    - Region wall colors are no longer the matching key. Generator deliberately chooses legend fill
      colors from a palette disjoint from wall colors, so color matching is visibly wrong.
    - At least one wall color is reused in two different quadrants with different directions. This
      makes the new rule observable: same-colored walls can induce different motions because location,
      not color, is now the key.
  behavior_trace:
    original:
      - Suppose a region is bounded mostly by wall color 5.
      - Solver counts adjacent wall colors, gets 5, looks up legend fill color 5, then slides by that
        legend's arrow.
    variant:
      - Two regions can both be bounded by wall color 5.
      - One lies in TL and another in BR.
      - TL uses the TL corner legend's arrow; BR uses the BR corner legend's arrow.
      - Therefore identical wall colors can yield different outputs if their quadrants differ.
  complexity_calibration:
    - Active legends: 2-4 corners, never just 1, because one legend would make the quadrant rule weak.
    - Regions: exactly 1 enclosed region per active quadrant. This keeps the layout readable inside the
      tighter quadrant-constrained space while still showing multiple quadrant-controlled cases.
    - Directions: at least two distinct directions per grid, so quadrant choice matters.
    - Wall colors: two colors reused across quadrants; at least one repeated-color pair has different
      directions across quadrants, which blocks accidental regression to the original color-lookup rule.
  implicit-assumption-audit_for_main:
    - _legend_directions_by_quadrant assumes legends are the only border-touching non-background
      rectangles with exactly 3 colors. Generator guarantees walls never touch the border.
    - _quadrant_from_points assumes each enclosed region belongs to one geometric quadrant. Generator
      grows each interior blob entirely within that quadrant's interior window.
    - main no longer reads border colors. Any code that previously depended on majority adjacent wall
      color is removed, because that premise is false in this variant.
    - If a region exists in a quadrant without a legend, movement would be undefined. Generator ensures
      every region is created only in an active-legend quadrant.
  placement_legality:
    - Legends stay in exact corners with a 1-cell reservation buffer.
    - Region interiors are grown on a quadrant-specific canvas excluding legend buffers and the outer
      border; resulting walls are checked to avoid border overlap and other occupied cells.
    - Reverse-slid yellow placement is retained and validated by running the solver.
  visual_observability:
    - Same wall color across different quadrants but different motion directions is the primary visual
      cue.
    - Legend fill colors do not match local walls, preventing a human from reasonably inferring the
      original color-based mechanism from examples.
  cleanliness:
    - One region per active quadrant avoids crowding in the reduced free space near corner legends.
    - Walls and legends remain visually distinct because they use disjoint colors.
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]  # non-{0 bg, 1 center, 2 indicator, 4 yellow}
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

# Smaller than the source task because each region is constrained to its controlling quadrant.
REGION_SIZE_BOUNDS = {
    2: (10, 18),
    3: (8, 14),
    4: (7, 12),
}
BLOB_GROW_RETRIES = 40
YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 400


def _corner_from_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> str | None:
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == w - 1:
        return "TR"
    if r1 == h - 1 and c0 == 0:
        return "BL"
    if r1 == h - 1 and c1 == w - 1:
        return "BR"
    return None


def _quadrant_from_points(pts: np.ndarray | list[tuple[int, int]], h: int, w: int) -> str:
    arr = np.asarray(pts, dtype=float)
    cr, cc = arr.mean(axis=0)
    return ("T" if cr < h / 2 else "B") + ("L" if cc < w / 2 else "R")


def _legend_directions_by_quadrant(g: np.ndarray, bg: int) -> dict[str, tuple[int, int]]:
    """
    Detect border legends exactly as in the source task, but key them by corner quadrant instead of
    their dominant fill color.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[str, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        corner = _corner_from_bbox(r0, c0, r1, c1, h, w)
        if corner is None:
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
            # Tie-break from the source task: the "center" color sits closer to the rectangle center.
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
            center_col = others[1][1]

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

        prev = best.get(corner)
        if prev is None or len(pts) > prev[0]:
            best[corner] = (len(pts), (dr, dc))

    return {corner: direction for corner, (_, direction) in best.items()}


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
    Slide yellow pixels inside enclosed regions.

    Source task key:
      region border color -> legend fill color -> direction

    Variant key:
      region quadrant -> same-corner legend -> direction
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions_by_quadrant(g, bg)
    assert dirs, "variant expects at least one corner legend"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        quadrant = _quadrant_from_points(pts, h, w)
        if quadrant not in dirs:
            continue
        dr, dc = dirs[quadrant]

        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in movers:
            out[r, c] = bg

        new_pts = _slide_points(pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = MOVER

    return out


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


def _random_direction(rng: random.Random) -> tuple[int, int]:
    if rng.random() < CARDINAL_PROB:
        return rng.choice(CARDINAL_DIRS)
    return rng.choice(DIAGONAL_DIRS)


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    n_legends = rng.randint(2, 4)
    corners = rng.sample(CORNERS, n_legends)
    fills = rng.sample(WALL_COLOR_CHOICES, n_legends)

    while True:
        dirs = [_random_direction(rng) for _ in range(n_legends)]
        if len(set(dirs)) >= 2:
            break

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, fill, (dr, dc) in zip(corners, fills, dirs):
        patch = _legend_5x5(fill, dr, dc) if (dr, dc) in CARDINAL_DIRS else _legend_6x7(fill, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)

        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"corner": corner, "fill": fill, "dir": (dr, dc)})

    return legend_occupied, legends


def _assign_wall_colors(rng: random.Random, legends: list[dict]) -> dict[str, int]:
    legend_fills = {int(leg["fill"]) for leg in legends}
    wall_palette = rng.sample([c for c in WALL_COLOR_CHOICES if c not in legend_fills], 2)
    shared_color, alt_color = wall_palette

    corners = [str(leg["corner"]) for leg in legends]
    dirs = {str(leg["corner"]): tuple(leg["dir"]) for leg in legends}
    differing_pairs = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1 :] if dirs[a] != dirs[b]]
    pair = rng.choice(differing_pairs)

    wall_by_corner = {pair[0]: shared_color, pair[1]: shared_color}
    remaining = [c for c in corners if c not in pair]
    rng.shuffle(remaining)
    if remaining:
        wall_by_corner[remaining[0]] = alt_color
    for corner in remaining[1:]:
        wall_by_corner[corner] = rng.choice(wall_palette)
    return wall_by_corner


def _quadrant_canvas(corner: str, occupied: np.ndarray) -> np.ndarray:
    h, w = occupied.shape
    canvas = np.zeros((h, w), dtype=bool)
    if corner == "TL":
        canvas[1:10, 1:10] = True
    elif corner == "TR":
        canvas[1:10, 10:19] = True
    elif corner == "BL":
        canvas[10:19, 1:10] = True
    elif corner == "BR":
        canvas[10:19, 10:19] = True
    else:
        raise ValueError(corner)
    canvas &= ~occupied
    return canvas


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


def _grow_regions(rng: random.Random, legends: list[dict], legend_occupied: np.ndarray) -> list[dict] | None:
    h, w = legend_occupied.shape
    occupied = legend_occupied.copy()
    wall_by_corner = _assign_wall_colors(rng, legends)
    size_min, size_max = REGION_SIZE_BOUNDS[len(legends)]
    regions = []

    # Harder top quadrants first when their legends are tall.
    legend_order = sorted(legends, key=lambda leg: (0 if leg["corner"] in ("TL", "TR") else 1, str(leg["corner"])))
    for leg in legend_order:
        corner = str(leg["corner"])
        canvas = _quadrant_canvas(corner, occupied)
        if canvas.sum() < size_min:
            return None

        target_size = rng.randint(size_min, size_max)
        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            return None
        if _quadrant_from_points(list(blob), h, w) != corner:
            return None

        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "corner": corner,
                "color": wall_by_corner[corner],
                "dir": tuple(leg["dir"]),
                "interior": blob,
                "walls": walls,
            }
        )
        for r, c in blob | walls:
            occupied[r, c] = True

    shared_color_groups: dict[int, list[tuple[int, int]]] = {}
    for reg in regions:
        shared_color_groups.setdefault(int(reg["color"]), []).append(tuple(reg["dir"]))
    if not any(len(set(dirs)) >= 2 for dirs in shared_color_groups.values() if len(dirs) >= 2):
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
    existing_inputs: list,
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

    # Variant-specific sanity: same-colored regions in different quadrants can move differently.
    by_color: dict[int, list[tuple[int, int]]] = {}
    for reg in regions:
        by_color.setdefault(int(reg["color"]), []).append(tuple(reg["dir"]))
    return any(len(set(dirs)) >= 2 for dirs in by_color.values() if len(dirs) >= 2)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

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
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
