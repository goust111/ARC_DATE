import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4


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
        # Legends are filled rectangles on the border.
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
        others.sort()  # ascending by count
        if others[0][0] == others[1][0]:
            # Tie: choose center as the color closer to the rectangle center.
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


def _slide_points(region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    if not points:
        return []
    # Process from the far end (closest to destination boundary) inward.
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            # For diagonal steps, disallow "corner cutting": both orthogonal intermediates must be inside the region.
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
    Slide yellow pixels inside walled regions in the direction dictated by border-legend arrows.

    Input structure:
    - bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
    - Border legends: small rectangles on the grid edge whose fill color C matches a region's border color. Each legend contains a center dot and an indicator dot encoding a direction.

    Rule:
    - For each enclosed region, identify its border color C and look up the legend with fill color C.
    - The direction from the legend's center dot to its indicator dot gives the slide direction.
    - All yellow pixels in that region slide in that direction, stacking against the far wall.

    Key insight: Legend fill color links to region border color; the center→indicator offset encodes the movement vector.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4  # yellow in ARC
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    # Find enclosed passable regions (not connected to the outer border).
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue  # outside background

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        # Determine this region's border color by counting adjacent wall colors.
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

        # Clear old movers.
        for r, c in movers:
            out[r, c] = bg

        # Slide and repaint.
        new_pts = _slide_points(pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover

    return out

# --------------- generator (claude-att1) ---------------

"""Generator for task 88e364bc (slide yellow in walled regions by legend arrows)."""
import random
import numpy as np

from arc_utils import label_mask, neighbors4, paste

# ── Configuration ─────────────────────────────────────────────────
GRID_H, GRID_W = 20, 20

# Legend
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ['TL', 'TR', 'BL', 'BR']
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]  # non-{0 bg, 1 center, 2 indicator, 4 yellow}
CARDINAL_PROB = 0.5                        # probability each legend is cardinal (vs diagonal)
LEGEND_BUFFER = 1                          # cells of padding around legend patches

# Regions
REGIONS_PER_COLOR = (1, 3)                 # min, max regions per wall color
MAX_TOTAL_REGIONS = 4                      # hard cap to fit in 20x20
REGION_SIZE_BOUNDS = {                     # {n_regions: (min_size, max_size)}
    1: (15, 85), 2: (15, 55), 3: (12, 35), 4: (12, 25),
}
BLOB_GROW_RETRIES = 25                     # attempts to grow a valid blob per region

# Yellows
YELLOWS_PER_REGION = [1, 2]                # choices per region
MAX_SLIDE_STEPS = 5                        # reverse-slide distance range [1, MAX]

# Outer loop
GEN_RETRIES = 300                          # max attempts in generate_input()


# ── Legend patches ────────────────────────────────────────────────

def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    """5x5 cardinal legend. Canonical direction: (0, +1) right."""
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1; g[1, 2] = 1; g[1, 3] = 2
    g[3, 1] = 1; g[3, 2] = 1; g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    """6x7 diagonal legend. Canonical direction: (+1, +1) down-right."""
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1; g[2, 4] = 1
    g[3, 2] = 1; g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1): return g
    if (dr, dc) == (-1, 1): return np.flipud(g)
    if (dr, dc) == (1, -1): return np.fliplr(g)
    if (dr, dc) == (-1, -1): return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    """Top-left coordinate for a patch placed at the given grid corner."""
    if corner == 'TL': return 0, 0
    if corner == 'TR': return 0, w - pw
    if corner == 'BL': return h - ph, 0
    if corner == 'BR': return h - ph, w - pw
    raise ValueError(corner)


# ── Blob growing / topology ──────────────────────────────────────

def _grow_blob(canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int,
               rng: random.Random) -> set[tuple[int, int]] | None:
    """Random-walk growth of a 4-connected blob on free cells."""
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier = []
    def push_neighbors(r, c):
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
    """Cells 4-adjacent to blob but not in blob."""
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    """Return blob with interior holes absorbed.

    Barrier = blob union its wall ring. Any non-barrier cell unreachable from the grid
    border through non-barrier cells is trapped and gets absorbed into the blob.
    """
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    # Flood-fill from border through non-barrier cells.
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


# ── Slide helpers ─────────────────────────────────────────────────

def _slide_step_valid(region_set: set, r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(region_set: set, start: tuple[int, int], dr: int, dc: int,
                   blocked: set) -> tuple[int, int]:
    """Simulate solver's slide: keep moving (dr, dc) until hitting a wall or blocker."""
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


# ── Generator phases ─────────────────────────────────────────────

def _place_legends(rng: random.Random, grid: np.ndarray,
                   ) -> tuple[np.ndarray, list[tuple[int, tuple[int,int]]]]:
    """Place 1-2 legend patches at grid corners.

    Returns (legend_occupied mask, list of (wall_color, direction) pairs).
    """
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
        # Reserve legend cells + buffer.
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))

    return legend_occupied, legends


def _grow_regions(rng: random.Random, legends: list[tuple[int, tuple[int,int]]],
                  legend_occupied: np.ndarray,
                  ) -> list[dict] | None:
    """Grow irregular closed regions, one per assigned wall color.

    Returns list of region dicts or None on failure.
    """
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    # Assign 1-3 regions per wall color, cap total.
    region_colors = []
    for c in all_colors:
        region_colors.extend([c] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
        dr, dc = color_dir[rc]
        target_size = rng.randint(size_min, size_max)

        # Build canvas: interior cells (not border) that are unoccupied.
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False; canvas[-1, :] = False
        canvas[:, 0] = False; canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        regions.append({'color': rc, 'dir': (dr, dc), 'interior': blob, 'walls': walls})

        # Reserve blob + walls + 1-cell buffer around walls.
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
    # Every legend color must have at least one region.
    if {reg['color'] for reg in regions} != set(all_colors):
        return None
    return regions


def _try_grow_valid_blob(rng: random.Random, canvas: np.ndarray,
                         occupied: np.ndarray, target_size: int,
                         h: int, w: int) -> set[tuple[int,int]] | None:
    """Attempt up to BLOB_GROW_RETRIES random seeds to grow a valid blob."""
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        # Fill interior holes until stable.
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        # Absorbed cells must stay within canvas.
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        # Walls must not overlap occupied cells or the grid border.
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1)
               for wr, wc in walls):
            continue
        return cand
    return None


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    """Check that every enclosed bg component matches exactly one region interior."""
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg['interior'] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _place_yellows(rng: random.Random, regions: list[dict],
                   grid: np.ndarray) -> bool:
    """Choose yellow output positions and reverse-slide to derive input positions.

    Modifies grid in-place by painting yellow(4) at input positions.
    Returns True on success.
    """
    for reg in regions:
        blob = reg['interior']
        dr, dc = reg['dir']
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        # Cells sorted by projection onto slide direction (farthest first).
        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        # Pick output positions: first at the farthest tier, second one step behind or at tier1.
        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        # Reverse-slide: process in solver order (far-to-near).
        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        yellows_input = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped,
                                            yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = 4
    return True


def _find_reverse_slide(blob: set, out_pos: tuple[int,int], dr: int, dc: int,
                        already_stopped: set, existing_inputs: list,
                        rng: random.Random) -> tuple[int,int] | None:
    """Walk backwards from out_pos to find a valid input position 1..MAX_SLIDE_STEPS away."""
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


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    """Run the solver on input_grid and verify correctness."""
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    # Non-yellow, non-bg cells must be identical.
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    # Yellow count must be preserved.
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False
    # All output yellows must lie inside known region interiors.
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg['interior']:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


# ── Main generator ────────────────────────────────────────────────

def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    # Paint walls.
    for reg in regions:
        for r, c in reg['walls']:
            grid[r, c] = reg['color']

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
