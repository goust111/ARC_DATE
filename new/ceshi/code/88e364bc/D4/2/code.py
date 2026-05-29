# Variant of 88e364bc
# Dimension: D4 — 滑动终止条件 (dst)
# Change: boundary_collision——黄色像素滑到区域边界（墙壁或区域外）时停止；多个黄色像素在同区域内会堆叠... → 遇到另一个黄色像素时不堆叠而是停在前一格

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior trace
#    Unchanged visible objects:
#    - 20x20 black background.
#    - 1-2 border legends whose fill color matches a wall color and whose red-vs-blue offset encodes a direction.
#    - Irregular colored wall loops enclosing passable regions.
#    - Yellow pixels inside those regions.
#    Core D4 change:
#    - Old task: if two yellows share the same slide lane, the nearer one stops adjacent to the farther one,
#      e.g. "... Y . . -> -> wall" becomes final "... . Y Y wall".
#    - New task: the farther yellow still reaches the wall, but every later yellow must leave one empty cell
#      before a previously stopped yellow, e.g. final "... Y . Y wall".
#    Small traced example on a 1-cell-high corridor moving right:
#      region cells = c1..c7, wall after c7
#      input yellows at c2 and c3
#      old solver: c3 -> c7, then c2 -> c6  => [c6, c7]
#      new solver: c3 -> c7, c2 cannot enter c6 because c6 is the reserved gap behind c7 => c2 stops at c5
#      difference = one visible empty spacer cell c6.
#    The affected pipeline stage is exactly the intra-region slide loop plus the generator's reverse-slide logic.
#
# b) Complexity calibration
#    The variant is intentionally narrow: legends, colors, wall generation, and region detection stay unchanged.
#    To make the new rule learnable from 2-4 samples, the generator guarantees at least one region per grid where
#    two yellows interact in the same slide lane, so the empty spacer is visibly required by the rule.
#
# c) Hidden-assumption review of main()
#    - `_legend_directions`: depends only on legend rectangles; D4 does not change it.
#    - `passable = (g == bg) | (g == mover)`: still valid because regions are still defined by bg/yellow cells.
#    - Iterating passable connected components: still valid; yellows remain inside enclosed passable regions.
#    - Border-color majority detection: unchanged.
#    - Old `_slide_points` assumed "next occupied cell means stop adjacent". That assumption is invalid now.
#      We replace it with "next cell cannot be a stopped yellow nor the cell immediately behind one".
#    - Clearing old movers before repainting remains safe because all destinations are computed from the original
#      mover list, and output still uses the same yellow color.
#    New input assumption introduced by this variant:
#    - A valid input should not start a yellow on the reserved spacer cell directly behind a farther yellow in the
#      same lane, because no legal 1-gap terminal can be inferred from such a state. The generator enforces this.
#
# d) Placement legality
#    The original region/wall/legend placement remains legal. The new generator only changes yellow placement:
#    - it chooses at least one output pair whose second yellow is terminal under the new solver,
#      but would move farther under the old contiguous-stacking solver;
#    - reverse-slid input yellows are checked against region bounds, diagonal no-corner-cut rules, reserved spacer
#      cells, and already assigned yellow inputs.
#
# e) Visual observability
#    The human-visible cue is not a hidden numeric sort; it is an empty black cell separating two final yellows
#    inside one region while both still move in the same legend-indicated direction. This is directly observable.
#
# f) Visual cleanliness
#    No extra markers are introduced. The output remains sparse: colored walls, one legend per wall color, and a
#    small number of yellows. The only new visual feature is a single empty gap in a slide lane.
#
# g) Color reasoning
#    Color semantics stay exactly the same as the base task: wall color selects legend; yellow is the mover.
#    No new color mapping is introduced.


# ---------------- Solver helpers ----------------

def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
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


def _project(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _blocked_cells(
    region_set: set[tuple[int, int]], stopped: set[tuple[int, int]], dr: int, dc: int, *, leave_gap: bool
) -> set[tuple[int, int]]:
    blocked = set(stopped)
    if leave_gap:
        for r, c in stopped:
            br, bc = r - dr, c - dc
            if (br, bc) in region_set:
                blocked.add((br, bc))
    return blocked


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []

    points_sorted = sorted(points, key=lambda p: (_project(p, dr, dc), p[0], p[1]), reverse=True)
    blocked: set[tuple[int, int]] = set()
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
            if (nr, nc) in blocked:
                break
            r, c = nr, nc
        out_pts.append((r, c))
        blocked.add((r, c))
        br, bc = r - dr, c - dc
        if (br, bc) in region_set:
            blocked.add((br, bc))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
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


# ---------------- Generator ----------------

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

MAX_SLIDE_STEPS = 5
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


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    stopped: set[tuple[int, int]],
    *,
    leave_gap: bool,
) -> tuple[int, int]:
    r, c = start
    blocked = _blocked_cells(region_set, stopped, dr, dc, leave_gap=leave_gap)
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
        regions.append({"color": rc, "dir": (dr, dc), "interior": blob, "walls": walls})

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


def _frontier_cells(blob: set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    cells = list(blob)
    if not cells:
        return []
    best_key = max(_project(p, dr, dc) for p in cells)
    return [p for p in cells if _project(p, dr, dc) == best_key]


def _pick_single_output(blob: set[tuple[int, int]], dr: int, dc: int, rng: random.Random) -> tuple[int, int] | None:
    frontier = _frontier_cells(blob, dr, dc)
    if not frontier:
        return None
    return frontier[rng.randrange(len(frontier))]


def _pick_interacting_outputs(
    blob: set[tuple[int, int]], dr: int, dc: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    frontier = _frontier_cells(blob, dr, dc)
    rng.shuffle(frontier)
    for first in frontier:
        candidates = []
        for second in blob:
            if second == first:
                continue
            if _forward_slide(blob, first, dr, dc, set(), leave_gap=True) != first:
                continue
            if _forward_slide(blob, second, dr, dc, {first}, leave_gap=True) != second:
                continue
            if _forward_slide(blob, second, dr, dc, {first}, leave_gap=False) == second:
                continue
            candidates.append(second)
        if candidates:
            candidates.sort(key=lambda p: (_project(p, dr, dc), p[0], p[1]), reverse=True)
            top_key = _project(candidates[0], dr, dc)
            top = [p for p in candidates if _project(p, dr, dc) == top_key]
            return [first, top[rng.randrange(len(top))]]
    return None


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
    blocked = _blocked_cells(blob, already_stopped, dr, dc, leave_gap=True)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in blocked or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped, leave_gap=True) == out_pos:
            return (r, c)
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    interactive_options: dict[int, list[tuple[int, int]]] = {}
    for idx, reg in enumerate(regions):
        pair = _pick_interacting_outputs(reg["interior"], reg["dir"][0], reg["dir"][1], rng)
        if pair is not None:
            interactive_options[idx] = pair
    if not interactive_options:
        return False

    focus_idx = rng.choice(list(interactive_options.keys()))

    for idx, reg in enumerate(regions):
        blob = reg["interior"]
        dr, dc = reg["dir"]
        if idx == focus_idx:
            output_positions = interactive_options[idx]
        else:
            single = _pick_single_output(blob, dr, dc, rng)
            if single is None:
                return False
            output_positions = [single]

        out_sorted = sorted(output_positions, key=lambda p: (_project(p, dr, dc), p[0], p[1]), reverse=True)
        already_stopped: set[tuple[int, int]] = set()
        yellows_input: list[tuple[int, int]] = []
        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
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
