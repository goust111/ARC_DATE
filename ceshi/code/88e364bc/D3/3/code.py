# Variant of 88e364bc
# Dimension: D3 — 滑动路径类型 (path)
# Change: directional_movement——黄色像素沿图例指定的固定方向（基数或对角线）直线滑动，遇到区域边界或已占据位... → 反射路径（遇墙反弹）

import random
from dataclasses import dataclass

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis for the D3 reflection-path variant
# --------------------------------------------------
# spec_variant:
#   preserved_structure:
#     grid: 20x20, bg=0
#     legends:
#       count: 1-2 at corners
#       role: legend fill color still selects the wall color / region family
#       direction: sign(centroid(indicator) - centroid(center)) still provides the initial movement vector
#     regions:
#       topology: enclosed irregular rooms formed by 1-cell-thick colored walls
#       passable: only bg/yellow cells inside a bounded room
#   core_change:
#     original_behavior:
#       - each yellow moved in a single fixed direction until the next step would leave the room
#       - multiple yellows stacked because already-settled yellows blocked later ones
#     new_behavior:
#       - each yellow starts with the legend direction but now has a fixed energy budget of REFLECT_STEPS moves
#       - movement is simulated step-by-step
#       - when the next step hits a wall, the direction reflects like a ray:
#         - cardinal step: flip the moving axis
#         - diagonal step:
#           - if the forward diagonal cell is blocked but both orthogonal support cells exist, treat it as a corner hit and flip both axes
#           - if only the row-side support is blocked, flip dr
#           - if only the col-side support is blocked, flip dc
#       - occupied yellow endpoints are blockers, not mirrors
#       - if a yellow reflects but still cannot take the reflected step, it stops immediately
#   concrete_trace:
#     - suppose a yellow starts at (6,6) with direction (+1,+1) in a room whose southeast corner is close
#     - original task: it would slide straight to the southeast-most reachable cell and stop there
#     - variant: it moves cell-by-cell; on touching the southeast wall it bounces to (-1,-1) and spends the remaining budget moving back across the room
#     - visible effect: output yellows are no longer always packed against the far wall; many end mid-room after one or more rebounds
#   complexity_calibration:
#     - legends are unchanged so the task still has one stable control signal per wall color
#     - the generator forces reflected examples where at least one bounce happens and the reflected endpoint differs from the original straight-slide endpoint
#     - energy is fixed globally, avoiding a new per-instance hidden parameter
#   main_assumptions_review:
#     - legend parsing still assumes border-touching solid rectangles with exactly 3 colors; generator preserves this
#     - room detection still assumes passable cells are bg/yellow enclosed by non-bg walls; generator preserves this
#     - border color lookup still uses majority adjacent wall color; generator keeps one wall color per room
#     - the old `_slide_points` assumed monotone travel to a boundary; this is invalid under reflection and is replaced with a finite-state simulator
#     - deterministic multi-yellow interaction still needs an order; we keep the original far-to-near ordering using the initial legend direction only
#   placement_legality:
#     - legends reserve a buffer zone as before
#     - grown rooms remain enclosed and disjoint
#     - yellow start cells are accepted only if the shared simulator keeps them inside the room and produces distinct final cells under the same processing order used by `main`
#   observability:
#     - generator rejects cases with zero bounces or straight-slide-equivalent outputs
#     - this makes the D3 change visible from a few examples: the yellow path must be mentally traced through rebounds rather than inferred as "move to the far wall"
#   cleanliness:
#     - no new colors or overlays are introduced
#     - the only changed visual behavior is the final yellow placement, so the puzzle remains readable


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

YELLOWS_PER_REGION = [1, 2]
REFLECT_STEPS = 8
MIN_MOVES_FOR_SAMPLE = 4
PLACEMENT_ATTEMPTS = 160

GEN_RETRIES = 300


@dataclass(frozen=True)
class TraceResult:
    final: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    bounces: int
    moves: int


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
        others.sort()
        if len(others) < 2:
            continue
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cents = []
            for col in (others[0][1], others[1][1]):
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


def _linear_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide_linear(
    region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int, blocked: set[tuple[int, int]]
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _linear_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _move_if_possible(
    region_set: set[tuple[int, int]], occupied: set[tuple[int, int]], r: int, c: int, dr: int, dc: int
) -> tuple[tuple[int, int] | None, str]:
    nr, nc = r + dr, c + dc
    if dr == 0 or dc == 0:
        if (nr, nc) not in region_set:
            return None, "wall"
        if (nr, nc) in occupied:
            return None, "occupied"
        return (nr, nc), "move"

    row_ok = (r + dr, c) in region_set
    col_ok = (r, c + dc) in region_set
    diag_ok = (nr, nc) in region_set
    if not (row_ok and col_ok and diag_ok):
        return None, "wall"
    if (nr, nc) in occupied:
        return None, "occupied"
    return (nr, nc), "move"


def _reflected_direction(
    region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int
) -> tuple[int, int] | None:
    if dr == 0 and dc == 0:
        return None
    if dr == 0:
        return (0, -dc) if (r, c + dc) not in region_set else None
    if dc == 0:
        return (-dr, 0) if (r + dr, c) not in region_set else None

    row_block = (r + dr, c) not in region_set
    col_block = (r, c + dc) not in region_set
    diag_block = (r + dr, c + dc) not in region_set
    if not (row_block or col_block or diag_block):
        return None

    new_dr, new_dc = dr, dc
    flipped = False
    if row_block:
        new_dr = -new_dr
        flipped = True
    if col_block:
        new_dc = -new_dc
        flipped = True
    if diag_block and not row_block and not col_block:
        new_dr = -new_dr
        new_dc = -new_dc
        flipped = True
    if not flipped:
        return None
    return int(new_dr), int(new_dc)


def _trace_reflection(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    occupied: set[tuple[int, int]],
    *,
    energy: int = REFLECT_STEPS,
) -> TraceResult:
    r, c = start
    cur_dr, cur_dc = int(dr), int(dc)
    path: list[tuple[int, int]] = []
    bounces = 0

    for _ in range(int(energy)):
        moved, reason = _move_if_possible(region_set, occupied, r, c, cur_dr, cur_dc)
        if moved is not None:
            r, c = moved
            path.append((r, c))
            continue
        if reason == "occupied":
            break

        new_dir = _reflected_direction(region_set, r, c, cur_dr, cur_dc)
        if new_dir is None or new_dir == (cur_dr, cur_dc):
            break
        cur_dr, cur_dc = new_dir
        bounces += 1

        moved, reason = _move_if_possible(region_set, occupied, r, c, cur_dr, cur_dc)
        if moved is None:
            break
        r, c = moved
        path.append((r, c))

    return TraceResult(final=(r, c), path=tuple(path), bounces=bounces, moves=len(path))


def _reflect_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> tuple[list[tuple[int, int]], list[TraceResult]]:
    if not points:
        return [], []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts: list[tuple[int, int]] = []
    traces: list[TraceResult] = []
    for start in points_sorted:
        trace = _trace_reflection(region_set, start, dr, dc, occupied)
        occupied.add(trace.final)
        out_pts.append(trace.final)
        traces.append(trace)
    return out_pts, traces


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

        new_pts, _traces = _reflect_points(pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover

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
    frontier: list[tuple[int, int]] = []

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
    stack: list[tuple[int, int]] = []
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


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, tuple[int, int]]] = []
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

    region_colors: list[int] = []
    for c in all_colors:
        region_colors.extend([c] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions: list[dict] = []

    for rc in region_colors:
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
        regions.append({"color": rc, "dir": color_dir[rc], "interior": blob, "walls": walls})

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


def _simulate_region_starts(
    blob: set[tuple[int, int]], starts: list[tuple[int, int]], dr: int, dc: int
) -> list[TraceResult]:
    ordered = sorted(starts, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    traces: list[TraceResult] = []
    blocked: set[tuple[int, int]] = set()
    for start in ordered:
        trace = _trace_reflection(blob, start, dr, dc, blocked)
        traces.append(trace)
        blocked.add(trace.final)
    return traces


def _place_region_yellows(
    rng: random.Random, reg: dict, grid: np.ndarray
) -> tuple[list[tuple[int, int]], list[TraceResult]] | None:
    blob = reg["interior"]
    dr, dc = reg["dir"]
    cells = list(blob)
    if len(cells) < 2:
        return None

    desired_counts = [rng.choice(YELLOWS_PER_REGION), 1]
    for n_yellows in desired_counts:
        for _ in range(PLACEMENT_ATTEMPTS):
            starts = rng.sample(cells, n_yellows)
            traces = _simulate_region_starts(blob, starts, dr, dc)
            ordered = sorted(starts, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)

            finals = [trace.final for trace in traces]
            if len(set(finals)) != len(finals):
                continue

            ok = True
            blocked: set[tuple[int, int]] = set()
            for start, trace in zip(ordered, traces):
                straight_final = _forward_slide_linear(blob, start, dr, dc, blocked)
                if trace.bounces < 1 or trace.moves < MIN_MOVES_FOR_SAMPLE or trace.final == straight_final:
                    ok = False
                    break
                blocked.add(trace.final)
            if not ok:
                continue

            for r, c in ordered:
                grid[r, c] = 4
            return ordered, traces
    return None


def _place_yellows(
    rng: random.Random, regions: list[dict], input_grid: np.ndarray
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[TraceResult]]] | None:
    region_starts: dict[int, list[tuple[int, int]]] = {}
    region_traces: dict[int, list[TraceResult]] = {}
    any_multi_bounce = False

    for idx, reg in enumerate(regions):
        placed = _place_region_yellows(rng, reg, input_grid)
        if placed is None:
            return None
        starts, traces = placed
        region_starts[idx] = starts
        region_traces[idx] = traces
        any_multi_bounce |= any(trace.bounces >= 2 for trace in traces)

    if not any_multi_bounce and regions:
        return None
    return region_starts, region_traces


def _expected_output(
    input_grid: np.ndarray, regions: list[dict], region_starts: dict[int, list[tuple[int, int]]]
) -> np.ndarray:
    out = input_grid.copy()
    for idx, reg in enumerate(regions):
        starts = region_starts[idx]
        dr, dc = reg["dir"]
        blob = reg["interior"]
        for r, c in starts:
            out[r, c] = 0
        finals, _traces = _reflect_points(blob, starts, dr, dc)
        for r, c in finals:
            out[r, c] = 4
    return out


def _oracle_check(
    input_grid: np.ndarray, expected_out: np.ndarray, regions: list[dict], region_traces: dict[int, list[TraceResult]]
) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    if not np.array_equal(solver_out, expected_out):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False

    h, w = input_grid.shape
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == 4])):
        return False

    return any(trace.bounces >= 1 for traces in region_traces.values() for trace in traces)


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
    placed = _place_yellows(rng, regions, input_grid)
    if placed is None:
        return None
    region_starts, region_traces = placed

    expected_out = _expected_output(input_grid, regions, region_starts)
    if not _oracle_check(input_grid, expected_out, regions, region_traces):
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
