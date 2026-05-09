# Combo variant of 88e364bc (index 13)
# combo_spec: D3/3+D8/2
# Name: 紧贴墙壁像素+反射路径

from __future__ import annotations

import itertools
import random
from typing import NamedTuple

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis for the D3/3 + D8/2 combo variant
# --------------------------------------------------
# a) Concrete behavior and trace
#    - Source task: legends still map wall color -> direction, and ordinary yellows slide straight
#      toward the legend direction until they stack at the far wall.
#    - D8/2 contribution: the input is now already in that far-wall stacked state, so a source-task
#      solver would usually predict "no change".
#    - D3/3 contribution must therefore act on that stacked input itself: each yellow still launches
#      using the legend direction, which means its first attempted step is into the wall it is
#      touching. That failed first step triggers an immediate reflection, so the yellow "pops off"
#      the wall before continuing. Later wall hits reflect again.
#    - Example corridor with legend direction right:
#      input yellow at the right wall -> try right -> bounce to left -> cross the room ->
#      hit the left wall -> bounce right -> stop after a fixed energy budget in a scattered spot.
#      The difference from plain D8/2 is that the output is not just the reverse-slide resting
#      place; it depends on the reflected multi-bounce route.
#
# b) Complexity calibration
#    - Keep the original legend machinery, wall colors, and enclosed-room topology.
#    - Use one global reflection budget instead of per-instance hidden parameters.
#    - Restrict yellow counts to the original 2-6 total, but allow empty regions.
#    - Generator rejects samples where the reflected result matches a simple reverse slide, because
#      that would make the reflection dimension visually weak.
#
# c) Implicit-assumption review of main()
#    - Legend parsing still assumes border-touching solid rectangles with exactly three colors.
#    - Region detection still assumes passable cells are bg/yellow enclosed by non-bg walls.
#    - Border-color lookup by adjacent wall majority still works because each room keeps one wall
#      family.
#    - The original monotone `_slide_points(..., dr, dc)` assumption is no longer valid for output.
#      We replace it with a stepwise finite-state reflection simulator.
#    - New combo assumption: input yellows are forward-fixed wall states. `generate_input()` now
#      guarantees that for every populated region.
#
# d) Placement legality
#    - Legends still reserve buffered corner patches.
#    - Region growth still reserves interior, walls, and a one-cell wall buffer, so rooms do not
#      overlap and walls stay off the outer border.
#    - Yellow candidates are accepted only if they are legal forward-stacked inputs and the shared
#      reflection simulator keeps all final positions inside the same room with distinct endpoints.
#
# e) Visual observability
#    - Across examples the repeated cue is consistent: input yellows hug the legend-facing wall, but
#      output yellows detach and land mid-room after rebounds.
#    - A solver using only D8/2 would reverse-slide them away from the wall in a straight monotone
#      fashion; the generator filters out cases where that prediction matches the combo output.
#
# f) Visual cleanliness
#    - No new colors or overlays are introduced.
#    - The only moving objects remain the yellow pixels, and they stay inside clean enclosed rooms.
#    - Empty rooms are allowed, but populated rooms are filtered to show a clear detach-and-scatter
#      effect rather than cluttered overlaps.
#
# g) Color reasoning
#    - No new color mapping is added. The only semantic color link remains:
#      legend fill color C <-> wall color C.
#
# Generator note
#    - The source task used output-first reverse placement. For this combo, the decisive hidden
#      property is the input being a forward-fixed wall state, so the generator samples legal inputs
#      directly, derives outputs with the same simulator as `main()`, and then oracle-checks them.

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
    1: (15, 65),
    2: (15, 45),
    3: (12, 30),
    4: (12, 22),
}
BLOB_GROW_RETRIES = 25

TOTAL_YELLOW_RANGE = (2, 6)
REFLECT_STEPS = 12
MIN_BOUNCES_FOR_SAMPLE = 2
MIN_MOVES_FOR_SAMPLE = 4
GEN_RETRIES = 300


class TraceResult(NamedTuple):
    final: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    bounces: int
    moves: int
    first_bounce_immediate: bool


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

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) < 2:
            continue
        others.sort()
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

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _linear_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts: list[tuple[int, int]] = []
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
    return (r, c)


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
    first_bounce_immediate = False

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
        if not path and bounces == 0:
            first_bounce_immediate = True
        cur_dr, cur_dc = new_dir
        bounces += 1

        moved, _reason = _move_if_possible(region_set, occupied, r, c, cur_dr, cur_dc)
        if moved is None:
            break
        r, c = moved
        path.append((r, c))

    return TraceResult(
        final=(r, c),
        path=tuple(path),
        bounces=bounces,
        moves=len(path),
        first_bounce_immediate=first_bounce_immediate,
    )


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


def _motion_regions(
    g: np.ndarray,
) -> tuple[int, list[tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]]]:
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == mover)
    regions: list[tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]] = []
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
        regions.append((pset, movers, dirs[border_col]))
    return bg, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch wall-jammed yellow pixels using the legend arrows, with the touching wall causing the first rebound and later wall hits continuing the reflected path.

    Input structure:
    - Background pockets enclosed by colored walls may contain yellow pixels that already sit in the forward-stacked, wall-touching state for that wall family.
    - Border legends are solid rectangles on the outer frame; their fill color matches a room-wall color, and their center/indicator dots encode an initial direction.

    Rule:
    - For each enclosed region, use the matching legend to get the launch direction.
    - Each yellow attempts to move with that direction for a fixed number of steps, reflecting off walls whenever the next step would leave the room; already-settled yellow endpoints still block later yellows.
    - Because the input yellows start tight against the legend-facing wall, the first attempted step typically bounces immediately, so the yellows pop away from the wall and may rebound again before stopping.

    Key insight:
    - The yellow pixels are not supposed to reverse-slide directly; they must be fired along the legend arrow from a wall-touching start, so the initial wall contact determines the rebound and the rest of the reflected route.
    """
    g = np.asarray(grid, dtype=int)
    bg, regions = _motion_regions(g)
    out = g.copy()

    for pset, movers, (dr, dc) in regions:
        for r, c in movers:
            out[r, c] = bg
        new_pts, _traces = _reflect_points(pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = 4
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
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
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


def _combo_state_good(
    blob: set[tuple[int, int]], starts: list[tuple[int, int]], dr: int, dc: int, traces: list[TraceResult]
) -> bool:
    finals = [trace.final for trace in traces]
    ordered_starts = sorted(starts, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    if len(set(finals)) != len(finals):
        return False
    if any(start == trace.final for start, trace in zip(ordered_starts, traces)):
        return False
    if set(_slide_points(blob, starts, dr, dc)) != set(starts):
        return False
    if set(finals) == set(starts):
        return False
    if any(
        (not trace.first_bounce_immediate)
        or trace.bounces < MIN_BOUNCES_FOR_SAMPLE
        or trace.moves < MIN_MOVES_FOR_SAMPLE
        for trace in traces
    ):
        return False
    reverse_pts = _slide_points(blob, starts, -dr, -dc)
    if set(finals) == set(reverse_pts):
        return False
    return True


def _combo_state_options(
    blob: set[tuple[int, int]], dr: int, dc: int
) -> dict[int, list[list[tuple[int, int]]]]:
    cells = sorted(blob, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    options: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}

    for p in cells:
        starts = [p]
        if set(_slide_points(blob, starts, dr, dc)) != set(starts):
            continue
        traces = _simulate_region_starts(blob, starts, dr, dc)
        if _combo_state_good(blob, starts, dr, dc, traces):
            options[1].append(starts)

    for p, q in itertools.combinations(cells, 2):
        starts = [p, q]
        if set(_slide_points(blob, starts, dr, dc)) != set(starts):
            continue
        traces = _simulate_region_starts(blob, starts, dr, dc)
        if _combo_state_good(blob, starts, dr, dc, traces):
            options[2].append(starts)

    return options


def _sample_region_counts(
    rng: random.Random, options_by_region: list[dict[int, list[list[tuple[int, int]]]]]
) -> list[int] | None:
    total_min, total_max = TOTAL_YELLOW_RANGE
    max_total = sum(2 if opts[2] else 1 if opts[1] else 0 for opts in options_by_region)
    if max_total < total_min:
        return None

    weighted_options = []
    for opts in options_by_region:
        counts = [0]
        if opts[1]:
            counts.extend([1, 1])
        if opts[2]:
            counts.extend([2, 2, 2])
        weighted_options.append(counts)

    for _ in range(120):
        counts = [rng.choice(opts) for opts in weighted_options]
        total = sum(counts)
        if total < total_min or total > total_max:
            continue
        if not any(count > 0 for count in counts):
            continue
        return counts
    return None


def _place_yellows(
    rng: random.Random, regions: list[dict], grid: np.ndarray
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[TraceResult]]] | None:
    options_by_region = []
    for reg in regions:
        dr, dc = reg["dir"]
        options_by_region.append(_combo_state_options(reg["interior"], dr, dc))

    counts = _sample_region_counts(rng, options_by_region)
    if counts is None:
        return None

    region_starts: dict[int, list[tuple[int, int]]] = {}
    region_traces: dict[int, list[TraceResult]] = {}
    for idx, (reg, opts, count) in enumerate(zip(regions, options_by_region, counts)):
        if count == 0:
            continue
        starts = rng.choice(opts[count])
        traces = _simulate_region_starts(reg["interior"], starts, *reg["dir"])
        if not _combo_state_good(reg["interior"], starts, *reg["dir"], traces):
            return None
        for r, c in starts:
            grid[r, c] = 4
        region_starts[idx] = list(starts)
        region_traces[idx] = traces

    if sum(len(v) for v in region_starts.values()) < TOTAL_YELLOW_RANGE[0]:
        return None
    if not region_starts:
        return None
    return region_starts, region_traces


def _expected_output(input_grid: np.ndarray, regions: list[dict], region_starts: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    out = input_grid.copy()
    for starts in region_starts.values():
        for r, c in starts:
            out[r, c] = 0
    for idx, starts in region_starts.items():
        blob = regions[idx]["interior"]
        dr, dc = regions[idx]["dir"]
        finals, _traces = _reflect_points(blob, starts, dr, dc)
        for r, c in finals:
            out[r, c] = 4
    return out


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    _bg, regions = _motion_regions(g)

    metrics = {
        "active_regions": 0,
        "forward_fixed_regions": 0,
        "changed_regions": 0,
        "reverse_divergent_regions": 0,
        "total_traces": 0,
        "immediate_bounce_traces": 0,
        "multi_bounce_traces": 0,
    }

    for pset, movers, (dr, dc) in regions:
        combo_pts, traces = _reflect_points(pset, movers, dr, dc)
        forward_pts = _slide_points(pset, movers, dr, dc)
        reverse_pts = _slide_points(pset, movers, -dr, -dc)

        metrics["active_regions"] += 1
        metrics["total_traces"] += len(traces)
        metrics["immediate_bounce_traces"] += sum(trace.first_bounce_immediate for trace in traces)
        metrics["multi_bounce_traces"] += sum(trace.bounces >= MIN_BOUNCES_FOR_SAMPLE for trace in traces)
        metrics["forward_fixed_regions"] += int(set(forward_pts) == set(movers))
        metrics["changed_regions"] += int(set(combo_pts) != set(movers))
        metrics["reverse_divergent_regions"] += int(set(combo_pts) != set(reverse_pts))

    return metrics


def _oracle_check(
    input_grid: np.ndarray,
    expected_out: np.ndarray,
    region_starts: dict[int, list[tuple[int, int]]],
) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    if not np.array_equal(solver_out, expected_out):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False
    if sum(len(v) for v in region_starts.values()) < TOTAL_YELLOW_RANGE[0]:
        return False

    metrics = behavior_metrics(input_grid)
    if metrics["active_regions"] != len(region_starts):
        return False
    if metrics["forward_fixed_regions"] != metrics["active_regions"]:
        return False
    if metrics["changed_regions"] != metrics["active_regions"]:
        return False
    if metrics["reverse_divergent_regions"] != metrics["active_regions"]:
        return False
    if metrics["immediate_bounce_traces"] != metrics["total_traces"]:
        return False
    if metrics["multi_bounce_traces"] != metrics["total_traces"]:
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
    placed = _place_yellows(rng, regions, input_grid)
    if placed is None:
        return None
    region_starts, _region_traces = placed

    expected_out = _expected_output(input_grid, regions, region_starts)
    if not _oracle_check(input_grid, expected_out, region_starts):
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
