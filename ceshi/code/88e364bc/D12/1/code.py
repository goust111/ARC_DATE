# Variant of 88e364bc
# Dimension: D12 — 图例放置位置 (symbol_instructions)
# Change: 图例放置在网格四角之一（TL/TR/BL/BR），1-2个图例占据不同角落... → 图例嵌入区域内部

"""
Design analysis for the D12 variant
===================================

spec_variant:
  grid: 20x20, bg=0
  legends:
    count: 1-2
    placement: embedded inside a matching-color region instead of at a grid corner
    structure: unchanged from the source task
      - 5x5 rectangle for cardinal directions
      - 6x7 rectangle for diagonal directions
      - colors are {fill C, blue 1 center mass, red 2 indicator mass}
      - counts are fixed: fill=(area-6), center=4, indicator=2
    visibility constraints:
      - each legend has a 1-cell moat of passable space around it
      - the legend sits strictly inside a region, never on the grid border
      - at least one yellow ends its slide against a legend face, so the legend is visibly an obstacle
  regions:
    count: 1-2 host regions, optionally one extra non-host region when space allows
    wall colors: 1-2 distinct colors from {3,5,6,7,8,9}; each legend color matches exactly one wall color
    shape:
      - base is a rectangular passable chamber
      - 0-3 shallow edge bites make the outer wall less trivial
      - host chambers contain one legend-shaped hole occupied by the legend patch
      - chambers remain 4-connected after the legend hole and edge bites
  yellow_pixels:
    1-2 per region
    host-region guarantee:
      - at least one yellow in a legend-hosting region reverse-slides to an input position whose forward slide
        stops because the next cell is legend, not the outer wall

Behavior trace
--------------
Original local mechanism:
  - `_legend_directions()` only inspects border-touching solid rectangles, so an internal legend is invisible.
  - `main()` treats every non-{bg,yellow} cell as a wall-like blocker. This is harmless in the source task because
    legends live outside every enclosed region.

Variant local mechanism:
  - legends are detected globally by their exact 3-color rectangular signature, not by border contact
  - the passable region still consists only of {bg, yellow}, so internal legend cells remain blocked
  - when deciding which legend controls a region, border-color counting must treat legend colors 1/2 as if they
    were the legend fill color C; otherwise an internal legend could make the region appear bordered by blue/red

Concrete example:
  - Suppose a grey-walled chamber slides right and contains a 5x5 grey legend near its right half.
  - In the source task, a yellow in that chamber would slide until it touches the outer right wall.
  - In this variant, if the yellow is aligned with the legend, it stops at the cell immediately left of the legend.
    The output therefore shows the yellow stacked against an internal obstacle rather than the far boundary.

Complexity calibration
----------------------
  - The legend appearance is unchanged, so the new difficulty is spatial: the solver must separate "same-color
    structure that is a directive symbol" from "same-color structure that is just a wall".
  - Keeping legend types to the original 5x5 / 6x7 vocabulary avoids adding an unrelated new symbol language.
  - The host-region guarantee prevents degenerate samples where the legend is present but irrelevant to the motion.

Implicit assumption audit for `main()`
--------------------------------------
  - Source assumption: legends touch the border.
    Fix: legend detection now scans all non-background components.
  - Source assumption: every non-{bg,yellow} neighbor around a region is a true border-color sample.
    Fix: legend cells colored 1/2 are normalized back to their owning fill color before border-color voting.
  - Source assumption: internal non-passable objects do not exist.
    Fix: the slide logic itself already handles blocked cells correctly, so only direction detection and border
    voting needed adjustment.

Placement legality and cleanliness
----------------------------------
  - Every host region is generated from a connected passable chamber, then its legend hole is carved out and
    filled by the legend patch; this guarantees no overlap and keeps the legend fully inside the chamber.
  - A buffer is reserved around each region's structure so different regions do not merge visually.
  - Edge bites are only 1 cell deep, avoiding hidden background pockets that would create misleading extra rooms.

New solver assumptions introduced here
--------------------------------------
  - Legends keep the original exact count pattern {fill: area-6, center: 4, indicator: 2}.
    The generator enforces this by using the unchanged source-task legend templates.
  - Each fill color has at most one legend.
    The generator uses unique wall colors per legend.
  - A host chamber remains connected after inserting the legend.
    The generator checks connectivity and rejects invalid placements.
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4
CENTER_COLOR = 1
INDICATOR_COLOR = 2

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]

HOST_REGION_TRIES = 120
EXTRA_REGION_TRIES = 80
GEN_RETRIES = 300
MAX_SLIDE_STEPS = 5
YELLOWS_PER_REGION = [1, 2]


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = CENTER_COLOR
    g[1, 2] = CENTER_COLOR
    g[1, 3] = INDICATOR_COLOR
    g[3, 1] = CENTER_COLOR
    g[3, 2] = CENTER_COLOR
    g[3, 3] = INDICATOR_COLOR
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = CENTER_COLOR
    g[2, 1] = CENTER_COLOR
    g[2, 4] = CENTER_COLOR
    g[3, 2] = CENTER_COLOR
    g[3, 5] = INDICATOR_COLOR
    g[4, 3] = INDICATOR_COLOR
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _legend_patch(fill: int, dr: int, dc: int, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return _legend_5x5(fill, dr, dc)
    return _legend_6x7(fill, dr, dc)


def _as_coord_set(pts: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in pts}


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _buffer_cells(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out = set(cells)
    frontier = set(cells)
    for _ in range(radius):
        nxt = set()
        for r, c in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in out:
                        nxt.add((nr, nc))
        out |= nxt
        frontier = nxt
    return out


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _edge_bite(
    rng: random.Random,
    passable: set[tuple[int, int]],
    pass_r0: int,
    pass_c0: int,
    pass_h: int,
    pass_w: int,
    protected: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        width = rng.randint(2, max(2, min(4, pass_w - 2)))
        c0 = rng.randint(pass_c0, pass_c0 + pass_w - width)
        row = pass_r0 if side == "top" else pass_r0 + pass_h - 1
        bite = {(row, c) for c in range(c0, c0 + width)}
    else:
        height = rng.randint(2, max(2, min(4, pass_h - 2)))
        r0 = rng.randint(pass_r0, pass_r0 + pass_h - height)
        col = pass_c0 if side == "left" else pass_c0 + pass_w - 1
        bite = {(r, col) for r in range(r0, r0 + height)}
    if bite & protected:
        return passable
    cand = passable - bite
    if cand and _is_connected(cand):
        return cand
    return passable


def _legend_positions(
    pass_r0: int,
    pass_c0: int,
    pass_h: int,
    pass_w: int,
    patch_h: int,
    patch_w: int,
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    row_min = pass_r0 + 1
    row_max = pass_r0 + pass_h - patch_h - 1
    col_min = pass_c0 + 1
    col_max = pass_c0 + pass_w - patch_w - 1
    if row_min > row_max or col_min > col_max:
        return []

    upstream_need = 3
    positions = []
    for r0 in range(row_min, row_max + 1):
        for c0 in range(col_min, col_max + 1):
            top_gap = r0 - pass_r0
            left_gap = c0 - pass_c0
            bottom_gap = (pass_r0 + pass_h - 1) - (r0 + patch_h - 1)
            right_gap = (pass_c0 + pass_w - 1) - (c0 + patch_w - 1)
            if dr == 1 and top_gap < upstream_need:
                continue
            if dr == -1 and bottom_gap < upstream_need:
                continue
            if dc == 1 and left_gap < upstream_need:
                continue
            if dc == -1 and right_gap < upstream_need:
                continue
            positions.append((r0, c0))

    positions.sort(key=lambda rc: dr * (rc[0] + patch_h / 2.0) + dc * (rc[1] + patch_w / 2.0), reverse=True)
    return positions


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


def _legend_specs(g: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int]]:
    best: dict[int, tuple[int, tuple[int, int], np.ndarray]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts) or area not in {25, 42}:
            continue
        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue
        fill_candidates = [col for col, cnt in counts.items() if cnt == area - 6]
        center_candidates = [col for col, cnt in counts.items() if cnt == 4]
        indicator_candidates = [col for col, cnt in counts.items() if cnt == 2]
        if len(fill_candidates) != 1 or len(center_candidates) != 1 or len(indicator_candidates) != 1:
            continue
        fill = fill_candidates[0]
        center_col = center_candidates[0]
        indicator_col = indicator_candidates[0]
        if fill in {bg, MOVER, CENTER_COLOR, INDICATOR_COLOR}:
            continue
        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indic = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        cc = center.mean(axis=0)
        ci = indic.mean(axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue
        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), (dr, dc), pts)

    dirs = {fill: info[1] for fill, info in best.items()}
    legend_owner: dict[tuple[int, int], int] = {}
    for fill, (_sz, _dir, pts) in best.items():
        for r, c in pts:
            legend_owner[(int(r), int(c))] = fill
    return dirs, legend_owner


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs, legend_owner = _legend_specs(g, bg)
    assert dirs, "legend detection failed: expected at least one embedded legend rectangle"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        pset = _as_coord_set(pts)
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                if (nr, nc) in legend_owner:
                    v = legend_owner[(nr, nc)]
                else:
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


def _make_host_region(
    rng: random.Random,
    color: int,
    direction: tuple[int, int],
    patch: np.ndarray,
    occupied: np.ndarray,
    frame: tuple[int, int, int, int] | None = None,
    height_choices: list[int] | None = None,
    width_choices: list[int] | None = None,
    reserve_radius: int = 1,
) -> dict | None:
    h, w = occupied.shape
    dr, dc = direction
    ph, pw = patch.shape

    if height_choices is None or width_choices is None:
        if ph == 5:
            height_choices = list(range(8, 12))
            width_choices = list(range(8, 12))
        else:
            height_choices = list(range(9, 12))
            width_choices = list(range(10, 13))

    if frame is None:
        frame = (2, 2, h - 3, w - 3)
    frame_r0, frame_c0, frame_r1, frame_c1 = frame

    for _ in range(HOST_REGION_TRIES):
        pass_h = rng.choice(height_choices)
        pass_w = rng.choice(width_choices)
        if pass_h > frame_r1 - frame_r0 + 1 or pass_w > frame_c1 - frame_c0 + 1:
            continue
        pass_r0 = rng.randint(frame_r0, frame_r1 - pass_h + 1)
        pass_c0 = rng.randint(frame_c0, frame_c1 - pass_w + 1)

        legend_positions = _legend_positions(pass_r0, pass_c0, pass_h, pass_w, ph, pw, dr, dc)
        if not legend_positions:
            continue
        pick_pool = legend_positions[: max(1, len(legend_positions) // 2)]
        leg_r0, leg_c0 = pick_pool[rng.randrange(len(pick_pool))]

        legend_cells = _rect_cells(leg_r0, leg_c0, ph, pw)
        passable = _rect_cells(pass_r0, pass_c0, pass_h, pass_w) - legend_cells
        protected = _buffer_cells(legend_cells, h, w, radius=1) & _rect_cells(pass_r0, pass_c0, pass_h, pass_w)

        for _bite in range(rng.randint(1, 3)):
            passable = _edge_bite(rng, passable, pass_r0, pass_c0, pass_h, pass_w, protected)

        if len(passable) < max(18, ph * pw // 2):
            continue
        if not _is_connected(passable):
            continue

        walls = _blob_wall_cells(passable, h, w) - legend_cells
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in walls):
            continue

        struct = passable | legend_cells | walls
        if any(occupied[r, c] for r, c in struct):
            continue

        blocked_targets = [p for p in passable if (p[0] + dr, p[1] + dc) in legend_cells]
        if not blocked_targets:
            continue

        reserve = _buffer_cells(struct, h, w, radius=reserve_radius)
        region = {
            "color": color,
            "dir": direction,
            "passable": passable,
            "walls": walls,
            "legend_cells": legend_cells,
            "legend_patch": patch,
            "legend_top_left": (leg_r0, leg_c0),
            "reserve": reserve,
            "host": True,
        }
        return region
    return None


def _make_extra_region(
    rng: random.Random,
    color: int,
    direction: tuple[int, int],
    occupied: np.ndarray,
) -> dict | None:
    h, w = occupied.shape
    for _ in range(EXTRA_REGION_TRIES):
        pass_h = rng.randint(6, 10)
        pass_w = rng.randint(6, 10)
        pass_r0 = rng.randint(2, h - pass_h - 3)
        pass_c0 = rng.randint(2, w - pass_w - 3)
        passable = _rect_cells(pass_r0, pass_c0, pass_h, pass_w)
        for _bite in range(rng.randint(1, 3)):
            passable = _edge_bite(rng, passable, pass_r0, pass_c0, pass_h, pass_w, protected=set())
        if len(passable) < 15 or not _is_connected(passable):
            continue
        walls = _blob_wall_cells(passable, h, w)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in walls):
            continue
        struct = passable | walls
        if any(occupied[r, c] for r, c in struct):
            continue
        region = {
            "color": color,
            "dir": direction,
            "passable": passable,
            "walls": walls,
            "legend_cells": set(),
            "legend_patch": None,
            "legend_top_left": None,
            "reserve": _buffer_cells(struct, h, w, radius=1),
            "host": False,
        }
        return region
    return None


def _find_reverse_slide(
    region_set: set[tuple[int, int]],
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
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in region_set or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in region_set or (pr, pc + dc) not in region_set:
                    ok = False
                    break
            r, c = pr, pc
        if not ok:
            continue
        if _forward_slide(region_set, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _frontier_targets(region_set: set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    frontier = []
    for p in region_set:
        if not _slide_step_valid(region_set, p[0], p[1], dr, dc):
            frontier.append(p)
    frontier.sort(key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    return frontier


def _place_region_yellows(rng: random.Random, region: dict, grid: np.ndarray) -> bool:
    blob = region["passable"]
    dr, dc = region["dir"]
    n_yellows = rng.choice(YELLOWS_PER_REGION)

    frontier = _frontier_targets(blob, dr, dc)
    if not frontier:
        return False

    if region["host"]:
        primary_pool = [p for p in frontier if (p[0] + dr, p[1] + dc) in region["legend_cells"]]
        if not primary_pool:
            return False
    else:
        primary_pool = frontier

    first_candidates = primary_pool[: max(1, min(len(primary_pool), 6))]
    rng.shuffle(first_candidates)

    for first in first_candidates:
        output_sets = [[first]]
        if n_yellows == 2:
            second_candidates = [p for p in frontier if p != first]
            preferred = []
            pr, pc = first[0] - dr, first[1] - dc
            if (pr, pc) in blob and (pr, pc) != first:
                preferred.append((pr, pc))
            preferred.extend(p for p in second_candidates if p not in preferred)
            for second in preferred[: max(1, min(len(preferred), 6))]:
                output_sets.append([first, second])

        for outputs in output_sets:
            out_sorted = sorted(outputs, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
            already_stopped = set()
            inputs = []
            ok = True
            for out_pos in out_sorted:
                candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, inputs, rng)
                if candidate is None:
                    ok = False
                    break
                inputs.append(candidate)
                already_stopped.add(out_pos)
            if not ok:
                continue
            for r, c in inputs:
                grid[r, c] = MOVER
            return True
    return False


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append(_as_coord_set(coords))
    region_sets = [reg["passable"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == MOVER).sum()) != int((input_grid == MOVER).sum()):
        return False
    allowed = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["passable"]:
            allowed[r, c] = True
    return bool(np.all(allowed[solver_out == MOVER]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    n_legends = rng.choice([1, 2])
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    occupied = np.zeros_like(grid, dtype=bool)
    regions = []

    host_specs = []
    if n_legends == 1:
        for color in wall_colors:
            direction = rng.choice(CARDINAL_DIRS + DIAGONAL_DIRS)
            patch = _legend_patch(color, direction[0], direction[1], rng)
            host_specs.append((color, direction, patch, None, None, None, 1))
    else:
        if rng.random() < 0.5:
            frames = [(2, 2, 9, 17), (11, 2, 18, 17)]
            dir_choices = [(0, 1), (0, -1)]
            height_choices = [7, 8]
            width_choices = [9, 10, 11]
        else:
            frames = [(2, 2, 17, 9), (2, 11, 17, 18)]
            dir_choices = [(1, 0), (-1, 0)]
            height_choices = [9, 10, 11]
            width_choices = [7, 8]
        for idx, color in enumerate(wall_colors):
            direction = rng.choice(dir_choices)
            patch = _legend_5x5(color, direction[0], direction[1])
            host_specs.append((color, direction, patch, frames[idx], height_choices, width_choices, 0))

    for color, direction, patch, frame, height_choices, width_choices, reserve_radius in host_specs:
        region = _make_host_region(
            rng,
            color,
            direction,
            patch,
            occupied,
            frame=frame,
            height_choices=height_choices,
            width_choices=width_choices,
            reserve_radius=reserve_radius,
        )
        if region is None:
            return None
        regions.append(region)
        for r, c in region["reserve"]:
            occupied[r, c] = True

    if n_legends == 1 or rng.random() < 0.35:
        extra_color = rng.choice(wall_colors)
        direction = next(reg["dir"] for reg in regions if reg["color"] == extra_color)
        extra = _make_extra_region(rng, extra_color, direction, occupied)
        if extra is not None:
            regions.append(extra)
            for r, c in extra["reserve"]:
                occupied[r, c] = True

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]
        if reg["host"]:
            paste(grid, reg["legend_patch"], reg["legend_top_left"][0], reg["legend_top_left"][1])

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    host_with_yellow = 0
    for reg in regions:
        if not _place_region_yellows(rng, reg, input_grid):
            return None
        if reg["host"] and any(input_grid[r, c] == MOVER for r, c in reg["passable"]):
            host_with_yellow += 1
    if host_with_yellow == 0:
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
