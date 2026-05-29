# Combo variant of 88e364bc (index 25)
# combo_spec: D5/1+D22/0
# Name: 多色移动像素+颜色距离属性
#
# Design analysis
# ===============
# fused_variant_spec:
#   concrete_visual_form:
#     - The original 20x20 fenced-region task stays intact: colored walls enclose background
#       interiors, and legends still sit on the outer border.
#     - The legend is upgraded to a fixed 9x9 solid rectangle keyed by wall color.
#       Inside that rectangle, yellow(4), pink(6), and orange(7) each appear exactly once.
#     - Each mover-colored dot has its own red(2) neighbor. The red offset relative to that
#       dot is that color's direction for regions surrounded by the same wall color.
#     - Thus one legend simultaneously says:
#         yellow goes one way, orange another, pink a third.
#   behavior_trace:
#     - Source task:
#       - a region finds one wall color -> one legend -> one direction;
#       - every yellow pixel in the region slides fully along that same direction.
#     - Fused task:
#       - the region still finds its wall color and matching legend;
#       - then each mover reads the arrow tied to its own color inside that legend;
#       - yellow takes the full free run, orange takes floor(free_run / 2),
#         pink stays at its input cell even if its arrow points into open space.
#     - Example:
#       - suppose a green-walled region uses a green legend where yellow points right,
#         orange points down-left, pink points up.
#       - a yellow mover slides all the way to the right wall;
#         an orange mover in the same region stops halfway down-left;
#         a pink mover remains where it started.
#     - The key fusion is not "direction by color" plus an unrelated recolor:
#       the same visible mover color now selects both which arrow to read and how much of that
#       available run is consumed.
#   complexity_calibration:
#     - Only three mover colors are used, with fixed reusable roles:
#         yellow=full slide, orange=half slide, pink=static.
#     - Wall colors stay in {3,5,8,9} so legends and movers remain visually separable.
#     - Every generated sample contains one showcase region with all three mover colors, making
#       both the direction split and the distance split observable in a single I/O pair.
#     - To avoid invisible tie-breaking between different directions, each region contains at most
#       one mover of each color, and the generator rejects overlapping paths/destinations.
#   main_hidden_assumptions_review:
#     - `mover = 4` is no longer valid; passability and repaint logic must cover {4,6,7}.
#     - Border-color voting can no longer ignore only yellow; it must ignore every mover color.
#     - A region no longer has a single direction. Direction lookup becomes
#       `(wall_color, mover_color) -> (dr, dc)`.
#     - The old "slide everything with the same stopping rule" assumption breaks because
#       orange and pink use different travel lengths from yellow.
#     - Legend parsing can no longer rely on a 3-color rectangle; it must extract three separate
#       mover-color sub-instructions from one 9x9 patch and exclude those legend cells from
#       region detection.
#     - New assumption introduced by this fused solver: cross-color movers are intended to be
#       non-interfering except for fixed pink anchors, so the generator guarantees disjoint traces
#       and unique final cells.
#   placement_legality:
#     - Legend centers are fixed and sufficiently separated; sampled red indicators must stay
#       inside the 9x9 patch and may not overlap any center or other indicator.
#     - Regions are grown first, then validated as enclosed background components exactly as in
#       the source task.
#     - The showcase planner keeps mover starts distinct, keeps yellow/orange paths disjoint,
#       blocks those paths from passing through pink, and rejects any duplicate output cells.
#   visual_observability:
#     - In one region the solver sees three visible colors under one wall color, but the output
#       sends yellow and orange in different directions and by different distances while pink
#       conspicuously stays put.
#     - Pink is only accepted when its own arrow has at least one open step ahead, so "pink does
#       not move" is visible rather than vacuous.
#     - Orange is only accepted when it truly moves fewer cells than the free run available,
#       making the half-distance rule observable instead of collapsing to full-slide cases.
#   cleanliness:
#     - The grid is still sparse: only a few single-pixel movers are placed inside otherwise empty
#       enclosed regions, and all blocking/interaction cases are filtered out at generation time.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

YELLOW = 4
PINK = 6
ORANGE = 7
INDICATOR_COLOR = 2
MOVER_COLORS = (YELLOW, PINK, ORANGE)
WALL_COLOR_CHOICES = (3, 5, 8, 9)

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]

LEGEND_H, LEGEND_W = 9, 9
LEGEND_BUFFER = 1
LEGEND_CENTERS = {
    YELLOW: (2, 2),
    PINK: (2, 6),
    ORANGE: (6, 4),
}

MAX_TOTAL_REGIONS = 3
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (15, 52),
    3: (12, 30),
}
BLOB_GROW_RETRIES = 25
GEN_RETRIES = 400


def _legend_direction_map(
    g: np.ndarray, bg: int
) -> tuple[dict[int, dict[int, tuple[int, int]]], set[tuple[int, int]]]:
    """Parse wall-color legends and return wall_color -> mover_color -> direction."""
    h, w = g.shape
    best: dict[int, tuple[int, dict[int, tuple[int, int]], set[tuple[int, int]]]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (LEGEND_H, LEGEND_W):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        coords_by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in pts:
            col = int(g[r, c])
            counts[col] = counts.get(col, 0) + 1
            coords_by_color.setdefault(col, []).append((int(r), int(c)))

        fill = max(counts, key=counts.get)
        if fill in MOVER_COLORS or fill == INDICATOR_COLOR:
            continue
        if any(counts.get(color, 0) != 1 for color in MOVER_COLORS):
            continue
        red_pts = coords_by_color.get(INDICATOR_COLOR, [])
        if len(red_pts) != len(MOVER_COLORS):
            continue

        remaining = set(red_pts)
        dir_map: dict[int, tuple[int, int]] = {}
        ok = True
        for color in MOVER_COLORS:
            cr, cc = coords_by_color[color][0]
            nearest = None
            for rr, rc in remaining:
                cand = (abs(rr - cr) + abs(rc - cc), rr, rc)
                if nearest is None or cand < nearest:
                    nearest = cand
            if nearest is None:
                ok = False
                break
            _, rr, rc = nearest
            dr = 1 if rr > cr else -1 if rr < cr else 0
            dc = 1 if rc > cc else -1 if rc < cc else 0
            if (dr, dc) == (0, 0):
                ok = False
                break
            if abs(rr - cr) > 1 or abs(rc - cc) > 1:
                ok = False
                break
            dir_map[color] = (dr, dc)
            remaining.remove((rr, rc))
        if not ok:
            continue

        cells = {(int(r), int(c)) for r, c in pts}
        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), dir_map, cells)

    dirs = {wall_color: dir_map for wall_color, (_, dir_map, _) in best.items()}
    legend_cells: set[tuple[int, int]] = set()
    for _, (_, _, cells) in best.items():
        legend_cells |= cells
    return dirs, legend_cells


def _move_steps_for_color(color: int, free_steps: int) -> int:
    if color == YELLOW:
        return free_steps
    if color == ORANGE:
        return free_steps // 2
    if color == PINK:
        return 0
    raise ValueError(f"unexpected mover color {color}")


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _path_until_blocked(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r, c = start
    path: list[tuple[int, int]] = []
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in occupied:
            break
        r, c = nr, nc
        path.append((r, c))
    return path


def _simulate_region_movers(
    region_set: set[tuple[int, int]],
    movers: list[tuple[int, int, int]],
    dir_map: dict[int, tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Apply the fused motion rule inside one enclosed region."""
    static_pts = [(r, c, color) for r, c, color in movers if color == PINK]
    moving_pts = [(r, c, color) for r, c, color in movers if color != PINK]

    occupied = {(r, c) for r, c, _ in static_pts}
    out_pts = list(static_pts)

    # Generator guarantees different-direction movers do not interfere; this priority only
    # resolves arbitrary out-of-distribution collisions.
    moving_pts.sort(key=lambda p: (0 if p[2] == YELLOW else 1, p[0], p[1]))
    for r0, c0, color in moving_pts:
        dr, dc = dir_map[color]
        path = _path_until_blocked(region_set, (r0, c0), dr, dc, occupied)
        steps = _move_steps_for_color(color, len(path))
        if steps > 0:
            r1, c1 = path[steps - 1]
        else:
            r1, c1 = r0, c0
        occupied.add((r1, c1))
        out_pts.append((r1, c1, color))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-colored mover pixels inside sealed wall regions using color-specific arrows and color-specific travel lengths.

    Input structure:
    - The background is color 0, and enclosed regions are surrounded by colored one-cell-thick walls.
    - Border legends are solid 9x9 rectangles whose fill color matches the wall color they control.
    - Each legend contains one yellow, one pink, and one orange pixel; each of those has its own adjacent red indicator showing a direction.
    - Region interiors may contain yellow/orange/pink single-pixel movers.

    Rule:
    - For each enclosed region, identify its wall color and read the matching legend.
    - A mover uses the red offset attached to its own color inside that legend as its movement direction.
    - Yellow slides through its full free run, orange slides only half of its free run, and pink does not move.
    - Diagonal motion still cannot cut a corner outside the region.

    Key insight:
    - Wall color chooses which legend applies, while mover color chooses both which arrow to read inside that legend and how much of the available path to traverse.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs, legend_cells = _legend_direction_map(g, bg)
    assert dirs, "legend parsing failed; generated inputs always include at least one border legend"

    mover_mask = np.isin(g, MOVER_COLORS)
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, c in legend_cells:
        legend_mask[r, c] = True
    passable = (g == bg) | (mover_mask & ~legend_mask)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c), int(g[r, c])) for r, c in pts if int(g[r, c]) in MOVER_COLORS]
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
                if v == bg or v in MOVER_COLORS:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        region_dirs = dirs.get(border_col)
        if region_dirs is None:
            continue

        for r, c, _ in movers:
            out[r, c] = bg
        for r, c, color in _simulate_region_movers(region_set, movers, region_dirs):
            out[r, c] = color

    return out


def _legend_patch(fill: int, dir_map: dict[int, tuple[int, int]]) -> np.ndarray:
    g = np.full((LEGEND_H, LEGEND_W), fill, dtype=int)
    for color, (cr, cc) in LEGEND_CENTERS.items():
        dr, dc = dir_map[color]
        ir, ic = cr + dr, cc + dc
        g[cr, cc] = color
        g[ir, ic] = INDICATOR_COLOR
    return g


def _sample_legend_dirs(rng: random.Random) -> dict[int, tuple[int, int]]:
    for _ in range(300):
        picked = rng.sample(ALL_DIRS, len(MOVER_COLORS))
        dir_map = dict(zip(MOVER_COLORS, picked))
        occupied = set(LEGEND_CENTERS.values())
        red_cells = set()
        ok = True
        for color, (dr, dc) in dir_map.items():
            cr, cc = LEGEND_CENTERS[color]
            ir, ic = cr + dr, cc + dc
            if not (0 <= ir < LEGEND_H and 0 <= ic < LEGEND_W):
                ok = False
                break
            if (ir, ic) in occupied or (ir, ic) in red_cells:
                ok = False
                break
            red_cells.add((ir, ic))
        if ok and dir_map[YELLOW] != dir_map[ORANGE]:
            return dir_map
    raise RuntimeError("_sample_legend_dirs: unable to place non-overlapping color-specific arrows")


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
    wall: set[tuple[int, int]] = set()
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


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(list(WALL_COLOR_CHOICES), n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[dict[str, object]] = []
    for corner, wall_color in zip(corners, wall_colors):
        dir_map = _sample_legend_dirs(rng)
        patch = _legend_patch(wall_color, dir_map)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"wall_color": wall_color, "dir_map": dict(dir_map)})
    return legend_occupied, legends


def _grow_regions(
    rng: random.Random, legends: list[dict[str, object]], legend_occupied: np.ndarray
) -> list[dict[str, object]] | None:
    h, w = legend_occupied.shape
    wall_colors = [int(legend["wall_color"]) for legend in legends]

    region_colors = wall_colors.copy()
    while len(region_colors) < MAX_TOTAL_REGIONS and rng.random() < 0.6:
        region_colors.append(rng.choice(wall_colors))
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions: list[dict[str, object]] = []

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
        legend = next(legend for legend in legends if int(legend["wall_color"]) == wall_color)
        regions.append(
            {
                "color": wall_color,
                "dir_map": dict(legend["dir_map"]),
                "interior": blob,
                "walls": walls,
                "placements": [],
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
    if {int(reg["color"]) for reg in regions} != set(wall_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [set(reg["interior"]) for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _showcase_candidate_ok(color: int, free_steps: int) -> bool:
    if color == YELLOW:
        return free_steps >= 1
    if color == ORANGE:
        return free_steps >= 2 and free_steps // 2 < free_steps
    if color == PINK:
        return free_steps >= 1
    return False


def _extra_candidate_ok(color: int, free_steps: int) -> bool:
    if color == YELLOW:
        return free_steps >= 1
    if color == ORANGE:
        return free_steps >= 2 and free_steps // 2 < free_steps
    if color == PINK:
        return free_steps >= 1
    return False


def _build_candidates(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dir_map: dict[int, tuple[int, int]],
    colors: tuple[int, ...],
    validator,
) -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {}
    for color in colors:
        dr, dc = dir_map[color]
        cands: list[dict[str, object]] = []
        for start in blob:
            path = _path_until_blocked(blob, start, dr, dc, set())
            free_steps = len(path)
            if not validator(color, free_steps):
                continue
            moved_steps = _move_steps_for_color(color, free_steps)
            dest = start if moved_steps == 0 else path[moved_steps - 1]
            cands.append(
                {
                    "color": color,
                    "start": start,
                    "output": dest,
                    "path": tuple(path),
                    "path_set": set(path),
                    "free_steps": free_steps,
                    "moved_steps": moved_steps,
                }
            )
        rng.shuffle(cands)
        cands.sort(key=lambda item: (item["free_steps"], item["moved_steps"]), reverse=True)
        out[color] = cands
    return out


def _cands_compatible(a: dict[str, object], b: dict[str, object]) -> bool:
    if a["start"] == b["start"] or a["output"] == b["output"]:
        return False
    if a["output"] == b["start"] or b["output"] == a["start"]:
        return False
    if a["path_set"] & b["path_set"]:
        return False
    if b["start"] in a["path_set"] or a["start"] in b["path_set"]:
        return False
    if b["output"] in a["path_set"] or a["output"] in b["path_set"]:
        return False
    return True


def _find_showcase_plan(
    rng: random.Random, region: dict[str, object]
) -> list[tuple[tuple[int, int], tuple[int, int], int]] | None:
    blob = set(region["interior"])
    dir_map = dict(region["dir_map"])
    if dir_map[YELLOW] == dir_map[ORANGE]:
        return None

    candidates = _build_candidates(rng, blob, dir_map, MOVER_COLORS, _showcase_candidate_ok)
    pinks = candidates[PINK][:60]
    yellows = candidates[YELLOW][:80]
    oranges = candidates[ORANGE][:80]

    for pink in pinks:
        for yellow in yellows:
            if not _cands_compatible(pink, yellow):
                continue
            for orange in oranges:
                if not _cands_compatible(pink, orange):
                    continue
                if not _cands_compatible(yellow, orange):
                    continue
                movers = [
                    (pink["start"][0], pink["start"][1], PINK),
                    (yellow["start"][0], yellow["start"][1], YELLOW),
                    (orange["start"][0], orange["start"][1], ORANGE),
                ]
                simulated = _simulate_region_movers(blob, movers, dir_map)
                out_map = {color: (r, c) for r, c, color in simulated}
                if out_map[PINK] != pink["start"]:
                    continue
                if out_map[YELLOW] != yellow["output"] or out_map[ORANGE] != orange["output"]:
                    continue
                if out_map[YELLOW] == out_map[ORANGE]:
                    continue
                return [
                    (pink["start"], out_map[PINK], PINK),
                    (yellow["start"], out_map[YELLOW], YELLOW),
                    (orange["start"], out_map[ORANGE], ORANGE),
                ]
    return None


def _find_extra_plan(
    rng: random.Random, region: dict[str, object]
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    blob = set(region["interior"])
    dir_map = dict(region["dir_map"])
    color_order = [YELLOW, ORANGE, PINK]
    rng.shuffle(color_order)
    candidates = _build_candidates(rng, blob, dir_map, tuple(color_order), _extra_candidate_ok)
    for color in color_order:
        for cand in candidates[color][:30]:
            movers = [(cand["start"][0], cand["start"][1], color)]
            simulated = _simulate_region_movers(blob, movers, dir_map)
            if len(simulated) != 1:
                continue
            r, c, _ = simulated[0]
            if (r, c) != cand["output"]:
                continue
            return [(cand["start"], cand["output"], color)]
    return []


def _place_movers(rng: random.Random, regions: list[dict[str, object]], grid: np.ndarray) -> bool:
    showcase_idx = None
    showcase_plan = None
    region_order = sorted(range(len(regions)), key=lambda i: len(regions[i]["interior"]), reverse=True)
    for idx in region_order:
        plan = _find_showcase_plan(rng, regions[idx])
        if plan is not None:
            showcase_idx = idx
            showcase_plan = plan
            break
    if showcase_idx is None or showcase_plan is None:
        return False

    regions[showcase_idx]["placements"] = showcase_plan
    for start, _out, color in showcase_plan:
        grid[start] = color

    extra_done = False
    for idx in region_order:
        if idx == showcase_idx:
            continue
        if rng.random() < 0.5:
            continue
        plan = _find_extra_plan(rng, regions[idx])
        if not plan:
            continue
        regions[idx]["placements"] = plan
        for start, _out, color in plan:
            grid[start] = color
        extra_done = True
        if extra_done:
            break

    present = {int(v) for v in grid.flatten() if int(v) in MOVER_COLORS}
    return present == set(MOVER_COLORS)


def _oracle_check(input_grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = input_grid.copy()
    all_placements = [placement for reg in regions for placement in reg.get("placements", [])]
    if not all_placements:
        return False

    for start, _out, _color in all_placements:
        expected[start] = 0
    for _start, out, color in all_placements:
        expected[out] = color

    if not np.array_equal(solver_out, expected):
        return False

    if not bool(np.any((input_grid == YELLOW) & (solver_out != YELLOW))):
        return False
    if not bool(np.any((input_grid == ORANGE) & (solver_out != ORANGE))):
        return False
    if not bool(np.array_equal(np.argwhere(input_grid == PINK), np.argwhere(solver_out == PINK))):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = int(reg["color"])

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
