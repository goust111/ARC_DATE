# Combo variant of 88e364bc (index 7)
# combo_spec: D3/3+D14/1
# Name: 反射路径+墙壁碰撞擦除

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


# Design summary
# --------------
# - Solver:
#   legend decoding and enclosed-room detection stay compatible with the original 88e364bc task.
#   The changed behavior is local to yellow motion:
#   1. yellow follows the legend direction for a fixed reflection budget;
#   2. wall hits reflect the direction like a ray;
#   3. every wall hit erases exactly one wall pixel in the output;
#   4. occupied yellow endpoints block motion but do not reflect.
# - Generator:
#   D14/1 makes diagonal impact choice visually less obvious, so generated samples intentionally use
#   cardinal legends only, while `main()` still supports diagonal reflection for completeness.
#   Each sample is built from clean rectangular rooms, and each room gets a start position that is
#   oracle-checked to guarantee:
#   - at least two wall-collision events overall,
#   - at least one reflected endpoint differs from the original straight-slide endpoint,
#   - the output contains visible wall holes rather than degenerating back to the base task.


Coord = tuple[int, int]

GRID_H, GRID_W = 20, 20
BG = 0
MOVER = 4
REFLECT_STEPS = 14

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
GEN_RETRIES = 500
ROOM_ATTEMPTS = 160


class TraceResult:
    __slots__ = ("final", "straight_final", "path", "impacts", "bounces", "moves")

    def __init__(
        self,
        final: Coord,
        straight_final: Coord,
        path: tuple[Coord, ...],
        impacts: tuple[Coord, ...],
        bounces: int,
        moves: int,
    ) -> None:
        self.final = final
        self.straight_final = straight_final
        self.path = path
        self.impacts = impacts
        self.bounces = int(bounces)
        self.moves = int(moves)


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


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> Coord:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, Coord]:
    h, w = g.shape
    best: dict[int, tuple[int, Coord]] = {}
    for pts in components_mask(g != bg, connectivity=4):
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

        fill = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill]
        if len(others) != 2:
            continue
        others.sort()

        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            scored = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                scored.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            scored.sort()
            center_col = scored[0][1]
            indicator_col = scored[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indic = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indic) == 0:
            continue

        delta = indic.mean(axis=0) - center.mean(axis=0)
        dr = 0 if abs(float(delta[0])) < 1e-6 else (1 if delta[0] > 0 else -1)
        dc = 0 if abs(float(delta[1])) < 1e-6 else (1 if delta[1] > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), (dr, dc))
    return {col: direction for col, (_, direction) in best.items()}


def _slide_step_valid(region_set: set[Coord], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide_linear(
    region_set: set[Coord],
    start: Coord,
    dr: int,
    dc: int,
    blocked: set[Coord],
) -> Coord:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _impact_wall_cell(
    g: np.ndarray,
    bg: int,
    mover: int,
    region_set: set[Coord],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> Coord | None:
    h, w = g.shape
    candidates: list[Coord] = []
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        candidates.append((nr, nc))
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set:
            candidates.append((r + dr, c))
        if (r, c + dc) not in region_set:
            candidates.append((r, c + dc))
    for rr, cc in candidates:
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != mover:
                return (rr, cc)
    return None


def _move_if_possible(
    region_set: set[Coord],
    occupied: set[Coord],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[Coord | None, str]:
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


def _reflected_direction(region_set: set[Coord], r: int, c: int, dr: int, dc: int) -> Coord | None:
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
    changed = False
    if row_block:
        new_dr = -new_dr
        changed = True
    if col_block:
        new_dc = -new_dc
        changed = True
    if diag_block and not row_block and not col_block:
        new_dr = -new_dr
        new_dc = -new_dc
        changed = True
    if not changed:
        return None
    return (int(new_dr), int(new_dc))


def _trace_reflection(
    g: np.ndarray,
    bg: int,
    mover: int,
    region_set: set[Coord],
    start: Coord,
    dr: int,
    dc: int,
    occupied: set[Coord],
    *,
    energy: int = REFLECT_STEPS,
) -> TraceResult:
    straight_final = _forward_slide_linear(region_set, start, dr, dc, occupied)
    r, c = start
    cur_dr, cur_dc = int(dr), int(dc)
    path: list[Coord] = []
    impacts: list[Coord] = []
    bounces = 0

    for _ in range(int(energy)):
        moved, reason = _move_if_possible(region_set, occupied, r, c, cur_dr, cur_dc)
        if moved is not None:
            r, c = moved
            path.append((r, c))
            continue
        if reason == "occupied":
            break

        impact = _impact_wall_cell(g, bg, mover, region_set, r, c, cur_dr, cur_dc)
        if impact is not None:
            impacts.append(impact)

        new_dir = _reflected_direction(region_set, r, c, cur_dr, cur_dc)
        if new_dir is None or new_dir == (cur_dr, cur_dc):
            break
        cur_dr, cur_dc = new_dir
        bounces += 1

        moved, _reason2 = _move_if_possible(region_set, occupied, r, c, cur_dr, cur_dc)
        if moved is None:
            break
        r, c = moved
        path.append((r, c))

    return TraceResult(
        final=(r, c),
        straight_final=straight_final,
        path=tuple(path),
        impacts=tuple(impacts),
        bounces=bounces,
        moves=len(path),
    )


def _reflect_points_with_impacts(
    g: np.ndarray,
    bg: int,
    mover: int,
    region_set: set[Coord],
    points: list[Coord],
    dr: int,
    dc: int,
) -> tuple[list[Coord], list[TraceResult]]:
    if not points:
        return [], []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[Coord] = set()
    out_pts: list[Coord] = []
    traces: list[TraceResult] = []
    for start in points_sorted:
        trace = _trace_reflection(g, bg, mover, region_set, start, dr, dc, occupied)
        occupied.add(trace.final)
        out_pts.append(trace.final)
        traces.append(trace)
    return out_pts, traces


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    out = g.copy()
    passable = (g == bg) | (g == MOVER)
    region_stats: list[dict] = []

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in region_set:
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

        finals, traces = _reflect_points_with_impacts(g, bg, MOVER, region_set, movers, dr, dc)
        erased: set[Coord] = set()
        for trace in traces:
            erased.update(trace.impacts)
        for r, c in erased:
            out[r, c] = bg
        for r, c in finals:
            out[r, c] = MOVER

        region_stats.append(
            {
                "region_set": region_set,
                "movers": tuple(movers),
                "dir": (dr, dc),
                "border_color": int(border_col),
                "traces": tuple(traces),
                "erased": tuple(sorted(erased)),
            }
        )

    return out, region_stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels by the legend direction, but now the path reflects on walls.

    Combo rule:
    - A yellow pixel moves step by step for a fixed reflection budget.
    - If its next step hits a wall, it bounces like a ray and keeps moving in the reflected direction.
    - Every wall collision erases the single collided wall cell in the output.
    - If the next step is blocked by an already-settled yellow, the mover simply stops with no erasure.
    """
    out, _stats = _solve_with_stats(np.asarray(grid, dtype=int))
    return out


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, Coord]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    occupied = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, Coord]] = []
    for corner, wall_color in zip(corners, wall_colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))
    return occupied, legends


def _room_dimensions_for_dir(dr: int, dc: int, rng: random.Random) -> tuple[int, int]:
    if dr == 0:
        return rng.randint(5, 6), rng.randint(8, 10)
    return rng.randint(8, 10), rng.randint(5, 6)


def _rect_room(r0: int, c0: int, oh: int, ow: int) -> tuple[set[Coord], set[Coord]]:
    walls: set[Coord] = set()
    interior: set[Coord] = set()
    for r in range(r0, r0 + oh):
        for c in range(c0, c0 + ow):
            if r in (r0, r0 + oh - 1) or c in (c0, c0 + ow - 1):
                walls.add((r, c))
            else:
                interior.add((r, c))
    return interior, walls


def _try_place_room(
    rng: random.Random,
    occupied: np.ndarray,
    wall_color: int,
    direction: Coord,
) -> dict | None:
    h, w = occupied.shape
    dr, dc = direction
    for _ in range(ROOM_ATTEMPTS):
        oh, ow = _room_dimensions_for_dir(dr, dc, rng)
        r0 = rng.randint(1, h - oh - 1)
        c0 = rng.randint(1, w - ow - 1)

        r_lo = max(0, r0 - 1)
        r_hi = min(h, r0 + oh + 1)
        c_lo = max(0, c0 - 1)
        c_hi = min(w, c0 + ow + 1)
        if occupied[r_lo:r_hi, c_lo:c_hi].any():
            continue

        interior, walls = _rect_room(r0, c0, oh, ow)
        for r in range(r_lo, r_hi):
            for c in range(c_lo, c_hi):
                occupied[r, c] = True
        return {
            "color": int(wall_color),
            "dir": (int(dr), int(dc)),
            "interior": interior,
            "walls": walls,
            "outer": (r0, c0, oh, ow),
        }
    return None


def _build_regions(rng: random.Random, legend_occupied: np.ndarray, legends: list[tuple[int, Coord]]) -> list[dict] | None:
    occupied = legend_occupied.copy()
    regions: list[dict] = []
    for wall_color, direction in legends:
        region = _try_place_room(rng, occupied, wall_color, direction)
        if region is None:
            return None
        regions.append(region)
    return regions


def _lane_cells(region: dict) -> list[Coord]:
    interior = region["interior"]
    dr, dc = region["dir"]
    rows = sorted({r for r, _c in interior})
    cols = sorted({c for _r, c in interior})
    if dr == 0:
        lane_r = rows[len(rows) // 2]
        return sorted([(r, c) for r, c in interior if r == lane_r], key=lambda p: p[1])
    lane_c = cols[len(cols) // 2]
    return sorted([(r, c) for r, c in interior if c == lane_c], key=lambda p: p[0])


def _choose_start_for_region(base_grid: np.ndarray, region: dict, rng: random.Random) -> tuple[Coord, TraceResult] | None:
    dr, dc = region["dir"]
    candidates = _lane_cells(region)
    rng.shuffle(candidates)
    for start in candidates:
        trace = _trace_reflection(base_grid, BG, MOVER, region["interior"], start, dr, dc, set())
        unique_impacts = set(trace.impacts)
        if trace.final == start:
            continue
        if trace.bounces < 2:
            continue
        if len(unique_impacts) < 2:
            continue
        if trace.final == trace.straight_final:
            continue
        if trace.moves < 6:
            continue
        return start, trace
    return None


def _paint_rooms(grid: np.ndarray, regions: list[dict]) -> None:
    for region in regions:
        for r, c in region["walls"]:
            grid[r, c] = region["color"]


def _expected_output_from_regions(input_grid: np.ndarray, regions: list[dict]) -> tuple[np.ndarray, list[TraceResult]]:
    out = input_grid.copy()
    out[out == MOVER] = BG
    traces: list[TraceResult] = []
    for region in regions:
        starts = region["starts"]
        dr, dc = region["dir"]
        finals, region_traces = _reflect_points_with_impacts(
            input_grid,
            BG,
            MOVER,
            region["interior"],
            list(starts),
            dr,
            dc,
        )
        erased: set[Coord] = set()
        for trace in region_traces:
            traces.append(trace)
            erased.update(trace.impacts)
        for r, c in erased:
            out[r, c] = BG
        for r, c in finals:
            out[r, c] = MOVER
    return out, traces


def _oracle_accepts(input_grid: np.ndarray, regions: list[dict]) -> bool:
    expected, traces = _expected_output_from_regions(input_grid, regions)
    try:
        solver_out, solver_stats = _solve_with_stats(input_grid)
    except Exception:
        return False
    if not np.array_equal(expected, solver_out):
        return False

    all_traces = list(traces)
    if not all_traces:
        return False
    if not any(trace.final != trace.straight_final for trace in all_traces):
        return False
    if sum(len(set(trace.impacts)) for trace in all_traces) < 2:
        return False
    if not any(trace.bounces >= 2 for trace in all_traces):
        return False

    erased_total = sum(len(stats["erased"]) for stats in solver_stats)
    return erased_total >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _build_regions(rng, legend_occupied, legends)
    if regions is None:
        return None

    _paint_rooms(grid, regions)

    for region in regions:
        chosen = _choose_start_for_region(grid, region, rng)
        if chosen is None:
            return None
        start, trace = chosen
        region["starts"] = [start]
        region["trace_preview"] = trace
        grid[start] = MOVER

    if not _oracle_accepts(grid, regions):
        return None
    return grid


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
