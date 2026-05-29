# Combo variant of 88e364bc (index 6)
# combo_spec: D3/1+D11/0
# Name: 沿墙滑动+轨迹涂色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


"""
Fusion summary
==============

- D3/1: 黄色像素不再穿过空腔直线滑动，而是沿内壁边界轨道滑动；legend 只编码顺/逆时针。
- D11/0: 输出要显示完整轨迹；轨迹上的非终点格涂成蓝色(1)，终点仍为黄色(4)。

Generator keeps the original closed-room stage, but now enforces that accepted samples contain:
- visible trail cells,
- at least one mover that actually turns a corner while tracing the wall.

That prevents degeneration into either:
- the original straight-slide task with trail only, or
- the boundary-trace task without visible trajectory.
"""


GRID_H, GRID_W = 20, 20
BG_COLOR = 0
TRAIL_COLOR = 1
MOVER = 4

# Legend
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

# Regions
REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (16, 55),
    3: (14, 36),
    4: (12, 26),
}
BLOB_GROW_RETRIES = 35
MIN_BOUNDARY_TRACK = 10

# Yellows / trail visibility
YELLOWS_PER_REGION = [1, 2]
MAX_BOUNDARY_STEPS = 8
MIN_NEW_TRAIL_CELLS = 3
MIN_LONGEST_TRAIL = 2

# Outer loop
GEN_RETRIES = 500


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


def _corner_point(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - 1
    if corner == "BL":
        return h - 1, 0
    if corner == "BR":
        return h - 1, w - 1
    raise ValueError(corner)


def _legend_corner(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> str:
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == w - 1:
        return "TR"
    if r1 == h - 1 and c0 == 0:
        return "BL"
    if r1 == h - 1 and c1 == w - 1:
        return "BR"
    raise ValueError("legend must sit in a corner")


def _legend_5x5(fill: int, clockwise: bool) -> np.ndarray:
    """5x5 legend: red on right => clockwise, red on left => counterclockwise."""
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[3, 1] = 1
    g[3, 2] = 1
    if clockwise:
        g[1, 3] = 2
        g[3, 3] = 2
    else:
        g[1, 0] = 2
        g[3, 0] = 2
    return g


def _legend_rotation_info(g: np.ndarray, bg: int) -> dict[int, tuple[bool, str]]:
    """
    Detect border legends and return {wall_color: (clockwise, corner)}.

    The patch format intentionally mirrors the D3/1 single variant:
    - most frequent color = fill color C = wall color
    - 4 blue cells = center
    - 2 red cells = indicator
    - indicator centroid right of center => clockwise, left => counterclockwise
    """
    h, w = g.shape
    best: dict[int, tuple[int, bool, str]] = {}
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
        if sorted(cnt for cnt, _ in others) != [2, 4]:
            continue
        others.sort()
        indicator_col = others[0][1]
        center_col = others[1][1]

        center_pts = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic_pts = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center_pts or not indic_pts:
            continue

        cc = np.mean(np.array(center_pts, dtype=float), axis=0)
        ci = np.mean(np.array(indic_pts, dtype=float), axis=0)
        dc = float(ci[1] - cc[1])
        if abs(dc) < 1e-6:
            continue
        clockwise = dc > 0
        corner = _legend_corner(r0, c0, r1, c1, h, w)

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), clockwise, corner)

    return {col: (clockwise, corner) for col, (_, clockwise, corner) in best.items()}


def _boundary_cells(region_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c in region_set
        if any((nr, nc) not in region_set for nr, nc in neighbors4(r, c))
    }


def _boundary_cycle_clockwise(region_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Trace the outer contour clockwise and convert it to an ordered list of boundary cells.

    Each exposed cell side contributes one directed edge with the region on its right.
    The generator rejects shapes where a boundary cell would need to appear twice
    non-consecutively, but the solver still collapses to first occurrence to stay total.
    """
    if not region_set:
        return []

    next_edge: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}

    def add_edge(start: tuple[int, int], end: tuple[int, int], cell: tuple[int, int]) -> None:
        if start in next_edge:
            raise ValueError("non-simple contour")
        next_edge[start] = (end, cell)

    for r, c in region_set:
        if (r - 1, c) not in region_set:
            add_edge((r, c), (r, c + 1), (r, c))
        if (r, c + 1) not in region_set:
            add_edge((r, c + 1), (r + 1, c + 1), (r, c))
        if (r + 1, c) not in region_set:
            add_edge((r + 1, c + 1), (r + 1, c), (r, c))
        if (r, c - 1) not in region_set:
            add_edge((r + 1, c), (r, c), (r, c))

    start = min(next_edge)
    cur = start
    raw_cells: list[tuple[int, int]] = []
    seen_vertices = set()
    while True:
        if cur in seen_vertices and cur != start:
            raise ValueError("contour should be a single cycle")
        seen_vertices.add(cur)
        nxt, cell = next_edge[cur]
        raw_cells.append(cell)
        cur = nxt
        if cur == start:
            break

    collapsed: list[tuple[int, int]] = []
    for cell in raw_cells:
        if not collapsed or cell != collapsed[-1]:
            collapsed.append(cell)
    if collapsed and collapsed[0] == collapsed[-1]:
        collapsed.pop()

    ordered_unique: list[tuple[int, int]] = []
    seen_cells: set[tuple[int, int]] = set()
    for cell in collapsed:
        if cell not in seen_cells:
            ordered_unique.append(cell)
            seen_cells.add(cell)

    return ordered_unique


def _ordered_boundary_track(
    region_set: set[tuple[int, int]], corner: str, clockwise: bool, h: int, w: int
) -> list[tuple[int, int]]:
    cycle = _boundary_cycle_clockwise(region_set)
    if not cycle:
        return []
    corner_pt = _corner_point(corner, h, w)
    anchor = min(
        cycle,
        key=lambda p: (abs(p[0] - corner_pt[0]) + abs(p[1] - corner_pt[1]), p[0], p[1]),
    )
    k = cycle.index(anchor)
    rotated = cycle[k:] + cycle[:k]
    if clockwise:
        return rotated
    return [rotated[0]] + list(reversed(rotated[1:]))


def _slide_along_track_with_paths(
    track: list[tuple[int, int]], points: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    if not points:
        return []
    if not track:
        return [(p, [p]) for p in points]

    index = {p: i for i, p in enumerate(track)}
    moving = [p for p in points if p in index]
    fixed = [p for p in points if p not in index]

    occupied: set[int] = set()
    out: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for p in sorted(moving, key=lambda q: index[q], reverse=True):
        start_idx = index[p]
        end_idx = start_idx
        path = [track[start_idx]]
        while end_idx + 1 < len(track) and (end_idx + 1) not in occupied:
            end_idx += 1
            path.append(track[end_idx])
        occupied.add(end_idx)
        out.append((track[end_idx], path))

    for p in fixed:
        out.append((p, [p]))
    return out


def _render_region_with_trails(
    out: np.ndarray, movers: list[tuple[int, int]], track: list[tuple[int, int]]
) -> None:
    for r, c in movers:
        out[r, c] = BG_COLOR

    slides = _slide_along_track_with_paths(track, movers)
    for _final, path in slides:
        for r, c in path[:-1]:
            out[r, c] = TRAIL_COLOR
    for final, _path in slides:
        out[final] = MOVER


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions along the inner boundary track and paint the trajectory.

    Input structure:
    - bg = most common color
    - yellow pixels lie on the inner boundary of enclosed passable regions
    - border legends are 5x5 rectangles; fill color matches wall color
    - red markers on the right/left side of the blue center encode clockwise/counterclockwise

    Rule:
    - For each enclosed region, identify its border color and matching legend.
    - Extract the ordered boundary ring of inner-wall cells.
    - Cut that ring at the boundary cell closest to the legend's corner.
    - Orient the track clockwise or counterclockwise from that cut and slide yellows forward,
      stacking at the far end of the track.
    - Paint every traversed non-final track cell blue(1); final resting cells stay yellow(4).
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_info = _legend_rotation_info(g, bg)
    assert legend_info, "grid must contain at least one border legend"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
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
                if v in (bg, MOVER):
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in legend_info:
            continue
        clockwise, corner = legend_info[border_col]
        track = _ordered_boundary_track(region_set, corner, clockwise, h, w)
        if not track:
            continue

        _render_region_with_trails(out, movers, track)

    return out


def _grow_blob(
    canvas_mask: np.ndarray,
    seed: tuple[int, int],
    target_size: int,
    rng: random.Random,
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


def _shape_has_simple_track(blob: set[tuple[int, int]], corner: str, clockwise: bool, h: int, w: int) -> bool:
    track = _ordered_boundary_track(blob, corner, clockwise, h, w)
    return len(track) >= MIN_BOUNDARY_TRACK and len(track) == len(set(track)) == len(_boundary_cells(blob))


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
    corner: str,
    clockwise: bool,
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
        if not _shape_has_simple_track(cand, corner, clockwise, h, w):
            continue
        return cand
    return None


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, bool, str]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, bool, str]] = []

    for corner, wall_color in zip(corners, wall_colors):
        clockwise = bool(rng.randrange(2))
        patch = _legend_5x5(wall_color, clockwise)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, clockwise, corner))

    return legend_occupied, legends


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, bool, str]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_info = {wall_color: (clockwise, corner) for wall_color, clockwise, corner in legends}
    all_colors = [wall_color for wall_color, _, _ in legends]

    region_colors = []
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for color in region_colors:
        clockwise, corner = color_info[color]
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w, corner, clockwise)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        track = _ordered_boundary_track(blob, corner, clockwise, h, w)
        regions.append(
            {
                "color": color,
                "clockwise": clockwise,
                "corner": corner,
                "interior": blob,
                "walls": walls,
                "track": track,
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
    labeled, n_comp = label_mask(grid == BG_COLOR, connectivity=4)
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


def _forward_slide_index(start_idx: int, blocked: set[int], track_len: int) -> int:
    j = start_idx
    while j + 1 < track_len and (j + 1) not in blocked:
        j += 1
    return j


def _path_has_turn(track: list[tuple[int, int]], start_idx: int, end_idx: int) -> bool:
    if end_idx - start_idx < 2:
        return False
    dirs = []
    for i in range(start_idx, end_idx):
        r0, c0 = track[i]
        r1, c1 = track[i + 1]
        dirs.append((r1 - r0, c1 - c0))
    return any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _find_reverse_track_index(
    track: list[tuple[int, int]],
    out_idx: int,
    blocked_outputs: set[int],
    existing_inputs: set[int],
    require_turn: bool,
) -> int | None:
    candidates: list[int] = []
    max_back = min(MAX_BOUNDARY_STEPS, out_idx)
    for steps in range(max_back, 0, -1):
        cand = out_idx - steps
        if cand in blocked_outputs or cand in existing_inputs:
            continue
        if _forward_slide_index(cand, blocked_outputs, len(track)) == out_idx:
            candidates.append(cand)

    if not candidates:
        return None

    turning = [cand for cand in candidates if _path_has_turn(track, cand, out_idx)]
    longish = [cand for cand in candidates if out_idx - cand >= MIN_LONGEST_TRAIL]

    if require_turn and turning:
        return turning[0]
    if longish:
        return longish[0]
    return candidates[0]


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> dict[str, int] | None:
    metrics = {
        "moved_any": 0,
        "trail_cells": 0,
        "longest_trail": 0,
        "turned_path": 0,
    }

    region_order = list(range(len(regions)))
    rng.shuffle(region_order)
    turning_region = region_order[0]

    for reg_idx, reg in enumerate(regions):
        track = reg["track"]
        track_len = len(track)
        if track_len < MIN_BOUNDARY_TRACK:
            return None

        n_yellows = rng.choice(YELLOWS_PER_REGION)
        if track_len < n_yellows + 2:
            return None

        output_indices = [track_len - 1 - i for i in range(n_yellows)]
        blocked_outputs: set[int] = set()
        input_indices: list[int] = []

        for j, out_idx in enumerate(output_indices):
            require_turn = metrics["turned_path"] == 0 and reg_idx == turning_region and j == 0
            cand = _find_reverse_track_index(
                track,
                out_idx,
                blocked_outputs,
                set(input_indices),
                require_turn=require_turn,
            )
            if cand is None:
                return None
            input_indices.append(cand)
            blocked_outputs.add(out_idx)

            steps = out_idx - cand
            if steps > 0:
                metrics["moved_any"] = 1
                metrics["trail_cells"] += steps
                metrics["longest_trail"] = max(metrics["longest_trail"], steps)
                if _path_has_turn(track, cand, out_idx):
                    metrics["turned_path"] = 1

        for idx in input_indices:
            r, c = track[idx]
            grid[r, c] = MOVER

    return metrics


def _expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for reg in regions:
        movers = [(r, c) for r, c in reg["interior"] if int(input_grid[r, c]) == MOVER]
        if not movers:
            continue
        _render_region_with_trails(out, movers, reg["track"])
    return out


def _oracle_check(input_grid: np.ndarray, regions: list[dict], metrics: dict[str, int]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = _expected_output(input_grid, regions)
    if not np.array_equal(solver_out, expected):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True

    trail_in_regions = int(np.sum((solver_out == TRAIL_COLOR) & regions_mask))
    if trail_in_regions < MIN_NEW_TRAIL_CELLS:
        return False
    if metrics["moved_any"] == 0:
        return False
    if metrics["longest_trail"] < MIN_LONGEST_TRAIL:
        return False
    if metrics["turned_path"] == 0:
        return False
    if int(np.sum(solver_out == MOVER)) != int(np.sum(input_grid == MOVER)):
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
    metrics = _place_yellows(rng, regions, input_grid)
    if metrics is None:
        return None
    if not _oracle_check(input_grid, regions, metrics):
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
