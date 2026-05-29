# Combo variant of 88e364bc (index 14)
# combo_spec: D9/2+D16/0
# Name: 骑士步移动+允许角切割

"""
Design analysis for the D9/2 + D16/0 fused variant.

This fusion keeps the original chamber-and-legend structure of 88e364bc, but changes how motion is
coded and how path legality is interpreted:
  - D9/2 replaces unit cardinal/diagonal sliding with repeated knight jumps.
  - D16/0 removes any intermediate-cell gating, so a jump is legal whenever its landing square is
    still inside the chamber, even if every plausible "through the corner" corridor is missing.

a) Concrete behavior and trace
  - Visual form:
    - Grid remains 20x20 with bg=0.
    - Each legend is still a solid border-touching rectangle whose fill color matches one wall
      color. The fused variant uses a single 7x7 legend family for all 8 directions.
    - Three blue cells at the legend center form a short bar; one red cell sits at one of the 8
      knight offsets from the central blue cell, directly encoding a 2:1 displacement.
    - Yellow movers still occupy enclosed chambers bounded by colored 1-cell-thick walls.
  - Trace on one chamber:
    - Suppose a legend encodes (+2,+1), a yellow starts at (8,6), and the chamber contains
      (10,7) but omits (9,6), (9,7), and (10,6) because of a corner notch.
    - Original task logic would collapse the legend to the sign (+1,+1) and then apply a diagonal
      step that fails because the orthogonal support cells are missing.
    - Fused logic preserves the full knight vector and permits the jump because only the landing
      square matters, so the yellow moves directly from (8,6) to (10,7).
    - Repeating the same offset may produce (12,8), (14,9), ... until the next landing square is
      outside the chamber or already occupied by an earlier-stopped yellow.
  - Difference:
    - D9/2 matters because the solver must keep the exact 2:1 displacement instead of reducing it
      to a sign-only arrow.
    - D16/0 matters because those knight jumps are allowed across visibly missing corner corridor
      cells; the generator enforces at least one such jump per sample.

b) Complexity calibration
  - Direction categories stay at 8, matching the original branch count while remaining visually
    learnable.
  - All legends use one 7x7 format, reducing incidental variation unrelated to the fused rule.
  - Samples still use 1-2 legend colors and 1-4 chambers, so one direction typically governs
    multiple movers/regions and the mapping is not a one-off coincidence.
  - The generator requires at least one yellow to travel at least two knight jumps and at least one
    executed jump to exploit the relaxed intermediate-cell rule.

c) Implicit assumption audit against the original main()
  - `_legend_directions(...)` in the original solver assumed sign(centroid delta) was sufficient.
    That would alias (+2,+1) with (+1,+1). The fused solver must preserve the full centroid delta
    and snap it to the nearest knight vector.
  - `_slide_points(...)` originally assumed one-cell steps and special diagonal corner checks.
    Those assumptions are both invalid now: movement is a repeated jump, and D16/0 explicitly says
    not to block it based on intermediate support cells.
  - The far-to-near processing order still matters for stacking, but the projection key must use
    the full knight vector so (+2,+1) and (+1,+2) do not collapse together.
  - Chamber detection via 4-connected passable components remains valid because the wall topology is
    unchanged and yellows are still passable.
  - Border-color inference from 4-neighbor wall counts still holds because walls are still painted
    as 1-cell orthogonal rings around chamber interiors.

d) Placement legality
  - Legends are padded away from chambers, so border legend rectangles never merge with wall rings.
  - Chambers are grown first as interior blobs, then their outer 4-neighbor rings are painted as
    walls; this guarantees valid enclosed passable components.
  - Reverse placement only accepts a yellow input position if a forward fused simulation returns
    exactly to the chosen output stop under the same stacking order.
  - The generator also checks a stricter "no corner cutting" knight baseline and rejects samples
    unless fused output differs from that stricter/original-style behavior.

e) Visual observability
  - The legend visibly shows an L-shaped offset, so the human can infer the move is not a straight
    ray.
  - At least one mover performs multiple repeated jumps, making the behavior clearly iterative.
  - At least one executed jump crosses a visible notch/corner gap that would block any unit-step or
    corridor-constrained interpretation, so the relaxed-path aspect is not hidden.

f) Visual cleanliness
  - No extra markers are added inside chambers beyond the existing yellow movers.
  - Chambers remain separated by wall/buffer spacing, legends stay in corners, and colors keep the
    original roles, so the puzzle stays readable despite the new movement rule.

g) Color inferability
  - Colors remain structurally meaningful rather than numeric:
      bg=0, center=1, indicator=2, mover=4, wall colors from {3,5,6,7,8,9}.
  - The rule depends only on visible geometry and color matching, never on comparing raw color ids.

h) New solver assumptions and generator guarantees
  - Each legend's center/indicator delta must decode unambiguously to one knight vector.
  - Every chamber containing yellow must admit at least one valid reverse knight path.
  - Every accepted grid must satisfy three non-degeneracy checks:
      1. at least one yellow moves for >= 2 knight steps,
      2. at least one executed jump is legal only because intermediate corridor cells are ignored,
      3. fused output differs from a stricter/original-style baseline on the same input.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

KNIGHT_DIRS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (20, 85),
    2: (18, 55),
    3: (15, 36),
    4: (14, 28),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 4
GEN_RETRIES = 450

LAST_GENERATION_INFO: dict[str, int] = {}


def _projection(pt: tuple[int, int], dr: int, dc: int) -> int:
    return dr * int(pt[0]) + dc * int(pt[1])


def _collapsed_dir(dr: int, dc: int) -> tuple[int, int]:
    return ((1 if dr > 0 else -1), (1 if dc > 0 else -1))


def _nearest_knight_dir(vr: float, vc: float) -> tuple[int, int] | None:
    best_dir = None
    best_dist = float("inf")
    for dr, dc in KNIGHT_DIRS:
        dist = (vr - dr) ** 2 + (vc - dc) ** 2
        if dist < best_dist:
            best_dist = dist
            best_dir = (dr, dc)
    if best_dist > 0.75:
        return None
    return best_dir


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and decode them as knight directions.

    Each legend is a border-touching filled rectangle with exactly three colors:
    the dominant fill color (wall link), the more central center color, and the red indicator.
    The full center->indicator centroid delta is snapped to one of the 8 knight vectors.
    """
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
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rect_center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tie_break = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tie_break.append((np.sum((pts_col.mean(axis=0) - rect_center) ** 2), col))
            tie_break.sort()
            center_col = tie_break[0][1]
            indicator_col = tie_break[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue

        delta = indicator.mean(axis=0) - center.mean(axis=0)
        knight_dir = _nearest_knight_dir(float(delta[0]), float(delta[1]))
        if knight_dir is None:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), knight_dir)

    return {k: v for k, (_, v) in best.items()}


def _knight_corridors(r: int, c: int, dr: int, dc: int) -> list[list[tuple[int, int]]]:
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    if abs(dr) == 2 and abs(dc) == 1:
        return [
            [(r + sr, c), (r + 2 * sr, c), (r + 2 * sr, c + sc)],
            [(r, c + sc), (r + sr, c + sc), (r + 2 * sr, c + sc)],
        ]
    return [
        [(r, c + sc), (r, c + 2 * sc), (r + sr, c + 2 * sc)],
        [(r + sr, c), (r + sr, c + sc), (r + sr, c + 2 * sc)],
    ]


def _strict_knight_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    landing = (r + dr, c + dc)
    if landing not in region_set:
        return False
    return any(all(cell in region_set for cell in corridor) for corridor in _knight_corridors(r, c, dr, dc))


def _jump_uses_corner_cut(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    landing = (r + dr, c + dc)
    return landing in region_set and not _strict_knight_step_valid(region_set, r, c, dr, dc)


def _slide_points_with_stats(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    if not points:
        return [], {"total_steps": 0, "cut_steps": 0, "max_steps": 0, "moved_points": 0}
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    total_steps = 0
    cut_steps = 0
    max_steps = 0
    moved_points = 0
    for r0, c0 in points_sorted:
        r, c = r0, c0
        steps = 0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set or (nr, nc) in occupied:
                break
            if _jump_uses_corner_cut(region_set, r, c, dr, dc):
                cut_steps += 1
            r, c = nr, nc
            steps += 1
        if steps > 0:
            moved_points += 1
        total_steps += steps
        max_steps = max(max_steps, steps)
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts, {
        "total_steps": total_steps,
        "cut_steps": cut_steps,
        "max_steps": max_steps,
        "moved_points": moved_points,
    }


def _legacy_slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
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


def _apply_solver(
    g: np.ndarray,
    dirs: dict[int, tuple[int, int]],
    *,
    legacy_mode: bool,
    collect_stats: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    passable = (g == bg) | (g == mover)
    out = g.copy()
    total_stats = {"total_steps": 0, "cut_steps": 0, "max_steps": 0, "moved_points": 0, "regions": 0}

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

        if legacy_mode:
            new_pts = _legacy_slide_points(pset, movers, dr, dc)
        else:
            new_pts, stats = _slide_points_with_stats(pset, movers, dr, dc)
            if collect_stats:
                total_stats["total_steps"] += stats["total_steps"]
                total_stats["cut_steps"] += stats["cut_steps"]
                total_stats["max_steps"] = max(total_stats["max_steps"], stats["max_steps"])
                total_stats["moved_points"] += stats["moved_points"]
                total_stats["regions"] += 1
        for r, c in new_pts:
            out[r, c] = mover

    return out, total_stats


def _legacy_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    raw_dirs = _legend_directions(g, bg)
    legacy_dirs = {col: _collapsed_dir(dr, dc) for col, (dr, dc) in raw_dirs.items()}
    out, _ = _apply_solver(g, legacy_dirs, legacy_mode=True, collect_stats=False)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside colored wall chambers by repeatedly applying knight-jump legends.

    Input structure:
    - The most common color is the background; yellow pixels sit inside enclosed passable chambers.
    - Chambers are bounded by colored wall rings, and border legends are solid rectangles whose fill
      color matches one wall color.
    - Each legend uses a blue center mark plus a red indicator offset at one of the 8 knight
      positions, so the legend encodes an exact L-shaped move rather than a unit direction sign.

    Rule:
    - For each enclosed chamber, infer its wall color from adjacent wall cells and match that color
      to the legend with the same fill color.
    - Every yellow in that chamber repeatedly jumps by the legend's knight vector.
    - A jump is legal whenever its landing square stays inside the chamber and is not already taken
      by a previously stopped yellow in that chamber; no intermediate corner or corridor cells are
      checked.

    Key insight:
    - The legend's fill color selects which chamber it controls, while the exact red-from-blue
      offset gives the full 2:1 knight displacement. Because corner cutting is allowed, the motion
      depends only on the sequence of landing squares, not on any cells between them.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"
    out, _ = _apply_solver(g, dirs, legacy_mode=False, collect_stats=False)
    return out


def _legend_7x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((7, 7), fill, dtype=int)
    g[3, 2:5] = 1
    g[3 + dr, 3 + dc] = 2
    return g


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


def _forward_slide_with_trace(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[tuple[int, int], dict[str, int]]:
    r, c = start
    steps = 0
    cut_steps = 0
    while True:
        nr, nc = r + dr, c + dc
        if (nr, nc) not in region_set or (nr, nc) in blocked:
            break
        if _jump_uses_corner_cut(region_set, r, c, dr, dc):
            cut_steps += 1
        r, c = nr, nc
        steps += 1
    return (r, c), {"steps": steps, "cut_steps": cut_steps}


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    directions = rng.sample(KNIGHT_DIRS, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []

    for corner, wall_color, (dr, dc) in zip(corners, wall_colors, directions):
        patch = _legend_7x7(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))

    return legend_occupied, legends


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
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
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wall_color: direction for wall_color, direction in legends}
    all_colors = [wall_color for wall_color, _ in legends]

    region_colors = []
    for wall_color in all_colors:
        region_colors.extend([wall_color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

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
        regions.append(
            {
                "color": wall_color,
                "dir": color_dir[wall_color],
                "interior": blob,
                "walls": walls,
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


def _candidate_output_groups(
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    frontier = [pt for pt in blob if (pt[0] + dr, pt[1] + dc) not in blob]
    if not frontier:
        return [], []
    frontier_sorted = sorted(frontier, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    best_key = _projection(frontier_sorted[0], dr, dc)
    top_tier = [pt for pt in frontier_sorted if _projection(pt, dr, dc) == best_key]
    return frontier_sorted, top_tier


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    min_steps: int,
    require_cut_jump: bool,
    rng: random.Random,
) -> tuple[tuple[int, int], dict[str, int]] | None:
    step_options = list(range(min_steps, MAX_SLIDE_STEPS + 1))
    rng.shuffle(step_options)
    step_options.sort(reverse=True)
    candidates = []

    for steps in step_options:
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            r, c = pr, pc
        if not path_ok:
            continue
        final_pos, trace = _forward_slide_with_trace(blob, (r, c), dr, dc, already_stopped)
        if final_pos != out_pos:
            continue
        candidate_stats = {"steps": trace["steps"], "cut_steps": trace["cut_steps"]}
        candidates.append(((r, c), candidate_stats))

    if not candidates:
        return None

    cut_candidates = [cand for cand in candidates if cand[1]["cut_steps"] > 0]
    if require_cut_jump and cut_candidates:
        return cut_candidates[rng.randrange(len(cut_candidates))]
    if require_cut_jump:
        return None
    return candidates[rng.randrange(len(candidates))]


def _place_yellows(
    rng: random.Random,
    regions: list[dict],
    grid: np.ndarray,
) -> dict[str, int] | None:
    has_long_jump = False
    has_cut_jump = False
    step_total = 0
    cut_step_total = 0

    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)
        frontier_sorted, top_tier = _candidate_output_groups(blob, dr, dc)
        if not frontier_sorted:
            return None

        first_pool = top_tier + frontier_sorted[: min(6, len(frontier_sorted))]
        first_pool = list(dict.fromkeys(first_pool))
        placed = False

        for _ in range(24):
            first = first_pool[rng.randrange(len(first_pool))]
            output_positions = [first]
            if n_yellows == 2:
                second_candidates = []
                prev = (first[0] - dr, first[1] - dc)
                if prev in blob:
                    second_candidates.append(prev)
                second_candidates.extend(pt for pt in top_tier if pt != first)
                second_candidates = list(dict.fromkeys(second_candidates))
                if not second_candidates:
                    continue
                output_positions.append(second_candidates[rng.randrange(len(second_candidates))])

            out_sorted = sorted(output_positions, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
            already_stopped: set[tuple[int, int]] = set()
            input_positions = []
            local_steps = []
            local_cut_steps = 0
            local_has_long = has_long_jump
            local_has_cut = has_cut_jump

            for out_pos in out_sorted:
                preferred_min = 2 if not local_has_long else 1
                need_cut = not local_has_cut
                candidate = _find_reverse_slide(
                    blob,
                    out_pos,
                    dr,
                    dc,
                    already_stopped,
                    input_positions,
                    preferred_min,
                    need_cut,
                    rng,
                )
                if candidate is None and preferred_min > 1:
                    candidate = _find_reverse_slide(
                        blob,
                        out_pos,
                        dr,
                        dc,
                        already_stopped,
                        input_positions,
                        1,
                        need_cut,
                        rng,
                    )
                if candidate is None and need_cut:
                    candidate = _find_reverse_slide(
                        blob,
                        out_pos,
                        dr,
                        dc,
                        already_stopped,
                        input_positions,
                        preferred_min,
                        False,
                        rng,
                    )
                if candidate is None and need_cut and preferred_min > 1:
                    candidate = _find_reverse_slide(
                        blob,
                        out_pos,
                        dr,
                        dc,
                        already_stopped,
                        input_positions,
                        1,
                        False,
                        rng,
                    )
                if candidate is None:
                    break

                start, stats = candidate
                input_positions.append(start)
                local_steps.append(stats["steps"])
                local_cut_steps += stats["cut_steps"]
                already_stopped.add(out_pos)
                if stats["steps"] >= 2:
                    local_has_long = True
                if stats["cut_steps"] > 0:
                    local_has_cut = True
            else:
                for r, c in input_positions:
                    grid[r, c] = 4
                step_total += sum(local_steps)
                cut_step_total += local_cut_steps
                has_long_jump = local_has_long
                has_cut_jump = local_has_cut
                placed = True
                break

        if not placed:
            return None

    if not has_long_jump or not has_cut_jump:
        return None
    return {
        "placed_total_steps": step_total,
        "placed_cut_steps": cut_step_total,
        "has_long_jump": int(has_long_jump),
        "has_cut_jump": int(has_cut_jump),
    }


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == 4).sum()) != int((input_grid == 4).sum()):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


def analyze_input(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    fused_out, fused_stats = _apply_solver(g, dirs, legacy_mode=False, collect_stats=True)
    legacy_out = _legacy_baseline(g)
    return {
        "total_knight_steps": fused_stats["total_steps"],
        "cut_jump_steps": fused_stats["cut_steps"],
        "max_knight_steps": fused_stats["max_steps"],
        "moved_points": fused_stats["moved_points"],
        "legacy_diff_cells": int(np.count_nonzero(fused_out != legacy_out)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    global LAST_GENERATION_INFO

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
    placement = _place_yellows(rng, regions, input_grid)
    if placement is None:
        return None
    if not _oracle_check(input_grid, regions):
        return None

    metrics = analyze_input(input_grid)
    if metrics["max_knight_steps"] < 2:
        return None
    if metrics["cut_jump_steps"] < 1:
        return None
    if metrics["legacy_diff_cells"] < 1:
        return None

    LAST_GENERATION_INFO = {**placement, **metrics}
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
