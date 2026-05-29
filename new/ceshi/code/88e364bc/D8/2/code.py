# Variant of 88e364bc
# Dimension: D8 — 黄色像素数量与分布 (src)
# Change: 每个区域1-2个黄色像素，总计2-6个；部分区域可能无黄色像素（空区域）... → 黄色像素初始位置紧贴墙壁（反向滑动）
#
# Design analysis for the D8 variant
# ==================================
# a) Concrete behavior and execution trace
#    - The legends stay unchanged: border rectangles still encode a direction (dr, dc), and the
#      legend fill color still matches the wall color of one or more enclosed regions.
#    - The walls and irregular enclosed regions also stay unchanged.
#    - The only semantic inversion is where the yellow pixels start:
#      original task: input yellows are somewhere inside the region, output yellows end stacked
#      against the far wall in direction (dr, dc).
#      this variant: input yellows already appear in that stacked/far-wall configuration, and the
#      output moves them in the opposite direction (-dr, -dc).
#    - Small trace in a horizontal corridor with legend direction right:
#      original: input at x=3, x=5 -> solver slides right -> output at x=8, x=9.
#      variant: input starts at x=8, x=9 -> solver slides left -> output at the leftward resting
#      positions for the same region. The visible change is "wall-adjacent cluster breaks away and
#      travels backward" instead of "interior pixels gather at the far wall".
#    - To keep the inversion visible, the generator rejects stacked inputs that would remain fixed
#      under reverse motion.
#
# b) Complexity calibration
#    - The task still uses only 1-2 legend colors and the same region/wall machinery, so the
#      concept count does not increase.
#    - We keep 1-2 yellows per populated region and 2-6 total yellows. Some regions may be empty.
#    - The generator only samples stacked states that are fixed points of the original forward
#      slide. This makes the variant visually different from the source task: a solver using the
#      old rule would usually leave the input unchanged, while the new rule moves the yellows away.
#
# c) Implicit-assumption review of main()
#    - _legend_directions(): unchanged. It only assumes legends exist on the border and still
#      encode directions by center->indicator centroid sign. D8 does not alter that.
#    - passable = (g == bg) | (g == mover): unchanged. Empty regions remain discoverable because
#      they are still background-only enclosed components.
#    - components_mask(passable): unchanged. Regions are identified exactly as before.
#    - movers extraction / "if not movers: continue": still valid, and now important because some
#      regions are intentionally empty.
#    - border color inference by adjacent wall counts: unchanged; it depends only on walls.
#    - The key source-task assumption was in the final slide step:
#        new_pts = _slide_points(pset, movers, dr, dc)
#      This assumed yellows should move toward the legend direction. In the variant that assumption
#      is false. The correct update is to move by (-dr, -dc).
#    - New solver assumption introduced by the variant:
#      input yellows should already be in a forward-stacked state so the reversal is visible.
#      The generator guarantees this by constructing only forward-fixed stacked states.
#
# d) Placement legality
#    - Legends are still placed at corners with a one-cell reserved buffer.
#    - Region growth still reserves interior + walls + wall-neighbor buffer, so regions do not
#      overlap and remain enclosed.
#    - Yellow placement now searches only among candidate sets that:
#      1. are fixed by the original forward slide,
#      2. do move under the reversed slide,
#      3. stay inside the same region under the new solver.
#      This keeps inputs legal and ensures main() can solve them.
#
# e) Visual observability
#    - Across 2-4 examples, the repeated pattern is:
#      same legend encoding as the source task, but yellow pixels begin jammed against the
#      direction-indicated far wall. The output consistently moves them in the opposite direction.
#    - A source-task solver would typically predict "no change" or keep the far-wall cluster,
#      while the correct variant pushes the cluster backward. That contrast is visible.
#
# f) Visual cleanliness
#    - No new markers are added.
#    - Empty regions remain clean background pockets bounded by colored walls.
#    - Yellow placement stays near a wall in the input, which makes the source-distribution change
#      easy to see without cluttering the region interiors.
#
# g) Color reasoning
#    - No new color mapping is introduced. The only color-dependent correspondence remains:
#      legend fill color C <-> region wall color C.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and map legend fill color -> direction."""
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


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    """Slide points inside a region until blocked by the wall or already-placed movers."""
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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Read the legend direction exactly as in the source task.
    - Input yellows already start stacked against the far wall for that direction.
    - Output moves them in the opposite direction, i.e. by (-dr, -dc).
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

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

        new_pts = _slide_points(pset, movers, -dr, -dc)
        for r, c in new_pts:
            out[r, c] = mover
    return out


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

TOTAL_YELLOW_RANGE = (2, 6)
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
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
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


def _stacked_state_options(
    blob: set[tuple[int, int]], dr: int, dc: int
) -> dict[int, list[list[tuple[int, int]]]]:
    """Enumerate 1- and 2-yellow inputs that are forward-stacked but reverse-movable."""
    cells = sorted(blob, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    options: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}

    for p in cells:
        pts = [p]
        if set(_slide_points(blob, pts, dr, dc)) != set(pts):
            continue
        if set(_slide_points(blob, pts, -dr, -dc)) == set(pts):
            continue
        options[1].append(pts)

    for p, q in itertools.combinations(cells, 2):
        pts = [p, q]
        if set(_slide_points(blob, pts, dr, dc)) != set(pts):
            continue
        if set(_slide_points(blob, pts, -dr, -dc)) == set(pts):
            continue
        options[2].append(pts)
    return options


def _sample_region_counts(
    rng: random.Random, regions: list[dict], options_by_region: list[dict[int, list[list[tuple[int, int]]]]]
) -> list[int] | None:
    total_min, total_max = TOTAL_YELLOW_RANGE
    weighted_options = []
    for opts in options_by_region:
        counts = [0]
        if opts[1]:
            counts.extend([1, 1, 1])
        if opts[2]:
            counts.extend([2, 2])
        weighted_options.append(counts)

    if sum(max(opts) for opts in weighted_options) < total_min:
        return None

    for _ in range(80):
        counts = [rng.choice(opts) for opts in weighted_options]
        total = sum(counts)
        if total < total_min or total > total_max:
            continue
        if len(regions) > 1 and all(c > 0 for c in counts):
            # Keep empty regions present sometimes; not mandatory in every sample.
            if rng.random() < 0.6:
                continue
        return counts
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    options_by_region = []
    for reg in regions:
        dr, dc = reg["dir"]
        opts = _stacked_state_options(reg["interior"], dr, dc)
        options_by_region.append(opts)

    counts = _sample_region_counts(rng, regions, options_by_region)
    if counts is None:
        return False

    placed_any = False
    for reg, opts, count in zip(regions, options_by_region, counts):
        if count == 0:
            continue
        pts = rng.choice(opts[count])
        for r, c in pts:
            grid[r, c] = 4
        placed_any = True
    return placed_any


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False
    if np.array_equal(solver_out, input_grid):
        return False

    h, w = input_grid.shape
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
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
