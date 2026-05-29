# Variant of 88e364bc
# Dimension: D8 — 黄色像素数量与分布 (src)
# Change: 每个区域1-2个黄色像素，总计2-6个；部分区域可能无黄色像素（空区域）... → 密集黄色像素（每区域3-5个）

"""
Design analysis for the D8 variant.

source_task: none

spec_variant:
  unchanged_core_rule:
    - Border legends still map wall color -> slide direction.
    - Walls still define enclosed passable regions.
    - Yellow cells still slide simultaneously by solver order: far-to-near along the legend direction.
  changed_dimension:
    - Every generated region now contains 3-5 yellow cells.
    - The final state in each region is a dense consecutive stack pressed against the far wall.
  concrete_visualization:
    - Original task often produced one terminal yellow plus at most one cell behind it.
    - This variant intentionally builds a straight "terminal lane" inside each region:
      wall-side cells [p0, p1, p2, ...] where p0 is the farthest stop, p1 is one step behind, etc.
    - Output yellows occupy p0..p{k-1} for k in [3,5], so the output shows a compact 3-5-cell strip.
    - Input yellows occupy a shifted slice p{d}..p{d+k-1}; after sliding, they collapse onto p0..p{k-1}.
  behavior_trace:
    - Example with direction right and lane cells listed from right wall inward as [p0, p1, p2, p3, p4].
    - Original logic with 2 yellows might use input [p2, p3] -> output [p0, p1].
    - Variant with 4 yellows can use input [p1, p2, p3, p4] -> output [p0, p1, p2, p3].
    - The first processed yellow reaches p0, the next stops at p1 because p0 is now occupied, etc.
    - The visible difference is not just "more yellow": most final yellows are stopped by earlier yellows,
      so the far-to-near processing order is now essential instead of incidental.
  complexity_calibration:
    - Region count is slightly tightened to at most 3, and region minima are raised, so 3-5 yellow stacks fit
      cleanly without making the 20x20 grid unreadable.
    - The generator still allows 1-2 legend colors and irregular regions; it does not add a second rule system.
    - Dense stacks are chosen from a single terminal lane per region, which keeps the logic legible for humans.
  implicit_assumption_audit_main:
    - _legend_directions assumes legends are border-touching solid rectangles with exactly three colors.
      D8 does not affect legends, so this remains valid.
    - main() treats passable cells as bg or yellow and finds enclosed components. This still works with 3-5 yellows
      because all extra yellows remain inside exactly one enclosed region.
    - _slide_points already supports arbitrary mover counts; the only sensitive assumption is the processing order.
      The generator mirrors the exact same projection ordering, so dense stacks stay solver-consistent.
    - Clearing all old yellow cells before repainting is still safe because yellow is used only for movers.
  placement_legality:
    - Regions are still grown as enclosed blobs with non-overlapping 1-cell walls and legend buffers.
    - A region is only accepted if it contains a terminal lane long enough to host 3-5 output cells plus at least
      one reverse offset cell; otherwise generation retries.
    - Input yellows are chosen from one shifted slice of the lane, so they never overlap each other or leave the region.
  visual_observability:
    - In 2-4 examples, humans can directly see that every region has a thicker yellow band in the output than before.
    - Because the band is contiguous and wall-pressed, the role of "stacking" is visually obvious.
  tidiness:
    - The generator avoids empty regions and caps total regions, which prevents the grid from becoming visually noisy.
  new_generator_assumptions:
    - Each generated region must contain a straight terminal lane of length at least 4.
    - generate_input() enforces this by searching for dense lane plans and rejecting failed attempts.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _projection_key(p: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int]:
    return dr * p[0] + dc * p[1], p[0], p[1]


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend blocks on the border.

    A legend is a border-touching non-background component whose most common color is the legend
    background `C` (and also the region border color it controls). The legend contains exactly
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
        if len(others) != 2:
            continue

        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            dists = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                dists.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            dists.sort()
            center_col = dists[0][1]
            indicator_col = dists[1][1]
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

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: _projection_key(p, dr, dc), reverse=True)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside each enclosed region using the legend-mapped direction.

    D8 does not change the solver mechanics; it changes the data distribution so regions now
    contain denser 3-5-cell stacks. The existing far-to-near slide order therefore becomes
    much more visible and must remain exact.
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

        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover

    return out


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 3
REGION_SIZE_BOUNDS = {
    1: (28, 85),
    2: (24, 55),
    3: (20, 38),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [3, 4, 5]
MAX_SLIDE_STEPS = 6

GEN_RETRIES = 400


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wall_color in zip(corners, wall_colors):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wall_color, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))

    return legend_occupied, legends


def _assigned_region_colors(rng: random.Random, all_colors: list[int]) -> list[int]:
    counts = {color: 1 for color in all_colors}
    remaining = MAX_TOTAL_REGIONS - len(all_colors)
    order = rng.sample(all_colors, len(all_colors))
    for color in order:
        if remaining <= 0:
            break
        extra_cap = REGIONS_PER_COLOR[1] - counts[color]
        if extra_cap <= 0:
            continue
        extra = rng.randint(0, min(extra_cap, remaining))
        counts[color] += extra
        remaining -= extra

    region_colors = []
    for color in all_colors:
        region_colors.extend([color] * counts[color])
    rng.shuffle(region_colors)
    return region_colors


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wall_color: direction for wall_color, direction in legends}
    all_colors = [wall_color for wall_color, _ in legends]
    region_colors = _assigned_region_colors(rng, all_colors)

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
            return None

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            return None

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

    return regions


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


def _terminal_lanes(blob: set[tuple[int, int]], dr: int, dc: int) -> list[list[tuple[int, int]]]:
    lanes = []
    for cell in blob:
        if _slide_step_valid(blob, cell[0], cell[1], dr, dc):
            continue
        lane = [cell]
        while True:
            prev = (lane[-1][0] - dr, lane[-1][1] - dc)
            if prev not in blob:
                break
            if not _slide_step_valid(blob, prev[0], prev[1], dr, dc):
                break
            lane.append(prev)
        if len(lane) >= 4:
            lanes.append(lane)
    lanes.sort(key=lambda lane: (_projection_key(lane[0], dr, dc), len(lane)), reverse=True)
    return lanes


def _choose_dense_plan(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> dict | None:
    plans = []
    for lane in _terminal_lanes(blob, dr, dc):
        lane_len = len(lane)
        max_count = min(max(YELLOWS_PER_REGION), lane_len - 1)
        for count in range(min(YELLOWS_PER_REGION), max_count + 1):
            max_offset = min(MAX_SLIDE_STEPS, lane_len - count)
            if max_offset < 1:
                continue
            offsets = list(range(1, max_offset + 1))
            if max_offset >= 2:
                offsets = list(range(2, max_offset + 1)) + [1]
            for offset in offsets:
                plans.append(
                    {
                        "count": count,
                        "offset": offset,
                        "lane_len": lane_len,
                        "output": lane[:count],
                        "input": lane[offset : offset + count],
                    }
                )
    if not plans:
        return None
    plans.sort(key=lambda plan: (plan["count"], plan["offset"], plan["lane_len"]), reverse=True)
    top = plans[: min(10, len(plans))]
    return top[rng.randrange(len(top))]


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        plan = _choose_dense_plan(rng, blob, dr, dc)
        if plan is None:
            return False
        reg["input_yellows"] = list(plan["input"])
        reg["output_yellows"] = list(plan["output"])
        reg["yellow_count"] = int(plan["count"])
        reg["slide_offset"] = int(plan["offset"])
        for r, c in reg["input_yellows"]:
            grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    expected = input_grid.copy()
    expected[expected == 4] = 0
    for reg in regions:
        if len(reg.get("output_yellows", [])) not in YELLOWS_PER_REGION:
            return False
        for r, c in reg["output_yellows"]:
            expected[r, c] = 4
    return np.array_equal(solver_out, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
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
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
