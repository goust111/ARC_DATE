# Variant of 88e364bc
# Dimension: D22 — 移动像素颜色与角色 (src)
# Change: 移动像素统一为黄色（颜色4），所有黄色像素行为相同... → 多种颜色的移动像素
#
# Design analysis
# - Concrete appearance:
#   - The border legends and colored wall regions stay the same as the source task.
#   - The change is inside each enclosed region: mover pixels are no longer all yellow.
#   - We use three visible mover colors with fixed roles:
#     - yellow (4): full slide, same as the source task
#     - orange (7): half slide, i.e. it moves floor(max_free_run / 2)
#     - pink (6): static anchor, it never moves
#   - Pink is still inside the region, so it is part of the same passable interior for region
#     detection, but during motion it behaves like an already-fixed blocker for pixels behind it.
# - Behavior trace on one channel:
#   - Source logic: if a yellow pixel has 5 free forward steps before a wall, it moves all 5.
#   - Variant logic:
#     - yellow moves all 5
#     - orange moves only 2
#     - pink moves 0 and can stop a later mover from reaching the wall
#   - This changes both the solver's per-pixel stopping rule and the generator's placement
#     strategy, because reverse placement for orange cannot reuse the yellow-only assumption.
# - Complexity calibration:
#   - Only three mover colors are used, and they repeat across samples.
#   - Wall colors are restricted to [3, 5, 8, 9] so orange/pink are never confused with walls.
#   - Each generated grid contains at least one yellow, one orange, and one pink so the color-role
#     mapping is observable from 2-4 examples.
# - Implicit assumptions in the source solver that break under D22:
#   - `mover = 4` assumed exactly one mover color. Now passability must include {4, 6, 7}.
#   - `if v == bg or v == mover` when reading border colors assumed only yellow was interior
#     noise. Now all mover colors must be excluded from border-color voting.
#   - Clearing old movers and repainting a single color was harmless before; now repaint must
#     preserve per-pixel mover color, otherwise orange/pink would be overwritten as yellow.
#   - The old `_slide_points` computed one stopping rule for every mover. Now motion depends on
#     color, so stopping distance must be computed per point after measuring its free run.
# - Placement legality:
#   - Regions are still grown first and validated as enclosed background components.
#   - Mover planning is done in solver order (farther starts first), so the generator uses the
#     same blocker model as `main()`.
#   - Orange candidates are only accepted when their free run is even and at least 2 cells, so
#     the "half distance" effect is visible and unambiguous in generated samples.
# - Visual observability:
#   - Every sample includes all three mover colors.
#   - Yellow and orange both move in the same legend-controlled direction but by different amounts,
#     while pink stays fixed; this is directly visible in input/output pairs.
# - Cleanliness:
#   - The legend system, wall thickness, and region topology are preserved from the source task, so
#     the only new visual burden is the extra mover colors inside the existing empty regions.

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]

YELLOW = 4
PINK = 6
ORANGE = 7
MOVER_COLORS = (YELLOW, PINK, ORANGE)
ACTIVE_MOVER_COLORS = (YELLOW, ORANGE)
WALL_COLOR_CHOICES = [3, 5, 8, 9]

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

GEN_RETRIES = 350
SHOWCASE_ORDER_OPTIONS = [
    [PINK, ORANGE, YELLOW],
    [PINK, YELLOW, ORANGE],
    [ORANGE, PINK, YELLOW],
]
EXTRA_COLORS = [YELLOW, ORANGE, PINK]


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


def _move_steps_for_color(color: int, free_steps: int) -> int:
    if color == YELLOW:
        return free_steps
    if color == ORANGE:
        return free_steps // 2
    if color == PINK:
        return 0
    raise ValueError(f"unexpected mover color {color}")


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
        nr, nc = r + dr, c + dc
        if (nr, nc) not in region_set:
            break
        if dr != 0 and dc != 0:
            if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                break
        if (nr, nc) in occupied:
            break
        r, c = nr, nc
        path.append((r, c))
    return path


def _slide_colored_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts: list[tuple[int, int, int]] = []
    for r0, c0, color in points_sorted:
        path = _path_until_blocked(region_set, (r0, c0), dr, dc, occupied)
        steps = _move_steps_for_color(color, len(path))
        if steps > 0:
            r, c = path[steps - 1]
        else:
            r, c = r0, c0
        occupied.add((r, c))
        out_pts.append((r, c, color))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = (g == bg)
    for color in MOVER_COLORS:
        passable |= (g == color)

    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c), int(g[r, c])) for r, c in pts if int(g[r, c]) in MOVER_COLORS]
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
                if v == bg or v in MOVER_COLORS:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c, _ in movers:
            out[r, c] = bg

        new_pts = _slide_colored_points(pset, movers, dr, dc)
        for r, c, color in new_pts:
            out[r, c] = color

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


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
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


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
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


def _forward_role(
    blob: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    color: int,
) -> tuple[tuple[int, int], int, int]:
    path = _path_until_blocked(blob, start, dr, dc, blocked)
    steps = _move_steps_for_color(color, len(path))
    if steps == 0:
        return start, 0, len(path)
    return path[steps - 1], steps, len(path)


def _is_valid_showcase(color: int, moved: int, free_steps: int) -> bool:
    if color == PINK:
        return True
    if color == YELLOW:
        return moved >= 1
    if color == ORANGE:
        return moved >= 1 and free_steps >= 2 and free_steps % 2 == 0
    return False


def _is_valid_extra(color: int, moved: int, free_steps: int) -> bool:
    if color == PINK:
        return True
    if color == YELLOW:
        return moved >= 1
    if color == ORANGE:
        return moved >= 1 and free_steps >= 2 and free_steps % 2 == 0
    return False


def _plan_region_colors(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    colors: list[int],
    validator,
) -> list[tuple[tuple[int, int], tuple[int, int], int]] | None:
    cells_sorted = sorted(blob, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)

    def dfs(
        color_idx: int,
        min_cell_idx: int,
        blocked_outputs: set[tuple[int, int]],
        used_inputs: set[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], tuple[int, int], int]] | None:
        if color_idx == len(colors):
            return []

        color = colors[color_idx]
        candidates = []
        for idx in range(min_cell_idx, len(cells_sorted)):
            start = cells_sorted[idx]
            if start in blocked_outputs or start in used_inputs:
                continue
            end, moved, free_steps = _forward_role(blob, start, dr, dc, blocked_outputs, color)
            if end in blocked_outputs:
                continue
            if not validator(color, moved, free_steps):
                continue
            candidates.append((idx, start, end))

        if not candidates:
            return None

        head = candidates[: min(len(candidates), 24)]
        tail = candidates[min(len(candidates), 24):]
        rng.shuffle(head)
        candidates = sorted(head, key=lambda x: x[0]) + tail

        for idx, start, end in candidates:
            rest = dfs(color_idx + 1, idx + 1, blocked_outputs | {end}, used_inputs | {start})
            if rest is not None:
                return [(start, end, color)] + rest
        return None

    return dfs(0, 0, set(), set())


def _place_movers(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    region_order = sorted(range(len(regions)), key=lambda i: len(regions[i]["interior"]), reverse=True)

    showcase_region = None
    showcase_plan = None
    for idx in region_order:
        reg = regions[idx]
        dr, dc = reg["dir"]
        for order in SHOWCASE_ORDER_OPTIONS:
            plan = _plan_region_colors(rng, reg["interior"], dr, dc, order, _is_valid_showcase)
            if plan is not None:
                showcase_region = idx
                showcase_plan = plan
                break
        if showcase_plan is not None:
            break
    if showcase_plan is None or showcase_region is None:
        return False

    for start, _end, color in showcase_plan:
        grid[start] = color

    for idx, reg in enumerate(regions):
        if idx == showcase_region:
            continue
        n_extra = rng.choice([0, 1])
        if n_extra == 0:
            continue
        colors = [rng.choice(EXTRA_COLORS) for _ in range(n_extra)]
        plan = _plan_region_colors(rng, reg["interior"], reg["dir"][0], reg["dir"][1], colors, _is_valid_extra)
        if plan is None:
            continue
        for start, _end, color in plan:
            grid[start] = color

    present = {int(v) for v in grid.flatten() if int(v) in MOVER_COLORS}
    return present == set(MOVER_COLORS)


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    mover_mask = np.isin(input_grid, list(MOVER_COLORS))
    struct_mask = (~mover_mask) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False

    for color in MOVER_COLORS:
        if int((solver_out == color).sum()) != int((input_grid == color).sum()):
            return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[np.isin(solver_out, list(MOVER_COLORS))])):
        return False

    if not bool(np.any((input_grid == YELLOW) & (solver_out != YELLOW))):
        return False
    if not bool(np.any((input_grid == ORANGE) & (solver_out != ORANGE))):
        return False
    if not bool(np.array_equal(np.argwhere(input_grid == PINK), np.argwhere(solver_out == PINK))):
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
    if not _place_movers(rng, regions, input_grid):
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
