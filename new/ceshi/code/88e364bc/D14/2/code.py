# Variant of 88e364bc
# Dimension: D14 — 滑动后变换效果 (transform)
# Change: 无额外变换——黄色像素仅移动位置，不改变颜色或产生其他效果... → 黄色像素消失、墙壁标记碰撞点

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis
# a) Concrete behavior:
#    - The input stays structurally identical to the original task: border legends encode a direction,
#      irregular colored walls enclose passable regions, and yellow pixels sit inside those regions.
#    - The physical simulation is still "slide yellows until blocked by the far side of the region".
#    - The output no longer shows the yellow terminals. Instead, every yellow disappears and the wall
#      face it ultimately reaches is painted red (color 2). For diagonal motion, the visible "impact"
#      is the front blocking wall face of the final frontier cell, so one terminal can color one or a
#      small 2x2-front subset of wall cells.
#    - Small trace:
#      original: yellow at (r,c) -> slide right -> stop at interior cell just before wall -> paint that
#      interior cell yellow.
#      variant: run the same slide, clear all yellows, then paint the wall cell(s) immediately beyond
#      that terminal frontier in red.
# b) Complexity calibration:
#    - Legends, wall colors, irregular regions, and cardinal/diagonal directions are preserved.
#    - Generator actively prefers distinct wall-impact signatures so the new transform is visible and
#      does not collapse into "one hidden slide result with no observable difference".
# c) Implicit-assumption audit for main():
#    - Region detection via passable=(bg|yellow) still holds: yellows remain inside passable interiors.
#    - Border-color inference still holds: walls are unchanged in the input.
#    - The original "repaint terminal yellow" step is no longer valid; unconditional repaint would
#      destroy the variant. It is replaced by "project terminal to blocking wall face, then mark red".
#    - A terminal may be blocked by another yellow in the original stack order. To avoid losing the
#      wall collision, impact projection must continue geometrically toward the wall while ignoring
#      yellow occupancy; otherwise only the front-most yellow would produce a marker.
# d) Placement legality:
#    - Generator keeps legend placement and region-growth legality from the original task.
#    - Chosen output terminals must have at least one valid reverse-slide origin and at least one wall
#      cell in their forward impact signature. Distinct terminals are chosen from distinct signatures.
# e) Visual observability:
#    - The wall-mark output is directly visible because yellows vanish entirely and new red cells appear
#      on walls, not inside regions. The generator avoids ambiguous same-signature duplicates when using
#      two yellows in one region.
# f) Visual cleanliness:
#    - Only wall cells are recolored in the output. Legends stay untouched and wall colors never use 2,
#      so red impact marks remain separable from ordinary walls.
# g) Color inference:
#    - The only new color behavior is "yellow motion manifests as red impact marks on walls". This is a
#      single fixed transform with strong positional structure, so it is inferable from a few examples.
# New solver assumptions guaranteed by the generator:
#    - Every region containing yellows is enclosed by non-red walls and has a matching legend direction.
#    - Every chosen terminal has a non-empty forward wall-impact signature.
#    - Distinct yellow terminals in one region use distinct impact signatures, so the variant remains
#      visually expressive instead of collapsing to repeated marks on a single wall face.

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
MAX_SLIDE_STEPS = 5
TERMINAL_CHOICE_RETRIES = 24

GEN_RETRIES = 300

BG = 0
CENTER = 1
MARKER = 2
MOVER = 4


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
        if len(others) < 2:
            continue
        others.sort()
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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
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


def _impact_wall_cells_grid(
    region_set: set[tuple[int, int]],
    g: np.ndarray,
    start: tuple[int, int],
    dr: int,
    dc: int,
    bg: int,
    mover: int,
) -> tuple[tuple[int, int], ...]:
    h, w = g.shape
    r, c = start
    while _slide_step_valid(region_set, r, c, dr, dc):
        r += dr
        c += dc

    if dr == 0 or dc == 0:
        frontier = [(r + dr, c + dc)]
    else:
        frontier = [(r + dr, c), (r, c + dc), (r + dr, c + dc)]

    marks = []
    for wr, wc in frontier:
        if not (0 <= wr < h and 0 <= wc < w):
            continue
        if (wr, wc) in region_set:
            continue
        v = int(g[wr, wc])
        if v == bg or v == mover:
            continue
        marks.append((wr, wc))
    return tuple(sorted(set(marks)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
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

        new_pts = _slide_points(pset, movers, dr, dc)
        impact_marks = set()
        for pt in new_pts:
            impact_marks.update(_impact_wall_cells_grid(pset, g, pt, dr, dc, bg, MOVER))
        for r, c in impact_marks:
            out[r, c] = MARKER

    return out


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = CENTER
    g[1, 2] = CENTER
    g[1, 3] = MARKER
    g[3, 1] = CENTER
    g[3, 2] = CENTER
    g[3, 3] = MARKER
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = CENTER
    g[2, 1] = CENTER
    g[2, 4] = CENTER
    g[3, 2] = CENTER
    g[3, 5] = MARKER
    g[4, 3] = MARKER
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


def _impact_wall_cells_region(
    region_set: set[tuple[int, int]],
    wall_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], ...]:
    r, c = start
    while _slide_step_valid(region_set, r, c, dr, dc):
        r += dr
        c += dc

    if dr == 0 or dc == 0:
        frontier = [(r + dr, c + dc)]
    else:
        frontier = [(r + dr, c), (r, c + dc), (r + dr, c + dc)]

    return tuple(sorted({cell for cell in frontier if cell in wall_set}))


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
    for c in all_colors:
        region_colors.extend([c] * rng.randint(*REGIONS_PER_COLOR))
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
    labeled, n_comp = label_mask(grid == BG, connectivity=4)
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


def _find_reverse_slide(
    blob: set[tuple[int, int]],
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
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        walls = reg["walls"]
        dr, dc = reg["dir"]

        by_signature: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
        for cell in blob:
            if _forward_slide(blob, cell, dr, dc, set()) != cell:
                continue
            sig = _impact_wall_cells_region(blob, walls, cell, dr, dc)
            if not sig:
                continue
            by_signature.setdefault(sig, []).append(cell)
        if not by_signature:
            return False

        possible_counts = [n for n in YELLOWS_PER_REGION if len(by_signature) >= n]
        if not possible_counts:
            return False

        success = False
        for _ in range(TERMINAL_CHOICE_RETRIES):
            n_yellows = rng.choice(possible_counts)
            chosen_sigs = rng.sample(list(by_signature), n_yellows)
            output_positions = [by_signature[sig][rng.randrange(len(by_signature[sig]))] for sig in chosen_sigs]
            out_sorted = sorted(
                output_positions, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True
            )
            already_stopped = set()
            yellows_input = []

            for out_pos in out_sorted:
                candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
                if candidate is None:
                    break
                yellows_input.append(candidate)
                already_stopped.add(out_pos)
            else:
                for r, c in yellows_input:
                    grid[r, c] = MOVER
                reg["impact_marks"] = set().union(*chosen_sigs)
                reg["output_terminals"] = set(output_positions)
                success = True
                break

        if not success:
            return False
    return True


def _expected_output(struct_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = struct_grid.copy()
    for reg in regions:
        for r, c in reg["impact_marks"]:
            out[r, c] = MARKER
    return out


def _oracle_check(input_grid: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    if not np.array_equal(solver_out, expected_out):
        return False
    return int((solver_out == MOVER).sum()) == 0


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

    expected_out = _expected_output(grid, regions)
    if not _oracle_check(input_grid, expected_out):
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
