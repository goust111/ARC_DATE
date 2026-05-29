# Variant of 88e364bc
# Dimension: D13 — 输出网格形状 (grid_shape)
# Change: same——输出与输入尺寸相同，墙壁和图例不变，仅黄色像素位置改变
#         → 裁剪到仅包含区域的范围

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# -----------------------------------------------------------------------------
# Design analysis for the D13 tight-crop variant
#
# a) Concrete behavior:
#    - Input stays close to the original task: a 20x20 board with 1-2 border legends in
#      corner positions, several enclosed irregular regions bounded by colored walls, and
#      yellow pixels inside those regions.
#    - The slide rule is unchanged: each region reads its wall color, looks up the border
#      legend with the same fill color, and slides its yellows along the legend direction.
#    - The new visible change happens after sliding: instead of returning the full 20x20
#      board, we crop to the minimal axis-aligned rectangle that contains every enclosed
#      region plus its surrounding wall cells. Legends and pure outside background are
#      removed.
#    - Small trace:
#      original: legends in the top corners, one region in the middle. Solver slides
#      yellow(s) but leaves the top legends and all outer black margins in place.
#      variant: the same sliding happens first, then the solver finds the bbox of the
#      region interiors and their walls, and returns only that subgrid.
#
# b) Complexity calibration:
#    - The crop must be visually obvious, so outputs must usually shrink.
#    - We keep 1-2 legends and multiple regions, but we do not allow arbitrary corner
#      combinations. If legends appeared in opposite corners while regions spanned the
#      middle, the tight crop could geometrically intersect a legend. To avoid ambiguous
#      examples, all legends are placed on one side family (top / bottom / left / right),
#      and all regions are generated in the opposite safe band.
#    - Region counts, wall-color linking, and cardinal/diagonal motion are preserved, so
#      the task still needs the original reasoning plus the new crop deduction.
#
# c) main() assumptions audit:
#    - Original `out = g.copy()` assumed output shape equals input shape. This is no longer
#      valid; we now crop at the end.
#    - Original passable-component scan only mattered for moving yellows. In the variant,
#      every enclosed region also contributes to the crop support, even if a region had no
#      mover.
#    - Original code could safely leave legends untouched because they stayed in output.
#      Now legends must be outside the crop rectangle. The solver assumes this geometric
#      separation; the generator enforces it.
#
# d) Placement legality:
#    - Legends still occupy border corners and reserve a buffer.
#    - Region interiors are grown only inside a side-dependent safe mask. This keeps the
#      region-wall union disjoint from legend rows/cols, so the crop rectangle excludes
#      legends rather than slicing through them.
#    - Walls may not touch the outer border or legend-reserved cells.
#
# e) Visual observability:
#    - Across examples, the slide behavior remains visible inside the cropped result, while
#      the output shape itself now changes noticeably.
#    - The crop rule is inferable because the output consistently removes legends and empty
#      margins but keeps all walled regions intact.
#
# f) Visual cleanliness:
#    - The crop support is the union of enclosed passable cells and their adjacent wall
#      cells. This keeps walls around regions visible in the cropped output instead of
#      trimming down to just yellow pixels.
#    - Restricting legends to one side keeps the final cropped rectangle clean.
#
# g) Color reasoning:
#    - No new color mapping is introduced. Wall-color↔legend-color matching is unchanged.
#
# 9) New solver assumptions and generator guarantees:
#    - Assumption: the bbox of all enclosed regions and their walls does not intersect any
#      legend cells. Guarantee: generator places every legend on one side family and every
#      region in the opposite safe band.
#    - Assumption: there is at least one enclosed region. Guarantee: generator rejects
#      empty region sets.
#    - Assumption: the crop should be strictly smaller than the input in at least one
#      dimension. Guarantee: oracle rejects uncropped outputs.
# -----------------------------------------------------------------------------


# ── Configuration ─────────────────────────────────────────────────────────────

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (15, 70),
    2: (15, 42),
    3: (12, 26),
    4: (10, 18),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5

GEN_RETRIES = 300


# ── Solver helpers ───────────────────────────────────────────────────────────

def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and decode their center→indicator direction."""
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


def _bounded_regions(
    g: np.ndarray,
    bg: int,
    mover: int,
) -> list[dict[str, object]]:
    """Return all enclosed passable regions plus their adjacent wall cells."""
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    regions: list[dict[str, object]] = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        border_counts: dict[int, int] = {}
        walls = set()
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                walls.add((nr, nc))
                border_counts[v] = border_counts.get(v, 0) + 1
        border_col = max(border_counts, key=border_counts.get) if border_counts else None
        regions.append(
            {
                "pset": pset,
                "movers": movers,
                "walls": walls,
                "border_col": border_col,
            }
        )
    return regions


def _crop_bbox_from_regions(regions: list[dict[str, object]]) -> tuple[int, int, int, int]:
    crop_cells: list[tuple[int, int]] = []
    for reg in regions:
        crop_cells.extend(reg["pset"])
        crop_cells.extend(reg["walls"])
    assert crop_cells, "tight-crop variant requires at least one enclosed region"
    return bbox_coords(np.array(crop_cells, dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions, then crop away legends and outer margins.

    The slide rule matches the original task. The D13 variant changes only the output
    shape: after movement, return the minimal rectangle containing all enclosed regions and
    their wall boundaries.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    regions = _bounded_regions(g, bg, mover)
    out = g.copy()

    for reg in regions:
        movers = reg["movers"]
        border_col = reg["border_col"]
        if not movers or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        for r, c in movers:
            out[r, c] = bg
        new_pts = _slide_points(reg["pset"], movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover

    r0, c0, r1, c1 = _crop_bbox_from_regions(regions)
    return out[r0 : r1 + 1, c0 : c1 + 1]


# ── Legend patches ───────────────────────────────────────────────────────────

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


def _corner_side(corner: str) -> str:
    if corner in ("TL", "TR"):
        return "top"
    if corner in ("BL", "BR"):
        return "bottom"
    raise ValueError(corner)


def _legend_layout(side: str, rng: random.Random) -> list[str]:
    if side == "top":
        corners = ["TL", "TR"]
    elif side == "bottom":
        corners = ["BL", "BR"]
    elif side == "left":
        corners = ["TL", "BL"]
    elif side == "right":
        corners = ["TR", "BR"]
    else:
        raise ValueError(side)
    n_legends = rng.choice([1, 2])
    return rng.sample(corners, n_legends)


def _side_safe_mask(
    shape: tuple[int, int],
    side: str,
    legend_specs: list[dict[str, object]],
) -> np.ndarray:
    """Interior cells where region blobs may live so crop bbox stays disjoint from legends."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side in ("top", "bottom"):
        reserve_depth = max(spec["patch"].shape[0] + LEGEND_BUFFER for spec in legend_specs)
        if side == "top":
            start = reserve_depth + 1
            stop = h - 1
            if start < stop:
                mask[start:stop, 1 : w - 1] = True
        else:
            start = 1
            stop = h - reserve_depth - 1
            if start < stop:
                mask[start:stop, 1 : w - 1] = True
    else:
        reserve_depth = max(spec["patch"].shape[1] + LEGEND_BUFFER for spec in legend_specs)
        if side == "left":
            start = reserve_depth + 1
            stop = w - 1
            if start < stop:
                mask[1 : h - 1, start:stop] = True
        else:
            start = 1
            stop = w - reserve_depth - 1
            if start < stop:
                mask[1 : h - 1, start:stop] = True
    return mask


# ── Blob growing / topology ─────────────────────────────────────────────────

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


# ── Slide helpers ────────────────────────────────────────────────────────────

def _slide_step_valid(region_set: set, r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(region_set: set, start: tuple[int, int], dr: int, dc: int, blocked: set) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


# ── Generator phases ────────────────────────────────────────────────────────

def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]], np.ndarray]:
    """
    Place 1-2 legends confined to one side family, and return a safe region mask.

    Top/bottom layouts separate legends from regions along rows.
    Left/right layouts separate legends from regions along columns.
    """
    h, w = grid.shape
    side = rng.choice(["top", "bottom", "left", "right"])
    corners = _legend_layout(side, rng)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, len(corners))

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    legend_specs: list[dict[str, object]] = []

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
        legend_specs.append({"corner": corner, "patch": patch})

    safe_mask = _side_safe_mask((h, w), side, legend_specs)
    safe_mask &= ~legend_occupied
    return legend_occupied, legends, safe_mask


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
    safe_mask: np.ndarray,
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

        canvas = safe_mask.copy()
        canvas &= ~occupied
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
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


def _find_reverse_slide(
    blob: set,
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set,
    existing_inputs: list,
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
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        yellows_input = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = 4
    return True


def _regions_crop_bbox(regions: list[dict]) -> tuple[int, int, int, int]:
    coords = []
    for reg in regions:
        coords.extend(reg["interior"])
        coords.extend(reg["walls"])
    return bbox_coords(np.array(coords, dtype=int))


def _expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray | None:
    try:
        solver_out = main(input_grid)
    except Exception:
        return None
    r0, c0, r1, c1 = _regions_crop_bbox(regions)
    full_out = input_grid.copy()
    bg = get_bg(input_grid)
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        movers = [(r, c) for r, c in blob if full_out[r, c] == 4]
        for r, c in movers:
            full_out[r, c] = bg
        new_pts = _slide_points(blob, movers, dr, dc)
        for r, c in new_pts:
            full_out[r, c] = 4
    expected = full_out[r0 : r1 + 1, c0 : c1 + 1]
    return expected if np.array_equal(solver_out, expected) else None


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    expected = _expected_output(input_grid, regions)
    if expected is None:
        return False
    if expected.shape == input_grid.shape:
        return False
    r0, c0, r1, c1 = _regions_crop_bbox(regions)
    if r0 == 0 and c0 == 0 and r1 == input_grid.shape[0] - 1 and c1 == input_grid.shape[1] - 1:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

    legend_occupied, legends, safe_mask = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied, safe_mask)
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
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
