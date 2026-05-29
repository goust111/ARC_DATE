# Variant of 88e364bc
# Dimension: D5 — 移动对象（源）类型 (src)
# Change: control_pixels——黄色单像素（颜色4），每个区域1-2个，作为被移动的对象... → 多像素精灵（2x2色块）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

# Design analysis for the D5 variant
# spec_variant:
#   core_change:
#     from: each mover is one yellow cell
#     to: each mover is a solid 2x2 yellow sprite; its top-left anchor is the logical position
#   concrete_visual_form:
#     legends, wall colors, irregular enclosed regions, and legend-to-wall color matching stay unchanged
#     every non-empty region now contains 1-2 separate 2x2 yellow squares in the input
#     the output shows the same 2x2 squares slid rigidly to the far side of the region
#   behavior_trace:
#     original:
#       mover extraction treated every yellow cell independently
#       slide step only checked one destination cell (plus diagonal corner-cut test)
#       stacking meant "next yellow occupies the nearest free cell behind the previous yellow"
#     variant:
#       mover extraction first groups yellow cells into 4-connected components and each component must be a filled 2x2 box
#       a slide step is legal only if the whole destination 2x2 box fits in the region
#       for diagonal motion, the intermediate horizontal-shifted and vertical-shifted 2x2 boxes must also fit
#       stacking happens between whole 2x2 boxes, so the next sprite stops when its next step would overlap an already-stopped sprite
#     small_example:
#       if direction is right and a sprite anchor is at (5, 7), the whole box covers cols 7-8
#       it can move to anchor (5, 8) only if cols 8-9 are all inside the region
#       if another stopped sprite already occupies cols 10-11, this sprite finally stops at cols 8-9
#       the old point solver would instead let the four yellow cells separate and fill cols 8-11, which is incorrect here
#   complexity_calibration:
#     keep the sprite monochrome and square so the only new concept is rigid-body occupancy, not internal pattern matching
#     keep sprite count low (1-2 per region) and region count moderate so the board remains readable
#     require generator-grown regions to have multiple valid 2x2 anchors, otherwise the variant becomes visually absent or physically impossible
#   hidden_assumptions_audited:
#     - passable = bg or yellow still works because yellow cells are traversable interior cells
#     - old movers = list(all yellow coords) no longer works; it assumed one cell == one object
#     - old _slide_points checked collision per cell, which would split a 2x2 sprite into four particles
#     - old reverse generation used a 1-cell "stack behind" offset; for 2x2 sprites the packed offset is 2 cells along the move vector
#     - input yellow components must stay 4-disconnected; otherwise solver could not recover individual sprites from the input
#   placement_legality:
#     generator only accepts regions with enough valid 2x2 anchors
#     reverse-slid input sprites may not overlap stopped output sprites or other input sprites
#     input sprites also may not be edge-touching, so each sprite remains a separate 4-connected yellow component
#   observability:
#     every generated puzzle uses 2x2 yellow areas, so the source-type change is visible before any reasoning
#     the same input run through the original point-based solver would usually produce a different output because point motion shears the blocks apart
#   cleanliness:
#     no extra markers are added; the visual layers remain legend / walls / empty interior / yellow sprites
#   new_solver_assumptions_and_generator_support:
#     - each yellow component is a full 2x2 box: enforced by generator placement
#     - each chosen input sprite has at least one valid forward slide step: enforced by reverse generation with steps >= 1
#     - regions provide enough 2x2 support under their assigned direction: enforced by blob validation plus oracle checking


GRID_H, GRID_W = 20, 20
SPRITE_SIZE = 2
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (20, 85),
    2: (20, 55),
    3: (18, 38),
    4: (18, 28),
}
BLOB_GROW_RETRIES = 25
MIN_ANCHOR_COUNT = 4

SPRITES_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5

GEN_RETRIES = 300


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


def _block_cells(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = anchor
    return [(r + dr, c + dc) for dr in range(SPRITE_SIZE) for dc in range(SPRITE_SIZE)]


def _block_cell_set(anchor: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_block_cells(anchor))


def _anchor_fits(region_set: set[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    return all(cell in region_set for cell in _block_cells(anchor))


def _anchor_cells(region_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    anchors = set()
    for r, c in region_set:
        anchor = (int(r), int(c))
        if _anchor_fits(region_set, anchor):
            anchors.add(anchor)
    return anchors


def _block_step_valid(region_set: set[tuple[int, int]], anchor: tuple[int, int], dr: int, dc: int) -> bool:
    r, c = anchor
    nxt = (r + dr, c + dc)
    if not _anchor_fits(region_set, nxt):
        return False
    if dr != 0 and dc != 0:
        if not _anchor_fits(region_set, (r + dr, c)):
            return False
        if not _anchor_fits(region_set, (r, c + dc)):
            return False
    return True


def _blocks_touch(a: tuple[int, int], b: tuple[int, int]) -> bool:
    aset = _block_cell_set(a)
    bset = _block_cell_set(b)
    if aset & bset:
        return True
    for r, c in aset:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _slide_blocks(
    region_set: set[tuple[int, int]],
    anchors: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not anchors:
        return []
    anchors_sorted = sorted(anchors, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied_cells: set[tuple[int, int]] = set()
    out_anchors = []
    for anchor0 in anchors_sorted:
        anchor = anchor0
        while True:
            if not _block_step_valid(region_set, anchor, dr, dc):
                break
            nxt = (anchor[0] + dr, anchor[1] + dc)
            if _block_cell_set(nxt) & occupied_cells:
                break
            anchor = nxt
        occupied_cells.update(_block_cells(anchor))
        out_anchors.append(anchor)
    return out_anchors


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


def _pointwise_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
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
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = MOVER
    return out


def _extract_sprite_anchors(
    g: np.ndarray, pts: np.ndarray, mover: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    yellow_set = {(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover}
    if not yellow_set:
        return [], []
    anchors = []
    used_cells = []
    remaining = set(yellow_set)
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in comp or cur not in remaining:
                continue
            comp.add(cur)
            remaining.remove(cur)
            for nxt in neighbors4(*cur):
                if nxt in remaining:
                    stack.append(nxt)
        if len(comp) != 4:
            raise ValueError("yellow component is not a 2x2 sprite")
        coords = np.array(list(comp), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (SPRITE_SIZE, SPRITE_SIZE):
            raise ValueError("yellow component bbox is not 2x2")
        if len(comp) != SPRITE_SIZE * SPRITE_SIZE:
            raise ValueError("yellow component area mismatch")
        expected = _block_cell_set((r0, c0))
        if comp != expected:
            raise ValueError("yellow component is not a filled 2x2 block")
        anchors.append((r0, c0))
        used_cells.extend(sorted(comp))
    return anchors, used_cells


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

        anchors, mover_cells = _extract_sprite_anchors(g, pts, MOVER)
        if not anchors:
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

        for r, c in mover_cells:
            out[r, c] = bg

        new_anchors = _slide_blocks(pset, anchors, dr, dc)
        for anchor in new_anchors:
            for r, c in _block_cells(anchor):
                out[r, c] = MOVER

    return out


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
        if len(_anchor_cells(cand)) < MIN_ANCHOR_COUNT:
            continue
        return cand
    return None


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


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
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
        anchors = _anchor_cells(blob)
        regions.append(
            {"color": rc, "dir": (dr, dc), "interior": blob, "walls": walls, "anchors": anchors}
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


def _forward_slide_block(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked_cells: set[tuple[int, int]],
) -> tuple[int, int]:
    anchor = start
    while True:
        if not _block_step_valid(region_set, anchor, dr, dc):
            break
        nxt = (anchor[0] + dr, anchor[1] + dc)
        if _block_cell_set(nxt) & blocked_cells:
            break
        anchor = nxt
    return anchor


def _non_overlapping_anchor_choices(
    tier: list[tuple[int, int]], chosen: tuple[int, int]
) -> list[tuple[int, int]]:
    return [anchor for anchor in tier if not (_block_cell_set(anchor) & _block_cell_set(chosen))]


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_anchor: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped_cells: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    existing_input_cells = set()
    for anchor in existing_inputs:
        existing_input_cells.update(_block_cells(anchor))

    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        anchor = out_anchor
        path_ok = True
        for _ in range(steps):
            prev = (anchor[0] - dr, anchor[1] - dc)
            if not _block_step_valid(blob, prev, dr, dc):
                path_ok = False
                break
            if _block_cell_set(prev) & already_stopped_cells:
                path_ok = False
                break
            if _block_cell_set(prev) & existing_input_cells:
                path_ok = False
                break
            anchor = prev
        if not path_ok:
            continue
        if any(_blocks_touch(anchor, other) for other in existing_inputs):
            continue
        if _forward_slide_block(blob, anchor, dr, dc, already_stopped_cells) == out_anchor:
            return anchor
    return None


def _place_sprites(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        anchors = sorted(reg["anchors"], key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
        if not anchors:
            return False

        desired = rng.choice(SPRITES_PER_REGION)
        far_key = dr * anchors[0][0] + dc * anchors[0][1]
        tier1 = [p for p in anchors if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_anchors = [first]

        if desired == 2:
            back = (first[0] - 2 * dr, first[1] - 2 * dc)
            cands = []
            if back in reg["anchors"] and not (_block_cell_set(back) & _block_cell_set(first)):
                cands.append(back)
            cands.extend(
                anchor
                for anchor in _non_overlapping_anchor_choices(tier1, first)
                if anchor not in cands
            )
            if cands:
                output_anchors.append(cands[rng.randrange(len(cands))])

        out_sorted = sorted(output_anchors, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
        already_stopped_cells: set[tuple[int, int]] = set()
        input_anchors: list[tuple[int, int]] = []

        for out_anchor in out_sorted:
            candidate = _find_reverse_slide(
                blob, out_anchor, dr, dc, already_stopped_cells, input_anchors, rng
            )
            if candidate is None:
                return False
            input_anchors.append(candidate)
            already_stopped_cells.update(_block_cells(out_anchor))

        for anchor in input_anchors:
            for r, c in _block_cells(anchor):
                grid[r, c] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == MOVER).sum()) != int((input_grid == MOVER).sum()):
        return False
    if int((solver_out == MOVER).sum()) % (SPRITE_SIZE * SPRITE_SIZE) != 0:
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == MOVER]))


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
    if not _place_sprites(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    if np.array_equal(main(input_grid), _pointwise_baseline(input_grid)):
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
