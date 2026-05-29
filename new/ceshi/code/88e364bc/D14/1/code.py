# Variant of 88e364bc
# Dimension: D14 — 滑动后变换效果 (transform)
# Change: 无额外变换——黄色像素仅移动位置，不改变颜色或产生其他效果... → 碰撞墙壁时墙壁局部擦除

# Design analysis for the D14 variant
# a) Concrete behavior:
#    - The input still has border legends, enclosed colored wall rings, and yellow movers inside.
#    - The slide destination is unchanged: yellows stack at the far side dictated by the legend.
#    - New effect: if a yellow stops because the next step is a wall, the single wall pixel directly
#      responsible for that stop is erased to bg=0 in the output.
#    - Small trace: in the original task, a right-moving yellow at "..4|wall" becomes "..04|wall"
#      after clearing old position and repainting at the far-right interior cell. In this variant it
#      becomes "..04 0" across the wall contact: the yellow still ends at the same interior cell, but
#      the adjacent wall pixel on its right is turned into background.
# b) Complexity calibration:
#    - The main relation is still "legend color -> wall color -> slide direction"; only one extra
#      post-slide transform is added.
#    - To keep the erased wall cell visually unambiguous, the generator uses cardinal directions only.
#      Diagonal motion would make the "which wall pixel was hit?" choice harder to infer from 2-3
#      examples, so this is the one compatibility adjustment made for readability.
#    - Samples still contain 1-2 legend colors and 1-4 enclosed regions, so the task keeps repeated
#      evidence rather than collapsing into a single isolated event.
# c) Hidden assumptions audit for main():
#    - `_legend_directions` assumes legends are unchanged rectangular border patches: still true.
#    - Region finding via passable=(bg|yellow) assumes input walls are intact and regions are enclosed:
#      still true in the input, even though the output now opens holes.
#    - Original `_slide_points` only needed final yellow cells; the variant needs the stop cause.
#      The new helper preserves the same far-to-near processing order, but also records whether the
#      stop was caused by a wall or by another yellow. Only wall-caused stops trigger erasure.
#    - Clearing old yellows before repainting is still safe because erased wall cells are outside the
#      passable region and never overlap yellow destinations.
# d) Placement legality:
#    - Legends stay buffered away from region-growth space.
#    - Regions are generated as before: closed in the input, non-overlapping, away from the border.
#    - Yellow target outputs are restricted to cells that are genuinely wall-adjacent in the slide
#      direction and also have a valid reverse-slide predecessor, so the new transform always has a
#      legal wall pixel to erase and the reverse construction remains solvable.
# e) Visual observability:
#    - Every generated sample includes at least one erased wall pixel in the output; otherwise the
#      variant would be visually indistinguishable from the original task.
#    - When a region gets 2 yellows, the generator prefers distinct wall-hit lanes so multiple holes
#      can appear on the same wall face, making the new rule easier to notice.
# f) Visual cleanliness:
#    - Walls remain 1-cell thick, legends stay in corners, and a one-cell reserve separates regions.
#      The output therefore shows isolated wall gaps rather than dense overlaps or ambiguous clutter.
# g) Color-change inferability:
#    - No new color mapping is introduced. The only color change is local wall -> background at impact
#      sites, and those sites are structurally marked by the final yellow positions.
# Additional solver/generator contract:
#    - `main()` now assumes that any generated yellow destined for the variant behavior is adjacent to
#      a real wall cell in its slide direction. `generate_input()` enforces this by only sampling
#      target outputs from verified impact-capable cells.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend blocks on the border.

    A legend is a border-touching non-background component whose most common color is the legend
    background `C` (and also the region border color it controls). The legend contains at least
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


def _impact_wall_cell(
    g: np.ndarray,
    bg: int,
    mover: int,
    region_set: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    """Return the wall cell responsible for the next blocked move, if identifiable."""
    h, w = g.shape
    candidates = []
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


def _slide_points_with_impacts(
    g: np.ndarray,
    bg: int,
    mover: int,
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if not points:
        return [], []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    impact_cells = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        impact = None
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                impact = _impact_wall_cell(g, bg, mover, region_set, r, c, dr, dc)
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    impact = _impact_wall_cell(g, bg, mover, region_set, r, c, dr, dc)
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
        if impact is not None:
            impact_cells.append(impact)
    return out_pts, impact_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions in the direction dictated by border-legend arrows.

    Variant rule:
    - Yellow pixels still slide and stack exactly as in the original task.
    - Additionally, whenever a yellow stops because it has reached a wall, the adjacent wall pixel
      it collided with is erased to background. Stops caused by another yellow do not erase walls.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

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

        new_pts, impact_cells = _slide_points_with_impacts(g, bg, mover, pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover
        for r, c in impact_cells:
            out[r, c] = bg

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 1.0
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
MAX_SLIDE_STEPS = 6

GEN_RETRIES = 300
REGION_OUTPUT_TRIES = 20


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


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


def _impact_cell_for_region(
    region_set: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    candidates = []
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        candidates.append((nr, nc))
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set:
            candidates.append((r + dr, c))
        if (r, c + dc) not in region_set:
            candidates.append((r, c + dc))
    for cand in candidates:
        if cand in walls:
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


def _candidate_outputs(
    blob: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates = []
    for r, c in blob:
        impact = _impact_cell_for_region(blob, walls, r, c, dr, dc)
        if impact is None:
            continue
        pr, pc = r - dr, c - dc
        if (pr, pc) not in blob:
            continue
        if dr != 0 and dc != 0:
            if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                continue
        if _forward_slide(blob, (pr, pc), dr, dc, set()) != (r, c):
            continue
        candidates.append(((r, c), impact))
    return candidates


def _reverse_inputs_for_outputs(
    rng: random.Random,
    blob: set[tuple[int, int]],
    outputs: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]] | None:
    out_sorted = sorted(outputs, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    already_stopped = set()
    yellows_input = []
    for out_pos in out_sorted:
        candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
        if candidate is None:
            return None
        yellows_input.append(candidate)
        already_stopped.add(out_pos)
    return yellows_input


def _choose_region_outputs(
    rng: random.Random, reg: dict
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    blob = reg["interior"]
    walls = reg["walls"]
    dr, dc = reg["dir"]
    candidates = _candidate_outputs(blob, walls, dr, dc)
    if not candidates:
        return None

    candidates.sort(key=lambda item: (dr * item[0][0] + dc * item[0][1], item[0][0], item[0][1]), reverse=True)
    by_impact: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pos, impact in candidates:
        by_impact.setdefault(impact, []).append(pos)

    max_yellows = 2 if len(by_impact) >= 2 else 1
    prefer_two = max_yellows == 2 and rng.random() < 0.55
    counts_to_try = [2, 1] if prefer_two else [1, 2]
    counts_to_try = [n for n in counts_to_try if n <= max_yellows]
    if 1 not in counts_to_try:
        counts_to_try.append(1)

    impacts = list(by_impact)
    for n_yellows in counts_to_try:
        for _ in range(REGION_OUTPUT_TRIES):
            chosen_impacts = rng.sample(impacts, n_yellows)
            outputs = [rng.choice(by_impact[impact]) for impact in chosen_impacts]
            if len(set(outputs)) != n_yellows:
                continue
            inputs = _reverse_inputs_for_outputs(rng, blob, outputs, dr, dc)
            if inputs is None:
                continue
            return outputs, chosen_impacts, inputs
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    total_impacts = 0
    for reg in regions:
        picked = _choose_region_outputs(rng, reg)
        if picked is None:
            return False
        outputs, impact_cells, input_cells = picked
        reg["target_outputs"] = list(outputs)
        reg["impact_cells"] = list(impact_cells)
        reg["input_cells"] = list(input_cells)
        total_impacts += len(impact_cells)
        for r, c in input_cells:
            grid[r, c] = 4
    return total_impacts > 0


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = input_grid.copy()
    expected[input_grid == 4] = 0
    for reg in regions:
        for r, c in reg["impact_cells"]:
            expected[r, c] = 0
        for r, c in reg["target_outputs"]:
            expected[r, c] = 4

    return np.array_equal(solver_out, expected)


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
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
