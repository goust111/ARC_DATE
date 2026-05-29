# Combo variant of 88e364bc (index 15)
# combo_spec: D11/0+D14/2
# Name: 轨迹涂色+碰撞标记

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

"""
Design analysis for the D11/0 + D14/2 combo variant.

a) Concrete behavior and trace:
   - The input keeps the original object grammar: 20x20 grid, corner legends, colored wall loops,
     and yellow movers inside enclosed passable regions.
   - The physical motion also stays the same: per region, decode a direction from the matching
     border legend, then slide yellows from the far side inward so they stack.
   - The rendering changes in two coupled places:
     1. every cell actually traversed by a mover becomes blue(1);
     2. the mover itself disappears, and the wall face aligned with its terminal direction becomes
        red(2).
   - Small trace:
     original: yellow path [(5,5),(5,6),(5,7)] with terminal at (5,7) -> output only shows (5,7)=4.
     fused: clear the yellow, paint (5,5),(5,6),(5,7)=1, then paint the wall cell just beyond the
     terminal frontier as 2. Output contains only the blue tail plus the red impact mark.
   - Difference from D11-only: there is no surviving yellow endpoint.
   - Difference from D14-only: the interior route is now visible, not just the wall impact.

b) Complexity calibration:
   - Keep one global trail color (blue) and one global impact color (red). This adds exactly two
     consistent visual roles without introducing per-color remapping.
   - Preserve 1-2 legends and up to 4 regions so examples still repeatedly support the original
     wall-color -> direction binding.
   - The generator enforces visible trail length and at least one new impact mark, preventing the
     combo from collapsing into a near-original or single-step-only sample.

c) Implicit-assumption audit for main():
   - `passable = (bg | yellow)` still works because trails and impact marks are output-only.
   - Region-border inference by adjacent wall colors still works because inputs keep the original
     wall palette and never pre-place trail or impact cells in interiors.
   - Returning only final terminals is no longer sufficient: the solver must retain every path cell.
   - Repainting terminals yellow is now invalid. The fused rendering paints the full path blue and
     omits yellow entirely.
   - A rear mover may stop behind another mover, but its impact marker should still refer to the far
     wall face in that direction. Impact projection therefore ignores final occupancies and advances
     geometrically to the region boundary.

d) Placement legality:
   - Region growth and legend placement stay within the original bounds, with legend buffers and
     closed-wall validation preserved.
   - Reverse-slide placement still guarantees each input yellow has a legal origin and moves at least
     one step.
   - For multi-yellow regions, the generator samples distinct wall-impact signatures so the red
     markings remain informative instead of duplicating onto one wall face.

e) Visual observability:
   - Human-solvable examples need both effects to be obvious. The accepted samples therefore contain
     visible new blue trail cells, visible new red wall cells, and no remaining yellow terminals.
   - Because the trail sits strictly inside regions while the impact mark sits on the wall frontier,
     the two outputs reinforce each other: one shows the route, the other shows where that route ends.

f) Visual cleanliness:
   - Only two added mark types appear inside/near regions: blue interior trails and red wall hits.
   - Trails never overwrite walls, and impact marks never spill into the passable interior.
   - Walls and legends otherwise keep their original shapes, so the scene stays readable.

g) Inference and new assumptions guaranteed by the generator:
   - Blue always means "cell traversed by a vanished yellow".
   - Red always means "wall location aligned with the mover's final collision direction".
   - Every generated sample has at least one mover with a 2+ step trail and at least one new impact
     mark, so both halves of the combo are evidenced in the training distribution.
   - The solver assumes legends are still detectable as border-touching 3-color rectangles and that
     the only yellow interior cells in the input are mover starts; the generator preserves both.
"""


GRID_H, GRID_W = 20, 20

BG = 0
TRAIL = 1
CENTER = 1
MARKER = 2
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
    1: (15, 85),
    2: (15, 55),
    3: (12, 35),
    4: (12, 25),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
TERMINAL_CHOICE_RETRIES = 24

MIN_NEW_TRAIL_CELLS = 3
MIN_LONGEST_TRAIL = 2
MIN_NEW_MARKER_CELLS = 1

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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _slide_paths(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        path = [(r0, c0)]
        while True:
            if not _slide_step_valid(region_set, r, c, dr, dc):
                break
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
            path.append((r, c))
        occupied.add((r, c))
        out.append(((r, c), path))
    return out


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels by the legend directions, draw their whole interior route, and mark the wall hit.

    Input structure:
    - The background is black, legends sit on the outer border, and each enclosed passable region is
      surrounded by a colored wall whose color matches one legend's fill color.
    - Yellow cells inside a region are movers; legend center and indicator dots encode that region's
      movement direction via their centroid offset.

    Rule:
    - For each enclosed region, decode its direction from the legend whose fill matches the wall color.
    - Slide all yellow movers maximally in that direction, stacking from the far side inward exactly as
      in the original task.
    - Clear the yellow starts, repaint every traversed interior cell blue, and then recolor the aligned
      collision wall face red. Do not leave any yellow terminal in the output.

    Key insight:
    - The original mechanics are unchanged; only the rendering is transformed. The blue trail reveals
      where each mover actually traveled inside the region, while the red wall mark reveals which wall
      face that directional motion ultimately targets.
    """
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

        slides = _slide_paths(pset, movers, dr, dc)
        impact_marks = set()
        for final, path in slides:
            for r, c in path:
                out[r, c] = TRAIL
            impact_marks.update(_impact_wall_cells_grid(pset, g, final, dr, dc, bg, MOVER))
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

    for region_color in region_colors:
        dr, dc = color_dir[region_color]
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
        regions.append({"color": region_color, "dir": (dr, dc), "interior": blob, "walls": walls})

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
) -> tuple[tuple[int, int], int] | None:
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
        start = (r, c)
        if _forward_slide(blob, start, dr, dc, already_stopped) == out_pos:
            return start, steps
    return None


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    total_steps = 0
    longest_trail = 0

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
            region_steps = 0
            region_longest = 0

            for out_pos in out_sorted:
                candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
                if candidate is None:
                    break
                start, steps = candidate
                yellows_input.append(start)
                already_stopped.add(out_pos)
                region_steps += steps
                region_longest = max(region_longest, steps)
            else:
                for r, c in yellows_input:
                    grid[r, c] = MOVER
                reg["input_movers"] = list(yellows_input)
                reg["output_terminals"] = set(output_positions)
                reg["impact_marks"] = set().union(*chosen_sigs)
                total_steps += region_steps
                longest_trail = max(longest_trail, region_longest)
                success = True
                break

        if not success:
            return False

    return total_steps >= MIN_NEW_TRAIL_CELLS and longest_trail >= MIN_LONGEST_TRAIL


def _expected_output(struct_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = struct_grid.copy()
    for reg in regions:
        blob = reg["interior"]
        walls = reg["walls"]
        dr, dc = reg["dir"]
        movers = reg.get("input_movers", [])
        slides = _slide_paths(blob, movers, dr, dc)
        for _final, path in slides:
            for r, c in path:
                out[r, c] = TRAIL
        impact_marks = set()
        for final, _path in slides:
            impact_marks.update(_impact_wall_cells_region(blob, walls, final, dr, dc))
        for r, c in impact_marks:
            out[r, c] = MARKER
    return out


def _variant_metrics(input_grid: np.ndarray, output_grid: np.ndarray) -> dict[str, int]:
    new_trail_mask = (output_grid == TRAIL) & (input_grid != TRAIL)
    new_marker_mask = (output_grid == MARKER) & (input_grid != MARKER)
    return {
        "new_trail_cells": int(new_trail_mask.sum()),
        "new_marker_cells": int(new_marker_mask.sum()),
        "remaining_yellows": int((output_grid == MOVER).sum()),
    }


def _oracle_check(input_grid: np.ndarray, expected_out: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    if not np.array_equal(solver_out, expected_out):
        return False

    metrics = _variant_metrics(input_grid, solver_out)
    if metrics["new_trail_cells"] < MIN_NEW_TRAIL_CELLS:
        return False
    if metrics["new_marker_cells"] < MIN_NEW_MARKER_CELLS:
        return False
    if metrics["remaining_yellows"] != 0:
        return False

    region_mask = np.zeros_like(input_grid, dtype=bool)
    wall_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            region_mask[r, c] = True
        for r, c in reg["walls"]:
            wall_mask[r, c] = True

    new_trail_mask = (solver_out == TRAIL) & (input_grid != TRAIL)
    new_marker_mask = (solver_out == MARKER) & (input_grid != MARKER)
    if not bool(np.all(region_mask[new_trail_mask])):
        return False
    if not bool(np.all(wall_mask[new_marker_mask])):
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
    if not _place_yellows(rng, regions, input_grid):
        return None

    expected_out = _expected_output(grid, regions)
    if not _oracle_check(input_grid, expected_out, regions):
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
